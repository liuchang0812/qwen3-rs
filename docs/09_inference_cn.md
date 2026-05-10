# 09 — 自回归推理：逐 Token 生成文本

语言模型如果无法生成文本，就毫无用处。训练赋予了模型*预测*下一个 token 的能力，而生成则是将这些预测转化为连贯输出的过程。本章解释自回归推理的工作原理：模型利用自身之前的输出作为每个新预测的上文，逐 token 地生成序列的过程。

---

## 1. 什么是自回归生成？

### "自回归"的含义

"自回归"（Auto-regressive）字面意思是"自我回归"——每个输出都依赖于同一过程之前的输出。前缀"auto"表示模型将自己的输出反馈给自己，"regressive"指的是统计学中的回归概念：根据先前的观测值预测一个值。

在语言模型的语境中，自回归生成意味着：

- 模型**每次生成一个 token**。
- 每个新 token 的预测都使用**所有之前生成的 token**作为上下文。
- 该过程持续进行，直到满足停止条件。

### 写作类比

想象你写一句话的过程。你不会同时决定每一个词。你先写第一个词，然后基于第一个词选择第二个词，再基于前两个词选择第三个词，依此类推。每个词都依赖于之前出现的所有内容。语言模型做的事情完全一样——它一次写一个词，始终考虑之前所写的内容。

### 为什么不一次性生成所有 Token？

原则上，模型可以在一次前向传播中，为长序列中的每个位置都产生一个概率分布。但这要求模型"提前规划"——在不知道第 1 到第 9 个 token 的情况下决定第 10 个 token。自回归生成通过始终保持每个预测的完整上下文来避免这个问题。每个 token 都是在完全了解其之前所有内容的情况下选择的，这就是为什么生成的文本具有连贯性。

代价是生成过程**本质上是顺序的**。你无法并行生成多个 token，因为每个 token 都依赖于前一个。这是自回归模型的根本限制，也是推理速度至关重要的关键原因。

---

## 2. 生成循环

自回归生成是一个循环。每次迭代产生一个 token 并将其加入不断增长的序列中。以下是逐步说明：

### 步骤 1：从提示文本开始

用户提供提示（prompt）——用于启动生成的初始文本。例如：

```
"The capital of France is"
```

### 步骤 2：分词

分词器将提示文本转换为整数 token ID 序列。例如：

```
"The capital of France is" → [791, 3187, 315, 5327, 374]
```

这些 token ID 是模型的"母语"。在处理之前，必须将每段文本转换为 ID。

### 步骤 3：预填充（Prefill）

模型在**一次前向传播中同时处理所有提示 token**。这被称为"预填充"阶段。模型并行计算每个提示位置的隐藏状态、注意力分数和 logits。输出是一个形状为 `[prompt_len, vocab_size]` 的 logits 张量。

我们只需要**最后一个**位置的 logits，因为该位置代表模型对整个提示之后下一个 token 的预测。

在预填充期间，KV cache 会为每个提示位置填充数据。这至关重要——这意味着在后续的 decode 步骤中，我们不需要为这些 token 中的任何一个重新计算 K 和 V。

### 步骤 4：采样下一个 Token

取最后一个位置的 logits（一个长度为 `vocab_size` 的向量），并应用采样策略来选择单个 token ID。这可以是贪婪解码（选择概率最高的 token）、温度采样、top-k 过滤、top-p（nucleus）过滤，或这些策略的组合。采样模块（文档第 10 章）将详细介绍这些策略。

### 步骤 5：解码（Decode）

将采样得到的 token 附加到输出序列中。然后仅将**新 token** 通过模型，使用之前步骤中的 KV cache。模型只需计算新 token 的 K 和 V；所有之前的 K 和 V 值已经存在于 cache 中。

这会产生下一个位置的 logits。从这些 logits 中采样得到下一个 token。

### 步骤 6：重复

重复步骤 4-5，直到满足停止条件（见第 5 节）。

### 生成循环伪代码

```
function generate(prompt, max_tokens, sampling_config):
    // 步骤 1-2：分词
    token_ids = tokenize(prompt)

    // 步骤 3：预填充 — 一次性处理所有提示 token
    logits = model.forward(token_ids, start_pos=0)
    next_token = sample(logits[-1], sampling_config)
    output_tokens = [next_token]

    // 跟踪 KV cache 的位置
    pos = len(token_ids)

    // 步骤 4-6：解码循环
    for i in 1..max_tokens:
        logits = model.forward([next_token], start_pos=pos)
        next_token = sample(logits[0], sampling_config)
        output_tokens.append(next_token)
        pos += 1

        if next_token == EOS_TOKEN:
            break

    // 将 token ID 转换回文本
    return decode(output_tokens)
```

注意两个截然不同的阶段：

1. **预填充**：`model.forward(token_ids, start_pos=0)` 一次性处理整个提示。KV cache 在此调用之前为空，调用之后被填充。

2. **解码循环**：`model.forward([next_token], start_pos=pos)` 仅处理一个 token。KV cache 已经包含了所有之前 token 的条目，因此模型只需计算新 token 的 K 和 V，并将它们附加到 cache 中。

`start_pos` 参数告诉模型当前 token 在整个序列中的起始位置。这需要用于两件事：(a) 查找正确的 RoPE 位置编码，以及 (b) 知道 KV cache 中哪些位置对应"过去"token，哪些对应"当前"token。

---

## 3. Prefill 与 Decode 的对比

理解 prefill 和 decode 之间的区别，对于理解推理性能的工作原理至关重要。

### Prefill：处理提示

在 prefill 期间，模型同时处理所有提示 token。如果提示长度为 50 个 token，模型执行一次输入形状为 `[50, hidden_size]` 的前向传播。

Prefill 的关键特征：

- **一次性处理多个 token**：整个提示被并行处理。
- **KV cache 被填充**：prefill 之后，cache 包含每个提示位置的 K 和 V 条目。对于 Qwen3-0.6B（8 个 KV head，head_dim 128），每层的 cache 增长到 `[prompt_len, 1024]`。
- **矩阵运算可以并行化**：QKV 投影、注意力分数计算和 FFN 都是大型矩阵乘法，可以从并行硬件（GPU、CPU 上的 SIMD）中受益。
- **O(n^2) 注意力**：注意力分数的形状为 `[num_heads, prompt_len, prompt_len]`，因此注意力计算随提示长度呈二次方扩展。对于非常长的提示，这可能变得昂贵。

10-token 提示的 prefill 期间形状追踪：

```
Input token IDs:    [10]
Embedding lookup:   [10, 1024]
After layer 0:      [10, 1024]
...
After layer 27:     [10, 1024]
After final norm:   [10, 1024]
After lm_head:      [10, 151936]      ← 每个位置的 logits

KV cache per layer: [10, 512] (K) + [10, 512] (V)
```

我们只需要位置 9（最后一个位置）的 logits 来采样第一个生成的 token。

### Decode：逐 Token 生成

在 decode 期间，模型每个步骤仅处理一个新 token。输入形状为 `[1, hidden_size]`。

Decode 的关键特征：

- **每次一个 token**：每个步骤向序列中添加一个 token。
- **每个步骤向 cache 添加一个 K/V**：cache 每个 decode 步骤增长一行。生成 20 个 token 后，每层的 cache 形状为 `[prompt_len + 20, 512]`。
- **本质上是顺序的**：无法跨 decode 步骤并行化，因为步骤 N+1 依赖于步骤 N 选择的 token。
- **O(n) 注意力**：注意力分数的形状为 `[num_heads, 1, n]`，其中 `n` 是到目前为止的总序列长度。这与序列长度呈线性扩展，而非二次方。

单个 decode 步骤期间的形状追踪（在 10-token prefill 之后生成 token）：

```
Input token ID:     [1]
Embedding lookup:   [1, 1024]
After layer 0:      [1, 1024]
...
After layer 27:     [1, 1024]
After final norm:   [1, 1024]
After lm_head:      [1, 151936]       ← 下一个 token 的 logits

KV cache per layer: [11, 512] (K) + [11, 512] (V)  ← 增长了一行
```

### 为什么这种区分很重要

Prefill 是计算密集型的（大型矩阵运算），而 decode 是内存带宽受限的（运算量小，但每一步都必须读取 KV cache）。在现代硬件上，prefill 通常是快速的阶段，因为它高效地利用了硬件。Decode 每个 token 的速度较慢，因为每一步都需要从内存中读取整个 KV cache，但计算量很少。

这就是为什么基准测试结果中会分别报告"首个 token 时间"（TTFT，由 prefill 速度决定）和"每秒 token 数"（由 decode 速度决定）。

---

## 4. KV Cache

KV cache 是自回归推理中最重要的优化。没有它，生成速度将慢得无法实用。

### 没有 Cache：O(n^2) 的总计算量

考虑生成一个长度为 N 的序列。在每一步 t，模型需要计算当前 token 与所有之前 token 之间的注意力。如果我们不缓存之前步骤的 K 和 V，就必须从头重新计算它们。

在步骤 1（prefill 之后），我们处理长度为 P 的提示。这需要 O(P^2) 的注意力计算。

在步骤 2，我们需要计算新 token 的 K 和 V，**并重新计算**所有 P 个之前 token 的 K 和 V 以计算注意力。这需要 O(P) 的重新计算量。

在步骤 t，我们必须重新计算所有 P + t - 1 个之前 token 的 K 和 V。N 个 decode 步骤的总成本为：

```
Total = sum_{t=1}^{N} O(P + t) = O(N*P + N^2)
```

对于一个 50-token 的提示和 500 个生成的 token，这大约是 27,500 个重新计算的 token——其中大多数在之前的步骤中已经计算过了。绝大部分工作都是冗余的。

### 使用 Cache：每步 O(n)

KV cache 存储每个之前处理过的 token 的 K 和 V 向量。在每个 decode 步骤：

1. 仅计算**新** token 的 K 和 V（O(1) 投影工作）。
2. 将新的 K 和 V 附加到 cache 中（O(1) 内存操作）。
3. 计算新 token 的 Q 与完整的缓存 K、V 之间的注意力（O(n)，其中 n 是到目前为止的总序列长度）。

每步成本是 O(n)，这是不可避免的——新 token 必须关注所有之前的 token。但我们避免了每一步都**重新计算** K 和 V。N 个 decode 步骤的总成本变为：

```
Total = sum_{t=1}^{N} O(P + t) 仅注意力部分
      = O(N*P + N^2/2) 注意力部分
      + O(N) K/V 投影（每步计算一次）
```

注意力部分的渐近复杂度相同（我们无法避免关注所有之前的 token），但我们消除了所有冗余的 K 和 V 投影计算。在实践中，节省是巨大的，因为 K 和 V 投影涉及大型矩阵乘法。

### Qwen3-0.6B 的内存成本计算

KV cache 每层存储两个张量（K 和 V）。每个张量的形状为 `[seq_len, num_kv_heads * head_dim]`。对于 Qwen3-0.6B：

```
num_kv_heads  = 8
head_dim      = 128
kv_dim        = 8 * 128 = 1024
num_layers    = 28
sizeof(f32)   = 4 bytes

KV cache per layer = 2 * kv_dim * seq_len * 4
                   = 2 * 1024 * seq_len * 4
                   = 8,192 * seq_len bytes

Total KV cache     = 8,192 * seq_len * 28
                   = 229,376 * seq_len bytes
```

在不同序列长度下：

| Sequence Length | Per Layer (KB) | Total (MB) |
|----------------|----------------|------------|
| 128            | 512            | 14         |
| 512            | 2,048          | 56         |
| 1,024          | 4,096          | 112        |
| 2,048          | 8,192          | 224        |
| 4,096          | 16,384         | 448        |
| 8,192          | 32,768         | 896        |
| 16,384         | 65,536         | 1,792      |

Cache 随序列长度线性增长。对于具有 4096-token 上下文的 Qwen3-0.6B，KV cache 消耗近 450 MB。对于具有更多层和 head 的更大模型，cache 可以轻松超过数 GB。

### Cache 随时间增长

Cache 开始时为空。在 prefill 期间，它被整个提示填充。在 decode 期间，每层每步增长一行。

```
Time    Tokens Processed    Cache Rows (per layer)    Total Cache (MB)
----    ----------------    ----------------------    ----------------
t=0     0 (empty)           0                          0
t=1     50 (prefill)        50                         5.5
t=2     51 (decode)         51                         5.6
t=3     52 (decode)         52                         5.7
...
t=450   499 (decode)        499                        54.7
t=451   500 (decode)        500                        54.8
```

在 50-token 提示之后进行 500 次 decode 步骤，cache 每层保存 550 行。每个 decode 步骤使 cache 略微增大，这意味着后续每个注意力计算都略微变慢（因为 Q 必须关注更多缓存的 K/V 条目）。

### 我们的实现

在我们的 Rust 代码中，`KVCache` 结构体定义在 `src/attention.rs` 中：

```rust
pub struct KVCache {
    pub key_cache:   Option<Tensor>,  // [seq_len_so_far, num_kv_heads * head_dim]
    pub value_cache: Option<Tensor>,  // [seq_len_so_far, num_kv_heads * head_dim]
}
```

`Option` 类型反映了 cache 开始时为空的事实。在第一次前向传播（prefill）时，计算得到的 K 和 V 被直接存储。在后续传播（decode）时，新的 K/V 行使用 `stack_rows` 附加：

```rust
let (k_full, v_full) = match (&kv_cache.key_cache, &kv_cache.value_cache) {
    (Some(prev_k), Some(prev_v)) => {
        // 将新的 K/V 行连接到缓存行之后。
        let full_k = prev_k.stack_rows(&k_flat);
        let full_v = prev_v.stack_rows(&v_flat);
        (full_k, full_v)
    }
    (None, None) => {
        // 第一次前向传播：直接使用当前的 K 和 V。
        (k_flat, v_flat)
    }
    _ => panic!("KV cache is in an inconsistent state"),
};
```

`QwenModel` 结构体拥有每个 transformer 层的一个 `KVCache`，存储在 `Vec<KVCache>` 中：

```rust
pub struct QwenModel {
    embed_tokens: Tensor,
    layers: Vec<TransformerBlock>,
    norm: RMSNorm,
    lm_head: Tensor,
    config: ModelConfig,
    kv_caches: Vec<KVCache>,  // 每层一个
}
```

---

## 5. 停止条件

生成循环最终必须停止。在每次 decode 步骤都应检查两个条件：

### 最大 Token 限制

调用者指定要生成的最大 token 数。这是一个硬性上限，防止模型无限生成。在我们的 CLI 中，这是 `--max-tokens` 标志，默认为 100。

检查很简单：

```
if len(generated_tokens) >= max_tokens:
    stop generation
```

这很重要，原因有几个：

- **成本控制**：生成更多 token 需要更多时间和内存。
- **安全性**：没有限制的情况下，退化的采样配置可能导致模型生成无界的 token 流。
- **用户预期**：用户通常对响应应该多长有一个大致的了解。

### EOS Token（序列结束）

模型有一个特殊的"序列结束"（EOS）token，表示它已完成生成。当模型采样到 EOS token 时，意味着"我没什么要说的了。"生成循环应该停止。

EOS token 在分词器配置中定义。对于 Qwen3 分词器，EOS token ID 是 151645（对应 `<|im_end|>`），还有一个文本结束 token，ID 为 151643。

检查是：

```
if next_token == EOS_TOKEN_ID:
    stop generation
```

在每一步检查两个条件很重要。EOS 检查提供了一个自然的停止点，而 max_tokens 检查为模型不产生 EOS token 的情况提供了安全网（这可能发生在某些提示或采样配置下）。

### 两个条件之间的交互

生成循环应该先检查 max_tokens 条件（或与 EOS 检查结合），因为：

- 如果用户设置 `max_tokens = 0`，模型应该完全不产生输出。
- 如果模型在步骤 5 生成了 EOS token，但 max_tokens 是 100，生成应该在步骤 5 停止。
- 如果模型在步骤 100 之前没有生成 EOS token，无论如何在步骤 100 停止。

伪代码：

```
for step in 0..max_tokens:
    logits = model.forward(...)
    next_token = sample(logits, config)
    if next_token == EOS_TOKEN_ID:
        break
    output_tokens.append(next_token)
```

---

## 6. 流式输出

### 为什么流式输出很重要

想象一下让语言模型写一篇文章。如果模型在显示任何输出之前生成所有 500 个 token，用户会盯着空白屏幕几秒钟，然后突然看到整篇文章出现。这是一种糟糕的用户体验。

流式输出通过在每个 token 生成后立即显示来解决此问题。用户看到文本逐词递增地出现，就像看着某人打字一样。这创造了速度感和响应感，即使总生成时间相同。

### 逐 Token 解码

在每个 decode 步骤之后，新采样的 token ID 使用分词器的 `decode` 函数转换回其字符串表示。然后该字符串立即发送到输出。关键的洞察是，我们不需要等待整个生成完成再开始显示输出。

然而，有一个微妙之处：某些 token 是部分 UTF-8 序列。例如，一个多字节 Unicode 字符可能被拆分到两个 token 中。尝试单独解码每个 token 可能产生无效的 UTF-8 或乱码文本。一个健壮的流式实现会缓冲 token，只将完整的字符刷新到输出。

### 我们的实现：generate_with_callback

我们的推理引擎通过回调机制支持流式输出。生成器不在最后累积所有 token 并返回它们，而是在每个 token 生成后调用用户提供的回调函数：

```rust
pub trait InferenceCallback {
    fn on_token(&mut self, token_id: usize, token_text: &str);
}
```

生成循环变为：

```
for step in 0..max_tokens:
    logits = model.forward(...)
    next_token = sample(logits, config)
    if next_token == EOS_TOKEN_ID:
        break

    token_text = tokenizer.decode(next_token)
    callback.on_token(next_token, &token_text)
    output_tokens.append(next_token)
```

这种设计将生成逻辑与输出显示分离。同一个 `generate` 函数适用于流式和非流式用例：

- **流式**：传递一个立即打印每个 token 的回调。
- **非流式**：传递一个将 token 累积到缓冲区中的回调，然后在生成完成后读取缓冲区。

---

## 7. 实现细节

### InferenceEngine

我们的 `src/inference.rs` 模块将提供一个 `InferenceEngine` 结构体，封装模型、分词器和采样配置：

```rust
pub struct InferenceEngine {
    model: QwenModel,
    tokenizer: Tokenizer,
    sampling_config: SamplingConfig,
    eos_token_id: usize,
}
```

引擎拥有生成所需的所有组件。`QwenModel` 持有权重和 KV cache，`Tokenizer` 处理文本到 ID 和 ID 到文本的转换，`SamplingConfig` 控制采样策略。

### generate() 方法

主要入口点是 `generate` 方法，它接受提示字符串和生成参数，并返回生成的文本：

```rust
impl InferenceEngine {
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
    ) -> String {
        // 步骤 1：对提示进行分词
        let token_ids = self.tokenizer.encode(prompt);

        // 步骤 2：预填充 — 处理所有提示 token
        let logits = self.model.forward(&token_ids, 0);
        let mut next_token = sample(&logits.last_row(), &self.sampling_config);

        let mut generated_ids = Vec::new();
        if next_token == self.eos_token_id {
            return self.tokenizer.decode(&generated_ids);
        }
        generated_ids.push(next_token);

        // 步骤 3：解码循环
        let mut pos = token_ids.len();
        for _ in 1..max_tokens {
            let logits = self.model.forward(&[next_token], pos);
            next_token = sample(&logits.last_row(), &self.sampling_config);
            pos += 1;

            if next_token == self.eos_token_id {
                break;
            }
            generated_ids.push(next_token);
        }

        // 步骤 4：将 token ID 解码回文本
        self.tokenizer.decode(&generated_ids)
    }
}
```

### start_pos 跟踪

`start_pos` 参数对于正确的推理至关重要。它告诉模型当前前向传播中第一个 token 的绝对位置。这用于：

1. **RoPE**：位置编码按位置索引查找。如果我们传递了错误的 `start_pos`，模型将对 Q 和 K 应用错误的旋转角度，产生乱码。

2. **因果掩码**：在 prefill 期间，`start_pos = 0`，模型应用标准下三角因果掩码。在 decode 期间，`start_pos = prompt_len + tokens_generated_so_far`，模型确定单个新 token 可以关注所有缓存的位置（不需要掩码）。

3. **KV cache 一致性**：cache 按顺序存储 K 和 V，因此 `start_pos` 必须与新 token 在整个序列中的实际位置匹配。如果 `start_pos` 错误，RoPE 和因果掩码将不正确，生成的文本将不连贯。

跟踪很简单：

```
After prefill:    start_pos = prompt_len
After 1 decode:   start_pos = prompt_len + 1
After 2 decodes:  start_pos = prompt_len + 2
...
After k decodes:  start_pos = prompt_len + k
```

在代码中，我们在 prefill 之后初始化 `pos = token_ids.len()`，并在每个 decode 步骤后将其递增 1。

### 新对话的 Cache 重置

KV cache 累积了所有之前前向传播的状态。当开始一个新对话（或一个新的独立生成）时，必须重置 cache。否则，新提示将在旧对话的上下文中被处理，产生无意义的输出。

我们的 `QwenModel` 提供了一个重置所有 KV cache 的方法：

```rust
impl QwenModel {
    pub fn reset_kv_cache(&mut self) {
        for cache in &mut self.kv_caches {
            cache.key_cache = None;
            cache.value_cache = None;
        }
    }
}
```

这将每层的 K 和 V cache 设置回 `None`，这与新创建模型的状态相同。下一次前向传播将是 prefill，从头开始填充 cache。

什么时候应该重置 cache？

- **在独立对话之间**：如果用户开始一个全新的话题，旧上下文无关紧要，应该被清除。
- **当上下文窗口已满时**：如果总序列长度（提示 + 生成）接近 `max_position_embeddings`，模型无法再正确关注所有 token。应该重置 cache，并重新开始对话（或使用滑动窗口/截断策略）。
- **在测试用例之间**：在单元测试中，每个测试都应该以干净的 cache 开始，以确保隔离。

在聊天应用程序中，cache 通常在每个新会话开始时重置。在一个会话内，cache 随着对话的进行而增长，使模型能够在多轮对话中保持上下文。

### 综合示例

以下是使用 CLI 中的推理引擎的完整示例：

```rust
fn main() {
    let args = Args::parse();

    // 从磁盘加载模型
    let mut engine = InferenceEngine::from_dir(&args.model_dir)
        .expect("Failed to load model");

    // 配置采样
    engine.sampling_config = SamplingConfig {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        seed: args.seed,
    };

    // 生成文本
    let output = engine.generate(&args.prompt, args.max_tokens);
    println!("{}", output);
}
```

`InferenceEngine` 将分词、prefill、decode、KV cache 管理和采样的细节隐藏在简单的 `generate` 方法背后。调用者提供提示和 token 限制，即可获得生成的文本。

### 完整流程可视化

```
User types: "Explain quantum computing"

                    TOKENIZE
"Explain quantum computing" → [4523, 18364, 18342, 15496]
                                 ↓
                    PREFILL (start_pos=0)
model.forward([4523, 18364, 18342, 15496], 0)
  - Embedding lookup:    [4, 1024]
  - 28 transformer blocks (KV cache populated)
  - Final norm + lm_head: [4, 151936]
  - Take logits at position 3
  - Sample: token 311 (="Quantum")
                                 ↓
                    DECODE LOOP (start_pos=4, 5, 6, ...)

Step 1: model.forward([311], 4) → sample → token 14758 (=" computing")
Step 2: model.forward([14758], 5) → sample → token 374 (=" is")
Step 3: model.forward([374], 6) → sample → token 264 (=" a")
Step 4: model.forward([264], 7) → sample → token 28354 (=" field")
  ...
Step N: model.forward([token], N+3) → sample → 151645 (EOS)
                                 ↓
                    DECODE
[311, 14758, 374, 264, 28354, ...] → "Quantum computing is a field..."
```

Decode 循环的每一步都为每层的 KV cache 添加恰好一行。`start_pos` 每一步增加 1，确保正确的 RoPE 编码和因果掩码。

---

## 总结

| 概念 | 核心思想 |
|---------|----------|
| 自回归生成 | 每个 token 依赖于所有之前的 token；一次生成一个 |
| Prefill | 一次性处理所有提示 token；填充 KV cache |
| Decode | 每步处理一个新 token；使用 KV cache 避免重新计算 |
| KV cache | 存储过去的 K、V，将每步成本从重新计算所有 K、V 降低到仅计算新的 |
| start_pos | 跟踪绝对位置以确保正确的 RoPE 和因果掩码 |
| 停止条件 | 最大 token 限制（用户配置）和 EOS token（模型决定） |
| 流式输出 | 在生成时逐个显示 token，提供响应迅速的用户体验 |
| Cache 重置 | 在独立对话之间清除 KV cache |

自回归推理是训练好的模型与可用输出之间的桥梁。模型的前向传播计算 logits；生成循环将这些 logits 转化为 token 序列；分词器将这些 token 转换回文本。KV cache 使这个过程高效，而流式输出使用户感觉它很快。
