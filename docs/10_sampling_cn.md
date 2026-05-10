# 10 — Token 采样：从 Logits 到文本

在自回归生成的每一步，模型会输出一个原始分数向量，称为 **logits**——词汇表中每个 token 对应一个分数。问题是：如何将这些分数转化为单个 token？这就是 **采样** 问题，我们选择的策略对生成文本的质量、多样性和风格有着巨大影响。

本章涵盖 `src/sampling.rs` 模块中实现的所有采样策略，从最简单的（贪心）到最复杂的（temperature + top-k + top-p），并解释我们 Rust 代码的实现细节。

---

## 1. 从 Logits 到 Tokens

### 什么是 Logits？

Transformer 的最后一层是一个线性投影（`lm_head`），将隐藏状态从 `hidden_size`（1024）映射到 `vocab_size`（151936）。输出是一个原始、未归一化的分数向量——词汇表中每个 token 对应一个。这些分数称为 **logits**。

Logits 可以是任意实数：正数、负数、大数或小数。一个 5.3 的 logit 并不意味着"5.3% 的概率"。Logits 必须先转换为概率才能用于采样。

### 通过 Softmax 将 Logits 转为概率

标准转换方法是 **softmax** 函数：

```
softmax(z_i) = exp(z_i) / sum_j(exp(z_j))
```

Softmax 将任意实数向量转换为概率分布：所有值都在 [0, 1] 范围内，且总和为 1。Logit 最高的 token 获得最高的概率，但每个 token 都会获得*一些*非零概率。

为了数值稳定性，我们使用最大值减法技巧：

```
softmax(z_i) = exp(z_i - max(z)) / sum_j(exp(z_j - max(z)))
```

这可以防止 logits 较大时 `exp()` 溢出。

### 选择问题

经过 softmax 后，我们得到了整个词汇表上的概率分布。现在我们必须选择一个 token。最简单的方法是始终选择概率最高的 token（贪心解码）。但在许多情况下，我们需要一些随机性——以生成有创意的文本、避免重复输出，或探索多种可能的续写。

这就是采样策略的用武之地。它们控制我们*如何*从概率分布中选择 token，在确定性、质量与多样性、创造性之间进行权衡。

---

## 2. 贪心解码（Greedy Decoding）

### 工作原理

贪心解码始终选择概率最高的 token：

```
token = argmax(logits)
```

不涉及任何随机性。给定相同的输入，贪心解码始终产生相同的输出。这是最简单的采样策略。

### 特性

- **确定性**：相同的输入始终产生相同的输出。这使得结果可复现，对调试和测试很有用。
- **可重复**：运行相同的提示词两次会得到相同的文本。
- **快速**：argmax 的时间复杂度是 O(vocab_size)，且不需要随机数生成。

### 问题

贪心解码有两个众所周知的问题：

1. **输出乏味**：通过始终选择最可能的 token，模型会收敛到最"平庸"或"安全"的续写。有创意、令人惊喜或有趣的词语选择永远不会被选中，因为它们的概率略低。

2. **重复循环**：一旦模型进入某种模式（例如，"The cat sat on the mat. The cat sat on the mat. The cat sat on the mat."），贪心解码没有机制来打破它。重复模式最可能的结果就是更多重复，从而形成无限循环。

### 示例

给定提示词 "The cat sat on the"，模型可能产生这些 logits（缩略显示前几名候选）：

```
"mat"   → logit 8.2   → probability 0.65
"couch" → logit 6.1   → probability 0.08
"floor" → logit 5.8   → probability 0.06
"table" → logit 5.5   → probability 0.04
"roof"  → logit 4.2   → probability 0.01
```

贪心解码始终选择 "mat"（概率最高）。每次运行这个提示词，你都会得到 "The cat sat on the mat." 没有任何变化。

### 我们的实现

```rust
pub fn sample_greedy(logits: &[f32]) -> usize {
    argmax(logits)
}
```

`argmax` 辅助函数找到最大值的索引：

```rust
fn argmax(slice: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = slice[0];
    for (i, &v) in slice.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}
```

在平局的情况下，第一个索引获胜（最左边的）。

---

## 3. Temperature（温度）

### 工作原理

Temperature 在 softmax 之前对 logits 进行缩放。给定温度 T：

```
scaled_logits[i] = logits[i] / T
probabilities = softmax(scaled_logits)
```

Temperature 不会改变 token 按概率排序的*顺序*——logit 最高的 token 仍然是概率最高的 token。相反，temperature 改变的是分布的**形状**：它是更尖锐还是更平坦。

### 不同温度的效果

**Temperature = 0**（贪心）：除以零是未定义的，因此我们将温度 0 视为特殊情况，直接返回 `argmax(logits)`。这相当于让分布无限尖锐——所有概率质量集中在单个 logit 最高的 token 上。

**Temperature < 1**（更尖锐）：将 logits 除以一个小数会使它们的幅度变大。经过 softmax 后，分布变得更加尖锐——概率最高的 token 获得更高的概率，低概率 token 被进一步抑制。模型变得更加"自信"和保守。

**Temperature = 1**（默认）：不应用缩放。Logits 按原样使用。这是模型的"自然"分布。

**Temperature > 1**（更平坦）：将 logits 除以一个大数会使它们的幅度变小。经过 softmax 后，分布变得更加平坦——概率质量更均匀地分布在各个 token 上。模型变得不那么自信，更加随机。

**Temperature 趋近于无穷大**：随着 T 增大，所有 logits 趋近于零。对接近零的值进行 softmax 会产生接近均匀的分布，每个 token 的概率大约相等（1/vocab_size）。

### 具体示例

假设模型为三个候选 token 产生了这些 logits：

```
"mat"   → logit 6.0
"couch" → logit 3.0
"roof"  → logit 0.0
```

**Temperature = 0.5**（更尖锐）：

```
scaled logits: [6.0/0.5, 3.0/0.5, 0.0/0.5] = [12.0, 6.0, 0.0]
softmax:       [0.9975,  0.0025, 0.0000]
                 ↑ "mat" 几乎完全占据主导
```

**Temperature = 1.0**（默认）：

```
scaled logits: [6.0, 3.0, 0.0]
softmax:       [0.9500, 0.0474, 0.0024]
                 ↑ "mat" 仍然非常可能，但其他 token 也有一些机会
```

**Temperature = 2.0**（更平坦）：

```
scaled logits: [6.0/2, 3.0/2, 0.0/2] = [3.0, 1.5, 0.0]
softmax:       [0.7054, 0.1419, 0.0317]
                 ↑ "mat" 仍然最有可能，但分布要平坦得多
```

**Temperature = 10.0**（非常平坦）：

```
scaled logits: [0.6, 0.3, 0.0]
softmax:       [0.3943, 0.2912, 0.2153]
                 ↑ 几乎均匀 —— 任何 token 都可能被选中
```

注意 "mat" 的概率如何从 0.9975（温度 0.5）下降到 0.3943（温度 10.0）。Temperature 让我们能够精细地控制模型输出看起来有多"随机"。

### 为什么 Temperature 有效

Softmax 函数是指数函数。当 logits 相差很大（幅度大）时，指数会放大差异，使分布非常尖锐。当 logits 彼此接近（幅度小）时，指数的作用较小，使分布更加均匀。Temperature 控制 logits 的幅度，进而控制分布的尖锐程度。

### 我们的实现

在我们的 `sample` 函数中，temperature 是第一个应用的变换：

```rust
if config.temperature == 0.0 {
    return sample_greedy(logits);
}

let mut scaled: Vec<f32> = logits.iter().map(|&l| l / config.temperature).collect();
```

如果温度为零，我们直接跳转到贪心解码。否则，在进入下一个采样阶段之前，我们将每个 logit 除以温度。

---

## 4. Top-k 采样

### 工作原理

Top-k 采样将候选集限制为 logits（或概率）最高的 `k` 个 token。所有其他 token 的概率设为零，剩余的 `k` 个 token 被重新归一化，使它们的总和为 1。

算法：

1. 按 logit（或概率）降序对 token 排序。
2. 只保留前 `k` 个 token。
3. 将所有其他 token 的 logits 设为负无穷（在 softmax 后变为概率 0）。
4. 对过滤后的 logits 应用 softmax。结果是一个仅包含前 `k` 个 token 的分布。

### 不同 k 值的效果

**k = 1**：只保留单个概率最高的 token。这等价于贪心解码。

**k = 小值（例如 10）**：只有 10 个最可能的 token 是候选。这会产生聚焦、连贯但多样性有限的文本。

**k = 50（典型值）**：50 个最可能的 token 是候选。这对大多数任务来说是在连贯性和多样性之间的良好平衡。

**k = vocab_size**：不应用过滤。每个 token 都是候选。这等价于完全不使用 top-k。

### 为什么 Top-k 有帮助

如果不进行过滤，模型偶尔会从分布的长尾中采样极不可能的 token。这些 token 可能会破坏文本的连贯性——想象一个格式良好的句子突然包含一个随机的中文字符或罕见的标点符号。Top-k 通过防止模型考虑它非常不确定的 token 来消除这些退化的样本。

### 具体示例

假设经过温度缩放后，前 6 个 token 具有以下概率（词汇表大小为 151,936）：

```
"mat"    → 0.50
"couch"  → 0.20
"floor"  → 0.10
"table"  → 0.05
"roof"   → 0.03
"bed"    → 0.02
...（151,930 个其他 token 共享剩余的 0.10 概率）
```

使用 **k = 3**：

```
保留：    "mat", "couch", "floor"
丢弃：    其他所有
重新归一化：
"mat"    → 0.50 / 0.80 = 0.625
"couch"  → 0.20 / 0.80 = 0.250
"floor"  → 0.10 / 0.80 = 0.125
```

现在模型只能生成 "mat"、"couch" 或 "floor"。非常不可能的 token（它们共同持有 0.10 的概率）被完全消除。剩余 token 的概率按比例放大。

### Top-k 的局限性

Top-k 使用**固定**数量的候选，而不管分布的形状。这会产生两个问题：

1. **当模型非常自信时**（一个 token 的概率为 0.95），保留 k=50 个 token 会迫使模型考虑 49 个极不可能的 token。这些 token 不应该成为候选，但 top-k 仍然包含了它们。

2. **当模型不确定时**（概率均匀分布在许多 token 上），保留 k=50 可能会排除几乎与第 50 个 token 一样可能的 token。截断是任意的，可能会丢弃好的候选。

接下来介绍的 Top-p（nucleus）采样解决了这两个问题。

### 我们的实现

```rust
if config.top_k > 0 && config.top_k < scaled.len() {
    let k = config.top_k;
    // 找到第 k 大的值。
    let mut sorted: Vec<f32> = scaled.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted[k - 1];
    for v in scaled.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}
```

我们找到第 k 大的 logit 值，并将低于它的所有值设为负无穷。在随后的 softmax 调用之后，这些位置的概率变为零。

注意 `top_k = 0` 会禁用 top-k 过滤（不应用截断）。这是用户不想要 top-k 时的默认行为。

---

## 5. Top-p（Nucleus）采样

### 工作原理

Top-p 采样（也称为 **nucleus 采样**）保留累积概率至少为 `p` 的最小 token 集合。与使用固定计数的 top-k 不同，top-p 会根据分布的形状进行自适应调整。

算法：

1. 按概率降序对 token 排序。
2. 从顶部开始累积概率，一旦累积和达到或超过 `p` 就停止。
3. 将截断点之外的所有 token 设为零。
4. 重新归一化剩余 token，使它们的总和为 1。

### 不同 p 值的效果

**p 接近 0**：保留非常少的 token。当 p = 0 时，只有单个概率最高的 token 存活（等价于贪心）。

**p = 0.9（典型值）**：保留累积概率达到 0.9 的最小 token 集合。当模型自信时，可能只有 2-3 个 token；当不确定时，可能有 20-30 个 token。

**p = 1.0**：不应用过滤。所有 token 都是候选。

### 为什么 Top-p 优于 Top-k

Top-p 根据模型的置信度动态适应：

- **当模型非常自信时**（一个 token 的概率为 0.96），top-p = 0.9 只保留 1-2 个 token。模型的强烈偏好得到尊重，没有不可能的 token 稀释分布。

- **当模型不确定时**（概率分布在许多 token 上），top-p = 0.9 可能保留 30 多个 token。由于模型对正确的续写不太确定，因此包含了更多候选。

这种自适应性产生了比 top-k 的固定截断方法更自然的文本。

### 具体示例

假设排序后的概率分布如下：

```
"mat"    → 0.60
"couch"  → 0.20
"floor"  → 0.08
"table"  → 0.04
"roof"   → 0.03
"bed"    → 0.02
...（其他 token：0.03）
```

使用 **p = 0.9**：

```
步骤 1："mat"    → cumulative = 0.60  (< 0.9，保留)
步骤 2："couch"  → cumulative = 0.80  (< 0.9，保留)
步骤 3："floor"  → cumulative = 0.88  (< 0.9，保留)
步骤 4："table"  → cumulative = 0.92  (>= 0.9，也保留这个，然后停止)
保留的 token："mat", "couch", "floor", "table"
丢弃的 token："roof", "bed" 和所有其他
重新归一化（保留的总和 = 0.92）：
"mat"    → 0.60 / 0.92 = 0.652
"couch"  → 0.20 / 0.92 = 0.217
"floor"  → 0.08 / 0.92 = 0.087
"table"  → 0.04 / 0.92 = 0.043
```

现在将其与模型非常自信的不同分布进行比较：

```
"mat"    → 0.95
"couch"  → 0.02
"floor"  → 0.01
"table"  → 0.01
...
```

使用 **p = 0.9**：

```
步骤 1："mat"    → cumulative = 0.95  (>= 0.9，保留这个，然后停止)
保留的 token：只有 "mat"
丢弃的 token：其他所有
重新归一化："mat" → 1.0
```

当模型自信时，top-p 保留更少的 token。当模型不确定时，top-p 保留更多。这正是我们想要的行为。

### 我们的实现

Top-p 过滤在 softmax 之后应用（softmax 将 logits 转换为概率）：

```rust
if config.top_p < 1.0 {
    let n = scaled.len();
    // 按概率降序排序索引。
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        scaled[b].partial_cmp(&scaled[a]).unwrap_or(std::cmp::Ordering::Equal)
    });

    // 找到截断点：保留 token 直到累积概率 >= top_p。
    let mut cumsum = 0.0f32;
    let mut cutoff = 0;
    for &idx in &indices {
        cumsum += scaled[idx];
        cutoff += 1;
        if cumsum >= config.top_p {
            break;
        }
    }

    // 将截断点之外的 token 设为零。
    for &idx in &indices[cutoff..] {
        scaled[idx] = 0.0;
    }

    // 重新归一化。
    let sum: f32 = scaled.iter().sum();
    if sum > 0.0 {
        for v in scaled.iter_mut() {
            *v /= sum;
        }
    }
}
```

关键步骤是：按概率降序排序，累积直到总和达到 `top_p`，将截断点之外的一切设为零，然后重新归一化。

---

## 6. 组合策略

### 完整流程

我们的 `sample` 函数按特定顺序应用策略：

```
原始 logits
    │
    ▼
1. Temperature 缩放：  logits / temperature
    │
    ▼
2. Top-k 过滤：  将低于第 k 大的 logits 设为 -inf
    │
    ▼
3. Softmax：  将过滤后的 logits 转换为概率
    │
    ▼
4. Top-p 过滤：  将累积概率超过阈值的 token 之外的部分设为零
    │
    ▼
5. 重新归一化：  确保概率总和为 1
    │
    ▼
6. CDF 采样：  生成一个随机数，遍历 CDF 来选择一个 token
```

### 为什么按此顺序应用

顺序很重要。以下是每个步骤放在此处的原因：

1. **Temperature 优先**：Temperature 改变 logit 分布的*形状*。它必须在任何过滤之前应用，因为过滤决策取决于 logits 的相对大小。在 top-k 之后应用 temperature 会以意想不到的方式改变已过滤 token 的概率。

2. **Top-k 在 softmax 之前**：Top-k 基于 *logit 幅度* 进行过滤，而不是概率。在 logit 级别进行过滤更自然，因为 logits 具有线性尺度。经过 softmax 后，概率处于指数尺度，"高概率"和"低概率"之间的区别被压缩了。在 softmax 之前将过滤后的 logits 设为 `-inf` 可确保它们变为概率恰好为 0。

3. **Top-p 在 softmax 之后**：Top-p 对 *概率* 进行操作，而不是 logits。它需要概率的累积和，这需要一个有效的概率分布（因此需要先 softmax）。在 logit 级别应用 top-p 是不正确的，因为 logit 值不具有概率解释。

4. **CDF 采样最后**：经过所有过滤和重新归一化后，我们得到一个干净的概率分布。CDF 采样从这个分布中抽取一个 token。

### 常见参数组合

不同的任务受益于不同的采样配置：

**创意写作**（故事、诗歌、头脑风暴）：

```
temperature = 0.9
top_k = 50
top_p = 0.95
```

相对较高的 temperature 鼓励多样化的词语选择。Top-k 和 top-p 通过防止模型采样极不可能的 token 来避免其偏离轨道。

**代码生成**（编程、结构化输出）：

```
temperature = 0.2
top_k = 10
top_p = 0.9
```

代码需要精确的语法和逻辑。低 temperature 使模型紧贴最可能的结果。小的 top-k 将候选限制在最合理的 token 上，避免因不可能的样本而产生语法错误。

**事实问答**（知识检索、摘要）：

```
temperature = 0.0  （贪心）
```

对于事实问题，通常只有一个正确答案。贪心解码确保模型始终选择最可能的（希望也是最准确的）token。没有随机性意味着同一个问题总是得到相同的答案。

**对话聊天**（通用助手）：

```
temperature = 0.7
top_k = 50
top_p = 0.9
```

适中的 temperature 产生多样化但连贯的响应。这些是我们 `SamplingConfig` 中的默认值。

### 为什么不仅仅使用 Temperature？

仅靠 Temperature 无法防止模型偶尔采样非常不可能的 token。即使在 temperature 0.5 的情况下，分布的长尾也包含数千个概率微小但非零的 token。偶尔，其中一个 token 会被采样，可能会破坏文本的连贯性。

Top-k 和 top-p 充当安全网：它们完全消除了长尾，确保只考虑了合理的候选。结合 temperature 进行形状控制，它们让用户可以精确控制质量与多样性之间的权衡。

---

## 7. 随机性与可复现性

### 伪随机数生成器（PRNG）

采样的最后一步——CDF 采样——需要一个随机数。但计算机程序中的"随机"意味着"伪随机"：由确定性算法生成的一系列数字，这些数字*看起来*是随机的，但完全由初始**种子**值决定。

这是一个特性，而不是 bug。确定性随机性实现了**可复现性**：给定相同的种子，PRNG 产生相同的随机数序列，这意味着相同的 logits 和采样配置会产生相同的输出 token。这对调试、测试和创建可复现的演示非常宝贵。

### 我们的 XorShift64 实现

我们使用 xorshift64 PRNG——最简单、最快的非加密 PRNG 之一。它在 `src/sampling.rs` 中实现，没有外部依赖：

```rust
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        assert!(seed != 0, "XorShift64 seed must not be zero");
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}
```

该算法是三个 XOR-移位操作序列，使用经典的移位三元组 (13, 7, 17)。从任何非零的 64 位种子开始，它产生一个周期为 2^64 - 1 的 64 位数字序列，然后重复。

`next_f32` 方法通过将顶部 24 位除以 2^24，将 u64 转换为 [0, 1) 范围内的浮点数。这提供了大约 7 位十进制精度，对于采样决策来说绰绰有余。

### 用于可复现性的种子

我们的 `SamplingConfig` 有一个可选的 `seed` 字段：

```rust
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: Option<u64>,
}
```

当 `seed` 为 `Some(value)` 时，采样是完全确定性的：相同的 logits 和配置始终产生相同的 token。这对以下情况很有用：

- **测试**：单元测试可以断言确切的 token ID，而不必担心随机性。
- **可复现性**：用户可以共享种子以复现特定的输出。
- **调试**：可以使用相同的种子精确复现失败的生成。

当 `seed` 为 `None` 时，我们使用默认种子 12345。这意味着采样在单次运行中是确定性的，但在不同调用之间不是（除非用户设置显式种子）。

### 为什么不使用加密级 RNG？

加密级 RNG（如 `/dev/urandom` 或 ChaCha20）产生高质量的随机性，即使对攻击者来说也是不可预测的。但对于 token 采样，加密质量是不必要的——我们只需要均匀分布且可复现的数字。加密级 RNG 会更慢，并且会使可复现性更难实现（因为它们的输出依赖于系统熵）。

XorShift64 的特点是：

- **快速**：每个随机数只需三次 XOR-移位操作，除了 8 字节的状态外不需要内存访问。
- **小巧**：只有 8 字节的状态。
- **确定性**：给定相同的种子，它总是产生相同的序列。
- **统计上足够**：它通过均匀性和独立性的标准统计测试，这正是采样所需要的。

唯一的警告是 xorshift64 **不是**加密安全的。给定几个输出，攻击者可以预测未来的输出。但由于我们生成的是文本，而不是加密密钥，所以这无关紧要。

---

## 8. 实现细节

### SamplingConfig

`SamplingConfig` 结构体捆绑了所有采样参数：

```rust
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub temperature: f32,     // 0.0 = 贪心，0.7 = 默认
    pub top_k: usize,         // 0 = 禁用，50 = 默认
    pub top_p: f32,           // 1.0 = 禁用，0.9 = 默认
    pub seed: Option<u64>,    // None = 默认种子
}
```

默认配置：

```rust
impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            seed: None,
        }
    }
}
```

这些默认值适用于通用文本生成：适中的 temperature 用于多样化但连贯的输出，top-k 为 50 以消除长尾，top-p 为 0.9 以根据置信度自适应过滤。

### sample() 函数逐步讲解

`sample` 函数是主要入口点。它接受一个 logits 切片和一个 `SamplingConfig`，返回一个 token ID。以下是带注释的完整流程：

```rust
pub fn sample(logits: &[f32], config: &SamplingConfig) -> usize {
    // 步骤 1：Temperature == 0 意味着贪心。
    // 立即短路以避免不必要的计算。
    if config.temperature == 0.0 {
        return sample_greedy(logits);
    }

    // 步骤 2：应用 temperature 缩放。
    // 将每个 logit 除以 temperature。
    // 这改变分布的形状而不改变顺序。
    let mut scaled: Vec<f32> = logits.iter().map(|&l| l / config.temperature).collect();

    // 步骤 3：应用 top-k 过滤。
    // 找到第 k 大的 logit 值，并将低于它的所有值设为 -inf。
    // 经过 softmax 后，-inf 变为概率 0，有效地将这些
    // token 从候选集中移除。
    if config.top_k > 0 && config.top_k < scaled.len() {
        let k = config.top_k;
        let mut sorted: Vec<f32> = scaled.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[k - 1];
        for v in scaled.iter_mut() {
            if *v < threshold {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // 步骤 4：Softmax 将过滤后的 logits 转换为概率。
    // 使用数值稳定的最大值减法技巧。
    // -inf logits 正确地变为概率 0（exp(-inf) = 0）。
    softmax_in_place(&mut scaled);

    // 步骤 5：应用 top-p（nucleus）过滤。
    // 按概率降序排序，累积直到 sum >= top_p，
    // 将剩余部分设为零，然后重新归一化。
    if config.top_p < 1.0 {
        let n = scaled.len();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            scaled[b].partial_cmp(&scaled[a]).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut cumsum = 0.0f32;
        let mut cutoff = 0;
        for &idx in &indices {
            cumsum += scaled[idx];
            cutoff += 1;
            if cumsum >= config.top_p {
                break;
            }
        }

        for &idx in &indices[cutoff..] {
            scaled[idx] = 0.0;
        }

        let sum: f32 = scaled.iter().sum();
        if sum > 0.0 {
            for v in scaled.iter_mut() {
                *v /= sum;
            }
        }
    }

    // 步骤 6：使用我们的 XorShift64 PRNG 进行 CDF 采样。
    // 在 [0, 1) 中抽取一个随机数 r，遍历累积分布，
    // 返回 cumsum > r 的第一个索引。
    let mut rng = XorShift64::new(config.seed.unwrap_or(12345));
    let r = rng.next_f32();
    let mut cumsum = 0.0f32;
    for (i, &p) in scaled.iter().enumerate() {
        cumsum += p;
        if cumsum > r {
            return i;
        }
    }

    // 回退：返回最后一个索引（处理浮点舍入）。
    scaled.len() - 1
}
```

### softmax_in_place

数值稳定的 softmax 实现：

```rust
fn softmax_in_place(logits: &mut [f32]) {
    // 步骤 1：找到最大值以确保数值稳定性。
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // 步骤 2：减去最大值并计算指数。
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }

    // 步骤 3：归一化。
    if sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    }
}
```

最大值减法技巧防止溢出：如果最大的 logit 是 1000，计算 `exp(1000)` 会溢出到无穷大。但 `exp(1000 - 1000) = exp(0) = 1`，这完全没问题。相对概率被保留，因为 softmax 是平移不变的：`softmax(z + c) = softmax(z)` 对于任意常数 c。

这个实现也正确处理 `-inf`：`exp(-inf - max) = exp(-inf) = 0`，所以来自 top-k 的过滤 token 变为概率零。

### CDF 采样

最后一步从 [0, 1) 中均匀抽取一个随机数 `r`，并遍历累积分布：

```rust
fn sample_from_cdf(probs: &[f32], rng: &mut XorShift64) -> usize {
    let r = rng.next_f32();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum > r {
            return i;
        }
    }
    probs.len() - 1  // 浮点舍入的回退
}
```

这是从分类分布中采样的方法。选择 token i 的概率恰好是 `probs[i]`，因为导致选择 token i 的 `r` 值区间的宽度为 `probs[i]`。

例如，如果分布是 [0.5, 0.3, 0.2]：

- r 在 [0, 0.5) 中选择 token 0（概率 0.5）
- r 在 [0.5, 0.8) 中选择 token 1（概率 0.3）
- r 在 [0.8, 1.0) 中选择 token 2（概率 0.2）

回退情况（`probs.len() - 1`）处理罕见的情况，即浮点舍入导致 `cumsum` 略小于 1.0，并且 `r` 落在 `cumsum` 和 1.0 之间的间隙中。在这种情况下，我们返回最后一个 token，这是最保守的选择。

### 边界情况

实现处理了几种边界情况：

- **Temperature 0**：短路到贪心解码，跳过所有其他步骤。这避免了除以零和不必要的计算。

- **top_k = 0**：禁用 top-k 过滤。没有 logits 被设为 `-inf`。

- **top_p = 1.0**：禁用 top-p 过滤。没有 token 被清零。

- **top_k >= vocab_size**：不需要过滤（所有 token 都是候选）。

- **所有 logits 相等**：经过 softmax 后，所有 token 具有相等的概率（1/vocab_size）。采样是真正均匀的。

- **单个非零 logit**：经过 softmax 后，一个 token 的概率为 1.0，所有其他 token 为 0。无论随机数是什么，CDF 采样总是返回这个 token。

### 独立函数

除了主要的 `sample` 函数外，我们还提供了每种采样策略的独立版本，可以单独使用：

- `sample_greedy(logits)`：贪心解码（argmax）。
- `sample_top_k(logits, k, rng)`：带显式 RNG 的 Top-k 采样。
- `sample_top_p(logits, p, rng)`：带显式 RNG 的 Top-p 采样。

这些对于实验和构建不同于默认顺序的自定义采样流程很有用。

### 测试

采样模块有全面的测试，验证：

- **贪心返回 argmax**：最高的 logit 总是获胜。
- **Temperature 0 等于贪心**：流程产生与 `sample_greedy` 相同的结果。
- **高温使分布平坦**：非常高的 temperature 产生接近均匀的分布。
- **低温使分布尖锐**：非常低的 temperature 将概率集中在顶部 token 上。
- **Top-k 限制候选**：过滤后只有前 k 个 token 具有非零概率。
- **Top-p 保持累积概率**：保留的 token 的累积概率至少为 `p`，移除最后一个保留的 token 会使累积概率低于 `p`。
- **XorShift64 是确定性的**：相同的种子产生相同的序列。
- **带种子的采样是确定性的**：相同的 logits、配置和种子产生相同的 token。
- **Softmax 处理 -inf**：过滤掉的 token 正确地得到概率 0。
- **Softmax 处理大值**：即使 logits 为 1000+ 也能数值稳定。
- **Softmax 总和为 1**：输出是有效的概率分布。

---

## 总结

| 策略 | 功能 | 使用场景 |
|----------|-------------|-------------|
| 贪心 | 始终选择概率最高的 token | 事实问答、调试 |
| Temperature | 缩放 logits 以控制分布形状 | 在质量与创造性之间权衡 |
| Top-k | 只保留 k 个最可能的 token | 防止采样极不可能的 token |
| Top-p | 保留累积概率覆盖 p 的最小集合 | 基于置信度的自适应过滤 |
| 组合 | Temperature、top-k、top-p，然后 CDF 采样 | 通用文本生成 |

采样流程让用户能够精细控制生成文本的特性。通过调整 temperature、top-k 和 top-p，你可以沿着从确定性、聚焦的输出到创造性、多样化的输出的光谱移动。XorShift64 PRNG 提供快速、可复现的随机性。这些组件共同将模型的原始 logits 转化为连贯、可控的文本。
