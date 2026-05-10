# 06 — 注意力机制：Transformer 的核心

注意力机制是赋予 Transformer 强大能力的核心。正是它让语言模型能够回溯句子中的前文，理解上下文、解析代词，并在长文本中保持连贯性。Transformer 中的每一个其他组件——嵌入层、前馈层、归一化层——都是为了支持注意力机制而存在的。

本章将从第一性原理出发解释注意力机制，然后逐步构建到 Grouped Query Attention (GQA) 与 KV 缓存——这正是 Qwen3-0.6B 所使用的变体。

---

## 1. 什么是注意力？

### 核心思想

想象你在读这个句子：

> "The cat sat on the mat because **it** was tired."

当你遇到单词 "it" 时，你的大脑不会孤立地看待它。你会自动回溯前面的词语，判断出 "it" 指的是 "the cat"，而不是 "the mat"。你**关注**了相关的上下文。

这正是注意力机制所做的事情：对于序列中的每个单词（token），它会查看所有其他单词，并决定每个单词对理解当前单词的重要程度。

### 图书馆类比

可以把注意力想象成在图书馆中搜索：

- **Query (Q)**：你要找什么——你的搜索问题。
  "我需要关于猫的信息。"
- **Key (K)**：每本书上的标签——每本书是关于什么的。
  一本书标着 "cats"，另一本标着 "dogs"，还有一本标着 "furniture。"
- **Value (V)**：每本书的实际内容。

注意力机制将你的 Query (Q) 与每个 Key (K) 进行比较以确定相关性，然后返回一个加权混合的 Value (V)。Key 与 Query 匹配的书获得更高的权重；不相关的书权重较低。

在**自注意力（self-attention）**中，Query、Key 和 Value 都来自**同一个**序列。每个 token 生成自己的 Q、K 和 V，然后每个 token 向所有其他 token "查询"。

---

## 2. 自注意力详解

### 步骤 1：Q、K、V 投影

每个 token 的隐藏状态向量（在 Qwen3-0.6B 中维度为 `hidden_size = 1024`）通过可学习的权重矩阵进行三次投影：

```
Q = x · W_q^T    → shape [seq_len, hidden_size]
K = x · W_k^T    → shape [seq_len, hidden_size]  (in MHA)
V = x · W_v^T    → shape [seq_len, hidden_size]  (in MHA)
```

这些投影将共享的表示转换为三个专门的角色：提问、提供标签和承载内容。

### 步骤 2：注意力分数

使用点积计算每次查询与每个 key 的对齐程度：

```
scores = Q · K^T    → shape [seq_len, seq_len]
```

结果是一个方阵，其中元素 `[i][j]` 衡量 token `i` 应该关注 token `j` 的程度。点积越大，表示对齐越强。

### 步骤 3：缩放

除以 `sqrt(d_k)`，其中 `d_k` 是 key 向量的维度：

```
scaled_scores = scores / sqrt(d_k)
```

为什么？维度为 `d` 的两个随机向量的点积，其方差与 `d` 成正比。如果不进行缩放，更大的维度会产生更大的分数，导致 softmax 进入饱和状态（非常尖锐、接近 one-hot 的分布）。除以 `sqrt(d_k)` 可以使方差保持在约 1，与维度无关。

### 步骤 4：Softmax

将每一行归一化为概率分布：

```
attn_weights = softmax(scaled_scores)    → shape [seq_len, seq_len]
```

每一行的和为 1。元素 `[i][j]` 是注意力权重：token `i` 对 token `j` 的关注程度。

### 步骤 5：加权求和

将注意力权重乘以 Value：

```
output = attn_weights · V    → shape [seq_len, hidden_size]
```

每个 token 的输出是所有 Value 向量的加权混合，权重由该 token 对其他每个 token 的关注程度决定。

### 一个具体的小例子

让我们用 4 个 token 和 4 维隐藏状态来追踪整个过程：

```
Input x (shape [4, 4]):
  Token 0: [1.0, 0.0, 0.0, 0.0]
  Token 1: [0.0, 1.0, 0.0, 0.0]
  Token 2: [0.0, 0.0, 1.0, 0.0]
  Token 3: [0.0, 0.0, 0.0, 1.0]
```

经过 Q、K、V 投影后（为简单起见使用恒等权重）：

```
Q = K = V = x

scores = Q · K^T:
  [[1, 0, 0, 0],
   [0, 1, 0, 0],
   [0, 0, 1, 0],
   [0, 0, 0, 1]]

scaled = scores / sqrt(4) = scores / 2:
  [[0.5, 0,   0,   0  ],
   [0,   0.5, 0,   0  ],
   [0,   0,   0.5, 0  ],
   [0,   0,   0,   0.5]]

attn_weights = softmax(scaled, dim=-1):
  [[1.0, 0,   0,   0  ],    ← Token 0 只关注自身
   [0,   1.0, 0,   0  ],    ← Token 1 只关注自身
   [0,   0,   1.0, 0  ],    ← Token 2 只关注自身
   [0,   0,   0,   1.0]]    ← Token 3 只关注自身

output = attn_weights · V = V = x
```

使用正交的 one-hot 输入和恒等投影时，每个 token 只关注自身。在实践中，通过可学习的投影和真实文本，注意力权重会分散到多个 token 上，形成丰富的上下文表示。

---

## 3. 多头注意力（MHA）

### 为什么需要多个头？

单个注意力头只能计算一组注意力模式。但语言有多种类型的关系：句法关系（主谓）、语义关系（词义）、位置关系（邻近词）和共指关系（代词解析）。单个头无法同时捕捉所有这些关系。

多头注意力通过将隐藏维度拆分为多个头来解决这个问题，每个头独立计算注意力。不同的头学会关注不同的模式。

### 工作原理

1. **投影**：像之前一样将输入投影到 Q、K、V（总维度 = `num_heads * head_dim = hidden_size`）。

2. **重塑**：拆分为独立的头：
   ```
   Q: [seq_len, hidden_size] → [seq_len, num_heads, head_dim]
   K: [seq_len, hidden_size] → [seq_len, num_heads, head_dim]
   V: [seq_len, hidden_size] → [seq_len, num_heads, head_dim]
   ```

3. **独立计算每个头的注意力**：
   ```
   For each head h:
     scores_h = Q_h · K_h^T / sqrt(head_dim)     → [seq_len, seq_len]
     weights_h = softmax(scores_h)                 → [seq_len, seq_len]
     output_h = weights_h · V_h                    → [seq_len, head_dim]
   ```

4. **拼接**头的输出：
   ```
   concat: [seq_len, num_heads, head_dim] → [seq_len, num_heads * head_dim]
   ```

5. **投影**：使用输出矩阵进行投影：
   ```
   output = concat · W_o^T    → [seq_len, hidden_size]
   ```

### Qwen3-0.6B 的形状追踪（MHA 变体）

```
Input:           [seq_len, 1024]
Q projection:    [seq_len, 2048]
K projection:    [seq_len, 1024]
V projection:    [seq_len, 1024]

Reshape Q:       [seq_len, 16, 128]
Reshape K:       [seq_len, 16, 128]
Reshape V:       [seq_len, 16, 128]

Scores:          [16, seq_len, seq_len]   (per head)
Attn weights:    [16, seq_len, seq_len]
Attn output:     [16, seq_len, 128]       (per head)

Concatenated:    [seq_len, 2048]
Output:          [seq_len, 1024]
```

---

## 4. 分组查询注意力（GQA）—— Qwen3 所使用的

### MHA 的问题

在 MHA 中，每个查询头都有自己的 key 头和 value 头。在自回归生成过程中，我们必须为每个头缓存 K 和 V。对于 Qwen3-0.6B：

```
KV cache per layer = 2 × num_heads × head_dim × seq_len × 4 bytes
                   = 2 × 16 × 128 × seq_len × 4
                   = 16,384 × seq_len bytes
```

在 `seq_len = 4096` 时，每层 **64 MB**，28 层总计 **1,792 MB**。对于更大的模型和更长的序列，这会变得难以承受。

### 谱系：MHA → MQA → GQA

| Variant | Q heads | K heads | V heads | KV cache size | Quality |
|---------|---------|---------|---------|---------------|---------|
| **MHA** | 16      | 16      | 16      | 1× (baseline) | Best    |
| **MQA** | 16      | 1       | 1       | 1/16×         | Good    |
| **GQA** | 16      | 8       | 8       | 1/2×          | Better  |

**多头注意力（MHA）**：标准版本。每个查询头都有自己的 K 和 V 头（比例 1:1:1）。质量最高，但 KV 缓存与头数成正比。

**多查询注意力（MQA）**：所有查询头共享一个 K 头和一个 V 头（比例 N:1:1）。KV 缓存非常小（MHA 的 1/N），但由于一组 K/V 无法很好地服务于所有查询模式，质量会下降。

**分组查询注意力（GQA）**：一种折中方案。查询头被分成若干组，每组共享一个 K 头和一个 V 头（比例 N:G:1）。与 MHA 相比，KV 缓存减少了 `num_heads/num_kv_heads` 倍，同时质量仍接近 MHA。

### Qwen3-0.6B 的 GQA 配置

```
num_attention_heads  = 16    (query heads)
num_key_value_heads  = 8     (KV heads)
kv_groups            = 16 / 8 = 2

每个 KV 头由 2 个查询头共享：
  Q heads 0, 1  →  KV head 0
  Q heads 2, 3  →  KV head 1
  Q heads 4, 5  →  KV head 2
  ...
  Q heads 14, 15 → KV head 7
```

### "Repeat KV" 步骤

为了使 GQA 能够与 MHA 使用相同的注意力代码，我们只需在计算注意力之前将每个 KV 头重复 `kv_groups` 次。这将 KV 张量从 `[seq_len, num_kv_heads, head_dim]` 转换为 `[seq_len, num_heads, head_dim]`：

```
Before expansion (8 KV heads):
  K: [seq_len, 8, 128]

After expansion (16 heads):
  K: [seq_len, 16, 128]
     Heads 0,1 are copies of KV head 0
     Heads 2,3 are copies of KV head 1
     ...

Value tensor is expanded the same way.
```

这种扩展不会增加 KV 缓存——我们只存储 8 个唯一的头，并在注意力计算时即时扩展。

```
┌──────────────────────────────────────────────┐
│            GQA Expansion Diagram             │
│                                              │
│  KV Heads (stored, 8 heads):                │
│  [h0] [h1] [h2] [h3] [h4] [h5] [h6] [h7]  │
│                                              │
│         ↓ repeat each 2× ↓                   │
│                                              │
│  Q Heads (after expansion, 16 heads):       │
│  [h0] [h0] [h1] [h1] [h2] [h2] ... [h7][h7]│
│   ↑    ↑                                         │
│   Q0   Q1    (both use KV head 0)               │
└──────────────────────────────────────────────┘
```

### KV 缓存节省

使用 GQA，缓存只存储 `num_kv_heads` 而不是 `num_heads`：

```
KV cache per layer = 2 × num_kv_heads × head_dim × seq_len × 4 bytes
                   = 2 × 8 × 128 × seq_len × 4
                   = 8,192 × seq_len bytes
```

在 `seq_len = 4096` 时：**每层约 32 MB**，**28 层约 896 MB**。与 MHA 相比减少了 50%。

---

## 5. 因果掩码（Causal Masking）

### 为什么需要因果掩码？

语言模型从左到右一次生成一个 token。在训练过程中，模型必须学会仅使用前面的 token 来预测下一个 token。如果位置 3 的 token 能够看到位置 5 的 token，模型就会"作弊"——它在预测之前就看到了答案。

因果掩码通过确保每个位置只能关注它自身及之前的位置来防止这种情况。

### 掩码矩阵

对于长度为 4 的序列，因果掩码如下所示：

```
        Key Position
       0   1   2   3
    ┌───────────────────
  0 │ ✓   ✗   ✗   ✗      Token 0 只看到自身
Q 1 │ ✓   ✓   ✗   ✗      Token 1 看到 token 0, 1
  2 │ ✓   ✓   ✓   ✗      Token 2 看到 token 0, 1, 2
  3 │ ✓   ✓   ✓   ✓      Token 3 看到 token 0, 1, 2, 3
```

在实现中，"✗" 位置在 softmax 之前被设置为负无穷（`-inf`）。由于 `softmax(-inf) = 0`，这些位置对加权求和没有贡献。

掩码是一个**下三角**矩阵：

```
    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]
```

### 不使用掩码会怎样？

如果没有因果掩码，每个 token 都可以关注所有其他 token，包括未来的 token。模型在训练期间看到了答案，因此永远学不会预测。在推理时，未来的 token 尚不存在，造成训练-测试不匹配。模型无法生成连贯的文本。

### Prefill 与 Decode 阶段的因果掩码

在 **prefill** 阶段（处理提示词，`seq_len > 1`），我们需要标准的下三角掩码：每个查询位置 `i` 可以关注 key 位置 `0..=i`。

在 **decode** 阶段（一次生成一个 token，`seq_len = 1`），新 token 位于最后一个位置。它可以关注所有缓存的位置以及自身。不需要掩码，因为缓存中没有"未来"的 token。

---

## 6. KV 缓存——关键优化

### 问题：O(n²) 重复计算

如果没有缓存，生成第 N 个 token 需要为所有之前的 N-1 个 token 加上新 token 计算 K 和 V。注意力计算本身在序列长度上是 O(N²) 的，因为每个查询都要关注每个 key。在没有缓存的情况下生成 1000 个 token 需要约 500,000 次注意力计算——而且每次都在重复已经完成的工作。

### 解决方案：缓存与增量计算

KV 缓存存储所有之前处理过的 token 的 K 和 V 向量。当生成新 token 时：

1. 仅为**新** token 计算 K 和 V（而不是整个序列）。
2. 将新的 K 和 V 追加到缓存中。
3. 计算新 token 的 Q 与完整缓存的 K、V 之间的注意力。

这将每步的成本从 O(N²) 降低到 O(N)——对于长序列来说是一个巨大的加速。

### Prefill 与 Decode

```
┌─────────────────────────────────────────────────────┐
│                    PREFILL                           │
│                                                     │
│  Input: "The cat sat on" (4 tokens)                 │
│  Process all 4 tokens at once.                      │
│  Compute Q, K, V for all 4 tokens.                  │
│  Attention: each token attends to preceding ones.    │
│  Cache: store K[0..4], V[0..4]                      │
│                                                     │
│  Q: [4, hidden]  K: [4, hidden]  V: [4, hidden]    │
│  Scores: [4, 4]  → 4×4 attention matrix             │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                    DECODE (step 1)                   │
│                                                     │
│  New token: "the" (token 5)                         │
│  Compute Q, K, V only for token 5.                  │
│  Append K[5], V[5] to cache.                        │
│  Attention: Q[5] attends to K[0..5].                │
│                                                     │
│  Q: [1, hidden]  K: [5, hidden]  V: [5, hidden]    │
│  Scores: [1, 5]  → just one row of attention        │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                    DECODE (step 2)                   │
│                                                     │
│  New token: "mat" (token 6)                         │
│  Compute Q, K, V only for token 6.                  │
│  Append K[6], V[6] to cache.                        │
│  Attention: Q[6] attends to K[0..6].                │
│                                                     │
│  Q: [1, hidden]  K: [6, hidden]  V: [6, hidden]    │
│  Scores: [1, 6]  → just one row of attention        │
└─────────────────────────────────────────────────────┘
```

### KV 缓存的内存成本

KV 缓存为每一层存储两个张量（K 和 V）。每个张量的形状为 `[seq_len, num_kv_heads, head_dim]`。每层的内存：

```
KV cache per layer = 2 × num_kv_heads × head_dim × seq_len × sizeof(f32)
                   = 2 × 8 × 128 × seq_len × 4 bytes
                   = 8,192 × seq_len bytes
```

对于拥有 28 层的 Qwen3-0.6B：

| Sequence Length | Per Layer | Total (28 layers) |
|----------------|-----------|-------------------|
| 512            | 4 MB      | 112 MB            |
| 1024           | 8 MB      | 224 MB            |
| 2048           | 16 MB     | 448 MB            |
| 4096           | 32 MB     | 896 MB            |
| 8192           | 64 MB     | 1,792 MB          |

缓存随序列长度线性增长。对于更长的序列或更大的模型（拥有更多层和头），缓存会消耗大量的 GPU/CPU 内存。

### 我们实现中的 KV 缓存

在我们的 Rust 代码中，`KVCache` 结构体将 K 和 V 存储为 2-D 张量（展平的头维度），以便高效地进行行拼接：

```rust
pub struct KVCache {
    pub key_cache:   Option<Tensor>,  // [seq_len_so_far, num_kv_heads * head_dim]
    pub value_cache: Option<Tensor>,  // [seq_len_so_far, num_kv_heads * head_dim]
}
```

在第一次前向传播（prefill）时，缓存为 `None`，我们直接存储计算出的 K 和 V。在后续传播（decode）时，我们使用 `stack_rows` 将新的 K 和 V 行追加到现有缓存中。

---

## 7. 逐步计算追踪

让我们用 Qwen3-0.6B 的维度来走一遍完整的注意力前向传播过程。我们将追踪一个 decode 步骤：处理位置 5 的 token，其中已有 5 个缓存的 token。

### 初始状态

```
KV Cache contains positions 0-4:
  key_cache:   [5, 1024]    (5 tokens, 8 KV heads × 128 head_dim)
  value_cache: [5, 1024]
```

### 步骤 1：将输入投影到 Q、K、V

```
Input x:  [1, 1024]

Q = x · W_q^T:  [1, 1024] × [1024, 2048] → [1, 2048]
K = x · W_k^T:  [1, 1024] × [1024, 1024]  → [1, 1024]
V = x · W_v^T:  [1, 1024] × [1024, 1024]  → [1, 1024]
```

### 步骤 2：重塑为独立的头

```
Q: [1, 2048] → [1, 16, 128]    (1 token, 16 query heads, 128 dims each)
K: [1, 1024] → [1, 8, 128]     (1 token, 8 KV heads, 128 dims each)
V: [1, 1024] → [1, 8, 128]
```

### 步骤 3：应用 RoPE

```
Q: [1, 16, 128]  → [1, 16, 128]   (rotated by position 5)
K: [1, 8, 128]   → [1, 8, 128]    (rotated by position 5)
```

### 步骤 4：更新 KV 缓存

```
K flattened:  [1, 8, 128] → [1, 1024]
V flattened:  [1, 8, 128] → [1, 1024]

Cache after concatenation:
  key_cache:   [5, 1024] stack_rows [1, 1024] → [6, 1024]
  value_cache: [5, 1024] stack_rows [1, 1024] → [6, 1024]

Reshape for attention:
  K_full: [6, 1024] → [6, 8, 128]
  V_full: [6, 1024] → [6, 8, 128]
```

### 步骤 5：为 GQA 扩展 KV 头

```
K: [6, 8, 128] → [6, 16, 128]    (each of 8 KV heads repeated 2×)
V: [6, 8, 128] → [6, 16, 128]
```

### 步骤 6：计算注意力分数

```
Transpose Q: [1, 16, 128] → [16, 1, 128]
Transpose K: [6, 16, 128] → [16, 6, 128]

scores = Q · K^T:  [16, 1, 128] × [16, 128, 6] → [16, 1, 6]
         (per head: [1, 128] × [128, 6] → [1, 6])

scaled = scores / sqrt(128) = scores / 11.314
```

### 步骤 7：应用因果掩码

```
Position 5 can attend to positions 0-5 (all 6 cached tokens).
No positions are masked during decode (q_pos = 5 >= all k positions).
```

### 步骤 8：Softmax

```
attn_weights = softmax(scaled, dim=2):  [16, 1, 6]
Each head's row of 6 values sums to 1.
```

### 步骤 9：计算输出

```
V transposed: [6, 16, 64] → [16, 6, 64]
attn_output = attn_weights · V:  [16, 1, 6] × [16, 6, 64] → [16, 1, 64]
```

### 步骤 10：重塑与投影

```
Transpose back: [16, 1, 64] → [1, 16, 64]
Flatten heads:  [1, 16, 64] → [1, 1024]
Output = attn_flat · W_o^T:  [1, 1024] × [1024, 1024] → [1, 1024]
```

### 完整形状摘要

```
Input:             [1, 1024]
After Q proj:      [1, 1024] → reshape [1, 16, 64]
After K proj:      [1, 512]  → reshape [1, 8, 64]
After V proj:      [1, 512]  → reshape [1, 8, 64]
After RoPE:        same shapes, values rotated
After KV cache:    K [6, 8, 64], V [6, 8, 64]
After GQA expand:  K [6, 16, 64], V [6, 16, 64]
Scores:            [16, 1, 6]
Attn weights:      [16, 1, 6]
Attn output:       [16, 1, 64] → [1, 16, 64] → [1, 1024]
Final output:      [1, 1024]
```

---

## 8. 实现细节

### 结构体定义

我们的 Rust 实现包含两个主要结构体：

```rust
pub struct Attention {
    q_proj: Tensor,     // [hidden_size, hidden_size]
    k_proj: Tensor,     // [kv_dim, hidden_size]
    v_proj: Tensor,     // [kv_dim, hidden_size]
    o_proj: Tensor,     // [hidden_size, num_heads * head_dim]
    num_heads: usize,   // 16
    num_kv_heads: usize, // 8
    head_dim: usize,    // 128
    kv_groups: usize,   // 2
    cos_table: Tensor,  // [max_seq_len, head_dim/2]
    sin_table: Tensor,  // [max_seq_len, head_dim/2]
}

pub struct KVCache {
    pub key_cache:   Option<Tensor>,  // [seq_len_so_far, kv_dim]
    pub value_cache: Option<Tensor>,  // [seq_len_so_far, kv_dim]
}
```

### Forward 方法逐步说明

`forward` 方法实现了第 7 节中描述的 10 步过程。以下是关键的实现决策：

**权重转置**：safetensors 格式以 `[out_features, in_features]` 存储权重。我们在 matmul 之前转置每个权重，以便 `x · W^T` 产生正确的投影。我们使用添加到 `Tensor` 的 `transpose_2d` 方法。

**KV 缓存使用 2-D**：我们将缓存存储为 `[seq_len, kv_dim]` 而不是 `[seq_len, num_kv_heads, head_dim]`，因为它能够通过 `stack_rows` 高效地进行行拼接。我们仅在注意力计算需要时重塑为 3-D。

**GQA 扩展**：`expand_kv_heads` 函数接受一个 3-D 张量 `[seq_len, num_kv_heads, head_dim]`，并通过将每个 KV 头重复 `kv_groups` 次来生成 `[seq_len, num_heads, head_dim]`。模式如下：

```
KV head 0 → Q heads 0, 1
KV head 1 → Q heads 2, 3
...
KV head i → Q heads i*kv_groups, i*kv_groups+1, ..., (i+1)*kv_groups-1
```

**批量 matmul**：由于我们的 `Tensor::matmul` 只支持 2-D，我们实现了自定义的批量 matmul 函数（`batch_matmul_qk` 和 `batch_matmul_attn_v`），它们显式地遍历各个头。每个头的计算都是标准的 2-D matmul。

**因果掩码**：`apply_causal_mask` 函数根据 `start_pos`（当前输入的位置偏移）确定哪些位置是"未来"的。在 prefill 期间，这会产生标准的下三角掩码。在 decode 期间，没有位置被掩码，因为单个新 token 位于序列的末尾。

**头转置**：我们使用 `transpose_heads` 和 `untranspose_heads` 在 `[seq_len, heads, dim]` 布局（适合投影和缓存）和 `[heads, seq_len, dim]` 布局（适合批量 matmul）之间转置 Q、K、V。

### 使用的关键张量操作

| Operation | Method | Purpose |
|-----------|--------|---------|
| Matrix multiply | `matmul` | Q/K/V projections, attention scores, output projection |
| Transpose 2-D | `transpose_2d` | Prepare weights for `x · W^T` |
| Row concatenation | `stack_rows` | KV cache update |
| Reshape | `reshape` | Separate/merge head dimensions |
| Row slicing | `rows` | Slice RoPE cos/sin tables for current positions |
| Scalar multiply | `mul_scalar` | Scale attention scores by `1/sqrt(d_k)` |
| Softmax | `softmax` | Normalize attention weights |
| RoPE | `apply_rope` | Apply positional encoding to Q and K |

### 性能考虑

我们的实现优先考虑清晰性而非速度。生产级的注意力实现会使用：

- **Flash Attention**：将 score-compute-softmax-multiply 流水线融合为单个 GPU kernel，避免生成完整的 `[heads, seq_len, seq_len]` 注意力矩阵。这节省了 O(N²) 内存，在 GPU 上速度更快。

- **Paged KV Cache**：不存储为一个连续的张量，而是将缓存块存储在不连续的页面中。这避免了内存碎片，并支持序列间的高效内存共享（例如用于 beam search）。

- **Optimized matmul**：使用 BLAS（例如 GPU 上的 cuBLAS，CPU 上的 OpenBLAS）而不是三重循环的 matmul。我们的实现使用教科书式算法，虽然清晰但比优化库慢约 100 倍。

- **Quantized KV Cache**：以更低精度（int8 甚至 int4）存储 K 和 V，使缓存内存减半或减至四分之一。注意力计算仍在 fp16 或 fp32 中进行。

这些优化复杂且特定于硬件，因此我们保持实现简单且具有教育意义。无论优化级别如何，数学运算都是相同的。

---

## 总结

| Concept | Key Idea |
|---------|----------|
| Self-Attention | Each token attends to all other tokens via Q·K^T similarity |
| Multi-Head | Split into multiple heads to capture different relationship types |
| GQA | Share KV heads across groups of Q heads to reduce cache size |
| Causal Mask | Prevent attending to future tokens during training/generation |
| KV Cache | Store past K,V to avoid recomputation during autoregressive decoding |

注意力机制是让 Transformer 工作的核心机制。每一个其他组件——嵌入层、RMSNorm、RoPE、前馈层——都是为了让输入准备好进入注意力机制，或者处理其输出。理解注意力就是理解 Transformer。
