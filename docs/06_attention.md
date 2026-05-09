# 06 — Attention: The Heart of the Transformer

Attention is the mechanism that gives transformers their power. It is the
reason a language model can look back at earlier words in a sentence to
understand context, resolve pronouns, and maintain coherence over long
passages. Every other component in a transformer — the embeddings, the
feed-forward layers, the normalization — exists to support the attention
mechanism.

This chapter explains attention from first principles, then builds up to
Grouped Query Attention (GQA) with KV caching — the exact variant used in
Qwen3-0.6B.

---

## 1. What Is Attention?

### The Core Idea

Imagine you are reading the sentence:

> "The cat sat on the mat because **it** was tired."

When you encounter the word "it," your brain does not treat it in
isolation. You automatically look back at the preceding words and
determine that "it" refers to "the cat," not "the mat." You pay
**attention** to the relevant context.

This is exactly what the attention mechanism does: for each word (token)
in a sequence, it looks at all other words and decides how much each one
matters for understanding the current word.

### The Library Metaphor

Think of attention like searching a library:

- **Query (Q)**: What you are looking for — your search question.
  "I need information about cats."
- **Key (K)**: The labels on each book — what each book is about.
  One book is labeled "cats," another "dogs," another "furniture."
- **Value (V)**: The actual content of each book.

The attention mechanism compares your query (Q) against every key (K) to
determine relevance, then returns a weighted mixture of the values (V).
Books whose keys match your query get more weight; irrelevant books get
less.

In **self-attention**, the queries, keys, and values all come from the
**same** sequence. Each token generates its own Q, K, and V, and then
every token "queries" every other token.

---

## 2. Self-Attention in Detail

### Step 1: Q, K, V Projections

Each token's hidden state vector (dimension `hidden_size = 1024` in
Qwen3-0.6B) is projected three times using learned weight matrices:

```
Q = x · W_q^T    → shape [seq_len, hidden_size]
K = x · W_k^T    → shape [seq_len, hidden_size]  (in MHA)
V = x · W_v^T    → shape [seq_len, hidden_size]  (in MHA)
```

These projections transform the shared representation into three
specialized roles: asking questions, providing labels, and carrying
content.

### Step 2: Attention Scores

Compute how much each query aligns with each key using a dot product:

```
scores = Q · K^T    → shape [seq_len, seq_len]
```

The result is a square matrix where entry `[i][j]` measures how much
token `i` should attend to token `j`. A larger dot product means
stronger alignment.

### Step 3: Scaling

Divide by `sqrt(d_k)` where `d_k` is the dimension of the key vectors:

```
scaled_scores = scores / sqrt(d_k)
```

Why? The dot product of two random vectors of dimension `d` has variance
proportional to `d`. Without scaling, larger dimensions produce larger
scores, which pushes softmax into saturation (very sharp, near-one-hot
distributions). Dividing by `sqrt(d_k)` keeps the variance at ~1
regardless of dimension.

### Step 4: Softmax

Normalize each row into a probability distribution:

```
attn_weights = softmax(scaled_scores)    → shape [seq_len, seq_len]
```

Each row sums to 1. Entry `[i][j]` is the attention weight: how much
token `i` attends to token `j`.

### Step 5: Weighted Sum

Multiply attention weights by values:

```
output = attn_weights · V    → shape [seq_len, hidden_size]
```

Each token's output is a weighted mixture of all value vectors, weighted
by how much that token attended to each other token.

### A Concrete Small Example

Let us trace through with 4 tokens and 4-dimensional hidden states:

```
Input x (shape [4, 4]):
  Token 0: [1.0, 0.0, 0.0, 0.0]
  Token 1: [0.0, 1.0, 0.0, 0.0]
  Token 2: [0.0, 0.0, 1.0, 0.0]
  Token 3: [0.0, 0.0, 0.0, 1.0]
```

After Q, K, V projections (with identity weights for simplicity):

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
  [[1.0, 0,   0,   0  ],    ← Token 0 attends only to itself
   [0,   1.0, 0,   0  ],    ← Token 1 attends only to itself
   [0,   0,   1.0, 0  ],    ← Token 2 attends only to itself
   [0,   0,   0,   1.0]]    ← Token 3 attends only to itself

output = attn_weights · V = V = x
```

With orthogonal one-hot inputs and identity projections, each token
attends only to itself. In practice, with learned projections and real
text, the attention weights spread across multiple tokens, creating rich
contextual representations.

---

## 3. Multi-Head Attention (MHA)

### Why Multiple Heads?

A single attention head computes one set of attention patterns. But
language has many types of relationships: syntactic (subject-verb),
semantic (word meaning), positional (nearby words), and coreference
(pronoun resolution). A single head cannot capture all of these
simultaneously.

Multi-Head Attention solves this by splitting the hidden dimension into
multiple heads, each of which independently computes attention. Different
heads learn to focus on different patterns.

### How It Works

1. **Project** the input to Q, K, V as before (total dimension =
   `num_heads * head_dim = hidden_size`).

2. **Reshape** into separate heads:
   ```
   Q: [seq_len, hidden_size] → [seq_len, num_heads, head_dim]
   K: [seq_len, hidden_size] → [seq_len, num_heads, head_dim]
   V: [seq_len, hidden_size] → [seq_len, num_heads, head_dim]
   ```

3. **Compute attention** independently for each head:
   ```
   For each head h:
     scores_h = Q_h · K_h^T / sqrt(head_dim)     → [seq_len, seq_len]
     weights_h = softmax(scores_h)                 → [seq_len, seq_len]
     output_h = weights_h · V_h                    → [seq_len, head_dim]
   ```

4. **Concatenate** the head outputs:
   ```
   concat: [seq_len, num_heads, head_dim] → [seq_len, num_heads * head_dim]
   ```

5. **Project** with the output matrix:
   ```
   output = concat · W_o^T    → [seq_len, hidden_size]
   ```

### Shape Trace for Qwen3-0.6B (MHA variant)

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

## 4. Grouped Query Attention (GQA) — What Qwen3 Uses

### The Problem with MHA

In MHA, every query head has its own key head and value head. During
autoregressive generation, we must cache the K and V for every head. For
Qwen3-0.6B:

```
KV cache per layer = 2 × num_heads × head_dim × seq_len × 4 bytes
                   = 2 × 16 × 128 × seq_len × 4
                   = 16,384 × seq_len bytes
```

At `seq_len = 4096`, that is **64 MB per layer**, and **1,792 MB for 28
layers**. For larger models and longer sequences, this becomes
prohibitive.

### The Spectrum: MHA → MQA → GQA

| Variant | Q heads | K heads | V heads | KV cache size | Quality |
|---------|---------|---------|---------|---------------|---------|
| **MHA** | 16      | 16      | 16      | 1× (baseline) | Best    |
| **MQA** | 16      | 1       | 1       | 1/16×         | Good    |
| **GQA** | 16      | 8       | 8       | 1/2×          | Better  |

**Multi-Head Attention (MHA)**: The standard. Each query head has its
own K and V head (ratio 1:1:1). Highest quality, but the KV cache is
proportional to the number of heads.

**Multi-Query Attention (MQA)**: All query heads share a single K head
and a single V head (ratio N:1:1). The KV cache is tiny (1/N of MHA),
but quality degrades because one set of K/V cannot serve all query
patterns well.

**Grouped Query Attention (GQA)**: A compromise. Query heads are divided
into groups, and each group shares one K head and one V head (ratio
N:G:1). The KV cache is reduced by a factor of `num_heads/num_kv_heads`
compared to MHA, while quality remains close to MHA.

### Qwen3-0.6B's GQA Configuration

```
num_attention_heads  = 16    (query heads)
num_key_value_heads  = 8     (KV heads)
kv_groups            = 16 / 8 = 2

Each KV head is shared by 2 query heads:
  Q heads 0, 1  →  KV head 0
  Q heads 2, 3  →  KV head 1
  Q heads 4, 5  →  KV head 2
  ...
  Q heads 14, 15 → KV head 7
```

### The "Repeat KV" Step

To make GQA work with the same attention code as MHA, we simply repeat
each KV head `kv_groups` times before computing attention. This
transforms the KV tensors from `[seq_len, num_kv_heads, head_dim]` to
`[seq_len, num_heads, head_dim]`:

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

This expansion does not increase the KV cache — we only store 8 unique
heads and expand on the fly during attention computation.

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

### KV Cache Savings

With GQA, the cache stores only `num_kv_heads` instead of `num_heads`:

```
KV cache per layer = 2 × num_kv_heads × head_dim × seq_len × 4 bytes
                   = 2 × 8 × 128 × seq_len × 4
                   = 8,192 × seq_len bytes
```

At `seq_len = 4096`: **~32 MB per layer**, **~896 MB for 28 layers**.
That is a 50% reduction compared to MHA.

---

## 5. Causal Masking

### Why Causal Masking?

Language models generate text one token at a time, from left to right.
During training, the model must learn to predict the next token using
only the tokens that precede it. If token at position 3 could see token
at position 5, the model would "cheat" — it would see the answer before
predicting it.

Causal masking prevents this by ensuring each position can only attend
to positions at or before it.

### The Mask Matrix

For a sequence of length 4, the causal mask looks like this:

```
        Key Position
       0   1   2   3
    ┌───────────────────
  0 │ ✓   ✗   ✗   ✗      Token 0 sees only itself
Q 1 │ ✓   ✓   ✗   ✗      Token 1 sees tokens 0, 1
  2 │ ✓   ✓   ✓   ✗      Token 2 sees tokens 0, 1, 2
  3 │ ✓   ✓   ✓   ✓      Token 3 sees tokens 0, 1, 2, 3
```

In implementation, the "✗" positions are set to negative infinity
(`-inf`) before softmax. Since `softmax(-inf) = 0`, these positions
contribute nothing to the weighted sum.

The mask is a **lower-triangular** matrix:

```
    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]
```

### What Happens Without Masking?

Without the causal mask, every token can attend to every other token,
including future tokens. The model sees the answer during training, so
it never learns to predict. At inference time, future tokens do not
exist yet, creating a train-test mismatch. The model fails to generate
coherent text.

### Causal Masking During Prefill vs. Decode

During **prefill** (processing the prompt, `seq_len > 1`), we need the
standard lower-triangular mask: each query position `i` can attend to
key positions `0..=i`.

During **decode** (generating one token at a time, `seq_len = 1`), the
new token is at the last position. It can attend to all cached positions
plus itself. No masking is needed because there are no "future" tokens
in the cache.

---

## 6. KV Cache — The Key Optimization

### The Problem: O(n²) Recomputation

Without a cache, generating the Nth token requires computing K and V
for all N-1 previous tokens plus the new one. The attention computation
itself is O(N²) in sequence length because every query attends to every
key. Generating 1000 tokens without caching would require ~500,000
attention computations — and each one repeats work already done.

### The Solution: Cache and Increment

The KV cache stores the K and V vectors for all previously processed
tokens. When generating a new token:

1. Compute K and V only for the **new** token (not the whole sequence).
2. Append the new K and V to the cache.
3. Compute attention between the new token's Q and the full cached K, V.

This reduces the per-step cost from O(N²) to O(N) — a huge speedup for
long sequences.

### Prefill vs. Decode

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

### Memory Cost of the KV Cache

The KV cache stores two tensors (K and V) for each layer. Each tensor
has shape `[seq_len, num_kv_heads, head_dim]`. The memory per layer:

```
KV cache per layer = 2 × num_kv_heads × head_dim × seq_len × sizeof(f32)
                   = 2 × 8 × 128 × seq_len × 4 bytes
                   = 8,192 × seq_len bytes
```

For Qwen3-0.6B with 28 layers:

| Sequence Length | Per Layer | Total (28 layers) |
|----------------|-----------|-------------------|
| 512            | 4 MB      | 112 MB            |
| 1024           | 8 MB      | 224 MB            |
| 2048           | 16 MB     | 448 MB            |
| 4096           | 32 MB     | 896 MB            |
| 8192           | 64 MB     | 1,792 MB          |

The cache grows linearly with sequence length. For longer sequences or
larger models (which have more layers and heads), the cache can consume
significant GPU/CPU memory.

### KV Cache in Our Implementation

In our Rust code, the `KVCache` struct stores K and V as 2-D tensors
(flattened head dimensions) for efficient row concatenation:

```rust
pub struct KVCache {
    pub key_cache:   Option<Tensor>,  // [seq_len_so_far, num_kv_heads * head_dim]
    pub value_cache: Option<Tensor>,  // [seq_len_so_far, num_kv_heads * head_dim]
}
```

On the first forward pass (prefill), the cache is `None` and we store
the computed K and V directly. On subsequent passes (decode), we use
`stack_rows` to append the new K and V rows to the existing cache.

---

## 7. Step-by-Step Computation Trace

Let us walk through the full attention forward pass with Qwen3-0.6B
dimensions. We will trace a single decode step: processing token at
position 5 with 5 cached tokens.

### Initial State

```
KV Cache contains positions 0-4:
  key_cache:   [5, 1024]    (5 tokens, 8 KV heads × 128 head_dim)
  value_cache: [5, 1024]
```

### Step 1: Project Input to Q, K, V

```
Input x:  [1, 1024]

Q = x · W_q^T:  [1, 1024] × [1024, 2048] → [1, 2048]
K = x · W_k^T:  [1, 1024] × [1024, 1024]  → [1, 1024]
V = x · W_v^T:  [1, 1024] × [1024, 1024]  → [1, 1024]
```

### Step 2: Reshape to Separate Heads

```
Q: [1, 2048] → [1, 16, 128]    (1 token, 16 query heads, 128 dims each)
K: [1, 1024] → [1, 8, 128]     (1 token, 8 KV heads, 128 dims each)
V: [1, 1024] → [1, 8, 128]
```

### Step 3: Apply RoPE

```
Q: [1, 16, 128]  → [1, 16, 128]   (rotated by position 5)
K: [1, 8, 128]   → [1, 8, 128]    (rotated by position 5)
```

### Step 4: Update KV Cache

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

### Step 5: Expand KV Heads for GQA

```
K: [6, 8, 128] → [6, 16, 128]    (each of 8 KV heads repeated 2×)
V: [6, 8, 128] → [6, 16, 128]
```

### Step 6: Compute Attention Scores

```
Transpose Q: [1, 16, 128] → [16, 1, 128]
Transpose K: [6, 16, 128] → [16, 6, 128]

scores = Q · K^T:  [16, 1, 128] × [16, 128, 6] → [16, 1, 6]
         (per head: [1, 128] × [128, 6] → [1, 6])

scaled = scores / sqrt(128) = scores / 11.314
```

### Step 7: Apply Causal Mask

```
Position 5 can attend to positions 0-5 (all 6 cached tokens).
No positions are masked during decode (q_pos = 5 >= all k positions).
```

### Step 8: Softmax

```
attn_weights = softmax(scaled, dim=2):  [16, 1, 6]
Each head's row of 6 values sums to 1.
```

### Step 9: Compute Output

```
V transposed: [6, 16, 64] → [16, 6, 64]
attn_output = attn_weights · V:  [16, 1, 6] × [16, 6, 64] → [16, 1, 64]
```

### Step 10: Reshape and Project

```
Transpose back: [16, 1, 64] → [1, 16, 64]
Flatten heads:  [1, 16, 64] → [1, 1024]
Output = attn_flat · W_o^T:  [1, 1024] × [1024, 1024] → [1, 1024]
```

### Full Shape Summary

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

## 8. Implementation Details

### Struct Definitions

Our Rust implementation consists of two main structs:

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

### Forward Method Step by Step

The `forward` method implements the 10-step process described in Section
7. Here are the key implementation decisions:

**Weight transposition**: The safetensors format stores weights as
`[out_features, in_features]`. We transpose each weight before the
matmul so that `x · W^T` produces the correct projection. We use the
`transpose_2d` method added to `Tensor`.

**KV cache as 2-D**: We store the cache as `[seq_len, kv_dim]` rather
than `[seq_len, num_kv_heads, head_dim]` because it enables efficient
row concatenation via `stack_rows`. We reshape to 3-D only when needed
for the attention computation.

**GQA expansion**: The `expand_kv_heads` function takes a 3-D tensor
`[seq_len, num_kv_heads, head_dim]` and produces
`[seq_len, num_heads, head_dim]` by repeating each KV head `kv_groups`
times. The pattern is:

```
KV head 0 → Q heads 0, 1
KV head 1 → Q heads 2, 3
...
KV head i → Q heads i*kv_groups, i*kv_groups+1, ..., (i+1)*kv_groups-1
```

**Batched matmul**: Since our `Tensor::matmul` only supports 2-D, we
implement custom batched matmul functions (`batch_matmul_qk` and
`batch_matmul_attn_v`) that loop over heads explicitly. Each head's
computation is a standard 2-D matmul.

**Causal masking**: The `apply_causal_mask` function determines which
positions are "future" based on `start_pos` (the position offset of the
current input). During prefill, this produces a standard lower-
triangular mask. During decode, no positions are masked because the
single new token is at the end of the sequence.

**Head transposition**: We transpose Q, K, V between the
`[seq_len, heads, dim]` layout (natural for projection and caching) and
the `[heads, seq_len, dim]` layout (natural for batched matmul) using
`transpose_heads` and `untranspose_heads`.

### Key Tensor Operations Used

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

### Performance Considerations

Our implementation prioritizes clarity over speed. A production-grade
attention implementation would use:

- **Flash Attention**: Fuses the score-compute-softmax-multiply pipeline
  into a single GPU kernel, avoiding materializing the full
  `[heads, seq_len, seq_len]` attention matrix. This saves O(N²) memory
  and is much faster on GPU.

- **Paged KV Cache**: Instead of one contiguous tensor, store cache
  blocks in non-contiguous pages. This avoids memory fragmentation and
  enables efficient memory sharing between sequences (e.g., for
  beam search).

- **Optimized matmul**: Use BLAS (e.g., cuBLAS on GPU, OpenBLAS on
  CPU) instead of triple-loop matmul. Our implementation uses the
  textbook algorithm which is clear but ~100× slower than optimized
  libraries.

- **Quantized KV Cache**: Store K and V in lower precision (int8 or
  even int4) to halve or quarter the cache memory. The attention
  computation is still done in fp16 or fp32.

These optimizations are complex and hardware-specific, which is why we
keep the implementation simple and educational. The mathematical
operations are identical regardless of the optimization level.

---

## Summary

| Concept | Key Idea |
|---------|----------|
| Self-Attention | Each token attends to all other tokens via Q·K^T similarity |
| Multi-Head | Split into multiple heads to capture different relationship types |
| GQA | Share KV heads across groups of Q heads to reduce cache size |
| Causal Mask | Prevent attending to future tokens during training/generation |
| KV Cache | Store past K,V to avoid recomputation during autoregressive decoding |

Attention is the mechanism that makes transformers work. Every other
component — embeddings, RMSNorm, RoPE, feed-forward layers — exists to
prepare inputs for attention or to process its outputs. Understanding
attention is understanding the transformer.
