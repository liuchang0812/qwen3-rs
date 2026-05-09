# 05. RoPE: Rotary Position Embedding

Transformers are powerful, but they have a fundamental blind spot: they do not
natively understand the order of tokens. The self-attention mechanism computes
how much each token attends to every other token, but it does this by comparing
the *content* of tokens, not their *positions*. Without position information,
a transformer treats the input as a bag of words -- it cannot tell the
difference between "dog bites man" and "man bites dog."

RoPE (Rotary Position Embedding) is the elegant solution used by Qwen3 and
most modern language models. This document explains why position encoding is
needed, how RoPE works, the beautiful mathematics behind it, and how we
implement it in Rust.

---

## 1. The Position Problem

### Permutation Invariance

The core operation in a transformer is self-attention, which computes a
weighted sum of all token representations. The attention weight between token
A and token B depends on their *content* (their vector representations), not
on where they appear in the sequence. If you shuffle the input tokens, the
attention mechanism will compute the same set of pairwise scores -- just in a
different order. The model is **permutation-invariant**.

This is a problem. Consider:

```
Sentence 1: "The dog bites the man"
Sentence 2: "The man bites the dog"
```

Both sentences contain the exact same words. A permutation-invariant model
would produce the exact same internal representations for both, because it
has no way to know which word came first. But the meanings are opposites!
In sentence 1, the dog is the attacker; in sentence 2, the man is. The
difference is entirely about position.

Without position information, a transformer cannot distinguish between these
sentences. It would be like reading a sentence where all the words have been
written on separate cards and then shuffled -- you can see all the words, but
you have no idea what order they go in.

### Early Approaches: Absolute Position Embeddings

The original Transformer paper (Vaswani et al., 2017) solved this by adding a
position vector to each token's embedding:

```
final_input[t] = token_embedding[t] + position_embedding[t]
```

Each position (0, 1, 2, ...) gets its own learned vector, and this vector is
added to the token embedding before the token enters the transformer layers.
The model can then learn to use these position signals to understand order.

This works, but it has limitations:

**1. Fixed maximum length.** The model can only handle positions it has
learned embeddings for. If you trained with positions 0..2047, the model has
no position embedding for position 2048 or beyond. You cannot simply
extrapolate -- the learned vectors for unseen positions are undefined.

**2. No explicit relative position.** Absolute embeddings tell the model
"you are at position 5," but they do not directly tell it "you are 3 tokens
away from token 2." The model must *learn* to infer relative positions from
absolute ones, which is an indirect and potentially inefficient process.

**3. Added at the wrong place.** Position embeddings are typically added at
the input layer, before any transformer blocks. By the time the signal
reaches deep layers, the position information has been mixed with content
information through multiple linear projections and nonlinearities. It may
become diluted.

These limitations motivated researchers to look for better ways to encode
position. The breakthrough came from a surprisingly elegant mathematical idea:
rotation.

---

## 2. How RoPE Works

### The Key Idea: Encode Position by Rotating

Instead of *adding* position information to the representation, RoPE
*rotates* the representation. The angle of rotation depends on the token's
position. Two tokens at different positions will be rotated by different
amounts, so their representations will be different even if their content
is the same.

But the real magic is subtler: the way RoPE is designed, the *dot product*
between a rotated query and a rotated key depends only on their **relative**
position, not their absolute positions. This means attention naturally
captures relative distance, which is exactly what we want.

### Starting Simple: 2D Rotation

Consider a 2-dimensional vector `v = [x0, x1]`. We can rotate it by an angle
theta using a rotation matrix:

```
         | cos(theta)  -sin(theta) |   | x0 |   | x0*cos - x1*sin |
R(theta) |                       | * |    | = |                  |
         | sin(theta)   cos(theta) |   | x1 |   | x0*sin + x1*cos |
```

This is standard 2D rotation. If theta = 0, the matrix is the identity and
nothing changes. If theta = pi/2, the vector rotates 90 degrees
counterclockwise.

```
Original:     [1, 0]  ---->
After 90 deg: [0, 1]  ^
                       |
```

Now, here is the key insight: suppose we rotate the query vector by an angle
proportional to its position `m`, and rotate the key vector by an angle
proportional to its position `n`. Then their dot product is:

```
(R(m*theta) * q)^T * (R(n*theta) * k)
  = q^T * R(m*theta)^T * R(n*theta) * k
  = q^T * R((n - m) * theta) * k
```

The dot product depends only on `(n - m)`, the **relative position**! The
absolute positions m and n have cancelled out. This is the mathematical heart
of RoPE.

### Extending to Higher Dimensions

A single rotation angle can only distinguish a limited range of positions
before the angles "wrap around" (cos and sin are periodic with period 2*pi).
To handle long sequences, we need multiple rotation angles at different
frequencies -- like a clock with multiple hands.

For a vector of dimension `head_dim` (e.g., 128 for Qwen3), we split it
into `head_dim / 2` pairs, and each pair gets its own rotation angle:

```
Pair 0: dimensions (0, 1)     rotate by position * freq_0
Pair 1: dimensions (2, 3)     rotate by position * freq_1
Pair 2: dimensions (4, 5)     rotate by position * freq_2
...
Pair 63: dimensions (126, 127) rotate by position * freq_63
```

The frequencies decrease geometrically:

```
freq_i = 1 / (base ^ (2i / head_dim))
```

where `base = 1000000.0` for Qwen3. This gives:

```
freq_0  = 1 / 1000000^(0/128)   = 1.0
freq_1  = 1 / 1000000^(2/128)   = 0.1803
freq_2  = 1 / 1000000^(4/128)   = 0.0325
...
freq_63 = 1 / 1000000^(126/128) = 0.000001
```

Think of this as a clock with 64 hands. The first hand (freq_0 = 1.0) rotates
fast -- it makes a full revolution every 2*pi ≈ 6.28 positions. It can
precisely distinguish nearby positions but wraps around quickly. The last hand
(freq_63 = 0.00001) rotates extremely slowly -- it barely moves over hundreds
of positions. It provides coarse positional information that never wraps
within practical sequence lengths.

Together, the fast-rotating and slow-rotating pairs give the model both
fine-grained positional resolution for nearby tokens and unambiguous positional
information for distant tokens. This multi-scale structure is what makes RoPE
effective.

### The Rotation Formula

For each pair of dimensions `(x_i, x_{i+1})` at position `p`:

```
x_i'     = x_i * cos(p * freq_i) - x_{i+1} * sin(p * freq_i)
x_{i+1}' = x_i * sin(p * freq_i) + x_{i+1} * cos(p * freq_i)
```

This is applied independently to every attention head. The same rotation
angles are used for all heads (they share the same `head_dim`), so the
cos/sin tables can be computed once and reused.

---

## 3. Why RoPE Works -- The Beautiful Math

The most elegant property of RoPE is that the dot product of two RoPE-encoded
vectors depends only on their **relative position**. Let us prove this.

### Setup

Let `q` and `k` be two d-dimensional vectors (the query and key for a single
attention head). Let `R_p` denote the rotation operator at position `p`. In
RoPE, this is a block-diagonal matrix with 2x2 rotation blocks:

```
       | cos(p*f_0)  -sin(p*f_0)     0            0          ... |
       | sin(p*f_0)   cos(p*f_0)     0            0          ... |
R_p  = |     0            0       cos(p*f_1)  -sin(p*f_1)    ... |
       |     0            0       sin(p*f_1)   cos(p*f_1)    ... |
       |    ...          ...         ...          ...        ... |
```

Each 2x2 block rotates one pair of dimensions by `p * f_i`.

### The Proof

We want to compute the dot product of the rotated query and rotated key:

```
(R_m * q) . (R_n * k)
```

First, expand the dot product as a matrix product:

```
(R_m * q)^T * (R_n * k) = q^T * R_m^T * R_n * k
```

Now, a crucial property: the transpose of a rotation matrix is its inverse.
Rotating by angle theta and then by -theta gives the identity:

```
R(theta)^T = R(-theta)
```

So `R_m^T = R(-m)`, and:

```
q^T * R_m^T * R_n * k = q^T * R(-m) * R(n) * k
```

The product of two rotation matrices with the same block structure is the
rotation matrix with the sum of the angles. Since both `R(-m)` and `R(n)`
have the same block-diagonal structure, their product is:

```
R(-m) * R(n) = R(n - m)
```

Therefore:

```
q^T * R(n - m) * k
```

The dot product depends only on the **relative position** `n - m`, not on
the absolute positions `m` and `n`! This is the defining property of RoPE.

### Why This Matters for Attention

In the attention mechanism, the attention weight between a query at position
`m` and a key at position `n` is proportional to their dot product. With
RoPE, this dot product naturally encodes the relative distance `(n - m)`:

```
attention_weight(m, n) proportional to q^T * R(n - m) * k
```

- If the key is right next to the query (`n - m = 1`), a specific rotation
  is applied.
- If the key is far away (`n - m = 1000`), a different rotation is applied.
- The model learns which relative positions are important for each attention
  head, and the rotation structure ensures that tokens at the same relative
  distance always interact in the same way, regardless of their absolute
  positions.

This is a much stronger inductive bias than absolute position embeddings. The
model does not need to *learn* that relative position matters -- it is built
into the mathematical structure.

---

## 4. Concrete Example

Let us walk through applying RoPE to a simple vector step by step. This will
make the abstract formulas concrete.

### Setup

We will use:
- `head_dim = 4` (just two dimension pairs, for simplicity)
- `theta_base = 10000.0`
- Input vector at position 3: `x = [1.0, 0.0, 0.0, 1.0]`

### Step 1: Compute the Frequencies

```
freq_0 = 1 / (10000 ^ (0/4))  = 1 / 1.0    = 1.0
freq_1 = 1 / (10000 ^ (2/4))  = 1 / 100.0  = 0.01
```

### Step 2: Compute Rotation Angles at Position 3

```
angle_0 = 3 * freq_0 = 3 * 1.0   = 3.0      (radians)
angle_1 = 3 * freq_1 = 3 * 0.01  = 0.03     (radians)
```

### Step 3: Compute Cos and Sin

```
cos(3.0)   ≈ -0.9900        sin(3.0)   ≈ 0.1411
cos(0.03)  ≈  1.0000        sin(0.03)  ≈ 0.0300
```

### Step 4: Apply the Rotation

**Pair 0** (dimensions 0 and 1): rotate by angle 3.0

```
x_0' = x_0 * cos(3.0) - x_1 * sin(3.0)
     = 1.0 * (-0.9900) - 0.0 * 0.1411
     = -0.9900

x_1' = x_0 * sin(3.0) + x_1 * cos(3.0)
     = 1.0 * 0.1411 + 0.0 * (-0.9900)
     = 0.1411
```

**Pair 1** (dimensions 2 and 3): rotate by angle 0.03

```
x_2' = x_2 * cos(0.03) - x_3 * sin(0.03)
     = 0.0 * 1.0000 - 1.0 * 0.0300
     = -0.0300

x_3' = x_2 * sin(0.03) + x_3 * cos(0.03)
     = 0.0 * 0.0300 + 1.0 * 1.0000
     = 1.0000
```

### Result at Position 3

```
x_at_3 = [-0.9900, 0.1411, -0.0300, 1.0000]
```

### What About Position 4?

At position 4:

```
angle_0 = 4 * 1.0  = 4.0
angle_1 = 4 * 0.01 = 0.04

cos(4.0)  ≈ -0.6536        sin(4.0)  ≈ -0.7568
cos(0.04) ≈  0.9992        sin(0.04) ≈  0.0400
```

**Pair 0:**

```
x_0' = 1.0 * (-0.6536) - 0.0 * (-0.7568) = -0.6536
x_1' = 1.0 * (-0.7568) + 0.0 * (-0.6536) = -0.7568
```

**Pair 1:**

```
x_2' = 0.0 * 0.9992 - 1.0 * 0.0400 = -0.0400
x_3' = 0.0 * 0.0400 + 1.0 * 0.9992 =  0.9992
```

```
x_at_4 = [-0.6536, -0.7568, -0.0400, 0.9992]
```

### Observations

Notice two things:

1. **Pair 0 changed a lot** between position 3 and position 4 (from
   [-0.9900, 0.1411] to [-0.6536, -0.7568]). This is because freq_0 = 1.0
   rotates fast -- each position moves the angle by a full radian.

2. **Pair 1 barely changed** (from [-0.0300, 1.0000] to [-0.0400, 0.9992]).
   This is because freq_1 = 0.01 rotates slowly -- each position moves the
   angle by only 0.01 radians. This pair provides stable, long-range position
   information.

This multi-scale structure is like the hands of a clock: the second hand
moves fast (resolving fine-grained positions), while the hour hand moves
slowly (distinguishing far-apart positions without ambiguity).

---

## 5. Precomputation for Efficiency

### Why Precompute?

The cos and sin values depend only on the position and the dimension pair
index -- they do not depend on the input data. This means we can compute them
once at model initialization and reuse them for every token, every sequence,
and every forward pass.

Without precomputation, we would need to compute `2 * seq_len * head_dim/2`
trigonometric functions on every forward pass. With precomputation, we
compute them once and then just look up the values. Since trigonometric
functions are relatively expensive (much more so than a memory read), this
is a significant optimization.

### The Precomputation Formula

```python
# Pseudocode
for i in range(head_dim // 2):
    freq[i] = 1.0 / (theta_base ** (2 * i / head_dim))

for p in range(max_seq_len):
    for i in range(head_dim // 2):
        angle = p * freq[i]
        cos_table[p][i] = cos(angle)
        sin_table[p][i] = sin(angle)
```

### Table Shapes

For Qwen3-0.6B:
- `head_dim = 128`, so `head_dim / 2 = 64`
- `max_seq_len = 40960`

```
cos_table shape: [4096, 64]   = 262,144 floats = ~1 MB at f32
sin_table shape: [4096, 64]   = 262,144 floats = ~1 MB at f32
```

These tables are tiny compared to the model weights (~1.2 GB), so the memory
cost is negligible.

### Slicing for Inference

During autoregressive inference, we generate one token at a time. If we have
already generated tokens at positions 0..99 and are now generating position
100, we only need the cos/sin values for position 100. We do this by slicing
the precomputed tables:

```python
# For a new token at position start_pos:
cos_slice = cos_table[start_pos : start_pos + seq_len]  # shape [seq_len, 64]
sin_slice = sin_table[start_pos : start_pos + seq_len]  # shape [seq_len, 64]
```

For the initial prompt (multiple tokens at once), we slice a range of rows.
For subsequent generation steps (one token at a time), we slice a single row.

---

## 6. Where RoPE Is Applied in Qwen3

### Inside the Attention Block

RoPE is applied to the **Q** (query) and **K** (key) tensors **after** the
linear projections but **before** the attention score computation.

The flow within a single attention head looks like this:

```
Input hidden state x
        |
        v
   +-----------+
   |  W_q * x  |  --> Q tensor  ----->  Apply RoPE  -->  Q_rotated
   +-----------+                           |
                                           v
   +-----------+                    Dot product (attention scores)
   |  W_k * x  |  --> K tensor  ----->  Apply RoPE  -->  K_rotated
   +-----------+                           |
                                           v
   +-----------+                    Softmax (attention weights)
   |  W_v * x  |  --> V tensor  ---------+--------->  Weighted sum
   +-----------+                    (V is NOT rotated!)
                                           |
                                           v
                                    Attention output
```

### Three Key Points

**1. RoPE is applied to Q and K only, NOT to V.**

The value (V) tensor does not need position encoding because it represents
the *content* that gets aggregated, not the *address* used to look it up.
Think of it like a dictionary: the query and key need to know where things
are (position), but the value is the content being retrieved -- it does not
need positional modification.

**2. RoPE is applied within each attention head independently.**

Each head has its own Q, K, V vectors of dimension `head_dim`. RoPE is
applied to each head's Q and K separately. The rotation angles are the same
across all heads (they share the same `head_dim`), but the rotation operates
on each head's specific Q and K values.

**3. RoPE is applied after the linear projections.**

The Q and K tensors are first computed by multiplying the input by learned
weight matrices (`W_q` and `W_k`). RoPE is applied to the result of this
projection. This is important because it means the model can learn to
distribute positionally-relevant information across the head dimensions
before the rotation is applied.

### Where in the Transformer Stack

Qwen3 has 28 transformer blocks, and each block has its own attention
layer. RoPE is applied in every single block -- not just the first one. This
means that position information is refreshed at every layer, allowing the
model to maintain positional awareness throughout the deep network.

```
Input Embedding
      |
      v
  +---------+
  | Block 0 | --> Apply RoPE to Q, K in attention
  +---------+
      |
      v
  +---------+
  | Block 1 | --> Apply RoPE to Q, K in attention
  +---------+
      |
     ...
      |
      v
  +---------+
  | Block 27| --> Apply RoPE to Q, K in attention
  +---------+
      |
      v
  Output
```

---

## 7. Implementation Details

### Our Rust Implementation

Our implementation consists of two functions: `precompute_freqs` and
`apply_rope`.

### The `precompute_freqs` Function

```rust
pub fn precompute_freqs(
    head_dim: usize,
    max_seq_len: usize,
    theta_base: f32,
) -> (Tensor, Tensor)
```

This function:

1. Computes the frequency for each of the `head_dim/2` dimension pairs:
   `freq_i = 1.0 / (theta_base ^ (2i / head_dim))`.

2. For each position `p` from 0 to `max_seq_len - 1`, computes the angle
   `p * freq_i` for each pair `i`, then takes `cos` and `sin`.

3. Returns two tensors of shape `[max_seq_len, head_dim/2]`: the cosine
   table and the sine table.

These tables are computed once when the model is loaded and then reused for
every forward pass.

### The `apply_rope` Function

```rust
pub fn apply_rope(
    x: &Tensor,
    cos_table: &Tensor,
    sin_table: &Tensor,
) -> Tensor
```

This function takes a Q or K tensor of shape `[seq_len, num_heads, head_dim]`
and the cos/sin tables for the current position range, and returns the
RoPE-rotated tensor of the same shape.

### The Split-Into-Halves Approach

The original RoPE paper describes rotating *adjacent* pairs of dimensions:
(x_0, x_1), (x_2, x_3), (x_4, x_5), ... This would require gathering
elements at even and odd indices, which is inefficient for row-major storage.

Instead, we use an equivalent formulation where we split the head dimension
into two contiguous halves:

```
Interleaved pairs:  (x_0, x_1), (x_2, x_3), (x_4, x_5), ...
                    pair 0       pair 1       pair 2

Half-split:         x_0, x_2, x_4, ...  |  x_1, x_3, x_5, ...
                    --- first half ---    --- second half ---
                    (all "left" elements) (all "right" elements)
```

With this layout, the rotation becomes:

```
x1 = x[..., :half_dim]          (first half:  indices 0..half_dim-1)
x2 = x[..., half_dim:]          (second half: indices half_dim..head_dim-1)

out1 = x1 * cos - x2 * sin      (rotate)
out2 = x1 * sin + x2 * cos      (rotate)

out = concat(out1, out2)         (reassemble)
```

This is mathematically equivalent to the interleaved version because the
mapping between the two formulations is just a permutation of dimensions.
The cos/sin tables are computed to match the half-split layout, so the
rotation angles are the same -- they are just applied to a different
arrangement of the same elements.

The practical benefit is significant: we can operate on contiguous slices of
memory, which is much more cache-friendly than gathering scattered elements.

### Step-by-Step Inside `apply_rope`

For each position `p` and each head `h`, the function does:

```
1. Load x1 = x[p][h][0..half_dim]       (first half of this head)
2. Load x2 = x[p][h][half_dim..dim]     (second half of this head)
3. Load cos = cos_table[p]              (half_dim cosines for this position)
4. Load sin = sin_table[p]              (half_dim sines for this position)
5. Compute out1 = x1 * cos - x2 * sin   (rotated first half)
6. Compute out2 = x1 * sin + x2 * cos   (rotated second half)
7. Store out[p][h][0..half_dim] = out1
8. Store out[p][h][half_dim..dim] = out2
```

Since our tensor uses row-major storage, the elements of each head are
contiguous in memory, making steps 1-2 and 7-8 efficient sequential reads
and writes.

### Qwen3-Specific Parameters

For the Qwen3-0.6B model:

| Parameter      | Value       |
|----------------|-------------|
| `head_dim`     | 128         |
| `num_heads`    | 16          |
| `theta_base`   | 1000000.0   |
| `max_seq_len`  | 40960       |
| `half_dim`     | 64          |

The `theta_base` of 1000000.0 is larger than the original RoPE paper's
suggestion of 10000.0. A larger base makes the low-frequency dimension pairs
rotate even more slowly, which improves the model's ability to handle longer
sequences. This is sometimes called "extended RoPE" or "RoPE scaling."

### Putting It All Together

During a forward pass, the sequence of operations in the attention layer is:

```
1. Compute Q, K, V from input via linear projections
   Q = input * W_q    shape: [seq_len, num_heads, head_dim]
   K = input * W_k    shape: [seq_len, num_kv_heads, head_dim]
   V = input * W_v    shape: [seq_len, num_kv_heads, head_dim]

2. Apply RoPE to Q and K
   Q_rotated = apply_rope(Q, cos_slice, sin_slice)
   K_rotated = apply_rope(K, cos_slice, sin_slice)

3. Compute attention scores
   scores = Q_rotated * K_rotated^T / sqrt(head_dim)

4. Apply softmax to get attention weights
   weights = softmax(scores)

5. Compute weighted sum of V (no RoPE on V!)
   output = weights * V
```

Steps 2 and 3 are where RoPE makes its contribution. The rotation ensures
that the dot product between Q and K naturally encodes relative position,
so the attention weights are position-aware without any explicit position
term.

---

## Summary

| Concept | Details |
|---------|---------|
| **What** | RoPE encodes position by rotating pairs of dimensions in Q and K vectors |
| **Why** | Transformers are permutation-invariant and need position information |
| **How** | Each dimension pair is rotated by angle `position * frequency` |
| **Frequencies** | Decrease geometrically: `freq_i = 1 / (base ^ (2i / head_dim))` |
| **Key property** | Dot product of two RoPE-encoded vectors depends only on relative position |
| **Applied to** | Q and K only, not V |
| **When** | After linear projections, before attention score computation |
| **Precomputation** | Cos/sin tables computed once, shape `[max_seq_len, head_dim/2]` |
| **Qwen3 base** | 1000000.0 (larger than original 10000.0 for better length extrapolation) |
| **Implementation** | Half-split approach for cache-friendly memory access |

RoPE is one of the most elegant contributions to transformer architecture in
recent years. It solves the position encoding problem with a simple, clean
mathematical operation that naturally captures relative position. No learned
parameters, no additional matrices, no complicated interpolation schemes --
just rotation. And yet this simple operation is critical: without it, the
model would be unable to distinguish "dog bites man" from "man bites dog."
