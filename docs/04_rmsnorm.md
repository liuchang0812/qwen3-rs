# 04. RMSNorm: Why and How We Normalize Activations

Neural networks are powerful, but they are also fragile. Left unchecked, the
numbers flowing through a deep network can explode to infinity or collapse to
zero, making training impossible. Normalization layers are the guardrails that
keep those numbers in a safe range. In Qwen3, the specific guardrail we use
is called **RMSNorm** (Root Mean Square Normalization).

This document explains why normalization is needed, how RMSNorm works, how it
compares to its predecessor LayerNorm, and how we implement it in Rust.

---

## 1. Why Do We Need Normalization?

### The Problem: Internal Covariate Shift

Imagine you are training a 28-layer neural network. During training, each layer
receives inputs from the layer before it. As the weights in earlier layers are
updated by gradient descent, the distribution of their outputs changes. This
means that a middle layer (say, layer 14) constantly sees a shifting input
distribution -- it never gets a stable "view" of what the data looks like.

This phenomenon is called **internal covariate shift**. The practical
consequence is devastating:

- **Exploding activations**: If the weights in early layers grow slightly
  larger, the outputs grow slightly larger, which causes the next layer's
  outputs to grow even more, and so on exponentially. By layer 28, the
  numbers are NaN (not a number) -- the network has blown up.

- **Vanishing activations**: If the weights in early layers shrink slightly,
  the outputs shrink, and the next layer's outputs shrink further. By layer
  28, the numbers are so close to zero that the gradients are also zero,
  and learning stops entirely.

- **Slow convergence**: Even when the network does not explode or vanish, the
  shifting input distributions force each layer to constantly adapt, making
  training slow and requiring careful tuning of learning rates.

### An Analogy

Think of a 28-person assembly line. Each person receives work from the person
before them, does their part, and passes it on. If person 3 starts sending
work that is too large (because they changed their technique), person 4 gets
overwhelmed, produces sloppy work, and person 5 gets even more confused. The
problem cascades down the line. Normalization is like a quality control
inspector between each pair of workers who rescales the work to a standard
size before passing it along.

### The Fix: Normalize

Normalization layers solve this by **re-scaling the activations at each layer**
so that they always fall within a consistent range. Regardless of what the
previous layer did -- whether it produced tiny numbers or huge numbers -- the
normalization layer transforms them back to a standard scale before passing
them to the next sub-layer.

This dramatically stabilizes training. With normalization:
- Activations stay in a reasonable range throughout the network.
- Gradients flow more smoothly during backpropagation.
- The network converges faster and is less sensitive to learning rate choice.

---

## 2. LayerNorm (The Predecessor)

Before RMSNorm, the dominant normalization in transformers was **LayerNorm**
(Layer Normalization), introduced by Ba et al. in 2016. The original
Transformer paper (Vaswani et al., 2017) used LayerNorm, and so did BERT,
GPT-1, GPT-2, and many other early models.

### The Formula

For an input vector **x** of length `n`, LayerNorm computes:

```
LayerNorm(x) = weight * (x - mean(x)) / sqrt(var(x) + eps) + bias

where:
  mean(x) = (1/n) * sum(x_i)
  var(x)  = (1/n) * sum((x_i - mean(x))^2)
  weight  = learnable scale parameter of shape [n]  (called gamma)
  bias    = learnable shift parameter of shape [n]  (called beta)
  eps     = small constant for numerical stability (e.g., 1e-5)
```

### Step-by-Step Example

Let us walk through LayerNorm with a concrete example. Suppose `x = [3.0, 4.0]`,
`weight = [1.0, 1.0]`, `bias = [0.0, 0.0]`, and `eps = 1e-6`.

**Step 1: Compute the mean.**
```
mean = (3.0 + 4.0) / 2 = 3.5
```

**Step 2: Subtract the mean (center the data).**
```
x_centered = [3.0 - 3.5, 4.0 - 3.5] = [-0.5, 0.5]
```

**Step 3: Compute the variance.**
```
var = ((-0.5)^2 + 0.5^2) / 2 = (0.25 + 0.25) / 2 = 0.25
```

**Step 4: Divide by the standard deviation (scale the data).**
```
std = sqrt(0.25 + 1e-6) = sqrt(0.250001) ≈ 0.5000
x_normalized = [-0.5 / 0.5000, 0.5 / 0.5000] = [-1.0, 1.0]
```

**Step 5: Multiply by weight and add bias.**
```
output = [1.0 * (-1.0) + 0.0, 1.0 * 1.0 + 0.0] = [-1.0, 1.0]
```

The result is a centered, scaled version of the input. The mean is zero and
the standard deviation is one. The weight and bias allow the network to learn
the optimal scale and offset.

### Why LayerNorm Works

LayerNorm works because it guarantees that the activations at each layer have
a predictable distribution: zero mean and unit variance (before the learned
weight and bias are applied). This prevents the exploding/vanishing problems
and makes optimization much easier.

The learned `weight` and `bias` parameters are important: they let the network
"undo" the normalization if that is beneficial. If the network determines that
the best representation for a particular layer has a mean of 5 and a standard
deviation of 3, it can learn `bias = 5` and `weight = 3` to achieve that.
The normalization provides a stable "default" that the network can build on.

---

## 3. RMSNorm (What We Use)

RMSNorm was proposed by Zhang and Sennrich in 2019. The key insight is simple:
**the mean subtraction in LayerNorm does not actually help much**, and removing
it makes the computation cheaper without hurting performance.

### The Formula

For an input vector **x** of length `n`, RMSNorm computes:

```
RMSNorm(x) = weight * x / sqrt(mean(x^2) + eps)

where:
  mean(x^2) = (1/n) * sum(x_i^2)
  weight    = learnable scale parameter of shape [n]  (called gamma)
  eps       = small constant for numerical stability (1e-6 in Qwen3)
```

Notice what is missing compared to LayerNorm:
- **No mean subtraction**: We do not center the data around zero.
- **No bias**: There is no learnable shift parameter.
- **No variance computation**: We use `mean(x^2)` instead of `var(x)`.

### Step-by-Step Example

Let us walk through RMSNorm with the same input: `x = [3.0, 4.0]`,
`weight = [1.0, 1.0]`, `eps = 1e-6`.

**Step 1: Square each element.**
```
x_squared = [3.0^2, 4.0^2] = [9.0, 16.0]
```

**Step 2: Compute the mean of squares.**
```
mean_sq = (9.0 + 16.0) / 2 = 25.0 / 2 = 12.5
```

**Step 3: Add epsilon for numerical stability.**
```
mean_sq + eps = 12.5 + 0.000001 ≈ 12.5
```

The epsilon makes essentially no difference here because `mean_sq` is already
well above zero. It only matters when all input elements are zero (or very
close to zero), in which case `mean_sq` would be zero and we would be dividing
by `sqrt(eps)` instead of zero.

**Step 4: Take the square root.**
```
rms = sqrt(12.5) = 3.5355339...
```

This is the **root mean square** of the input -- the square root of the mean
of the squared elements. It is a measure of the "magnitude" of the vector,
similar to the L2 norm but scaled by `1/sqrt(n)`.

**Step 5: Divide each element by the RMS.**
```
x_normalized = [3.0 / 3.5355, 4.0 / 3.5355]
             = [0.84853..., 1.13137...]
```

After this step, the root mean square of the output vector is exactly 1.0
(ignoring the tiny epsilon). The vector has been "re-scaled" so that its
magnitude is standardized, regardless of the original magnitude.

**Step 6: Multiply by the learned weight.**
```
output = [1.0 * 0.8485, 1.0 * 1.1314]
       = [0.8485, 1.1314]
```

Since the weight is all ones in this example, the output is the same as the
normalized vector. In practice, the weight is learned during training and
allows the model to re-scale individual dimensions.

### Comparing LayerNorm and RMSNorm on the Same Input

Let us put the two side by side for `x = [3.0, 4.0]`:

| Step | LayerNorm | RMSNorm |
|------|-----------|---------|
| Mean | 3.5 | (not computed) |
| Centering | [-0.5, 0.5] | (not done) |
| Variance/Mean-of-squares | 0.25 | 12.5 |
| Std/RMS | 0.5 | 3.5355 |
| Normalized | [-1.0, 1.0] | [0.8485, 1.1314] |
| After weight+bias | [-1.0, 1.0] | [0.8485, 1.1314] |

The key difference is visible in the normalized output. LayerNorm centers the
data (negative and positive values), while RMSNorm only scales it (all values
keep their original sign). LayerNorm produces a zero-mean output; RMSNorm does
not.

### A Subtlety: What Does "mean(x^2)" Have to Do With Variance?

If you look closely, you might notice that `mean(x^2)` is related to variance:

```
var(x) = mean(x^2) - mean(x)^2
```

So:

```
mean(x^2) = var(x) + mean(x)^2
```

RMSNorm uses `mean(x^2)` where LayerNorm uses `var(x)`. The difference is that
RMSNorm includes `mean(x)^2` in its calculation. When the mean is large
relative to the variance, RMSNorm and LayerNorm produce quite different
results. When the mean is near zero, they are similar.

In practice, the mean of activations in a well-trained transformer tends to
be small relative to the variance, which is part of why removing the mean
subtraction does not hurt performance.

---

## 4. Why RMSNorm Over LayerNorm?

You might wonder: if LayerNorm has been working well for years, why switch to
RMSNorm? There are three reasons.

### 4.1 Simpler (Fewer Operations)

LayerNorm requires these operations:
1. Compute the mean of the input.
2. Subtract the mean from each element.
3. Compute the variance of the centered input.
4. Divide by the standard deviation.
5. Multiply by weight.
6. Add bias.

RMSNorm requires:
1. Compute the mean of squares (one multiply-accumulate pass).
2. Divide by the root mean square.
3. Multiply by weight.

That is roughly half the operations. In a model like Qwen3 with 57 RMSNorm
layers (more on that below), the savings add up.

### 4.2 Just As Effective

The original RMSNorm paper (Zhang & Sennrich, 2019) showed that RMSNorm
matches or slightly exceeds LayerNorm in quality across a range of language
modeling benchmarks. The mean subtraction, it turns out, is not doing much
useful work. The most important part of normalization is the **re-scaling**
(dividing by a measure of magnitude), not the **centering** (subtracting the
mean).

This makes intuitive sense: what matters for stability is that the activations
do not explode or vanish. Scaling them to have a consistent magnitude achieves
this. Whether they are centered around zero or not is a secondary concern that
the rest of the network can easily adapt to.

### 4.3 Slightly Faster

Removing the mean subtraction and bias not only simplifies the code but also
reduces memory traffic and computation:

- **One fewer learnable parameter per dimension**: LayerNorm has both `weight`
  and `bias` (2 parameters per dimension), while RMSNorm has only `weight`
  (1 parameter per dimension). For hidden_size = 1024, that saves 1024
  parameters per normalization layer, or 1024 * 57 = 58,368 parameters
  total. This is negligible in terms of memory but slightly reduces the
  amount of data to load during the forward pass.

- **One fewer pass over the data**: LayerNorm needs to compute the mean first,
  then use it to center the data before computing the variance. RMSNorm
  computes `mean(x^2)` in a single pass.

In practice, the speed difference is small (normalization layers are a tiny
fraction of total compute in a transformer -- attention and FFN dominate). But
the simplicity advantage is real: simpler code is easier to implement
correctly, easier to optimize, and easier to reason about.

### 4.4 The Modern Standard

RMSNorm has become the default choice for modern LLMs:

| Model | Year | Normalization |
|-------|------|---------------|
| BERT | 2018 | LayerNorm |
| GPT-2 | 2019 | LayerNorm |
| GPT-3 | 2020 | LayerNorm |
| LLaMA | 2023 | RMSNorm |
| LLaMA 2 | 2023 | RMSNorm |
| Mistral | 2023 | RMSNorm |
| Qwen | 2023 | RMSNorm |
| Qwen2 | 2024 | RMSNorm |
| Qwen3 | 2025 | RMSNorm |

The transition happened around 2023 with the LLaMA family. Once it was
demonstrated that RMSNorm works just as well with fewer operations, the
community adopted it rapidly. Today, nearly all new LLM architectures use
RMSNorm.

---

## 5. Where Is RMSNorm Used in Qwen3?

RMSNorm appears in three places in the Qwen3 model architecture:

### 5.1 Before Self-Attention (input_layernorm)

In each transformer block, the input is first normalized by RMSNorm before
being passed to the self-attention mechanism. This is called the
**input layer norm** or **pre-attention norm**.

```
x_norm = RMSNorm(x)                    # normalize
attn_out = self_attention(x_norm)       # attend
output = x + attn_out                   # residual connection
```

The weight for this layer is stored under the key:
`model.layers.{i}.input_layernorm.weight`

where `{i}` ranges from 0 to 27 (for the 28 transformer blocks).

### 5.2 Before FFN (post_attention_layernorm)

After the self-attention sub-layer (and its residual connection), the result
is normalized again before being passed to the FFN. This is called the
**post-attention layer norm** or **pre-FFN norm**.

```
x_norm = RMSNorm(x')                   # normalize (x' is the attention residual output)
ffn_out = ffn(x_norm)                   # feed-forward
output = x' + ffn_out                   # residual connection
```

The weight for this layer is stored under the key:
`model.layers.{i}.post_attention_layernorm.weight`

### 5.3 Final Normalization (model.norm)

After all 28 transformer blocks, there is one final RMSNorm layer that
normalizes the output before it is projected to vocabulary logits by the
`lm_head`. This ensures that the logits are computed from a well-scaled
representation.

```
hidden = block_27_output
hidden_norm = RMSNorm(hidden)           # final normalization
logits = lm_head(hidden_norm)           # project to vocab
```

The weight for this layer is stored under the key:
`model.norm.weight`

### 5.4 Counting the RMSNorm Layers

Let us count them all:

- Per transformer block: 2 (input_layernorm + post_attention_layernorm)
- Number of transformer blocks: 28
- Blocks subtotal: 28 * 2 = 56
- Final norm: 1
- **Total: 56 + 1 = 57 RMSNorm layers**

Each of these 57 layers has its own independent `weight` parameter of shape
`[1024]`. The total number of RMSNorm parameters is:

```
57 * 1024 = 58,368 parameters
```

At f32 (4 bytes each), that is only 233,472 bytes -- about 228 KB. This is
utterly negligible compared to the ~580 million total parameters. The
normalization layers contribute almost nothing to the model's memory
footprint, but they are essential for training stability.

### 5.5 Pre-Norm vs. Post-Norm

An important architectural detail: Qwen3 uses the **pre-norm** design,
where normalization is applied *before* the sub-layer (attention or FFN),
not after. The original Transformer paper used **post-norm**, where
normalization was applied *after* the sub-layer.

Pre-norm:
```
output = x + sublayer(RMSNorm(x))
```

Post-norm:
```
output = RMSNorm(x + sublayer(x))
```

Pre-norm has become standard because it is more stable for training deep
networks. The reason: in pre-norm, the residual connection carries the
unmodified input `x` directly to the output, so gradients can always flow
through the identity path. In post-norm, the normalization is applied *after*
the addition, which can still cause gradient issues because the normalization
itself can distort the signal.

In our Qwen3 implementation, both the attention and FFN sub-layers follow
the pre-norm pattern.

---

## 6. Implementation Details

Now let us look at how we implement RMSNorm in Rust.

### 6.1 The RMSNorm Struct

Our implementation wraps the low-level `Tensor::rms_norm` operation in a
reusable struct that stores the weight and epsilon:

```rust
pub struct RMSNorm {
    weight: Tensor,  // shape [hidden_size], the learned scaling parameter
    eps: f32,        // small constant for numerical stability (1e-6)
}
```

The struct is intentionally simple. It owns a weight tensor and an epsilon
value. When we create an `RMSNorm` instance for a specific layer, we load the
weight from the safetensors file and store it in the struct. The epsilon is
always `1e-6` for Qwen3.

### 6.2 Construction

```rust
impl RMSNorm {
    pub fn new(weight: Tensor, eps: f32) -> Self {
        assert_eq!(weight.ndim(), 1, "weight must be 1-D");
        Self { weight, eps }
    }
}
```

The constructor validates that the weight is a 1-D tensor (a vector) and
stores it along with the epsilon. In practice, this is called once during
model initialization:

```rust
// When loading the model:
let input_ln_weight = weights.load("model.layers.0.input_layernorm.weight");
let input_layernorm = RMSNorm::new(input_ln_weight, 1e-6);
```

### 6.3 The Forward Pass

```rust
impl RMSNorm {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.rms_norm(&self.weight, self.eps)
    }
}
```

The forward method delegates entirely to `Tensor::rms_norm`. This is a
deliberate design choice: the tensor module owns the low-level math (loops
over rows, accumulation, square root), and the `RMSNorm` struct provides a
clean, reusable interface that stores the layer's parameters.

### 6.4 The Tensor::rms_norm Backend

The actual computation happens in `Tensor::rms_norm` (in `tensor.rs`). Here
is what it does for each row of the input:

```rust
pub fn rms_norm(&self, weight: &Tensor, eps: f32) -> Tensor {
    // For each row of the input tensor:
    for r in 0..num_rows {
        // Step 1: Compute mean of squares for this row.
        let mut sum_sq = 0.0f32;
        for j in 0..last_dim {
            let v = self.data[row_start + j];
            sum_sq += v * v;                     // accumulate x_i^2
        }
        let mean_sq = sum_sq / (last_dim as f32); // (1/n) * sum(x_i^2)

        // Step 2: Compute the reciprocal of the RMS.
        //   rms = sqrt(mean_sq + eps)
        //   1/rms = 1 / sqrt(mean_sq + eps)
        let rms_inv = 1.0 / (mean_sq + eps).sqrt();

        // Step 3: Normalize and scale by weight.
        for j in 0..last_dim {
            result[row_start + j] = self.data[row_start + j] * rms_inv * weight.data[j];
        }
    }
}
```

A few implementation notes:

**Why compute `rms_inv` instead of `rms`?** We compute the reciprocal of the
RMS (`1/rms`) rather than the RMS itself, and then multiply by `rms_inv`
instead of dividing by `rms`. This is a standard optimization: multiplication
is faster than division, and we only need one division (to compute `1/rms`)
instead of `n` divisions (one per element).

**Why iterate over rows?** The input to RMSNorm is a 2-D tensor of shape
`[seq_len, hidden_size]`. Each row represents one token's hidden state, and
normalization is applied independently to each row. Two different tokens in
the same sequence should not influence each other's normalization.

**Why multiply weight inside the loop?** We could normalize first and then
multiply by weight in a separate step. But combining them in one loop avoids
an extra pass over the data and an extra temporary tensor allocation. For a
hidden_size of 1024 and a typical sequence length, this saves a non-trivial
number of memory operations.

### 6.5 Numerical Stability

The `eps` parameter is crucial for numerical stability. Consider what happens
when the input is all zeros:

```
x = [0.0, 0.0, ..., 0.0]
mean_sq = 0.0
rms = sqrt(0.0 + eps) = sqrt(1e-6) ≈ 0.001
output = [0.0 / 0.001, 0.0 / 0.001, ...] * weight = [0.0, 0.0, ...]
```

Without `eps`, we would have `sqrt(0.0) = 0.0` and division by zero. With
`eps`, we get a small but non-zero denominator, and the output is correctly
zero (because the numerator is also zero).

In practice, activations are rarely exactly zero. But they can be very small,
and without `eps`, floating-point rounding could cause the denominator to be
zero or extremely small, leading to numerical instability (NaN or Inf values
propagating through the network). The epsilon ensures this never happens.

The value `1e-6` is the standard choice for Qwen3 (specified in
`config.json` under the key `rms_norm_eps`). It is small enough that it does
not affect the computation when activations are normal-sized (e.g., when
`mean_sq` is around 1.0, adding `1e-6` makes no difference), but large
enough to prevent underflow when activations are tiny.

### 6.6 Memory Layout

The weight tensor has shape `[hidden_size]` = `[1024]`, stored as a contiguous
array of 1024 f32 values. During the forward pass, this array is accessed
sequentially for each row of the input. Since the access pattern is sequential
and the array is small (only 4 KB), it fits entirely in L1 cache on modern
CPUs. This means the weight access is essentially free from a memory
perspective.

The input and output tensors have shape `[seq_len, hidden_size]`, stored in
row-major order. This means each row (each token's hidden state) is contiguous
in memory, which is optimal for the row-by-row normalization loop.

### 6.7 Usage in the Transformer Block

Here is how RMSNorm fits into the transformer block's forward pass:

```rust
fn forward(&self, x: &Tensor) -> Tensor {
    // Pre-attention norm
    let x_norm = self.input_layernorm.forward(x);

    // Self-attention
    let attn_out = self.attention.forward(&x_norm);

    // Residual connection
    let x = x.add(&attn_out);

    // Pre-FFN norm
    let x_norm = self.post_attention_layernorm.forward(&x);

    // Feed-forward network
    let ffn_out = self.ffn.forward(&x_norm);

    // Residual connection
    x.add(&ffn_out)
}
```

Notice the pattern: normalize, then process, then add the residual. This
happens twice per block (once for attention, once for FFN), and the
normalization is always applied to the *input* of the sub-layer, never to
the output.

---

## Summary

Let us recap the key points about RMSNorm:

| Concept | Summary |
|---------|---------|
| **What it does** | Normalizes a vector by its root mean square, then scales by a learned weight |
| **Formula** | `RMSNorm(x) = weight * x / sqrt(mean(x^2) + eps)` |
| **vs. LayerNorm** | Removes mean subtraction and bias; simpler but equally effective |
| **Epsilon** | 1e-6 in Qwen3; prevents division by zero |
| **Weight** | Learnable parameter of shape `[hidden_size]`; one per normalization layer |
| **Where used** | 57 times total: 2 per transformer block (56) + 1 final norm |
| **When applied** | Before each sub-layer (pre-norm architecture) |
| **Parameters** | 57 * 1024 = 58,368 total; negligible compared to the full model |

RMSNorm is one of the simplest components in the transformer, but it is also
one of the most important. Without normalization, training a 28-layer
network would be extremely difficult. RMSNorm provides the necessary stability
with minimal computational overhead, which is why it has become the standard
choice in modern LLMs.

In the next document, we will look at how position information is encoded
using Rotary Position Embedding (RoPE): [05_rope.md](05_rope.md).
