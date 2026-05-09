# 07 — The SwiGLU Feed-Forward Network

The feed-forward network (FFN) is the "thinking" part of the transformer. If
attention is the mechanism that lets tokens *look* at each other, the FFN is
the mechanism that lets each token *process* what it has gathered. Every
transformer block applies an FFN after attention, and in modern LLMs the FFN
accounts for roughly half of all model parameters.

This document walks through the FFN from its simplest form to the SwiGLU
variant used in Qwen3, with concrete numerical examples and a look at our
Rust implementation.

---

## 1. What Is a Feed-Forward Network?

At its core, a feed-forward network is the simplest kind of neural network
layer: take an input vector, multiply it by a weight matrix, maybe add a bias,
apply an activation function, and produce an output vector. There are no
loops, no recurrence, no attention — just a linear transformation followed by
a nonlinearity.

In a transformer, the FFN is applied to **each token independently**. After
attention has gathered contextual information from other positions, the FFN
transforms each token's representation on its own. There is no cross-token
interaction inside the FFN. This is a crucial design choice: it keeps the FFN
simple and parallelizable, while the attention layer handles all the
cross-position communication.

You can think of the transformer block as having two stages:

1. **Attention** — "Look around." Each token queries all other tokens and
   aggregates relevant information.
2. **FFN** — "Think about what you saw." Each token processes its updated
   representation through a nonlinear transformation.

This division of labor — attention for communication, FFN for computation — is
the fundamental architecture of every transformer model.

---

## 2. The Vanilla FFN

The original transformer paper (Vaswani et al., 2017) used a simple two-layer
FFN with ReLU activation:

```text
FFN(x) = W2 · ReLU(W1 · x + b1) + b2
```

Where:
- `W1` is the "up" projection, expanding from `hidden_size` to
  `intermediate_size` (also called `d_ff`)
- `W2` is the "down" projection, contracting from `intermediate_size` back to
  `hidden_size`
- `ReLU(x) = max(0, x)` clips negative values to zero
- `b1`, `b2` are bias vectors (modern LLMs typically omit these)

The dimension expansion is the key design choice. In the original transformer,
`hidden_size = 512` and `intermediate_size = 2048` — a 4x expansion. The idea
is that the wider intermediate layer gives the network more capacity to learn
complex patterns. You can think of it as: the input is decomposed into a
higher-dimensional space where it is easier to separate and transform
different features, then projected back down to the original space.

Here is a concrete example with tiny dimensions. Say `hidden_size = 4` and
`intermediate_size = 8`:

```text
Input x: [1.0, -0.5, 0.3, 2.0]              shape [4]

W1 (up projection):                           shape [8, 4]
  (a random 8x4 matrix)

h = W1 · x                                    shape [8]
  (8 intermediate features, some negative)

h_activated = ReLU(h)                         shape [8]
  (negative values become 0)

W2 (down projection):                         shape [4, 8]
  (a random 4x8 matrix)

Output = W2 · h_activated                     shape [4]
  (back to the original hidden dimension)
```

The ReLU activation is what makes this nonlinear. Without it, the two linear
layers would collapse into a single linear transformation (since `W2 · W1` is
just another matrix), and the network would lose most of its expressive power.

---

## 3. The GLU Variant

In 2017, Dauphin et al. introduced the Gated Linear Unit (GLU), which adds a
*gating mechanism* to the feed-forward network. The idea is inspired by LSTMs
and GRUs, which use gates to control information flow.

The GLU FFN replaces the single up projection with two parallel projections,
and uses one to gate the other:

```text
GLU(x) = W_down · (gate ⊙ W_up · x)
```

Where:
- `W_up` projects the input to the intermediate dimension (the "value" path)
- `gate` is computed from the input using a separate projection, then passed
  through a sigmoid activation to produce values between 0 and 1
- `⊙` denotes element-wise multiplication (Hadamard product)
- `W_down` projects the gated result back to the hidden dimension

The gate acts like a **water valve**: for each intermediate feature, the gate
decides how much of that feature to let through. A gate value near 1 means
"pass this feature through fully," while a gate value near 0 means "block this
feature completely."

Why is this better than ReLU? ReLU applies the same hard cutoff (zero for
negative, identity for positive) to every element. The GLU gate, on the other
hand, is *input-dependent*: the gate values change based on what the input is,
allowing the network to learn adaptive, context-sensitive feature selection.

---

## 4. SwiGLU — What Qwen3 Uses

SwiGLU (Swish-Gated Linear Unit) is the particular GLU variant used in
Qwen3 and virtually every modern LLM. It was proposed by Shazeer (2020) in
the "GLU Variants Improve Transformer" paper and adopted by PaLM, LLaMA,
Mistral, Gemma, and Qwen.

The SwiGLU formula is:

```text
gate = SiLU(W_gate · x)         ← the "swish" gate
up   = W_up · x                 ← the value path
output = W_down · (gate ⊙ up)   ← gate modulates value, then project down
```

This requires **three** weight matrices instead of two: `W_gate`, `W_up`, and
`W_down`. The extra matrix is the price of the gating mechanism, but the
performance gain is worth it.

### The SiLU Activation

SiLU (Sigmoid Linear Unit), also called the "Swish" function, is defined as:

```text
SiLU(x) = x · sigmoid(x) = x / (1 + e^(-x))
```

Let us compute SiLU for several input values to understand its shape:

| x    | sigmoid(x) | SiLU(x)    |
|------|------------|------------|
| -3.0 | 0.0474     | -0.1422    |
| -2.0 | 0.1192     | -0.2384    |
| -1.0 | 0.2689     | -0.2689    |
| -0.5 | 0.3775     | -0.1888    |
|  0.0 | 0.5000     |  0.0000    |
|  0.5 | 0.6225     |  0.3112    |
|  1.0 | 0.7311     |  0.7311    |
|  2.0 | 0.8808     |  1.7616    |
|  3.0 | 0.9526     |  2.8578    |
|  5.0 | 0.9933     |  4.9665    |
| 10.0 | 0.9999     |  9.9995    |

Key observations about SiLU:
- **For large positive x**: SiLU(x) approaches x (since sigmoid(x) approaches
  1). The function becomes nearly linear, just like ReLU.
- **At x = 0**: SiLU(0) = 0, same as ReLU.
- **For negative x**: SiLU(x) dips slightly negative (reaching about -0.28
  near x = -1.77) then approaches 0 from below. This is *non-monotonic* — the
  function goes negative and then comes back toward zero. ReLU simply clips
  to zero.
- **Smooth everywhere**: Unlike ReLU which has a sharp corner at x = 0, SiLU
  is a smooth (infinitely differentiable) curve. This means smooth gradient
  flow during backpropagation — no discontinuity in the derivative.

### Comparing SiLU and ReLU

```
ReLU:                   SiLU:
    |                       |
  3 |       /               |      .--
  2 |      /                |    .-
  1 |     /                 |  .-
  0 |____/_____             |._/ ._____
    |                        |  /
 -1 |                        | /
    |                        |.     (dips slightly below 0)
```

ReLU is a hard ramp: zero below, identity above. SiLU is a soft ramp that
smoothly transitions, with a slight negative dip. The smoothness and the
self-gating property (the gate value depends on the input magnitude) make
SiLU empirically superior for deep networks.

---

## 5. Why SwiGLU Over ReLU?

The shift from ReLU FFNs to SwiGLU FFNs is one of the clearest empirical
improvements in modern LLM architecture. Here is why:

### Smooth Gradient Flow

ReLU has a zero gradient for all negative inputs. This means that any neuron
whose pre-activation is negative receives no gradient and cannot update — the
"dead neuron" problem. In a deep network with millions of neurons, a
significant fraction can become permanently stuck at zero output.

SiLU has a non-zero gradient everywhere (except at negative infinity). Even
for moderately negative inputs, the gradient is small but non-zero, so every
neuron can potentially recover. This leads to better optimization dynamics
throughout training.

### Self-Gating

In the SwiGLU formula, the gate is `SiLU(W_gate · x)`. Because SiLU(x) = x *
sigmoid(x), the gate value naturally scales with the input magnitude. For
large positive inputs, the gate is wide open (SiLU(x) ~ x). For near-zero
inputs, the gate is nearly closed. This input-dependent gating is more
flexible than ReLU's fixed threshold at zero.

### Empirical Evidence

The original GLU paper (Shazeer 2020) tested many GLU variants (with ReLU,
GELU, Swish/SiLU as the gate activation) on language modeling benchmarks.
SwiGLU consistently outperformed the vanilla ReLU FFN. This result was
confirmed at scale by PaLM (540B parameters), which adopted SwiGLU, and then
by LLaMA, which made SwiGLU the standard for open-weight models.

Today, every major open LLM family uses SwiGLU or a close variant:
- **LLaMA** (Meta) — SwiGLU
- **Qwen** (Alibaba) — SwiGLU
- **Mistral** (Mistral AI) — SwiGLU
- **Gemma** (Google) — GeGLU (GELU-gated, very similar)
- **Phi** (Microsoft) — SwiGLU

The SwiGLU FFN has become as standard as the multi-head attention mechanism
itself.

---

## 6. Concrete Computation Example

Let us walk through the full SwiGLU computation with small, hand-computable
numbers. We will use `hidden_size = 2` and `intermediate_size = 3`.

### Setup

```text
Input x = [1.0, 2.0]                       shape [2] (1 token, 2 features)

gate_proj = [[1, 0],                        shape [3, 2]
             [0, 1],
             [1, 1]]

up_proj   = [[2, 0],                        shape [3, 2]
             [0, 2],
             [1, -1]]

down_proj = [[1, 0, 0],                     shape [2, 3]
             [0, 1, 0]]
```

### Step 1: Gate path — project and apply SiLU

First, compute the pre-activation values by projecting x through gate_proj:

```text
gate_pre = x · gate_proj^T

gate_proj^T = [[1, 0, 1],
               [0, 1, 1]]

gate_pre = [1.0, 2.0] · [[1, 0, 1],
                          [0, 1, 1]]

gate_pre[0] = 1*1 + 2*0 = 1.0
gate_pre[1] = 1*0 + 2*1 = 2.0
gate_pre[2] = 1*1 + 2*1 = 3.0

gate_pre = [1.0, 2.0, 3.0]
```

Now apply SiLU element-wise:

```text
gate[0] = SiLU(1.0) = 1.0 * sigmoid(1.0) = 1.0 / (1 + e^(-1)) = 1.0 * 0.7311 = 0.7311
gate[1] = SiLU(2.0) = 2.0 * sigmoid(2.0) = 2.0 / (1 + e^(-2)) = 2.0 * 0.8808 = 1.7616
gate[2] = SiLU(3.0) = 3.0 * sigmoid(3.0) = 3.0 / (1 + e^(-3)) = 3.0 * 0.9526 = 2.8578

gate = [0.7311, 1.7616, 2.8578]
```

### Step 2: Up path — project (no activation)

```text
up = x · up_proj^T

up_proj^T = [[2, 0, 1],
             [0, 2, -1]]

up[0] = 1*2 + 2*0 = 2.0
up[1] = 1*0 + 2*2 = 4.0
up[2] = 1*1 + 2*(-1) = -1.0

up = [2.0, 4.0, -1.0]
```

### Step 3: Element-wise multiply (gate modulates up)

```text
gated = gate ⊙ up

gated[0] = 0.7311 * 2.0  = 1.4622
gated[1] = 1.7616 * 4.0  = 7.0464
gated[2] = 2.8578 * (-1.0) = -2.8578

gated = [1.4622, 7.0464, -2.8578]
```

Notice something important here: even though the up path has a negative value
(-1.0 at position 2), the gate value at position 2 is large and positive
(2.8578), so the gated result at position 2 is -2.8578. The gate does not
simply suppress negative values — it modulates whatever the up path produces.
The sign of the output depends on the up path; the gate controls the
*magnitude*.

### Step 4: Down projection — project back to hidden_size

```text
output = gated · down_proj^T

down_proj^T = [[1, 0],
               [0, 1],
               [0, 0]]

output[0] = 1.4622 * 1 + 7.0464 * 0 + (-2.8578) * 0 = 1.4622
output[1] = 1.4622 * 0 + 7.0464 * 1 + (-2.8578) * 0 = 7.0464

output = [1.4622, 7.0464]
```

The final output has the same shape as the input: `[2]` (one token with two
features). The down_proj we chose here only uses the first two intermediate
features (the third column is all zeros), so the third intermediate dimension
is effectively discarded.

### Summary of the data flow

```text
x          [1.0, 2.0]                    hidden_size=2
              |
         +----+----+
         |         |
     gate_proj   up_proj
         |         |
    gate_pre     up
    [1,2,3]     [2,4,-1]                intermediate_size=3
         |         |
       SiLU        |
         |         |
      gate         |
   [0.73,1.76,2.86]|
         |         |
         +----+----+
              |
          gate ⊙ up
       [1.46, 7.05, -2.86]
              |
          down_proj
              |
        output [1.46, 7.05]              hidden_size=2
```

---

## 7. Parameter Count in Qwen3-0.6B

Now let us look at the real numbers. Qwen3-0.6B has:
- `hidden_size = 1024`
- `intermediate_size = 3072` (3x expansion)
- `num_hidden_layers = 28`
- No bias terms in the FFN (standard for modern LLMs)

### Per FFN

Each FFN layer has three weight matrices:

| Matrix     | Shape           | Parameters |
|------------|-----------------|------------|
| gate_proj  | [3072, 1024]    | 3,145,728  |
| up_proj    | [3072, 1024]    | 3,145,728  |
| down_proj  | [1024, 3072]    | 3,145,728  |
| **Total**  |                 | **9,437,184** |

Each matrix contributes exactly 3,145,728 parameters (3072 * 1024), for a
total of about 9.44 million parameters per FFN.

### All FFNs combined

```text
28 layers * 9,437,184 params/layer = 264,241,152 total FFN parameters
```

That is approximately **264.2 million parameters**, or about 46% of the
model's total ~596 million parameters. The FFN is the single largest
component of the model — bigger than the attention layers, bigger than the
embeddings.

### Comparison with other components

| Component                | Parameters  | Percentage |
|--------------------------|-------------|------------|
| Token embedding          | ~156M       | ~27%       |
| 28x Attention layers     | ~176M       | ~30%       |
| 28x FFN layers           | ~264M       | ~46%       |
| Output norm + lm_head    | ~1K        | ~0%        |

The FFN's dominance in parameter count is why optimizing the FFN — through
techniques like mixture-of-experts (MoE), where only a subset of the FFN is
activated per token — can dramatically reduce inference cost. Models like
Mixtral and Qwen3-MoE use this approach.

### Why 3x expansion?

The 3x expansion ratio (intermediate_size = 3 * hidden_size) is specific to
the SwiGLU architecture. The original transformer used a 4x expansion with a
2-matrix FFN. SwiGLU adds a third matrix (the gate), so to keep the total
parameter count roughly the same, the expansion ratio is reduced from 4x to
approximately 4x * 2/3 = 8/3 ~ 2.67x. In practice, Qwen3 uses a clean 3x,
which gives slightly more parameters than the equivalent 2-matrix 4x FFN:

```text
2-matrix FFN (4x):   2 * hidden * (4 * hidden) = 8 * hidden^2
3-matrix FFN (3x):   3 * hidden * (3 * hidden) = 9 * hidden^2
```

So the SwiGLU FFN with 3x expansion uses about 12.5% more parameters than a
vanilla 4x FFN, but the performance improvement more than justifies the cost.

---

## 8. Implementation Details

Here is our Rust implementation of the SwiGLU FFN:

```rust
pub struct FeedForward {
    gate_proj: Tensor,  // [intermediate_size, hidden_size] = [3072, 1024]
    up_proj: Tensor,    // [intermediate_size, hidden_size] = [3072, 1024]
    down_proj: Tensor,  // [hidden_size, intermediate_size] = [1024, 3072]
}

impl FeedForward {
    pub fn new(gate_proj: Tensor, up_proj: Tensor, down_proj: Tensor) -> Self {
        // Validate dimensions...
        Self { gate_proj, up_proj, down_proj }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Step 1: Gate path — project then activate with SiLU.
        let gate = x.matmul(&self.gate_proj.transpose_2d()).silu();

        // Step 2: Up path — project (no activation).
        let up = x.matmul(&self.up_proj.transpose_2d());

        // Step 3: Element-wise multiply — gate modulates up path.
        let gated = gate.mul_elementwise(&up);

        // Step 4: Down projection — project back to hidden_size.
        gated.matmul(&self.down_proj.transpose_2d())
    }
}
```

### Understanding the transpose

The weight matrices from safetensors are stored as `[out_features, in_features]`,
which is the PyTorch convention. For a linear layer computing `y = x · W^T`,
the weight `W` has shape `[out_dim, in_dim]`. Since our `matmul` expects
`[M, K] × [K, N]`, we need to transpose `W` first:

```text
x:     [seq_len, hidden_size]       = [seq_len, 1024]
W^T:   [hidden_size, intermediate]  = [1024, 3072]
result: [seq_len, intermediate]     = [seq_len, 3072]
```

Without the transpose, the dimensions would not line up for matmul. The
`transpose_2d()` method swaps rows and columns: `[M, N]` becomes `[N, M]`.

### The four-step computation

The `forward` method implements exactly the SwiGLU formula:

1. **Gate path**: `x.matmul(&gate_proj.transpose_2d())` computes `W_gate · x^T`,
   then `.silu()` applies the SiLU activation. This produces the gate signal
   that controls information flow.

2. **Up path**: `x.matmul(&up_proj.transpose_2d())` computes `W_up · x^T`.
   No activation is applied — this is the raw value signal.

3. **Gate modulation**: `gate.mul_elementwise(&up)` performs element-wise
   multiplication. Where the gate is large, the up-path value passes through;
   where the gate is near zero, the value is suppressed.

4. **Down projection**: `gated.matmul(&down_proj.transpose_2d())` projects
   the intermediate representation back to the hidden dimension, producing
   the final output.

### Why no biases?

Modern LLMs (LLaMA, Qwen, Mistral, Gemma) omit bias terms from the FFN
linear layers. The reasoning is:
- Biases add relatively few parameters compared to the weight matrices
- With RMSNorm applied before the FFN (pre-norm architecture), the
  normalization already centers the activations, reducing the need for biases
- Empirically, removing biases does not hurt performance and simplifies the
  implementation

### Constructor validation

The `new` constructor validates that the three weight matrices have compatible
dimensions:
- `gate_proj` and `up_proj` must have the same shape `[intermediate_size,
  hidden_size]`
- `down_proj` must have shape `[hidden_size, intermediate_size]` — the reverse
  of `gate_proj`

These checks catch configuration errors early, before they produce confusing
dimension mismatches during `forward()`.

### Key properties verified by tests

Our test suite verifies several important properties:

1. **Output shape matches input shape**: After all the expansion and
   contraction, the FFN outputs the same shape it received. This is essential
   for the residual connection in the transformer block: `output = x + FFN(x)`.

2. **SiLU gate behavior**: For positive inputs with positive weights, the
   pre-activation values are positive, and SiLU of positive values is always
   positive. The gate is thus a soft, non-negative modulator.

3. **No cross-token interaction**: Feeding two identical tokens produces two
   identical output rows. The FFN processes each token independently — there
   is no cross-position communication. That is attention's job.

4. **Different inputs produce different outputs**: A basic sanity check that
   the computation is not collapsing to a constant function.

5. **Hand-computed values**: A small-dimension test with manually computed
   expected outputs ensures the arithmetic is correct end-to-end.

---

## Summary

The SwiGLU FFN is a deceptively simple component: three matrix multiplications,
one activation function, and one element-wise multiply. But this simple recipe
accounts for nearly half the parameters in Qwen3 and provides the model's
primary nonlinear processing capacity.

The key innovations over the vanilla ReLU FFN are:
1. **Gating** — the gate_proj path learns input-dependent feature selection,
   rather than applying a fixed threshold
2. **SiLU activation** — smooth gradients, no dead neurons, self-gating
   behavior
3. **Three projections** — the extra matrix gives the gating mechanism its
   own learnable parameters

In the full transformer block, the FFN sits after attention and is wrapped by
a residual connection:

```text
output = x + FFN(RMSNorm(x + Attention(RMSNorm(x))))
```

The residual connection allows the FFN to learn incremental modifications to
the token representations, while the pre-norm (RMSNorm before each sub-layer)
stabilizes the magnitude of the inputs. Together, these design choices make
the modern transformer both powerful and trainable at scale.
