# 10 — Token Sampling: From Logits to Text

At each step of autoregressive generation, the model outputs a vector of raw
scores called **logits** — one score for every token in the vocabulary. The
question is: how do we turn these scores into a single token? This is the
problem of **sampling**, and the strategy we choose has an enormous impact on
the quality, diversity, and character of the generated text.

This chapter covers every sampling strategy implemented in our
`src/sampling.rs` module, from the simplest (greedy) to the most
sophisticated (temperature + top-k + top-p), and explains the
implementation details of our Rust code.

---

## 1. From Logits to Tokens

### What Are Logits?

The final layer of the transformer is a linear projection (`lm_head`) that
maps the hidden state from `hidden_size` (1024) to `vocab_size` (151936).
The output is a vector of raw, unnormalized scores — one per vocabulary
token. These scores are called **logits**.

Logits can be any real number: positive, negative, large, or small. A logit
of 5.3 does not mean "5.3% probability." Logits must be converted to
probabilities before they can be used for sampling.

### Logits to Probabilities via Softmax

The standard conversion is the **softmax** function:

```
softmax(z_i) = exp(z_i) / sum_j(exp(z_j))
```

Softmax converts a vector of arbitrary real numbers into a probability
distribution: all values are in [0, 1] and they sum to 1. The token with the
highest logit gets the highest probability, but every token gets *some*
nonzero probability.

For numerical stability, we use the max-subtraction trick:

```
softmax(z_i) = exp(z_i - max(z)) / sum_j(exp(z_j - max(z)))
```

This prevents overflow in `exp()` when logits are large.

### The Choice Problem

After softmax, we have a probability distribution over the entire vocabulary.
Now we must choose a single token. The simplest approach is to always pick
the most probable token (greedy decoding). But there are many situations
where we want some randomness — to generate creative text, to avoid
repetitive outputs, or to explore multiple possible continuations.

This is where sampling strategies come in. They control **how** we select a
token from the probability distribution, trading off determinism and quality
against diversity and creativity.

---

## 2. Greedy Decoding

### How It Works

Greedy decoding always picks the token with the highest probability:

```
token = argmax(logits)
```

No randomness is involved. Given the same input, greedy decoding always
produces the same output. It is the simplest possible sampling strategy.

### Properties

- **Deterministic**: The same input always produces the same output. This
  makes results reproducible, which is useful for debugging and testing.
- **Repeatable**: Running the same prompt twice yields identical text.
- **Fast**: argmax is O(vocab_size) and requires no random number generation.

### Problems

Greedy decoding has two well-known problems:

1. **Boring output**: By always picking the most likely token, the model
   converges on the most "average" or "safe" continuation. Creative,
   surprising, or interesting word choices are never selected because they
   have slightly lower probability.

2. **Repetition loops**: Once the model enters a pattern (e.g., "The cat
   sat on the mat. The cat sat on the mat. The cat sat on the mat."), greedy
   decoding has no mechanism to break out. The most likely continuation of
   the repeated pattern is more repetition, creating an infinite loop.

### Example

Given the prompt "The cat sat on the", the model might produce these logits
(abbreviated to the top candidates):

```
"mat"   → logit 8.2   → probability 0.65
"couch" → logit 6.1   → probability 0.08
"floor" → logit 5.8   → probability 0.06
"table" → logit 5.5   → probability 0.04
"roof"  → logit 4.2   → probability 0.01
```

Greedy decoding always picks "mat" (the highest probability). Every single
time you run this prompt, you get "The cat sat on the mat." There is no
variation.

### Our Implementation

```rust
pub fn sample_greedy(logits: &[f32]) -> usize {
    argmax(logits)
}
```

The `argmax` helper function finds the index of the maximum value:

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

In case of ties, the first index wins (leftmost).

---

## 3. Temperature

### How It Works

Temperature scales the logits before softmax. Given a temperature T:

```
scaled_logits[i] = logits[i] / T
probabilities = softmax(scaled_logits)
```

Temperature does not change the *order* of tokens by probability — the
highest-logit token remains the highest-probability token. Instead,
temperature changes the **shape** of the distribution: how peaked or flat it
is.

### Effect of Different Temperatures

**Temperature = 0** (greedy): Division by zero is undefined, so we treat
temperature 0 as a special case that directly returns `argmax(logits)`. This
is equivalent to making the distribution infinitely peaked — all probability
mass concentrates on the single highest-logit token.

**Temperature < 1** (sharper): Dividing logits by a small number makes them
larger in magnitude. After softmax, the distribution becomes more peaked —
the highest-probability tokens get even more probability, and low-probability
tokens are further suppressed. The model becomes more "confident" and
conservative.

**Temperature = 1** (default): No scaling is applied. The logits are used as-
is. This is the model's "natural" distribution.

**Temperature > 1** (flatter): Dividing logits by a large number makes them
smaller in magnitude. After softmax, the distribution becomes flatter —
probability mass spreads more evenly across tokens. The model becomes less
confident and more random.

**Temperature approaching infinity**: As T grows, all logits approach zero.
Softmax of near-zero values produces a near-uniform distribution where every
token has approximately equal probability (1/vocab_size).

### Concrete Example

Suppose the model produces these logits for three candidate tokens:

```
"mat"   → logit 6.0
"couch" → logit 3.0
"roof"  → logit 0.0
```

**Temperature = 0.5** (sharper):

```
scaled logits: [6.0/0.5, 3.0/0.5, 0.0/0.5] = [12.0, 6.0, 0.0]
softmax:       [0.9975,  0.0025, 0.0000]
                 ↑ "mat" dominates almost completely
```

**Temperature = 1.0** (default):

```
scaled logits: [6.0, 3.0, 0.0]
softmax:       [0.9500, 0.0474, 0.0024]
                 ↑ "mat" is still very likely, but others have some chance
```

**Temperature = 2.0** (flatter):

```
scaled logits: [6.0/2, 3.0/2, 0.0/2] = [3.0, 1.5, 0.0]
softmax:       [0.7054, 0.1419, 0.0317]
                 ↑ "mat" is still most likely, but the distribution is much flatter
```

**Temperature = 10.0** (very flat):

```
scaled logits: [0.6, 0.3, 0.0]
softmax:       [0.3943, 0.2912, 0.2153]
                 ↑ almost uniform — any token could be chosen
```

Notice how the probability of "mat" drops from 0.9975 (temperature 0.5) to
0.3943 (temperature 10.0). Temperature gives us fine-grained control over
how "random" the model's output appears.

### Why Temperature Works

The softmax function is an exponential. When logits are far apart (large
magnitude), the exponential amplifies the differences, making the
distribution very peaked. When logits are close together (small magnitude),
the exponential has less effect, making the distribution more uniform.
Temperature controls the magnitude of the logits, which in turn controls the
peakedness of the distribution.

### Our Implementation

In our `sample` function, temperature is the first transformation applied:

```rust
if config.temperature == 0.0 {
    return sample_greedy(logits);
}

let mut scaled: Vec<f32> = logits.iter().map(|&l| l / config.temperature).collect();
```

If temperature is zero, we short-circuit to greedy decoding. Otherwise, we
divide every logit by the temperature before proceeding to the next sampling
stage.

---

## 4. Top-k Sampling

### How It Works

Top-k sampling restricts the candidate set to the `k` tokens with the highest
logits (or probabilities). All other tokens are set to probability zero, and
the remaining `k` tokens are renormalized so they sum to 1.

The algorithm:

1. Sort tokens by logit (or probability) in descending order.
2. Keep only the top `k` tokens.
3. Set all other tokens' logits to negative infinity (which becomes
   probability 0 after softmax).
4. Apply softmax to the filtered logits. The result is a distribution over
   only the top `k` tokens.

### Effect of Different k Values

**k = 1**: Only the single highest-probability token is kept. This is
equivalent to greedy decoding.

**k = small (e.g., 10)**: Only the 10 most likely tokens are candidates.
This produces focused, coherent text with limited diversity.

**k = 50 (typical)**: The 50 most likely tokens are candidates. This is a
good balance between coherence and diversity for most tasks.

**k = vocab_size**: No filtering is applied. Every token is a candidate.
This is equivalent to not using top-k at all.

### Why Top-k Helps

Without filtering, the model occasionally samples extremely unlikely tokens
from the long tail of the distribution. These tokens can break the coherence
of the text — imagine a well-formed sentence suddenly containing a random
Chinese character or a rare punctuation mark. Top-k eliminates these
degenerate samples by preventing the model from considering tokens it is very
unsure about.

### Concrete Example

Suppose after temperature scaling, the top 6 tokens have these probabilities
(out of a vocabulary of 151,936):

```
"mat"    → 0.50
"couch"  → 0.20
"floor"  → 0.10
"table"  → 0.05
"roof"   → 0.03
"bed"    → 0.02
... (151,930 other tokens share the remaining 0.10 probability)
```

With **k = 3**:

```
Keep:    "mat", "couch", "floor"
Discard: everything else

Renormalized:
"mat"    → 0.50 / 0.80 = 0.625
"couch"  → 0.20 / 0.80 = 0.250
"floor"  → 0.10 / 0.80 = 0.125
```

Now the model can only generate "mat", "couch", or "floor". The very unlikely
tokens (which together held 0.10 probability) are eliminated entirely. The
remaining tokens' probabilities are scaled up proportionally.

### Limitations of Top-k

Top-k uses a **fixed** number of candidates regardless of the distribution
shape. This creates two problems:

1. **When the model is very confident** (one token at 0.95 probability),
   keeping k=50 tokens forces the model to consider 49 extremely unlikely
   tokens. These tokens should not be candidates, but top-k includes them
   anyway.

2. **When the model is uncertain** (probability spread evenly across many
   tokens), keeping k=50 might exclude tokens that are almost as likely as
   the 50th token. The cutoff is arbitrary and may discard good candidates.

Top-p (nucleus) sampling, described next, addresses both of these problems.

### Our Implementation

```rust
if config.top_k > 0 && config.top_k < scaled.len() {
    let k = config.top_k;
    // Find the k-th largest value.
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

We find the k-th largest logit value and set everything below it to negative
infinity. After the subsequent softmax call, these positions become
probability zero.

Note that `top_k = 0` disables top-k filtering (no cutoff is applied). This
is the default behavior when the user does not want top-k.

---

## 5. Top-p (Nucleus) Sampling

### How It Works

Top-p sampling (also called **nucleus sampling**) keeps the smallest set of
tokens whose cumulative probability is at least `p`. Unlike top-k, which uses
a fixed count, top-p adapts to the shape of the distribution.

The algorithm:

1. Sort tokens by probability in descending order.
2. Accumulate probabilities from the top, stopping as soon as the cumulative
   sum reaches or exceeds `p`.
3. Zero out all tokens beyond the cutoff.
4. Renormalize the remaining tokens so they sum to 1.

### Effect of Different p Values

**p close to 0**: Very few tokens are kept. At p = 0, only the single
highest-probability token survives (equivalent to greedy).

**p = 0.9 (typical)**: The smallest set of tokens whose cumulative
probability reaches 0.9 is kept. When the model is confident, this might be
just 2-3 tokens; when uncertain, it could be 20-30 tokens.

**p = 1.0**: No filtering is applied. All tokens are candidates.

### Why Top-p Is Better Than Top-k

Top-p adapts dynamically to the model's confidence:

- **When the model is very confident** (one token at 0.96 probability), a
  top-p of 0.9 keeps just 1-2 tokens. The model's strong preference is
  respected, and no unlikely tokens dilute the distribution.

- **When the model is uncertain** (probability spread across many tokens), a
  top-p of 0.9 might keep 30+ tokens. More candidates are included because
  the model is less sure about the correct continuation.

This adaptivity produces more natural text than the fixed-cutoff approach of
top-k.

### Concrete Example

Suppose the sorted probability distribution is:

```
"mat"    → 0.60
"couch"  → 0.20
"floor"  → 0.08
"table"  → 0.04
"roof"   → 0.03
"bed"    → 0.02
... (other tokens: 0.03)
```

With **p = 0.9**:

```
Step 1: "mat"    → cumulative = 0.60  (< 0.9, keep)
Step 2: "couch"  → cumulative = 0.80  (< 0.9, keep)
Step 3: "floor"  → cumulative = 0.88  (< 0.9, keep)
Step 4: "table"  → cumulative = 0.92  (>= 0.9, keep this one too, then stop)

Kept tokens: "mat", "couch", "floor", "table"
Discarded:   "roof", "bed", and all others

Renormalized (sum of kept = 0.92):
"mat"    → 0.60 / 0.92 = 0.652
"couch"  → 0.20 / 0.92 = 0.217
"floor"  → 0.08 / 0.92 = 0.087
"table"  → 0.04 / 0.92 = 0.043
```

Now compare this with a different distribution where the model is very
confident:

```
"mat"    → 0.95
"couch"  → 0.02
"floor"  → 0.01
"table"  → 0.01
...
```

With **p = 0.9**:

```
Step 1: "mat"    → cumulative = 0.95  (>= 0.9, keep this one, then stop)

Kept tokens: "mat" only
Discarded:   everything else

Renormalized: "mat" → 1.0
```

When the model is confident, top-p keeps fewer tokens. When the model is
uncertain, top-p keeps more. This is exactly the behavior we want.

### Our Implementation

Top-p filtering is applied after softmax (which converts logits to
probabilities):

```rust
if config.top_p < 1.0 {
    let n = scaled.len();
    // Sort indices by probability descending.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        scaled[b].partial_cmp(&scaled[a]).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Find cutoff: keep tokens until cumulative probability >= top_p.
    let mut cumsum = 0.0f32;
    let mut cutoff = 0;
    for &idx in &indices {
        cumsum += scaled[idx];
        cutoff += 1;
        if cumsum >= config.top_p {
            break;
        }
    }

    // Zero out tokens beyond the cutoff.
    for &idx in &indices[cutoff..] {
        scaled[idx] = 0.0;
    }

    // Renormalize.
    let sum: f32 = scaled.iter().sum();
    if sum > 0.0 {
        for v in scaled.iter_mut() {
            *v /= sum;
        }
    }
}
```

The key steps are: sort by probability descending, accumulate until the sum
reaches `top_p`, zero out everything beyond the cutoff, and renormalize.

---

## 6. Combining Strategies

### The Full Pipeline

Our `sample` function applies strategies in a specific order:

```
Raw logits
    │
    ▼
1. Temperature scaling:  logits / temperature
    │
    ▼
2. Top-k filtering:  set logits below the k-th largest to -inf
    │
    ▼
3. Softmax:  convert filtered logits to probabilities
    │
    ▼
4. Top-p filtering:  zero out tokens beyond the cumulative probability threshold
    │
    ▼
5. Renormalize:  ensure probabilities sum to 1
    │
    ▼
6. CDF sampling:  draw a random number, walk the CDF to pick a token
```

### Why Apply in This Order

The order matters. Here is why each step comes where it does:

1. **Temperature first**: Temperature changes the *shape* of the logit
   distribution. It must be applied before any filtering, because the
   filtering decisions depend on the relative magnitudes of the logits.
   Applying temperature after top-k would change the probabilities of the
   already-filtered tokens in an unintended way.

2. **Top-k before softmax**: Top-k filters based on *logit magnitude*, not
   probability. Filtering at the logit level is more natural because logits
   have a linear scale. After softmax, probabilities are on an exponential
   scale, and the distinction between "high probability" and "low
   probability" is compressed. Setting filtered logits to `-inf` before
   softmax ensures they become exactly 0 probability.

3. **Top-p after softmax**: Top-p operates on *probabilities*, not logits.
   It needs the cumulative sum of probabilities, which requires a valid
   probability distribution (hence softmax first). Applying top-p at the
   logit level would be incorrect because logit values do not have a
   probabilistic interpretation.

4. **CDF sampling last**: After all filtering and renormalization, we have a
   clean probability distribution. CDF sampling draws a single token from
   this distribution.

### Common Parameter Combinations

Different tasks benefit from different sampling configurations:

**Creative writing** (stories, poems, brainstorming):

```
temperature = 0.9
top_k = 50
top_p = 0.95
```

A relatively high temperature encourages diverse word choices. Top-k and top-p
prevent the model from going off the rails by sampling extremely unlikely
tokens.

**Code generation** (programming, structured output):

```
temperature = 0.2
top_k = 10
top_p = 0.9
```

Code requires precise syntax and logic. Low temperature makes the model
stick close to the most likely continuation. Small top-k restricts candidates
to the most plausible tokens, avoiding syntactic errors from unlikely
samples.

**Factual Q&A** (knowledge retrieval, summarization):

```
temperature = 0.0  (greedy)
```

For factual questions, there is typically one correct answer. Greedy decoding
ensures the model always picks the most probable (and hopefully most
accurate) token. No randomness means the same question always gets the same
answer.

**Conversational chat** (general-purpose assistants):

```
temperature = 0.7
top_k = 50
top_p = 0.9
```

A moderate temperature produces varied but coherent responses. These are the
default values in our `SamplingConfig`.

### Why Not Just Use Temperature?

Temperature alone cannot prevent the model from occasionally sampling very
unlikely tokens. Even at temperature 0.5, the long tail of the distribution
contains thousands of tokens with tiny but nonzero probability. Occasionally,
one of these tokens will be sampled, potentially breaking the coherence of
the text.

Top-k and top-p act as safety nets: they eliminate the long tail entirely,
ensuring that only reasonable candidates are ever considered. Combined with
temperature for shape control, they give the user precise control over the
trade-off between quality and diversity.

---

## 7. Randomness and Reproducibility

### Pseudo-Random Number Generators (PRNG)

The final step of sampling — CDF sampling — requires a random number. But
"random" in a computer program means "pseudo-random": generated by a
deterministic algorithm that produces a sequence of numbers that *appear*
random but are entirely determined by an initial **seed** value.

This is a feature, not a bug. Deterministic randomness enables
**reproducibility**: given the same seed, the PRNG produces the same
sequence of random numbers, which means the same logits and sampling
configuration produce the same output token. This is invaluable for
debugging, testing, and creating reproducible demonstrations.

### Our XorShift64 Implementation

We use a xorshift64 PRNG — one of the simplest and fastest non-cryptographic
PRNGs. It is implemented in `src/sampling.rs` with no external dependencies:

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

The algorithm is a sequence of three XOR-and-shift operations with the
classic shift triple (13, 7, 17). Starting from any nonzero 64-bit seed,
it produces a sequence of 64-bit numbers with period 2^64 - 1 before
repeating.

The `next_f32` method converts a u64 to a float in [0, 1) by taking the top
24 bits and dividing by 2^24. This gives approximately 7 decimal digits of
precision, which is more than sufficient for sampling decisions.

### Seeds for Reproducibility

Our `SamplingConfig` has an optional `seed` field:

```rust
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: Option<u64>,
}
```

When `seed` is `Some(value)`, sampling is fully deterministic: the same
logits and config always produce the same token. This is useful for:

- **Testing**: Unit tests can assert exact token IDs without worrying about
  randomness.
- **Reproducibility**: Users can share seeds to reproduce specific outputs.
- **Debugging**: A failing generation can be reproduced exactly by using the
  same seed.

When `seed` is `None`, we use a default seed of 12345. This means sampling
is deterministic within a single run but not across different invocations
(unless the user sets an explicit seed).

### Why Not Use a Cryptographic RNG?

Cryptographic RNGs (like `/dev/urandom` or ChaCha20) produce high-quality
randomness that is unpredictable even to an attacker. But for token sampling,
cryptographic quality is unnecessary — we just need numbers that are
uniformly distributed and reproducible. A cryptographic RNG would be slower
and would make reproducibility harder (since their output depends on system
entropy).

XorShift64 is:

- **Fast**: Three XOR-shift operations per random number, no memory access
  beyond the 8-byte state.
- **Small**: Only 8 bytes of state.
- **Deterministic**: Given the same seed, it always produces the same
  sequence.
- **Statistically adequate**: It passes standard statistical tests for
  uniformity and independence, which is all we need for sampling.

The only caveat is that xorshift64 is **not** cryptographically secure.
Given a few outputs, an attacker could predict future outputs. But since we
are generating text, not encryption keys, this is irrelevant.

---

## 8. Implementation Details

### SamplingConfig

The `SamplingConfig` struct bundles all sampling parameters:

```rust
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub temperature: f32,     // 0.0 = greedy, 0.7 = default
    pub top_k: usize,         // 0 = disabled, 50 = default
    pub top_p: f32,           // 1.0 = disabled, 0.9 = default
    pub seed: Option<u64>,    // None = default seed
}
```

The default configuration:

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

These defaults are suitable for general-purpose text generation: moderate
temperature for diverse but coherent output, top-k of 50 to eliminate the
long tail, and top-p of 0.9 to adaptively filter based on confidence.

### The sample() Function Step by Step

The `sample` function is the main entry point. It takes a slice of logits
and a `SamplingConfig`, and returns a single token ID. Here is the complete
pipeline with annotations:

```rust
pub fn sample(logits: &[f32], config: &SamplingConfig) -> usize {
    // Step 1: Temperature == 0 means greedy.
    // Short-circuit immediately to avoid unnecessary computation.
    if config.temperature == 0.0 {
        return sample_greedy(logits);
    }

    // Step 2: Apply temperature scaling.
    // Divide every logit by the temperature.
    // This changes the shape of the distribution without changing the order.
    let mut scaled: Vec<f32> = logits.iter().map(|&l| l / config.temperature).collect();

    // Step 3: Apply top-k filtering.
    // Find the k-th largest logit value and set everything below it to -inf.
    // After softmax, -inf becomes probability 0, effectively removing those
    // tokens from the candidate set.
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

    // Step 4: Softmax to convert filtered logits to probabilities.
    // Uses the numerically stable max-subtraction trick.
    // -inf logits correctly become 0 probability (exp(-inf) = 0).
    softmax_in_place(&mut scaled);

    // Step 5: Apply top-p (nucleus) filtering.
    // Sort by probability descending, accumulate until sum >= top_p,
    // zero out the rest, and renormalize.
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

    // Step 6: CDF sampling with our XorShift64 PRNG.
    // Draw a random number r in [0, 1), walk the cumulative distribution,
    // and return the first index where cumsum > r.
    let mut rng = XorShift64::new(config.seed.unwrap_or(12345));
    let r = rng.next_f32();
    let mut cumsum = 0.0f32;
    for (i, &p) in scaled.iter().enumerate() {
        cumsum += p;
        if cumsum > r {
            return i;
        }
    }

    // Fallback: return the last index (handles floating-point rounding).
    scaled.len() - 1
}
```

### softmax_in_place

The numerically stable softmax implementation:

```rust
fn softmax_in_place(logits: &mut [f32]) {
    // Step 1: find max for numerical stability.
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Step 2: subtract max and exponentiate.
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }

    // Step 3: normalize.
    if sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    }
}
```

The max-subtraction trick prevents overflow: if the largest logit is 1000,
computing `exp(1000)` would overflow to infinity. But `exp(1000 - 1000) =
exp(0) = 1`, which is perfectly fine. The relative probabilities are
preserved because softmax is shift-invariant: `softmax(z + c) = softmax(z)`
for any constant c.

This implementation also handles `-inf` correctly: `exp(-inf - max) =
exp(-inf) = 0`, so filtered-out tokens from top-k become probability zero.

### CDF Sampling

The final step draws a random number `r` uniformly from [0, 1) and walks the
cumulative distribution:

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
    probs.len() - 1  // Fallback for floating-point rounding
}
```

This is the standard method for sampling from a categorical distribution. The
probability of selecting token i is exactly `probs[i]`, because the interval
of `r` values that leads to selecting token i has width `probs[i]`.

For example, if the distribution is [0.5, 0.3, 0.2]:

- r in [0, 0.5) selects token 0 (probability 0.5)
- r in [0.5, 0.8) selects token 1 (probability 0.3)
- r in [0.8, 1.0) selects token 2 (probability 0.2)

The fallback case (`probs.len() - 1`) handles the rare situation where
floating-point rounding causes `cumsum` to be slightly less than 1.0, and
`r` falls in the gap between `cumsum` and 1.0. In this case, we return the
last token, which is the most conservative choice.

### Edge Cases

The implementation handles several edge cases:

- **Temperature 0**: Short-circuits to greedy decoding, skipping all other
  steps. This avoids division by zero and unnecessary computation.

- **top_k = 0**: Disables top-k filtering. No logits are set to `-inf`.

- **top_p = 1.0**: Disables top-p filtering. No tokens are zeroed out.

- **top_k >= vocab_size**: No filtering needed (all tokens are candidates).

- **All logits equal**: After softmax, all tokens have equal probability
  (1/vocab_size). Sampling is truly uniform.

- **Single nonzero logit**: After softmax, one token has probability 1.0 and
  all others have 0. CDF sampling always returns this token regardless of
  the random number.

### Standalone Functions

In addition to the main `sample` function, we provide standalone versions of
each sampling strategy that can be used independently:

- `sample_greedy(logits)`: Greedy decoding (argmax).
- `sample_top_k(logits, k, rng)`: Top-k sampling with explicit RNG.
- `sample_top_p(logits, p, rng)`: Top-p sampling with explicit RNG.

These are useful for experimentation and for building custom sampling
pipelines that differ from the default ordering.

### Testing

The sampling module has comprehensive tests that verify:

- **Greedy returns argmax**: The highest logit always wins.
- **Temperature 0 equals greedy**: The pipeline produces the same result as
  `sample_greedy`.
- **High temperature flattens**: Very high temperature produces a near-uniform
  distribution.
- **Low temperature sharpens**: Very low temperature concentrates probability
  on the top token.
- **Top-k limits candidates**: Only the top k tokens have nonzero
  probability after filtering.
- **Top-p maintains cumulative probability**: The kept tokens' cumulative
  probability is at least `p`, and removing the last kept token would drop
  below `p`.
- **XorShift64 is deterministic**: Same seed produces the same sequence.
- **Sample is deterministic with seed**: Same logits, config, and seed produce
  the same token.
- **Softmax handles -inf**: Filtered-out tokens correctly get probability 0.
- **Softmax handles large values**: Numerically stable even with logits of
  1000+.
- **Softmax sums to 1**: The output is a valid probability distribution.

---

## Summary

| Strategy | What It Does | When to Use |
|----------|-------------|-------------|
| Greedy | Always pick the highest-probability token | Factual Q&A, debugging |
| Temperature | Scale logits to control distribution shape | Trade quality vs. creativity |
| Top-k | Keep only the k most likely tokens | Prevent sampling very unlikely tokens |
| Top-p | Keep the smallest set covering cumulative probability p | Adaptive filtering based on confidence |
| Combined | Temperature, top-k, top-p, then CDF sample | General-purpose text generation |

The sampling pipeline gives the user fine-grained control over the character
of generated text. By adjusting temperature, top-k, and top-p, you can move
along the spectrum from deterministic, focused output to creative, diverse
output. The XorShift64 PRNG provides fast, reproducible randomness. Together,
these components turn the model's raw logits into coherent, controllable text.
