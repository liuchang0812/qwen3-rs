//! Token sampling strategies for autoregressive language model generation.
//!
//! During autoregressive generation, the model produces a logits vector of
//! shape `[vocab_size]` at each step. This module converts those logits into
//! a single token ID using configurable sampling strategies.
//!
//! # Sampling pipeline
//!
//! The main entry point is [`sample`], which applies strategies in order:
//!
//! 1. **Temperature scaling** — controls the "sharpness" of the distribution.
//!    Low temperature (< 1.0) makes the distribution more peaked, favouring
//!    high-probability tokens. High temperature (> 1.0) flattens the
//!    distribution, producing more diverse output. Temperature 0.0 is
//!    equivalent to greedy decoding (always pick the most likely token).
//!
//! 2. **Top-k filtering** — restricts sampling to the `k` tokens with the
//!    highest logits. This prevents the model from sampling extremely unlikely
//!    tokens while still allowing some randomness. It is a simple, effective
//!    way to truncate the long tail of the distribution.
//!
//! 3. **Top-p (nucleus) filtering** — restricts sampling to the smallest set
//!    of tokens whose cumulative probability exceeds `p`. Unlike top-k, which
//!    uses a fixed count, top-p adapts to the shape of the distribution:
//!    when the model is confident, it may keep only 1–2 tokens; when it is
//!    uncertain, it may keep many. This produces more natural text than top-k
//!    alone.
//!
//! 4. **CDF sampling** — after filtering, we draw a single random number and
//!    walk the cumulative distribution to pick a token. This is standard
//!    categorical sampling.
//!
//! # Why these strategies matter
//!
//! - **Temperature** lets you trade off quality vs. creativity. Near-zero
//!   temperature is good for code generation; higher temperature is better
//!   for brainstorming or storytelling.
//!
//! - **Top-k** prevents degenerate outputs where the model occasionally
//!   samples a very low-probability token that breaks coherence.
//!
//! - **Top-p** is more principled than top-k because it adapts to the
//!   confidence of the model. A confident prediction (one token at 0.95)
//!   should not be diluted by keeping 50 candidates.
//!
//! # PRNG
//!
//! Randomness is provided by [`XorShift64`], a tiny xorshift64 PRNG
//! implemented without external crates. It is deterministic and fast,
//! making it suitable for reproducible inference, but it is **not**
//! cryptographically secure.

// ─────────────────────────────────────────────────────────────────────────────
// PRNG
// ─────────────────────────────────────────────────────────────────────────────

/// Simple xorshift64 pseudo-random number generator.
///
/// This is a classic xorshift generator with shift triple (13, 7, 17).
/// It passes standard statistical tests and has a period of 2^64 - 1,
/// which is more than sufficient for token sampling. It is **not**
/// cryptographically secure — do not use it for key generation, etc.
///
/// # Example
///
/// ```ignore
/// let mut rng = XorShift64::new(42);
/// let val = rng.next_f32(); // random f32 in [0, 1)
/// ```
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Create a new PRNG with the given seed.
    ///
    /// # Panics
    ///
    /// Panics if `seed == 0` because xorshift cannot escape the zero state.
    pub fn new(seed: u64) -> Self {
        assert!(seed != 0, "XorShift64 seed must not be zero");
        Self { state: seed }
    }

    /// Generate the next pseudo-random `u64`.
    ///
    /// Uses the xorshift64 algorithm with shift triple (13, 7, 17).
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a random `f32` uniformly in `[0, 1)`.
    ///
    /// Takes the top 24 bits of `next_u64()` and divides by 2^24 so that
    /// the result falls in the half-open interval [0, 1). This gives
    /// approximately 7 decimal digits of precision, which is adequate for
    /// sampling decisions.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for token sampling.
///
/// All strategies are optional — setting `top_k = 0` disables top-k filtering,
/// and setting `top_p = 1.0` disables top-p filtering. Temperature 0.0
/// triggers greedy decoding and skips all other strategies.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature for scaling logits. 0.0 = greedy (argmax).
    ///
    /// Values < 1.0 sharpen the distribution (prefer high-probability tokens),
    /// values > 1.0 flatten it (more uniform / creative), and 0.0 selects
    /// the single most probable token deterministically.
    pub temperature: f32,

    /// Top-k: only consider the `k` highest-probability tokens.
    ///
    /// Set to 0 to disable top-k filtering (consider the full vocabulary).
    pub top_k: usize,

    /// Top-p (nucleus): only consider tokens with cumulative probability >= p.
    ///
    /// Set to 1.0 to disable top-p filtering. Values closer to 0.0 keep
    /// fewer tokens; values closer to 1.0 keep more.
    pub top_p: f32,

    /// Random seed for reproducibility.
    ///
    /// When `Some(seed)`, sampling is deterministic: the same logits and
    /// config always produce the same token. When `None`, a default seed
    /// is derived from a simple counter (still deterministic within a
    /// single process run, but not across runs unless you set an explicit
    /// seed).
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    /// Default configuration: temperature 0.7, top-k 50, top-p 0.9, no seed.
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            seed: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Find the index of the maximum value in a slice.
///
/// If multiple elements share the maximum value, returns the **first**
/// such index (leftmost).
fn argmax(slice: &[f32]) -> usize {
    assert!(!slice.is_empty(), "argmax: cannot operate on empty slice");
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

/// Numerically stable in-place softmax.
///
/// Converts `logits` into a probability distribution (all values in `[0, 1]`
/// and summing to 1). Uses the max-subtraction trick to prevent overflow
/// in `exp()`:
///
/// ```text
/// softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
/// ```
///
/// Elements that are `-inf` (e.g. from top-k filtering) correctly become
/// zero probability because `exp(-inf) = 0`.
fn softmax_in_place(logits: &mut [f32]) {
    assert!(!logits.is_empty(), "softmax_in_place: cannot operate on empty slice");

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

// ─────────────────────────────────────────────────────────────────────────────
// Public sampling functions
// ─────────────────────────────────────────────────────────────────────────────

/// Greedy decoding: always pick the token with the highest logit.
///
/// This is the simplest sampling strategy — it is deterministic and
/// always produces the same output for the same input. It is equivalent
/// to temperature = 0.
///
/// # Arguments
///
/// * `logits` — raw model output scores, one per vocabulary token.
///
/// # Returns
///
/// The index (token ID) of the maximum logit value.
pub fn sample_greedy(logits: &[f32]) -> usize {
    argmax(logits)
}

/// Top-k sampling: restrict candidates to the `k` highest-scoring tokens.
///
/// After applying temperature scaling, this function finds the top-`k`
/// logits by value, sets all other logits to `-inf`, then samples from
/// the resulting distribution via softmax + CDF sampling.
///
/// Top-k is useful because it eliminates the long tail of very unlikely
/// tokens that can still be sampled when using pure temperature scaling,
/// especially with large vocabularies.
///
/// # Arguments
///
/// * `logits` — raw model output scores (temperature should already be
///   applied by the caller).
/// * `k` — number of candidate tokens to keep. If `k >= logits.len()`,
///   no filtering is applied.
/// * `rng` — random number generator for sampling.
///
/// # Returns
///
/// A sampled token ID.
pub fn sample_top_k(logits: &[f32], k: usize, rng: &mut XorShift64) -> usize {
    let n = logits.len();
    assert!(n > 0, "sample_top_k: logits must not be empty");

    // If k covers everything, just softmax and sample.
    if k >= n {
        let mut probs = logits.to_vec();
        softmax_in_place(&mut probs);
        return sample_from_cdf(&probs, rng);
    }

    // Find the k-th largest value as a threshold.
    // We collect (index, value) pairs, sort descending by value, and use
    // the k-th value as the cutoff.
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let threshold = indexed[k - 1].1;

    // Build a filtered logits vector: keep values >= threshold,
    // set the rest to -inf.
    let mut filtered = logits.to_vec();
    for v in filtered.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }

    // Softmax over the remaining logits, then sample.
    softmax_in_place(&mut filtered);
    sample_from_cdf(&filtered, rng)
}

/// Top-p (nucleus) sampling: keep the smallest set of tokens whose
/// cumulative probability exceeds `p`.
///
/// After softmax, tokens are sorted by probability in descending order.
/// We accumulate probabilities until the sum reaches `p`, then zero out
/// all remaining tokens and renormalize. This adapts to the model's
/// confidence: when the model is very sure, only a few tokens are kept;
/// when it is uncertain, more candidates are included.
///
/// Top-p is generally preferred over top-k because it adapts dynamically
/// to the shape of the probability distribution.
///
/// # Arguments
///
/// * `logits` — raw model output scores (temperature should already be
///   applied by the caller).
/// * `p` — cumulative probability threshold in (0, 1]. Set to 1.0 to
///   disable top-p filtering.
/// * `rng` — random number generator for sampling.
///
/// # Returns
///
/// A sampled token ID.
pub fn sample_top_p(logits: &[f32], p: f32, rng: &mut XorShift64) -> usize {
    let n = logits.len();
    assert!(n > 0, "sample_top_p: logits must not be empty");

    // Step 1: compute probabilities via softmax.
    let mut probs = logits.to_vec();
    softmax_in_place(&mut probs);

    // Step 2: sort indices by probability descending.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));

    // Step 3: find the cutoff — keep tokens until cumulative prob >= p.
    let mut cumsum = 0.0f32;
    let mut cutoff = 0; // number of tokens to keep
    for &idx in &indices {
        cumsum += probs[idx];
        cutoff += 1;
        if cumsum >= p {
            break;
        }
    }

    // Step 4: zero out tokens beyond the cutoff.
    for &idx in &indices[cutoff..] {
        probs[idx] = 0.0;
    }

    // Step 5: renormalize.
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for v in probs.iter_mut() {
            *v /= sum;
        }
    }

    // Step 6: sample from the renormalized distribution.
    sample_from_cdf(&probs, rng)
}

/// Sample from a probability distribution using the cumulative distribution
/// function (CDF) method.
///
/// Given a slice of probabilities that sum to 1.0, draw a random number `r`
/// in `[0, 1)` and return the first index where the cumulative sum of
/// probabilities exceeds `r`.
fn sample_from_cdf(probs: &[f32], rng: &mut XorShift64) -> usize {
    let r = rng.next_f32();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum > r {
            return i;
        }
    }
    // Fallback: return the last index (handles floating-point rounding).
    probs.len() - 1
}

/// Main sampling function that applies the full pipeline.
///
/// The pipeline applies strategies in this order:
///
/// 1. **Temperature scaling**: if temperature is 0.0, return greedy result
///    immediately. Otherwise, divide all logits by temperature.
///
/// 2. **Top-k filtering**: if `top_k > 0`, restrict to the `k` highest
///    logits (set the rest to `-inf`).
///
/// 3. **Softmax**: convert filtered logits to probabilities.
///
/// 4. **Top-p filtering**: sort by probability descending, keep the smallest
///    set whose cumulative probability >= `top_p`, zero out the rest, and
///    renormalize.
///
/// 5. **CDF sampling**: draw a random number and walk the cumulative
///    distribution to pick a token.
///
/// # Arguments
///
/// * `logits` — raw model output scores of length `vocab_size`.
/// * `config` — sampling configuration (temperature, top-k, top-p, seed).
///
/// # Returns
///
/// A sampled token ID (index into the logits slice).
///
/// # Example
///
/// ```ignore
/// let config = SamplingConfig {
///     temperature: 0.8,
///     top_k: 50,
///     top_p: 0.95,
///     seed: Some(42),
/// };
/// let token_id = sample(&logits, &config);
/// ```
pub fn sample(logits: &[f32], config: &SamplingConfig) -> usize {
    assert!(!logits.is_empty(), "sample: logits must not be empty");

    // Step 1: temperature == 0 => greedy.
    if config.temperature == 0.0 {
        return sample_greedy(logits);
    }

    // Step 2: apply temperature scaling.
    let mut scaled: Vec<f32> = logits.iter().map(|&l| l / config.temperature).collect();

    // Step 3: apply top-k filtering (set non-top-k logits to -inf).
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

    // Step 4: softmax to get probabilities.
    softmax_in_place(&mut scaled);

    // Step 5: apply top-p filtering.
    if config.top_p < 1.0 {
        let n = scaled.len();
        // Sort indices by probability descending.
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            scaled[b].partial_cmp(&scaled[a]).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find cutoff.
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

    // Step 6: CDF sampling with our own PRNG.
    let mut rng = XorShift64::new(config.seed.unwrap_or(12345));
    let r = rng.next_f32();
    let mut cumsum = 0.0f32;
    for (i, &p) in scaled.iter().enumerate() {
        cumsum += p;
        if cumsum > r {
            return i;
        }
    }

    // Fallback: return last index (handles floating-point rounding).
    scaled.len() - 1
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_returns_argmax() {
        let logits = [0.1, 0.5, 3.0, 0.2, -1.0];
        assert_eq!(sample_greedy(&logits), 2, "greedy should pick index 2 (value 3.0)");

        let logits2 = [5.0, 1.0, 2.0];
        assert_eq!(sample_greedy(&logits2), 0, "greedy should pick index 0 (value 5.0)");

        // Tie-breaking: first index wins.
        let logits3 = [1.0, 1.0, 1.0];
        assert_eq!(sample_greedy(&logits3), 0, "greedy should pick first index on tie");
    }

    #[test]
    fn test_temperature_zero_is_greedy() {
        let logits = [0.1, 0.5, 3.0, 0.2, -1.0];
        let config = SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        };
        assert_eq!(
            sample(&logits, &config),
            sample_greedy(&logits),
            "temperature 0 should behave identically to greedy"
        );
    }

    #[test]
    fn test_high_temperature_flattens() {
        // With very high temperature, the distribution should be nearly uniform.
        let logits = [0.0, 10.0, 20.0];
        let high_temp = 1000.0;

        let mut scaled: Vec<f32> = logits.iter().map(|&l| l / high_temp).collect();
        softmax_in_place(&mut scaled);

        // All probabilities should be close to 1/3 ≈ 0.333.
        let uniform = 1.0 / 3.0;
        for (i, &p) in scaled.iter().enumerate() {
            assert!(
                (p - uniform).abs() < 0.05,
                "prob[{}] = {} should be close to uniform {}",
                i, p, uniform
            );
        }
    }

    #[test]
    fn test_low_temperature_sharpens() {
        // With very low temperature, the highest logit should dominate.
        let logits = [1.0, 2.0, 3.0];
        let low_temp = 0.01;

        let mut scaled: Vec<f32> = logits.iter().map(|&l| l / low_temp).collect();
        softmax_in_place(&mut scaled);

        // The last element (logit 3.0) should have probability very close to 1.
        assert!(
            scaled[2] > 0.99,
            "low temperature should concentrate probability on the max, got {}",
            scaled[2]
        );
    }

    #[test]
    fn test_top_k_limits_candidates() {
        // Create logits where 5 tokens have high values and the rest are low.
        let mut logits = vec![-10.0; 10];
        logits[2] = 5.0;
        logits[5] = 4.0;
        logits[7] = 3.0;
        logits[1] = 2.0;
        logits[8] = 1.0;

        let k = 3;
        // Apply top-k manually to verify the logic.
        let mut sorted: Vec<f32> = logits.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[k - 1]; // 3rd largest = 3.0

        let mut filtered = logits.clone();
        for v in filtered.iter_mut() {
            if *v < threshold {
                *v = f32::NEG_INFINITY;
            }
        }
        softmax_in_place(&mut filtered);

        // Only indices 2, 5, 7 should have non-zero probability.
        let non_zero: Vec<usize> = filtered
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(i, _)| i)
            .collect();

        assert_eq!(non_zero, vec![2, 5, 7], "top-k=3 should keep only the 3 highest logits");
    }

    #[test]
    fn test_top_p_nucleus_sampling() {
        // Create a distribution where one token dominates.
        let logits = [10.0, 1.0, 0.5, 0.1, -5.0];
        let mut probs = logits.to_vec();
        softmax_in_place(&mut probs);

        // The first token should have very high probability.
        // With top_p = 0.9, only the first 1-2 tokens should be kept.
        let p = 0.9f32;

        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0f32;
        let mut cutoff = 0;
        for &idx in &indices {
            cumsum += probs[idx];
            cutoff += 1;
            if cumsum >= p {
                break;
            }
        }

        // Verify the kept tokens have cumulative probability >= p.
        let kept_prob: f32 = indices[..cutoff].iter().map(|&idx| probs[idx]).sum();
        assert!(
            kept_prob >= p - 1e-6,
            "kept probability {} should be >= top_p {}",
            kept_prob, p
        );

        // Verify that removing one more token from the front would drop below p.
        if cutoff > 1 {
            let reduced_prob: f32 = indices[..cutoff - 1].iter().map(|&idx| probs[idx]).sum();
            assert!(
                reduced_prob < p,
                "removing the last kept token should drop below top_p: {} >= {}",
                reduced_prob, p
            );
        }
    }

    #[test]
    fn test_xorshift64_deterministic() {
        let mut rng1 = XorShift64::new(42);
        let mut rng2 = XorShift64::new(42);

        for _ in 0..100 {
            assert_eq!(
                rng1.next_u64(),
                rng2.next_u64(),
                "same seed should produce identical sequences"
            );
        }
    }

    #[test]
    fn test_sample_deterministic_with_seed() {
        let logits = [0.1, 0.5, 3.0, 0.2, -1.0, 2.0, 0.0, 1.5, -0.3, 0.8];

        let config1 = SamplingConfig {
            temperature: 0.8,
            top_k: 5,
            top_p: 0.9,
            seed: Some(12345),
        };
        let config2 = SamplingConfig {
            temperature: 0.8,
            top_k: 5,
            top_p: 0.9,
            seed: Some(12345),
        };

        let result1 = sample(&logits, &config1);
        let result2 = sample(&logits, &config2);
        assert_eq!(
            result1, result2,
            "same seed should produce identical sampling results"
        );
    }

    // ── Additional helper tests ─────────────────────────────────────────────

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[1.0, 5.0, 3.0]), 1);
        assert_eq!(argmax(&[9.0, 1.0, 2.0]), 0);
        assert_eq!(argmax(&[1.0, 2.0, 9.0]), 2);
    }

    #[test]
    fn test_softmax_in_place_sums_to_one() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        softmax_in_place(&mut v);
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax should sum to 1.0, got {}", sum);
    }

    #[test]
    fn test_softmax_in_place_stability() {
        let mut v = vec![1000.0, 1001.0, 1002.0];
        softmax_in_place(&mut v);
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "numerically stable softmax should sum to 1.0, got {}", sum);
    }

    #[test]
    fn test_softmax_handles_neg_inf() {
        let mut v = vec![1.0, f32::NEG_INFINITY, 2.0, f32::NEG_INFINITY];
        softmax_in_place(&mut v);
        assert_eq!(v[1], 0.0, "-inf should become probability 0");
        assert_eq!(v[3], 0.0, "-inf should become probability 0");
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax should sum to 1.0, got {}", sum);
    }

    #[test]
    fn test_xorshift64_next_f32_range() {
        let mut rng = XorShift64::new(999);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!(v >= 0.0 && v < 1.0, "next_f32 should be in [0, 1), got {}", v);
        }
    }

    #[test]
    fn test_xorshift64_rejects_zero_seed() {
        let result = std::panic::catch_unwind(|| XorShift64::new(0));
        assert!(result.is_err(), "XorShift64::new(0) should panic");
    }

    #[test]
    fn test_sample_greedy_via_config() {
        // Temperature = 0 with top_k and top_p set should still be greedy.
        let logits = [0.1, 0.5, 3.0, 0.2];
        let config = SamplingConfig {
            temperature: 0.0,
            top_k: 2,
            top_p: 0.5,
            seed: Some(42),
        };
        assert_eq!(sample(&logits, &config), 2);
    }

    #[test]
    fn test_default_config() {
        let config = SamplingConfig::default();
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_k, 50);
        assert!((config.top_p - 0.9).abs() < 1e-6);
        assert!(config.seed.is_none());
    }
}
