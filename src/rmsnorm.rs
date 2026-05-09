//! RMSNorm (Root Mean Square Normalization).
//!
//! This module implements RMSNorm as a reusable struct that wraps the
//! low-level `Tensor::rms_norm` operation. In the Qwen3 model, RMSNorm
//! is used in three places:
//!
//! - Before self-attention in each transformer block (`input_layernorm`)
//! - Before the FFN in each transformer block (`post_attention_layernorm`)
//! - After all transformer blocks, before the `lm_head` projection
//!
//! # Why RMSNorm?
//!
//! RMSNorm is a simplified variant of LayerNorm that removes the mean
//! subtraction and the learnable bias term. The formula is:
//!
//! ```text
//! RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
//! ```
//!
//! Compared to LayerNorm:
//!
//! ```text
//! LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
//! ```
//!
//! RMSNorm is cheaper to compute (no mean subtraction, no bias parameter)
//! while achieving nearly identical performance. It is the standard
//! normalization in modern LLMs including LLaMA, Qwen, and Mistral.

use crate::tensor::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// RMSNorm struct
// ─────────────────────────────────────────────────────────────────────────────

/// Root Mean Square Normalization layer.
///
/// RMSNorm normalizes a vector by dividing each element by the root mean
/// square (RMS) of the vector, then multiplying element-wise by a learned
/// weight parameter.
///
/// # Formula
///
/// For an input vector **x** of length `n`:
///
/// ```text
/// RMSNorm(x) = weight * (x / sqrt(mean(x²) + eps))
///
/// where:
///   mean(x²) = (1/n) * Σ_i x_i²
///   eps      = a small constant to prevent division by zero (1e-6 in Qwen3)
///   weight   = a learnable parameter vector of shape [n]
/// ```
///
/// # Fields
///
/// - `weight`: A 1-D tensor of shape `[hidden_size]`. This is the learned
///   scaling parameter (called `gamma` in some implementations). After
///   normalization, each element is multiplied by the corresponding weight
///   element, allowing the model to re-scale individual dimensions.
///
/// - `eps`: A small floating-point constant added inside the square root
///   to avoid division by zero when all input elements are zero. Qwen3
///   uses `1e-6`.
///
/// # When to use this
///
/// In a transformer, RMSNorm is applied before each sub-layer (attention
/// and FFN). This "pre-norm" design stabilizes training by ensuring that
/// activations never grow too large or too small before they enter a
/// sub-layer.
pub struct RMSNorm {
    /// Learned scaling parameter of shape `[hidden_size]`.
    /// After normalizing the input, each dimension is multiplied by the
    /// corresponding weight element, allowing the network to learn an
    /// optimal per-dimension scale.
    weight: Tensor,

    /// Epsilon constant for numerical stability.
    /// Added to the mean of squares before taking the square root, so
    /// that we never divide by zero. Qwen3 uses 1e-6.
    eps: f32,
}

impl RMSNorm {
    /// Create a new RMSNorm layer from a weight tensor and epsilon value.
    ///
    /// # Arguments
    ///
    /// * `weight` - A 1-D tensor of shape `[hidden_size]` containing the
    ///   learned per-dimension scaling factors. This is typically loaded
    ///   from the model's safetensors file (e.g., key
    ///   `model.layers.0.input_layernorm.weight`).
    ///
    /// * `eps` - A small constant added to the mean of squares before the
    ///   square root to prevent division by zero. For Qwen3, this is
    ///   always `1e-6` (from the `rms_norm_eps` field in `config.json`).
    ///
    /// # Panics
    ///
    /// Panics if `weight` is not a 1-D tensor.
    ///
    /// # Example (pseudocode)
    ///
    /// ```ignore
    /// // Load weight from safetensors, shape [1024]
    /// let weight = safetensors.load("model.layers.0.input_layernorm.weight");
    /// let norm = RMSNorm::new(weight, 1e-6);
    /// let normalized = norm.forward(&hidden_states);
    /// ```
    pub fn new(weight: Tensor, eps: f32) -> Self {
        assert_eq!(
            weight.ndim(),
            1,
            "RMSNorm::new: weight must be 1-D, got shape {:?}",
            weight.shape(),
        );
        Self { weight, eps }
    }

    /// Apply RMSNorm to the input tensor.
    ///
    /// # The computation, step by step
    ///
    /// For each row (i.e., each vector along the last dimension) of the
    /// input tensor:
    ///
    /// 1. **Square each element**: compute `x_i²` for every element.
    ///
    /// 2. **Compute the mean of squares**:
    ///    ```text
    ///    mean_sq = (1/n) * Σ x_i²
    ///    ```
    ///    where `n` is the length of the last dimension (`hidden_size`).
    ///
    /// 3. **Add epsilon** for numerical stability:
    ///    ```text
    ///    mean_sq + eps
    ///    ```
    ///
    /// 4. **Take the square root** to get the root mean square:
    ///    ```text
    ///    rms = sqrt(mean_sq + eps)
    ///    ```
    ///
    /// 5. **Divide each element by the RMS**: this normalizes the vector
    ///    so that its RMS value becomes 1.
    ///    ```text
    ///    normalized_i = x_i / rms
    ///    ```
    ///
    /// 6. **Multiply by the learned weight**:
    ///    ```text
    ///    output_i = weight_i * normalized_i
    ///    ```
    ///
    /// This produces the final output. The weight allows the model to
    /// recover any scaling that the normalization removed, on a
    /// per-dimension basis.
    ///
    /// # Input shapes
    ///
    /// - Input: `[seq_len, hidden_size]` (typical) or `[1, hidden_size]`
    ///   (single-token decode step)
    /// - Output: same shape as input
    ///
    /// The normalization is computed independently for each row (each
    /// token's hidden state), over the last dimension (`hidden_size`).
    ///
    /// # Delegation
    ///
    /// The actual math is performed by `Tensor::rms_norm`, which
    /// implements the loop over rows and the per-row computation
    /// described above. This method simply stores the weight and eps
    /// and delegates to that operation.
    ///
    /// # Example (pseudocode)
    ///
    /// ```ignore
    /// // hidden_states has shape [seq_len, 1024]
    /// let normalized = rmsnorm.forward(&hidden_states);
    /// // normalized has shape [seq_len, 1024]
    /// ```
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Step 1-5: Compute x / sqrt(mean(x²) + eps) for each row.
        // Step 6: Multiply by the learned weight element-wise.
        // Both steps are handled by Tensor::rms_norm, which iterates
        // over rows, computes the RMS, normalizes, and scales by weight.
        x.rms_norm(&self.weight, self.eps)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Test RMSNorm with a simple known vector and unit weight.
    ///
    /// Input:  x = [3.0, 4.0]
    /// Weight: w = [1.0, 1.0]
    /// Eps:    1e-6
    ///
    /// Step 1: Square each element → [9.0, 16.0]
    /// Step 2: Mean of squares → (9.0 + 16.0) / 2 = 12.5
    /// Step 3: Add eps → 12.5 + 1e-6 ≈ 12.5
    /// Step 4: Square root → sqrt(12.5) ≈ 3.5355
    /// Step 5: Divide → [3.0/3.5355, 4.0/3.5355] ≈ [0.8485, 1.1314]
    /// Step 6: Multiply by weight [1.0, 1.0] → [0.8485, 1.1314]
    #[test]
    fn test_rmsnorm_simple_vector_unit_weight() {
        let x = Tensor::new(vec![2], vec![3.0, 4.0]);
        let w = Tensor::new(vec![2], vec![1.0, 1.0]);
        let norm = RMSNorm::new(w, 1e-6);

        let output = norm.forward(&x);

        // Expected: [3.0 / sqrt(12.5), 4.0 / sqrt(12.5)]
        // sqrt(12.5) = 3.5355339...
        // 3.0 / 3.5355 ≈ 0.8485
        // 4.0 / 3.5355 ≈ 1.1314
        assert!(
            (output.data()[0] - 0.8485).abs() < 1e-3,
            "expected ~0.8485, got {}",
            output.data()[0],
        );
        assert!(
            (output.data()[1] - 1.1314).abs() < 1e-3,
            "expected ~1.1314, got {}",
            output.data()[1],
        );
    }

    /// Test RMSNorm with a non-trivial weight.
    ///
    /// Input:  x = [3.0, 4.0]
    /// Weight: w = [2.0, 0.5]
    /// Eps:    1e-6
    ///
    /// After normalization: [0.8485, 1.1314] (same as unit weight test)
    /// After weight multiply: [0.8485 * 2.0, 1.1314 * 0.5] ≈ [1.6971, 0.5657]
    #[test]
    fn test_rmsnorm_nontrivial_weight() {
        let x = Tensor::new(vec![2], vec![3.0, 4.0]);
        let w = Tensor::new(vec![2], vec![2.0, 0.5]);
        let norm = RMSNorm::new(w, 1e-6);

        let output = norm.forward(&x);

        // After normalization: [~0.8485, ~1.1314]
        // After weight multiply: [0.8485 * 2.0, 1.1314 * 0.5] ≈ [1.6971, 0.5657]
        assert!(
            (output.data()[0] - 1.6971).abs() < 1e-3,
            "expected ~1.6971, got {}",
            output.data()[0],
        );
        assert!(
            (output.data()[1] - 0.5657).abs() < 1e-3,
            "expected ~0.5657, got {}",
            output.data()[1],
        );
    }

    /// Test RMSNorm with a 2D input (seq_len > 1).
    ///
    /// Each row should be normalized independently.
    ///
    /// Row 0: [3.0, 4.0] → normalized ≈ [0.8485, 1.1314] (with weight [1.0, 1.0])
    /// Row 1: [6.0, 8.0] → same direction, different magnitude
    ///   mean_sq = (36 + 64) / 2 = 50.0
    ///   rms = sqrt(50.0) ≈ 7.0711
    ///   normalized = [6.0/7.0711, 8.0/7.0711] ≈ [0.8485, 1.1314]
    ///
    /// Notice that [3, 4] and [6, 8] point in the same direction (one is
    /// 2x the other), so after RMSNorm they should produce the same
    /// output. This is a key property: RMSNorm is invariant to scalar
    /// multiplication of the input.
    #[test]
    fn test_rmsnorm_2d_input() {
        let x = Tensor::new(vec![2, 2], vec![3.0, 4.0, 6.0, 8.0]);
        let w = Tensor::new(vec![2], vec![1.0, 1.0]);
        let norm = RMSNorm::new(w, 1e-6);

        let output = norm.forward(&x);

        // Both rows should produce the same output (direction-invariant)
        assert!(
            (output.data()[0] - 0.8485).abs() < 1e-3,
            "row 0, element 0: expected ~0.8485, got {}",
            output.data()[0],
        );
        assert!(
            (output.data()[1] - 1.1314).abs() < 1e-3,
            "row 0, element 1: expected ~1.1314, got {}",
            output.data()[1],
        );
        assert!(
            (output.data()[2] - 0.8485).abs() < 1e-3,
            "row 1, element 0: expected ~0.8485, got {}",
            output.data()[2],
        );
        assert!(
            (output.data()[3] - 1.1314).abs() < 1e-3,
            "row 1, element 1: expected ~1.1314, got {}",
            output.data()[3],
        );
    }

    /// Test that the output is different from the input (i.e., the
    /// normalization actually does something).
    ///
    /// This is a basic sanity check: if forward() returned the input
    /// unchanged, something would be fundamentally wrong. The only case
    /// where the output equals the input is when every element of the
    /// input already has an RMS of 1 *and* every weight element is 1.
    #[test]
    fn test_rmsnorm_output_differs_from_input() {
        let x = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]);
        let w = Tensor::new(vec![3], vec![1.0, 1.0, 1.0]);
        let norm = RMSNorm::new(w, 1e-6);

        let output = norm.forward(&x);

        // The output should differ from the input for at least one element.
        let any_different = x
            .data()
            .iter()
            .zip(output.data().iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            any_different,
            "RMSNorm output should differ from input when input RMS is not 1"
        );
    }

    /// Test that the RMS of the normalized output (before weight scaling)
    /// equals 1 when weight is all-ones.
    ///
    /// This verifies the core invariant of RMSNorm: after normalization
    /// (ignoring weight), the root mean square of the output should be 1.
    #[test]
    fn test_rmsnorm_output_rms_is_one() {
        let x = Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let w = Tensor::new(vec![4], vec![1.0, 1.0, 1.0, 1.0]);
        let norm = RMSNorm::new(w, 1e-6);

        let output = norm.forward(&x);

        // Compute RMS of the output
        let sum_sq: f32 = output.data().iter().map(|v| v * v).sum();
        let rms = (sum_sq / output.len() as f32).sqrt();

        assert!(
            (rms - 1.0).abs() < 1e-4,
            "RMS of normalized output should be 1.0, got {}",
            rms,
        );
    }
}
