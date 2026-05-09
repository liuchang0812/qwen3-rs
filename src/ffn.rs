//! SwiGLU Feed-Forward Network.
//!
//! This module implements the feed-forward network (FFN) used in each
//! transformer block of the Qwen3 model. The architecture is SwiGLU
//! (Swish-Gated Linear Unit), which is the standard FFN variant in all
//! modern large language models including LLaMA, Qwen, Mistral, and Gemma.
//!
//! # SwiGLU Formula
//!
//! ```text
//! output = W_down · (SiLU(W_gate · x) ⊙ (W_up · x))
//! ```
//!
//! Where:
//! - `W_gate` projects the input to the intermediate dimension, then applies
//!   the SiLU activation to produce a *gate* signal
//! - `W_up` projects the input to the intermediate dimension as the *value*
//!   path
//! - `⊙` is element-wise multiplication (the gate modulates the value)
//! - `W_down` projects back from the intermediate dimension to the hidden
//!   dimension
//!
//! # Dimensions in Qwen3-0.6B
//!
//! | Component    | Shape                  | Parameters |
//! |-------------|------------------------|------------|
//! | `gate_proj` | [3072, 1024]           | 3,145,728  |
//! | `up_proj`   | [3072, 1024]           | 3,145,728  |
//! | `down_proj` | [1024, 3072]           | 3,145,728  |
//! | **Total**   |                        | **9,437,184** |
//!
//! # Weight Layout Convention
//!
//! The weight matrices from safetensors are stored as
//! `[out_features, in_features]`, which is the PyTorch convention.
//! For a linear layer computing `y = x · W^T`, the weight `W` has shape
//! `[out_dim, in_dim]`. We must transpose before matmul:
//!
//! ```text
//! y = x.matmul(&W.transpose_2d())
//! ```

use crate::tensor::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// FeedForward struct
// ─────────────────────────────────────────────────────────────────────────────

/// SwiGLU Feed-Forward Network.
///
/// The FFN is the "thinking" part of each transformer block. While attention
/// gathers information from other tokens, the FFN processes each token's
/// representation independently — it transforms features without looking at
/// other positions.
///
/// # Architecture
///
/// The SwiGLU FFN has three learned projections:
///
/// 1. **gate_proj** — projects input to the intermediate dimension; the
///    result passes through SiLU activation, producing a soft gate that
///    controls information flow.
///
/// 2. **up_proj** — projects input to the intermediate dimension; this is
///    the "value" path whose information is gated.
///
/// 3. **down_proj** — projects from the intermediate dimension back to the
///    hidden dimension, producing the final output.
///
/// The computation proceeds as:
///
/// ```text
/// gate = SiLU(x · gate_proj^T)    // [seq_len, intermediate_size]
/// up   = x · up_proj^T            // [seq_len, intermediate_size]
/// out  = (gate ⊙ up) · down_proj^T  // [seq_len, hidden_size]
/// ```
///
/// # Why three projections instead of two?
///
/// A standard (vanilla) FFN has only two projections: an "up" projection
/// that expands the dimension and a "down" projection that contracts it.
/// SwiGLU adds a third projection (`gate_proj`) to implement a gating
/// mechanism. The gate learns which features to pass through and which
/// to suppress, giving the model more expressive power per parameter.
pub struct FeedForward {
    /// Gate projection weight of shape `[intermediate_size, hidden_size]`.
    /// Projects the input to the intermediate dimension; the result is
    /// passed through SiLU to form the gate signal.
    gate_proj: Tensor,

    /// Up projection weight of shape `[intermediate_size, hidden_size]`.
    /// Projects the input to the intermediate dimension as the value path.
    up_proj: Tensor,

    /// Down projection weight of shape `[hidden_size, intermediate_size]`.
    /// Projects the gated intermediate representation back to the hidden
    /// dimension.
    down_proj: Tensor,
}

impl FeedForward {
    /// Create a new SwiGLU Feed-Forward Network from its three weight
    /// matrices.
    ///
    /// # Arguments
    ///
    /// * `gate_proj` — Weight of shape `[intermediate_size, hidden_size]`,
    ///   e.g. `[3072, 1024]` in Qwen3-0.6B. Loaded from safetensors keys
    ///   like `model.layers.N.mlp.gate_proj.weight`.
    ///
    /// * `up_proj` — Weight of shape `[intermediate_size, hidden_size]`,
    ///   e.g. `[3072, 1024]`. Loaded from keys like
    ///   `model.layers.N.mlp.up_proj.weight`.
    ///
    /// * `down_proj` — Weight of shape `[hidden_size, intermediate_size]`,
    ///   e.g. `[1024, 3072]`. Loaded from keys like
    ///   `model.layers.N.mlp.down_proj.weight`.
    ///
    /// # Panics
    ///
    /// Panics if any weight is not 2-D, or if the dimensions are
    /// incompatible (gate_proj and up_proj must share the same shape,
    /// and down_proj's dimensions must be the reverse of gate_proj's).
    pub fn new(gate_proj: Tensor, up_proj: Tensor, down_proj: Tensor) -> Self {
        // Validate that all weights are 2-D.
        assert_eq!(
            gate_proj.ndim(),
            2,
            "FeedForward::new: gate_proj must be 2-D, got shape {:?}",
            gate_proj.shape(),
        );
        assert_eq!(
            up_proj.ndim(),
            2,
            "FeedForward::new: up_proj must be 2-D, got shape {:?}",
            up_proj.shape(),
        );
        assert_eq!(
            down_proj.ndim(),
            2,
            "FeedForward::new: down_proj must be 2-D, got shape {:?}",
            down_proj.shape(),
        );

        // Validate that gate_proj and up_proj have the same shape.
        assert_eq!(
            gate_proj.shape(),
            up_proj.shape(),
            "FeedForward::new: gate_proj shape {:?} must match up_proj shape {:?}",
            gate_proj.shape(),
            up_proj.shape(),
        );

        // Validate that down_proj's dimensions are the reverse of gate_proj's.
        // gate_proj: [intermediate_size, hidden_size]
        // down_proj: [hidden_size, intermediate_size]
        assert_eq!(
            down_proj.shape()[0],
            gate_proj.shape()[1],
            "FeedForward::new: down_proj rows ({}) must equal gate_proj cols ({})",
            down_proj.shape()[0],
            gate_proj.shape()[1],
        );
        assert_eq!(
            down_proj.shape()[1],
            gate_proj.shape()[0],
            "FeedForward::new: down_proj cols ({}) must equal gate_proj rows ({})",
            down_proj.shape()[1],
            gate_proj.shape()[0],
        );

        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// Run the SwiGLU feed-forward computation.
    ///
    /// # The computation, step by step
    ///
    /// Given input `x` of shape `[seq_len, hidden_size]`:
    ///
    /// 1. **Gate path**: Compute `x · gate_proj^T` to get a tensor of shape
    ///    `[seq_len, intermediate_size]`, then apply SiLU activation.
    ///    The SiLU function is `SiLU(z) = z * sigmoid(z) = z / (1 + e^{-z})`.
    ///    This produces a soft gate: values are near-zero for negative inputs
    ///    and roughly linear for large positive inputs.
    ///
    /// 2. **Up path**: Compute `x · up_proj^T` to get a tensor of shape
    ///    `[seq_len, intermediate_size]`. This is the value signal that
    ///    the gate will modulate.
    ///
    /// 3. **Gate modulation**: Element-wise multiply the gate and up tensors.
    ///    Where the gate is near zero, information is suppressed; where the
    ///    gate is large, information passes through.
    ///
    /// 4. **Down projection**: Compute `(gate * up) · down_proj^T` to
    ///    project back to `[seq_len, hidden_size]`.
    ///
    /// # Input shapes
    ///
    /// - Input: `[seq_len, hidden_size]`
    /// - Output: `[seq_len, hidden_size]` (same shape as input)
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Step 1: Gate path — project then activate with SiLU.
        // gate_proj is [intermediate_size, hidden_size], transpose to
        // [hidden_size, intermediate_size] so x [seq_len, hidden_size]
        // matmul gives [seq_len, intermediate_size].
        let gate = x.matmul(&self.gate_proj.transpose_2d()).silu();

        // Step 2: Up path — project (no activation).
        let up = x.matmul(&self.up_proj.transpose_2d());

        // Step 3: Element-wise multiply — the gate modulates the up path.
        let gated = gate.mul_elementwise(&up);

        // Step 4: Down projection — project back to hidden_size.
        // down_proj is [hidden_size, intermediate_size], transpose to
        // [intermediate_size, hidden_size] so gated [seq_len, intermediate_size]
        // matmul gives [seq_len, hidden_size].
        gated.matmul(&self.down_proj.transpose_2d())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Test FFN forward pass with small dimensions (hidden=4, intermediate=8).
    ///
    /// This creates an FFN with:
    /// - gate_proj: [8, 4]
    /// - up_proj: [8, 4]
    /// - down_proj: [4, 8]
    ///
    /// Input shape: [2, 4] (seq_len=2, hidden_size=4)
    /// Expected output shape: [2, 4] (same as input)
    #[test]
    fn test_ffn_forward_small_dimensions() {
        let hidden = 4;
        let intermediate = 8;

        // Create weights filled with small values so the output is
        // numerically reasonable. Using 0.1 everywhere.
        let gate_proj = Tensor::new(
            vec![intermediate, hidden],
            vec![0.1; intermediate * hidden],
        );
        let up_proj = Tensor::new(
            vec![intermediate, hidden],
            vec![0.1; intermediate * hidden],
        );
        let down_proj = Tensor::new(
            vec![hidden, intermediate],
            vec![0.1; hidden * intermediate],
        );

        let ffn = FeedForward::new(gate_proj, up_proj, down_proj);

        // Input: 2 tokens, each with 4 features.
        let x = Tensor::new(vec![2, hidden], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let output = ffn.forward(&x);

        // Output shape must match input shape.
        assert_eq!(
            output.shape(),
            &[2, hidden],
            "output shape should be [2, 4], got {:?}",
            output.shape(),
        );
    }

    /// Test that the output shape matches the input shape for various
    /// seq_len values.
    #[test]
    fn test_ffn_output_shape_matches_input() {
        let hidden = 4;
        let intermediate = 8;

        let gate_proj = Tensor::new(
            vec![intermediate, hidden],
            vec![0.1; intermediate * hidden],
        );
        let up_proj = Tensor::new(
            vec![intermediate, hidden],
            vec![0.1; intermediate * hidden],
        );
        let down_proj = Tensor::new(
            vec![hidden, intermediate],
            vec![0.1; hidden * intermediate],
        );

        let ffn = FeedForward::new(gate_proj, up_proj, down_proj);

        // Test with seq_len = 1
        let x1 = Tensor::new(vec![1, hidden], vec![1.0, 2.0, 3.0, 4.0]);
        let out1 = ffn.forward(&x1);
        assert_eq!(out1.shape(), &[1, hidden]);

        // Test with seq_len = 3
        let x3 = Tensor::new(vec![3, hidden], vec![1.0; 3 * hidden]);
        let out3 = ffn.forward(&x3);
        assert_eq!(out3.shape(), &[3, hidden]);

        // Test with seq_len = 10
        let x10 = Tensor::new(vec![10, hidden], vec![1.0; 10 * hidden]);
        let out10 = ffn.forward(&x10);
        assert_eq!(out10.shape(), &[10, hidden]);
    }

    /// Test SiLU gate behavior: the gate path produces non-negative values
    /// for positive inputs.
    ///
    /// When the gate_proj weight is positive and the input is positive,
    /// the pre-activation values are positive. SiLU of a positive number
    /// is also positive (SiLU(x) > 0 when x > 0). Therefore the gate
    /// should be all non-negative for a positive input.
    #[test]
    fn test_silu_gate_nonnegative_for_positive_input() {
        let hidden = 4;
        let intermediate = 8;

        // Use all-positive weights.
        let gate_proj = Tensor::new(
            vec![intermediate, hidden],
            vec![0.5; intermediate * hidden],
        );
        let up_proj = Tensor::new(
            vec![intermediate, hidden],
            vec![0.3; intermediate * hidden],
        );
        let down_proj = Tensor::new(
            vec![hidden, intermediate],
            vec![0.2; hidden * intermediate],
        );

        let ffn = FeedForward::new(gate_proj, up_proj, down_proj);

        // All-positive input.
        let x = Tensor::new(vec![1, hidden], vec![1.0, 2.0, 3.0, 4.0]);

        // Compute the gate manually: x · gate_proj^T, then SiLU.
        let gate_pre = x.matmul(&ffn.gate_proj.transpose_2d());
        let gate = gate_pre.silu();

        // For positive pre-activation values, SiLU should be positive.
        // With all-positive input and weights, every pre-activation
        // value is positive, so every SiLU value should be positive.
        for (i, &v) in gate.data().iter().enumerate() {
            assert!(
                v >= 0.0,
                "gate element {} should be non-negative for positive input, got {}",
                i,
                v,
            );
        }
    }

    /// Test that different inputs produce different outputs.
    ///
    /// If the FFN is working correctly, changing the input should change
    /// the output. This is a basic sanity check that the computation
    /// isn't collapsing to a constant regardless of input.
    #[test]
    fn test_different_inputs_different_outputs() {
        let hidden = 4;
        let intermediate = 8;

        // Use non-trivial weights (not all the same value).
        let gate_proj_data: Vec<f32> = (0..intermediate * hidden)
            .map(|i| i as f32 * 0.1 - 0.4)
            .collect();
        let up_proj_data: Vec<f32> = (0..intermediate * hidden)
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let down_proj_data: Vec<f32> = (0..hidden * intermediate)
            .map(|i| i as f32 * 0.05 - 0.2)
            .collect();

        let gate_proj = Tensor::new(vec![intermediate, hidden], gate_proj_data);
        let up_proj = Tensor::new(vec![intermediate, hidden], up_proj_data);
        let down_proj = Tensor::new(vec![hidden, intermediate], down_proj_data);

        let ffn = FeedForward::new(gate_proj, up_proj, down_proj);

        let x1 = Tensor::new(vec![1, hidden], vec![1.0, 0.0, -1.0, 0.5]);
        let x2 = Tensor::new(vec![1, hidden], vec![0.0, 1.0, 0.5, -1.0]);

        let out1 = ffn.forward(&x1);
        let out2 = ffn.forward(&x2);

        // Outputs should differ.
        let any_different = out1
            .data()
            .iter()
            .zip(out2.data().iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);

        assert!(
            any_different,
            "different inputs should produce different outputs"
        );
    }

    /// Test the full SwiGLU computation with hand-computed values.
    ///
    /// Using very small dimensions (hidden=2, intermediate=3) so we can
    /// verify the arithmetic by hand.
    ///
    /// gate_proj = [[1, 0], [0, 1], [1, 1]]   shape [3, 2]
    /// up_proj   = [[1, 0], [0, 1], [1, 1]]   shape [3, 2]
    /// down_proj = [[1, 0, 0], [0, 1, 0]]     shape [2, 3]
    ///
    /// x = [[1, 2]]  shape [1, 2]
    ///
    /// Step 1: gate_pre = x · gate_proj^T
    ///   gate_proj^T = [[1, 0, 1], [0, 1, 1]]
    ///   gate_pre = [[1*1+2*0, 1*0+2*1, 1*1+2*1]] = [[1, 2, 3]]
    ///
    /// Step 2: gate = SiLU(gate_pre)
    ///   SiLU(1) = 1 * sigmoid(1) ≈ 0.7311
    ///   SiLU(2) = 2 * sigmoid(2) ≈ 2 * 0.8808 ≈ 1.7616
    ///   SiLU(3) = 3 * sigmoid(3) ≈ 3 * 0.9526 ≈ 2.8578
    ///   gate ≈ [[0.7311, 1.7616, 2.8578]]
    ///
    /// Step 3: up = x · up_proj^T
    ///   up = [[1, 2, 3]]  (same as gate_pre since same weights)
    ///
    /// Step 4: gated = gate ⊙ up
    ///   gated ≈ [[0.7311, 3.5232, 8.5733]]
    ///
    /// Step 5: output = gated · down_proj^T
    ///   down_proj^T = [[1, 0], [0, 1], [0, 0]]
    ///   output = [[0.7311*1 + 3.5232*0 + 8.5733*0,
    ///              0.7311*0 + 3.5232*1 + 8.5733*0]]
    ///          = [[0.7311, 3.5232]]
    #[test]
    fn test_ffn_hand_computed() {
        let gate_proj = Tensor::new(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let up_proj = Tensor::new(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let down_proj = Tensor::new(vec![2, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

        let ffn = FeedForward::new(gate_proj, up_proj, down_proj);

        let x = Tensor::new(vec![1, 2], vec![1.0, 2.0]);
        let output = ffn.forward(&x);

        assert_eq!(output.shape(), &[1, 2]);

        // Expected values (from hand computation above):
        // output[0] ≈ SiLU(1) ≈ 0.7311
        // output[1] ≈ SiLU(2) * 2 ≈ 1.7616 * 2 ≈ 3.5232
        // Wait — let me recalculate. gate ⊙ up:
        //   SiLU(1) * 1 = 0.7311
        //   SiLU(2) * 2 = 1.7616 * 2 = 3.5232
        //   SiLU(3) * 3 = 2.8578 * 3 = 8.5733
        // Then down_proj^T = [[1,0],[0,1],[0,0]], so:
        //   output[0] = 0.7311
        //   output[1] = 3.5232
        assert!(
            (output.data()[0] - 0.7311).abs() < 0.01,
            "output[0] expected ~0.7311, got {}",
            output.data()[0],
        );
        assert!(
            (output.data()[1] - 3.5232).abs() < 0.01,
            "output[1] expected ~3.5232, got {}",
            output.data()[1],
        );
    }

    /// Test that FFN constructor validates weight shapes.
    ///
    /// If gate_proj and up_proj have different shapes, the constructor
    /// should panic.
    #[test]
    #[should_panic(expected = "must match up_proj shape")]
    fn test_ffn_mismatched_gate_up_shapes() {
        let gate_proj = Tensor::new(vec![8, 4], vec![0.1; 32]);
        let up_proj = Tensor::new(vec![6, 4], vec![0.1; 24]); // different intermediate
        let down_proj = Tensor::new(vec![4, 8], vec![0.1; 32]);

        FeedForward::new(gate_proj, up_proj, down_proj);
    }

    /// Test that FFN constructor validates down_proj shape.
    ///
    /// If down_proj's dimensions don't match the reverse of gate_proj's,
    /// the constructor should panic.
    #[test]
    #[should_panic(expected = "must equal gate_proj cols")]
    fn test_ffn_invalid_down_proj_rows() {
        let gate_proj = Tensor::new(vec![8, 4], vec![0.1; 32]);
        let up_proj = Tensor::new(vec![8, 4], vec![0.1; 32]);
        let down_proj = Tensor::new(vec![3, 8], vec![0.1; 24]); // rows should be 4

        FeedForward::new(gate_proj, up_proj, down_proj);
    }

    /// Test that the FFN processes each token independently.
    ///
    /// If we feed two identical rows, the output should have two
    /// identical rows. This confirms there is no cross-token
    /// interaction in the FFN.
    #[test]
    fn test_ffn_no_cross_token_interaction() {
        let hidden = 4;
        let intermediate = 8;

        let gate_proj_data: Vec<f32> = (0..intermediate * hidden)
            .map(|i| i as f32 * 0.1 - 0.4)
            .collect();
        let up_proj_data: Vec<f32> = (0..intermediate * hidden)
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let down_proj_data: Vec<f32> = (0..hidden * intermediate)
            .map(|i| i as f32 * 0.05 - 0.2)
            .collect();

        let gate_proj = Tensor::new(vec![intermediate, hidden], gate_proj_data);
        let up_proj = Tensor::new(vec![intermediate, hidden], up_proj_data);
        let down_proj = Tensor::new(vec![hidden, intermediate], down_proj_data);

        let ffn = FeedForward::new(gate_proj, up_proj, down_proj);

        // Two identical tokens.
        let token = vec![1.0, -0.5, 0.3, 2.0];
        let x = Tensor::new(vec![2, hidden], [&token[..], &token[..]].concat().to_vec());
        let output = ffn.forward(&x);

        // Both rows of the output should be identical.
        let row0: Vec<f32> = output.data()[..hidden].to_vec();
        let row1: Vec<f32> = output.data()[hidden..].to_vec();
        for (i, (a, b)) in row0.iter().zip(row1.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "identical inputs should produce identical outputs: row0[{}] = {}, row1[{}] = {}",
                i, a, i, b,
            );
        }
    }
}
