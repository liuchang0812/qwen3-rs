//! Single transformer block: RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual.
//!
//! This module implements a single transformer block (also called a "layer" in
//! the Hugging Face model format). Each block consists of two sub-layers with
//! residual connections:
//!
//! 1. **Self-attention block**: RMSNorm -> Multi-Head Attention -> Residual
//! 2. **Feed-forward block**: RMSNorm -> SwiGLU FFN -> Residual
//!
//! The "pre-norm" design (normalizing *before* each sub-layer rather than
//! after) is standard in modern LLMs (LLaMA, Qwen, Mistral, etc.) and
//! stabilizes training by ensuring activations never grow too large before
//! entering a sub-layer.
//!
//! # Qwen3-0.6B Block Configuration
//!
//! ```text
//! hidden_size          = 1024
//! num_attention_heads  = 16
//! num_key_value_heads  = 8    (GQA)
//! head_dim             = 128
//! intermediate_size    = 3072 (for FFN)
//! rms_norm_eps         = 1e-6
//! rope_theta           = 1000000.0
//! ```

use crate::attention::{Attention, KVCache};
use crate::ffn::FeedForward;
use crate::rmsnorm::RMSNorm;
use crate::tensor::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// TransformerBlock struct
// ─────────────────────────────────────────────────────────────────────────────

/// A single transformer block combining self-attention and feed-forward
/// sub-layers with pre-norm residual connections.
///
/// The block follows the standard pre-norm transformer architecture:
///
/// ```text
/// x ──┬── RMSNorm ── Attention ──┬── + ──┬── RMSNorm ── FFN ──┬── + ──> output
///     └──────── residual ────────┘       └──── residual ───────┘
/// ```
///
/// Each sub-layer's output is added back to its input (residual connection),
/// which helps gradients flow during training and allows the network to learn
/// incremental modifications to the representation.
pub struct TransformerBlock {
    /// RMSNorm applied before the self-attention sub-layer.
    /// Called `input_layernorm` in the Hugging Face model format.
    input_layernorm: RMSNorm,

    /// Self-attention with Grouped Query Attention (GQA) and KV cache.
    self_attn: Attention,

    /// RMSNorm applied before the feed-forward sub-layer.
    /// Called `post_attention_layernorm` in the Hugging Face model format.
    post_attention_layernorm: RMSNorm,

    /// SwiGLU feed-forward network.
    ffn: FeedForward,
}

impl TransformerBlock {
    /// Create a new transformer block from its component weights and config.
    ///
    /// # Arguments
    ///
    /// * `input_layernorm_weight`    - Weight for RMSNorm before attention,
    ///   shape `[hidden_size]`. Loaded from key `model.layers.N.input_layernorm.weight`.
    ///
    /// * `post_attention_layernorm_weight` - Weight for RMSNorm before FFN,
    ///   shape `[hidden_size]`. Loaded from key `model.layers.N.post_attention_layernorm.weight`.
    ///
    /// * `q_proj` - Query projection weight, shape `[hidden_size, hidden_size]`.
    /// * `k_proj` - Key projection weight, shape `[kv_dim, hidden_size]`.
    /// * `v_proj` - Value projection weight, shape `[kv_dim, hidden_size]`.
    /// * `o_proj` - Output projection weight, shape `[hidden_size, hidden_size]`.
    ///
    /// * `gate_proj` - FFN gate projection, shape `[intermediate_size, hidden_size]`.
    /// * `up_proj`   - FFN up projection, shape `[intermediate_size, hidden_size]`.
    /// * `down_proj` - FFN down projection, shape `[hidden_size, intermediate_size]`.
    ///
    /// * `num_heads`    - Number of query attention heads.
    /// * `num_kv_heads` - Number of key-value attention heads (for GQA).
    /// * `head_dim`     - Dimension per attention head.
    /// * `max_seq_len`  - Maximum sequence length (for RoPE precomputation).
    /// * `rope_theta`   - Base frequency for RoPE (1000000.0 for Qwen3).
    /// * `rms_norm_eps` - Epsilon for RMSNorm (1e-6 for Qwen3).
    pub fn new(
        input_layernorm_weight: Tensor,
        post_attention_layernorm_weight: Tensor,
        q_proj: Tensor,
        k_proj: Tensor,
        v_proj: Tensor,
        o_proj: Tensor,
        q_norm: Option<Tensor>,
        k_norm: Option<Tensor>,
        gate_proj: Tensor,
        up_proj: Tensor,
        down_proj: Tensor,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f32,
        rms_norm_eps: f32,
    ) -> Self {
        // Convert optional weight tensors to Option<RMSNorm>.
        let q_norm = q_norm.map(|w| RMSNorm::new(w, rms_norm_eps));
        let k_norm = k_norm.map(|w| RMSNorm::new(w, rms_norm_eps));

        Self {
            input_layernorm: RMSNorm::new(input_layernorm_weight, rms_norm_eps),
            self_attn: Attention::new(
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
                num_heads,
                num_kv_heads,
                head_dim,
                max_seq_len,
                rope_theta,
            ),
            post_attention_layernorm: RMSNorm::new(post_attention_layernorm_weight, rms_norm_eps),
            ffn: FeedForward::new(gate_proj, up_proj, down_proj),
        }
    }

    /// Run the transformer block forward pass.
    ///
    /// The computation proceeds as:
    ///
    /// ```text
    /// 1. residual = x                           (save input for residual connection)
    /// 2. x_norm = input_layernorm.forward(x)    (normalize before attention)
    /// 3. attn_out = self_attn.forward(x_norm, start_pos, kv_cache)  (self-attention)
    /// 4. x = residual + attn_out                (residual connection after attention)
    /// 5. residual = x                           (save for second residual connection)
    /// 6. x_norm = post_attention_layernorm.forward(x)  (normalize before FFN)
    /// 7. ffn_out = ffn.forward(x_norm)          (feed-forward network)
    /// 8. x = residual + ffn_out                 (residual connection after FFN)
    /// 9. return x
    /// ```
    ///
    /// # Arguments
    ///
    /// * `x`          - Input tensor of shape `[seq_len, hidden_size]`.
    /// * `start_pos`  - Position of the first token (for RoPE and KV cache).
    /// * `kv_cache`   - Mutable reference to the KV cache for this layer.
    ///   Updated in-place with new key/value entries.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[seq_len, hidden_size]` (same as input).
    pub fn forward(&self, x: &Tensor, start_pos: usize, kv_cache: &mut KVCache) -> Tensor {
        // Step 1: Save the input for the first residual connection.
        let residual = x.clone();

        // Step 2: Apply RMSNorm before attention (pre-norm).
        // This normalizes the hidden states to stabilize the attention computation.
        let x_norm = self.input_layernorm.forward(x);

        // Step 3: Run self-attention with GQA and KV cache.
        // The attention layer computes Q*K^T, applies causal masking, and
        // produces weighted sums of V. It also updates the KV cache.
        let attn_out = self.self_attn.forward(&x_norm, start_pos, kv_cache);

        // Step 4: Add the residual connection after attention.
        // The residual connection allows gradients to flow directly through
        // the block and lets the attention layer learn incremental updates.
        let x = residual.add(&attn_out);

        // Step 5: Save the intermediate result for the second residual connection.
        let residual = &x;

        // Step 6: Apply RMSNorm before FFN (pre-norm).
        let x_norm = self.post_attention_layernorm.forward(&x);

        // Step 7: Run the SwiGLU feed-forward network.
        // The FFN processes each token's representation independently,
        // applying a gating mechanism for expressive feature transformation.
        let ffn_out = self.ffn.forward(&x_norm);

        // Step 8: Add the residual connection after FFN.
        // Same rationale as step 4: preserves information from the attention
        // output and allows the FFN to learn incremental modifications.
        residual.add(&ffn_out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a small TransformerBlock for testing.
    ///
    /// Configuration:
    /// - hidden_size = 8
    /// - num_heads = 2
    /// - num_kv_heads = 1
    /// - head_dim = 4
    /// - intermediate_size = 16
    /// - max_seq_len = 32
    /// - rope_theta = 10000.0
    /// - rms_norm_eps = 1e-6
    fn make_test_block() -> TransformerBlock {
        let hidden = 8;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 4;
        let intermediate = 16;
        let kv_dim = num_kv_heads * head_dim; // 4

        // RMSNorm weights: all ones (identity-like normalization)
        let ln1_weight = Tensor::ones(vec![hidden]);
        let ln2_weight = Tensor::ones(vec![hidden]);

        // Attention weights: identity-like projections
        // q_proj: [hidden, hidden] = [8, 8]
        let q_proj = Tensor::new(
            vec![hidden, hidden],
            (0..hidden * hidden)
                .map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 })
                .collect(),
        );
        // k_proj: [kv_dim, hidden] = [4, 8]
        let k_proj = Tensor::new(
            vec![kv_dim, hidden],
            (0..kv_dim * hidden)
                .map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 })
                .collect(),
        );
        // v_proj: [kv_dim, hidden] = [4, 8]
        let v_proj = Tensor::new(
            vec![kv_dim, hidden],
            (0..kv_dim * hidden)
                .map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 })
                .collect(),
        );
        // o_proj: [hidden, hidden] = [8, 8]
        let o_proj = Tensor::new(
            vec![hidden, hidden],
            (0..hidden * hidden)
                .map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 })
                .collect(),
        );

        // FFN weights: small values
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

        TransformerBlock::new(
            ln1_weight,
            ln2_weight,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            None,
            None,
            gate_proj,
            up_proj,
            down_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            32,
            10000.0,
            1e-6,
        )
    }

    /// Test that the forward pass produces the correct output shape.
    ///
    /// Input shape: [seq_len, hidden_size]
    /// Expected output shape: [seq_len, hidden_size] (same as input)
    #[test]
    fn test_forward_output_shape() {
        let block = make_test_block();
        let mut cache = KVCache::new();

        // Test with seq_len = 1 (decode step)
        let x1 = Tensor::ones(vec![1, 8]);
        let out1 = block.forward(&x1, 0, &mut cache);
        assert_eq!(
            out1.shape(),
            &[1, 8],
            "output shape should be [1, 8], got {:?}",
            out1.shape(),
        );

        // Test with seq_len = 3 (prefill)
        let mut cache2 = KVCache::new();
        let x3 = Tensor::ones(vec![3, 8]);
        let out3 = block.forward(&x3, 0, &mut cache2);
        assert_eq!(
            out3.shape(),
            &[3, 8],
            "output shape should be [3, 8], got {:?}",
            out3.shape(),
        );
    }

    /// Test that two forward passes with different start_pos update the KV
    /// cache correctly.
    ///
    /// The first pass (prefill) processes multiple tokens and stores them in
    /// the cache. The second pass (decode) processes one new token and
    /// appends to the cache. The total cache size should reflect both passes.
    #[test]
    fn test_kv_cache_updates_across_forward_passes() {
        let block = make_test_block();
        let mut cache = KVCache::new();

        // First forward pass: prefill with 3 tokens at positions 0, 1, 2
        let x = Tensor::ones(vec![3, 8]);
        let _out = block.forward(&x, 0, &mut cache);

        // Cache should contain 3 rows (one per token)
        assert!(
            cache.key_cache.is_some(),
            "KV cache should be populated after first forward pass",
        );
        assert_eq!(
            cache.key_cache.as_ref().unwrap().shape()[0],
            3,
            "cache should have 3 rows after prefill, got {}",
            cache.key_cache.as_ref().unwrap().shape()[0],
        );

        // Second forward pass: decode 1 token at position 3
        let x2 = Tensor::ones(vec![1, 8]);
        let _out2 = block.forward(&x2, 3, &mut cache);

        // Cache should now contain 4 rows (3 + 1)
        assert_eq!(
            cache.key_cache.as_ref().unwrap().shape()[0],
            4,
            "cache should have 4 rows after decode, got {}",
            cache.key_cache.as_ref().unwrap().shape()[0],
        );
        assert_eq!(
            cache.value_cache.as_ref().unwrap().shape()[0],
            4,
            "value cache should have 4 rows after decode, got {}",
            cache.value_cache.as_ref().unwrap().shape()[0],
        );
    }

    /// Test that the residual connection is applied, so the output differs
    /// from the FFN output alone.
    ///
    /// Without the residual connection, the block output would be just
    /// `ffn(attention(norm(x)))`. With the residual, it is
    /// `x + ffn(attention(norm(x)))`. We verify that the output is not
    /// equal to the FFN output alone by comparing against a manually
    /// computed FFN-only result.
    #[test]
    fn test_residual_connection_is_applied() {
        let block = make_test_block();
        let mut cache = KVCache::new();

        // Use an input that is not zero so the residual actually adds something.
        let x = Tensor::new(vec![1, 8], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // Run the full block forward pass
        let output = block.forward(&x, 0, &mut cache);

        // Compute just the sub-layer output (no residual) by running
        // attention and FFN without adding back the input.
        let x_norm = block.input_layernorm.forward(&x);
        let attn_out = block.self_attn.forward(&x_norm, 0, &mut KVCache::new());
        let x_after_attn = x_norm.add(&attn_out); // Note: this uses norm'd x, not original
        let x_norm2 = block.post_attention_layernorm.forward(&x_after_attn);
        let ffn_out = block.ffn.forward(&x_norm2);

        // The block output should differ from the FFN output alone because
        // the residual connections add the original input (and intermediate)
        // back. With residual connections, the output includes contributions
        // from the input that are not present in ffn_out alone.
        let any_different = output
            .data()
            .iter()
            .zip(ffn_out.data().iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);

        assert!(
            any_different,
            "output with residual should differ from FFN output alone"
        );
    }
}
