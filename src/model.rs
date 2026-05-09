//! Full transformer model: embedding -> N x TransformerBlock -> RMSNorm -> lm_head.
//!
//! This module implements the complete Qwen3 model as a single struct that
//! owns all parameters and orchestrates the forward pass. The architecture is:
//!
//! ```text
//! token_ids ──> Embedding lookup ──> N x TransformerBlock ──> RMSNorm ──> lm_head ──> logits
//! ```
//!
//! # Qwen3-0.6B Architecture
//!
//! ```text
//! vocab_size            = 151936
//! hidden_size           = 1024
//! num_hidden_layers     = 28
//! num_attention_heads   = 16
//! num_key_value_heads   = 8
//! intermediate_size     = 3072
//! max_position_embeddings = 40960
//! rms_norm_eps          = 1e-6
//! rope_theta            = 1000000.0
//! head_dim              = 128
//! ```
//!
//! # Forward Pass
//!
//! 1. **Embedding lookup**: Convert token IDs to dense vectors by looking up
//!    rows in the embedding table.
//! 2. **Transformer blocks**: Pass through N transformer blocks, each with
//!    self-attention and feed-forward sub-layers with residual connections.
//! 3. **Final RMSNorm**: Normalize the output of the last block.
//! 4. **lm_head projection**: Project from hidden_size to vocab_size to
//!    produce logits for each token position.

use crate::attention::KVCache;
use crate::config::ModelConfig;
use crate::rmsnorm::RMSNorm;
use crate::safetensors::read_safetensors_as_tensors;
use crate::tensor::Tensor;
use crate::transformer_block::TransformerBlock;

use std::collections::HashMap;
use std::path::Path;

// ─────────────────────────────────────────────────────────────────────────────
// QwenModel struct
// ─────────────────────────────────────────────────────────────────────────────

/// The complete Qwen3 transformer model for next-token prediction.
///
/// The model takes a sequence of token IDs, embeds them, processes them
/// through N transformer blocks, and produces logits over the vocabulary
/// for each position. During generation, these logits are used to sample
/// the next token.
///
/// # Ownership
///
/// The model owns all its parameters (embedding table, transformer block
/// weights, final norm, and lm_head). The KV caches are also owned by the
/// model so they persist across forward passes during autoregressive
/// generation.
pub struct QwenModel {
    /// Token embedding table of shape `[vocab_size, hidden_size]`.
    /// Each row is the dense vector representation of one token.
    /// During the forward pass, we look up the row corresponding to each
    /// input token ID.
    embed_tokens: Tensor,

    /// The N transformer blocks that process the hidden states.
    /// Each block contains self-attention (with GQA and KV cache) and
    /// a SwiGLU feed-forward network, both with pre-norm residual
    /// connections.
    layers: Vec<TransformerBlock>,

    /// Final RMSNorm applied after the last transformer block.
    /// This normalizes the hidden states before the lm_head projection.
    norm: RMSNorm,

    /// Output projection (language model head) of shape
    /// `[vocab_size, hidden_size]`. Projects from hidden_size back to
    /// vocab_size to produce logits for each token in the vocabulary.
    /// In Qwen3, this weight is tied to the embedding table when
    /// `tie_word_embeddings` is `true`.
    lm_head: Tensor,

    /// Model configuration (hyperparameters).
    config: ModelConfig,

    /// One KV cache per transformer layer. Each cache stores the key and
    /// value tensors from previous forward passes so that during
    /// autoregressive generation, we only need to compute K/V for the
    /// new token and append to the cache.
    kv_caches: Vec<KVCache>,
}

impl QwenModel {
    /// Create a new model from pre-built components.
    ///
    /// This constructor takes individual components that have already been
    /// assembled (embedding table, transformer blocks, final norm, lm_head).
    /// The KV caches are initialized as empty — one per layer.
    ///
    /// Later, we will add a `load_from_safetensors()` method that
    /// constructs the model directly from a safetensors weight file.
    ///
    /// # Arguments
    ///
    /// * `embed_tokens` - Embedding table of shape `[vocab_size, hidden_size]`.
    /// * `layers`       - Vector of transformer blocks (one per layer).
    /// * `norm`         - Final RMSNorm layer.
    /// * `lm_head`      - Output projection of shape `[vocab_size, hidden_size]`.
    /// * `config`       - Model configuration.
    ///
    /// # Panics
    ///
    /// Panics if the number of layers does not match `config.num_hidden_layers`.
    pub fn new(
        embed_tokens: Tensor,
        layers: Vec<TransformerBlock>,
        norm: RMSNorm,
        lm_head: Tensor,
        config: ModelConfig,
    ) -> Self {
        assert_eq!(
            layers.len(),
            config.num_hidden_layers,
            "QwenModel::new: number of layers ({}) must match config.num_hidden_layers ({})",
            layers.len(),
            config.num_hidden_layers,
        );

        // Create one empty KV cache per layer.
        let kv_caches = (0..config.num_hidden_layers)
            .map(|_| KVCache::new())
            .collect();

        Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config,
            kv_caches,
        }
    }

    /// Load a QwenModel from a model directory.
    ///
    /// The directory should contain:
    /// - `config.json`: model hyperparameters
    /// - `model.safetensors`: model weights (or `model-00001-of-0000X.safetensors` for sharded)
    ///
    /// # Weight Name Mapping
    ///
    /// The safetensors file contains tensors named according to the HuggingFace
    /// convention. This method maps them to the corresponding model components:
    ///
    /// - `model.embed_tokens.weight` → embedding table
    /// - `model.layers.{i}.input_layernorm.weight` → pre-attention RMSNorm
    /// - `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` → attention projections
    /// - `model.layers.{i}.post_attention_layernorm.weight` → pre-FFN RMSNorm
    /// - `model.layers.{i}.mlp.{gate,up,down}_proj.weight` → FFN projections
    /// - `model.norm.weight` → final RMSNorm
    /// - `lm_head.weight` → output projection (optional if `tie_word_embeddings` is true)
    ///
    /// # Errors
    ///
    /// Returns a descriptive error if:
    /// - `config.json` cannot be read or parsed
    /// - `model.safetensors` cannot be read
    /// - A required tensor is missing from the safetensors file
    /// - A tensor's shape does not match the expected dimensions
    pub fn load(model_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        // Step 1: Load config.
        let config = ModelConfig::from_file(&model_dir.join("config.json"))?;

        // Step 2: Load weights.
        let weights = read_safetensors_as_tensors(&model_dir.join("model.safetensors"))?;

        // Helper: extract a tensor by name, returning a descriptive error if missing.
        let get_tensor = |weights: &HashMap<String, Tensor>, name: &str| -> Result<Tensor, Box<dyn std::error::Error>> {
            weights.get(name).cloned().ok_or_else(|| {
                format!("Missing weight: {}", name).into()
            })
        };

        // Helper: validate that a tensor's shape matches the expected shape.
        let check_shape = |tensor: &Tensor, expected: &[usize], name: &str| -> Result<(), Box<dyn std::error::Error>> {
            if tensor.shape() != expected {
                Err(format!(
                    "Shape mismatch for {}: expected {:?}, got {:?}",
                    name, expected, tensor.shape()
                ).into())
            } else {
                Ok(())
            }
        };

        let head_dim = config.head_dim();
        let kv_dim = config.num_key_value_heads * head_dim;
        let q_dim = config.num_attention_heads * head_dim;

        // Step 3: Extract embed_tokens.
        let embed_tokens = get_tensor(&weights, "model.embed_tokens.weight")?;
        check_shape(&embed_tokens, &[config.vocab_size, config.hidden_size], "model.embed_tokens.weight")?;

        // Step 4: Build transformer blocks.
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            // Input layernorm.
            let ln1_name = format!("model.layers.{}.input_layernorm.weight", i);
            let ln1_weight = get_tensor(&weights, &ln1_name)?;
            check_shape(&ln1_weight, &[config.hidden_size], &ln1_name)?;

            // Attention projections.
            let q_name = format!("model.layers.{}.self_attn.q_proj.weight", i);
            let q_proj = get_tensor(&weights, &q_name)?;
            check_shape(&q_proj, &[q_dim, config.hidden_size], &q_name)?;

            let k_name = format!("model.layers.{}.self_attn.k_proj.weight", i);
            let k_proj = get_tensor(&weights, &k_name)?;
            check_shape(&k_proj, &[kv_dim, config.hidden_size], &k_name)?;

            let v_name = format!("model.layers.{}.self_attn.v_proj.weight", i);
            let v_proj = get_tensor(&weights, &v_name)?;
            check_shape(&v_proj, &[kv_dim, config.hidden_size], &v_name)?;

            let o_name = format!("model.layers.{}.self_attn.o_proj.weight", i);
            let o_proj = get_tensor(&weights, &o_name)?;
            check_shape(&o_proj, &[config.hidden_size, q_dim], &o_name)?;

            // Optional Q/K normalization (present in Qwen3 models).
            // q_norm.weight: shape [head_dim] — RMSNorm on Q after projection, per head
            // k_norm.weight: shape [head_dim] — RMSNorm on K after projection, per head
            let q_norm_weight = weights.get(&format!("model.layers.{}.self_attn.q_norm.weight", i))
                .map(|t| t.clone());
            if let Some(ref w) = q_norm_weight {
                check_shape(w, &[head_dim], &format!("model.layers.{}.self_attn.q_norm.weight", i))?;
            }

            let k_norm_weight = weights.get(&format!("model.layers.{}.self_attn.k_norm.weight", i))
                .map(|t| t.clone());
            if let Some(ref w) = k_norm_weight {
                check_shape(w, &[head_dim], &format!("model.layers.{}.self_attn.k_norm.weight", i))?;
            }

            // Post-attention layernorm.
            let ln2_name = format!("model.layers.{}.post_attention_layernorm.weight", i);
            let ln2_weight = get_tensor(&weights, &ln2_name)?;
            check_shape(&ln2_weight, &[config.hidden_size], &ln2_name)?;

            // FFN projections.
            let gate_name = format!("model.layers.{}.mlp.gate_proj.weight", i);
            let gate_proj = get_tensor(&weights, &gate_name)?;
            check_shape(&gate_proj, &[config.intermediate_size, config.hidden_size], &gate_name)?;

            let up_name = format!("model.layers.{}.mlp.up_proj.weight", i);
            let up_proj = get_tensor(&weights, &up_name)?;
            check_shape(&up_proj, &[config.intermediate_size, config.hidden_size], &up_name)?;

            let down_name = format!("model.layers.{}.mlp.down_proj.weight", i);
            let down_proj = get_tensor(&weights, &down_name)?;
            check_shape(&down_proj, &[config.hidden_size, config.intermediate_size], &down_name)?;

            let block = TransformerBlock::new(
                ln1_weight,
                ln2_weight,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm_weight,
                k_norm_weight,
                gate_proj,
                up_proj,
                down_proj,
                config.num_attention_heads,
                config.num_key_value_heads,
                head_dim,
                config.max_position_embeddings,
                config.rope_theta_f32(),
                config.eps_f32(),
            );

            layers.push(block);
        }

        // Step 5: Extract final RMSNorm.
        let norm_weight = get_tensor(&weights, "model.norm.weight")?;
        check_shape(&norm_weight, &[config.hidden_size], "model.norm.weight")?;
        let norm = RMSNorm::new(norm_weight, config.eps_f32());

        // Step 6: Extract lm_head (use embed_tokens if tied).
        let lm_head = if config.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            get_tensor(&weights, "lm_head.weight")?
        };
        check_shape(&lm_head, &[config.vocab_size, config.hidden_size], "lm_head.weight")?;

        // Step 7: Assemble the model.
        Ok(QwenModel::new(embed_tokens, layers, norm, lm_head, config))
    }

    /// Run a forward pass through the entire model.
    ///
    /// # The computation, step by step
    ///
    /// 1. **Embedding lookup**: For each token ID, extract the corresponding
    ///    row from the embedding table. This converts discrete token IDs
    ///    into continuous dense vectors.
    ///
    ///    - Input: `token_ids` of length `seq_len`
    ///    - Output: tensor of shape `[seq_len, hidden_size]`
    ///
    /// 2. **Transformer blocks**: Pass the embeddings through each
    ///    transformer block sequentially. Each block applies self-attention
    ///    (with GQA and KV cache) and a feed-forward network with residual
    ///    connections.
    ///
    /// 3. **Final RMSNorm**: Normalize the output of the last block.
    ///
    /// 4. **Logit projection**: Multiply by `lm_head^T` to project from
    ///    `hidden_size` to `vocab_size`, producing raw logits for each
    ///    vocabulary token at each position.
    ///
    /// # Arguments
    ///
    /// * `token_ids`  - Slice of token IDs to process. Length = `seq_len`.
    /// * `start_pos`  - Position of the first token (for KV cache and RoPE).
    ///   During prefill this is 0; during decode it is the number of tokens
    ///   already processed.
    ///
    /// # Returns
    ///
    /// Logits tensor of shape `[seq_len, vocab_size]`.
    pub fn forward(&mut self, token_ids: &[usize], start_pos: usize) -> Tensor {
        let seq_len = token_ids.len();
        assert!(seq_len > 0, "QwenModel::forward: token_ids must not be empty");

        // ── Step 1: Embedding lookup ─────────────────────────────────────
        //
        // For each token ID, extract the corresponding row from the
        // embedding table. The embedding table has shape
        // [vocab_size, hidden_size], so each row is a vector of length
        // hidden_size.
        //
        // If there is a single token, we use row() directly. For multiple
        // tokens, we extract each row and stack them into a 2-D tensor.

        let x = if seq_len == 1 {
            // Single token: extract one row, reshape to [1, hidden_size].
            let row = self.embed_tokens.row(token_ids[0]);
            row.reshape(vec![1, self.config.hidden_size])
        } else {
            // Multiple tokens: extract each row and stack them.
            let first_row = self.embed_tokens.row(token_ids[0]);
            let mut result = first_row.reshape(vec![1, self.config.hidden_size]);
            for &tid in &token_ids[1..] {
                let row = self.embed_tokens.row(tid);
                let row_2d = row.reshape(vec![1, self.config.hidden_size]);
                result = result.stack_rows(&row_2d);
            }
            result
        };

        // x now has shape [seq_len, hidden_size].

        // ── Step 2: Pass through transformer blocks ─────────────────────
        //
        // Each block applies: RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual
        // The KV cache for each layer is updated in place.

        let mut hidden = x;
        for (layer_idx, block) in self.layers.iter().enumerate() {
            hidden = block.forward(
                &hidden,
                start_pos,
                &mut self.kv_caches[layer_idx],
            );
        }

        // ── Step 3: Final RMSNorm ───────────────────────────────────────
        //
        // Apply the final normalization before the lm_head projection.
        let hidden = self.norm.forward(&hidden);

        // ── Step 4: Logit projection ────────────────────────────────────
        //
        // Project from hidden_size to vocab_size:
        //   logits = hidden · lm_head^T
        //
        // hidden:  [seq_len, hidden_size]
        // lm_head^T: [hidden_size, vocab_size]
        // logits: [seq_len, vocab_size]
        let lm_head_t = self.lm_head.transpose_2d();
        let logits = hidden.matmul(&lm_head_t);

        logits
    }

    /// Reset all KV caches, clearing stored key and value tensors.
    ///
    /// Call this when starting a new conversation turn so the model does
    /// not attend to tokens from a previous conversation. After calling
    /// this method, the next forward pass will behave as if the model
    /// has never seen any tokens (i.e., it will start a fresh prefill).
    pub fn reset_caches(&mut self) {
        for cache in &mut self.kv_caches {
            cache.clear();
        }
    }

    /// Get the model's vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Get a reference to the KV caches.
    ///
    /// Each layer has one [`KVCache`] entry. This accessor is useful for
    /// inspecting cache state in tests.
    pub fn kv_caches(&self) -> &[KVCache] {
        &self.kv_caches
    }

    /// Get a mutable reference to the lm_head weight tensor.
    ///
    /// The lm_head has shape `[vocab_size, hidden_size]` and projects from
    /// hidden states to vocabulary logits. This accessor is useful in tests
    /// that need to manipulate the output projection.
    pub fn lm_head_mut(&mut self) -> &mut Tensor {
        &mut self.lm_head
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a tiny model for testing.
    ///
    /// Configuration:
    /// - vocab_size = 32
    /// - hidden_size = 16
    /// - num_hidden_layers = 1
    /// - num_attention_heads = 2
    /// - num_key_value_heads = 1
    /// - intermediate_size = 32
    /// - max_position_embeddings = 64
    /// - rms_norm_eps = 1e-6
    /// - rope_theta = 10000.0
    /// - head_dim = 8
    fn make_test_model() -> QwenModel {
        let vocab = 32;
        let hidden = 16;
        let num_layers = 1;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 8;
        let intermediate = 32;
        let kv_dim = num_kv_heads * head_dim; // 8

        let config = ModelConfig {
            vocab_size: vocab,
            hidden_size: hidden,
            num_hidden_layers: num_layers,
            num_attention_heads: num_heads,
            num_key_value_heads: num_kv_heads,
            intermediate_size: intermediate,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            hidden_act: "silu".to_string(),
            tie_word_embeddings: false,
            head_dim: None,
        };

        // Embedding table: [vocab, hidden] with small values
        let embed_tokens = Tensor::new(
            vec![vocab, hidden],
            (0..vocab * hidden).map(|i| (i as f32 * 0.01).sin()).collect(),
        );

        // Build one transformer block
        let ln1_weight = Tensor::ones(vec![hidden]);
        let ln2_weight = Tensor::ones(vec![hidden]);

        // q_proj: [hidden, hidden] = [16, 16] — identity
        let q_proj = Tensor::new(
            vec![hidden, hidden],
            (0..hidden * hidden)
                .map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 })
                .collect(),
        );
        // k_proj: [kv_dim, hidden] = [8, 16]
        let k_proj = Tensor::new(
            vec![kv_dim, hidden],
            (0..kv_dim * hidden)
                .map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 })
                .collect(),
        );
        // v_proj: [kv_dim, hidden] = [8, 16]
        let v_proj = Tensor::new(
            vec![kv_dim, hidden],
            (0..kv_dim * hidden)
                .map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 })
                .collect(),
        );
        // o_proj: [hidden, hidden] = [16, 16] — identity
        let o_proj = Tensor::new(
            vec![hidden, hidden],
            (0..hidden * hidden)
                .map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 })
                .collect(),
        );

        // FFN weights
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

        let block = TransformerBlock::new(
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
            config.max_position_embeddings,
            config.rope_theta_f32(),
            config.eps_f32(),
        );

        let layers = vec![block];

        // Final norm weight
        let norm_weight = Tensor::ones(vec![hidden]);
        let norm = RMSNorm::new(norm_weight, config.eps_f32());

        // lm_head: [vocab, hidden]
        let lm_head = Tensor::new(
            vec![vocab, hidden],
            (0..vocab * hidden).map(|i| (i as f32 * 0.02).cos()).collect(),
        );

        QwenModel::new(embed_tokens, layers, norm, lm_head, config)
    }

    /// Test model forward with a single token.
    ///
    /// The output should be a tensor of shape [1, vocab_size].
    #[test]
    fn test_forward_single_token() {
        let mut model = make_test_model();
        let token_ids = &[5usize];
        let logits = model.forward(token_ids, 0);

        assert_eq!(
            logits.shape(),
            &[1, 32],
            "output shape should be [1, 32], got {:?}",
            logits.shape(),
        );

        // Logits should be finite
        for &v in logits.data() {
            assert!(v.is_finite(), "logits should be finite, got {}", v);
        }
    }

    /// Test model forward with multiple tokens (prefill).
    ///
    /// The output should be a tensor of shape [seq_len, vocab_size].
    #[test]
    fn test_forward_multiple_tokens() {
        let mut model = make_test_model();
        let token_ids: &[usize] = &[3, 7, 15];
        let logits = model.forward(token_ids, 0);

        assert_eq!(
            logits.shape(),
            &[3, 32],
            "output shape should be [3, 32], got {:?}",
            logits.shape(),
        );

        for &v in logits.data() {
            assert!(v.is_finite(), "logits should be finite, got {}", v);
        }
    }

    /// Test that the output shape is [seq_len, vocab_size] for various
    /// sequence lengths.
    #[test]
    fn test_output_shape_various_seq_lens() {
        let mut model = make_test_model();
        let vocab = 32;

        for seq_len in &[1, 2, 5, 10] {
            let token_ids: Vec<usize> = (0..*seq_len).map(|i| i % vocab).collect();
            let logits = model.forward(&token_ids, 0);
            assert_eq!(
                logits.shape(),
                &[*seq_len, vocab],
                "output shape should be [{}, {}], got {:?}",
                seq_len,
                vocab,
                logits.shape(),
            );
        }
    }

    /// Test that KV caches are created for each layer.
    ///
    /// After construction, there should be one KVCache per layer, and
    /// each should start empty. After a forward pass, each cache should
    /// be populated.
    #[test]
    fn test_kv_caches_created_per_layer() {
        let mut model = make_test_model();

        // Before forward: caches should exist but be empty
        assert_eq!(
            model.kv_caches.len(),
            model.config.num_hidden_layers,
            "should have one KV cache per layer",
        );
        for (i, cache) in model.kv_caches.iter().enumerate() {
            assert!(
                cache.key_cache.is_none(),
                "layer {} key cache should start empty",
                i,
            );
            assert!(
                cache.value_cache.is_none(),
                "layer {} value cache should start empty",
                i,
            );
        }

        // After forward pass: caches should be populated
        let token_ids: &[usize] = &[1, 2, 3];
        let _logits = model.forward(token_ids, 0);

        for (i, cache) in model.kv_caches.iter().enumerate() {
            assert!(
                cache.key_cache.is_some(),
                "layer {} key cache should be populated after forward",
                i,
            );
            assert!(
                cache.value_cache.is_some(),
                "layer {} value cache should be populated after forward",
                i,
            );
            // Each cache should have seq_len rows
            assert_eq!(
                cache.key_cache.as_ref().unwrap().shape()[0],
                3,
                "layer {} key cache should have 3 rows",
                i,
            );
        }
    }

    /// Test that the model can perform a prefill followed by a decode step,
    /// and that the KV cache grows correctly.
    #[test]
    fn test_prefill_then_decode() {
        let mut model = make_test_model();

        // Prefill: process 3 tokens
        let prefill_ids: &[usize] = &[0, 5, 10];
        let logits_prefill = model.forward(prefill_ids, 0);
        assert_eq!(logits_prefill.shape(), &[3, 32]);

        // Decode: process 1 more token
        let decode_id: &[usize] = &[15];
        let logits_decode = model.forward(decode_id, 3);
        assert_eq!(logits_decode.shape(), &[1, 32]);

        // KV cache should now have 4 rows (3 + 1)
        for (i, cache) in model.kv_caches.iter().enumerate() {
            assert_eq!(
                cache.key_cache.as_ref().unwrap().shape()[0],
                4,
                "layer {} cache should have 4 rows after prefill+decode",
                i,
            );
        }

        // All logits should be finite
        for &v in logits_decode.data() {
            assert!(v.is_finite(), "decode logits should be finite, got {}", v);
        }
    }

    /// Test that different token IDs produce different logits.
    ///
    /// This is a basic sanity check that the embedding lookup actually
    /// uses the token ID and that the model computation is non-trivial.
    #[test]
    fn test_different_tokens_different_logits() {
        let mut model1 = make_test_model();
        let mut model2 = make_test_model();

        let logits1 = model1.forward(&[0], 0);
        let logits2 = model2.forward(&[10], 0);

        let any_different = logits1
            .data()
            .iter()
            .zip(logits2.data().iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);

        assert!(
            any_different,
            "different token IDs should produce different logits"
        );
    }

    /// Test QwenModel::load from a directory with config.json and model.safetensors.
    ///
    /// This creates a temp directory, writes a tiny config and safetensors file
    /// with all required tensors at tiny dimensions, then calls `QwenModel::load`
    /// and verifies that loading succeeds and the model can do a forward pass.
    #[test]
    fn test_load_from_directory() {
        use std::io::Write;

        // Tiny model dimensions for testing.
        let vocab: usize = 8;
        let hidden: usize = 4;
        let num_layers: usize = 1;
        let num_heads: usize = 2;
        let num_kv_heads: usize = 1;
        let head_dim: usize = hidden / num_heads; // 2
        let kv_dim: usize = num_kv_heads * head_dim; // 2
        let intermediate: usize = 8;

        // Create a temporary directory.
        let dir = std::env::temp_dir().join("qwen35_rs_model_load_test");
        std::fs::create_dir_all(&dir).expect("should create temp dir");

        // Write config.json.
        let config_json = format!(r#"{{
            "vocab_size": {vocab},
            "hidden_size": {hidden},
            "num_hidden_layers": {num_layers},
            "num_attention_heads": {num_heads},
            "num_key_value_heads": {num_kv_heads},
            "intermediate_size": {intermediate},
            "max_position_embeddings": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "hidden_act": "silu",
            "tie_word_embeddings": false
        }}"#);
        let config_path = dir.join("config.json");
        let mut config_file = std::fs::File::create(&config_path).expect("should create config.json");
        write!(config_file, "{}", config_json).expect("should write config.json");

        // Build all tensor data and the safetensors header.
        // We accumulate raw data bytes and track offsets.
        let mut data_bytes: Vec<u8> = Vec::new();
        let mut header_entries: Vec<String> = Vec::new();

        // Helper to add a tensor to the file.
        let mut add_tensor = |name: &str, shape: &[usize], values: &[f32]| {
            let num_elements: usize = shape.iter().product();
            assert_eq!(values.len(), num_elements);
            let start = data_bytes.len();
            for &v in values {
                data_bytes.extend_from_slice(&v.to_le_bytes());
            }
            let end = data_bytes.len();
            let shape_str = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
            header_entries.push(format!(
                r#""{}":{{"dtype":"F32","shape":[{}],"data_offsets":[{},{}]}}"#,
                name, shape_str, start, end
            ));
        };

        // embed_tokens: [vocab, hidden]
        let embed_data: Vec<f32> = (0..vocab * hidden).map(|i| (i as f32 * 0.01).sin()).collect();
        add_tensor("model.embed_tokens.weight", &[vocab, hidden], &embed_data);

        // Layer 0 weights.
        // input_layernorm: [hidden]
        let ln1_data: Vec<f32> = vec![1.0; hidden];
        add_tensor("model.layers.0.input_layernorm.weight", &[hidden], &ln1_data);

        // q_proj: [hidden, hidden]
        let q_data: Vec<f32> = (0..hidden * hidden)
            .map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 })
            .collect();
        add_tensor("model.layers.0.self_attn.q_proj.weight", &[hidden, hidden], &q_data);

        // k_proj: [kv_dim, hidden]
        let k_data: Vec<f32> = (0..kv_dim * hidden)
            .map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 })
            .collect();
        add_tensor("model.layers.0.self_attn.k_proj.weight", &[kv_dim, hidden], &k_data);

        // v_proj: [kv_dim, hidden]
        let v_data: Vec<f32> = (0..kv_dim * hidden)
            .map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 })
            .collect();
        add_tensor("model.layers.0.self_attn.v_proj.weight", &[kv_dim, hidden], &v_data);

        // o_proj: [hidden, hidden]
        add_tensor("model.layers.0.self_attn.o_proj.weight", &[hidden, hidden], &q_data);

        // post_attention_layernorm: [hidden]
        let ln2_data: Vec<f32> = vec![1.0; hidden];
        add_tensor("model.layers.0.post_attention_layernorm.weight", &[hidden], &ln2_data);

        // FFN weights.
        let gate_data: Vec<f32> = vec![0.1; intermediate * hidden];
        add_tensor("model.layers.0.mlp.gate_proj.weight", &[intermediate, hidden], &gate_data);

        let up_data: Vec<f32> = vec![0.1; intermediate * hidden];
        add_tensor("model.layers.0.mlp.up_proj.weight", &[intermediate, hidden], &up_data);

        let down_data: Vec<f32> = vec![0.1; hidden * intermediate];
        add_tensor("model.layers.0.mlp.down_proj.weight", &[hidden, intermediate], &down_data);

        // Final norm: [hidden]
        let norm_data: Vec<f32> = vec![1.0; hidden];
        add_tensor("model.norm.weight", &[hidden], &norm_data);

        // lm_head: [vocab, hidden]
        let lm_head_data: Vec<f32> = (0..vocab * hidden).map(|i| (i as f32 * 0.02).cos()).collect();
        add_tensor("lm_head.weight", &[vocab, hidden], &lm_head_data);

        // Build the safetensors binary file.
        let header_json = format!("{{{}}}", header_entries.join(","));
        let header_bytes = header_json.as_bytes();
        let json_len = header_bytes.len();
        let aligned_len = ((json_len + 7) / 8) * 8;
        let padding = aligned_len - json_len;
        let header_size = aligned_len as u64;

        let mut file_data = Vec::new();
        file_data.extend_from_slice(&header_size.to_le_bytes());
        file_data.extend_from_slice(header_bytes);
        file_data.extend_from_slice(&vec![b' '; padding]);
        file_data.extend_from_slice(&data_bytes);

        let st_path = dir.join("model.safetensors");
        let mut st_file = std::fs::File::create(&st_path).expect("should create safetensors file");
        st_file.write_all(&file_data).expect("should write safetensors file");

        // Load the model.
        let mut model = QwenModel::load(&dir).expect("QwenModel::load should succeed");

        // Verify config was loaded correctly.
        assert_eq!(model.config.vocab_size, vocab);
        assert_eq!(model.config.hidden_size, hidden);
        assert_eq!(model.config.num_hidden_layers, num_layers);
        assert_eq!(model.config.num_attention_heads, num_heads);
        assert_eq!(model.config.num_key_value_heads, num_kv_heads);
        assert_eq!(model.config.intermediate_size, intermediate);

        // Verify the model can do a forward pass.
        let token_ids: &[usize] = &[0, 3, 7];
        let logits = model.forward(token_ids, 0);

        assert_eq!(
            logits.shape(),
            &[3, vocab],
            "output shape should be [3, {}], got {:?}",
            vocab,
            logits.shape(),
        );

        for &v in logits.data() {
            assert!(v.is_finite(), "logits should be finite, got {}", v);
        }

        // Clean up temp files.
        let _ = std::fs::remove_file(&config_path);
        let _ = std::fs::remove_file(&st_path);
        let _ = std::fs::remove_dir(&dir);
    }

    /// Test that QwenModel::load returns a descriptive error when a required
    /// tensor is missing from the safetensors file.
    #[test]
    fn test_load_missing_weight_error() {
        use std::io::Write;

        let vocab: usize = 8;
        let hidden: usize = 4;

        let dir = std::env::temp_dir().join("qwen35_rs_model_load_missing_test");
        std::fs::create_dir_all(&dir).expect("should create temp dir");

        // Write a minimal config.
        let config_json = format!(r#"{{
            "vocab_size": {vocab},
            "hidden_size": {hidden},
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "intermediate_size": 8,
            "max_position_embeddings": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "hidden_act": "silu",
            "tie_word_embeddings": false
        }}"#);
        let config_path = dir.join("config.json");
        let mut config_file = std::fs::File::create(&config_path).expect("should create config.json");
        write!(config_file, "{}", config_json).expect("should write config.json");

        // Write a safetensors file with ONLY embed_tokens (missing all other weights).
        let embed_data: Vec<f32> = (0..vocab * hidden).map(|i| (i as f32 * 0.01).sin()).collect();
        let num_elements = vocab * hidden;
        let data_bytes: Vec<u8> = embed_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let header_json = format!(
            r#"{{"model.embed_tokens.weight":{{"dtype":"F32","shape":[{vocab},{hidden}],"data_offsets":[0,{}]}}}}"#,
            num_elements * 4
        );
        let header_bytes = header_json.as_bytes();
        let json_len = header_bytes.len();
        let aligned_len = ((json_len + 7) / 8) * 8;
        let padding = aligned_len - json_len;
        let header_size = aligned_len as u64;

        let mut file_data = Vec::new();
        file_data.extend_from_slice(&header_size.to_le_bytes());
        file_data.extend_from_slice(header_bytes);
        file_data.extend_from_slice(&vec![b' '; padding]);
        file_data.extend_from_slice(&data_bytes);

        let st_path = dir.join("model.safetensors");
        let mut st_file = std::fs::File::create(&st_path).expect("should create safetensors file");
        st_file.write_all(&file_data).expect("should write safetensors file");

        // Loading should fail with a descriptive error about the missing weight.
        let err_msg = match QwenModel::load(&dir) {
            Ok(_) => panic!("expected load to fail due to missing weight"),
            Err(e) => e.to_string(),
        };
        assert!(
            err_msg.contains("Missing weight: model.layers.0.input_layernorm.weight"),
            "error should mention the missing weight, got: {}",
            err_msg,
        );

        // Clean up.
        let _ = std::fs::remove_file(&config_path);
        let _ = std::fs::remove_file(&st_path);
        let _ = std::fs::remove_dir(&dir);
    }

    /// Test that QwenModel::load returns an error when a tensor has the wrong shape.
    #[test]
    fn test_load_shape_mismatch_error() {
        use std::io::Write;

        let vocab: usize = 8;
        let hidden: usize = 4;
        let num_layers: usize = 1;
        let num_heads: usize = 2;
        let num_kv_heads: usize = 1;
        let head_dim: usize = hidden / num_heads; // 2
        let kv_dim: usize = num_kv_heads * head_dim; // 2
        let intermediate: usize = 8;

        let dir = std::env::temp_dir().join("qwen35_rs_model_load_shape_test");
        std::fs::create_dir_all(&dir).expect("should create temp dir");

        let config_json = format!(r#"{{
            "vocab_size": {vocab},
            "hidden_size": {hidden},
            "num_hidden_layers": {num_layers},
            "num_attention_heads": {num_heads},
            "num_key_value_heads": {num_kv_heads},
            "intermediate_size": {intermediate},
            "max_position_embeddings": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "hidden_act": "silu",
            "tie_word_embeddings": false
        }}"#);
        let config_path = dir.join("config.json");
        let mut config_file = std::fs::File::create(&config_path).expect("should create config.json");
        write!(config_file, "{}", config_json).expect("should write config.json");

        // Build safetensors with a wrong shape for k_proj (using [hidden, hidden] instead of [kv_dim, hidden]).
        let mut data_bytes: Vec<u8> = Vec::new();
        let mut header_entries: Vec<String> = Vec::new();

        let mut add_tensor = |name: &str, shape: &[usize], values: &[f32]| {
            let num_elements: usize = shape.iter().product();
            assert_eq!(values.len(), num_elements);
            let start = data_bytes.len();
            for &v in values {
                data_bytes.extend_from_slice(&v.to_le_bytes());
            }
            let end = data_bytes.len();
            let shape_str = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
            header_entries.push(format!(
                r#""{}":{{"dtype":"F32","shape":[{}],"data_offsets":[{},{}]}}"#,
                name, shape_str, start, end
            ));
        };

        let embed_data: Vec<f32> = (0..vocab * hidden).map(|i| (i as f32 * 0.01).sin()).collect();
        add_tensor("model.embed_tokens.weight", &[vocab, hidden], &embed_data);
        let ln1_data: Vec<f32> = vec![1.0; hidden];
        add_tensor("model.layers.0.input_layernorm.weight", &[hidden], &ln1_data);

        // q_proj with correct shape.
        let q_data: Vec<f32> = (0..hidden * hidden).map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 }).collect();
        add_tensor("model.layers.0.self_attn.q_proj.weight", &[hidden, hidden], &q_data);

        // k_proj with WRONG shape: [hidden, hidden] instead of [kv_dim, hidden].
        let k_data_wrong: Vec<f32> = vec![0.0; hidden * hidden];
        add_tensor("model.layers.0.self_attn.k_proj.weight", &[hidden, hidden], &k_data_wrong);

        // Fill in the rest with correct shapes so we get past the k_proj error only.
        let v_data: Vec<f32> = (0..kv_dim * hidden).map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 }).collect();
        add_tensor("model.layers.0.self_attn.v_proj.weight", &[kv_dim, hidden], &v_data);
        add_tensor("model.layers.0.self_attn.o_proj.weight", &[hidden, hidden], &q_data);
        let ln2_data: Vec<f32> = vec![1.0; hidden];
        add_tensor("model.layers.0.post_attention_layernorm.weight", &[hidden], &ln2_data);
        let gate_data: Vec<f32> = vec![0.1; intermediate * hidden];
        add_tensor("model.layers.0.mlp.gate_proj.weight", &[intermediate, hidden], &gate_data);
        let up_data: Vec<f32> = vec![0.1; intermediate * hidden];
        add_tensor("model.layers.0.mlp.up_proj.weight", &[intermediate, hidden], &up_data);
        let down_data: Vec<f32> = vec![0.1; hidden * intermediate];
        add_tensor("model.layers.0.mlp.down_proj.weight", &[hidden, intermediate], &down_data);
        let norm_data: Vec<f32> = vec![1.0; hidden];
        add_tensor("model.norm.weight", &[hidden], &norm_data);
        let lm_head_data: Vec<f32> = (0..vocab * hidden).map(|i| (i as f32 * 0.02).cos()).collect();
        add_tensor("lm_head.weight", &[vocab, hidden], &lm_head_data);

        let header_json = format!("{{{}}}", header_entries.join(","));
        let header_bytes = header_json.as_bytes();
        let json_len = header_bytes.len();
        let aligned_len = ((json_len + 7) / 8) * 8;
        let padding = aligned_len - json_len;
        let header_size = aligned_len as u64;

        let mut file_data = Vec::new();
        file_data.extend_from_slice(&header_size.to_le_bytes());
        file_data.extend_from_slice(header_bytes);
        file_data.extend_from_slice(&vec![b' '; padding]);
        file_data.extend_from_slice(&data_bytes);

        let st_path = dir.join("model.safetensors");
        let mut st_file = std::fs::File::create(&st_path).expect("should create safetensors file");
        st_file.write_all(&file_data).expect("should write safetensors file");

        // Loading should fail with a shape mismatch error.
        let err_msg = match QwenModel::load(&dir) {
            Ok(_) => panic!("expected load to fail due to shape mismatch"),
            Err(e) => e.to_string(),
        };
        assert!(
            err_msg.contains("Shape mismatch for model.layers.0.self_attn.k_proj.weight"),
            "error should mention shape mismatch for k_proj, got: {}",
            err_msg,
        );

        // Clean up.
        let _ = std::fs::remove_file(&config_path);
        let _ = std::fs::remove_file(&st_path);
        let _ = std::fs::remove_dir(&dir);
    }

    /// Test that QwenModel::load works when tie_word_embeddings is true
    /// (lm_head.weight is not present in the safetensors file).
    #[test]
    fn test_load_tied_embeddings() {
        use std::io::Write;

        let vocab: usize = 8;
        let hidden: usize = 4;
        let num_layers: usize = 1;
        let num_heads: usize = 2;
        let num_kv_heads: usize = 1;
        let head_dim: usize = hidden / num_heads; // 2
        let kv_dim: usize = num_kv_heads * head_dim; // 2
        let intermediate: usize = 8;

        let dir = std::env::temp_dir().join("qwen35_rs_model_load_tied_test");
        std::fs::create_dir_all(&dir).expect("should create temp dir");

        // Config with tie_word_embeddings = true.
        let config_json = format!(r#"{{
            "vocab_size": {vocab},
            "hidden_size": {hidden},
            "num_hidden_layers": {num_layers},
            "num_attention_heads": {num_heads},
            "num_key_value_heads": {num_kv_heads},
            "intermediate_size": {intermediate},
            "max_position_embeddings": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "hidden_act": "silu",
            "tie_word_embeddings": true
        }}"#);
        let config_path = dir.join("config.json");
        let mut config_file = std::fs::File::create(&config_path).expect("should create config.json");
        write!(config_file, "{}", config_json).expect("should write config.json");

        // Build safetensors WITHOUT lm_head.weight.
        let mut data_bytes: Vec<u8> = Vec::new();
        let mut header_entries: Vec<String> = Vec::new();

        let mut add_tensor = |name: &str, shape: &[usize], values: &[f32]| {
            let num_elements: usize = shape.iter().product();
            assert_eq!(values.len(), num_elements);
            let start = data_bytes.len();
            for &v in values {
                data_bytes.extend_from_slice(&v.to_le_bytes());
            }
            let end = data_bytes.len();
            let shape_str = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
            header_entries.push(format!(
                r#""{}":{{"dtype":"F32","shape":[{}],"data_offsets":[{},{}]}}"#,
                name, shape_str, start, end
            ));
        };

        let embed_data: Vec<f32> = (0..vocab * hidden).map(|i| (i as f32 * 0.01).sin()).collect();
        add_tensor("model.embed_tokens.weight", &[vocab, hidden], &embed_data);
        let ln1_data: Vec<f32> = vec![1.0; hidden];
        add_tensor("model.layers.0.input_layernorm.weight", &[hidden], &ln1_data);

        let q_data: Vec<f32> = (0..hidden * hidden).map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 }).collect();
        add_tensor("model.layers.0.self_attn.q_proj.weight", &[hidden, hidden], &q_data);

        let kv_data: Vec<f32> = (0..kv_dim * hidden).map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 }).collect();
        add_tensor("model.layers.0.self_attn.k_proj.weight", &[kv_dim, hidden], &kv_data);
        add_tensor("model.layers.0.self_attn.v_proj.weight", &[kv_dim, hidden], &kv_data);
        add_tensor("model.layers.0.self_attn.o_proj.weight", &[hidden, hidden], &q_data);
        let ln2_data: Vec<f32> = vec![1.0; hidden];
        add_tensor("model.layers.0.post_attention_layernorm.weight", &[hidden], &ln2_data);
        let gate_data: Vec<f32> = vec![0.1; intermediate * hidden];
        add_tensor("model.layers.0.mlp.gate_proj.weight", &[intermediate, hidden], &gate_data);
        let up_data: Vec<f32> = vec![0.1; intermediate * hidden];
        add_tensor("model.layers.0.mlp.up_proj.weight", &[intermediate, hidden], &up_data);
        let down_data: Vec<f32> = vec![0.1; hidden * intermediate];
        add_tensor("model.layers.0.mlp.down_proj.weight", &[hidden, intermediate], &down_data);
        let norm_data: Vec<f32> = vec![1.0; hidden];
        add_tensor("model.norm.weight", &[hidden], &norm_data);
        // No lm_head.weight — it should use embed_tokens instead.

        let header_json = format!("{{{}}}", header_entries.join(","));
        let header_bytes = header_json.as_bytes();
        let json_len = header_bytes.len();
        let aligned_len = ((json_len + 7) / 8) * 8;
        let padding = aligned_len - json_len;
        let header_size = aligned_len as u64;

        let mut file_data = Vec::new();
        file_data.extend_from_slice(&header_size.to_le_bytes());
        file_data.extend_from_slice(header_bytes);
        file_data.extend_from_slice(&vec![b' '; padding]);
        file_data.extend_from_slice(&data_bytes);

        let st_path = dir.join("model.safetensors");
        let mut st_file = std::fs::File::create(&st_path).expect("should create safetensors file");
        st_file.write_all(&file_data).expect("should write safetensors file");

        // Load should succeed even without lm_head.weight.
        let mut model = QwenModel::load(&dir).expect("QwenModel::load should succeed with tied embeddings");
        assert!(model.config.tie_word_embeddings);

        // Forward pass should work.
        let logits = model.forward(&[0, 3], 0);
        assert_eq!(logits.shape(), &[2, vocab]);
        for &v in logits.data() {
            assert!(v.is_finite(), "logits should be finite, got {}", v);
        }

        // Clean up.
        let _ = std::fs::remove_file(&config_path);
        let _ = std::fs::remove_file(&st_path);
        let _ = std::fs::remove_dir(&dir);
    }
}
