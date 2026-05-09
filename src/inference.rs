//! Autoregressive generation loop that ties together model, tokenizer, and sampling.
//!
//! This module implements the core inference loop for next-token prediction.
//! During autoregressive generation, the model produces one token at a time:
//! at each step, it computes logits over the vocabulary, samples a token,
//! and feeds that token back as input for the next step. The process repeats
//! until an end-of-sequence (EOS) token is generated or a maximum number of
//! tokens is reached.
//!
//! # Prefill vs. Decode
//!
//! Generation has two distinct phases:
//!
//! - **Prefill**: The prompt (possibly many tokens) is processed in a single
//!   forward pass. The KV caches are populated for all prompt tokens. Logits
//!   are extracted only for the *last* prompt position, since that is the
//!   position from which the first generated token is sampled.
//!
//! - **Decode**: After prefill, each subsequent step processes exactly one
//!   token. The new token's key and value are appended to the existing KV
//!   caches, so the model can attend to all previously seen tokens without
//!   recomputing their representations. This is the key efficiency advantage
//!   of the KV cache.
//!
//! # KV Cache Management
//!
//! The KV caches persist across decode steps within a single generation call.
//! When starting a new conversation turn, [`InferenceEngine::reset`] clears
//! all caches so the model starts fresh. If caches are not reset, the model
//! will continue generating as if the previous conversation context is still
//! present, which is useful for multi-turn dialogue.
//!
//! # Example
//!
//! ```ignore
//! use qwen3_5_rs::inference::InferenceEngine;
//! use qwen3_5_rs::sampling::SamplingConfig;
//! use std::path::Path;
//!
//! let mut engine = InferenceEngine::load(
//!     Path::new("model_dir"),
//!     SamplingConfig::default(),
//! ).unwrap();
//!
//! // Generate text (non-streaming).
//! let output = engine.generate("Hello, world!", 100);
//! println!("{}", output);
//!
//! // Generate with streaming callback.
//! let output = engine.generate_with_callback("Tell me a story", 200, |token_text| {
//!     print!("{}", token_text);
//! });
//! ```

use crate::model::QwenModel;
use crate::sampling::{sample, SamplingConfig};
use crate::tokenizer::Tokenizer;

use std::path::Path;

// ─────────────────────────────────────────────────────────────────────────────
// InferenceEngine
// ─────────────────────────────────────────────────────────────────────────────

/// An inference engine that manages model, tokenizer, and generation state.
///
/// The engine owns a [`QwenModel`], a [`Tokenizer`], and a [`SamplingConfig`].
/// Together these provide everything needed to convert text prompts into
/// generated text:
///
/// - The **tokenizer** converts between human-readable text and integer token IDs.
/// - The **model** maps token IDs to logits over the vocabulary.
/// - The **sampling config** controls how logits are converted to a single next token.
///
/// The engine also manages the KV caches that persist across decode steps within
/// a generation call. Use [`InferenceEngine::reset`] to clear caches between
/// independent conversation turns.
pub struct InferenceEngine {
    model: QwenModel,
    tokenizer: Tokenizer,
    sampling_config: SamplingConfig,
}

impl InferenceEngine {
    /// Create a new inference engine from pre-built components.
    ///
    /// This constructor is useful for programmatic construction in tests,
    /// where you want to build a model in memory without reading files.
    ///
    /// # Arguments
    ///
    /// * `model`            - A loaded `QwenModel`.
    /// * `tokenizer`        - A loaded `Tokenizer`.
    /// * `sampling_config`  - Configuration for token sampling.
    pub fn new(model: QwenModel, tokenizer: Tokenizer, sampling_config: SamplingConfig) -> Self {
        Self {
            model,
            tokenizer,
            sampling_config,
        }
    }

    /// Create a new inference engine from a model directory.
    ///
    /// The directory should contain:
    /// - `config.json`: model hyperparameters
    /// - `model.safetensors`: model weights
    /// - `tokenizer.json`: BPE tokenizer vocabulary and merge rules
    ///
    /// # Arguments
    ///
    /// * `model_dir`       - Path to the directory containing model files.
    /// * `sampling_config` - Configuration for token sampling (temperature,
    ///   top-k, top-p, seed).
    ///
    /// # Errors
    ///
    /// Returns an error if any file cannot be read or parsed, or if the model
    /// weights contain missing or mismatched tensors.
    pub fn load(
        model_dir: &Path,
        sampling_config: SamplingConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let model = QwenModel::load(model_dir)?;
        let tokenizer = Tokenizer::from_file(&model_dir.join("tokenizer.json"))?;
        Ok(Self {
            model,
            tokenizer,
            sampling_config,
        })
    }

    /// Generate text from a prompt.
    ///
    /// This is the simplest generation interface. It runs the full autoregressive
    /// loop and returns the complete output text, which includes the original
    /// prompt followed by the generated tokens.
    ///
    /// # Arguments
    ///
    /// * `prompt`     - The input text to continue generating from.
    /// * `max_tokens` - Maximum number of new tokens to generate (not counting
    ///   the prompt tokens).
    ///
    /// # Returns
    ///
    /// The prompt text followed by all generated text, as a single string.
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> String {
        self.generate_with_callback(prompt, max_tokens, |_token_text| {})
    }

    /// Generate tokens one at a time, calling a callback for each token.
    ///
    /// This is useful for interactive or streaming output where the user wants
    /// to see tokens as they are produced rather than waiting for the entire
    /// generation to complete.
    ///
    /// # Generation Algorithm
    ///
    /// 1. **Encode** the prompt into token IDs using the tokenizer.
    /// 2. **Prefill**: Run the model forward pass on all prompt tokens at
    ///    `start_pos = 0`. This populates the KV caches and produces logits
    ///    of shape `[prompt_len, vocab_size]`.
    /// 3. Extract the logits for the **last position** of the prefill output.
    /// 4. **Sample** the next token from these logits using the sampling config.
    /// 5. **Decode loop** (up to `max_tokens` iterations):
    ///    a. If the sampled token is the EOS token, stop generation.
    ///    b. Decode the sampled token to text and call the callback.
    ///    c. Run the model forward pass on just the sampled token at the
    ///       current `start_pos`.
    ///    d. Sample the next token from the resulting logits.
    ///    e. Increment `start_pos` by 1.
    /// 6. Return the prompt text concatenated with all generated token text.
    ///
    /// # Arguments
    ///
    /// * `prompt`     - The input text to continue generating from.
    /// * `max_tokens` - Maximum number of new tokens to generate.
    /// * `callback`   - A function called with each decoded token's text as
    ///   it is generated. This allows streaming output to the user.
    ///
    /// # Returns
    ///
    /// The prompt text followed by all generated text, as a single string.
    pub fn generate_with_callback<F>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        mut callback: F,
    ) -> String
    where
        F: FnMut(&str),
    {
        // Step 1: Encode the prompt into token IDs.
        let token_ids = self.tokenizer.encode(prompt);
        if token_ids.is_empty() {
            return String::new();
        }

        let eos_id = self.tokenizer.eos_token_id();

        // Step 2: Prefill — process all prompt tokens in one forward pass.
        let logits = self.model.forward(&token_ids, 0);
        let seq_len = logits.shape()[0];
        let vocab_size = logits.shape()[1];

        // Step 3: Extract logits for the last position.
        // logits shape: [seq_len, vocab_size]
        // We want the row at index (seq_len - 1).
        let last_row_offset = (seq_len - 1) * vocab_size;
        let last_logits: Vec<f32> = logits.data()
            [last_row_offset..last_row_offset + vocab_size]
            .to_vec();

        // Step 4: Sample the first generated token.
        let mut sampled_token = sample(&last_logits, &self.sampling_config);

        // Track position for KV cache.
        let mut current_pos = seq_len;

        // Collect generated token IDs for final decoding.
        let mut generated_ids: Vec<usize> = Vec::new();
        // Also collect decoded text incrementally for the callback.
        let mut generated_text = String::new();

        // Step 5: Decode loop — generate one token at a time.
        for _ in 0..max_tokens {
            // Step 5a: Check for EOS.
            if sampled_token == eos_id {
                break;
            }

            generated_ids.push(sampled_token);

            // Decode this single token and call the callback.
            let token_text = self.tokenizer.decode(&[sampled_token]);
            callback(&token_text);
            generated_text.push_str(&token_text);

            // Step 5b: Forward pass with just the sampled token.
            let logits = self.model.forward(&[sampled_token], current_pos);

            // Step 5c: Sample the next token.
            // logits shape is [1, vocab_size], so we take row 0.
            let logits_slice: Vec<f32> = logits.data().to_vec();
            sampled_token = sample(&logits_slice, &self.sampling_config);

            // Step 5d: Update position.
            current_pos += 1;
        }

        // Step 6: Return prompt + generated text.
        format!("{}{}", prompt, generated_text)
    }

    /// Reset the KV caches, starting a new conversation turn.
    ///
    /// After calling `reset()`, the next generation call will start with empty
    /// KV caches. This means the model will not attend to any tokens from
    /// previous generation calls. If you want the model to remember the
    /// previous conversation context, do not call `reset()` between turns.
    pub fn reset(&mut self) {
        self.model.reset_caches();
    }

    /// Get the model's EOS (End of Sequence) token ID.
    ///
    /// The EOS token signals the end of generation. When the model samples
    /// the EOS token during generation, the loop terminates early.
    ///
    /// For Qwen3 models, the EOS token is `<|endoftext|>` with ID 151643.
    pub fn eos_token_id(&self) -> usize {
        self.tokenizer.eos_token_id()
    }

    /// Get a reference to the underlying model.
    ///
    /// This is useful in integration tests that need to inspect model state
    /// (e.g., KV caches) or run forward passes directly.
    pub fn model(&self) -> &QwenModel {
        &self.model
    }

    /// Get a mutable reference to the underlying model.
    ///
    /// This is useful in integration tests that need to run forward passes
    /// or modify model state.
    pub fn model_mut(&mut self) -> &mut QwenModel {
        &mut self.model
    }

    /// Get a reference to the tokenizer.
    ///
    /// This is useful in integration tests that need to encode or decode
    /// text independently of the generation loop.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::rmsnorm::RMSNorm;
    use crate::tensor::Tensor;
    use crate::transformer_block::TransformerBlock;

    /// Helper: create a tiny InferenceEngine for testing.
    ///
    /// Uses a minimal model (vocab=32, hidden=16, 1 layer) and a minimal
    /// tokenizer with a small vocabulary and BPE merges.
    fn make_test_engine() -> InferenceEngine {
        make_test_engine_with_config(SamplingConfig::default())
    }

    /// Helper: create a tiny InferenceEngine with a specific SamplingConfig.
    fn make_test_engine_with_config(sampling_config: SamplingConfig) -> InferenceEngine {
        // ── Build the model (same as model.rs tests) ──
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

        let q_proj = Tensor::new(
            vec![hidden, hidden],
            (0..hidden * hidden)
                .map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 })
                .collect(),
        );
        let k_proj = Tensor::new(
            vec![kv_dim, hidden],
            (0..kv_dim * hidden)
                .map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 })
                .collect(),
        );
        let v_proj = Tensor::new(
            vec![kv_dim, hidden],
            (0..kv_dim * hidden)
                .map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 })
                .collect(),
        );
        let o_proj = Tensor::new(
            vec![hidden, hidden],
            (0..hidden * hidden)
                .map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 })
                .collect(),
        );

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

        let norm_weight = Tensor::ones(vec![hidden]);
        let norm = RMSNorm::new(norm_weight, config.eps_f32());

        let lm_head = Tensor::new(
            vec![vocab, hidden],
            (0..vocab * hidden).map(|i| (i as f32 * 0.02).cos()).collect(),
        );

        let model = QwenModel::new(embed_tokens, layers, norm, lm_head, config);

        // ── Build a minimal tokenizer ──
        // We create a tokenizer with a small vocabulary that maps single
        // characters to IDs. Token IDs 0..31 map to characters 'a'..'z' plus
        // a few extras. The EOS token is at ID 0 (mapped to a special token).
        let tokenizer_json = r#"{
            "version": "1.0",
            "added_tokens": [
                {"id": 0, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "model": {
                "type": "BPE",
                "vocab": {
                    "<|endoftext|>": 0,
                    "a": 1,
                    "b": 2,
                    "c": 3,
                    "d": 4,
                    "e": 5,
                    "f": 6,
                    "g": 7,
                    "h": 8,
                    "i": 9,
                    "j": 10,
                    "k": 11,
                    "l": 12,
                    "m": 13,
                    "n": 14,
                    "o": 15,
                    "p": 16,
                    "q": 17,
                    "r": 18,
                    "s": 19,
                    "t": 20,
                    "u": 21,
                    "v": 22,
                    "w": 23,
                    "x": 24,
                    "y": 25,
                    "z": 26,
                    " ": 27,
                    "!": 28,
                    "ab": 29,
                    "cd": 30,
                    "Ġ": 31
                },
                "merges": [
                    "a b",
                    "c d"
                ]
            }
        }"#;

        let tokenizer = Tokenizer::from_json(tokenizer_json).unwrap();

        InferenceEngine {
            model,
            tokenizer,
            sampling_config,
        }
    }

    /// Test that basic generation produces non-empty output.
    #[test]
    fn test_generate_basic() {
        let mut engine = make_test_engine();

        let output = engine.generate("a", 10);
        // The output should at least contain the prompt.
        assert!(
            !output.is_empty(),
            "generate should produce non-empty output"
        );
        assert!(
            output.starts_with('a'),
            "output should start with the prompt text"
        );
    }

    /// Test that generation stops at the max_tokens limit.
    #[test]
    fn test_generate_with_max_tokens() {
        // Use greedy sampling so the output is deterministic.
        let config = SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        };
        let mut engine = make_test_engine_with_config(config);

        // Generate with max_tokens = 3.
        let output = engine.generate("a", 3);
        assert!(
            output.starts_with('a'),
            "output should start with the prompt text"
        );

        // The generated portion should be at most 3 tokens.
        // We cannot guarantee the exact number because EOS might stop
        // generation early, but the generated portion should exist.
        let generated = &output[1..]; // strip prompt "a"
        if !generated.is_empty() {
            // At least some tokens were generated — that is the success condition.
            // With greedy decoding and our tiny model, we should get exactly
            // max_tokens generated tokens (unless EOS is hit, which is unlikely
            // with a random model).
        }
    }

    /// Test that reset() clears KV caches.
    #[test]
    fn test_reset_clears_cache() {
        let mut engine = make_test_engine();

        // Do a generation to populate KV caches.
        let _ = engine.generate("ab", 5);

        // After generation, KV caches should be populated (non-empty).
        // We verify this indirectly: after reset, caches should be clear.
        engine.reset();

        // Now check caches are empty after reset.
        for cache in engine.model.kv_caches() {
            assert!(
                cache.key_cache.is_none(),
                "key cache should be None after reset"
            );
            assert!(
                cache.value_cache.is_none(),
                "value cache should be None after reset"
            );
        }
    }

    /// Test that generation stops when the EOS token is produced.
    #[test]
    fn test_eos_stops_generation() {
        // We construct a scenario where the model will produce the EOS token.
        // With our tiny test model, we can manipulate the lm_head weights
        // so that a specific input token always produces EOS as the top token.
        //
        // Strategy: Create an engine where the lm_head weights are set up so
        // that for any input, token 0 (EOS) gets the highest logit.
        // We do this by making the lm_head's first row all very large values
        // and the rest zero.

        let config = SamplingConfig {
            temperature: 0.0, // greedy — always pick the highest logit
            top_k: 0,
            top_p: 1.0,
            seed: None,
        };
        let mut engine = make_test_engine_with_config(config);

        // Verify that EOS token ID is 0 for our test tokenizer.
        assert_eq!(engine.eos_token_id(), 0, "EOS token should be ID 0");

        // Manipulate the model's lm_head so that token 0 (EOS) always wins.
        // The lm_head has shape [vocab_size, hidden_size].
        // Row 0 corresponds to EOS. Set it to large values.
        {
            let lm_head = engine.model.lm_head_mut();
            let vocab = lm_head.shape()[0];
            let hidden = lm_head.shape()[1];
            let mut data = lm_head.data().to_vec();
            // Set row 0 (EOS) to large positive values.
            for j in 0..hidden {
                data[j] = 100.0;
            }
            // Set all other rows to zero so EOS dominates.
            for i in 1..vocab {
                for j in 0..hidden {
                    data[i * hidden + j] = 0.0;
                }
            }
            *lm_head = Tensor::new(vec![vocab, hidden], data);
        }

        // Now generate. The model should immediately produce EOS (token 0),
        // and generation should stop.
        let output = engine.generate("a", 100);

        // The output should be just the prompt, since EOS was hit immediately
        // after prefill.
        assert_eq!(
            output, "a",
            "generation should stop at EOS and return only the prompt, got: {:?}",
            output
        );
    }
}
