//! Model configuration parsing from config.json
//!
//! This module defines [`ModelConfig`], a struct that captures the
//! hyperparameters of a Qwen3 transformer model as stored in the
//! `config.json` file distributed with HuggingFace model weights.
//!
//! # Usage
//!
//! ```no_run
//! use qwen3_5_rs::config::ModelConfig;
//! use std::path::Path;
//!
//! let config = ModelConfig::from_file(Path::new("model_dir/config.json")).unwrap();
//! println!("head_dim = {}", config.head_dim());
//! ```
//!
//! # Why f64 for floating-point fields?
//!
//! `rms_norm_eps` and `rope_theta` are stored as `f64` in this struct because
//! JSON numbers can carry more precision than `f32` provides (e.g.
//! `1e-6` is exactly representable in `f64` but rounds in `f32`). The
//! convenience methods [`ModelConfig::eps_f32`] and
//! [`ModelConfig::rope_theta_f32`] cast to `f32` at the call site where the
//! value is actually used in tensor computations.

use serde::Deserialize;
use std::fs;
use std::path::Path;

// ─────────────────────────────────────────────────────────────────────────────
// ModelConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Model configuration parsed from a HuggingFace `config.json`.
///
/// Every field corresponds to a key in the JSON file. Fields that are present
/// in the Qwen3 config but not needed at inference time (e.g.
/// `architectures`, `model_type`) are intentionally omitted — serde will
/// silently ignore unknown fields by default.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    /// Vocabulary size — the number of distinct tokens the model can encode.
    ///
    /// This determines the width of the embedding matrix and the output
    /// projection (`lm_head`). For Qwen3-0.6B this is 151 936.
    pub vocab_size: usize,

    /// Hidden size — the dimensionality of the transformer's internal
    /// representations (often called `d_model`).
    ///
    /// Every residual stream vector has this length. For Qwen3-0.6B this
    /// is 1024.
    pub hidden_size: usize,

    /// Number of transformer blocks (layers) in the model.
    ///
    /// Each block contains one self-attention layer and one feed-forward
    /// network. For Qwen3-0.6B this is 28.
    pub num_hidden_layers: usize,

    /// Number of query (attention) heads.
    ///
    /// The hidden state is split into this many parallel attention heads,
    /// each of dimension `head_dim`. For Qwen3-0.6B this is 16.
    pub num_attention_heads: usize,

    /// Number of key-value heads for Grouped Query Attention (GQA).
    ///
    /// When this is less than `num_attention_heads`, multiple query heads
    /// share the same key and value projections, reducing KV-cache memory.
    /// For Qwen3-0.6B this is 8 (each KV head serves 2 query heads).
    pub num_key_value_heads: usize,

    /// Intermediate (feed-forward) size — the dimensionality of the
    /// up-projection inside the SwiGLU FFN.
    ///
    /// For Qwen3-0.6B this is 3072 (3x the hidden size).
    pub intermediate_size: usize,

    /// Maximum position embeddings — the longest sequence the model was
    /// trained to handle.
    ///
    /// This bounds the RoPE frequency table size and the KV-cache length.
    /// For Qwen3-0.6B this is 40 960.
    pub max_position_embeddings: usize,

    /// Epsilon used in RMSNorm for numerical stability.
    ///
    /// RMSNorm computes `x / sqrt(mean(x^2) + eps)`. The small epsilon
    /// prevents division by zero when the input vector is near-zero.
    /// Stored as `f64` for full JSON precision; use [`Self::eps_f32`]
    /// for the `f32` value used in tensor math. For Qwen3-0.6B this
    /// is 1e-6.
    pub rms_norm_eps: f64,

    /// Base frequency for Rotary Position Embeddings (RoPE).
    ///
    /// RoPE encodes position information by rotating query and key vectors
    /// at frequencies that decrease geometrically from `1/theta` to
    /// `max_pos/theta`. A larger theta gives slower frequency decay,
    /// enabling longer context. Stored as `f64` for precision; use
    /// [`Self::rope_theta_f32`] for the `f32` value. For Qwen3-0.6B
    /// this is 1 000 000.0.
    pub rope_theta: f64,

    /// Hidden activation function used in the FFN.
    ///
    /// This is the name of the non-linearity applied to the gate projection
    /// in the SwiGLU FFN. For Qwen3-0.6B this is `"silu"`
    /// (Sigmoid Linear Unit, a.k.a. Swish).
    pub hidden_act: String,

    /// Whether the word embedding matrix and the output projection
    /// (`lm_head`) share the same weights.
    ///
    /// When `true`, the model saves parameters by reusing the embedding
    /// matrix for the final logits projection. For Qwen3-0.6B this is
    /// `true`.
    pub tie_word_embeddings: bool,

    /// Explicit dimension per attention head.
    ///
    /// When present in the config, this overrides the default computation
    /// of `hidden_size / num_attention_heads`. For Qwen3-0.6B this is 128
    /// (instead of the default 1024/16 = 64).
    ///
    /// This field is optional because not all model configs include it.
    /// When absent, [`Self::head_dim()`] falls back to the division.
    #[serde(default)]
    pub head_dim: Option<usize>,
}

impl ModelConfig {
    /// Dimension of each attention head.
    ///
    /// If `head_dim` is explicitly set in the config (e.g. Qwen3-0.6B uses
    /// `head_dim = 128` instead of the default `1024/16 = 64`), return that
    /// value. Otherwise, fall back to `hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Load a model configuration from a `config.json` file on disk.
    ///
    /// The file is expected to be in the HuggingFace format — a JSON object
    /// whose keys match the fields of [`ModelConfig`]. Any extra keys (e.g.
    /// `architectures`, `model_type`) are silently ignored by serde.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or the JSON does not
    /// contain the required fields with the expected types.
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let json = fs::read_to_string(path)?;
        Self::from_json(&json)
    }

    /// Parse a model configuration from a JSON string.
    ///
    /// This is useful when the config JSON has already been loaded into
    /// memory (e.g. from a safetensors metadata bundle) or in tests.
    ///
    /// # Errors
    ///
    /// Returns an error if the string is not valid JSON or is missing
    /// required fields.
    pub fn from_json(json: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let config: ModelConfig = serde_json::from_str(json)?;
        Ok(config)
    }

    /// Return `rms_norm_eps` as an `f32`.
    ///
    /// The value is stored as `f64` in this struct for full JSON precision,
    /// but tensor operations use `f32`. This method performs the cast.
    pub fn eps_f32(&self) -> f32 {
        self.rms_norm_eps as f32
    }

    /// Return `rope_theta` as an `f32`.
    ///
    /// The value is stored as `f64` in this struct for full JSON precision,
    /// but tensor operations use `f32`. This method performs the cast.
    pub fn rope_theta_f32(&self) -> f32 {
        self.rope_theta as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// JSON string matching the Qwen3-0.6B config.json format.
    const QWEN3_CONFIG_JSON: &str = r#"{
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": 151936,
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "intermediate_size": 3072,
        "max_position_embeddings": 40960,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "hidden_act": "silu",
        "tie_word_embeddings": true,
        "head_dim": 128
    }"#;

    #[test]
    fn test_parse_from_json_string() {
        let config = ModelConfig::from_json(QWEN3_CONFIG_JSON)
            .expect("parsing Qwen3 config should succeed");

        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.intermediate_size, 3072);
        assert_eq!(config.max_position_embeddings, 40960);
        assert!((config.rms_norm_eps - 1e-6).abs() < 1e-18);
        assert!((config.rope_theta - 1000000.0).abs() < 1e-10);
        assert_eq!(config.hidden_act, "silu");
        assert!(config.tie_word_embeddings);
    }

    #[test]
    fn test_head_dim_explicit() {
        let config = ModelConfig::from_json(QWEN3_CONFIG_JSON).unwrap();
        // Explicitly set to 128 in the JSON (overrides default 1024/16=64)
        assert_eq!(config.head_dim(), 128);
        assert_eq!(config.head_dim, Some(128));
    }

    #[test]
    fn test_head_dim_fallback() {
        // When head_dim is not in the config, it should fall back to
        // hidden_size / num_attention_heads.
        let json = r#"{
            "vocab_size": 151936,
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "intermediate_size": 3072,
            "max_position_embeddings": 40960,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "hidden_act": "silu",
            "tie_word_embeddings": true
        }"#;
        let config = ModelConfig::from_json(json).unwrap();
        // No explicit head_dim — should compute 1024 / 16 = 64
        assert_eq!(config.head_dim, None);
        assert_eq!(config.head_dim(), 64);
    }

    #[test]
    fn test_head_dim_override() {
        // Qwen3-0.6B has head_dim=128 which differs from hidden_size/num_heads=64.
        let json = r#"{
            "vocab_size": 151936,
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "intermediate_size": 3072,
            "max_position_embeddings": 40960,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "hidden_act": "silu",
            "tie_word_embeddings": true,
            "head_dim": 128
        }"#;
        let config = ModelConfig::from_json(json).unwrap();
        // Explicit head_dim should override the default computation
        assert_eq!(config.head_dim, Some(128));
        assert_eq!(config.head_dim(), 128);
    }

    #[test]
    fn test_eps_f32() {
        let config = ModelConfig::from_json(QWEN3_CONFIG_JSON).unwrap();
        let eps = config.eps_f32();
        // The f64 value 1e-6 cast to f32 should be very close to 1e-6.
        assert!(
            (eps - 1e-6f32).abs() < 1e-12f32,
            "eps_f32 should be close to 1e-6, got {}",
            eps,
        );
    }

    #[test]
    fn test_rope_theta_f32() {
        let config = ModelConfig::from_json(QWEN3_CONFIG_JSON).unwrap();
        let theta = config.rope_theta_f32();
        // The f64 value 1000000.0 is exactly representable in f32.
        assert!(
            (theta - 1000000.0f32).abs() < 1.0,
            "rope_theta_f32 should be close to 1000000.0, got {}",
            theta,
        );
    }

    #[test]
    fn test_from_file() {
        // Create a temporary config.json file.
        let dir = std::env::temp_dir().join("qwen3_rs_config_test");
        fs::create_dir_all(&dir).expect("should create temp dir");
        let path = dir.join("config.json");

        let mut file = fs::File::create(&path).expect("should create temp file");
        write!(file, "{}", QWEN3_CONFIG_JSON).expect("should write config");

        let config = ModelConfig::from_file(&path)
            .expect("from_file should succeed with a valid config.json");

        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.head_dim(), 128);

        // Clean up the temp file.
        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir(&dir);
    }

    #[test]
    fn test_extra_json_fields_ignored() {
        // Configs from HuggingFace often contain extra fields like
        // "architectures", "model_type", "bos_token_id", etc.
        // serde should silently ignore them.
        let json = r#"{
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "vocab_size": 151936,
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "intermediate_size": 3072,
            "max_position_embeddings": 40960,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "hidden_act": "silu",
            "tie_word_embeddings": true,
            "head_dim": 128,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.40.0"
        }"#;

        let config = ModelConfig::from_json(json)
            .expect("parsing config with extra fields should succeed");
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.head_dim(), 128);
    }

    #[test]
    fn test_missing_field_returns_error() {
        let json = r#"{
            "vocab_size": 151936,
            "hidden_size": 1024
        }"#;

        let result = ModelConfig::from_json(json);
        assert!(result.is_err(), "parsing config with missing fields should fail");
    }

    #[test]
    fn test_tie_word_embeddings_true() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 512,
            "num_hidden_layers": 6,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "intermediate_size": 1536,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "hidden_act": "gelu",
            "tie_word_embeddings": true
        }"#;

        let config = ModelConfig::from_json(json).unwrap();
        assert!(config.tie_word_embeddings);
        assert_eq!(config.head_dim(), 64); // 512 / 8 (fallback, no explicit head_dim)
        assert_eq!(config.head_dim, None); // not explicitly set
    }
}
