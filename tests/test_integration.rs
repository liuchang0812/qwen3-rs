//! Integration test for the qwen3.5-rs model pipeline without file I/O.
//!
//! This test constructs a tiny model programmatically (using QwenModel::new
//! with in-memory tensors) and exercises the full forward/generate pipeline.
//! It covers model construction, forward passes, generation, KV cache
//! management, and sampling behavior — all without touching the filesystem.

use qwen3_5_rs::config::ModelConfig;
use qwen3_5_rs::inference::InferenceEngine;
use qwen3_5_rs::model::QwenModel;
use qwen3_5_rs::rmsnorm::RMSNorm;
use qwen3_5_rs::sampling::{sample, sample_greedy, SamplingConfig};
use qwen3_5_rs::tensor::Tensor;
use qwen3_5_rs::tokenizer::Tokenizer;
use qwen3_5_rs::transformer_block::TransformerBlock;

/// Tiny model dimensions used throughout this test.
const VOCAB: usize = 32;
const HIDDEN: usize = 16;
const NUM_LAYERS: usize = 1;
const NUM_HEADS: usize = 2;
const NUM_KV_HEADS: usize = 1;
const HEAD_DIM: usize = 8;
const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM;
const INTERMEDIATE: usize = 32;

/// Build a tiny QwenModel in memory (same pattern as model.rs unit tests).
fn make_test_model() -> QwenModel {
    let config = ModelConfig {
        vocab_size: VOCAB,
        hidden_size: HIDDEN,
        num_hidden_layers: NUM_LAYERS,
        num_attention_heads: NUM_HEADS,
        num_key_value_heads: NUM_KV_HEADS,
        intermediate_size: INTERMEDIATE,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        hidden_act: "silu".to_string(),
        tie_word_embeddings: false,
        head_dim: None,
    };

    let embed_tokens = Tensor::new(
        vec![VOCAB, HIDDEN],
        (0..VOCAB * HIDDEN).map(|i| (i as f32 * 0.01).sin()).collect(),
    );

    let ln1_weight = Tensor::ones(vec![HIDDEN]);
    let ln2_weight = Tensor::ones(vec![HIDDEN]);

    let q_proj = Tensor::new(
        vec![HIDDEN, HIDDEN],
        (0..HIDDEN * HIDDEN)
            .map(|i| if i / HIDDEN == i % HIDDEN { 1.0 } else { 0.0 })
            .collect(),
    );
    let k_proj = Tensor::new(
        vec![KV_DIM, HIDDEN],
        (0..KV_DIM * HIDDEN)
            .map(|i| if i / HIDDEN == i % HIDDEN { 1.0 } else { 0.0 })
            .collect(),
    );
    let v_proj = Tensor::new(
        vec![KV_DIM, HIDDEN],
        (0..KV_DIM * HIDDEN)
            .map(|i| if i / HIDDEN == i % HIDDEN { 1.0 } else { 0.0 })
            .collect(),
    );
    let o_proj = Tensor::new(
        vec![HIDDEN, HIDDEN],
        (0..HIDDEN * HIDDEN)
            .map(|i| if i / HIDDEN == i % HIDDEN { 1.0 } else { 0.0 })
            .collect(),
    );

    let gate_proj = Tensor::new(vec![INTERMEDIATE, HIDDEN], vec![0.1; INTERMEDIATE * HIDDEN]);
    let up_proj = Tensor::new(vec![INTERMEDIATE, HIDDEN], vec![0.1; INTERMEDIATE * HIDDEN]);
    let down_proj = Tensor::new(vec![HIDDEN, INTERMEDIATE], vec![0.1; HIDDEN * INTERMEDIATE]);

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
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        config.max_position_embeddings,
        config.rope_theta_f32(),
        config.eps_f32(),
    );

    let norm_weight = Tensor::ones(vec![HIDDEN]);
    let norm = RMSNorm::new(norm_weight, config.eps_f32());

    let lm_head = Tensor::new(
        vec![VOCAB, HIDDEN],
        (0..VOCAB * HIDDEN).map(|i| (i as f32 * 0.02).cos()).collect(),
    );

    QwenModel::new(embed_tokens, vec![block], norm, lm_head, config)
}

/// Build a minimal Tokenizer in memory.
fn make_test_tokenizer() -> Tokenizer {
    let json = r#"{
        "version": "1.0",
        "added_tokens": [
            {"id": 0, "content": "<eos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "model": {
            "type": "BPE",
            "vocab": {
                "<eos>": 0,
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
                "Ġ": 27,
                "!": 28,
                "ab": 29,
                "cd": 30,
                "Ġa": 31
            },
            "merges": [
                "a b",
                "c d",
                "Ġ a"
            ]
        }
    }"#;
    Tokenizer::from_json(json).unwrap()
}

/// Build a tiny InferenceEngine in memory (no file I/O).
fn make_test_engine() -> InferenceEngine {
    make_test_engine_with_config(SamplingConfig::default())
}

fn make_test_engine_with_config(sampling_config: SamplingConfig) -> InferenceEngine {
    InferenceEngine::new(
        make_test_model(),
        make_test_tokenizer(),
        sampling_config,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Model forward pass tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_model_forward_single_token() {
    let mut model = make_test_model();
    let logits = model.forward(&[5], 0);
    assert_eq!(logits.shape(), &[1, VOCAB]);
    for &v in logits.data() {
        assert!(v.is_finite(), "logit should be finite, got {}", v);
    }
}

#[test]
fn test_model_forward_multiple_tokens() {
    let mut model = make_test_model();
    let logits = model.forward(&[1, 2, 3], 0);
    assert_eq!(logits.shape(), &[3, VOCAB]);
    for &v in logits.data() {
        assert!(v.is_finite(), "logit should be finite, got {}", v);
    }
}

#[test]
fn test_model_prefill_then_decode() {
    let mut model = make_test_model();

    // Prefill: 3 tokens
    let logits = model.forward(&[0, 5, 10], 0);
    assert_eq!(logits.shape(), &[3, VOCAB]);

    // Decode: 1 more token
    let logits = model.forward(&[15], 3);
    assert_eq!(logits.shape(), &[1, VOCAB]);

    for &v in logits.data() {
        assert!(v.is_finite(), "decode logit should be finite, got {}", v);
    }
}

#[test]
fn test_model_different_tokens_different_logits() {
    let mut model1 = make_test_model();
    let mut model2 = make_test_model();

    let logits1 = model1.forward(&[0], 0);
    let logits2 = model2.forward(&[10], 0);

    let any_different = logits1
        .data()
        .iter()
        .zip(logits2.data().iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);

    assert!(any_different, "different token IDs should produce different logits");
}

// ─────────────────────────────────────────────────────────────────────────────
// KV cache management tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_kv_cache_populated_after_forward() {
    let mut model = make_test_model();
    let _ = model.forward(&[1, 2, 3], 0);

    for cache in model.kv_caches() {
        assert!(cache.key_cache.is_some(), "key cache should be populated after forward");
        assert!(cache.value_cache.is_some(), "value cache should be populated after forward");
        assert_eq!(cache.key_cache.as_ref().unwrap().shape()[0], 3);
    }
}

#[test]
fn test_kv_cache_cleared_after_reset() {
    let mut model = make_test_model();
    let _ = model.forward(&[1, 2], 0);
    model.reset_caches();

    for cache in model.kv_caches() {
        assert!(cache.key_cache.is_none(), "key cache should be None after reset");
        assert!(cache.value_cache.is_none(), "value cache should be None after reset");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// InferenceEngine generation tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_engine_generate_nonempty() {
    let mut engine = make_test_engine();
    let output = engine.generate("a", 5);
    assert!(!output.is_empty(), "generate should produce non-empty output");
    assert!(output.starts_with('a'), "output should start with prompt");
}

#[test]
fn test_engine_generate_greedy_deterministic() {
    let config = SamplingConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        seed: None,
    };
    let mut engine1 = make_test_engine_with_config(config.clone());
    let mut engine2 = make_test_engine_with_config(config);

    let out1 = engine1.generate("abc", 5);
    let out2 = engine2.generate("abc", 5);
    assert_eq!(out1, out2, "greedy generation should be deterministic");
}

#[test]
fn test_engine_generate_with_callback() {
    let config = SamplingConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        seed: Some(42),
    };
    let mut engine = make_test_engine_with_config(config);

    let mut callback_count = 0;
    let output = engine.generate_with_callback("ab", 3, |_token_text| {
        callback_count += 1;
    });

    assert!(output.starts_with("ab"), "output should start with prompt");
    assert!(
        callback_count >= 1,
        "callback should be called at least once, was called {} times",
        callback_count
    );
}

#[test]
fn test_engine_reset_then_generate() {
    let config = SamplingConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        seed: Some(99),
    };
    let mut engine = make_test_engine_with_config(config);

    let out1 = engine.generate("a", 3);
    engine.reset();
    let out2 = engine.generate("a", 3);

    assert_eq!(out1, out2, "generation after reset should match fresh generation");
}

// ─────────────────────────────────────────────────────────────────────────────
// Sampling integration tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_sample_from_model_logits() {
    let mut model = make_test_model();
    let logits = model.forward(&[1], 0);
    let last_row = logits.data();

    // Greedy sample should return a valid vocab index.
    let greedy_token = sample_greedy(last_row);
    assert!(greedy_token < VOCAB, "greedy token should be a valid vocab index");

    // Stochastic sample should also return a valid vocab index.
    let config = SamplingConfig {
        temperature: 0.8,
        top_k: 10,
        top_p: 0.9,
        seed: Some(42),
    };
    let sampled_token = sample(last_row, &config);
    assert!(sampled_token < VOCAB, "sampled token should be a valid vocab index");
}

#[test]
fn test_sample_deterministic_with_seed() {
    let mut model = make_test_model();
    let logits = model.forward(&[1], 0);
    let last_row = logits.data().to_vec();

    let config1 = SamplingConfig {
        temperature: 0.7,
        top_k: 5,
        top_p: 0.9,
        seed: Some(12345),
    };
    let config2 = SamplingConfig {
        temperature: 0.7,
        top_k: 5,
        top_p: 0.9,
        seed: Some(12345),
    };

    let t1 = sample(&last_row, &config1);
    let t2 = sample(&last_row, &config2);
    assert_eq!(t1, t2, "same seed should produce same sampled token");
}

// ─────────────────────────────────────────────────────────────────────────────
// Tokenizer roundtrip tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_tokenizer_encode_decode_roundtrip() {
    let tokenizer = make_test_tokenizer();

    let texts = vec!["a", "ab", "cd", "abc"];
    for text in texts {
        let ids = tokenizer.encode(text);
        assert!(!ids.is_empty(), "encoding '{}' should produce token IDs", text);
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, text, "roundtrip failed for '{}': got '{:?}'", text, decoded);
    }
}

#[test]
fn test_tokenizer_eos_token_id() {
    let tokenizer = make_test_tokenizer();
    assert_eq!(tokenizer.eos_token_id(), 0, "EOS token ID should be 0");
}

// ─────────────────────────────────────────────────────────────────────────────
// Config tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_model_config_values() {
    let model = make_test_model();
    assert_eq!(model.vocab_size(), VOCAB);
    assert_eq!(model.kv_caches().len(), NUM_LAYERS);
}
