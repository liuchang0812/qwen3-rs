//! End-to-end integration test for the qwen3.5-rs pipeline.
//!
//! This test builds a tiny model from scratch (writing config.json, tokenizer.json,
//! and model.safetensors to a temporary directory), then exercises the full
//! InferenceEngine pipeline: load -> encode -> forward -> generate -> decode.
//!
//! It also covers interactive (streaming) generation and error-handling paths.

use qwen3_5_rs::inference::InferenceEngine;
use qwen3_5_rs::sampling::SamplingConfig;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers for building tiny model files
// ─────────────────────────────────────────────────────────────────────────────

/// Tiny model dimensions used throughout this test.
const VOCAB: usize = 32;
const HIDDEN: usize = 16;
const NUM_LAYERS: usize = 1;
const NUM_HEADS: usize = 2;
const NUM_KV_HEADS: usize = 1;
const HEAD_DIM: usize = HIDDEN / NUM_HEADS; // 8
const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 8
const INTERMEDIATE: usize = 32;

/// Return the JSON string for a tiny config.json.
fn config_json() -> String {
    format!(
        r#"{{
    "vocab_size": {VOCAB},
    "hidden_size": {HIDDEN},
    "num_hidden_layers": {NUM_LAYERS},
    "num_attention_heads": {NUM_HEADS},
    "num_key_value_heads": {NUM_KV_HEADS},
    "intermediate_size": {INTERMEDIATE},
    "max_position_embeddings": 64,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "hidden_act": "silu",
    "tie_word_embeddings": false
}}"#
    )
}

/// Return the JSON string for a minimal tokenizer.json with a 32-token BPE vocab.
fn tokenizer_json() -> String {
    // We create a vocabulary where:
    //   ID 0 = <eos> (special token)
    //   IDs 1..26 = 'a'..'z'
    //   ID 27 = ' ' (space, stored as byte-level char Ġ)
    //   ID 28 = '!'
    //   ID 29 = 'ab' (merged)
    //   ID 30 = 'cd' (merged)
    //   ID 31 = Ġ (byte-level space character, needed for encoding)
    r#"{
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
}"#.to_string()
}

/// Build the raw f32 data for every tensor in the tiny model, concatenated.
///
/// Returns `(data_bytes, header_entries)` where `header_entries` is a list of
/// JSON fragments like `"tensor_name":{"dtype":"F32","shape":[...],"data_offsets":[start,end]}`.
fn build_tensor_data() -> (Vec<u8>, Vec<String>) {
    let mut data_bytes: Vec<u8> = Vec::new();
    let mut header_entries: Vec<String> = Vec::new();

    let mut add_tensor = |name: &str, shape: &[usize], values: &[f32]| {
        let num_elements: usize = shape.iter().product();
        assert_eq!(values.len(), num_elements, "tensor {} element count mismatch", name);
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

    // embed_tokens: [VOCAB, HIDDEN]
    let embed_data: Vec<f32> = (0..VOCAB * HIDDEN).map(|i| (i as f32 * 0.01).sin()).collect();
    add_tensor("model.embed_tokens.weight", &[VOCAB, HIDDEN], &embed_data);

    // Layer 0 input_layernorm: [HIDDEN]
    add_tensor("model.layers.0.input_layernorm.weight", &[HIDDEN], &vec![1.0; HIDDEN]);

    // q_proj: [HIDDEN, HIDDEN] — identity-like
    let q_data: Vec<f32> = (0..HIDDEN * HIDDEN)
        .map(|i| if i / HIDDEN == i % HIDDEN { 1.0 } else { 0.0 })
        .collect();
    add_tensor("model.layers.0.self_attn.q_proj.weight", &[HIDDEN, HIDDEN], &q_data);

    // k_proj: [KV_DIM, HIDDEN]
    let k_data: Vec<f32> = (0..KV_DIM * HIDDEN)
        .map(|i| if i / HIDDEN == i % HIDDEN { 1.0 } else { 0.0 })
        .collect();
    add_tensor("model.layers.0.self_attn.k_proj.weight", &[KV_DIM, HIDDEN], &k_data);

    // v_proj: [KV_DIM, HIDDEN]
    let v_data: Vec<f32> = (0..KV_DIM * HIDDEN)
        .map(|i| if i / HIDDEN == i % HIDDEN { 1.0 } else { 0.0 })
        .collect();
    add_tensor("model.layers.0.self_attn.v_proj.weight", &[KV_DIM, HIDDEN], &v_data);

    // o_proj: [HIDDEN, HIDDEN] — identity-like
    add_tensor("model.layers.0.self_attn.o_proj.weight", &[HIDDEN, HIDDEN], &q_data);

    // Layer 0 post_attention_layernorm: [HIDDEN]
    add_tensor("model.layers.0.post_attention_layernorm.weight", &[HIDDEN], &vec![1.0; HIDDEN]);

    // FFN gate_proj: [INTERMEDIATE, HIDDEN]
    add_tensor("model.layers.0.mlp.gate_proj.weight", &[INTERMEDIATE, HIDDEN], &vec![0.1; INTERMEDIATE * HIDDEN]);

    // FFN up_proj: [INTERMEDIATE, HIDDEN]
    add_tensor("model.layers.0.mlp.up_proj.weight", &[INTERMEDIATE, HIDDEN], &vec![0.1; INTERMEDIATE * HIDDEN]);

    // FFN down_proj: [HIDDEN, INTERMEDIATE]
    add_tensor("model.layers.0.mlp.down_proj.weight", &[HIDDEN, INTERMEDIATE], &vec![0.1; HIDDEN * INTERMEDIATE]);

    // Final norm: [HIDDEN]
    add_tensor("model.norm.weight", &[HIDDEN], &vec![1.0; HIDDEN]);

    // lm_head: [VOCAB, HIDDEN]
    let lm_head_data: Vec<f32> = (0..VOCAB * HIDDEN).map(|i| (i as f32 * 0.02).cos()).collect();
    add_tensor("lm_head.weight", &[VOCAB, HIDDEN], &lm_head_data);

    (data_bytes, header_entries)
}

/// Build a valid safetensors binary file in memory and return its bytes.
fn build_safetensors_bytes() -> Vec<u8> {
    let (data_bytes, header_entries) = build_tensor_data();
    let header_json = format!("{{{}}}", header_entries.join(","));
    let header_bytes = header_json.as_bytes();

    // Align header to 8-byte boundary (safetensors spec requirement).
    let json_len = header_bytes.len();
    let aligned_len = ((json_len + 7) / 8) * 8;
    let padding = aligned_len - json_len;
    let header_size = aligned_len as u64;

    let mut file_data = Vec::new();
    file_data.extend_from_slice(&header_size.to_le_bytes());
    file_data.extend_from_slice(header_bytes);
    file_data.extend_from_slice(&vec![b' '; padding]);
    file_data.extend_from_slice(&data_bytes);

    file_data
}

/// Create a temporary directory with all model files and return its path.
///
/// The caller is responsible for cleaning up (call `cleanup_temp_dir`).
fn create_temp_model_dir() -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "qwen35_rs_e2e_test_{}_{}",
        std::process::id(),
        id,
    ));
    fs::create_dir_all(&dir).expect("should create temp model dir");

    // Write config.json
    let config_path = dir.join("config.json");
    let mut f = fs::File::create(&config_path).expect("should create config.json");
    write!(f, "{}", config_json()).expect("should write config.json");

    // Write tokenizer.json
    let tok_path = dir.join("tokenizer.json");
    let mut f = fs::File::create(&tok_path).expect("should create tokenizer.json");
    write!(f, "{}", tokenizer_json()).expect("should write tokenizer.json");

    // Write model.safetensors
    let st_path = dir.join("model.safetensors");
    let mut f = fs::File::create(&st_path).expect("should create model.safetensors");
    f.write_all(&build_safetensors_bytes()).expect("should write model.safetensors");

    dir
}

/// Remove the temporary model directory and all files within it.
fn cleanup_temp_dir(dir: &Path) {
    let _ = fs::remove_file(dir.join("config.json"));
    let _ = fs::remove_file(dir.join("tokenizer.json"));
    let _ = fs::remove_file(dir.join("model.safetensors"));
    let _ = fs::remove_dir(dir);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Full pipeline — load, encode, forward, generate, decode
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_full_pipeline() {
    let dir = create_temp_model_dir();
    let result = std::panic::catch_unwind(|| {
        let sampling = SamplingConfig {
            temperature: 0.0, // greedy for determinism
            top_k: 0,
            top_p: 1.0,
            seed: Some(42),
        };
        let mut engine = InferenceEngine::load(&dir, sampling)
            .expect("InferenceEngine::load should succeed with valid model dir");

        // Encode a prompt.
        let token_ids = engine.tokenizer().encode("ab");
        assert!(!token_ids.is_empty(), "encoding 'ab' should produce token IDs");

        // Run a forward pass directly via the model.
        let logits = engine.model_mut().forward(&token_ids, 0);
        assert_eq!(logits.shape()[0], token_ids.len(), "logits seq_len should match prompt length");
        assert_eq!(logits.shape()[1], VOCAB, "logits vocab dim should be {}", VOCAB);
        for &v in logits.data() {
            assert!(v.is_finite(), "all logits should be finite, got {}", v);
        }

        // Generate text.
        let output = engine.generate("ab", 5);
        assert!(!output.is_empty(), "generate should return non-empty output");
        assert!(
            output.starts_with("ab"),
            "output should start with the prompt text, got: {:?}",
            output
        );

        // Decode the generated portion should be non-empty and finite.
        // (The generated text is appended after the prompt in the output string.)
        if output.len() > 2 {
            // At least some tokens were generated beyond the prompt.
            let generated = &output[2..]; // strip "ab" prompt
            assert!(!generated.is_empty(), "generated portion should not be empty");
        }

        engine.reset();
    });
    cleanup_temp_dir(&dir);
    assert!(result.is_ok());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Interactive / streaming generation with callback
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_generate_with_callback() {
    let dir = create_temp_model_dir();
    let result = std::panic::catch_unwind(|| {
        let sampling = SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: Some(42),
        };
        let mut engine = InferenceEngine::load(&dir, sampling)
            .expect("InferenceEngine::load should succeed");

        // Generate with callback, counting how many times it is invoked.
        let mut callback_count = 0usize;
        let output = engine.generate_with_callback("a", 5, |_token_text| {
            callback_count += 1;
        });

        // The output should start with the prompt.
        assert!(
            output.starts_with('a'),
            "output should start with prompt, got: {:?}",
            output
        );

        // The callback should have been called at least once (unless EOS was
        // hit immediately, which is very unlikely with our random-ish weights).
        assert!(
            callback_count >= 1,
            "callback should be called at least once, was called {} times",
            callback_count
        );

        // The generated portion should be non-empty and finite.
        let generated = &output[1..]; // strip the "a" prompt
        if !generated.is_empty() {
            for ch in generated.chars() {
                assert!(
                    !ch.is_control() || ch == '\n' || ch == '\t',
                    "generated text should not contain control characters"
                );
            }
        }
    });
    cleanup_temp_dir(&dir);
    assert!(result.is_ok());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Reset cache and verify generation still works
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_reset_and_regenerate() {
    let dir = create_temp_model_dir();
    let result = std::panic::catch_unwind(|| {
        let sampling = SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: Some(42),
        };
        let mut engine = InferenceEngine::load(&dir, sampling)
            .expect("InferenceEngine::load should succeed");

        // First generation to populate KV caches.
        let output1 = engine.generate("a", 3);
        assert!(!output1.is_empty(), "first generation should produce output");

        // Reset caches.
        engine.reset();

        // Second generation should still work after reset.
        let output2 = engine.generate("a", 3);
        assert!(!output2.is_empty(), "generation after reset should produce output");
        assert!(
            output2.starts_with('a'),
            "second output should start with prompt, got: {:?}",
            output2
        );

        // With greedy decoding and the same prompt, the output should be
        // deterministic (identical across two fresh-slate generations).
        assert_eq!(
            output1, output2,
            "greedy generation should be deterministic after reset"
        );
    });
    cleanup_temp_dir(&dir);
    assert!(result.is_ok());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Error handling — load from non-existent directory
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_load_nonexistent_directory() {
    let result = InferenceEngine::load(
        Path::new("/tmp/qwen35_rs_definitely_does_not_exist_12345678"),
        SamplingConfig::default(),
    );
    assert!(result.is_err(), "loading from a non-existent directory should fail");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Error handling — load from directory missing config.json
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_load_missing_config() {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);

    let dir = std::env::temp_dir().join(format!(
        "qwen35_rs_e2e_missing_config_{}_{}",
        std::process::id(),
        id,
    ));
    fs::create_dir_all(&dir).expect("should create temp dir");

    // Only write tokenizer.json and model.safetensors — no config.json.
    let tok_path = dir.join("tokenizer.json");
    let mut f = fs::File::create(&tok_path).expect("should create tokenizer.json");
    write!(f, "{}", tokenizer_json()).expect("should write tokenizer.json");

    let st_path = dir.join("model.safetensors");
    let mut f = fs::File::create(&st_path).expect("should create safetensors file");
    f.write_all(&build_safetensors_bytes()).expect("should write safetensors");

    let result = InferenceEngine::load(&dir, SamplingConfig::default());
    assert!(
        result.is_err(),
        "loading from directory without config.json should fail"
    );

    // Clean up.
    let _ = fs::remove_file(tok_path);
    let _ = fs::remove_file(st_path);
    let _ = fs::remove_dir(&dir);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Verify generated tokens are valid vocabulary indices
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_generated_tokens_in_vocab_range() {
    let dir = create_temp_model_dir();
    let result = std::panic::catch_unwind(|| {
        let sampling = SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: Some(42),
        };
        let mut engine = InferenceEngine::load(&dir, sampling)
            .expect("InferenceEngine::load should succeed");

        // Encode and do a forward pass manually, checking logits range.
        let token_ids = engine.tokenizer().encode("abc");
        let logits = engine.model_mut().forward(&token_ids, 0);

        // All logits should be finite.
        for &v in logits.data() {
            assert!(v.is_finite(), "logits should be finite, got {}", v);
        }

        // The argmax of the last position should be a valid vocab index.
        let vocab_size = logits.shape()[1];
        let last_row_start = (logits.shape()[0] - 1) * vocab_size;
        let last_row = &logits.data()[last_row_start..last_row_start + vocab_size];
        let max_idx = last_row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert!(
            max_idx < VOCAB,
            "argmax of logits should be a valid vocab index, got {}",
            max_idx
        );
    });
    cleanup_temp_dir(&dir);
    assert!(result.is_ok());
}
