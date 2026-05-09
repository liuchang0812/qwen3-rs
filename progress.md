# Progress Log — qwen3.5-rs

## Session: 2026-05-09

### All Tasks Complete + Bug Fixes Applied

All 29 original tasks complete. Then tested with real Qwen3-0.6B model and found/fixed 5 bugs:

#### Bug Fixes After Real Model Testing

1. **CLI flag conflict**: `-m` used for both `--model-dir` and `--max-tokens` → removed `-m` short for `--max-tokens`
2. **head_dim mismatch**: Qwen3 config has `head_dim=128` explicitly (not `hidden_size/num_heads=64`) → added `head_dim: Option<usize>` to ModelConfig
3. **BF16 weights**: Real model uses BF16, not F32 → added BF16→F32 and F16→F32 conversion in safetensors.rs
4. **Tokenizer merge format**: Qwen3 uses array-format merges `[["a","b"],...]` instead of string-format `["a b",...]` → custom deserializer handles both
5. **Missing q_norm/k_norm**: Qwen3 applies RMSNorm to Q and K per-head before RoPE → added optional q_norm/k_norm to Attention

### Final Stats
- **162 total tests passing**
- **Model runs successfully** on real Qwen3-0.6B weights
- **Speed**: ~0.1 tokens/sec (single-threaded f32 CPU, expected for unoptimized)
- **Correct output**: "1+1=" → "2, 1+1=2" (greedy), ChatML prompts generate coherent text

### Model Files Downloaded
- `model/config.json` — Qwen3-0.6B configuration
- `model/tokenizer.json` — 151,643 vocab, 151,387 merges
- `model/model.safetensors` — 1.5GB BF16 weights

### How to Run
```bash
cargo run --release -- --model-dir ./model --prompt "Hello" --max-tokens 20 --temperature 0.7
# ChatML format for best results:
cargo run --release -- --model-dir ./model --prompt "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n" --max-tokens 50
```
