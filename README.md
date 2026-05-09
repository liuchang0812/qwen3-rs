# qwen3.5-rs

*A from-scratch implementation of Qwen3-0.6B inference in Rust, built for education.*

This project implements a decoder-only transformer inference engine for the Qwen3-0.6B model, written entirely in Rust with minimal dependencies. Every component — from tensor math to BPE tokenization — is implemented from scratch so students can understand exactly how large language models work.

## Why This Project?

- **Educational**: Every module has a companion doc explaining *what it is*, *how it works*, and *how we implement it*
- **From scratch**: No ML frameworks (PyTorch, Candle, Burn) — just plain Rust
- **Minimal dependencies**: Only 4 crates (clap, serde, serde_json, byteorder)
- **Readable**: Clear code with extensive comments, no clever abstractions

## Architecture

```
Token IDs → Embedding → N × TransformerBlock → RMSNorm → lm_head → Logits
                         ┌──────────────────────┐
                         │ RMSNorm → Attention   │
                         │   ↓ (residual add)    │
                         │ RMSNorm → FFN (SwiGLU)│
                         │   ↓ (residual add)    │
                         └──────────────────────┘
```

## Project Structure

```
src/
├── main.rs              # CLI entry point
├── config.rs            # Model config parsing
├── tokenizer.rs         # BPE tokenizer
├── tensor.rs            # Tensor math from scratch
├── safetensors.rs       # Weight file reader
├── model.rs             # Full model assembly
├── transformer_block.rs # Single transformer block
├── rmsnorm.rs           # RMSNorm normalization
├── rope.rs              # Rotary position embedding
├── attention.rs         # Grouped query attention
├── ffn.rs               # SwiGLU feed-forward
├── sampling.rs          # Token sampling strategies
└── inference.rs         # Autoregressive generation
```

## Getting Started

### Prerequisites

- Rust 1.70+ (install via https://rustup.rs)
- ~3GB RAM for loading the model
- Qwen3-0.6B model files (config.json, tokenizer.json, model.safetensors)

### Download Model

```bash
# Using huggingface-cli
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./model

# Or using git lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-0.6B ./model
```

### Build and Run

```bash
# Build
cargo build --release

# Single prompt
cargo run --release -- --model-dir ./model --prompt "Explain Rust in one sentence"

# Interactive chat
cargo run --release -- --model-dir ./model --interactive

# With sampling parameters
cargo run --release -- --model-dir ./model --prompt "Hello" --temperature 0.7 --top-k 50 --top-p 0.9

# Greedy decoding (deterministic)
cargo run --release -- --model-dir ./model --prompt "1+1=" --temperature 0
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-dir` | required | Path to model directory |
| `--prompt` | - | Input prompt (single-turn) |
| `--interactive` | false | Interactive chat mode |
| `--max-tokens` | 100 | Max tokens to generate |
| `--temperature` | 1.0 | Sampling temperature (0=greedy) |
| `--top-k` | 50 | Top-k sampling |
| `--top-p` | 0.9 | Nucleus sampling threshold |
| `--seed` | random | Random seed for reproducibility |

## Documentation

Educational docs explaining each component in depth:

- [01_transformer_basics.md](docs/01_transformer_basics.md) — How transformers process text
- [02_tokenizer.md](docs/02_tokenizer.md) — BPE tokenization explained
- [03_embeddings.md](docs/03_embeddings.md) — From words to vectors
- [04_rmsnorm.md](docs/04_rmsnorm.md) — Why RMSNorm over LayerNorm
- [05_rope.md](docs/05_rope.md) — Rotary position embeddings
- [06_attention.md](docs/06_attention.md) — Self-attention and GQA
- [07_ffn.md](docs/07_ffn.md) — SwiGLU feed-forward networks
- [08_safetensors.md](docs/08_safetensors.md) — How weights are stored
- [09_inference.md](docs/09_inference.md) — Autoregressive generation
- [10_sampling.md](docs/10_sampling.md) — Sampling strategies

## Qwen3-0.6B Model Specs

| Parameter | Value |
|-----------|-------|
| Vocab size | 151,936 |
| Hidden size | 1,024 |
| Layers | 28 |
| Attention heads | 16 (query), 8 (KV) |
| Head dim | 128 |
| FFN intermediate | 3,072 |
| Max context | 40,960 |
| Total params | ~0.6B |

## Running Tests

```bash
# Run all tests
cargo test

# Run specific module tests
cargo test --lib tensor
cargo test --lib attention

# Run with output
cargo test -- --nocapture
```

## Performance Notes

- Single-threaded CPU inference at f32 precision
- ~0.5-2 tokens/second on a modern CPU (model-dependent)
- This is NOT optimized for speed — it's optimized for readability
- For production inference, use vLLM, llama.cpp, or candle

## Limitations

- CPU only (no GPU/CUDA)
- f32 only (no quantization)
- Single request at a time (no batching)
- Simplified tokenizer (may not exactly match HuggingFace)
- No quantization (f16/bf16 weights not supported yet)

## License

MIT
