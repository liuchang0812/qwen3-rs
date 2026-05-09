# Task List — qwen3.5-rs

Status: `[ ]` pending | `[~]` in progress | `[x]` done

## Phase 0: Project Setup
- [x] T0: Create design.md
- [x] T0: Create todo.md and progress.md
- [x] T1: Project scaffold — Cargo.toml, src/lib.rs, src/main.rs skeleton
- [x] T2: Install Rust toolchain (user action — documented in progress.md)

## Phase 1: Foundation — Tensor & Math
- [x] T3: Implement `tensor.rs` — N-dimensional tensor with basic ops (matmul, add, mul, reshape, etc.)
- [x] T4: Write `docs/01_transformer_basics.md` — What is a transformer? How does it process text?
- [x] T5: Write `docs/03_embeddings.md` — What are embeddings? How do words become numbers?

## Phase 2: Building Blocks
- [x] T6: Implement `rmsnorm.rs` — RMSNorm normalization
- [x] T7: Write `docs/04_rmsnorm.md` — What is RMSNorm? Why not BatchNorm/LayerNorm?
- [x] T8: Implement `rope.rs` — Rotary Position Embedding
- [x] T9: Write `docs/05_rope.md` — What is RoPE? How does it encode position?
- [x] T10: Implement `attention.rs` — Grouped Query Attention with KV cache
- [x] T11: Write `docs/06_attention.md` — What is attention? GQA vs MHA?
- [x] T12: Implement `ffn.rs` — SwiGLU Feed-Forward Network
- [x] T13: Write `docs/07_ffn.md` — What is an FFN? Why SwiGLU?

## Phase 3: Model Assembly
- [x] T14: Implement `transformer_block.rs` — Wire up attention + FFN + residuals + norms
- [x] T15: Implement `model.rs` — Full model: embedding → blocks → lm_head
- [x] T16: Implement `config.rs` — Parse config.json into Rust struct
- [x] T17: Write `docs/03_embeddings.md` update with code reference

## Phase 4: Loading Weights
- [x] T18: Implement `safetensors.rs` — Read .safetensors file format
- [x] T19: Write `docs/08_safetensors.md` — What is safetensors? How are weights stored?
- [x] T20: Implement weight loading in `model.rs` — Map tensor names to struct fields

## Phase 5: Tokenizer
- [x] T21: Implement `tokenizer.rs` — BPE tokenizer from tokenizer.json
- [x] T22: Write `docs/02_tokenizer.md` — What is BPE? How does tokenization work?

## Phase 6: Inference
- [x] T23: Implement `sampling.rs` — Greedy, temperature, top-k, top-p sampling
- [x] T24: Implement `inference.rs` — Autoregressive generation loop with KV cache
- [x] T25: Write `docs/09_inference.md` — How does autoregressive generation work?
- [x] T26: Write `docs/10_sampling.md` — What are sampling strategies?

## Phase 7: CLI & Integration
- [x] T27: Implement `main.rs` — CLI with clap, interactive mode, prompt mode
- [x] T28: End-to-end test — Run model on a simple prompt and verify output
- [x] T29: Write README.md with build & run instructions

## Notes

- Tasks are done **one at a time** via subagents
- Each code task includes a companion educational doc
- `progress.md` is updated after each task completion
