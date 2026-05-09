# qwen3.5-rs: A Rust Implementation of Qwen3 Inference

> **Purpose**: Educational project to understand how large language models work by
> implementing inference from scratch in Rust, with minimal dependencies.

## 1. Project Overview

This project implements a **decoder-only transformer** inference engine for the
**Qwen3-0.6B** (Qwen3-0.6B) model in pure Rust. The goal is **not** to build a
production-grade inference server, but to create a **readable, well-documented**
codebase that teaches students how LLMs work end-to-end.

### Design Principles

1. **Simplicity over performance** — No CUDA, no quantization, no batching.
   We run one token at a time on CPU with f32 precision.
2. **Minimal dependencies** — Only essential crates: `tokio` (async I/O for
   interactive REPL), `clap` (CLI args), and a handful of small utility crates.
   All math (matrix multiply, softmax, RoPE, etc.) is written from scratch.
3. **Heavy documentation** — Every module has a companion `.md` doc explaining
   *what it is*, *how it works*, and *how we implement it*.
4. **Local file tracing** — A `progress.md` file tracks completed tasks so work
   can resume across sessions.

## 2. Qwen3-0.6B Model Architecture

Qwen3-0.6B (marketed as Qwen3-0.6B) is a **decoder-only transformer** with
Grouped Query Attention (GQA). Its architecture is nearly identical to LLaMA
with some differences in normalization and activation functions.

### 2.1 Model Hyperparameters

| Parameter              | Value        | Description                              |
|------------------------|--------------|------------------------------------------|
| `vocab_size`           | 151,936      | Size of the tokenizer vocabulary         |
| `hidden_size`          | 1,024        | Dimension of hidden representations (d_model) |
| `num_hidden_layers`    | 28           | Number of transformer blocks             |
| `num_attention_heads`  | 16           | Number of query heads (n_heads)          |
| `num_key_value_heads`  | 8            | Number of KV heads (GQA ratio = 2:1)     |
| `head_dim`             | 128          | Dimension per attention head (explicit in config) |
| `intermediate_size`    | 3,072        | FFN hidden dimension (3x hidden_size)    |
| `max_position_embeddings` | 40,960   | Maximum sequence length                  |
| `rms_norm_eps`         | 1e-6         | Epsilon for RMSNorm                      |
| `rope_theta`           | 1,000,000.0  | Base frequency for Rotary Position Embedding |
| `tie_emb_heads`        | true         | Whether output projection shares weights with embedding |

### 2.2 Single Transformer Block

```
Input x (shape: [seq_len, hidden_size])
│
├── RMSNorm(x) ──────────────────┐
│                                 ▼
│                    ┌────────────────────────┐
│                    │  Grouped Query Attn    │
│                    │  (with RoPE & Causal   │
│                    │   Mask & KV Cache)     │
│                    └────────────────────────┘
│                                 │
├──────────────────────────────── + ◄─── Residual Connection
│                                 │
├── RMSNorm(…) ──────────────────┐
│                                 ▼
│                    ┌────────────────────────┐
│                    │  SwiGLU FFN            │
│                    │  gate = SiLU(W_gate·x) │
│                    │  up   = W_up·x         │
│                    │  out  = W_down·(gate*up)│
│                    └────────────────────────┘
│                                 │
├──────────────────────────────── + ◄─── Residual Connection
│
▼
Output (shape: [seq_len, hidden_size])
```

### 2.3 Full Model Forward Pass

```
Token IDs ──► Embedding Layer ──► N × Transformer Block ──► RMSNorm ──► Linear (lm_head) ──► Logits
```

### 2.4 Key Components

| Component               | Purpose                                          |
|-------------------------|--------------------------------------------------|
| **Token Embedding**     | Maps token IDs → dense vectors (vocab_size × hidden_size) |
| **RMSNorm**             | Normalizes activations (simpler than LayerNorm)  |
| **RoPE**                | Encodes position via rotation matrices            |
| **GQA**                 | Shares KV heads across query heads for efficiency |
| **SwiGLU FFN**          | Feed-forward with SiLU gating (better than ReLU)  |
| **Causal Mask**         | Prevents attending to future tokens               |
| **KV Cache**            | Caches past key/value to avoid recomputation      |
| **lm_head**             | Projects hidden states → vocabulary logits        |

## 3. Model File Format

We support loading from **HuggingFace safetensors** format. The model weights
are stored as individual tensors in safetensors files, with a `config.json`
for hyperparameters and a `tokenizer.json` for the tokenizer.

### Expected File Structure

```
model_dir/
├── config.json           # Model hyperparameters
├── tokenizer.json        # Tokenizer vocabulary and merges
├── model.safetensors     # Model weights (single file for 0.6B)
└── generation_config.json  # Generation parameters
```

### Weight Name Mapping

| Weight Name Pattern                                | Shape                         |
|----------------------------------------------------|-------------------------------|
| `model.embed_tokens.weight`                        | [151936, 1024]               |
| `model.layers.{i}.input_layernorm.weight`          | [1024]                       |
| `model.layers.{i}.self_attn.q_proj.weight`         | [2048, 1024]                 |
| `model.layers.{i}.self_attn.k_proj.weight`         | [1024, 1024]                  |
| `model.layers.{i}.self_attn.v_proj.weight`         | [1024, 1024]                  |
| `model.layers.{i}.self_attn.o_proj.weight`         | [1024, 2048]                 |
| `model.layers.{i}.post_attention_layernorm.weight`  | [1024]                       |
| `model.layers.{i}.mlp.gate_proj.weight`            | [3072, 1024]                 |
| `model.layers.{i}.mlp.up_proj.weight`              | [3072, 1024]                 |
| `model.layers.{i}.mlp.down_proj.weight`            | [1024, 3072]                 |
| `model.norm.weight`                                | [1024]                       |
| `lm_head.weight`                                   | [151936, 1024] (tied with embed_tokens) |

## 4. Project Structure

```
qwen3.5-rs/
├── Cargo.toml                  # Project manifest
├── design.md                   # This file — architecture design
├── todo.md                     # Task list and status tracking
├── progress.md                 # Detailed progress log
├── docs/                       # Educational documentation
│   ├── 01_transformer_basics.md
│   ├── 02_tokenizer.md
│   ├── 03_embeddings.md
│   ├── 04_rmsnorm.md
│   ├── 05_rope.md
│   ├── 06_attention.md
│   ├── 07_ffn.md
│   ├── 08_safetensors.md
│   ├── 09_inference.md
│   └── 10_sampling.md
├── src/
│   ├── main.rs                 # CLI entry point
│   ├── lib.rs                  # Module declarations
│   ├── config.rs               # Model configuration (from config.json)
│   ├── tokenizer.rs            # BPE tokenizer (from tokenizer.json)
│   ├── tensor.rs               # Simple N-dimensional tensor with math ops
│   ├── safetensors.rs          # safetensors file format reader
│   ├── model.rs                # Full model: embedding → blocks → lm_head
│   ├── transformer_block.rs    # Single transformer block
│   ├── rmsnorm.rs              # RMSNorm implementation
│   ├── rope.rs                 # Rotary Position Embedding
│   ├── attention.rs            # Grouped Query Attention with KV cache
│   ├── ffn.rs                  # SwiGLU Feed-Forward Network
│   ├── sampling.rs             # Token sampling strategies (greedy, top-k, top-p, temperature)
│   └── inference.rs            # Inference loop and KV cache management
└── tests/
    ├── test_tensor.rs          # Unit tests for tensor operations
    ├── test_rmsnorm.rs         # Unit tests for RMSNorm
    ├── test_rope.rs            # Unit tests for RoPE
    ├── test_attention.rs       # Unit tests for attention
    └── test_tokenizer.rs       # Unit tests for tokenizer
```

## 5. Dependencies (Minimal)

```toml
[dependencies]
clap = { version = "4", features = ["derive"] }   # CLI argument parsing
serde = { version = "1", features = ["derive"] }   # JSON deserialization
serde_json = "1"                                    # JSON parsing (config.json, tokenizer.json)
byteorder = "1"                                     # Endian-aware byte reading (safetensors)
```

That's it. **4 crates** total. All math is implemented from scratch.

### What we do NOT depend on

| Skipped crate         | Why                                     | What we do instead                  |
|-----------------------|-----------------------------------------|-------------------------------------|
| `ndarray` / `tch`     | Too abstract, hides the math            | Custom `Tensor` struct with loops   |
| `candle` / `burn`     | ML frameworks — defeats the purpose     | Hand-written forward pass           |
| `tokenizers`          | HuggingFace's Rust tokenizer (heavy)    | Simple BPE from scratch             |
| `rayon`               | Parallelism (premature for education)   | Single-threaded, clear control flow |

## 6. Inference Flow

```
1. User provides prompt text
2. Tokenizer encodes text → token IDs
3. For each token position:
   a. Embedding lookup → x
   b. For each transformer block:
      i.   RMSNorm → x_norm
      ii.  GQA with RoPE + causal mask + KV cache → attn_out
      iii. Residual add: x = x + attn_out
      iv.  RMSNorm → x_norm
      v.   SwiGLU FFN → ffn_out
      vi.  Residual add: x = x + ffn_out
   c. Final RMSNorm
   d. lm_head projection → logits
   e. Sample next token from logits
   f. Append token to output, update KV cache
4. Tokenizer decodes output token IDs → text
```

## 7. CLI Interface

```bash
# Download model first (outside this tool)
# Place config.json, tokenizer.json, model.safetensors in a directory

# Run inference
cargo run -- --model-dir ./model --prompt "Hello, world!" --max-tokens 100

# Interactive chat mode
cargo run -- --model-dir ./model --interactive

# With sampling parameters
cargo run -- --model-dir ./model --prompt "Explain Rust" --temperature 0.7 --top-k 50 --top-p 0.9
```
