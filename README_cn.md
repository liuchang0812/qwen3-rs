# qwen3.5-rs

*一个用 Rust 从零实现的 Qwen3-0.6B 推理引擎，专为教学而构建。*

本项目为 Qwen3-0.6B 模型实现了一个仅解码器（decoder-only）的 Transformer 推理引擎，完全使用 Rust 编写，依赖极少。每个组件——从张量运算到 BPE 分词——都是从零开始实现的，方便学生完全理解大语言模型的工作原理。

## 为什么选择本项目？

- **教学导向**：每个模块都配有配套文档，解释*它是什么*、*如何工作*以及*我们如何实现它*
- **从零开始**：不使用任何 ML 框架（PyTorch、Candle、Burn）——只用纯 Rust
- **依赖极少**：仅依赖 4 个 crate（clap、serde、serde_json、byteorder）
- **可读性强**：代码清晰，注释详尽，没有晦涩的抽象

## 架构

```
Token IDs → Embedding → N × TransformerBlock → RMSNorm → lm_head → Logits
                         ┌──────────────────────┐
                         │ RMSNorm → Attention   │
                         │   ↓ (residual add)    │
                         │ RMSNorm → FFN (SwiGLU)│
                         │   ↓ (residual add)    │
                         └──────────────────────┘
```

## 项目结构

```
src/
├── main.rs              # CLI 入口
├── config.rs            # 模型配置解析
├── tokenizer.rs         # BPE 分词器
├── tensor.rs            # 从零实现的张量运算
├── safetensors.rs       # 权重文件读取器
├── model.rs             # 完整模型组装
├── transformer_block.rs # 单个 Transformer 块
├── rmsnorm.rs           # RMSNorm 归一化
├── rope.rs              # 旋转位置编码
├── attention.rs         # 分组查询注意力
├── ffn.rs               # SwiGLU 前馈网络
├── sampling.rs          # Token 采样策略
└── inference.rs         # 自回归生成
```

## 快速开始

### 前置条件

- Rust 1.70+（通过 https://rustup.rs 安装）
- 约 3GB RAM 用于加载模型
- Qwen3-0.6B 模型文件（config.json、tokenizer.json、model.safetensors）

### 下载模型

```bash
# 使用 huggingface-cli
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./model

# 或使用 git lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-0.6B ./model
```

### 构建与运行

```bash
# 构建
cargo build --release

# 单轮提示
cargo run --release -- --model-dir ./model --prompt "用一句话解释 Rust"

# 交互式对话
cargo run --release -- --model-dir ./model --interactive

# 带采样参数
cargo run --release -- --model-dir ./model --prompt "你好" --temperature 0.7 --top-k 50 --top-p 0.9

# 贪心解码（确定性输出）
cargo run --release -- --model-dir ./model --prompt "1+1=" --temperature 0
```

## CLI 选项

| 标志 | 默认值 | 说明 |
|------|---------|-------------|
| `--model-dir` | 必填 | 模型目录路径 |
| `--prompt` | - | 输入提示（单轮） |
| `--interactive` | false | 交互式对话模式 |
| `--max-tokens` | 100 | 最大生成 token 数 |
| `--temperature` | 1.0 | 采样温度（0=贪心） |
| `--top-k` | 50 | Top-k 采样 |
| `--top-p` | 0.9 | 核采样阈值 |
| `--seed` | 随机 | 随机种子，用于复现 |

## 文档

深入讲解每个组件的教学文档：

- [01_transformer_basics.md](docs/01_transformer_basics.md) — Transformer 如何处理文本
- [02_tokenizer.md](docs/02_tokenizer.md) — BPE 分词详解
- [03_embeddings.md](docs/03_embeddings.md) — 从词到向量
- [04_rmsnorm.md](docs/04_rmsnorm.md) — 为什么选择 RMSNorm 而非 LayerNorm
- [05_rope.md](docs/05_rope.md) — 旋转位置编码
- [06_attention.md](docs/06_attention.md) — 自注意力与 GQA
- [07_ffn.md](docs/07_ffn.md) — SwiGLU 前馈网络
- [08_safetensors.md](docs/08_safetensors.md) — 权重如何存储
- [09_inference.md](docs/09_inference.md) — 自回归生成
- [10_sampling.md](docs/10_sampling.md) — 采样策略

## Qwen3-0.6B 模型规格

| 参数 | 值 |
|-----------|-------|
| 词表大小 | 151,936 |
| 隐藏层维度 | 1,024 |
| 层数 | 28 |
| 注意力头数 | 16（查询）、8（KV） |
| 头维度 | 128 |
| FFN 中间层维度 | 3,072 |
| 最大上下文长度 | 40,960 |
| 总参数量 | ~0.6B |

## 运行测试

```bash
# 运行所有测试
cargo test

# 运行特定模块测试
cargo test --lib tensor
cargo test --lib attention

# 显示输出运行
cargo test -- --nocapture
```

## 性能说明

- 单线程 CPU 推理，f32 精度
- 在现代 CPU 上约 0.5-2 tokens/秒（取决于模型）
- 本项目未针对速度优化——而是针对可读性优化
- 如需生产级推理，请使用 vLLM、llama.cpp 或 candle

## 局限性

- 仅支持 CPU（无 GPU/CUDA）
- 仅支持 f32（无量化）
- 单次处理一个请求（无批处理）
- 分词器简化（可能与 HuggingFace 不完全一致）
- 无量化支持（尚不支持 f16/bf16 权重）

## 许可证

MIT
