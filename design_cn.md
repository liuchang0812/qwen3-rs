# qwen3.5-rs：用 Rust 实现的 Qwen3 推理引擎

> **目的**：一个教育项目，通过用 Rust 从零实现推理过程，帮助理解大语言模型的工作原理，依赖项尽可能少。

## 1. 项目概述

本项目用纯 Rust 实现了一个针对 **Qwen3-0.6B** 模型的**仅解码器 Transformer** 推理引擎。目标**不是**构建一个生产级的推理服务器，而是创建一个**可读性好、文档完善**的代码库，用于教学端到端 LLM 工作原理。

### 设计原则

1. **简单优于性能** — 不使用 CUDA、量化或批处理。在 CPU 上以 f32 精度逐 token 运行。
2. **依赖最小化** — 仅使用必要的 crate：`tokio`（交互式 REPL 的异步 I/O）、`clap`（CLI 参数），以及少量小型工具 crate。所有数学运算（矩阵乘法、softmax、RoPE 等）均从零编写。
3. **详尽的文档** — 每个模块都有配套的 `.md` 文档，解释*它是什么*、*如何工作*以及*我们如何实现它*。
4. **本地文件追踪** — 使用 `progress.md` 文件记录已完成的任务，以便跨会话恢复工作。

## 2. Qwen3-0.6B 模型架构

Qwen3-0.6B（市场名称为 Qwen3-0.6B）是一个采用**分组查询注意力（GQA）**的**仅解码器 Transformer**。其架构与 LLaMA 几乎相同，仅在归一化和激活函数方面存在一些差异。

### 2.1 模型超参数

| 参数                    | 值           | 描述                                      |
|-------------------------|--------------|------------------------------------------|
| `vocab_size`            | 151,936      | Tokenizer 词表大小                       |
| `hidden_size`           | 1,024        | 隐藏表示的维度（d_model）                |
| `num_hidden_layers`     | 28           | Transformer 块的数量                      |
| `num_attention_heads`   | 16           | 查询头数量（n_heads）                    |
| `num_key_value_heads`   | 8            | KV 头数量（GQA 比例 = 2:1）             |
| `head_dim`              | 128          | 每个注意力头的维度（配置中显式指定）      |
| `intermediate_size`     | 3,072        | FFN 隐藏维度（3 × hidden_size）          |
| `max_position_embeddings` | 40,960     | 最大序列长度                              |
| `rms_norm_eps`          | 1e-6         | RMSNorm 的 epsilon 值                     |
| `rope_theta`            | 1,000,000.0  | 旋转位置编码（RoPE）的基频                |
| `tie_emb_heads`         | true         | 输出投影是否与嵌入层共享权重              |

### 2.2 单个 Transformer 块

```
输入 x（形状：[seq_len, hidden_size]）
│
├── RMSNorm(x) ──────────────────┐
│                                 ▼
│                    ┌────────────────────────┐
│                    │  Grouped Query Attn    │
│                    │  (with RoPE & Causal   │
│                    │   Mask & KV Cache)     │
│                    └────────────────────────┘
│                                 │
├──────────────────────────────── + ◄─── 残差连接
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
├──────────────────────────────── + ◄─── 残差连接
│
▼
输出（形状：[seq_len, hidden_size]）
```

### 2.3 完整模型前向传播

```
Token IDs ──► Embedding Layer ──► N × Transformer Block ──► RMSNorm ──► Linear (lm_head) ──► Logits
```

### 2.4 关键组件

| 组件                    | 用途                                          |
|-------------------------|------------------------------------------------|
| **Token Embedding**     | 将 token ID 映射为稠密向量（vocab_size × hidden_size） |
| **RMSNorm**             | 激活归一化（比 LayerNorm 更简单）             |
| **RoPE**                | 通过旋转矩阵编码位置信息                       |
| **GQA**                 | 在查询头之间共享 KV 头以提升效率               |
| **SwiGLU FFN**          | 带 SiLU 门控的前馈网络（优于 ReLU）            |
| **Causal Mask**         | 防止注意到未来 token                           |
| **KV Cache**            | 缓存历史 key/value 以避免重复计算              |
| **lm_head**             | 将隐藏状态投影为词表 logits                    |

## 3. 模型文件格式

我们支持从 **HuggingFace safetensors** 格式加载模型。模型权重以独立张量的形式存储在 safetensors 文件中，同时配有 `config.json`（超参数）和 `tokenizer.json`（分词器）。

### 预期文件结构

```
model_dir/
├── config.json           # 模型超参数
├── tokenizer.json        # 分词器词表和合并规则
├── model.safetensors     # 模型权重（0.6B 为单文件）
└── generation_config.json  # 生成参数
```

### 权重名称映射

| 权重名称模式                                    | 形状                         |
|------------------------------------------------|-------------------------------|
| `model.embed_tokens.weight`                    | [151936, 1024]               |
| `model.layers.{i}.input_layernorm.weight`      | [1024]                       |
| `model.layers.{i}.self_attn.q_proj.weight`     | [2048, 1024]                 |
| `model.layers.{i}.self_attn.k_proj.weight`      | [1024, 1024]                  |
| `model.layers.{i}.self_attn.v_proj.weight`      | [1024, 1024]                  |
| `model.layers.{i}.self_attn.o_proj.weight`      | [1024, 2048]                 |
| `model.layers.{i}.post_attention_layernorm.weight` | [1024]                       |
| `model.layers.{i}.mlp.gate_proj.weight`        | [3072, 1024]                 |
| `model.layers.{i}.mlp.up_proj.weight`          | [3072, 1024]                 |
| `model.layers.{i}.mlp.down_proj.weight`        | [1024, 3072]                 |
| `model.norm.weight`                            | [1024]                       |
| `lm_head.weight`                               | [151936, 1024]（与 embed_tokens 绑定） |
```

## 4. 项目结构

```
qwen3.5-rs/
├── Cargo.toml                  # 项目清单
├── design.md                   # 本文件 — 架构设计
├── todo.md                     # 任务列表和状态追踪
├── progress.md                 # 详细进度日志
├── docs/                       # 教学文档
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
│   ├── main.rs                 # CLI 入口点
│   ├── lib.rs                  # 模块声明
│   ├── config.rs               # 模型配置（来自 config.json）
│   ├── tokenizer.rs            # BPE 分词器（来自 tokenizer.json）
│   ├── tensor.rs               # 带数学运算的简单 N 维张量
│   ├── safetensors.rs          # safetensors 文件格式读取器
│   ├── model.rs                # 完整模型：embedding → blocks → lm_head
│   ├── transformer_block.rs    # 单个 Transformer 块
│   ├── rmsnorm.rs              # RMSNorm 实现
│   ├── rope.rs                 # 旋转位置编码（RoPE）
│   ├── attention.rs            # 带 KV Cache 的分组查询注意力
│   ├── ffn.rs                  # SwiGLU 前馈网络
│   ├── sampling.rs             # Token 采样策略（greedy、top-k、top-p、temperature）
│   └── inference.rs            # 推理循环和 KV Cache 管理
└── tests/
    ├── test_tensor.rs          # 张量运算单元测试
    ├── test_rmsnorm.rs         # RMSNorm 单元测试
    ├── test_rope.rs            # RoPE 单元测试
    ├── test_attention.rs       # 注意力机制单元测试
    └── test_tokenizer.rs       # 分词器单元测试
```

## 5. 依赖项（最小化）

```toml
[dependencies]
clap = { version = "4", features = ["derive"] }   # CLI 参数解析
serde = { version = "1", features = ["derive"] }   # JSON 反序列化
serde_json = "1"                                    # JSON 解析（config.json、tokenizer.json）
byteorder = "1"                                     # 端感知字节读取（safetensors）
```

就这些。**共 4 个 crate**。所有数学运算均从零实现。

### 我们未依赖的 crate

| 跳过的 crate           | 原因                                       | 我们的替代方案                      |
|-----------------------|--------------------------------------------|-------------------------------------|
| `ndarray` / `tch`     | 过于抽象，掩盖了数学细节                   | 自定义 `Tensor` 结构体，用循环实现   |
| `candle` / `burn`     | 机器学习框架 — 违背项目初衷               | 手写前向传播                         |
| `tokenizers`          | HuggingFace 的 Rust 分词器（过于庞大）     | 从零实现简易 BPE                     |
| `rayon`               | 并行化（对教学而言为时过早）               | 单线程，控制流清晰                   |

## 6. 推理流程

```
1. 用户提供提示文本
2. 分词器将文本编码为 token IDs
3. 对每个 token 位置：
   a. 嵌入查找 → x
   b. 对每个 Transformer 块：
      i.   RMSNorm → x_norm
      ii.  带 RoPE + causal mask + KV Cache 的 GQA → attn_out
      iii. 残差相加：x = x + attn_out
      iv.  RMSNorm → x_norm
      v.   SwiGLU FFN → ffn_out
      vi.  残差相加：x = x + ffn_out
   c. 最终 RMSNorm
   d. lm_head 投影 → logits
   e. 从 logits 中采样下一个 token
   f. 将 token 追加到输出，更新 KV Cache
4. 分词器将输出 token IDs 解码为文本
```

## 7. CLI 接口

```bash
# 先下载模型（在本工具之外操作）
# 将 config.json、tokenizer.json、model.safetensors 放入同一目录

# 运行推理
cargo run -- --model-dir ./model --prompt "Hello, world!" --max-tokens 100

# 交互式聊天模式
cargo run -- --model-dir ./model --interactive

# 带采样参数
cargo run -- --model-dir ./model --prompt "Explain Rust" --temperature 0.7 --top-k 50 --top-p 0.9
```
