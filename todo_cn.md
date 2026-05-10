# 任务清单 — qwen3.5-rs

状态：`[ ]` 待处理 | `[~]` 进行中 | `[x]` 已完成

## 阶段 0：项目设置
- [x] T0：创建 design.md
- [x] T0：创建 todo.md 和 progress.md
- [x] T1：项目脚手架 — Cargo.toml、src/lib.rs、src/main.rs 骨架
- [x] T2：安装 Rust 工具链（用户操作 — 记录在 progress.md 中）

## 阶段 1：基础 — 张量与数学
- [x] T3：实现 `tensor.rs` — 支持基本操作的 N 维张量（matmul、add、mul、reshape 等）
- [x] T4：编写 `docs/01_transformer_basics.md` — 什么是 Transformer？它如何处理文本？
- [x] T5：编写 `docs/03_embeddings.md` — 什么是嵌入（Embedding）？词如何变成数字？

## 阶段 2：构建模块
- [x] T6：实现 `rmsnorm.rs` — RMSNorm 归一化
- [x] T7：编写 `docs/04_rmsnorm.md` — 什么是 RMSNorm？为什么不用 BatchNorm/LayerNorm？
- [x] T8：实现 `rope.rs` — 旋转位置编码（Rotary Position Embedding）
- [x] T9：编写 `docs/05_rope.md` — 什么是 RoPE？它如何编码位置信息？
- [x] T10：实现 `attention.rs` — 带 KV 缓存的分组查询注意力（Grouped Query Attention）
- [x] T11：编写 `docs/06_attention.md` — 什么是注意力机制？GQA 与 MHA 有何区别？
- [x] T12：实现 `ffn.rs` — SwiGLU 前馈网络（Feed-Forward Network）
- [x] T13：编写 `docs/07_ffn.md` — 什么是 FFN？为什么使用 SwiGLU？

## 阶段 3：模型组装
- [x] T14：实现 `transformer_block.rs` — 连接注意力 + FFN + 残差连接 + 归一化
- [x] T15：实现 `model.rs` — 完整模型：embedding → blocks → lm_head
- [x] T16：实现 `config.rs` — 将 config.json 解析为 Rust 结构体
- [x] T17：编写 `docs/03_embeddings.md` 更新，补充代码引用

## 阶段 4：加载权重
- [x] T18：实现 `safetensors.rs` — 读取 .safetensors 文件格式
- [x] T19：编写 `docs/08_safetensors.md` — 什么是 safetensors？权重如何存储？
- [x] T20：在 `model.rs` 中实现权重加载 — 将张量名称映射到结构体字段

## 阶段 5：分词器
- [x] T21：实现 `tokenizer.rs` — 基于 tokenizer.json 的 BPE 分词器
- [x] T22：编写 `docs/02_tokenizer.md` — 什么是 BPE？分词是如何工作的？

## 阶段 6：推理
- [x] T23：实现 `sampling.rs` — 贪心搜索、温度采样、top-k、top-p 采样
- [x] T24：实现 `inference.rs` — 带 KV 缓存的自回归生成循环
- [x] T25：编写 `docs/09_inference.md` — 自回归生成是如何工作的？
- [x] T26：编写 `docs/10_sampling.md` — 什么是采样策略？

## 阶段 7：CLI 与集成
- [x] T27：实现 `main.rs` — 基于 clap 的 CLI，支持交互模式和提示词模式
- [x] T28：端到端测试 — 在简单提示词上运行模型并验证输出
- [x] T29：编写 README.md，包含构建与运行说明

## 备注

- 任务通过子代理**一次处理一个**
- 每个代码任务都包含配套的教育文档
- 每个任务完成后会更新 `progress.md`
