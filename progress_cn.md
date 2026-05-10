# 进度日志 — qwen3.5-rs

## 会话：2026-05-09

### 全部任务完成 + Bug 修复已应用

最初的 29 个任务全部完成。随后使用真实 Qwen3-0.6B 模型进行测试，发现并修复了 5 个 Bug：

#### 真实模型测试后的 Bug 修复

1. **CLI 标志冲突**：`-m` 同时用于 `--model-dir` 和 `--max-tokens` → 移除了 `--max-tokens` 的 `-m` 短标志
2. **head_dim 不匹配**：Qwen3 配置中显式指定了 `head_dim=128`（而非 `hidden_size/num_heads=64`）→ 在 ModelConfig 中添加了 `head_dim: Option<usize>`
3. **BF16 权重**：真实模型使用 BF16 而非 F32 → 在 safetensors.rs 中添加了 BF16→F32 和 F16→F32 的转换
4. **Tokenizer 合并格式**：Qwen3 使用数组格式合并 `[["a","b"],...]`，而非字符串格式 `["a b",...]` → 自定义反序列化器同时处理两种格式
5. **缺少 q_norm/k_norm**：Qwen3 在 RoPE 之前对每头 Q 和 K 应用 RMSNorm → 在 Attention 中添加了可选的 q_norm/k_norm

### 最终统计

- **162 个测试全部通过**
- **模型在真实 Qwen3-0.6B 权重上成功运行**
- **速度**：约 0.1 tokens/sec（单线程 f32 CPU，未优化状态下符合预期）
- **输出正确**：`"1+1="` → `"2, 1+1=2"`（贪心解码），ChatML 提示词可生成连贯文本

### 已下载的模型文件

- `model/config.json` — Qwen3-0.6B 配置文件
- `model/tokenizer.json` — 151,643 词表，151,387 个合并规则
- `model/model.safetensors` — 1.5GB BF16 权重

### 运行方式

```bash
cargo run --release -- --model-dir ./model --prompt "Hello" --max-tokens 20 --temperature 0.7
# 使用 ChatML 格式以获得最佳效果：
cargo run --release -- --model-dir ./model --prompt "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n" --max-tokens 50
```
