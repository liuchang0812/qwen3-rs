# 08 — Safetensors：模型权重在磁盘上的存储方式

训练好的语言模型本质上是一组大型矩阵和向量——数十亿个浮点数，编码了模型在训练过程中学到的所有内容。在运行推理之前，我们需要将这些数字从磁盘加载到内存中。用于存储它们的格式比你想象的要重要得多。

本章解释了 safetensors 格式：它为什么存在，它在字节级别上是如何工作的，以及我们的 Rust 代码如何读取它。

---

## 1. 模型权重是如何存储的？

### 问题的规模

Qwen3-0.6B 大约有 5.96 亿个参数。每个 `f32` 参数占 4 字节，这大约是 2.3 GB 的原始数值数据。更大的模型更加惊人：7B 模型约 28 GB，70B 模型约 280 GB。

这些数字必须在训练后保存到磁盘，并在推理前加载回来。文件格式决定了：

- **加载速度有多快**（几秒钟还是几分钟）。
- **需要多少内存**（我们能否避免一次性加载所有内容？）。
- **是否安全**（打开陌生人的模型文件是否安全）。

### 传统格式及其问题

**PyTorch `.pt` / `.pth` 文件**使用 Python 的 `pickle` 模块进行序列化。Pickle 可以序列化任意 Python 对象——这意味着加载文件时它也可以反序列化并**执行任意代码**。打开恶意的 `.pt` 文件可能让攻击者完全控制你的机器。这不是理论上的风险；概念验证漏洞利用是存在的。

此外，`.pt` 文件加载速度慢，因为 pickle 必须重建整个 Python 对象图，包括 PyTorch 与原始张量一起存储的所有框架元数据。

**NumPy `.npz` 文件**更安全（无代码执行），但设计用于通用数值数据，并非专门针对 ML 模型权重。它们在内部使用 ZIP 压缩，这会增加开销。它们也缺乏一种标准方式来附加张量名称和形状等元数据，以便 ML 框架能够达成一致。

**HDF5** 效率高并支持内存映射，但它是一个复杂的格式，有庞大的 C 库依赖。规范超过 150 页。对于"存储一个命名张量字典"这个特定任务来说，这是杀鸡用牛刀。

ML 社区需要的是一种满足以下条件的格式：

1. **安全**：绝不执行代码。
2. **快速**：加载开销最小。
3. **简单**：在任何语言中都能轻松实现。
4. **可内存映射**：可选按需加载张量，无需读取整个文件。

---

## 2. 什么是 Safetensors？

[Safetensors](https://github.com/huggingface/safetensors) 是由 HuggingFace 专门设计的用于存储 ML 模型权重的文件格式。它满足上述所有四个要求。

### 关键特性

**安全。** 该格式只包含 JSON 头部（元数据）和原始二进制数据（张量值）。没有脚本语言，没有字节码，没有序列化框架，也没有嵌入可执行代码的方式。加载 safetensors 文件永远不会执行任何内容——它只是读取字节。

**快速。** JSON 头部很小（通常只有几 KB），并且只解析一次。张量数据以连续的原始字节存储，与内存布局完全匹配。加载本质上就是一次文件读取加上几次指针调整——无需反序列化，无需对象重建，无需解压缩。

**简单。** 整个规范只用几段话就能说清楚（见第 3 节）。一个基础的读取器可以用任何语言在 100 行代码内实现。没有复杂的帧结构，没有压缩方案，没有需要协商的可选功能。

**可内存映射。** 因为张量数据以连续的原始字节存储在已知的偏移量处，你可以 `mmap` 该文件并按需访问单个张量，而无需将整个文件加载到 RAM 中。这对于大型模型至关重要：F16 格式的 70B 模型约为 140 GB，你可能没有那么多 RAM。使用 mmap，只有你实际访问的页面才会被加载到物理内存中。

### 行业采用

Safetensors 现在是 HuggingFace Model Hub 上的**默认格式**。当你下载像 `Qwen/Qwen3-0.6B` 这样的模型时，你得到的权重文件就是 `.safetensors` 文件。出于安全和性能原因，PyTorch `.bin` 格式正在逐步淘汰。

---

## 3. 文件格式规范

### 二进制布局

一个 safetensors 文件由三个连续的区域组成：

```
┌─────────────────────────────────┐
│ 8 字节：header_size (u64 LE)   │  ← JSON 头部的长度
├─────────────────────────────────┤
│ header_size 字节：JSON 头部    │  ← 张量元数据（填充到 8 字节对齐）
├─────────────────────────────────┤
│ 剩余字节：张量数据              │  ← 原始张量值，拼接存储
└─────────────────────────────────┘
```

**区域 1 — 头部大小（8 字节）。** 一个小端字节序的 `u64`。它告诉我们需要为 JSON 头部读取多少字节。`8 + header_size` 的总和始终是 8 的倍数（如有必要，头部会用空格填充），确保数据部分从 8 字节对齐的偏移量开始。

**区域 2 — JSON 头部。** 一个 UTF-8 JSON 对象，将张量名称映射到它们的元数据。这是"目录"——它告诉我们每个张量的数据位于数据部分的哪个位置，以及如何解释它。

**区域 3 — 数据部分。** 包含实际张量值的原始字节，依次拼接存储。顺序和偏移量由头部决定。每个张量的数据以行优先（C）顺序连续存储，使用小端字节序。

### JSON 头部结构

头部中的每个张量条目有三个字段：

```json
{
  "model.embed_tokens.weight": {
    "dtype": "F32",
    "shape": [151936, 1024],
    "data_offsets": [0, 155705344]
  },
  "model.layers.0.self_attn.q_proj.weight": {
    "dtype": "F32",
    "shape": [1024, 1024],
    "data_offsets": [155705344, 156754304]
  }
}
```

- **`dtype`**（字符串）：每个元素的数据类型。常见值：
  - `"F32"` — 32 位 IEEE 754 浮点数（每个元素 4 字节）
  - `"F16"` — 16 位 IEEE 754 浮点数（每个元素 2 字节）
  - `"BF16"` — BFloat16（每个元素 2 字节，指数范围比 F16 更广）
  - 其他：`"F64"`、`"I8"`、`"I32"`、`"I64"`、`"U8"` 等。

- **`shape`**（整数数组）：张量的维度。这正是你会传递给 `Tensor::new` 的内容。对于线性层中的权重矩阵，这是 `[out_features, in_features]`。

- **`data_offsets`**（两个整数数组）：数据部分中的字节偏移量 `[start, end]`。张量的原始数据占用数据部分的字节 `start` 到 `end`（不包括）。注意：这些偏移量是相对于数据部分的起始位置，而不是文件的起始位置。

头部还可能包含一个特殊的 `"__metadata__"` 键，带有任意的键值对（例如 `{"format": "pt"}`）。这不是一个张量——读取器必须跳过它。

### 真实示例：Qwen3-0.6B 张量名称和形状

Qwen3-0.6B 模型具有以下权重张量（形状以 F32 版本显示）：

```
model.embed_tokens.weight              [151936, 1024]    ← token 嵌入
model.layers.0.self_attn.q_proj.weight [2048, 1024]      ← 查询投影（16 头 * 128）
model.layers.0.self_attn.k_proj.weight [1024, 1024]      ← 键投影（GQA：8 头 * 128）
model.layers.0.self_attn.v_proj.weight [1024, 1024]      ← 值投影（GQA：8 头 * 128）
model.layers.0.self_attn.o_proj.weight [1024, 2048]      ← 输出投影
model.layers.0.mlp.gate_proj.weight    [3072, 1024]      ← SwiGLU 门
model.layers.0.mlp.up_proj.weight      [3072, 1024]      ← SwiGLU 上
model.layers.0.mlp.down_proj.weight    [1024, 3072]      ← SwiGLU 下
model.layers.0.input_layernorm.weight  [1024]            ← 注意力前的 RMSNorm
model.layers.0.post_attention_layernorm.weight [1024]    ← FFN 前的 RMSNorm
...
model.norm.weight                      [1024]            ← 最终 RMSNorm
lm_head.weight                         [151936, 1024]    ← 输出投影（与嵌入共享权重）
```

28 个 transformer 层中的每一个都有相同的 10 个权重张量，总共 28 × 10 + 3 = 283 个张量。嵌入矩阵和 `lm_head` 共享相同的权重（`tie_word_embeddings = true`），因此合并大小为 151936 × 1024 × 4 = 约 590 MB（F32 格式）。

### data_offsets 的工作原理

`data_offsets` 字段指定每个张量的字节在数据部分中的位置。张量通常是连续排列的，因此一个张量的结束偏移量等于下一个张量的开始偏移量：

```
数据部分：
┌──────────────────┬──────────────────┬──────────────────┬─────
│ embed_tokens      │ q_proj (layer 0)  │ k_proj (layer 0)  │ ...
│ [0 .. 155705344)  │ [155705344 ..      │ [156754304 ..      │
│                   │  156754304)       │  157287168)       │
└──────────────────┴──────────────────┴──────────────────┴─────
```

读取第 0 层的查询投影权重：
1. 在头部中查找 `"model.layers.0.self_attn.q_proj.weight"`。
2. 获取 `data_offsets = [155705344, 156754304]`。
3. 在文件中定位到字节 `8 + header_size + 155705344`。
4. 读取 `156754304 - 155705344 = 1048960` 字节（对于 `[1024, 1024]` 的 F32 张量，等于 1024 × 1024 × 4）。
5. 将这些字节解释为 1,048,576 个小端 `f32` 值。

偏移量是字节级别的，而不是元素级别的，这使得该格式与 dtype 无关——相同的偏移量方案适用于 F32、F16、BF16 或任何其他类型。

---

## 4. 支持的数据类型

### F32 (float32) — 我们使用的类型

F32 是 IEEE 754 单精度浮点格式：1 个符号位，8 个指数位，23 个尾数位。每个值占用 4 字节。可表示的范围大约是 ±3.4 × 10^38，精度约为 7 位十进制数字。

对于我们的教育项目，F32 是自然的选择：
- 它与 Rust 的 `f32` 类型完全匹配——不需要转换。
- 它为推理提供了足够的精度（训练通常需要更高的精度来进行梯度累积，但推理不需要）。
- 这是最简单的读取格式：只需将 4 个字节解释为一个小端 `f32`。

### F16 (float16) — 半精度

F16 是 IEEE 754 半精度格式：1 个符号位，5 个指数位，10 个尾数位。每个值只有 2 字节。可表示的范围大约是 ±65,504，精度约为 3 位十进制数字。

F16 在 GPU 上用于推理很流行，因为它比 F32 减少了一半的内存使用和带宽。然而，有限的范围会导致数值问题（溢出到无穷大，下溢到零），需要仔细的缩放。现代 GPU 有原生的 F16 算术单元，使其非常快速。

### BF16 (bfloat16) — 脑浮点

BF16 是 Google 对 F16 的替代方案：1 个符号位，8 个指数位，7 个尾数位。它具有与 F32 相同的指数范围（因此没有溢出问题），但精度只有大约 2-3 位十进制数字。与 F16 一样，每个值占 2 字节。

BF16 是专门为深度学习设计的。其洞察是神经网络对指数范围（以避免溢出）的敏感度远高于尾数位精度（F16 精度的额外位通常被浪费）。现代训练框架（PyTorch、JAX）使用 BF16 作为默认的混合精度格式，许多模型以 BF16 格式分发。

### 为什么我们只支持 F32

我们的实现会拒绝 F16 和 BF16 张量，并给出清晰的错误消息：

```
Unsupported dtype BF16 for tensor model.embed_tokens.weight -- only F32 is supported
```

这是教育项目的刻意选择：
- 支持 F16/BF16 需要转换步骤（读取后反量化为 F32），在不教授新概念的情况下增加了复杂性。
- Qwen3-0.6B 模型在 HuggingFace 上有 F32 版本，因此我们的代码有一个可用的权重文件。
- 对于生产推理引擎，F16/BF16 支持对于减少内存使用是必不可少的。这将是一个自然的扩展练习。

### 大小比较

| Dtype | 每个元素的字节数 | Qwen3-0.6B 总计 | 典型用途 |
|-------|-------------------|---------------------|-------------|
| F32   | 4                 | ~2.3 GB             | 训练、教育推理 |
| F16   | 2                 | ~1.7 GB             | GPU 推理 |
| BF16  | 2                 | ~1.7 GB             | 训练和 GPU 推理 |
| Q8    | 1                 | ~0.85 GB            | 量化 CPU 推理 |

---

## 5. 在我们的代码中读取 Safetensors

让我们逐步讲解 `read_safetensors` 函数中的解析步骤，解释每个步骤的作用和原因。

### 步骤 1：读取 8 字节的头部大小

```rust
let header_size = file.read_u64::<LittleEndian>()? as usize;
```

我们使用 `byteorder` crate 的 `ReadU64` trait 以小端字节序读取一个 `u64`。这给出了 JSON 头部中的字节数。`as usize` 转换是安全的，因为头部大小通常只有几 KB——远低于任何平台上的 `usize` 限制。

### 步骤 2：读取并解析 JSON 头部

```rust
let mut header_bytes = vec![0u8; header_size];
file.read_exact(&mut header_bytes)?;
let header_str = std::str::from_utf8(&header_bytes)?;
let header_map: HashMap<String, serde_json::Value> =
    serde_json::from_str(header_str)?;
```

我们精确读取 `header_size` 字节，验证它们是有效的 UTF-8，并将 JSON 解析为 `HashMap<String, Value>`。我们对顶层使用 `serde_json::Value`，因为头部包含异构条目：张量元数据对象和特殊的 `__metadata__` 对象。

### 步骤 3：计算数据部分的起始位置

```rust
let data_offset_start = 8 + header_size;
```

数据部分在 8 字节大小前缀和头部字节之后立即开始。因为 `header_size` 包含任何填充（safetensors 规范要求 `8 + header_size` 的 8 字节对齐），所以数据部分始终是对齐的。

### 步骤 4：遍历张量条目

对于头部中的每个条目，我们跳过 `__metadata__` 并将值反序列化为我们的 `TensorHeader` 结构体：

```rust
if name == "__metadata__" {
    continue;
}
let tensor_header: TensorHeader = serde_json::from_value(value.clone())?;
```

使用类型化结构体（`TensorHeader`）可以自动提取字段和进行类型检查——如果张量条目缺少 `dtype` 或 `shape`，serde 会返回清晰的错误。

### 步骤 5：验证 dtype

```rust
if tensor_header.dtype != "F32" {
    return Err(format!(
        "Unsupported dtype {} for tensor {} -- only F32 is supported",
        tensor_header.dtype, name
    ).into());
}
```

我们在进行任何 I/O 之前检查 dtype。如果张量使用不支持的格式，我们会快速失败，并给出包含 dtype 和张量名称的清晰消息，以便用户确切知道哪个张量导致了问题。

### 步骤 6：验证数据大小一致性

```rust
let num_elements: usize = tensor_header.shape.iter().product();
let expected_bytes = num_elements * 4;
let [start, end] = tensor_header.data_offsets;
let actual_bytes = end - start;
if actual_bytes != expected_bytes {
    return Err(...);
}
```

我们从形状（维度的乘积 × 每个 F32 元素 4 字节）计算预期的字节数，并验证它与 `data_offsets` 范围匹配。这能捕获头部不一致的畸形文件——这是一个有用的健全性检查，能产生比缓冲区溢出更好的错误消息。

### 步骤 7：读取原始字节

```rust
file.seek(SeekFrom::Start((data_offset_start + start) as u64))?;
let mut raw_bytes = vec![0u8; actual_bytes];
file.read_exact(&mut raw_bytes)?;
```

我们定位到文件中的正确位置（数据部分起始位置 + 张量在数据部分内的字节偏移量），并精确读取所需的字节数。使用 `read_exact` 确保我们获得所有数据或得到清晰的错误——不会出现读取不足的情况。

### 步骤 8：将字节转换为 Vec<f32>

```rust
let mut data = Vec::with_capacity(num_elements);
let mut cursor = std::io::Cursor::new(&raw_bytes);
for _ in 0..num_elements {
    data.push(cursor.read_f32::<LittleEndian>()?);
}
```

这就是 `byteorder` crate 发挥作用的地方。safetensors 文件中的每个 F32 值都以小端字节序存储（现代 x86 和 ARM 处理器的标准）。`read_f32::<LittleEndian>()` 方法读取 4 个字节并将它们解释为一个小端 IEEE 754 浮点数，返回一个 Rust `f32`。

为什么使用 byteorder 而不是 `f32::from_le_bytes`？两者都有效，但 byteorder 提供了一个方便的基于 `Read` 的接口，为我们处理逐字节迭代。它也与我们读取头部大小（步骤 1）的方式一致。使用 `from_le_bytes` 的替代方法如下：

```rust
// 替代方案：使用 from_le_bytes（基于块）
let data: Vec<f32> = raw_bytes
    .chunks_exact(4)
    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
    .collect();
```

两者都是正确的；当你已经使用该 crate 进行其他读取时，byteorder 方法稍微更符合习惯用法。

### 步骤 9：创建条目并返回

```rust
result.insert(
    name.clone(),
    SafeTensorEntry {
        name: name.clone(),
        shape: tensor_header.shape,
        dtype: tensor_header.dtype,
        data,
    },
);
```

我们将每个张量存储为一个 `SafeTensorEntry`——一个包含张量名称、形状、dtype 字符串和 f32 数据的简单结构体。`read_safetensors_as_tensors` 便捷函数然后通过调用 `Tensor::new(entry.shape, entry.data)` 将每个条目转换为我们的 `Tensor` 类型。

---

## 6. 内存考虑

### 我们要加载多少数据？

Qwen3-0.6B（F32 格式）大约有 5.96 亿个参数。每个 F32 参数占 4 字节，因此张量数据总量为：

```
596M × 4 字节 ≈ 2.3 GB
```

该模型的 JSON 头部相比之下非常小——只有几十 KB。所以文件几乎完全是原始张量数据。

### 一次性全部加载

我们的 `read_safetensors` 函数将每个张量读取到 `Vec<f32>` 中，并将它们全部存储在 `HashMap` 中。对于 Qwen3-0.6B，这意味着：

- 约 2.3 GB 用于 `Vec<f32>` 缓冲区（张量数据本身）。
- `HashMap` 结构和 `String` 键的一些开销（可忽略不计）。
- 在读取期间，我们还分配临时的 `raw_bytes` 缓冲区（转换为 `Vec<f32>` 后释放）。

读取期间的峰值内存使用量大约是数据大小的两倍（一份在 `raw_bytes` 中，一份在最终的 `Vec<f32>` 中），读取完成后稳定在约 2.3 GB。对于适合 RAM 的模型来说，这没问题。

### 张量生命周期

加载后，`HashMap<String, Tensor>` 通常被模型构建代码消费，该代码按名称提取每个张量并将其分配给相应的模型组件。然后 HashMap 被丢弃，只有模型的字段持有对数据的引用。模型构建期间的峰值内存与加载的数据大致相同——张量是被移动，而不是被复制。

### 大型模型的内存映射

对于太大而无法放入 RAM 的模型（或者当你想避免前期加载成本时），safetensors 格式支持**内存映射**（mmap）。其思路很简单：

1. 对文件调用 `mmap` 将其映射到进程的虚拟地址空间。这实际上并不读取数据——它只是设置页表项。
2. 当代码访问张量的数据时，操作系统会按需从磁盘加载相应的页面（"页错误"）。
3. 最近未被访问的页面可以被操作系统从物理 RAM 中驱逐，自动释放内存。

使用 mmap，你可以在只有 16 GB RAM 的机器上"加载"一个 70 GB 的模型——操作系统会将最近使用的张量保留在内存中，其余的换出到磁盘。这比将所有内容都放在 RAM 中要慢，但它使大型模型可以在适中的硬件上访问。

我们的实现不使用 mmap（它会为教育项目增加显著的复杂性），但 safetensors 格式是为了支持它而设计的。启用 mmap 的关键设计选择是：

- **连续的数据部分**：所有张量数据都在文件的一个连续区域中，使得一次性映射整个数据部分变得容易。
- **已知偏移量**：头部确切地告诉你每个张量的数据位于何处，因此你可以在不复制的情况下构建对 mmap 区域的"视图"。
- **无压缩**：磁盘上的字节与内存中的字节完全相同，因此 mmap 的数据可以直接使用，无需解压缩。

### Qwen3-0.6B 推理的大致内存预算

| 组件 | 内存（F32） |
|-----------|-------------|
| 模型权重 | ~2.3 GB |
| KV 缓存（28 层，seq_len=2048） | ~224 MB |
| 激活（每次前向传播） | ~10 MB |
| Tokenizer、配置、杂项 | ~10 MB |
| **总计** | **~3.7 GB** |

这适合任何具有 8+ GB RAM 的现代笔记本电脑。对于 F32 格式的 7B 模型，仅权重就约为 28 GB，这需要一台具有 32+ GB RAM 的机器或使用 F16/量化权重。

---

## 总结

| 概念 | 要点 |
|---------|-----------|
| 为什么选择 safetensors？ | 安全（无代码执行）、快速（原始字节）、简单（JSON + 二进制） |
| 文件布局 | 8 字节头部大小，然后是 JSON 头部，然后是原始张量数据 |
| JSON 头部 | 将张量名称映射到 `{dtype, shape, data_offsets}` |
| data_offsets | 数据部分中的字节范围 `[start, end]` |
| 我们的 dtype 支持 | 仅 F32；F16/BF16 会产生清晰的错误消息 |
| byteorder crate | 以指定的字节序读取多字节值（u64、f32） |
| 0.8B 的内存 | F32 格式约 2.3 GB——适合 RAM，不需要 mmap |
| 内存映射 | 通过按需分页数据，使加载比 RAM 更大的模型成为可能 |

safetensors 格式是良好设计的一个绝佳示例：它以尽可能小的复杂性解决了一个真正的问题（安全、快速的模型加载）。读取器只需要三个操作——读取 8 个字节、解析 JSON、读取字节范围——并且可以在一个下午内实现。然而，这种简单的格式现在支撑着 HuggingFace Hub 上数十万个模型的分布。
