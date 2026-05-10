# 04. RMSNorm：为何及如何归一化激活值

神经网络非常强大，但也十分脆弱。如果不加控制，流经深度网络的数字可能爆炸到无穷大或坍缩到零，导致训练无法进行。归一化层就是将这些数字保持在安全范围内的护栏。在 Qwen3 中，我们使用的具体护栏叫做 **RMSNorm**（均方根归一化）。

本文档将解释为什么需要归一化、RMSNorm 的工作原理、它与其前身 LayerNorm 的对比，以及我们如何在 Rust 中实现它。

---

## 1. 为什么需要归一化？

### 问题：内部协变量偏移

想象你在训练一个 28 层的神经网络。训练过程中，每一层都接收前一层的输出作为输入。随着前面层的权重通过梯度下降不断更新，其输出的分布也在不断变化。这意味着中间层（比如第 14 层）会不断面对不断变化的输入分布——它永远无法获得一个稳定的数据"视角"。

这种现象被称为**内部协变量偏移**。其实际后果是灾难性的：

- **激活值爆炸**：如果前面层的权重略微增大，输出就会略微增大，进而导致下一层的输出更大，依此类推，呈指数级增长。到了第 28 层，数字变成了 NaN（非数字）——网络已经崩溃。

- **激活值消失**：如果前面层的权重略微缩小，输出也会缩小，下一层的输出进一步缩小。到了第 28 层，数字已经接近零，梯度也变为零，学习完全停止。

- **收敛缓慢**：即使网络没有爆炸或消失，不断变化的输入分布迫使每一层不断适应，导致训练缓慢，并且需要仔细调整学习率。

### 一个类比

想象一条 28 人的装配流水线。每个人从前一个人那里接收工件，完成自己的部分，然后传递下去。如果第 3 个人开始发送过大的工件（因为他改变了操作方法），第 4 个人就会不堪重负，产出粗制滥造的工件，第 5 个人则会更加困惑。问题沿着流水线级联下去。归一化就像在每两个工人之间设置一名质检员，在传递之前将工件重新缩放到标准尺寸。

### 解决方案：归一化

归一化层通过**在每一层重新缩放激活值**来解决这个问题，使其始终落在一致的范围内。无论前一层做了什么——无论它产生的是极小的数字还是巨大的数字——归一化层都会在传递给下一个子层之前，将它们转换回标准尺度。

这极大地稳定了训练。使用归一化后：
- 激活值在整个网络中保持在合理范围内。
- 梯度在反向传播过程中流动更顺畅。
- 网络收敛更快，对学习率的选择不那么敏感。

---

## 2. LayerNorm（前身）

在 RMSNorm 之前，Transformer 中主流的归一化方法是 **LayerNorm**（层归一化），由 Ba 等人在 2016 年提出。原始的 Transformer 论文（Vaswani 等人，2017）使用了 LayerNorm，BERT、GPT-1、GPT-2 以及许多其他早期模型也是如此。

### 公式

对于长度为 `n` 的输入向量 **x**，LayerNorm 计算如下：

```
LayerNorm(x) = weight * (x - mean(x)) / sqrt(var(x) + eps) + bias

where:
  mean(x) = (1/n) * sum(x_i)
  var(x)  = (1/n) * sum((x_i - mean(x))^2)
  weight  = learnable scale parameter of shape [n]  (called gamma)
  bias    = learnable shift parameter of shape [n]  (called beta)
  eps     = small constant for numerical stability (e.g., 1e-5)
```

### 分步示例

让我们通过一个具体示例来走一遍 LayerNorm。假设 `x = [3.0, 4.0]`，`weight = [1.0, 1.0]`，`bias = [0.0, 0.0]`，`eps = 1e-6`。

**步骤 1：计算均值。**
```
mean = (3.0 + 4.0) / 2 = 3.5
```

**步骤 2：减去均值（将数据中心化）。**
```
x_centered = [3.0 - 3.5, 4.0 - 3.5] = [-0.5, 0.5]
```

**步骤 3：计算方差。**
```
var = ((-0.5)^2 + 0.5^2) / 2 = (0.25 + 0.25) / 2 = 0.25
```

**步骤 4：除以标准差（缩放数据）。**
```
std = sqrt(0.25 + 1e-6) = sqrt(0.250001) ≈ 0.5000
x_normalized = [-0.5 / 0.5000, 0.5 / 0.5000] = [-1.0, 1.0]
```

**步骤 5：乘以权重并加上偏置。**
```
output = [1.0 * (-1.0) + 0.0, 1.0 * 1.0 + 0.0] = [-1.0, 1.0]
```

结果是一个经过中心化和缩放的输入版本。均值为零，标准差为一。权重和偏置让网络能够学习最优的缩放和偏移量。

### LayerNorm 为何有效

LayerNorm 之所以有效，是因为它保证了每一层的激活值具有可预测的分布：零均值和单位方差（在应用可学习的权重和偏置之前）。这防止了爆炸/消失问题，并使优化变得更加容易。

可学习的 `weight` 和 `bias` 参数非常重要：它们让网络在有益的情况下能够"撤销"归一化。如果网络确定某个特定层的最佳表示应该是均值 5、标准差 3，它就可以通过学习 `bias = 5` 和 `weight = 3` 来实现。归一化为网络提供了一个稳定的"默认值"，网络可以在此基础上构建。

---

## 3. RMSNorm（我们所使用的方法）

RMSNorm 由 Zhang 和 Sennrich 在 2019 年提出。其核心见解很简单：**LayerNorm 中的均值减法实际上并没有太大帮助**，去掉它可以让计算更廉价，且不会损害性能。

### 公式

对于长度为 `n` 的输入向量 **x**，RMSNorm 计算如下：

```
RMSNorm(x) = weight * x / sqrt(mean(x^2) + eps)

where:
  mean(x^2) = (1/n) * sum(x_i^2)
  weight    = learnable scale parameter of shape [n]  (called gamma)
  eps       = small constant for numerical stability (1e-6 in Qwen3)
```

注意与 LayerNorm 相比缺少了什么：
- **没有均值减法**：我们不对数据做零中心化。
- **没有偏置**：没有可学习的偏移参数。
- **没有方差计算**：我们使用 `mean(x^2)` 而不是 `var(x)`。

### 分步示例

让我们用相同的输入走一遍 RMSNorm：`x = [3.0, 4.0]`，`weight = [1.0, 1.0]`，`eps = 1e-6`。

**步骤 1：对每个元素求平方。**
```
x_squared = [3.0^2, 4.0^2] = [9.0, 16.0]
```

**步骤 2：计算平方的均值。**
```
mean_sq = (9.0 + 16.0) / 2 = 25.0 / 2 = 12.5
```

**步骤 3：为数值稳定性加上 epsilon。**
```
mean_sq + eps = 12.5 + 0.000001 ≈ 12.5
```

在这里 epsilon 基本上没有区别，因为 `mean_sq` 已经远大于零。它只有在所有输入元素都为零（或非常接近零）时才有意义，此时 `mean_sq` 为零，我们将除以 `sqrt(eps)` 而不是零。

**步骤 4：取平方根。**
```
rms = sqrt(12.5) = 3.5355339...
```

这就是输入的**均方根**——平方元素均值的平方根。它是向量"幅度"的一种度量，类似于 L2 范数，但按 `1/sqrt(n)` 进行了缩放。

**步骤 5：将每个元素除以 RMS。**
```
x_normalized = [3.0 / 3.5355, 4.0 / 3.5355]
             = [0.84853..., 1.13137...]
```

在此步骤之后，输出向量的均方根恰好为 1.0（忽略微小的 epsilon）。向量已被"重新缩放"，使其幅度标准化，无论原始幅度如何。

**步骤 6：乘以可学习的权重。**
```
output = [1.0 * 0.8485, 1.0 * 1.1314]
       = [0.8485, 1.1314]
```

由于本例中权重全为一，输出与归一化后的向量相同。在实践中，权重是在训练过程中学习的，允许模型重新缩放各个维度。

### 对比相同输入下的 LayerNorm 和 RMSNorm

让我们将两者并排对比，输入为 `x = [3.0, 4.0]`：

| 步骤 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 均值 | 3.5 | （不计算） |
| 中心化 | [-0.5, 0.5] | （不做） |
| 方差/平方均值 | 0.25 | 12.5 |
| 标准差/RMS | 0.5 | 3.5355 |
| 归一化后 | [-1.0, 1.0] | [0.8485, 1.1314] |
| 权重+偏置后 | [-1.0, 1.0] | [0.8485, 1.1314] |

关键差异在归一化后的输出中可见。LayerNorm 将数据中心化（有正负值），而 RMSNorm 只进行缩放（所有值保持原有符号）。LayerNorm 产生零均值输出；RMSNorm 则不然。

### 一个细节："mean(x^2)"与方差有什么关系？

如果仔细观察，你可能会注意到 `mean(x^2)` 与方差有关：

```
var(x) = mean(x^2) - mean(x)^2
```

因此：

```
mean(x^2) = var(x) + mean(x)^2
```

RMSNorm 使用 `mean(x^2)`，而 LayerNorm 使用 `var(x)`。区别在于 RMSNorm 的计算中包含了 `mean(x)^2`。当均值相对于方差较大时，RMSNorm 和 LayerNorm 的结果会有很大不同。当均值接近零时，它们则相似。

在实践中，训练良好的 Transformer 中激活值的均值往往相对于方差较小，这也是为什么去掉均值减法不会损害性能的部分原因。

---

## 4. 为什么选择 RMSNorm 而非 LayerNorm？

你可能会想：如果 LayerNorm 多年来一直运行良好，为什么要切换到 RMSNorm？原因有三。

### 4.1 更简单（操作更少）

LayerNorm 需要以下操作：
1. 计算输入的均值。
2. 从每个元素中减去均值。
3. 计算中心化输入的方差。
4. 除以标准差。
5. 乘以权重。
6. 加上偏置。

RMSNorm 需要：
1. 计算平方的均值（一次乘加遍历）。
2. 除以均方根。
3. 乘以权重。

操作数量大约减少了一半。在像 Qwen3 这样拥有 57 个 RMSNorm 层的模型中（详见下文），节省的代价会累积起来。

### 4.2 同样有效

RMSNorm 的原始论文（Zhang & Sennrich, 2019）表明，在一系列语言建模基准测试中，RMSNorm 的质量与 LayerNorm 相当甚至略胜一筹。事实证明，均值减法并没有做太多有用的工作。归一化最重要的部分是**重新缩放**（除以幅度度量），而不是**中心化**（减去均值）。

这在直觉上是合理的：稳定性关键在于激活值不会爆炸或消失。将它们缩放到一致的幅度即可实现这一点。它们是否以零为中心是一个次要问题，网络的其余部分可以轻松适应。

### 4.3 略微更快

去掉均值减法和偏置不仅简化了代码，还减少了内存访问和计算量：

- **每个维度少一个可学习参数**：LayerNorm 同时拥有 `weight` 和 `bias`（每个维度 2 个参数），而 RMSNorm 只有 `weight`（每个维度 1 个参数）。对于 hidden_size = 1024，每层归一化节省 1024 个参数，总共节省 1024 * 57 = 58,368 个参数。这在内存方面微不足道，但略微减少了前向传播过程中需要加载的数据量。

- **少一次数据遍历**：LayerNorm 需要先计算均值，然后用它来中心化数据，再计算方差。RMSNorm 可以在一次遍历中计算 `mean(x^2)`。

在实践中，速度差异很小（归一化层在 Transformer 的总计算量中只占极小一部分——注意力和 FFN 占主导）。但简单性的优势是实实在在的：更简单的代码更容易正确实现、更容易优化、更容易推理。

### 4.4 现代标准

RMSNorm 已成为现代 LLM 的默认选择：

| 模型 | 年份 | 归一化方法 |
|-------|------|---------------|
| BERT | 2018 | LayerNorm |
| GPT-2 | 2019 | LayerNorm |
| GPT-3 | 2020 | LayerNorm |
| LLaMA | 2023 | RMSNorm |
| LLaMA 2 | 2023 | RMSNorm |
| Mistral | 2023 | RMSNorm |
| Qwen | 2023 | RMSNorm |
| Qwen2 | 2024 | RMSNorm |
| Qwen3 | 2025 | RMSNorm |

这一转变大约发生在 2023 年，伴随着 LLaMA 系列。一旦证明 RMSNorm 以更少的操作达到同等效果，社区便迅速采用了它。如今，几乎所有新的 LLM 架构都使用 RMSNorm。

---

## 5. RMSNorm 在 Qwen3 中的使用位置

RMSNorm 出现在 Qwen3 模型架构的三个地方：

### 5.1 自注意力之前（input_layernorm）

在每个 Transformer 块中，输入首先通过 RMSNorm 归一化，然后再传递给自注意力机制。这被称为**输入层归一化**或**注意力前归一化**。

```
x_norm = RMSNorm(x)                    # 归一化
attn_out = self_attention(x_norm)       # 注意力
output = x + attn_out                   # 残差连接
```

该层的权重存储在以下键下：
`model.layers.{i}.input_layernorm.weight`

其中 `{i}` 的范围从 0 到 27（对应 28 个 Transformer 块）。

### 5.2 FFN 之前（post_attention_layernorm）

在自注意力子层（及其残差连接）之后，结果在传递给 FFN 之前再次归一化。这被称为**注意力后层归一化**或**FFN 前归一化**。

```
x_norm = RMSNorm(x')                   # 归一化（x' 是注意力残差输出）
ffn_out = ffn(x_norm)                   # 前馈网络
output = x' + ffn_out                   # 残差连接
```

该层的权重存储在以下键下：
`model.layers.{i}.post_attention_layernorm.weight`

### 5.3 最终归一化（model.norm）

在所有 28 个 Transformer 块之后，还有一个最终的 RMSNorm 层，在通过 `lm_head` 投影到词表 logits 之前对输出进行归一化。这确保了 logits 是从良好缩放的表示中计算出来的。

```
hidden = block_27_output
hidden_norm = RMSNorm(hidden)           # 最终归一化
logits = lm_head(hidden_norm)           # 投影到词表
```

该层的权重存储在以下键下：
`model.norm.weight`

### 5.4 统计 RMSNorm 层数量

让我们统计一下总数：

- 每个 Transformer 块：2 个（input_layernorm + post_attention_layernorm）
- Transformer 块数量：28
- 块小计：28 * 2 = 56
- 最终归一化：1
- **总计：56 + 1 = 57 个 RMSNorm 层**

这 57 个层各自拥有独立的 `weight` 参数，形状为 `[1024]`。RMSNorm 参数的总数为：

```
57 * 1024 = 58,368 个参数
```

以 f32（每个 4 字节）计算，仅为 233,472 字节——约 228 KB。与约 5.8 亿的总参数量相比，这完全可以忽略不计。归一化层对模型的内存占用几乎没有任何贡献，但它们对训练稳定性至关重要。

### 5.5 Pre-Norm 与 Post-Norm

一个重要的架构细节：Qwen3 使用 **pre-norm** 设计，即归一化应用于子层（注意力或 FFN）*之前*，而不是之后。原始 Transformer 论文使用 **post-norm**，即归一化应用于子层*之后*。

Pre-norm：
```
output = x + sublayer(RMSNorm(x))
```

Post-norm：
```
output = RMSNorm(x + sublayer(x))
```

Pre-norm 已成为标准，因为它在训练深度网络时更稳定。原因是：在 pre-norm 中，残差连接将未修改的输入 `x` 直接传递到输出，因此梯度始终可以通过恒等路径流动。而在 post-norm 中，归一化应用于加法*之后*，这仍可能导致梯度问题，因为归一化本身可能扭曲信号。

在我们的 Qwen3 实现中，注意力子层和 FFN 子层都遵循 pre-norm 模式。

---

## 6. 实现细节

现在让我们看看如何在 Rust 中实现 RMSNorm。

### 6.1 RMSNorm 结构体

我们的实现将底层的 `Tensor::rms_norm` 操作封装在一个可复用的结构体中，该结构体存储权重和 epsilon：

```rust
pub struct RMSNorm {
    weight: Tensor,  // shape [hidden_size], the learned scaling parameter
    eps: f32,        // small constant for numerical stability (1e-6)
}
```

该结构体故意设计得很简单。它拥有一个权重张量和一个 epsilon 值。当我们为特定层创建一个 `RMSNorm` 实例时，我们从 safetensors 文件中加载权重并存储在结构体中。对于 Qwen3，epsilon 始终为 `1e-6`。

### 6.2 构造

```rust
impl RMSNorm {
    pub fn new(weight: Tensor, eps: f32) -> Self {
        assert_eq!(weight.ndim(), 1, "weight must be 1-D");
        Self { weight, eps }
    }
}
```

构造函数验证权重是一个 1-D 张量（向量），并将其与 epsilon 一起存储。在实践中，这在模型初始化期间调用一次：

```rust
// 加载模型时：
let input_ln_weight = weights.load("model.layers.0.input_layernorm.weight");
let input_layernorm = RMSNorm::new(input_ln_weight, 1e-6);
```

### 6.3 前向传播

```rust
impl RMSNorm {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.rms_norm(&self.weight, self.eps)
    }
}
```

forward 方法完全委托给 `Tensor::rms_norm`。这是一个深思熟虑的设计选择：tensor 模块拥有底层数学运算（按行循环、累加、平方根），而 `RMSNorm` 结构体提供了一个干净的、可复用的接口来存储层的参数。

### 6.4 Tensor::rms_norm 后端

实际计算发生在 `Tensor::rms_norm`（在 `tensor.rs` 中）。以下是它对输入每一行所做的操作：

```rust
pub fn rms_norm(&self, weight: &Tensor, eps: f32) -> Tensor {
    // 对输入张量的每一行：
    for r in 0..num_rows {
        // 步骤 1：计算该行的平方均值。
        let mut sum_sq = 0.0f32;
        for j in 0..last_dim {
            let v = self.data[row_start + j];
            sum_sq += v * v;                     // 累加 x_i^2
        }
        let mean_sq = sum_sq / (last_dim as f32); // (1/n) * sum(x_i^2)

        // 步骤 2：计算 RMS 的倒数。
        //   rms = sqrt(mean_sq + eps)
        //   1/rms = 1 / sqrt(mean_sq + eps)
        let rms_inv = 1.0 / (mean_sq + eps).sqrt();

        // 步骤 3：归一化并按权重缩放。
        for j in 0..last_dim {
            result[row_start + j] = self.data[row_start + j] * rms_inv * weight.data[j];
        }
    }
}
```

几个实现要点：

**为什么计算 `rms_inv` 而不是 `rms`？** 我们计算 RMS 的倒数（`1/rms`），然后乘以 `rms_inv` 而不是除以 `rms`。这是一个标准优化：乘法比除法快，我们只需要一次除法（计算 `1/rms`）而不是 `n` 次除法（每个元素一次）。

**为什么要按行迭代？** RMSNorm 的输入是一个形状为 `[seq_len, hidden_size]` 的 2-D 张量。每一行代表一个 token 的隐藏状态，归一化独立应用于每一行。同一序列中的两个不同 token 不应相互影响各自的归一化结果。

**为什么在循环内乘以权重？** 我们可以先归一化，然后在单独的步骤中乘以权重。但将它们合并到一个循环中可以避免对数据的额外遍历和额外的临时张量分配。对于 hidden_size 为 1024 和典型序列长度的情况，这可以节省相当数量的内存操作。

### 6.5 数值稳定性

`eps` 参数对数值稳定性至关重要。考虑输入全为零的情况：

```
x = [0.0, 0.0, ..., 0.0]
mean_sq = 0.0
rms = sqrt(0.0 + eps) = sqrt(1e-6) ≈ 0.001
output = [0.0 / 0.001, 0.0 / 0.001, ...] * weight = [0.0, 0.0, ...]
```

如果没有 `eps`，我们将得到 `sqrt(0.0) = 0.0` 和除以零。有了 `eps`，我们得到一个很小但非零的分母，输出正确地为零（因为分子也为零）。

在实践中，激活值很少恰好为零。但它们可能非常小，如果没有 `eps`，浮点舍入可能导致分母为零或极小，从而导致数值不稳定（NaN 或 Inf 值在网络中传播）。Epsilon 确保了这种情况永远不会发生。

值 `1e-6` 是 Qwen3 的标准选择（在 `config.json` 中以键 `rms_norm_eps` 指定）。它足够小，不会影响激活值正常大小的计算（例如，当 `mean_sq` 约为 1.0 时，加上 `1e-6` 没有任何区别），但又足够大，可以在激活值极小时防止下溢。

### 6.6 内存布局

权重张量的形状为 `[hidden_size]` = `[1024]`，存储为 1024 个 f32 值的连续数组。在前向传播过程中，该数组对输入的每一行按顺序访问。由于访问模式是顺序的且数组很小（仅 4 KB），它可以完全放入现代 CPU 的 L1 缓存中。这意味着从内存角度来看，权重访问基本上是免费的。

输入和输出张量的形状为 `[seq_len, hidden_size]`，按行主序存储。这意味着每一行（每个 token 的隐藏状态）在内存中是连续的，这对于逐行归一化循环来说是最优的。

### 6.7 在 Transformer 块中的使用

以下是 RMSNorm 如何融入 Transformer 块的前向传播：

```rust
fn forward(&self, x: &Tensor) -> Tensor {
    // 注意力前归一化
    let x_norm = self.input_layernorm.forward(x);

    // 自注意力
    let attn_out = self.attention.forward(&x_norm);

    // 残差连接
    let x = x.add(&attn_out);

    // FFN 前归一化
    let x_norm = self.post_attention_layernorm.forward(&x);

    // 前馈网络
    let ffn_out = self.ffn.forward(&x_norm);

    // 残差连接
    x.add(&ffn_out)
}
```

注意这个模式：先归一化，再处理，然后加上残差。每个块中发生两次（一次用于注意力，一次用于 FFN），且归一化始终应用于子层的*输入*，而不是输出。

---

## 总结

让我们回顾一下关于 RMSNorm 的要点：

| 概念 | 总结 |
|---------|---------|
| **功能** | 按均方根归一化向量，然后按可学习的权重缩放 |
| **公式** | `RMSNorm(x) = weight * x / sqrt(mean(x^2) + eps)` |
| **与 LayerNorm 对比** | 去掉均值减法和偏置；更简单但同样有效 |
| **Epsilon** | Qwen3 中为 1e-6；防止除以零 |
| **权重** | 形状为 `[hidden_size]` 的可学习参数；每个归一化层一个 |
| **使用位置** | 共 57 处：每个 Transformer 块 2 个（56 个）+ 1 个最终归一化 |
| **应用时机** | 在每个子层之前（pre-norm 架构） |
| **参数量** | 57 * 1024 = 58,368 总计；与完整模型相比可忽略不计 |

RMSNorm 是 Transformer 中最简单的组件之一，但也是最重要的之一。如果没有归一化，训练一个 28 层的网络将极其困难。RMSNorm 以最小的计算开销提供了必要的稳定性，这就是它成为现代 LLM 标准选择的原因。

在下一篇文档中，我们将探讨如何使用旋转位置编码（RoPE）来编码位置信息：[05_rope.md](05_rope.md)。
