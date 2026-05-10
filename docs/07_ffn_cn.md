# 07 — SwiGLU 前馈网络

前馈网络（FFN）是 Transformer 的"思考"部分。如果说注意力是让 token 之间*互相查看*的机制，那么 FFN 就是让每个 token *处理*它所收集信息的机制。每个 Transformer 块在注意力之后都会应用一个 FFN，在现代大语言模型中，FFN 约占模型总参数量的一半。

本文档将从最简单的 FFN 形式逐步讲解到 Qwen3 使用的 SwiGLU 变体，包含具体的数值示例以及对我们的 Rust 实现的介绍。

---

## 1. 什么是前馈网络？

前馈网络本质上是最简单的一种神经网络层：接收一个输入向量，乘以一个权重矩阵，可能加上偏置，应用激活函数，然后产生一个输出向量。没有循环，没有递归，没有注意力——只有线性变换加上非线性。

在 Transformer 中，FFN 是**独立应用于每个 token** 的。在注意力从其他位置收集了上下文信息之后，FFN 自行变换每个 token 的表示。FFN 内部不存在跨 token 的交互。这是一个关键的设计选择：它使 FFN 保持简单且可并行化，而注意力层则负责所有的跨位置通信。

你可以将 Transformer 块视为两个阶段：

1. **注意力**——"环顾四周。"每个 token 查询所有其他 token 并聚合相关信息。
2. **FFN**——"思考你所看到的。"每个 token 通过非线性变换处理其更新后的表示。

这种分工——注意力负责通信，FFN 负责计算——是每个 Transformer 模型的基础架构。

---

## 2. 标准 FFN

原始 Transformer 论文（Vaswani et al., 2017）使用了一种简单的两层 FFN，带有 ReLU 激活：

```text
FFN(x) = W2 · ReLU(W1 · x + b1) + b2
```

其中：
- `W1` 是"上"投影，从 `hidden_size` 扩展到 `intermediate_size`（也称为 `d_ff`）
- `W2` 是"下"投影，从 `intermediate_size` 缩放回 `hidden_size`
- `ReLU(x) = max(0, x)` 将负值截断为零
- `b1`、`b2` 是偏置向量（现代大语言模型通常省略这些）

维度扩展是关键的设计选择。在原始 Transformer 中，`hidden_size = 512`，`intermediate_size = 2048`——即 4 倍扩展。其思想是，更宽的中间层赋予网络更强的学习能力来捕捉复杂模式。你可以这样理解：输入被分解到一个更高维的空间，在那里更容易分离和变换不同的特征，然后再投影回原始空间。

以下是一个使用微小维度的具体示例。假设 `hidden_size = 4`，`intermediate_size = 8`：

```text
Input x: [1.0, -0.5, 0.3, 2.0]              shape [4]

W1 (up projection):                           shape [8, 4]
  (a random 8x4 matrix)

h = W1 · x                                    shape [8]
  (8 intermediate features, some negative)

h_activated = ReLU(h)                         shape [8]
  (negative values become 0)

W2 (down projection):                         shape [4, 8]
  (a random 4x8 matrix)

Output = W2 · h_activated                     shape [4]
  (back to the original hidden dimension)
```

ReLU 激活函数使整个网络具有非线性。如果没有它，两个线性层将退化为单个线性变换（因为 `W2 · W1` 只是另一个矩阵），网络将失去大部分表达能力。

---

## 3. GLU 变体

2017 年，Dauphin 等人引入了门控线性单元（GLU），为前馈网络添加了*门控机制*。其灵感来自 LSTM 和 GRU，它们使用门来控制信息流。

GLU FFN 用两个并行投影替代了单一的上投影，并使用其中一个来门控另一个：

```text
GLU(x) = W_down · (gate ⊙ W_up · x)
```

其中：
- `W_up` 将输入投影到中间维度（"值"路径）
- `gate` 通过单独的投影从输入计算得出，然后经过 sigmoid 激活产生 0 到 1 之间的值
- `⊙` 表示逐元素乘法（Hadamard 积）
- `W_down` 将门控结果投影回隐藏维度

门的作用就像一个**水阀**：对于每个中间特征，门决定让多少该特征通过。门值接近 1 表示"完全放行该特征"，而门值接近 0 表示"完全阻挡该特征"。

为什么这比 ReLU 更好？ReLU 对每个元素应用相同的硬截断（负值为零，正值为恒等）。而 GLU 门是*依赖于输入的*：门值会根据输入内容而变化，使网络能够学习自适应的、对上下文敏感的特征选择。

---

## 4. SwiGLU —— Qwen3 所使用的

SwiGLU（Swish 门控线性单元）是 Qwen3 以及几乎所有现代大语言模型所使用的特定 GLU 变体。它由 Shazeer（2020）在《GLU Variants Improve Transformer》论文中提出，并被 PaLM、LLaMA、Mistral、Gemma 和 Qwen 所采用。

SwiGLU 公式如下：

```text
gate = SiLU(W_gate · x)         ←  "swish" 门
up   = W_up · x                 ←  值路径
output = W_down · (gate ⊙ up)   ←  门调制值，然后向下投影
```

这需要**三个**权重矩阵，而不是两个：`W_gate`、`W_up` 和 `W_down`。额外的矩阵是门控机制的代价，但性能提升是值得的。

### SiLU 激活函数

SiLU（Sigmoid 线性单元），也称为"Swish"函数，定义为：

```text
SiLU(x) = x · sigmoid(x) = x / (1 + e^(-x))
```

让我们计算几个输入值的 SiLU 以理解其形状：

| x    | sigmoid(x) | SiLU(x)    |
|------|------------|------------|
| -3.0 | 0.0474     | -0.1422    |
| -2.0 | 0.1192     | -0.2384    |
| -1.0 | 0.2689     | -0.2689    |
| -0.5 | 0.3775     | -0.1888    |
|  0.0 | 0.5000     |  0.0000    |
|  0.5 | 0.6225     |  0.3112    |
|  1.0 | 0.7311     |  0.7311    |
|  2.0 | 0.8808     |  1.7616    |
|  3.0 | 0.9526     |  2.8578    |
|  5.0 | 0.9933     |  4.9665    |
| 10.0 | 0.9999     |  9.9995    |

关于 SiLU 的关键观察：
- **对于较大的正 x**：SiLU(x) 趋近于 x（因为 sigmoid(x) 趋近于 1）。函数变得几乎线性，就像 ReLU 一样。
- **在 x = 0 处**：SiLU(0) = 0，与 ReLU 相同。
- **对于负的 x**：SiLU(x) 略微变为负值（在 x ≈ -1.77 附近达到约 -0.28），然后从下方趋近于 0。这是*非单调的*——函数变为负值然后回到零附近。ReLU 只是简单地截断为零。
- **处处平滑**：与 ReLU 在 x = 0 处有尖角不同，SiLU 是一条平滑（无限可微）的曲线。这意味着在反向传播过程中梯度流是平滑的——导数没有不连续性。

### 比较 SiLU 和 ReLU

```
ReLU:                   SiLU:
    |                       |
  3 |       /               |      .--
  2 |      /                |    .-
  1 |     /                 |  .-
  0 |____/_____             |._/ ._____
    |                        |  /
 -1 |                        | /
    |                        |.     (轻微低于 0)
```

ReLU 是一个硬斜坡：下方为零，上方为恒等。SiLU 是一个平滑过渡的软斜坡，带有轻微的负向倾斜。平滑性和自门控特性（门值取决于输入幅度）使 SiLU 在深度网络中具有更好的经验表现。

---

## 5. 为什么选择 SwiGLU 而非 ReLU？

从 ReLU FFN 到 SwiGLU FFN 的转变是现代大语言模型架构中最明确的经验改进之一。原因如下：

### 平滑的梯度流

ReLU 对所有负输入都有零梯度。这意味着任何预激活值为负的神经元都接收不到梯度，无法更新——即"死神经元"问题。在一个拥有数百万神经元的深度网络中，相当大比例的神经元可能永久卡在零输出。

SiLU 在所有地方都有非零梯度（负无穷处除外）。即使对于中等程度的负输入，梯度虽然小但非零，因此每个神经元都有可能恢复。这带来了更好的训练优化动态。

### 自门控

在 SwiGLU 公式中，门是 `SiLU(W_gate · x)`。因为 SiLU(x) = x * sigmoid(x)，门值自然随输入幅度缩放。对于较大的正输入，门完全打开（SiLU(x) ~ x）。对于接近零的输入，门几乎关闭。这种依赖于输入的门控比 ReLU 在零处的固定阈值更加灵活。

### 经验证据

原始的 GLU 论文（Shazeer 2020）在语言建模基准上测试了许多 GLU 变体（使用 ReLU、GELU、Swish/SiLU 作为门激活）。SwiGLU 始终优于标准 ReLU FFN。这一结果在 PaLM（540B 参数）的大规模实验中得到证实，随后被 LLaMA 采用，使 SwiGLU 成为开放权重模型的标准。

如今，每个主要的开放大语言模型系列都使用 SwiGLU 或类似的变体：
- **LLaMA**（Meta）—— SwiGLU
- **Qwen**（阿里巴巴）—— SwiGLU
- **Mistral**（Mistral AI）—— SwiGLU
- **Gemma**（Google）—— GeGLU（GELU 门控，非常相似）
- **Phi**（Microsoft）—— SwiGLU

SwiGLU FFN 已经变得和多头注意力机制本身一样标准。

---

## 6. 具体计算示例

让我们用小的、可手工计算的数字来完成完整的 SwiGLU 计算过程。我们将使用 `hidden_size = 2` 和 `intermediate_size = 3`。

### 设置

```text
Input x = [1.0, 2.0]                       shape [2] (1 token, 2 features)

gate_proj = [[1, 0],                        shape [3, 2]
             [0, 1],
             [1, 1]]

up_proj   = [[2, 0],                        shape [3, 2]
             [0, 2],
             [1, -1]]

down_proj = [[1, 0, 0],                     shape [2, 3]
             [0, 1, 0]]
```

### 步骤 1：门路径——投影并应用 SiLU

首先，通过 gate_proj 投影 x 来计算预激活值：

```text
gate_pre = x · gate_proj^T

gate_proj^T = [[1, 0, 1],
               [0, 1, 1]]

gate_pre = [1.0, 2.0] · [[1, 0, 1],
                          [0, 1, 1]]

gate_pre[0] = 1*1 + 2*0 = 1.0
gate_pre[1] = 1*0 + 2*1 = 2.0
gate_pre[2] = 1*1 + 2*1 = 3.0

gate_pre = [1.0, 2.0, 3.0]
```

现在逐元素应用 SiLU：

```text
gate[0] = SiLU(1.0) = 1.0 * sigmoid(1.0) = 1.0 / (1 + e^(-1)) = 1.0 * 0.7311 = 0.7311
gate[1] = SiLU(2.0) = 2.0 * sigmoid(2.0) = 2.0 / (1 + e^(-2)) = 2.0 * 0.8808 = 1.7616
gate[2] = SiLU(3.0) = 3.0 * sigmoid(3.0) = 3.0 / (1 + e^(-3)) = 3.0 * 0.9526 = 2.8578

gate = [0.7311, 1.7616, 2.8578]
```

### 步骤 2：上路径——投影（无激活）

```text
up = x · up_proj^T

up_proj^T = [[2, 0, 1],
             [0, 2, -1]]

up[0] = 1*2 + 2*0 = 2.0
up[1] = 1*0 + 2*2 = 4.0
up[2] = 1*1 + 2*(-1) = -1.0

up = [2.0, 4.0, -1.0]
```

### 步骤 3：逐元素乘法（门调制上路径）

```text
gated = gate ⊙ up

gated[0] = 0.7311 * 2.0  = 1.4622
gated[1] = 1.7616 * 4.0  = 7.0464
gated[2] = 2.8578 * (-1.0) = -2.8578

gated = [1.4622, 7.0464, -2.8578]
```

这里请注意一件重要的事情：即使上路径有一个负值（位置 2 处的 -1.0），位置 2 处的门值很大且为正（2.8578），所以位置 2 处的门控结果为 -2.8578。门并不是简单地抑制负值——它调制上路径产生的任何值。输出的符号取决于上路径；门控制的是*幅度*。

### 步骤 4：下投影——投影回 hidden_size

```text
output = gated · down_proj^T

down_proj^T = [[1, 0],
               [0, 1],
               [0, 0]]

output[0] = 1.4622 * 1 + 7.0464 * 0 + (-2.8578) * 0 = 1.4622
output[1] = 1.4622 * 0 + 7.0464 * 1 + (-2.8578) * 0 = 7.0464

output = [1.4622, 7.0464]
```

最终输出的形状与输入相同：`[2]`（一个 token，两个特征）。我们这里选择的 down_proj 只使用了前两个中间特征（第三列全为零），所以第三个中间维度实际上被丢弃了。

### 数据流总结

```text
x          [1.0, 2.0]                    hidden_size=2
              |
         +----+----+
         |         |
     gate_proj   up_proj
         |         |
    gate_pre     up
    [1,2,3]     [2,4,-1]                intermediate_size=3
         |         |
       SiLU        |
         |         |
      gate         |
   [0.73,1.76,2.86]|
         |         |
         +----+----+
              |
          gate ⊙ up
       [1.46, 7.05, -2.86]
              |
          down_proj
              |
        output [1.46, 7.05]              hidden_size=2
```

---

## 7. Qwen3-0.6B 中的参数量

现在让我们看看实际数字。Qwen3-0.6B 具有：
- `hidden_size = 1024`
- `intermediate_size = 3072`（3 倍扩展）
- `num_hidden_layers = 28`
- FFN 中没有偏置项（现代大语言模型的标准）

### 每个 FFN

每个 FFN 层有三个权重矩阵：

| Matrix     | Shape           | Parameters |
|------------|-----------------|------------|
| gate_proj  | [3072, 1024]    | 3,145,728  |
| up_proj    | [3072, 1024]    | 3,145,728  |
| down_proj  | [1024, 3072]    | 3,145,728  |
| **Total**  |                 | **9,437,184** |

每个矩阵恰好贡献 3,145,728 个参数（3072 * 1024），每个 FFN 总计约 944 万参数。

### 所有 FFN 合计

```text
28 layers * 9,437,184 params/layer = 264,241,152 total FFN parameters
```

这大约是 **2.642 亿参数**，约占模型总参数量（约 5.96 亿）的 46%。FFN 是模型中最大的单一组件——比注意力层更大，比嵌入层更大。

### 与其他组件的比较

| Component                | Parameters  | Percentage |
|--------------------------|-------------|------------|
| Token embedding          | ~156M       | ~27%       |
| 28x Attention layers     | ~176M       | ~30%       |
| 28x FFN layers           | ~264M       | ~46%       |
| Output norm + lm_head    | ~1K        | ~0%        |

FFN 在参数量上的主导地位，正是为什么通过诸如混合专家（MoE）等技术优化 FFN——即每个 token 只激活 FFN 的一个子集——可以显著降低推理成本。Mixtral 和 Qwen3-MoE 等模型就使用了这种方法。

### 为什么是 3 倍扩展？

3 倍扩展比率（intermediate_size = 3 * hidden_size）是 SwiGLU 架构特有的。原始 Transformer 使用 4 倍扩展和 2 矩阵 FFN。SwiGLU 增加了第三个矩阵（门），所以为了保持总参数量大致相同，扩展比率从 4 倍降低到约 4 倍 * 2/3 = 8/3 ≈ 2.67 倍。在实践中，Qwen3 使用整洁的 3 倍，这比等效的 2 矩阵 4 倍 FFN 略多一些参数：

```text
2-matrix FFN (4x):   2 * hidden * (4 * hidden) = 8 * hidden^2
3-matrix FFN (3x):   3 * hidden * (3 * hidden) = 9 * hidden^2
```

因此，3 倍扩展的 SwiGLU FFN 比标准 4 倍 FFN 多使用约 12.5% 的参数，但性能提升足以证明这一成本是合理的。

---

## 8. 实现细节

以下是我们的 SwiGLU FFN 的 Rust 实现：

```rust
pub struct FeedForward {
    gate_proj: Tensor,  // [intermediate_size, hidden_size] = [3072, 1024]
    up_proj: Tensor,    // [intermediate_size, hidden_size] = [3072, 1024]
    down_proj: Tensor,  // [hidden_size, intermediate_size] = [1024, 3072]
}

impl FeedForward {
    pub fn new(gate_proj: Tensor, up_proj: Tensor, down_proj: Tensor) -> Self {
        // Validate dimensions...
        Self { gate_proj, up_proj, down_proj }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Step 1: Gate path — project then activate with SiLU.
        let gate = x.matmul(&self.gate_proj.transpose_2d()).silu();

        // Step 2: Up path — project (no activation).
        let up = x.matmul(&self.up_proj.transpose_2d());

        // Step 3: Element-wise multiply — gate modulates up path.
        let gated = gate.mul_elementwise(&up);

        // Step 4: Down projection — project back to hidden_size.
        gated.matmul(&self.down_proj.transpose_2d())
    }
}
```

### 理解转置

来自 safetensors 的权重矩阵存储为 `[out_features, in_features]`，这是 PyTorch 的惯例。对于一个计算 `y = x · W^T` 的线性层，权重 `W` 的形状为 `[out_dim, in_dim]`。由于我们的 `matmul` 期望 `[M, K] × [K, N]`，我们需要先转置 `W`：

```text
x:     [seq_len, hidden_size]       = [seq_len, 1024]
W^T:   [hidden_size, intermediate]  = [1024, 3072]
result: [seq_len, intermediate]     = [seq_len, 3072]
```

如果不进行转置，维度将无法对齐进行 matmul。`transpose_2d()` 方法交换行和列：`[M, N]` 变为 `[N, M]`。

### 四步计算

`forward` 方法准确地实现了 SwiGLU 公式：

1. **门路径**：`x.matmul(&gate_proj.transpose_2d())` 计算 `W_gate · x^T`，然后 `.silu()` 应用 SiLU 激活。这产生控制信息流的门信号。

2. **上路径**：`x.matmul(&up_proj.transpose_2d())` 计算 `W_up · x^T`。不应用激活——这是原始值信号。

3. **门调制**：`gate.mul_elementwise(&up)` 执行逐元素乘法。门值大的地方，上路径值通过；门值接近零的地方，值被抑制。

4. **下投影**：`gated.matmul(&down_proj.transpose_2d())` 将中间表示投影回隐藏维度，产生最终输出。

### 为什么没有偏置？

现代大语言模型（LLaMA、Qwen、Mistral、Gemma）省略了 FFN 线性层中的偏置项。原因是：
- 与权重矩阵相比，偏置增加的参数相对较少
- 在 FFN 之前应用了 RMSNorm（pre-norm 架构），归一化已经将激活值居中，减少了对偏置的需求
- 经验表明，移除偏置不会损害性能，并且简化了实现

### 构造函数验证

`new` 构造函数验证三个权重矩阵具有兼容的维度：
- `gate_proj` 和 `up_proj` 必须具有相同的形状 `[intermediate_size, hidden_size]`
- `down_proj` 必须具有形状 `[hidden_size, intermediate_size]`——即 `gate_proj` 的反向

这些检查可以在早期捕获配置错误，避免在 `forward()` 期间产生令人困惑的维度不匹配。

### 测试验证的关键属性

我们的测试套件验证了几个重要属性：

1. **输出形状与输入形状匹配**：经过所有的扩展和收缩后，FFN 输出的形状与接收的形状相同。这对于 Transformer 块中的残差连接至关重要：`output = x + FFN(x)`。

2. **SiLU 门行为**：对于具有正权重的正数输入，预激活值为正，而正值的 SiLU 始终为正。因此门是一个平滑的非负调制器。

3. **无跨 token 交互**：输入两个相同的 token 会产生两个相同的输出行。FFN 独立处理每个 token——不存在跨位置通信。那是注意力的工作。

4. **不同的输入产生不同的输出**：一个基本的健全性检查，确保计算不会退化为常数函数。

5. **手工计算的值**：一个使用手动计算预期输出的小维度测试，确保算术从头到尾都是正确的。

---

## 总结

SwiGLU FFN 是一个看似简单的组件：三次矩阵乘法、一个激活函数、一次逐元素乘法。但这个简单的配方占据了 Qwen3 近一半的参数，并提供了模型主要的非线性处理能力。

与标准 ReLU FFN 相比，关键创新在于：
1. **门控**——gate_proj 路径学习依赖于输入的特征选择，而不是应用固定阈值
2. **SiLU 激活**——平滑梯度、无死神经元、自门控行为
3. **三个投影**——额外的矩阵为门控机制提供了自己的可学习参数

在完整的 Transformer 块中，FFN 位于注意力之后，并包裹在残差连接中：

```text
output = x + FFN(RMSNorm(x + Attention(RMSNorm(x))))
```

残差连接使 FFN 能够学习对 token 表示的增量修改，而 pre-norm（每个子层之前的 RMSNorm）则稳定了输入的幅度。这些设计选择共同使现代 Transformer 在规模化时既强大又可训练。
