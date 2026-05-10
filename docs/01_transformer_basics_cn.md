# 1. Transformer 基础知识：大语言模型如何工作

本文档是理解现代大语言模型（LLM）工作原理的起点。我们将从基础概念开始，一步一步构建知识体系，到最后你将完全理解本项目所实现的 Qwen3-0.6B 模型的每一个组件。

无需预先了解 Transformer。你只需要熟悉基础线性代数（向量、矩阵、矩阵乘法）以及 Python 或 Rust 语法，能够阅读代码片段即可。

---

## 1. 什么是 Transformer？

### 1.1 改变一切的论文

2017 年 6 月，Google 的一组研究人员发表了一篇题为**"Attention Is All You Need"**（Vaswani 等人，2017）的论文。它引入了一种称为**Transformer**的神经网络架构。当时，序列建模——如机器翻译、语音识别和文本生成等任务——的主流方法是**循环神经网络（RNN）**及其改进变体**长短期记忆网络（LSTM）**。

Transformer 旨在解决 RNN 的两个根本问题：

**问题 1：顺序处理阻碍并行化。**
RNN 一次处理序列中的一个步骤。要计算位置 t 的隐藏状态，必须先计算位置 t-1 的隐藏状态，而这又需要 t-2，依此类推。这种串行依赖意味着在训练期间无法跨时间步并行化。对于一个包含 1,000 个 token 的句子，你必须等待 999 次顺序计算才能处理最后一个 token。在现代 GPU 上，GPU 擅长并行计算，这种串行处理成为巨大的瓶颈。

**问题 2：长距离依赖难以学习。**
即使使用专门设计来缓解此问题的 LSTM，长序列中早期位置的信息在 RNN 到达后期位置时也往往会被"冲刷掉"。如果位置 500 的代词指代位置 5 的名词，RNN 必须通过 495 个中间步骤传递该信息。在实践中，允许网络学习这些连接的梯度信号要么消失（变为零）要么爆炸（变得巨大），导致训练不稳定。

Transformer 同时解决了这两个问题：
- 它在训练期间**并行处理所有位置**，因为时间步之间没有顺序依赖。
- 它使用**自注意力（self-attention）**，在每对位置之间创建直接连接，无论它们相距多远。位置 500 可以在一次操作中关注位置 5，没有任何信息退化。

### 1.3 Transformer 的三种变体

自原始论文以来，Transformer 架构已经分化为三个主要变体。理解这些差异很重要，因为它们服务于不同的目的。

**编码器-解码器（原始架构，如 2017 年论文所述）：**
模型有两个堆栈。**编码器**读取完整的输入序列并为每个 token 生成上下文化表示。**解码器**随后一次生成一个输出 token，同时关注自己之前生成的 token（因果方式）和编码器的表示（交叉注意力）。这种设计天然适合序列到序列的任务，如翻译：将英语句子编码，然后解码为法语。

```
English: "The cat sat" ──► [Encoder] ──► context vectors
                                              │
                                              ▼ (cross-attention)
French:  "<s>" ──► [Decoder] ──► "Le" ──► [Decoder] ──► "chat" ──► ...
```

**仅编码器（如 BERT，2018）：**
只保留编码器堆栈。每个 token 可以双向关注所有其他 token。这产生了丰富的双向表示，非常适合理解任务：分类、命名实体识别、问答（给定段落）等。BERT **不是**文本生成器——它是文本理解器。你遮蔽一些 token，训练它根据上下文预测这些 token。

**仅解码器（如 GPT 系列、LLaMA、Qwen）：**
只保留解码器堆栈，但有一个关键修改：没有交叉注意力（因为没有编码器），且自注意力是**因果的**——每个 token 只能关注自身及其之前的 token。这种限制是必要的，因为模型一次生成一个 token，绝不能"偷看"未来的 token。仅解码器模型通过预测下一个 token 进行训练，事实证明这是一个极其强大的训练目标。在足够的数据和计算资源下，这个简单的目标可以训练出能够写文章、编写代码并推理复杂问题的模型。

```
Input:  "The cat sat on"
         │   │   │   │
         ▼   ▼   ▼   ▼
       ┌───────────────────┐
       │  Causal Self-Attn  │  ← 每个 token 只能看到更早的 token
       │  FFN               │
       │  ... (N layers)    │
       └───────────────────┘
         │   │   │   │
         ▼   ▼   ▼   ▼
       logits for each position
                          │
                          ▼
               predict: "the" (next token)
```

**本项目实现的是仅解码器 Transformer**——具体来说，是 Qwen3-0.6B 模型。从此处开始，当我们说"Transformer"时，指的是仅解码器变体。

---

## 2. 大局观：Transformer 如何处理文本？

让我们走一遍完整的流程，从一段文本字符串到预测的下一个 token。我们将使用一个具体示例：输入文本为 `"Hello world"`。

### 2.1 步骤 1：文本转换为 Token ID（分词器）

计算机不理解文本，它们理解数字。第一步是将字符串转换为称为**token ID**的整数序列。

**分词器（tokenizer）**将文本分割为称为**token**的子词单元，并将每个 token 映射到固定词汇表中的整数 ID。Qwen3 使用字节对编码（BPE）分词器，词汇表大小为 151,936 个 token。

```
Text:     "Hello world"
           │         │
           ▼         ▼
Tokens:   [Hello]   [world]
           │         │
           ▼         ▼
Token IDs: [15496]   [995]
```

实际上，根据 BPE 合并规则，分词可能会以不同方式分割单词。"Hello"可能是一个 token，也可能被分割为 "He" + "llo"。详细信息请参见 `02_tokenizer.md`。现在，只需将其视为查找操作：每个文本片段获得一个数字。

### 2.2 步骤 2：Token ID 转换为嵌入向量

像 15496 这样的 token ID 只是一个数字——它没有内在含义。数字 15496 在语义上并不"接近"15497。我们需要将这些离散的 ID 转换为能够捕捉语义关系的连续向量。

**嵌入表**（也称为嵌入矩阵）是一个巨大的查找表。它每行对应一个词汇表条目，每行是一个大小为 `hidden_size` 的向量（Qwen3-0.6B 为 1,024）。要嵌入一个 token ID，只需查找对应的行。

```
Embedding Table (shape: [151936, 1024])
┌──────────────────────────────┐
│ Row 0:    [0.01, -0.03, ...] │  ← token ID 0
│ Row 1:    [0.12,  0.05, ...] │  ← token ID 1
│ ...                          │
│ Row 15496: [0.33, -0.17, ...]│  ← "Hello"
│ ...                          │
│ Row 995:  [-0.08, 0.22, ...] │  ← "world"
│ ...                          │
│ Row 151935: [...]            │  ← last token
└──────────────────────────────┘

Token IDs: [15496, 995]
                │      │
                ▼      ▼
Embeddings: [vec_15496, vec_995]   (shape: [2, 1024])
```

在此步骤之后，我们得到一个向量序列，每个 token 对应一个，每个向量长度为 1,024。这些向量是在训练期间学习的——出现在相似上下文中的 token 最终会得到相似的嵌入向量。

### 2.3 步骤 3：Transformer 块（核心）

这是魔法发生的地方。嵌入向量序列通过一堆**Transformer 块**（也称为层）。Qwen3-0.6B 有 28 个这样的块。每个块将其输入转换为更丰富、更上下文化的表示。

```
Embeddings: [vec_15496, vec_995]   (shape: [2, 1024])
                │
                ▼
       ┌─────────────────┐
       │  Block 0         │
       └─────────────────┘
                │
                ▼
       ┌─────────────────┐
       │  Block 1         │
       └─────────────────┘
                │
                ▼
               ...
                │
                ▼
       ┌─────────────────┐
       │  Block 27        │
       └─────────────────┘
                │
                ▼
Hidden states: [h_0, h_1]   (shape: [2, 1024])
```

每个块接收一个向量序列，并输出一个形状相同的向量序列。每个块内部的转换涉及两个子层：**自注意力**（让 token 之间相互通信）和**前馈网络**（独立处理每个 token）。我们将在第 3 节详细研究这些内容。

核心洞察在于**每个块都添加更多上下文**。在块 0 之后，"world"的表示可能编码了它跟在"Hello"之后。在块 27 之后，表示可能编码了"Hello world"在上下文中的完整语义——它是一个问候语，它是一个著名的编程传统，等等。

### 2.4 步骤 4：最终隐藏状态转换为 Logits（lm_head）

在最后一个 Transformer 块之后，我们有一系列隐藏状态向量。为了预测下一个 token，我们取**最后一个位置**的隐藏状态（我们将在第 4 节关于自回归生成中解释原因），并使用称为 **lm_head** 的线性层将其投影回词汇表空间。

lm_head 是一个形状为 `[vocab_size, hidden_size]` = `[151936, 1024]` 的矩阵。将一个隐藏状态向量（1,024 维）乘以该矩阵，产生一个包含 151,936 个数字的向量，称为 **logits**。每个 logit 对应词汇表中的一个 token，logit 越高意味着模型认为该 token 越有可能出现在下一个位置。

```
Last hidden state: [h_1]   (shape: [1, 1024])
                       │
                       ▼
              lm_head (shape: [151936, 1024])
                       │
                       ▼
Logits: [l_0, l_1, ..., l_151935]   (shape: [151936])

  l_0     = score for token ID 0    (可能非常低)
  l_15496 = score for "Hello"       (可能中等)
  l_3140  = score for "!"           (可能很高)
  ...
```

### 2.5 步骤 5：Logits 转换为下一个 Token（采样）

Logits 是原始分数，不是概率。要将 logits 转换为概率，我们应用 **softmax** 函数：

```
P(token_i) = exp(logit_i) / sum(exp(logit_j) for all j)
```

经过 softmax 后，每个概率都在 0 和 1 之间，且它们的总和为 1。

选择下一个 token 最简单的方法是**贪心解码（greedy decoding）**：总是选择概率最高的 token。但这会产生无聊、重复的文本。在实践中，我们使用引入可控随机性的**采样**方法：

- **温度（Temperature）**：在 softmax 之前将 logits 除以一个值 T。T < 1 使分布更尖锐（更自信）；T > 1 使分布更平坦（更随机）。
- **Top-k**：只考虑 k 个最可能的 token，将其余的置零。
- **Top-p（核采样，nucleus）**：只考虑累积概率超过 p 的最小 token 集合。

```
Logits: [..., 2.1, 0.5, 5.3, 1.8, ...]
                    │
                    ▼  softmax
Probabilities: [..., 0.03, 0.005, 0.81, 0.02, ...]
                                │
                                ▼  sample
Selected token: ID 3140 ("!")
```

### 2.6 一览完整流程

```
 ┌──────────┐     ┌───────────┐     ┌─────────────────┐     ┌─────────┐     ┌──────────┐
 │  Text    │────►│ Tokenizer │────►│  Embedding      │────►│ 28 x    │────►│ lm_head  │
 │"Hello    │     │ (BPE)     │     │  Table lookup   │     │ Blocks  │     │ Linear   │
 │  world"  │     │           │     │                 │     │         │     │          │
 └──────────┘     └───────────┘     └─────────────────┘     └─────────┘     └──────────┘
                        │                   │                    │                │
                        ▼                   ▼                    ▼                ▼
                   [15496, 995]       [2, 1024]            [2, 1024]        [151936]
                   Token IDs          Embeddings           Hidden states     Logits
                                                                  │
                                                                  ▼
                                                           ┌──────────┐
                                                           │ Sampling │
                                                           │ (softmax │
                                                           │  + pick) │
                                                           └──────────┘
                                                                  │
                                                                  ▼
                                                           Next token ID
                                                           (e.g., 3140 → "!")
```

**形状总结**（输入为 "Hello world"，2 个 token）：

| 阶段               | 形状           | 描述                           |
|---------------------|-----------------|---------------------------------------|
| Token ID           | [2]             | 每个 token 一个整数                 |
| 嵌入          | [2, 1024]       | 每个 token 一个 1024 维向量         |
| 每个块之后    | [2, 1024]       | 相同形状（块保持形状）    |
| lm_head 之后       | [2, 151936]     | 每个位置每个词汇条目一个 logit |
| 下一个 token logits   | [151936]        | 仅最后一个位置的 logits      |

---

## 3. Transformer 块内部是什么？

现在让我们放大观察单个 Transformer 块。这是核心计算单元，在 Qwen3-0.6B 中重复 28 次。理解一个块就足以理解整个模型。

### 3.1 块架构

仅解码器 Transformer 块（遵循 LLaMA/Qwen 风格）具有以下结构：

```
Input x
│
├─── RMSNorm ──────────────────────────┐
│                                      ▼
│                           ┌──────────────────┐
│                           │  Self-Attention   │
│                           │  (with RoPE &     │
│                           │   causal mask &   │
│                           │   KV cache)       │
│                           └──────────────────┘
│                                      │
├────────────────────────────────────── +  ◄── 残差连接
│                                      │
│                            x' = x + attn_out
│                                      │
├─── RMSNorm ──────────────────────────┐
│                                      ▼
│                           ┌──────────────────┐
│                           │  FFN (SwiGLU)    │
│                           │  gate = SiLU(W_g·x')
│                           │  up   = W_u·x'
│                           │  out  = W_d·(gate * up)
│                           └──────────────────┘
│                                      │
├────────────────────────────────────── +  ◄── 残差连接
│
│                         output = x' + ffn_out
│
▼
Output (same shape as input)
```

让我们逐步了解每个组件。

### 3.2 自注意力：让 Token 相互通信

自注意力是 Transformer 的核心。它允许序列中的每个 token "查看"其他每个 token，并决定在多大程度上关注（聚焦）它。

**类比**：想象你在读句子"猫坐在垫子上，因为它累了。"当你遇到单词"它"时，你会本能地回头看前面的词，弄清楚"它"指的是什么。你更多地关注"猫"而不是"垫子"，因为"猫"更可能累了。自注意力做类似的事情：对于每个 token，它计算其他 token 表示的加权平均，权重取决于每个 token 的相关性。

**工作原理**（简化版，单头）：

对于每个 token，我们从其当前表示中计算三个向量：

- **Query（Q）**："我在寻找什么？"——表示这个 token 想要什么信息。
- **Key（K）**："我包含什么？"——表示这个 token 提供什么信息。
- **Value（V）**："这是我的内容。"——要聚合的实际信息。

token i 和 token j 之间的注意力分数是 Q_i 和 K_j 的点积，除以 key 维度的平方根（以保持数值稳定）。这些分数经过 softmax 得到注意力权重，然后 token i 的输出是所有 Value 向量的加权和：

```
attention_score(i, j) = (Q_i . K_j) / sqrt(d_k)

attention_weight(i, j) = softmax_j(attention_score(i, :))

output_i = sum_j(attention_weight(i, j) * V_j)
```

在仅解码器（因果）变体中，我们还应用**因果掩码（causal mask）**，防止 token i 关注 token j > i（未来的 token）。这是通过在 softmax 之前将未来位置的注意力分数设置为负无穷大来实现的，在 softmax 之后它们变为零。

```
Causal Mask (4 个 token):

       Position:  0   1   2   3
Token 0 attends: [ok,  X,  X,  X]    ← 只能看到自己
Token 1 attends: [ok, ok,  X,  X]    ← 可以看到 0 和 1
Token 2 attends: [ok, ok, ok,  X]    ← 可以看到 0, 1, 2
Token 3 attends: [ok, ok, ok, ok]    ← 可以看到所有

X = 被遮蔽（在 softmax 之前设置为 -inf，之后变为 0）
```

**多头注意力（Multi-Head Attention）**：模型不是计算单组 Q、K、V，而是并行计算多组，每组称为一个**注意力头**。每个头可以学习关注不同类型的关系——一个头可能关注句法关系（主谓一致），另一个关注共指（"它"指代什么），另一个关注位置邻近性，等等。

Qwen3-0.6B 使用 **16 个查询头**和 **8 个键值头**。这称为**分组查询注意力（GQA）**，其中多个查询头共享相同的 key 和 value 头。具体来说，每 2 个查询头共享 1 个 KV 头。GQA 在质量损失最小的情况下减少了 KV 缓存的内存使用和计算。详细信息请参见 `06_attention.md`。

### 3.3 FFN（前馈网络）：处理每个 Token

自注意力在 token 之间混合信息后，FFN 独立处理每个 token 的表示。它以相同的方式应用于每个位置——使用相同的权重，但处理不同的输入向量。

Qwen3 使用 **SwiGLU** 变体的 FFN，它有三条权重矩阵，而不是传统的两条：

```
Traditional FFN:     output = W_2 * ReLU(W_1 * x)

SwiGLU FFN:          gate = SiLU(W_gate * x)    ← 门控路径
                     up   = W_up * x             ← 上投影路径
                     output = W_down * (gate * up)
```

其中 **SiLU**（Sigmoid Linear Unit，也称为 Swish）定义为：
```
SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

直观理解：门控路径决定*哪些*信息可以通过，上投影路径提供*什么*信息。它们之间的逐元素乘法创建了一个选择性过滤机制。

**维度**：FFN 的输入和输出都是 `hidden_size = 1,024`。中间（上投影）维度是 `intermediate_size = 3,072`，是隐藏大小的 3 倍。这种扩展允许网络在投影回较小维度之前学习更丰富的内部表示。

```
x (1,024)
│
├─── W_gate (3,072 x 1,024) ──► (3,072) ──► SiLU ──┐
├─── W_up   (3,072 x 1,024) ──► (3,072)             ├──► element-wise * ──► (3,072)
│                                                     │
│                                  W_down (1,024 x 3,072)
│                                                     │
└─────────────────────────────────────────────────────┘
                                                      │
                                                      ▼
                                              output (1,024)
```

### 3.4 残差连接：帮助梯度流动

**残差连接**（也称为跳跃连接）将块的输入直接添加到其输出：

```
output = x + sublayer(x)
```

子层不需要从头开始产生整个输出，它只需要学习**残差**——与输入的差异。这有两个主要好处：

1. **训练期间的梯度流动**：在反向传播期间，梯度可以直接通过加法操作流动，完全绕过子层。这防止了深度网络中的梯度消失问题。如果没有残差连接，28 层的网络将极其难以训练。

2. **增量细化**：每个 Transformer 块只需要学习对其输入的小修改。早期层可以学习基本模式，后期层可以在这些模式的基础上学习更复杂的关系。

在我们的 Transformer 块中，有两个残差连接：
```
x' = x + self_attention(rmsnorm(x))
output = x' + ffn(rmsnorm(x'))
```

输入始终被保留并加回。子层只产生修正。

### 3.5 RMSNorm：激活归一化

**RMSNorm**（均方根归一化）是 LayerNorm 的简化变体。两者的目的相同：它们归一化激活，防止其在前向传播过程中变得过大或过小，从而稳定训练。

LayerNorm 通过均值和方差进行归一化：
```
LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
```

RMSNorm 只使用均方根，速度更快：
```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
```

其中 `eps` 是一个小常数（Qwen3 中为 1e-6），用于防止除以零，`gamma` 是一个形状为 `[hidden_size]` 的可学习参数，独立缩放每个维度。

关键区别：RMSNorm 不减去均值，也没有可学习的偏置（beta）。这使得它在实现类似性能的同时计算成本更低。这个设计选择遵循了 LLaMA 系列模型。

归一化应用在每个子层**之前**（这称为"pre-norm"架构），这比原始"post-norm"架构（归一化应用在子层之后）更稳定。

---

## 4. 自回归生成

### 4.1 一次一个 Token

仅解码器 Transformer **自回归地**生成文本：一次一个 token。给定一系列 token，它预测下一个 token。然后，给定序列加上预测的 token，它预测再下一个，依此类推。

这与模型在**训练**期间（所有位置并行计算）和**推理**期间（必须顺序生成，因为每个新 token 依赖于之前的预测）处理文本的方式根本不同。

```
Step 0: Input:  "Hello world"          → Predict: "!"
Step 1: Input:  "Hello world !"        → Predict: "How"
Step 2: Input:  "Hello world ! How"    → Predict: "are"
Step 3: Input:  "Hello world ! How are" → Predict: "you"
...
```

### 4.2 为什么是"因果"？

"因果"一词来自时间中的因果概念：现在可以受过去影响，但不能受未来影响。在因果语言模型中，位置 t 的预测只能依赖于位置 0, 1, ..., t-1。

这不仅仅是一个设计偏好——它是自回归生成的**必要条件**。当模型生成 token t 时，token t+1、t+2 等还不存在。因果掩码确保模型永远不会学会依赖未来信息，因为在生成过程中它永远无法访问未来信息。

在训练期间，我们可以并行处理整个序列，但对自注意力应用因果掩码，这样每个位置只能看到之前的位置。这让我们既获得了并行训练的好处，又保持了生成所需的因果属性。

### 4.3 KV 缓存：避免冗余计算

考虑在没有优化的情况下自回归生成时会发生什么：

```
Step 0: Process tokens [0, 1, 2, 3]  → predict token 4
Step 1: Process tokens [0, 1, 2, 3, 4]  → predict token 5
Step 2: Process tokens [0, 1, 2, 3, 4, 5]  → predict token 6
```

在每个步骤中，我们重新计算**所有**先前 token 的 Key 和 Value 向量，即使它们没有变化。token 0 的 Key 和 Value 在步骤 0、步骤 1 和步骤 2 中完全相同——唯一变化的是我们有了新 token 的新 Query。

**KV 缓存**解决了这个低效问题。在处理每个 token 之后，我们缓存其 Key 和 Value 向量。在下一步，我们只需要：

1. 仅为**新 token** 计算 Q、K、V。
2. 将新的 K 和 V 追加到缓存。
3. 使用新的 Q 对**所有缓存的 K 和 V**（包括新的）计算注意力。

```
Without KV Cache:                     With KV Cache:
Step 0: Compute K,V for [0,1,2,3]    Step 0: Compute K,V for [0,1,2,3], cache them
Step 1: Compute K,V for [0,1,2,3,4]  Step 1: Compute K,V for [4] only, append to cache
Step 2: Compute K,V for [0,1,2,3,    Step 2: Compute K,V for [5] only, append to cache
              4,5]
Cost without cache: O(n^2) total      Cost with cache: O(n) total
```

KV 缓存在生成过程中大幅减少了计算。权衡的是内存：我们必须为每个过去的 token 存储 Key 和 Value 向量。对于长对话，这个缓存可能变得相当大。这就是为什么 GQA（分组查询注意力）对 Qwen3 很重要——通过只有 8 个 KV 头而不是 16 个，缓存比标准多头注意力小 50%。

### 4.4 生成循环（伪代码）

以下是完整的自回归生成循环，为清晰起见进行了简化：

```
function generate(prompt_tokens, max_tokens):
    # Step 1: 一次性处理整个提示（预填充）
    hidden = forward_pass(prompt_tokens)      # shape: [seq_len, hidden_size]
    logits = lm_head(hidden[-1])              # take last position's output
    next_token = sample(logits)               # apply temperature, top-k, top-p
    
    # Initialize KV cache with prompt's keys and values
    kv_cache = get_kv_cache_from_forward_pass()
    
    generated = [next_token]
    
    # Step 2: 一次生成一个 token（解码）
    for i in range(max_tokens - 1):
        # Process only the new token
        hidden = forward_pass([next_token], kv_cache=kv_cache)
        logits = lm_head(hidden[0])
        next_token = sample(logits)
        
        # Update KV cache with new token's keys and values
        kv_cache.append(new_keys, new_values)
        
        generated.append(next_token)
        
        if next_token == EOS_TOKEN:
            break
    
    return generated
```

生成有两个阶段：

- **预填充（Prefill）**：一次性处理整个提示。这是一大批工作，但只发生一次。所有提示 token 的 K 和 V 并行计算并存储在缓存中。

- **解码（Decode）**：一次处理一个新 token。每个步骤非常快（只需要计算一个 token 的 Q、K、V），但必须顺序执行。

这个区别对于理解推理性能很重要。预填充阶段是计算密集型的（大量矩阵乘法），而解码阶段是内存密集型的（读取 KV 缓存占主导地位）。

---

## 5. 关键数字：Qwen3-0.6B

让我们用我们正在实现的模型的具体数字来巩固所有这些概念。理解这些数字及其来源对于建立对模型规模的直观认识至关重要。

### 5.1 模型超参数

| 参数                | 值       | 描述                                    |
|--------------------------|-------------|------------------------------------------------|
| `vocab_size`             | 151,936     | 词汇表中的 token 数量             |
| `hidden_size` (d_model)  | 1,024       | 隐藏表示的维度            |
| `num_hidden_layers`      | 28          | Transformer 块数量                   |
| `num_attention_heads`    | 16          | 查询头数量                          |
| `num_key_value_heads`    | 8           | 键/值头数量（GQA 比例 2:1）      |
| `head_dim`               | 128         | 每个注意力头的维度（配置中显式指定） |
| `intermediate_size`      | 3,072       | FFN 中间维度（3倍 hidden_size）    |
| `max_position_embeddings`| 40,960      | 最大序列长度                        |
| `rms_norm_eps`           | 1e-6        | RMSNorm 数值稳定性的 epsilon        |
| `rope_theta`             | 1,000,000.0 | 旋转位置编码的基础频率   |

### 5.2 参数数量分解

了解参数位于何处有助于你理解模型规模以及哪些组件占主导地位。

**嵌入层**：将 token ID 映射到向量。

```
embed_tokens: vocab_size x hidden_size = 151,936 x 1,024 = 155,580,224  (~155.6M)
```

**每个 Transformer 块**：

| 权重              | 形状            | 参数       | 描述               |
|---------------------|------------------|------------------|---------------------------|
| `q_proj`            | [2048, 1024]     | 2,097,152        | 查询投影（16 heads x 128） |
| `k_proj`            | [1024, 1024]     | 1,048,576        | 键投影（8 heads x 128）    |
| `v_proj`            | [1024, 1024]     | 1,048,576        | 值投影（8 heads x 128）  |
| `o_proj`            | [1024, 2048]     | 2,097,152        | 输出投影         |
| `gate_proj`         | [3072, 1024]     | 3,145,728        | FFN 门控路径           |
| `up_proj`           | [3072, 1024]     | 3,145,728        | FFN 上投影         |
| `down_proj`         | [1024, 3072]     | 3,145,728        | FFN 下投影       |
| `input_layernorm`   | [1024]           | 1,024            | 注意力前 RMSNorm     |
| `post_attn_layernorm`| [1024]          | 1,024            | FFN 前 RMSNorm           |
| **块总计**     |                  | **13,186,560**   | **每块约 13.2M**      |

关于投影形状的说明：`q_proj` 的输出维度是 `num_attention_heads * head_dim = 16 * 128 = 2,048`，而 `k_proj` 和 `v_proj` 的输出维度是 `num_key_value_heads * head_dim = 8 * 128 = 1,024`。q_proj 的 2,048 与 k/v_proj 的 1,024 之间的差异是 GQA 带来的参数节省。如果使用标准多头注意力（16 个 KV 头），它们各自会是 [2048, 1024]，每个块再增加约 2M 参数。

**全部 28 个块**：
```
28 x 13,186,560 = 369,223,680  (~369.2M)
```

**lm_head**（输出投影）：将隐藏状态映射回词汇表 logits。
```
lm_head: vocab_size x hidden_size = 151,936 x 1,024 = 155,580,224  (~155.6M)
```

**最终 RMSNorm**：lm_head 之前的单个归一化层。
```
model.norm: [1024] = 1,024  (可忽略)
```

### 5.3 总参数数量

```
Embedding:         155,580,224  (~155.6M)
28 Blocks:         369,223,680  (~369.2M)
lm_head:           (与嵌入绑定 — 无额外参数)
Final norm:              1,024  (~0.001M)
─────────────────────────────────────────
Total:             524,804,928  (~524.8M ≈ 0.6B)
```

这接近 0.6B，这就是模型被称为 Qwen3-0.6B 的原因。LLM 的命名约定通常四舍五入到最方便的数字。由于 `tie_word_embeddings = true`，`lm_head` 权重矩阵与嵌入矩阵共享参数，所以那约 155.6M 参数只存储一次。

### 5.4 f32 下的内存占用

每个参数存储为 32 位浮点数（f32），使用 4 字节。

```
524,804,928 parameters x 4 bytes = 2,099,219,712 bytes ≈ 2.1 GB
```

仅将模型权重加载到内存中就需要约 2.1 GB RAM。在推理期间，你还需要内存用于：

- KV 缓存：对于长度为 L 的序列，缓存存储 `2 x num_kv_heads x head_dim x L x num_layers x 4 bytes` = `2 x 8 x 128 x L x 28 x 4` = `229,376 x L` 字节。在 L = 4,096 token 时，约为 920 MB。
- 前向传播期间的中间激活。

因此，f32 下推理的实际内存使用量对于短序列约为 3-4 GB，并随上下文长度增长。

### 5.5 参数位于何处？

查看分解，参数分布为：

```
Embedding:  155.6M  (29.7%)
Blocks:     369.2M  (70.3%)
lm_head:    (与嵌入绑定)
Norm:         0.001M (0.0%)
```

FFN 是每个块中最大的组件（约 9.4M，总共 13.2M）。注意力投影每个块约占 6.3M（q_proj、k_proj、v_proj、o_proj 合计），FFN 每个块约占 9.4M。嵌入（由于权重绑定也用作 lm_head）约占所有参数的 30%，这是大词汇表（151,936 个 token）的结果。

对于 Qwen 系列中的更大模型（1.5B、7B、14B 等），hidden_size、intermediate_size 和 num_layers 都会增加，但词汇表大小保持不变，因此嵌入/lm_head 的比例会下降。

---

## 6. 本项目如何实现

### 6.1 代码结构

`qwen3.5-rs` 项目用 Rust 实现了完整的推理流程，每个组件在自己的模块中：

```
src/
├── main.rs              # CLI 入口点（参数解析、交互模式）
├── lib.rs               # 模块声明
├── config.rs            # 将 config.json 解析为 Config 结构体
├── tokenizer.rs         # BPE 分词器（读取 tokenizer.json）
├── tensor.rs            # 具有数学运算的简单 N 维张量
├── safetensors.rs       # 读取 .safetensors 权重文件
├── model.rs             # 完整模型：嵌入 → 块 → lm_head
├── transformer_block.rs # 单个 Transformer 块（注意力 + FFN + 残差）
├── rmsnorm.rs           # RMSNorm 实现
├── rope.rs              # 旋转位置编码
├── attention.rs         # 带 KV 缓存的分组查询注意力
├── ffn.rs               # SwiGLU 前馈网络
├── sampling.rs          # Token 采样策略（贪心、top-k、top-p）
└── inference.rs         # 自回归推理循环和 KV 缓存管理
```

这些模块反映了本文档中的概念分解。每个模块都是自包含的，只处理拼图的一个部分。

### 6.2 设计选择

本项目做出了几个与生产性推理引擎不同的深思熟虑的选择。这些选择优先考虑清晰性和教育性，而非速度：

**单线程，仅 CPU**：没有 CUDA，没有多线程，没有批处理。每个操作都在单个 CPU 核心上发生。这使代码更容易理解——没有竞态条件，没有 GPU 同步，没有批处理维度使张量操作复杂化。性能不是目标；理解才是。

**f32 精度（无量化）**：所有权重和计算使用 32 位浮点数。生产系统通常使用 16 位（f16、bf16）或 8 位（int8、int4）量化来减少内存并加速推理。我们坚持使用 f32，因为它避免了量化方案的复杂性和降低精度的数值微妙之处。

**无外部 ML 框架**：我们不使用 PyTorch、TensorFlow、Candle、Burn 或任何 ML 框架。所有数学运算——矩阵乘法、softmax、RMSNorm、RoPE——都是使用循环和基础算术从头实现的。这是教育方面最重要的设计选择：当你阅读代码时，你看到的是确切发生的计算，没有隐藏的抽象。

**最小依赖**：项目只使用 4 个外部 crate：`clap` 用于 CLI 参数解析，`serde` 和 `serde_json` 用于读取 JSON 配置文件，`byteorder` 用于读取二进制 safetensors 文件。其他一切都是在这个项目中实现的。

### 6.3 接下来是什么

本文档为你提供了大局观。`docs/` 目录中的后续文档深入探讨每个组件：

| 文档                    | 涵盖内容                                  |
|-----------------------------|-------------------------------------------------|
| `02_tokenizer.md`           | BPE 分词：文本如何变为 token ID    |
| `03_embeddings.md`          | Token 嵌入和位置编码       |
| `04_rmsnorm.md`             | RMSNorm：为什么以及我们如何归一化               |
| `05_rope.md`                | 旋转位置编码：编码位置    |
| `06_attention.md`           | 分组查询注意力：核心机制     |
| `07_ffn.md`                 | SwiGLU FFN：逐 token 处理层      |
| `08_safetensors.md`         | 从 safetensors 文件加载模型权重    |
| `09_inference.md`           | 完整推理循环和 KV 缓存管理 |
| `10_sampling.md`            | Token 采样：温度、top-k、top-p       |

每个文档都解释了概念、数学和 Rust 实现。按顺序阅读，或者跳到你最感兴趣的组件。

---

## 快速参考：术语表

| 术语              | 含义                                                        |
|-------------------|----------------------------------------------------------------|
| Token             | 文本的 subword 单元；模型的原子输入          |
| Token ID          | 表示词汇表中 token 的整数              |
| Embedding         | Token 的稠密向量表示                       |
| Hidden state      | 某个层中 token 的内部表示           |
| Logits            | 每个词汇 token 的原始分数，在 softmax 之前           |
| Self-attention    | Token 相互关注的机制                    |
| Causal mask       | 防止 token 关注未来位置             |
| KV cache          | 来自先前 token 的缓存 Key 和 Value 向量              |
| FFN               | 前馈网络，独立处理每个 token       |
| RMSNorm           | 均方根归一化，稳定激活         |
| RoPE              | 旋转位置编码，编码 token 位置              |
| GQA               | 分组查询注意力，在查询头之间共享 KV 头    |
| SwiGLU            | 使用 SiLU 激活的门控 FFN 变体                        |
| Residual connection| 将输入直接添加到输出以帮助梯度流动          |
| Autoregressive    | 一次生成一个 token，每个依赖于前一个 |
| Prefill           | 通过模型处理初始提示                |
| Decode            | 预填充后一次生成一个新 token              |
| lm_head           | 将隐藏状态投影到词汇表 logits 的线性层     |

---

## 延伸阅读

- Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017.
  原始 Transformer 论文。
- Touvron, H., et al. "LLaMA: Open and Efficient Foundation Language Models."
  2023. Qwen3 紧密遵循的架构。
- Shazeer, N. "GLU Variants Improve Transformer." 2020.
  介绍 SwiGLU 和其他门控 FFN 变体。
- Zhang, B., and Sennrich, R. "Root Mean Square Layer Normalization." NeurIPS 2019.
  RMSNorm 论文。
- Su, J., et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding."
  2022. RoPE 论文。
- Ainslie, J., et al. "GQA: Training Generalized Multi-Query Transformer Models
  from Multi-Head Checkpoints." EMNIPS 2023. GQA 论文。
