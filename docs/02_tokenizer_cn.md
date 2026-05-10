# 2. 分词：文本如何变成数字

本文档解释了大语言模型如何将人类可读的文本转换为它实际操作的整型 token ID。这个过程被称为**分词（tokenization）**，执行该操作的组件就是**分词器（tokenizer）**。

如果你已经读过 `01_transformer_basics.md`，你就已经知道分词器位于推理管道的最前端。这里我们深入讲解：有哪些分词方案、字节对编码（BPE）如何工作、为什么现代模型使用字节级 BPE，以及 Qwen3 分词器是如何构建的。

---

## 1. 什么是分词？

神经网络处理的是数字，而不是文本。在 transformer 能够对像 "Hello world" 这样的句子做任何处理之前，该句子必须被转换为一个整数序列。分词就是连接这两个世界的桥梁。

但我们应该如何将文本映射到数字呢？有几种策略，每种都有不同的权衡。

### 1.1 字符级分词

最简单的方法：每个字符分配一个独立的 ID。词表很小（只有 128 个 ASCII 字符，或约 1,000 个常用 Unicode 码点），并且任何可能的文本都可以被表示。

```
"Hello" → [72, 101, 108, 108, 111]
          H=72   e=101  l=108  l=108  o=111
```

**问题**：字符本身携带的含义非常少。字母 "e" 几乎出现在每个英文单词中，因此模型必须学习到 "e" 后面跟着 "l"，再跟着 "l"，再跟着 "o" 意味着 "ello"——一个常见后缀——这完全只能从上下文中学习。这给模型带来了不必要的负担。序列也会变得非常长：一篇 1,000 词的文章可能有 5,000 个字符，而 transformer 必须处理其中的每一个。

### 1.2 词级分词

另一个极端：每个词分配一个独立的 ID。"Hello" 是一个 token，"world" 是另一个。

```
"Hello world" → [15496, 995]
```

**问题 1 — 词表爆炸**：仅英语就有数十万个单词。加上人名、科技术语和多语言文本，很快就会超过一百万。一个 1,000,000 x 1,024 = 4 GB 的嵌入表是不切实际的。

**问题 2 — 词表外（OOV）词**：当模型遇到一个从未见过的词时会发生什么？对于词级分词，答案是：它根本无法表示该词。模型必须回退到一个特殊的 `<UNK>`（未知）token，从而丢失该词的所有信息。这对于形态丰富的语言（如芬兰语、土耳其语）来说是灾难性的，因为这些语言通过添加后缀来高效创造新的词形。

**问题 3 — 同一词的不同形式**："run"、"running"、"runs"、"ran" 都获得独立的 ID。模型必须独立学习它们之间的语义关系，尽管它们共享同一个词根。

### 1.3 子词级分词：最佳平衡点

子词分词将文本拆分为比单个字符大、但比完整单词小的单元。常见词保持完整（"the" 是一个 token），而罕见或复杂的词则被拆分成有意义的片段：

```
"unbelievable" → ["un", "believable"]
"tokenization" → ["token", "ization"]
"hamburger"    → ["ham", "burger"]
```

这优雅地解决了两个极端的问题：

- **词表有界**：一个 30,000-150,000 的子词词表几乎可以覆盖任何语言的所有文本。
- **无 OOV 词**：任何文本都可以被表示，因为任何词都可以先分解为字符，再从子词片段重建。
- **有意义的单元**："un" 和 "ization" 携带了单个字符所不具备的含义，从而减轻了模型的负担。

三种主要的子词分词算法如下：

| 算法 | 使用者 | 核心思想 |
|-----------|---------|----------|
| BPE | GPT-2, GPT-4, LLaMA, Qwen | 迭代合并最频繁的相邻 token 对 |
| WordPiece | BERT, DistilBERT | 合并使似然最大化的 token 对 |
| Unigram | T5, ALBERT | 从大模型开始，剪枝最无用的 token |

BPE 在现代仅解码器模型中远为最常见，因此这是我们关注的重点。

---

## 2. 字节对编码（BPE）—— 算法

字节对编码最初是作为一种文本压缩算法开发的（Sennrich et al., 2016 将其改编用于神经机器翻译）。其思想简单而优雅：从由单个字符组成的词表开始，然后反复将最频繁的相邻 token 对合并为一个新 token，直到词表达到所需大小。

### 2.1 训练：学习合并规则

假设我们的训练语料库由以下带频率的词组成：

```
"low"     × 5
"lower"   × 2
"newest"  × 6
"widest"  × 3
```

我们首先将每个词拆分为字符（我们使用特殊的词尾符号 `</w>` 来标记词边界，以便模型能够重建词在何处结束）：

```
l o w </w>           × 5
l o w e r </w>       × 2
n e w e s t </w>     × 6
w i d e s t </w>     × 3
```

**步骤 1**：统计所有相邻对，找出最频繁的一个。

```
Pair frequencies:
  (e, s) = 6 + 3 = 9    ← 最频繁
  (l, o) = 5 + 2 = 7
  (o, w) = 5 + 2 = 7
  (w, e) = 2 + 6 = 8
  (s, t) = 6 + 3 = 9
  ...
```

(e, s) 和 (s, t) 出现了平局。我们任意选择 (e, s)。我们将其合并为一个新 token "es" 并加入词表。

```
l o w </w>             × 5
l o w es r </w>        × 2
n es w es t </w>       × 6   （两个 "e s" 对都被合并）
w i d es t </w>        × 3
```

**步骤 2**：再次统计对。现在最频繁的对是 (es, t)，出现了 6 + 3 = 9 次。

```
l o w </w>              × 5
l o w es r </w>         × 2
n es w est </w>         × 6
w i d est </w>          × 3
```

**步骤 3**：现在最频繁的对是 (l, o)，出现 7 次。将其合并。

```
lo w </w>               × 5
lo w es r </w>          × 2
n es w est </w>         × 6
w i d est </w>          × 3
```

**步骤 4**：(lo, w) 出现 7 次。合并。

```
low </w>                × 5
low es r </w>           × 2
n es w est </w>         × 6
w i d est </w>          × 3
```

依此类推。经过 K 次合并后，我们的词表包含原始字符加上 K 个新 token，以及一个包含 K 条合并规则的有序列表。

### 2.2 编码：应用合并规则

在推理时，我们将学习到的合并规则应用于新文本。算法如下：

1. 将输入词拆分为单个字符。
2. 找到相邻 token 对中合并**排名最低**（优先级最高，即最早被学习到）的那一对。
3. 合并该对的所有出现。
4. 重复直到无法再进行合并。

关键洞察：我们总是先应用**优先级最高**的合并，而不是最频繁的那一个。在训练期间，频率决定了顺序。在编码期间，我们只需遵循该顺序。

让我们追踪一下对词 "lowest" 的编码过程：

```
Start:  l o w e s t

Merge (e, s) → es (rank 0, highest priority):
  l o w es t

Merge (es, t) → est (rank 1):
  l o w est

Merge (l, o) → lo (rank 2):
  lo w est

Merge (lo, w) → low (rank 3):
  low est

No more applicable merges.
Result: [low, est]
```

词 "lowest" 被分词为两个子词："low" 和 "est"。两者都携带含义——"low" 是一个可识别的词根，"est" 是一个常见的比较级后缀。模型可以独立学习这两个片段的表示并将它们组合起来。

### 2.3 为什么效果这么好

BPE 自然地产生反映训练语料统计结构的子词。在英语中：

- 常见词如 "the"、"and"、"is" 获得自己的 token，因为它们是由早期合并形成的。
- 不太常见的词如 "unbelievable" 被拆分为有意义的子词："un"、"believ"、"able"。
- 非常罕见的词如 "supercalifragilistic" 被拆分为单个字符或非常小的子词。

词表大小是一个可以调节的旋钮：更多的合并 = 更大的词表 = 更短的序列 = 每 token 含义更丰富，代价是更大的嵌入表。

---

## 3. 字节级 BPE

标准 BPE 在 Unicode 字符上操作，但这会带来实际问题。GPT-2 论文（Radford et al., 2019）引入了**字节级 BPE** 来解决这些问题。

### 3.1 Unicode 问题

标准 BPE 根据训练数据中的字符构建其基础词表。对于英语，大约是 70 个字符（a-z、A-Z、0-9、标点符号）。对于中文，是数万个字符。对于表情符号和特殊符号，则更多。

这造成了几个问题：

1. **词表大小不一致**：在英语上训练的模型可能有 70 个字符的基础词表，而在中文上训练的模型则有 30,000+。
2. **跨语言干扰**：如果一个罕见的中文字符出现在英语训练语料中，它会占用一个词表槽位但几乎毫无用处。
3. **词表外字符**：任何在训练期间未见过的字符根本无法被表示。

### 3.2 字节级解决方案

字节级 BPE 通过在**字节**而非字符上操作，完全绕过了这些问题。任何语言的任何文本都可以使用 UTF-8 编码表示为字节序列。只有 256 种可能的字节值，因此基础词表始终恰好是 256——小巧、固定且通用。

但有一个问题：原始字节包含控制字符（字节 0 = null，字节 10 = 换行，字节 32 = 空格），这对 BPE 来说是有问题的。BPE 通过查看 token 对来操作，如果某些"token"是不可见的空白或控制字符，就很难推理合并过程。

解决方案是**字节到 Unicode 的映射**。我们将 256 个字节值中的每一个映射到一个唯一的、可见的 Unicode 字符：

- **直接映射**（188 个字节）：可打印的 ASCII 字节 33-126（`!` 到 `~`），加上 Latin-1 补充字节 161-172 和 174-255，映射到其对应的 Unicode 码点。这些字节已经有了可见的字符表示，因此我们保持原样。

- **偏移映射**（68 个字节）：其余的字节值——控制字符（0-32）、删除和 C1 控制字符（127-160）以及软连字符（173）——被映射到从 U+0100 开始的 Unicode 码点。因此字节 0 映射到 `Ā`（U+0100），字节 1 映射到 `ā`（U+0101），依此类推。

以下是偏移字节中最常遇到的部分映射一览：

| 字节值 | Unicode 字符 | 码点 | 常见含义 |
|-----------|-------------|-----------|---------------|
| 0         | Ā | U+0100 | 空字节 |
| 9         | Ĩ | U+0128 | 制表符 |
| 10        | ĩ | U+0129 | 换行符 |
| 13        | ļ | U+013C | 回车符 |
| 32        | Ġ | U+0120 | **空格** |
| 127       | ł | U+0142 | 删除符 |

需要记住的最重要的映射：**空格（字节 32）映射到 Ġ**。当你在分词器的词表中看到 `Ġ` 时，它代表一个空格字符。像 `Ġthe` 这样的 token 意味着"前面有空格的单词 'the'"——即句中出现而非句首出现。

188 个直接映射的字节包括你通常在键盘上输入的所有字符：字母、数字和常用标点符号。它们映射到自身：

| 字节范围 | 字符 | 示例 |
|-----------|-----------|---------|
| 33-126 | `!` 到 `~` | `A` = 字节 65 = 字符 'A' |
| 161-172 | `¡` 到 `¬` | `©` = 字节 169 = 字符 '©' |
| 174-255 | `®` 到 `ÿ` | `ü` = 字节 252 = 字符 'ü' |

### 3.3 字节级 BPE 的工作原理

完整的流程如下：

1. 将输入文本转换为 UTF-8 字节。
2. 使用上表将每个字节映射为其字节级 Unicode 字符。
3. 对得到的字节级字符序列应用 BPE 合并。
4. 在词表中查找每个得到的 token 以获取其 ID。

对于解码：

1. 查找每个 token ID 以获取 token 字符串。
2. 拼接所有 token 字符串。
3. 将每个字节级字符映射回其原始字节。
4. 将得到的字节解码为 UTF-8。

### 3.4 为什么能处理所有语言

UTF-8 可以表示任何 Unicode 字符。像 `你` 这样的中文字符被编码为三个字节：`0xE4 0xBD 0xA0`。在字节级 BPE 中，这变成三个字节级字符，然后可以由 BPE 合并为更大的子词单元。

对于中文文本，BPE 会迅速学会将对应于常用字符的常见三字节序列合并为单个 token。不太常见的字符则保持为多 token 序列。这意味着：

- 分词器无需特殊配置即可适用于**任何语言**。
- 无论训练数据中有多少种语言，词表大小都保持有界。
- 跨语言迁移是可能的：如果模型学到 `Ġun` 在英语中表示 "un-"，它可以在混合语言语境中应用这一知识。

对于代码，同样的原理适用。Python 关键字如 `def`、`class`、`return` 成为单个 token。不太常见的标识符如 `quantize` 变成 `["quant", "ize"]`。特殊字符如 `{`、`}`、`=`、`==` 都是单个 token。这就是为什么现代 LLM 能够相当好地编写代码——分词器理解编程语言的"词汇"。

---

## 4. Qwen3 分词器

Qwen3 使用基于 tiktoken/cl100k_base 系列的字节级 BPE 分词器，类似于 GPT-4 的分词器。它以 HuggingFace `tokenizer.json` 文件的形式分发。

### 4.1 关键参数

| 参数 | 值 | 描述 |
|-----------|-------|-------------|
| 词表大小 | 151,936 | token 总数（基础 256 + 合并 + 特殊 token） |
| 分词器类型 | 字节级 BPE | 在映射到 Unicode 字符的 UTF-8 字节上操作 |
| 特殊 token | 2+ | EOS 和其他控制 token |
| 预分词器 | GPT-2 风格正则表达式 | 在 BPE 之前将文本拆分为词 |
| 解码器 | 字节级 | 将字节级字符转换回字节 |

### 4.2 特殊 Token

Qwen3 定义了几个在模型输入输出格式中起特定作用的特殊 token：

| Token | ID | 用途 |
|-------|----|---------|
| `<\|endoftext\|>` | 151643 | 序列结束（EOS）标记 |
| `<\|im_start\|>` | 151644 | 聊天消息开始 |
| `<\|im_end\|>` | 151645 | 聊天消息结束 |

EOS token 在推理时是最重要的：它标志着模型已完成生成。在自回归生成过程中，一旦模型输出 EOS token ID (151643)，我们就停止。

`im_start` 和 `im_end` token 用于以 ChatML 格式格式化对话。一个典型的对话如下所示：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
2+2 equals 4.<|im_end|>
```

每条消息都被包裹在 `im_start` 和 `im_end` 标记中，`im_start` 后面跟着角色（system、user、assistant）。这种格式由 OpenAI 引入，并被包括 Qwen 在内的许多模型采用。

### 4.3 tokenizer.json 文件

HuggingFace 以如下结构的 JSON 文件形式分发分词器：

```json
{
  "version": "1.0",
  "added_tokens": [
    {"id": 151643, "content": "<|endoftext|>", "special": true, ...},
    {"id": 151644, "content": "<|im_start|>", "special": true, ...},
    {"id": 151645, "content": "<|im_end|>", "special": true, ...}
  ],
  "model": {
    "type": "BPE",
    "vocab": {
      "!": 0,
      "\"": 1,
      ...
      "Ġthe": 367,
      ...
    },
    "merges": [
      "Ġ t",
      "Ġt he",
      ...
    ]
  },
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokens": [
      {"type": "Split", "pattern": {"Regex": "GPT-2 pattern here"}, ...},
      {"type": "ByteLevel", ...}
    ]
  },
  "decoder": {
    "type": "ByteLevel"
  }
}
```

关键部分包括：

- **`added_tokens`**：带有 ID 的特殊 token。这些在 BPE 处理之前被添加到词表中，并在输入文本中被字面匹配。
- **`model.vocab`**：从 token 字符串到 ID 的完整词表映射。这包括 256 个基础字节级字符、所有 BPE 合并结果以及任何添加的 token。
- **`model.merges`**：BPE 合并规则的有序列表。每个条目是 `"token_a token_b"`，排名即列表中的位置。
- **`pre_tokenizer`**：配置在 BPE 之前将文本拆分为词的规则。GPT-2 正则表达式模式确保 BPE 合并永远不会跨越词边界。
- **`decoder`**：告诉我们要如何将 token 字符串转换回文本。对于字节级 BPE，这是 `"ByteLevel"`，意味着我们反转字节到 Unicode 的映射。

### 4.4 多语言支持

Qwen3 分词器高效处理中文文本。常见中文字符是单个 token，而不太常见的字符则被拆分为 2-3 个 token（UTF-8 字节序列）。相比早期的分词器，这是一个显著的改进——早期分词器中文字本产生的 token 数是同等英文文本的 2-3 倍，使得中文用户的模型更慢且更昂贵。

凭借词表中 151,936 个 token，Qwen3 可以将很大一部分分配给中文、代码和其他专业领域，同时仍然很好地覆盖英文。结果是，对于相同语义内容的英文和中文文本，分词器产生的 token 数量大致相似。

---

## 5. 编码：文本到 Token ID

现在让我们逐步走过完整的编码过程，使用示例文本 `"Hello world"`。

### 5.1 步骤 1：预分词

预分词将输入文本拆分为"词"，以便 BPE 合并永远不会跨越词边界。GPT-2 预分词器使用正则表达式模式来捕获：

- 缩约形式：`'s`、`'t`、`'re`、`'ve`、`'m`、`'ll`、`'d`
- 字母序列（可选前导空格）：`Hello`、` world`
- 数字序列（可选前导空格）：`42`、` 123`
- 标点序列（可选前导空格）：`!`、` .`
- 空白字符：换行符、尾部空格

对于我们的示例：

```
Input:  "Hello world"
Split:  ["Hello", " world"]
```

注意 "world" 前面的空格作为前导空格附加到了词上。这是 GPT-2 的约定：空格是后续词的一部分，而不是独立的 token。

### 5.2 步骤 2：转换为字节级字符

每个词被转换为 UTF-8 字节，每个字节被映射为其字节级 Unicode 字符。

对于 "Hello"：
```
H → byte 72 → 'H'（直接映射）
e → byte 101 → 'e'（直接映射）
l → byte 108 → 'l'（直接映射）
l → byte 108 → 'l'（直接映射）
o → byte 111 → 'o'（直接映射）
Result: ["H", "e", "l", "l", "o"]
```

对于 " world"（带前导空格）：
```
(空格) → byte 32 → 'Ġ'（偏移映射）
w → byte 119 → 'w'（直接映射）
o → byte 111 → 'o'（直接映射）
r → byte 114 → 'r'（直接映射）
l → byte 108 → 'l'（直接映射）
d → byte 100 → 'd'（直接映射）
Result: ["Ġ", "w", "o", "r", "l", "d"]
```

由于 "Hello world" 中的所有字符都是可打印的 ASCII，字节级映射几乎是平凡的——只有空格字符被转换为 `Ġ`。

对于像 "你好" 这样的中文文本，转换更有趣：
```
你 → bytes [0xE4, 0xBD, 0xA0] → ['ä', '½', ' ']
好 → bytes [0xE5, 0xA5, 0xBD] → ['å', '¥', '½']
```

等等——那些看起来不对。这是因为字节 0xE4、0xBD、0xA0 落在"直接映射"范围内（161-255），因此它们映射到其 Latin-1 对应字符：`ä`（0xE4）、`½`（0xBD）等。这些看起来很奇怪，但它们只是中间表示——BPE 算法会迅速将这样的常见序列合并为正确的中文字符 token。

### 5.3 步骤 3：应用 BPE 合并

对于每个词，我们应用 BPE 合并算法：

**"Hello"**——从 `["H", "e", "l", "l", "o"]` 开始：

BPE 算法查看所有相邻对，找到合并列表中排名最低（优先级最高）的那一个。它合并该对并重复。

在实际的 Qwen3 分词器中，"Hello"（大写 H）可能被分词为 `["Hello"]`，如果它是训练数据中足够常见的词的话。如果它不是单个 token，它可能会被拆分为 `["H", "ello"]` 或 `["He", "llo"]`，这取决于学到了哪些合并。

**" world"**——从 `["Ġ", "w", "o", "r", "l", "d"]` 开始：

实际 Qwen3 分词器中 " world" 的合并序列可能如下：
1. `Ġ` + `w` → `Ġw`（空格 + w 是非常常见的序列）
2. `o` + `r` → `or`（英文中常见）
3. `Ġw` + `or` → `Ġwor`
4. `l` + `d` → `ld`（常见结尾）
5. `Ġwor` + `ld` → `Ġworld`

结果：`["Ġworld"]`——一个表示 " world"（空格 + world）的单个 token。

### 5.4 步骤 4：词表查找

每个得到的 token 字符串在词表中查找以获取其整型 ID：

```
"Hello" → ID 15496（示例；实际 ID 取决于分词器）
"Ġworld" → ID 995（示例）
```

最终编码输出为：`[15496, 995]`

这些就是作为 transformer 前向传播第一步被送入嵌入表的整型 ID。

---

## 6. 解码：Token ID 到文本

解码是逆向过程。给定一系列 token ID，我们重建原始文本。

### 6.1 步骤 1：反向词表查找

每个 token ID 在反向词表中查找（ID → token 字符串）：

```
[15496, 995] → ["Hello", "Ġworld"]
```

### 6.2 步骤 2：拼接 Token 字符串

token 字符串被简单拼接：

```
"Hello" + "Ġworld" = "HelloĠworld"
```

### 6.3 步骤 3：将字节级字符转换回字节

拼接字符串中的每个字符使用字节到 Unicode 映射的逆映射被映射回其字节值：

```
'H' → byte 72
'e' → byte 101
'l' → byte 108
'l' → byte 108
'o' → byte 111
'Ġ' → byte 32（空格！）
'w' → byte 119
'o' → byte 111
'r' → byte 114
'l' → byte 108
'd' → byte 100
```

关键步骤：`Ġ` 映射回字节 32，即空格字符。

### 6.4 步骤 4：将字节解码为 UTF-8

字节序列 `[72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]` 被解码为 UTF-8：

```
"Hello world"
```

我们已经恢复了原始文本。

### 6.5 解码中的边界情况

**不完整的 token**：有时一个 token 可能在一个多字节 UTF-8 序列的中间结束。例如，中文字符 `你` 是三个字节（0xE4、0xBD、0xA0）。如果分词器将其拆分到两个 token 中，第一个 token 可能以 0xE4 结束（一个不完整的 UTF-8 序列），第二个以 0xBD、0xA0 开始。解码器必须首先拼接所有 token 字符串，将它们全部转换为字节，然后将完整的字节序列解码为 UTF-8。逐个 token 解码会失败。

**特殊 token**：像 `<|endoftext|>` 这样的特殊 token 不是字节级编码的一部分。它们按原样存储在词表中，并作为字面字符串解码。在我们的实现中，不在字节解码器中的字符（因为它们是特殊 token 字符串的一部分）通过直接将其编码为 UTF-8 表示来处理。

---

## 7. 实现细节

我们在 `src/tokenizer.rs` 中的 Rust 实现遵循上述算法。这里我们重点介绍关键的设计决策。

### 7.1 Tokenizer 结构体

```rust
pub struct Tokenizer {
    vocab: HashMap<String, usize>,           // token 字符串 → ID
    id_to_token: HashMap<usize, String>,      // ID → token 字符串
    merges: Vec<(String, String)>,            // 合并规则（有序）
    merge_ranks: HashMap<(String, String), usize>,  // 合并对 → 排名
    special_tokens: HashMap<String, usize>,   // 特殊 token → ID
    byte_encoder: [char; 256],                // 字节 → unicode 字符
    byte_decoder: HashMap<char, u8>,          // unicode 字符 → 字节
}
```

`merge_ranks` HashMap 是在构造时从 `merges` 派生出来的。它允许 BPE 算法在 O(1) 时间内检查任何相邻对的优先级，而不是线性扫描合并列表。

### 7.2 字节编码器

`build_byte_encoder` 函数构建字节到 Unicode 的映射：

```rust
fn build_byte_encoder() -> ([char; 256], HashMap<char, u8>) {
    let mut encoder = ['\0'; 256];
    let mut decoder = HashMap::new();

    // 直接映射字节（共 188 个）
    let direct_bytes: Vec<u8> = (33..=126)
        .chain(161..=172)
        .chain(174..=255)
        .collect();

    for &b in &direct_bytes {
        let c = char::from_u32(b as u32).unwrap();
        encoder[b as usize] = c;
        decoder.insert(c, b);
    }

    // 偏移字节（剩余 68 个）→ U+0100 及以上
    let mut n = 0u32;
    for b in 0u8..=255 {
        if encoder[b as usize] == '\0' {
            let c = char::from_u32(256 + n).unwrap();
            encoder[b as usize] = c;
            decoder.insert(c, b);
            n += 1;
        }
    }

    (encoder, decoder)
}
```

这产生的正是 GPT-2 字节编码器。结果是确定性的，与 HuggingFace 的 tokenizers 库和 OpenAI 的 tiktoken 所使用的相匹配。

### 7.3 BPE 合并算法

`apply_bpe` 方法实现了标准 BPE 合并算法：

```rust
fn apply_bpe(&self, tokens: &[String]) -> Vec<String> {
    let mut tokens = tokens.to_vec();

    loop {
        // 找到合并排名最低的对。
        let mut best_pair = None;
        let mut best_rank = usize::MAX;

        for i in 0..tokens.len() - 1 {
            let pair = (tokens[i].clone(), tokens[i + 1].clone());
            if let Some(&rank) = self.merge_ranks.get(&pair) {
                if rank < best_rank {
                    best_rank = rank;
                    best_pair = Some(pair);
                }
            }
        }

        let best_pair = match best_pair {
            Some(pair) => pair,
            None => break,  // 无法再进行合并
        };

        // 合并所有出现的最佳对。
        let merged = format!("{}{}", best_pair.0, best_pair.1);
        let mut new_tokens = Vec::new();
        let mut i = 0;
        while i < tokens.len() {
            if i < tokens.len() - 1
                && tokens[i] == best_pair.0
                && tokens[i + 1] == best_pair.1
            {
                new_tokens.push(merged.clone());
                i += 2;
            } else {
                new_tokens.push(tokens[i].clone());
                i += 1;
            }
        }
        tokens = new_tokens;

        if tokens.len() < 2 { break; }
    }

    tokens
}
```

循环的每次迭代需要 O(n) 时间来扫描最佳对，以及 O(n) 来执行合并，其中 n 是当前 token 的数量。在最坏的情况下，我们执行 O(m) 次迭代（其中 m 是适用的合并数），总共 O(m * n)。对于一个典型的词，n 很小（5-15 个字符），m 最多为 n-1，因此对于我们的教学目的来说足够快。

生产级分词器（如 tiktoken）通过使用更复杂的数据结构进一步优化，但算法是相同的。

### 7.4 预分词简化

我们的实现使用了一个简化的预分词器，它根据字符类别（字母、数字、标点、空白）进行拆分，而不是实现完整的 GPT-2 正则表达式。这意味着：

- **对大多数英文文本正确**：带前导空格的词、标点和数字都被正确处理。
- **缩约形式略有不同**："don't" 的拆分方式可能与参考分词器不同（参考分词器会特殊处理 `'s`、`'t`、`'re` 等）。
- **多字符标点略有不同**：像 `...` 或 `==` 这样的序列可能会以不同方式拆分。

出于教学目的，这没问题。BPE 算法本身是正确的，以后可以通过添加 `regex` crate 作为依赖来改进预分词器。

### 7.5 局限性

我们的实现有几个有意的简化：

1. **无正则表达式预分词器**：我们使用基于字符类别的拆分器，而不是完整的 GPT-2 正则表达式模式。这导致某些输入的词化结果略有不同。

2. **无标准化**：某些分词器在编码之前应用 Unicode 标准化（NFC、NFD）。我们跳过了这一步，这意味着具有不同 Unicode 表示但视觉上相同的字符串将被不同地分词。

3. **无截断/填充**：分词器不处理截断长输入或将短输入填充到固定长度。这些通常由调用代码处理，而不是分词器本身。

4. **性能**：我们的实现未针对速度进行优化。对于生产使用，你可能需要缓存预分词结果、使用更高效的合并算法，并可能实现并行编码。

尽管有这些简化，该实现正确地演示了字节级 BPE 分词的所有核心概念。如果你向它提供 Qwen3 的 `tokenizer.json` 文件，它将正确地对大多数英文和中文文本进行分词，并将结果解码回原始文本。

---

## 总结

| 概念 | 要点 |
|---------|-------------|
| 分词 | 将文本转换为模型使用的整数 |
| 子词分词 | 平衡词表大小和序列长度 |
| BPE | 迭代合并最频繁的 token 对 |
| 字节级 BPE | 在字节上操作，而非字符；能处理所有语言 |
| 字节到 Unicode 映射 | 使所有字节对 BPE"可见"；空格 = Ġ |
| 预分词 | 将文本拆分为词，使 BPE 不跨越边界 |
| 编码 | 预分词 → 字节级转换 → BPE 合并 → 词表查找 |
| 解码 | 反向词表查找 → 拼接 → 字节级解码 → UTF-8 |

---

## 延伸阅读

- Sennrich, R., Haddow, B., and Birch, A. "Neural Machine Translation of Rare Words with Subword Units." ACL 2016. 将 BPE 引入 NLP 的论文。

- Radford, A., et al. "Language Models are Unsupervised Multitask Learners." 2019. 引入字节级 BPE 的 GPT-2 论文。

- Kudo, T. "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates." ACL 2018. 介绍了 SentencePiece 使用的 Unigram 分词器。

- HuggingFace Tokenizers 文档：
  https://huggingface.co/docs/tokenizers/ -- 我们读取其格式的库。

- OpenAI Tiktoken：
  https://github.com/openai/tiktoken -- GPT-4 使用的快速 BPE 分词器。
