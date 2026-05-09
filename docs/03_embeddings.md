# 03. Embeddings: Turning Tokens into Vectors

Before a transformer can do anything useful with text, it needs to convert
discrete token IDs into continuous numerical representations that carry
meaning. This is the job of the **embedding layer** -- the very first
operation inside the model, and conceptually the simplest. But despite its
simplicity, the embedding layer is one of the most important components:
everything the model "knows" about the meaning of words is initialized here.

---

## 1. The Problem: Computers Don't Understand Words

Computers work with numbers. They can add, multiply, and compare numbers at
blazing speed. But they have no native understanding of text. When you type
"hello," the CPU does not see a greeting -- it sees a sequence of bytes.

So the fundamental question of natural language processing is: **how do we
turn words into numbers in a way that preserves meaning?**

### The Naive Approach: One-Hot Encoding

The most straightforward idea is **one-hot encoding**. Assign each word in
your vocabulary a unique index, then represent that word as a vector that is
all zeros except for a single 1 at its index position.

Suppose we have a tiny vocabulary of just five words:

| Word    | Index | One-Hot Vector            |
|---------|-------|---------------------------|
| cat     | 0     | [1, 0, 0, 0, 0]          |
| dog     | 1     | [0, 1, 0, 0, 0]          |
| bird    | 2     | [0, 0, 1, 0, 0]          |
| runs    | 3     | [0, 0, 0, 1, 0]          |
| the     | 4     | [0, 0, 0, 0, 1]          |

Each vector has length 5 (the vocabulary size). Only one position is "hot"
(equal to 1), and the rest are "cold" (equal to 0).

### Why One-Hot Fails

One-hot encoding has three fatal flaws:

**1. No semantic meaning.** The vectors carry no information about what the
words *mean*. "cat" and "dog" are semantically similar (both are animals,
both are pets), but their one-hot vectors are just as different from each
other as "cat" and "the". The dot product of any two distinct one-hot
vectors is always zero -- the model sees no relationship whatsoever.

```
dot(cat, dog) = 1*0 + 0*1 + 0*0 + 0*0 + 0*0 = 0
dot(cat, the) = 1*0 + 0*0 + 0*0 + 0*0 + 0*1 = 0
```

"cat" is equally dissimilar to "dog" and "the." That is clearly wrong.

**2. Huge dimensionality.** In a real model like Qwen3, the vocabulary
contains 151,936 tokens. Each one-hot vector would have 151,936 dimensions,
with 151,935 of them being zero. That is incredibly wasteful. A single
one-hot vector takes 151,936 * 4 = ~608 KB of memory at f32, and the vast
majority of those bytes store the number zero.

**3. All vectors are equidistant.** The Euclidean distance (or cosine
distance) between any two distinct one-hot vectors is always the same:
sqrt(2). There is no notion of "closer" or "farther." The model cannot
express that "cat" is more similar to "dog" than to "the."

These problems make one-hot encoding useless for anything beyond the
simplest toy examples. We need a better way.

---

## 2. What Are Embeddings?

An **embedding** is a **dense vector representation** of a token. Instead of
a sparse vector with a single 1 and thousands of zeros, an embedding is a
compact vector of real numbers where every dimension carries information.

For example, in Qwen3, each token is represented by a vector of **1,024
floating-point numbers**. These 1,024 numbers encode the semantic meaning of
the token in a continuous space.

### Embeddings Capture Semantic Relationships

The key insight of embeddings is that semantically similar tokens end up
with similar vectors. After training, the geometric relationships between
embedding vectors reflect semantic relationships between words.

The classic example:

```
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```

This works because the embedding space learns a "gender" direction. Moving
from "king" to "queen" involves moving in the same direction as moving from
"man" to "woman." The vectors encode that "king" is to "man" as "queen" is
to "woman."

Another way to think about it: if you plot word embeddings in 2D (using
dimensionality reduction), semantically related words cluster together.
Animal names would be near each other, country names would be near each
other, verbs of motion would be near each other, and so on.

### How Are Embeddings Learned?

The embedding vectors are **not hand-crafted**. They are **learned during
training** via backpropagation, just like every other weight in the model.

When the model is initialized, the embedding matrix is filled with small
random numbers. At first, the vectors are meaningless. But as the model
trains on billions of words, it adjusts these vectors so that:

- Tokens that appear in similar contexts get pulled closer together.
- Tokens that never share contexts get pushed apart.
- The vectors encode enough information for the rest of the model (attention
  layers, FFN layers) to predict the next token accurately.

Every time the model makes a wrong prediction, the loss signal propagates
all the way back to the embedding layer, nudging the numbers in the
embedding matrix slightly. After enough training, these numbers settle into
representations that capture meaning, syntax, and even some world knowledge.

---

## 3. How Embeddings Work in a Transformer

The embedding layer is essentially a **lookup table** implemented as a
matrix.

### The Embedding Matrix

The entire embedding layer is just one matrix of shape
`[vocab_size, hidden_size]`:

```
For Qwen3-0.6B:
  vocab_size  = 151,936
  hidden_size = 1,024

Embedding matrix shape: [151936, 1024]
```

That is it. No complicated computation, no neural network layers, no
activation functions. Just a big table of numbers.

### The Lookup Operation

Given a sequence of token IDs, the embedding layer simply looks up the
corresponding row for each ID:

```
Input:  token_ids = [151644, 1036, 3837, 1079, ...]    (a sequence of integers)
Output: embeddings = [row_151644, row_1036, row_3837, row_1079, ...]  (a sequence of vectors)
```

Each integer token ID is used as an index into the matrix. The result is a
2D tensor of shape `[seq_len, hidden_size]`, where `seq_len` is the number
of tokens in the input.

```
Token ID 151644 ──► Row 151644 of the embedding matrix ──► [0.023, -0.145, 0.678, ..., 0.234]
Token ID 1036   ──► Row 1036   of the embedding matrix ──► [-0.089, 0.567, -0.012, ..., -0.345]
Token ID 3837   ──► Row 3837   of the embedding matrix ──► [0.456, -0.234, 0.089, ..., 0.567]
  ...                          ...
```

### Why a Matrix, Not a Dictionary?

You might wonder: why store this as a matrix instead of a hash map from
token ID to vector? The answer is **efficiency**. A contiguous block of
memory (the matrix) allows the lookup to be a simple memory offset
calculation:

```
address_of_row_i = base_address + i * hidden_size * sizeof(f32)
```

This is an O(1) operation with excellent cache locality. A hash map would
involve hashing the key, handling collisions, and chasing pointers -- all
much slower.

Conceptually, the embedding lookup is equivalent to multiplying a one-hot
vector by the embedding matrix:

```
one_hot(token_id) @ embedding_matrix = embedding_matrix[token_id]
```

But actually constructing the one-hot vector would be absurdly wasteful
(151,936 dimensions for a single token!), so we just do a row lookup
instead. The mathematical equivalence is nice to know, but the
implementation is a simple index.

### The Simplest Layer in the Entire Model

The embedding layer has no learned computation -- no weights that transform
the input, no non-linearities, no parameters that mix information across
positions. It is purely a storage mechanism. The "learning" happens only in
the sense that the values stored in the matrix are updated during training.

This makes it the simplest layer in the entire transformer. Everything that
comes after (attention, feed-forward networks, layer normalization) is far
more complex.

---

## 4. Token vs. Word vs. Subword

A natural question: if we are embedding "words," does that mean we have one
embedding per word? Not quite. Modern language models operate on **subword
tokens**, not whole words.

### The Problem with Whole-Word Embeddings

If we assigned one embedding per word, we would face several problems:

- **Huge vocabulary.** English alone has hundreds of thousands of words, and
  a multilingual model would need millions. The embedding matrix would be
  impossibly large.
- **Cannot handle unseen words.** If the model encounters a word not in its
  vocabulary during inference, it has no embedding for it and simply
  fails. This is the **out-of-vocabulary (OOV)** problem.
- **Cannot share structure.** The words "run," "running," "ran," and "runs"
  are related, but with whole-word embeddings, each gets a completely
  independent vector. The model must relearn their relationship from
  scratch.

### Subword Tokenization to the Rescue

Modern models solve this by splitting words into smaller units called
**subword tokens**, using an algorithm called **Byte Pair Encoding (BPE)**.
The key idea:

- Common words stay as single tokens ("the," "cat," "is").
- Rare or complex words are split into reusable subword units
  ("unfriendly" might become ["un", "friendly"] or ["un", "friend", "ly"]).
- The vocabulary is fixed at a manageable size (151,936 for Qwen3).

This means:

- Every possible text can be tokenized -- there are no out-of-vocabulary
  problems, because even unknown characters can fall back to byte-level
  tokens.
- Morphologically related words share subword tokens, so the model can
  generalize: if it learns something about "friend," that knowledge helps
  with "friendly" and "friendship."
- The vocabulary size stays bounded, keeping the embedding matrix finite.

### Example

Consider the word "unhappiness." With BPE, it might be tokenized as:

```
"unhappiness" → ["un", "happiness"]
```

or perhaps:

```
"unhappiness" → ["un", "happy", "ness"]
```

Each subword token has its own embedding. The model sees the sequence of
embeddings for ["un", "happy", "ness"], and the attention layers can learn
to combine the meanings of the prefix "un" (negation), the root "happy,"
and the suffix "ness" (state of being) to understand the full word.

For a detailed explanation of how BPE works -- including how the vocabulary
is built from training data and how the merge algorithm decides which
subwords to combine -- see the companion document
[02_tokenizer.md](02_tokenizer.md).

### Qwen3's Vocabulary

Qwen3 uses a vocabulary of **151,936 tokens**. This vocabulary was built
by running BPE on a massive multilingual corpus. It includes:

- Common English words and subwords
- Chinese characters and common character combinations
- Tokens for other languages present in the training data
- Special tokens (like `<|endoftext|>`, chat markers, etc.)
- Byte-level fallback tokens for handling any arbitrary byte sequence

This vocabulary size is a design trade-off: a larger vocabulary means fewer
tokens per word (shorter sequences, faster attention), but a bigger
embedding matrix (more memory, more parameters to train).

---

## 5. The Embedding Matrix in Detail

Let us look at what the embedding matrix actually contains.

### Visualizing the Matrix

```
              dim_0    dim_1    dim_2   ...  dim_1023
           ┌──────────────────────────────────────────┐
token_0    │  0.023  -0.145   0.678   ...   0.234     │
token_1    │ -0.089   0.567  -0.012   ...  -0.345     │
token_2    │  0.456  -0.234   0.089   ...   0.567     │
  ...      │   ...      ...     ...    ...    ...      │
token_151935│  0.123   0.456  -0.789   ...   0.012     │
           └──────────────────────────────────────────┘

Shape: [151936, 1024]
Total entries: 151,936 * 1,024 = 155,581,184 floating-point numbers
```

Each row is a learned representation of one token in the vocabulary. The
columns (dimensions) do not correspond to interpretable human concepts --
they are learned features that the model has found useful for prediction.

### What Do the Dimensions Mean?

Individual dimensions of the embedding are not directly interpretable. You
cannot point to dimension 47 and say "this encodes animacy" or dimension
312 and say "this encodes plurality." The dimensions are distributed
representations: meaning is spread across all dimensions, and each
dimension contributes to many semantic features.

However, if you project the embeddings down to 2D or 3D using techniques
like PCA or t-SNE, you can see that semantically related tokens cluster
together. This is emergent structure -- nobody told the model to organize
the space this way. It learned it from the data.

### Looking Up a Token

When the model receives a token ID (say, 1036), it extracts row 1036 from
the embedding matrix. This is equivalent to:

```python
# Pseudocode
embedding_vector = embedding_matrix[1036]  # Shape: [1024]
```

The result is a 1D tensor (vector) of 1,024 floats, which becomes the
initial representation of that token in the model.

For a sequence of token IDs, we extract multiple rows:

```python
# Pseudocode
token_ids = [151644, 1036, 3837, 1079]
embeddings = embedding_matrix[token_ids]  # Shape: [4, 1024]
```

The result is a 2D tensor where the first dimension is the sequence length
and the second dimension is the embedding size.

---

## 6. Position Information

There is a critical problem with embeddings as described so far: **they
carry no position information**.

### The Problem

Consider two sentences:

1. "The dog bites the man"
2. "The man bites the dog"

Both sentences contain the exact same tokens: ["The", "dog", "bites",
"the", "man"]. With plain embeddings, both sentences would produce the
exact same set of vectors (ignoring the trivial difference between "The"
and "the"). The model would have no way to know which word is the subject
and which is the object -- and therefore no way to understand the very
different meanings of these two sentences.

This is a fundamental issue: **the meaning of a word depends on its
position in the sentence**.

### Why Embeddings Alone Cannot Solve This

The embedding for "dog" is always the same vector, regardless of whether
"dog" appears at position 1 or position 4. The embedding layer has no
concept of position -- it only knows about token identity.

You might think: why not just add the position index to the embedding?
The problem is that a simple integer index (0, 1, 2, 3, ...) does not
carry useful relational information. Position 3 and position 4 are
consecutive, but the model cannot easily learn this from a raw integer.
Worse, the model needs a way to generalize to sequence lengths it has not
seen during training.

### The Solution: RoPE (Preview)

Qwen3 uses **Rotary Position Embedding (RoPE)** to encode position
information. Unlike some earlier approaches that add a position vector to
the embedding, RoPE operates on the query and key vectors inside the
attention mechanism.

The key idea: instead of adding position information at the embedding
layer, RoPE modifies the attention computation itself so that the relative
position between two tokens influences how much they attend to each other.

We will cover RoPE in detail in [05_rope.md](05_rope.md). For now, the
important takeaway is:

> The embedding layer produces **position-agnostic** representations.
> Position information is injected later, inside the attention layers,
> via RoPE.

This design choice is common in modern transformers (LLaMA, Mistral,
Qwen, etc.) and is one of the differences from the original Transformer
paper, which used absolute positional embeddings added directly to the
token embeddings.

---

## 7. Memory Footprint

The embedding matrix is big. Let us calculate exactly how big.

### Calculation

```
Number of entries = vocab_size * hidden_size
                  = 151,936 * 1,024
                  = 155,581,184

Size at f32 (4 bytes per float) = 155,581,184 * 4
                                 = 622,324,736 bytes
                                 ≈ 624 MB (about 594 MiB)
```

That is over **600 megabytes** just for the embedding matrix. For
reference, the entire Qwen3-0.6B model is about 1.2 GB in f32, so
the embedding layer accounts for roughly **half** of the total model
size.

### Why So Large?

The embedding matrix scales with vocabulary size, which is 151,936 in
Qwen3. This is one of the largest vocabularies among open-source models
(LLaMA-2 uses 32,000, GPT-2 uses 50,257). A large vocabulary is a design
choice: it means fewer tokens per word, which means shorter sequences and
faster attention. But the trade-off is a bigger embedding matrix.

Here is a comparison of embedding matrix sizes across different models:

| Model          | Vocab Size | Hidden Size | Embedding Size (f32) |
|----------------|------------|-------------|----------------------|
| GPT-2          | 50,257     | 768         | ~148 MB              |
| LLaMA-2-7B     | 32,000     | 4,096       | ~500 MB              |
| Qwen3-0.6B   | 151,936    | 1,024       | ~624 MB              |
| Qwen3-7B     | 151,936    | 4,096       | ~2.4 GB              |

Notice how the Qwen3-7B model, with the same vocabulary but 4x the
hidden size, has an embedding matrix that is 4x larger.

### Weight Tying (and Why Qwen3 Uses It)

Some models reduce the memory footprint by **tying** the embedding matrix
and the output projection (`lm_head`). In weight tying, the same matrix
is used for both:

- **Embedding layer**: maps token IDs to vectors (lookup).
- **lm_head**: maps final hidden states to vocabulary logits (matrix
  multiply).

If the weights are tied, you only need to store one copy of the matrix
instead of two, cutting the embedding-related memory roughly in half.

Qwen3 **does** use weight tying (`tie_word_embeddings = true`).
The embedding matrix and the `lm_head` weight matrix share the same
parameters. This means the model only needs to store one copy of this
large matrix:

```
Embedding matrix:  151,936 * 1,024 * 4 = ~624 MB
lm_head:           (tied with embedding — 0 extra memory)
                                           ─────────
Total for both:                             ~624 MB
```

Without weight tying, the total would be ~1,248 MB. Weight tying saves
about 624 MB of memory for the Qwen3-0.6B model, which is significant
for a model this size. The trade-off is slightly less flexibility — the
input embedding and output projection must use the same weight values —
but empirical results show this has minimal impact on model quality.

---

## 8. How We Implement It

In our Rust implementation, the embedding layer is straightforward. The
embedding matrix is stored as a 2D tensor of shape `[151936, 1024]`, and
looking up token IDs means selecting rows from this tensor.

### Storage

The embedding weights are loaded from the safetensors file under the key
`model.embed_tokens.weight`. The shape is `[151936, 1024]`, stored as f32
floats.

### Lookup Pseudocode

```rust
fn embed(token_ids: &[usize], embedding_table: &Tensor) -> Tensor {
    // For each token ID, extract the corresponding row
    let rows: Vec<Tensor> = token_ids.iter()
        .map(|&id| embedding_table.row(id))
        .collect();
    // Stack rows into a 2D tensor [seq_len, hidden_size]
    Tensor::stack(rows, 0)
}
```

This function:

1. Iterates over the input token IDs.
2. For each ID, extracts the corresponding row from the embedding table
   (a 1D tensor of length 1,024).
3. Stacks all the rows into a 2D tensor of shape `[seq_len, 1024]`.

### A More Concrete Walkthrough

Let us trace through a tiny example. Suppose the input text is "Hello
world" and the tokenizer produces token IDs `[1036, 1079]` (these are
hypothetical IDs for illustration).

```
Step 1: Tokenizer output
  "Hello world" → token_ids = [1036, 1079]

Step 2: Embedding lookup
  embedding_table.row(1036) = [-0.042, 0.156, 0.891, ..., 0.345]  (1024 floats)
  embedding_table.row(1079) = [0.234, -0.567, 0.012, ..., -0.789] (1024 floats)

Step 3: Stack into a tensor
  embeddings = [[-0.042, 0.156, 0.891, ..., 0.345],   ← "Hello"
                [0.234, -0.567, 0.012, ..., -0.789]]   ← "world"
  Shape: [2, 1024]
```

This 2D tensor is then passed into the first transformer block, where the
attention and FFN layers will process it.

### Implementation Notes

A few things worth noting about our implementation:

- **No gradient computation.** Since we are doing inference only (not
  training), we never need to compute gradients through the embedding
  layer. The embedding values are fixed after loading from the model file.
- **Row selection is efficient.** Because our tensor stores data in a
  contiguous row-major layout, selecting a row is just a pointer offset.
  There is no copy needed for a row view (though stacking does copy).
- **Input validation.** We must ensure that every token ID is in the range
  `[0, vocab_size)`. An out-of-range ID would access memory outside the
  embedding matrix, causing undefined behavior or a crash.

---

## Summary

Let us recap the key points about embeddings:

| Concept | Summary |
|---------|---------|
| **What** | Dense vector representations of tokens, stored in a matrix |
| **Shape** | `[vocab_size, hidden_size]` = `[151936, 1024]` for Qwen3-0.6B |
| **Operation** | Lookup: token ID (integer) → corresponding row (vector) |
| **Learning** | Values are learned during training via backpropagation |
| **Semantics** | Similar tokens get similar vectors; relationships are encoded geometrically |
| **Position** | Embeddings are position-agnostic; RoPE adds position later in attention |
| **Memory** | ~624 MB at f32; significant portion of the total model |
| **Weight tying** | Used in Qwen3; embedding and lm_head share the same weight matrix |

The embedding layer is the bridge between the discrete world of text and
the continuous world of neural networks. It is the first step in the
forward pass, and its quality directly affects everything that follows.
Without good embeddings, no amount of attention or feed-forward computation
can produce meaningful language understanding.

In the next document, we will look at how the model normalizes these
embedding vectors before passing them into attention: [04_rmsnorm.md](04_rmsnorm.md).
