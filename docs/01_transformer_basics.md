# 1. Transformer Basics: How Large Language Models Work

This document is the starting point for understanding how modern large language
models (LLMs) work. We will build up from the fundamental ideas, one step at a
time, and by the end you will understand every component of the Qwen3-0.6B
model that this project implements.

No prior knowledge of transformers is assumed. You should be comfortable with
basic linear algebra (vectors, matrices, matrix multiplication) and Python or
Rust syntax for reading code snippets.

---

## 1. What Is a Transformer?

### 1.1 The Paper That Changed Everything

In June 2017, a team of researchers at Google published a paper titled
**"Attention Is All You Need"** (Vaswani et al., 2017). It introduced a neural
network architecture called the **Transformer**. At the time, the dominant
approach for sequence modeling -- tasks like machine translation, speech
recognition, and text generation -- was the **Recurrent Neural Network (RNN)**
and its improved variant, the **Long Short-Term Memory network (LSTM)**.

The Transformer was designed to solve two fundamental problems with RNNs:

**Problem 1: Sequential processing prevents parallelism.**
An RNN processes a sequence one step at a time. To compute the hidden state at
position t, it must first compute the hidden state at position t-1, which
requires t-2, and so on. This serial dependency means you cannot parallelize
across time steps during training. For a sentence of 1,000 tokens, you must
wait for 999 sequential computations before processing the last token. On modern
GPUs, which excel at parallel computation, this is a massive bottleneck.

**Problem 2: Long-range dependencies are hard to learn.**
Even with LSTMs, which were specifically designed to mitigate this, information
from early positions in a long sequence tends to get "washed out" by the time
the RNN reaches later positions. If a pronoun at position 500 refers to a noun
at position 5, the RNN must carry that information through 495 intermediate
steps. In practice, the gradient signals that allow the network to learn these
connections either vanish (become zero) or explode (become enormous), making
training unstable.

The Transformer solved both problems simultaneously:
- It processes **all positions in parallel** during training, because there is
  no sequential dependency between time steps.
- It uses **self-attention**, which creates direct connections between every
  pair of positions, no matter how far apart they are. Position 500 can attend
  to position 5 in a single operation with no degradation.

### 1.3 Three Flavors of Transformers

Since the original paper, the Transformer architecture has branched into three
main variants. Understanding the differences is important because they serve
different purposes.

**Encoder-Decoder (original, as in the 2017 paper):**
The model has two stacks. The **encoder** reads the full input sequence and
produces contextualized representations of every token. The **decoder** then
generates the output sequence one token at a time, attending to both its own
previously generated tokens (causally) and the encoder's representations
(cross-attention). This design is natural for sequence-to-sequence tasks like
translation: encode the English sentence, then decode it into French.

```
English: "The cat sat" ──► [Encoder] ──► context vectors
                                              │
                                              ▼ (cross-attention)
French:  "<s>" ──► [Decoder] ──► "Le" ──► [Decoder] ──► "chat" ──► ...
```

**Encoder-Only (e.g., BERT, 2018):**
Only the encoder stack is kept. Every token can attend to every other token in
both directions. This produces rich, bidirectional representations that are
excellent for understanding tasks: classification, named entity recognition,
question answering (given a passage), and so on. BERT is **not** a text
generator -- it is a text understander. You mask out some tokens and train it
to predict them from context.

**Decoder-Only (e.g., GPT series, LLaMA, Qwen):**
Only the decoder stack is kept, but with a crucial modification: there is no
cross-attention (since there is no encoder), and self-attention is **causal** --
each token can only attend to itself and tokens that came before it. This
restriction is necessary because the model generates text one token at a time
and must not "peek" at future tokens. Decoder-only models are trained by
predicting the next token, which turns out to be an incredibly powerful
training objective. With enough data and compute, this simple objective
produces models that can write essays, code, and reason about complex problems.

```
Input:  "The cat sat on"
         │   │   │   │
         ▼   ▼   ▼   ▼
       ┌───────────────────┐
       │  Causal Self-Attn  │  ← each token only sees earlier tokens
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

**This project implements a decoder-only Transformer** -- specifically, the
Qwen3-0.6B model. From here on, when we say "Transformer," we mean the
decoder-only variant.

---

## 2. The Big Picture: How Does a Transformer Process Text?

Let us walk through the entire pipeline, from a string of text to a predicted
next token. We will use a concrete example: the input text is `"Hello world"`.

### 2.1 Step 1: Text to Token IDs (Tokenizer)

Computers do not understand text. They understand numbers. The first step is to
convert a string into a sequence of integers called **token IDs**.

A **tokenizer** splits text into subword units called **tokens** and maps each
token to an integer ID from a fixed vocabulary. Qwen3 uses a Byte Pair
Encoding (BPE) tokenizer with a vocabulary of 151,936 tokens.

```
Text:     "Hello world"
           │         │
           ▼         ▼
Tokens:   [Hello]   [world]
           │         │
           ▼         ▼
Token IDs: [15496]   [995]
```

In reality, the tokenization might split words differently depending on the BPE
merges. "Hello" might be one token, or it could be split into "He" + "llo".
The details are covered in `02_tokenizer.md`. For now, just think of it as a
lookup: each piece of text gets a number.

### 2.2 Step 2: Token IDs to Embedding Vectors

A token ID like 15496 is just a number -- it has no inherent meaning. The number
15496 is not "close to" 15497 in any semantic sense. We need to convert these
discrete IDs into continuous vectors that can capture semantic relationships.

The **embedding table** (also called the embedding matrix) is a giant lookup
table. It has one row per vocabulary entry, and each row is a vector of size
`hidden_size` (1,024 for Qwen3-0.6B). To embed a token ID, you simply look
up the corresponding row.

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

After this step, we have a sequence of vectors, one per token, each of length
1,024. These vectors are learned during training -- tokens that appear in
similar contexts will end up with similar embedding vectors.

### 2.3 Step 3: Transformer Blocks (The Core)

This is where the magic happens. The sequence of embedding vectors passes
through a stack of **transformer blocks** (also called layers). Qwen3-0.6B
has 28 such blocks. Each block transforms its input into a new representation
that is richer and more contextualized.

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

Each block takes in a sequence of vectors and outputs a sequence of vectors with
the same shape. The transformation inside each block involves two sub-layers:
**self-attention** (which lets tokens communicate with each other) and a
**feed-forward network** (which processes each token independently). We will
examine these in detail in Section 3.

The key insight is that **every block adds more context**. After Block 0, the
representation of "world" might encode that it follows "Hello." After Block 27,
the representation might encode the full semantic meaning of "Hello world" in
context -- that it is a greeting, that it is a famous programming tradition,
and so on.

### 2.4 Step 4: Final Hidden State to Logits (lm_head)

After the last transformer block, we have a sequence of hidden state vectors.
To predict the next token, we take the hidden state at the **last position**
(we will see why in Section 4 on autoregressive generation) and project it
back into vocabulary space using a linear layer called **lm_head**.

The lm_head is a matrix of shape `[vocab_size, hidden_size]` = `[151936, 1024]`.
Multiplying a hidden state vector (1,024 dimensions) by this matrix produces a
vector of 151,936 numbers called **logits**. Each logit corresponds to one token
in the vocabulary, and a higher logit means the model thinks that token is more
likely to come next.

```
Last hidden state: [h_1]   (shape: [1, 1024])
                       │
                       ▼
              lm_head (shape: [151936, 1024])
                       │
                       ▼
Logits: [l_0, l_1, ..., l_151935]   (shape: [151936])

  l_0     = score for token ID 0    (probably very low)
  l_15496 = score for "Hello"       (maybe moderate)
  l_3140  = score for "!"           (maybe high)
  ...
```

### 2.5 Step 5: Logits to Next Token (Sampling)

Logits are raw scores, not probabilities. To convert logits to probabilities,
we apply a **softmax** function:

```
P(token_i) = exp(logit_i) / sum(exp(logit_j) for all j)
```

After softmax, every probability is between 0 and 1, and they all sum to 1.

The simplest way to pick the next token is **greedy decoding**: always pick the
token with the highest probability. But this produces boring, repetitive text.
In practice, we use **sampling** methods that introduce controlled randomness:

- **Temperature**: Divides logits by a value T before softmax. T < 1 makes the
  distribution sharper (more confident); T > 1 makes it flatter (more random).
- **Top-k**: Only consider the k most likely tokens, zeroing out the rest.
- **Top-p (nucleus)**: Only consider the smallest set of tokens whose
  cumulative probability exceeds p.

```
Logits: [..., 2.1, 0.5, 5.3, 1.8, ...]
                    │
                    ▼  softmax
Probabilities: [..., 0.03, 0.005, 0.81, 0.02, ...]
                                │
                                ▼  sample
Selected token: ID 3140 ("!")
```

### 2.6 The Full Pipeline at a Glance

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

**Shapes summary** for input "Hello world" (2 tokens):

| Stage               | Shape           | Description                           |
|---------------------|-----------------|---------------------------------------|
| Token IDs           | [2]             | One integer per token                 |
| Embeddings          | [2, 1024]       | One 1024-dim vector per token         |
| After each block    | [2, 1024]       | Same shape (blocks preserve shape)    |
| After lm_head       | [2, 151936]     | One logit per vocab entry per position |
| Next token logits   | [151936]        | Logits at the last position only      |

---

## 3. What Is Inside a Transformer Block?

Now let us zoom in on a single transformer block. This is the core computational
unit, repeated 28 times in Qwen3-0.6B. Understanding one block is enough to
understand the entire model.

### 3.1 The Block Architecture

A decoder-only transformer block (following the LLaMA/Qwen style) has this
structure:

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
├────────────────────────────────────── +  ◄── Residual Connection
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
├────────────────────────────────────── +  ◄── Residual Connection
│
│                         output = x' + ffn_out
│
▼
Output (same shape as input)
```

Let us walk through each component.

### 3.2 Self-Attention: Letting Tokens Communicate

Self-attention is the heart of the Transformer. It allows each token in the
sequence to "look at" every other token and decide how much to attend to (focus
on) it.

**Analogy**: Imagine you are reading the sentence "The cat sat on the mat because
it was tired." When you encounter the word "it," you instinctively look back at
earlier words to figure out what "it" refers to. You attend more to "cat" than
to "mat" because "cat" is more likely to be tired. Self-attention does something
similar: for each token, it computes weighted combinations of other tokens'
representations, where the weights depend on how relevant each token is.

**How it works** (simplified, single head):

For each token, we compute three vectors from its current representation:

- **Query (Q)**: "What am I looking for?" -- represents what information this
  token wants.
- **Key (K)**: "What do I contain?" -- represents what information this token
  offers.
- **Value (V)**: "Here is my content." -- the actual information to be
  aggregated.

The attention score between token i and token j is the dot product of Q_i and
K_j, divided by the square root of the key dimension (to keep magnitudes
stable). These scores are passed through softmax to get attention weights, and
then the output for token i is a weighted sum of all Value vectors:

```
attention_score(i, j) = (Q_i . K_j) / sqrt(d_k)

attention_weight(i, j) = softmax_j(attention_score(i, :))

output_i = sum_j(attention_weight(i, j) * V_j)
```

In the decoder-only (causal) variant, we also apply a **causal mask** that
prevents token i from attending to tokens j > i (future tokens). This is
implemented by setting the attention scores for future positions to negative
infinity before softmax, which turns them into zero after softmax.

```
Causal Mask (for 4 tokens):

       Position:  0   1   2   3
Token 0 attends: [ok,  X,  X,  X]    ← can only see itself
Token 1 attends: [ok, ok,  X,  X]    ← can see 0 and 1
Token 2 attends: [ok, ok, ok,  X]    ← can see 0, 1, 2
Token 3 attends: [ok, ok, ok, ok]    ← can see everything

X = masked (set to -inf before softmax, becomes 0 after)
```

**Multi-Head Attention**: Instead of computing a single set of Q, K, V, the
model computes multiple sets in parallel, each called an **attention head**.
Each head can learn to attend to different types of relationships -- one head
might focus on syntactic relationships (subject-verb agreement), another on
coreference (what "it" refers to), another on positional proximity, and so on.

Qwen3-0.6B uses **16 query heads** and **8 key-value heads**. This is called
**Grouped Query Attention (GQA)**, where multiple query heads share the same
key and value heads. Specifically, every 2 query heads share 1 KV head. GQA
reduces memory usage and computation for the KV cache with minimal quality loss.
The details are covered in `06_attention.md`.

### 3.3 FFN (Feed-Forward Network): Processing Each Token

After self-attention has mixed information across tokens, the FFN processes each
token's representation independently. It is applied identically to every position
-- the same weights are used, but on different input vectors.

Qwen3 uses the **SwiGLU** variant of the FFN, which has three weight matrices
instead of the traditional two:

```
Traditional FFN:     output = W_2 * ReLU(W_1 * x)

SwiGLU FFN:          gate = SiLU(W_gate * x)    ← gating path
                     up   = W_up * x             ← up-projection path
                     output = W_down * (gate * up)
```

Where **SiLU** (Sigmoid Linear Unit, also called Swish) is defined as:
```
SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

The intuition: the gate path decides *which* information to pass through, and
the up path provides *what* information. The element-wise multiplication
between them creates a selective filtering mechanism.

**Dimensions**: The input and output of the FFN are `hidden_size = 1,024`. The
intermediate (up-projected) dimension is `intermediate_size = 3,072`, which is
3x the hidden size. This expansion allows the network to learn a richer
internal representation before projecting back down.

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

### 3.4 Residual Connections: Helping Gradients Flow

A **residual connection** (also called a skip connection) adds the block's input
directly to its output:

```
output = x + sublayer(x)
```

Instead of the sublayer having to produce the entire output from scratch, it
only needs to learn the **residual** -- the difference from the input. This has
two major benefits:

1. **Gradient flow during training**: During backpropagation, gradients can flow
   directly through the addition operation, bypassing the sublayer entirely.
   This prevents the vanishing gradient problem in deep networks. Without
   residual connections, a 28-layer network would be extremely difficult to
   train.

2. **Incremental refinement**: Each transformer block only needs to learn a
   small modification to its input. Early layers can learn basic patterns, and
   later layers can build on top of those patterns to learn more complex
   relationships.

In our transformer block, there are two residual connections:
```
x' = x + self_attention(rmsnorm(x))
output = x' + ffn(rmsnorm(x'))
```

The input is always preserved and added back. The sublayers only produce
corrections.

### 3.5 RMSNorm: Normalizing Activations

**RMSNorm** (Root Mean Square Normalization) is a simplified variant of
LayerNorm. Both serve the same purpose: they normalize the activations to
prevent them from growing too large or too small during the forward pass,
which stabilizes training.

LayerNorm normalizes by both the mean and variance:
```
LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
```

RMSNorm only uses the root mean square, which is faster:
```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
```

Where `eps` is a small constant (1e-6 in Qwen3) to prevent division by zero,
and `gamma` is a learnable parameter of shape `[hidden_size]` that scales each
dimension independently.

The key difference: RMSNorm does not subtract the mean and does not have a
learnable bias (beta). This makes it computationally cheaper while achieving
similar performance. The design choice follows the LLaMA family of models.

The norm is applied **before** each sublayer (this is called "pre-norm"
architecture), which is more stable for training than the original "post-norm"
architecture where normalization is applied after the sublayer.

---

## 4. Autoregressive Generation

### 4.1 One Token at a Time

A decoder-only Transformer generates text **autoregressively**: one token at a
time. Given a sequence of tokens, it predicts the next token. Then, given the
sequence plus that predicted token, it predicts the one after that, and so on.

This is fundamentally different from how the model processes text during
**training** (where all positions are computed in parallel) versus during
**inference** (where we must generate sequentially because each new token
depends on the previous prediction).

```
Step 0: Input:  "Hello world"          → Predict: "!"
Step 1: Input:  "Hello world !"        → Predict: "How"
Step 2: Input:  "Hello world ! How"    → Predict: "are"
Step 3: Input:  "Hello world ! How are" → Predict: "you"
...
```

### 4.2 Why "Causal"?

The word "causal" comes from the concept of causality in time: the present can
be influenced by the past, but not by the future. In a causal language model,
the prediction for position t can only depend on positions 0, 1, ..., t-1.

This is not just a design preference -- it is a **necessity** for autoregressive
generation. When the model is generating token t, tokens t+1, t+2, etc. do not
exist yet. The causal mask ensures that the model never learns to rely on
future information, because it will never have access to it during generation.

During training, we can process the entire sequence in parallel but apply the
causal mask to self-attention, so each position only sees previous positions.
This gives us the benefit of parallel training while maintaining the causal
property required for generation.

### 4.3 The KV Cache: Avoiding Redundant Computation

Consider what happens during autoregressive generation without any optimization:

```
Step 0: Process tokens [0, 1, 2, 3]  → predict token 4
Step 1: Process tokens [0, 1, 2, 3, 4]  → predict token 5
Step 2: Process tokens [0, 1, 2, 3, 4, 5]  → predict token 6
```

At each step, we recompute the Key and Value vectors for **all** previous
tokens, even though they have not changed. The Key and Value for token 0 are
exactly the same at Step 0, Step 1, and Step 2 -- the only thing that changes
is that we have a new Query for the new token.

The **KV cache** solves this inefficiency. After processing each token, we
cache its Key and Value vectors. At the next step, we only need to:

1. Compute Q, K, V for the **new token only**.
2. Append the new K and V to the cache.
3. Compute attention using the new Q against **all cached K and V** (including
  the new ones).

```
Without KV Cache:                     With KV Cache:
Step 0: Compute K,V for [0,1,2,3]    Step 0: Compute K,V for [0,1,2,3], cache them
Step 1: Compute K,V for [0,1,2,3,4]  Step 1: Compute K,V for [4] only, append to cache
Step 2: Compute K,V for [0,1,2,3,    Step 2: Compute K,V for [5] only, append to cache
              4,5]

Cost without cache: O(n^2) total      Cost with cache: O(n) total
```

The KV cache dramatically reduces computation during generation. The trade-off
is memory: we must store the Key and Value vectors for every past token. For a
long conversation, this cache can grow quite large. This is why GQA (Grouped
Query Attention) is important for Qwen3 -- by having only 8 KV heads instead
of 16, the cache is 50% smaller than it would be with standard multi-head
attention.

### 4.4 The Generation Loop (Pseudocode)

Here is the full autoregressive generation loop, simplified for clarity:

```
function generate(prompt_tokens, max_tokens):
    # Step 1: Process the entire prompt at once (prefill)
    hidden = forward_pass(prompt_tokens)      # shape: [seq_len, hidden_size]
    logits = lm_head(hidden[-1])              # take last position's output
    next_token = sample(logits)               # apply temperature, top-k, top-p
    
    # Initialize KV cache with prompt's keys and values
    kv_cache = get_kv_cache_from_forward_pass()
    
    generated = [next_token]
    
    # Step 2: Generate one token at a time (decode)
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

The generation has two phases:

- **Prefill**: Process the entire prompt at once. This is a large batch of work
  but it only happens once. All prompt tokens' K and V are computed in parallel
  and stored in the cache.

- **Decode**: Process one new token at a time. Each step is very fast (only one
  token's Q, K, V to compute), but it must be done sequentially.

This distinction is important for understanding inference performance. The
prefill phase is compute-bound (lots of matrix multiplications), while the
decode phase is memory-bound (reading the KV cache dominates the time).

---

## 5. Key Numbers: Qwen3-0.6B

Let us ground all of this in the concrete numbers of the model we are
implementing. Understanding these numbers and where they come from is essential
for building intuition about model scale.

### 5.1 Model Hyperparameters

| Parameter                | Value       | Description                                    |
|--------------------------|-------------|------------------------------------------------|
| `vocab_size`             | 151,936     | Number of tokens in the vocabulary             |
| `hidden_size` (d_model)  | 1,024       | Dimension of hidden representations            |
| `num_hidden_layers`      | 28          | Number of transformer blocks                   |
| `num_attention_heads`    | 16          | Number of query heads                          |
| `num_key_value_heads`    | 8           | Number of key/value heads (GQA ratio 2:1)      |
| `head_dim`               | 128         | Dimension per attention head (explicit in config) |
| `intermediate_size`      | 3,072       | FFN intermediate dimension (3x hidden_size)    |
| `max_position_embeddings`| 40,960      | Maximum sequence length                        |
| `rms_norm_eps`           | 1e-6        | Epsilon for RMSNorm numerical stability        |
| `rope_theta`             | 1,000,000.0 | Base frequency for Rotary Position Embedding   |

### 5.2 Parameter Count Breakdown

Understanding where the parameters live helps you understand model scale and
which components dominate.

**Embedding layer**: Maps token IDs to vectors.

```
embed_tokens: vocab_size x hidden_size = 151,936 x 1,024 = 155,580,224  (~155.6M)
```

**Per transformer block**:

| Weight              | Shape            | Parameters       | Description               |
|---------------------|------------------|------------------|---------------------------|
| `q_proj`            | [2048, 1024]     | 2,097,152        | Query projection (16 heads x 128) |
| `k_proj`            | [1024, 1024]     | 1,048,576        | Key projection (8 heads x 128)    |
| `v_proj`            | [1024, 1024]     | 1,048,576        | Value projection (8 heads x 128)  |
| `o_proj`            | [1024, 2048]     | 2,097,152        | Output projection         |
| `gate_proj`         | [3072, 1024]     | 3,145,728        | FFN gating path           |
| `up_proj`           | [3072, 1024]     | 3,145,728        | FFN up-projection         |
| `down_proj`         | [1024, 3072]     | 3,145,728        | FFN down-projection       |
| `input_layernorm`   | [1024]           | 1,024            | Pre-attention RMSNorm     |
| `post_attn_layernorm`| [1024]          | 1,024            | Pre-FFN RMSNorm           |
| **Block total**     |                  | **13,186,560**   | **~13.2M per block**      |

Note on the projection shapes: `q_proj` has output dimension
`num_attention_heads * head_dim = 16 * 128 = 2,048`, while `k_proj` and `v_proj`
have output dimension `num_key_value_heads * head_dim = 8 * 128 = 1,024`. The
difference between q_proj's 2,048 and k/v_proj's 1,024 is the parameter savings
from GQA. With standard multi-head attention (16 KV heads), they would each be
[2048, 1024], adding another ~2M parameters per block.

**All 28 blocks**:
```
28 x 13,186,560 = 369,223,680  (~369.2M)
```

**lm_head** (output projection): Maps hidden states back to vocabulary logits.
```
lm_head: vocab_size x hidden_size = 151,936 x 1,024 = 155,580,224  (~155.6M)
```

**Final RMSNorm**: A single normalization layer before lm_head.
```
model.norm: [1024] = 1,024  (negligible)
```

### 5.3 Total Parameter Count

```
Embedding:         155,580,224  (~155.6M)
28 Blocks:         369,223,680  (~369.2M)
lm_head:           (tied with embedding — 0 extra parameters)
Final norm:              1,024  (~0.001M)
─────────────────────────────────────────
Total:             524,804,928  (~524.8M ≈ 0.6B)
```

This is close to 0.6B, which is why the model is called Qwen3-0.6B.
The naming convention for LLMs typically rounds to the nearest convenient
number. Because `tie_word_embeddings = true`, the `lm_head` weight matrix
shares its parameters with the embedding matrix, so those ~155.6M parameters
are only stored once.

### 5.4 Memory Footprint at f32

Each parameter is stored as a 32-bit floating point number (f32), which uses
4 bytes.

```
524,804,928 parameters x 4 bytes = 2,099,219,712 bytes ≈ 2.1 GB
```

Just loading the model weights into memory requires about 2.1 GB of RAM. During
inference, you also need memory for:

- The KV cache: For a sequence of length L, the cache stores
  `2 x num_kv_heads x head_dim x L x num_layers x 4 bytes`
  = `2 x 8 x 128 x L x 28 x 4` = `229,376 x L` bytes.
  At L = 4,096 tokens, that is about 920 MB.
- Intermediate activations during the forward pass.

So realistic memory usage for inference at f32 is around 3-4 GB for short
sequences and grows with context length.

### 5.5 Where Do the Parameters Live?

Looking at the breakdown, the parameter distribution is:

```
Embedding:  155.6M  (29.7%)
Blocks:     369.2M  (70.3%)
lm_head:    (tied with embedding)
Norm:         0.001M (0.0%)
```

The FFN is the largest component within each block (about 9.4M out of 13.2M).
The attention projections account for about 6.3M per block (q_proj, k_proj,
v_proj, o_proj combined), and the FFN accounts for about 9.4M per block. The
embedding (which also serves as lm_head due to weight tying) accounts for
about 30% of all parameters, which is a consequence of the large vocabulary
(151,936 tokens).

For larger models in the Qwen series (1.5B, 7B, 14B, etc.), the hidden_size,
intermediate_size, and num_layers all increase, but the vocabulary size stays
the same, so the embedding/lm_head fraction decreases.

---

## 6. How This Project Implements It

### 6.1 Code Structure

The `qwen3.5-rs` project implements the full inference pipeline in Rust, with
each component in its own module:

```
src/
├── main.rs              # CLI entry point (argument parsing, interactive mode)
├── lib.rs               # Module declarations
├── config.rs            # Parses config.json into a Config struct
├── tokenizer.rs         # BPE tokenizer (reads tokenizer.json)
├── tensor.rs            # Simple N-dimensional tensor with math operations
├── safetensors.rs       # Reads .safetensors weight files
├── model.rs             # Full model: embedding → blocks → lm_head
├── transformer_block.rs # Single transformer block (attention + FFN + residuals)
├── rmsnorm.rs           # RMSNorm implementation
├── rope.rs              # Rotary Position Embedding
├── attention.rs         # Grouped Query Attention with KV cache
├── ffn.rs               # SwiGLU Feed-Forward Network
├── sampling.rs          # Token sampling strategies (greedy, top-k, top-p)
└── inference.rs         # Autoregressive inference loop and KV cache management
```

The modules mirror the conceptual decomposition from this document. Each module
is self-contained and handles exactly one piece of the puzzle.

### 6.2 Design Choices

This project makes several deliberate choices that differ from production
inference engines. These choices prioritize clarity and education over speed:

**Single-threaded, CPU-only**: There is no CUDA, no multi-threading, no
batching. Every operation happens on a single CPU core. This makes the code
easier to follow -- there are no race conditions, no GPU synchronization, and
no batch dimension complicating the tensor operations. Performance is not the
goal; understanding is.

**f32 precision (no quantization)**: All weights and computations use 32-bit
floating point numbers. Production systems often use 16-bit (f16, bf16) or
8-bit (int8, int4) quantization to reduce memory and speed up inference. We
stick with f32 because it avoids the complexity of quantization schemes and
the numerical subtleties of reduced precision.

**No external ML frameworks**: We do not use PyTorch, TensorFlow, Candle, Burn,
or any ML framework. All math -- matrix multiplication, softmax, RMSNorm,
RoPE -- is implemented from scratch using loops and basic arithmetic. This is
the most important design choice for education: when you read the code, you
see exactly what computation is happening, with no hidden abstractions.

**Minimal dependencies**: The project uses only 4 external crates: `clap` for
CLI argument parsing, `serde` and `serde_json` for reading JSON configuration
files, and `byteorder` for reading binary safetensors files. Everything else
is implemented in this project.

### 6.3 What Comes Next

This document gave you the big picture. The subsequent documents in the `docs/`
directory dive deep into each component:

| Document                    | What It Covers                                  |
|-----------------------------|-------------------------------------------------|
| `02_tokenizer.md`           | BPE tokenization: how text becomes token IDs    |
| `03_embeddings.md`          | Token embeddings and positional encodings       |
| `04_rmsnorm.md`             | RMSNorm: why and how we normalize               |
| `05_rope.md`                | Rotary Position Embedding: encoding position    |
| `06_attention.md`           | Grouped Query Attention: the core mechanism     |
| `07_ffn.md`                 | SwiGLU FFN: the per-token processing layer      |
| `08_safetensors.md`         | Loading model weights from safetensors files    |
| `09_inference.md`           | The full inference loop and KV cache management |
| `10_sampling.md`            | Token sampling: temperature, top-k, top-p       |

Each document explains the concept, the math, and the Rust implementation.
Read them in order, or jump to whichever component interests you most.

---

## Quick Reference: Glossary

| Term              | Meaning                                                        |
|-------------------|----------------------------------------------------------------|
| Token             | A subword unit of text; the atomic input to the model          |
| Token ID          | An integer representing a token in the vocabulary              |
| Embedding         | A dense vector representation of a token                       |
| Hidden state      | The internal representation of a token at some layer           |
| Logits            | Raw scores for each vocabulary token, before softmax           |
| Self-attention    | Mechanism where tokens attend to each other                    |
| Causal mask       | Prevents tokens from attending to future positions             |
| KV cache          | Cached Key and Value vectors from previous tokens              |
| FFN               | Feed-Forward Network, processes each token independently       |
| RMSNorm           | Root Mean Square Normalization, stabilizes activations         |
| RoPE              | Rotary Position Embedding, encodes token position              |
| GQA               | Grouped Query Attention, shares KV heads across query heads    |
| SwiGLU            | Gated FFN variant using SiLU activation                        |
| Residual connection| Adding input directly to output to help gradient flow          |
| Autoregressive    | Generating one token at a time, each depending on the previous |
| Prefill           | Processing the initial prompt through the model                |
| Decode            | Generating new tokens one at a time after prefill              |
| lm_head           | Linear layer projecting hidden states to vocabulary logits     |

---

## Further Reading

- Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017.
  The original Transformer paper.
- Touvron, H., et al. "LLaMA: Open and Efficient Foundation Language Models."
  2023. The architecture that Qwen3 follows closely.
- Shazeer, N. "GLU Variants Improve Transformer." 2020.
  Introduces SwiGLU and other gated FFN variants.
- Zhang, B., and Sennrich, R. "Root Mean Square Layer Normalization." NeurIPS 2019.
  The RMSNorm paper.
- Su, J., et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding."
  2022. The RoPE paper.
- Ainslie, J., et al. "GQA: Training Generalized Multi-Query Transformer Models
  from Multi-Head Checkpoints." EMNIPS 2023. The GQA paper.
