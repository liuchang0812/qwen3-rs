# 09 — Autoregressive Inference: Generating Text One Token at a Time

A language model is useless if it cannot produce text. Training gives the
model the ability to *predict* the next token, but generation is what turns
those predictions into coherent output. This chapter explains how
autoregressive inference works: the process by which the model generates a
sequence of tokens, one at a time, using its own previous outputs as context
for each new prediction.

---

## 1. What Is Autoregressive Generation?

### The Meaning of "Auto-Regressive"

"Auto-regressive" literally means "self-regressing" — each output depends on
previous outputs from the same process. The prefix "auto" signals that the
model feeds its own outputs back into itself, and "regressive" refers to the
statistical notion of regression: predicting a value from prior observations.

In the context of language models, autoregressive generation means:

- The model generates **one token at a time**.
- Each new token is predicted using **all previously generated tokens** as
  context.
- The process continues until a stopping condition is met.

### The Writing Metaphor

Think of how you write a sentence. You do not decide on every word
simultaneously. You write the first word, then choose the second word based
on the first, then the third based on the first two, and so on. Each word
depends on everything that came before it. The language model does exactly
the same thing — it writes one word at a time, always considering what came
before.

### Why Not Generate All Tokens at Once?

The model could, in principle, produce a probability distribution over every
position in a long sequence in a single forward pass. But that would require
the model to "plan ahead" — to decide on token 10 without knowing what tokens
1 through 9 will be. Autoregressive generation avoids this by always having
full context for each prediction. Each token is chosen with complete knowledge
of everything that precedes it, which is why the resulting text is coherent.

The trade-off is that generation is **inherently sequential**. You cannot
parallelize the generation of multiple tokens because each one depends on the
previous one. This is a fundamental limitation of autoregressive models and a
key reason why inference speed matters.

---

## 2. The Generation Loop

Autoregressive generation is a loop. Each iteration produces one token and
adds it to the growing sequence. Here is the step-by-step walkthrough:

### Step 1: Start with Prompt Text

The user provides a prompt — the initial text that seeds the generation. For
example:

```
"The capital of France is"
```

### Step 2: Tokenize

The tokenizer converts the prompt text into a sequence of integer token IDs.
For example:

```
"The capital of France is" → [791, 3187, 315, 5327, 374]
```

These token IDs are the model's native language. Every piece of text must be
converted to IDs before the model can process it.

### Step 3: Prefill

The model processes **all prompt tokens at once** in a single forward pass.
This is called the "prefill" phase. The model computes the hidden states,
attention, and logits for every prompt position in parallel. The output is a
logits tensor of shape `[prompt_len, vocab_size]`.

We only need the logits for the **last** position, because that position
represents the model's prediction for the next token after the entire prompt.

During prefill, the KV cache is populated for every prompt position. This is
critical — it means we will not need to recompute K and V for any of these
tokens during subsequent decode steps.

### Step 4: Sample the Next Token

Take the logits at the last position (a vector of length `vocab_size`) and
apply a sampling strategy to select a single token ID. This could be greedy
decoding (pick the highest-probability token), temperature sampling, top-k
filtering, top-p (nucleus) filtering, or a combination. The sampling module
(doc Chapter 10) covers these strategies in detail.

### Step 5: Decode

Append the sampled token to the output sequence. Then pass **just the new
token** through the model, using the KV cache from previous steps. The model
only needs to compute K and V for the new token; all previous K and V values
are already in the cache.

This produces logits for the next position. Sample from those logits to get
the next token.

### Step 6: Repeat

Repeat steps 4-5 until a stopping condition is met (see Section 5).

### Pseudocode for the Generation Loop

```
function generate(prompt, max_tokens, sampling_config):
    // Step 1-2: Tokenize
    token_ids = tokenize(prompt)

    // Step 3: Prefill — process all prompt tokens at once
    logits = model.forward(token_ids, start_pos=0)
    next_token = sample(logits[-1], sampling_config)
    output_tokens = [next_token]

    // Track position for KV cache
    pos = len(token_ids)

    // Steps 4-6: Decode loop
    for i in 1..max_tokens:
        logits = model.forward([next_token], start_pos=pos)
        next_token = sample(logits[0], sampling_config)
        output_tokens.append(next_token)
        pos += 1

        if next_token == EOS_TOKEN:
            break

    // Convert token IDs back to text
    return decode(output_tokens)
```

Notice the two distinct phases:

1. **Prefill**: `model.forward(token_ids, start_pos=0)` processes the entire
   prompt at once. The KV cache is empty before this call and populated after.

2. **Decode loop**: `model.forward([next_token], start_pos=pos)` processes
   exactly one token. The KV cache already contains entries for all previous
   tokens, so the model only computes K and V for the new token and appends
   them to the cache.

The `start_pos` parameter tells the model where the current tokens begin in
the overall sequence. This is needed for two things: (a) looking up the
correct RoPE positional encodings, and (b) knowing which positions in the KV
cache correspond to "past" vs. "current" tokens.

---

## 3. Prefill vs Decode

Understanding the difference between prefill and decode is essential for
understanding how inference performance works.

### Prefill: Processing the Prompt

During prefill, the model processes all prompt tokens simultaneously. If the
prompt is 50 tokens long, the model does a single forward pass with input
shape `[50, hidden_size]`.

Key characteristics of prefill:

- **Many tokens at once**: The entire prompt is processed in parallel.
- **KV cache is populated**: After prefill, the cache contains K and V
  entries for every prompt position. For Qwen3-0.6B with 8 KV heads and
  head_dim 128, each layer's cache grows to `[prompt_len, 1024]`.
- **Matrix operations can be parallelized**: The QKV projections, attention
  score computation, and FFN are all large matrix multiplications that
  benefit from parallel hardware (GPUs, SIMD on CPU).
- **O(n^2) attention**: Attention scores have shape
  `[num_heads, prompt_len, prompt_len]`, so attention computation scales
  quadratically with prompt length. For very long prompts, this can become
  expensive.

Shape trace during prefill for a 10-token prompt:

```
Input token IDs:    [10]
Embedding lookup:   [10, 1024]
After layer 0:      [10, 1024]
...
After layer 27:     [10, 1024]
After final norm:   [10, 1024]
After lm_head:      [10, 151936]      ← logits for every position

KV cache per layer: [10, 512] (K) + [10, 512] (V)
```

We only use the logits at position 9 (the last position) to sample the first
generated token.

### Decode: Generating One Token at a Time

During decode, the model processes exactly one new token per step. The input
shape is `[1, hidden_size]`.

Key characteristics of decode:

- **One token at a time**: Each step adds exactly one token to the sequence.
- **Each step adds one K/V to the cache**: The cache grows by one row per
  decode step. After generating 20 tokens, each layer's cache has shape
  `[prompt_len + 20, 512]`.
- **Inherently sequential**: You cannot parallelize across decode steps
    because step N+1 depends on the token chosen at step N.
- **O(n) attention**: Attention scores have shape `[num_heads, 1, n]` where
  `n` is the total sequence length so far. This scales linearly, not
  quadratically, with sequence length.

Shape trace during a single decode step (generating token after 10-token
prefill):

```
Input token ID:     [1]
Embedding lookup:   [1, 1024]
After layer 0:      [1, 1024]
...
After layer 27:     [1, 1024]
After final norm:   [1, 1024]
After lm_head:      [1, 151936]       ← logits for the next token

KV cache per layer: [11, 512] (K) + [11, 512] (V)  ← grew by 1 row
```

### Why the Distinction Matters

Prefill is compute-heavy (large matrix operations), while decode is
memory-bandwidth-bound (small operations, but the KV cache must be read at
every step). On modern hardware, prefill is often the fast phase because it
uses the hardware efficiently. Decode is slower per-token because each step
requires reading the entire KV cache from memory but does very little
computation.

This is why "time to first token" (TTFT, determined by prefill speed) and
"tokens per second" (determined by decode speed) are reported separately in
benchmark results.

---

## 4. The KV Cache

The KV cache is the single most important optimization for autoregressive
inference. Without it, generation would be impractically slow.

### Without the Cache: O(n^2) Total Computation

Consider generating a sequence of length N. At each step t, the model needs
to compute attention between the current token and all previous tokens. If we
do not cache K and V from previous steps, we must recompute them from
scratch.

At step 1 (after prefill), we process the prompt of length P. This costs
O(P^2) attention.

At step 2, we need to compute K and V for the new token AND recompute K and V
for all P previous tokens to compute attention. This costs O(P) for the
recomputation.

At step t, we must recompute K and V for all P + t - 1 previous tokens. The
total cost over N decode steps is:

```
Total = sum_{t=1}^{N} O(P + t) = O(N*P + N^2)
```

For a 50-token prompt and 500 generated tokens, that is approximately
27,500 recomputed tokens — most of which were already computed in previous
steps. The vast majority of this work is redundant.

### With the Cache: O(n) Per Step

The KV cache stores the K and V vectors for every previously processed token.
At each decode step:

1. Compute K and V for the **new** token only (O(1) projection work).
2. Append the new K and V to the cache (O(1) memory operation).
3. Compute attention between the new token's Q and the full cached K, V
   (O(n) where n is the total sequence length so far).

The per-step cost is O(n), which is unavoidable — the new token must attend
to all previous tokens. But we avoid the O(n) **recomputation** of K and V at
every step. The total cost over N decode steps becomes:

```
Total = sum_{t=1}^{N} O(P + t) for attention only
      = O(N*P + N^2/2) for attention
      + O(N) for K/V projection (computed once per step)
```

This is the same asymptotic attention cost (we cannot avoid attending to all
previous tokens), but we eliminate all redundant K and V projection
computations. In practice, the savings are enormous because K and V
projections involve large matrix multiplications.

### Memory Cost Calculation for Qwen3-0.6B

The KV cache stores two tensors (K and V) per layer. Each tensor has shape
`[seq_len, num_kv_heads * head_dim]`. For Qwen3-0.6B:

```
num_kv_heads  = 8
head_dim      = 128
kv_dim        = 8 * 128 = 1024
num_layers    = 28
sizeof(f32)   = 4 bytes

KV cache per layer = 2 * kv_dim * seq_len * 4
                   = 2 * 1024 * seq_len * 4
                   = 8,192 * seq_len bytes

Total KV cache     = 8,192 * seq_len * 28
                   = 229,376 * seq_len bytes
```

At various sequence lengths:

| Sequence Length | Per Layer (KB) | Total (MB) |
|----------------|----------------|------------|
| 128            | 512            | 14         |
| 512            | 2,048          | 56         |
| 1,024          | 4,096          | 112        |
| 2,048          | 8,192          | 224        |
| 4,096          | 16,384         | 448        |
| 8,192          | 32,768         | 896        |
| 16,384         | 65,536         | 1,792      |

The cache grows linearly with sequence length. For Qwen3-0.6B with a
4096-token context, the KV cache consumes nearly 450 MB. For larger models
with more layers and heads, the cache can easily exceed several gigabytes.

### Cache Growth Over Time

The cache starts empty. During prefill, it is populated with the entire
prompt. During decode, it grows by one row per step per layer.

```
Time    Tokens Processed    Cache Rows (per layer)    Total Cache (MB)
----    ----------------    ----------------------    ----------------
t=0     0 (empty)           0                          0
t=1     50 (prefill)        50                         5.5
t=2     51 (decode)         51                         5.6
t=3     52 (decode)         52                         5.7
...
t=450   499 (decode)        499                        54.7
t=451   500 (decode)        500                        54.8
```

After 500 decode steps following a 50-token prompt, the cache holds 550 rows
per layer. Each decode step makes the cache slightly larger, which means each
subsequent attention computation is slightly slower (because Q must attend to
more cached K/V entries).

### Our Implementation

In our Rust code, the `KVCache` struct is defined in `src/attention.rs`:

```rust
pub struct KVCache {
    pub key_cache:   Option<Tensor>,  // [seq_len_so_far, num_kv_heads * head_dim]
    pub value_cache: Option<Tensor>,  // [seq_len_so_far, num_kv_heads * head_dim]
}
```

The `Option` type reflects the fact that the cache starts empty. On the first
forward pass (prefill), the computed K and V are stored directly. On
subsequent passes (decode), the new K/V rows are appended using
`stack_rows`:

```rust
let (k_full, v_full) = match (&kv_cache.key_cache, &kv_cache.value_cache) {
    (Some(prev_k), Some(prev_v)) => {
        // Concatenate new K/V rows after the cached rows.
        let full_k = prev_k.stack_rows(&k_flat);
        let full_v = prev_v.stack_rows(&v_flat);
        (full_k, full_v)
    }
    (None, None) => {
        // First forward pass: just use the current K and V.
        (k_flat, v_flat)
    }
    _ => panic!("KV cache is in an inconsistent state"),
};
```

The `QwenModel` struct owns one `KVCache` per transformer layer, stored in a
`Vec<KVCache>`:

```rust
pub struct QwenModel {
    embed_tokens: Tensor,
    layers: Vec<TransformerBlock>,
    norm: RMSNorm,
    lm_head: Tensor,
    config: ModelConfig,
    kv_caches: Vec<KVCache>,  // One per layer
}
```

---

## 5. Stopping Conditions

The generation loop must eventually stop. Two conditions should be checked at
every decode step:

### Maximum Token Limit

The caller specifies a maximum number of tokens to generate. This is a hard
upper bound that prevents the model from generating indefinitely. In our CLI,
this is the `--max-tokens` flag, defaulting to 100.

The check is simple:

```
if len(generated_tokens) >= max_tokens:
    stop generation
```

This is important for several reasons:

- **Cost control**: Generating more tokens takes more time and memory.
- **Safety**: Without a limit, a degenerate sampling configuration could
  cause the model to generate an unbounded stream of tokens.
- **User expectation**: The user typically has a rough idea of how long the
  response should be.

### EOS Token (End of Sequence)

The model has a special "end of sequence" (EOS) token that signals it has
finished generating. When the model samples the EOS token, it means "I have
nothing more to say." The generation loop should stop.

The EOS token is defined in the tokenizer configuration. For the Qwen3
tokenizer, the EOS token ID is 151645 (corresponding to `<|im_end|>`), and
there is also an end-of-text token at ID 151643.

The check is:

```
if next_token == EOS_TOKEN_ID:
    stop generation
```

It is important to check both conditions at every step. The EOS check
provides a natural stopping point, while the max_tokens check provides a
safety net for cases where the model does not produce an EOS token (which can
happen with certain prompts or sampling configurations).

### Interaction Between the Two Conditions

The generation loop should check the max_tokens condition first (or in
conjunction with the EOS check), because:

- If the user sets `max_tokens = 0`, the model should produce no output at
  all.
- If the model generates the EOS token at step 5, but max_tokens is 100, the
  generation should stop at step 5.
- If the model has not generated an EOS token by step 100, the generation
  should stop regardless.

In pseudocode:

```
for step in 0..max_tokens:
    logits = model.forward(...)
    next_token = sample(logits, config)
    if next_token == EOS_TOKEN_ID:
        break
    output_tokens.append(next_token)
```

---

## 6. Streaming Output

### Why Streaming Matters

Imagine asking a language model to write an essay. If the model generates all
500 tokens before showing any output, the user stares at a blank screen for
several seconds, then suddenly sees the entire essay appear. This is a poor
user experience.

Streaming output solves this by displaying each token as soon as it is
generated. The user sees the text appearing incrementally, word by word, just
like watching someone type. This creates the perception of speed and
responsiveness, even if the total generation time is the same.

### Token-by-Token Decoding

After each decode step, the newly sampled token ID is converted back to its
string representation using the tokenizer's `decode` function. That string is
then immediately sent to the output. The key insight is that we do not need
to wait for the entire generation to finish before starting to display
output.

However, there is a subtlety: some tokens are partial UTF-8 sequences. For
example, a multi-byte Unicode character might be split across two tokens.
Attempting to decode each token individually could produce invalid UTF-8 or
garbled text. A robust streaming implementation buffers tokens and only
flushes complete characters to the output.

### Our Implementation: generate_with_callback

Our inference engine supports streaming through a callback mechanism. Instead
of accumulating all tokens and returning them at the end, the generator calls
a user-provided callback function after each token is generated:

```rust
pub trait InferenceCallback {
    fn on_token(&mut self, token_id: usize, token_text: &str);
}
```

The generation loop becomes:

```
for step in 0..max_tokens:
    logits = model.forward(...)
    next_token = sample(logits, config)
    if next_token == EOS_TOKEN_ID:
        break

    token_text = tokenizer.decode(next_token)
    callback.on_token(next_token, &token_text)
    output_tokens.append(next_token)
```

This design separates the generation logic from the output display. The same
`generate` function works for both streaming and non-streaming use cases:

- **Streaming**: Pass a callback that prints each token immediately.
- **Non-streaming**: Pass a callback that accumulates tokens into a buffer,
  then read the buffer after generation completes.

---

## 7. Implementation Details

### The InferenceEngine

Our `src/inference.rs` module will provide an `InferenceEngine` struct that
encapsulates the model, tokenizer, and sampling configuration:

```rust
pub struct InferenceEngine {
    model: QwenModel,
    tokenizer: Tokenizer,
    sampling_config: SamplingConfig,
    eos_token_id: usize,
}
```

The engine owns all the components needed for generation. The `QwenModel`
holds the weights and KV caches, the `Tokenizer` handles text-to-ID and
ID-to-text conversion, and the `SamplingConfig` controls the sampling
strategy.

### The generate() Method

The main entry point is the `generate` method, which takes a prompt string
and generation parameters, and returns the generated text:

```rust
impl InferenceEngine {
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
    ) -> String {
        // Step 1: Tokenize the prompt
        let token_ids = self.tokenizer.encode(prompt);

        // Step 2: Prefill — process all prompt tokens
        let logits = self.model.forward(&token_ids, 0);
        let mut next_token = sample(&logits.last_row(), &self.sampling_config);

        let mut generated_ids = Vec::new();
        if next_token == self.eos_token_id {
            return self.tokenizer.decode(&generated_ids);
        }
        generated_ids.push(next_token);

        // Step 3: Decode loop
        let mut pos = token_ids.len();
        for _ in 1..max_tokens {
            let logits = self.model.forward(&[next_token], pos);
            next_token = sample(&logits.last_row(), &self.sampling_config);
            pos += 1;

            if next_token == self.eos_token_id {
                break;
            }
            generated_ids.push(next_token);
        }

        // Step 4: Decode token IDs back to text
        self.tokenizer.decode(&generated_ids)
    }
}
```

### start_pos Tracking

The `start_pos` parameter is critical for correct inference. It tells the
model the absolute position of the first token in the current forward pass.
This is used for:

1. **RoPE**: Positional encodings are looked up by position index. If we
   pass the wrong `start_pos`, the model will apply the wrong rotational
   angles to Q and K, producing gibberish.

2. **Causal masking**: During prefill, `start_pos = 0` and the model
   applies a standard lower-triangular causal mask. During decode,
   `start_pos = prompt_len + tokens_generated_so_far`, and the model
   determines that the single new token can attend to all cached positions
   (no masking needed).

3. **KV cache consistency**: The cache stores K and V in order, so the
   `start_pos` must match the actual position of the new token in the
   overall sequence. If `start_pos` is wrong, the RoPE and causal mask
   will be incorrect, and the generated text will be incoherent.

The tracking is straightforward:

```
After prefill:    start_pos = prompt_len
After 1 decode:   start_pos = prompt_len + 1
After 2 decodes:  start_pos = prompt_len + 2
...
After k decodes:  start_pos = prompt_len + k
```

In the code, we initialize `pos = token_ids.len()` after prefill and
increment it by 1 after each decode step.

### Cache Reset for New Conversations

The KV cache accumulates state from all previous forward passes. When
starting a new conversation (or a new independent generation), the cache must
be reset. Otherwise, the new prompt would be processed in the context of the
old conversation, producing nonsensical output.

Our `QwenModel` provides a method to reset all KV caches:

```rust
impl QwenModel {
    pub fn reset_kv_cache(&mut self) {
        for cache in &mut self.kv_caches {
            cache.key_cache = None;
            cache.value_cache = None;
        }
    }
}
```

This sets every layer's K and V caches back to `None`, which is the same
state as a freshly created model. The next forward pass will be a prefill
that populates the caches from scratch.

When should the cache be reset?

- **Between independent conversations**: If the user starts a completely new
  topic, the old context is irrelevant and should be cleared.
- **When the context window is full**: If the total sequence length
  (prompt + generated) approaches `max_position_embeddings`, the model can
  no longer attend to all tokens correctly. The cache should be reset, and
  the conversation should start fresh (or use a sliding window / truncation
  strategy).
- **Between test cases**: In unit tests, each test should start with a clean
  cache to ensure isolation.

In a chat application, the cache is typically reset at the start of each new
session. Within a session, the cache grows as the conversation progresses,
allowing the model to maintain context across multiple turns.

### Putting It All Together

Here is a complete example of using the inference engine from the CLI:

```rust
fn main() {
    let args = Args::parse();

    // Load the model from disk
    let mut engine = InferenceEngine::from_dir(&args.model_dir)
        .expect("Failed to load model");

    // Configure sampling
    engine.sampling_config = SamplingConfig {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        seed: args.seed,
    };

    // Generate text
    let output = engine.generate(&args.prompt, args.max_tokens);
    println!("{}", output);
}
```

The `InferenceEngine` hides the details of tokenization, prefill, decode,
KV cache management, and sampling behind a simple `generate` method. The
caller provides a prompt and a token limit, and gets back generated text.

### The Full Pipeline, Visualized

```
User types: "Explain quantum computing"

                    TOKENIZE
"Explain quantum computing" → [4523, 18364, 18342, 15496]
                                 ↓
                    PREFILL (start_pos=0)
model.forward([4523, 18364, 18342, 15496], 0)
  - Embedding lookup:    [4, 1024]
  - 28 transformer blocks (KV cache populated)
  - Final norm + lm_head: [4, 151936]
  - Take logits at position 3
  - Sample: token 311 (="Quantum")
                                 ↓
                    DECODE LOOP (start_pos=4, 5, 6, ...)

Step 1: model.forward([311], 4) → sample → token 14758 (=" computing")
Step 2: model.forward([14758], 5) → sample → token 374 (=" is")
Step 3: model.forward([374], 6) → sample → token 264 (=" a")
Step 4: model.forward([264], 7) → sample → token 28354 (=" field")
  ...
Step N: model.forward([token], N+3) → sample → 151645 (EOS)
                                 ↓
                    DECODE
[311, 14758, 374, 264, 28354, ...] → "Quantum computing is a field..."
```

Each step of the decode loop adds exactly one row to every layer's KV cache.
The `start_pos` increases by 1 each step, ensuring correct RoPE encodings
and causal masking.

---

## Summary

| Concept | Key Idea |
|---------|----------|
| Autoregressive generation | Each token depends on all previous tokens; generate one at a time |
| Prefill | Process all prompt tokens at once; populate KV cache |
| Decode | Process one new token per step; use KV cache to avoid recomputation |
| KV cache | Store past K,V to reduce per-step cost from recomputing all K,V to computing only the new one |
| start_pos | Track absolute position for correct RoPE and causal masking |
| Stopping conditions | Max token limit (user-configured) and EOS token (model-decided) |
| Streaming | Display each token as it is generated for responsive user experience |
| Cache reset | Clear KV cache between independent conversations |

Autoregressive inference is the bridge between a trained model and usable
output. The model's forward pass computes logits; the generation loop turns
those logits into a sequence of tokens; and the tokenizer turns those tokens
back into text. The KV cache makes this process efficient, and streaming
output makes it feel fast to the user.
