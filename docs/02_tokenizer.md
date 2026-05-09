# 2. Tokenization: How Text Becomes Numbers

This document explains how a large language model converts human-readable text
into the integer token IDs it actually operates on. The process is called
**tokenization**, and the component that performs it is the **tokenizer**.

If you have read `01_transformer_basics.md`, you already know that a tokenizer
sits at the very beginning of the inference pipeline. Here we go deep: what
tokenization schemes exist, how Byte Pair Encoding (BPE) works, why modern
models use byte-level BPE, and how the Qwen3 tokenizer is built.

---

## 1. What Is Tokenization?

Neural networks operate on numbers, not text. Before a transformer can do
anything with a sentence like "Hello world", that sentence must be converted
into a sequence of integers. Tokenization is the bridge between the two worlds.

But how should we map text to numbers? There are several strategies, each with
different trade-offs.

### 1.1 Character-Level Tokenization

The simplest approach: each character gets its own ID. The vocabulary is small
(there are only 128 ASCII characters, or about 1,000 common Unicode code
points), and every possible text can be represented.

```
"Hello" → [72, 101, 108, 108, 111]
          H=72   e=101  l=108  l=108  o=111
```

**Problem**: Characters carry very little meaning on their own. The letter "e"
appears in almost every English word, so the model must learn that "e" followed
by "l" followed by "l" followed by "o" means "ello" — a common suffix — purely
from context. This puts an unnecessary burden on the model. Sequences also
become very long: a 1,000-word essay might be 5,000 characters, and the
transformer must process every one of them.

### 1.2 Word-Level Tokenization

The opposite extreme: each word gets its own ID. "Hello" is one token, "world"
is another.

```
"Hello world" → [15496, 995]
```

**Problem 1 — Vocabulary explosion**: English alone has hundreds of thousands
of words. Add in names, technical terms, and multilingual text, and you quickly
exceed a million. An embedding table of 1,000,000 x 1,024 = 4 GB is
impractical.

**Problem 2 — Out-of-vocabulary (OOV) words**: What happens when the model
encounters a word it has never seen before? With word-level tokenization, the
answer is: it cannot represent it at all. The model must fall back to a special
`<UNK>` (unknown) token, losing all information about that word. This is
catastrophic for languages with rich morphology (Finnish, Turkish), where new
word forms are created productively by adding suffixes.

**Problem 3 — Different forms of the same word**: "run", "running", "runs",
"ran" all get separate IDs. The model must independently learn the semantic
relationship between them, even though they share a common root.

### 1.3 Subword-Level Tokenization: The Sweet Spot

Subword tokenization splits text into units that are larger than individual
characters but smaller than whole words. Common words stay intact ("the" is one
token), while rare or complex words are broken into meaningful pieces:

```
"unbelievable" → ["un", "believable"]
"tokenization" → ["token", "ization"]
"hamburger"    → ["ham", "burger"]
```

This elegantly solves the problems of both extremes:

- **Vocabulary is bounded**: A vocabulary of 30,000-150,000 subwords covers
  virtually all text in any language.
- **No OOV words**: Any text can be represented, because any word can be
  decomposed into characters and then rebuilt from subword pieces.
- **Meaningful units**: "un" and "ization" carry meaning that individual
  characters do not, reducing the burden on the model.

The three main subword tokenization algorithms are:

| Algorithm | Used By | Key Idea |
|-----------|---------|----------|
| BPE | GPT-2, GPT-4, LLaMA, Qwen | Merge most frequent pair iteratively |
| WordPiece | BERT, DistilBERT | Merge pair that maximizes likelihood |
| Unigram | T5, ALBERT | Start large, prune least useful tokens |

BPE is by far the most common for modern decoder-only models, so that is what
we focus on.

---

## 2. Byte Pair Encoding (BPE) — The Algorithm

Byte Pair Encoding was originally developed as a text compression algorithm
(Sennrich et al., 2016 adapted it for neural machine translation). The idea is
simple and elegant: start with a vocabulary of individual characters, then
repeatedly merge the most frequent pair of adjacent tokens into a new token
until the vocabulary reaches the desired size.

### 2.1 Training: Learning the Merge Rules

Suppose our training corpus consists of the following words with their
frequencies:

```
"low"     × 5
"lower"   × 2
"newest"  × 6
"widest"  × 3
```

We first split each word into characters (we use the special end-of-word symbol
`</w>` to mark word boundaries, so the model can reconstruct where words end):

```
l o w </w>           × 5
l o w e r </w>       × 2
n e w e s t </w>     × 6
w i d e s t </w>     × 3
```

**Step 1**: Count all adjacent pairs and find the most frequent one.

```
Pair frequencies:
  (e, s) = 6 + 3 = 9    ← most frequent
  (l, o) = 5 + 2 = 7
  (o, w) = 5 + 2 = 7
  (w, e) = 2 + 6 = 8
  (s, t) = 6 + 3 = 9
  ...
```

We have a tie between (e, s) and (s, t). Let us pick (e, s) arbitrarily.
We merge it into a new token "es" and add it to the vocabulary.

```
l o w </w>             × 5
l o w es r </w>        × 2
n es w es t </w>       × 6   (both "e s" pairs merged)
w i d es t </w>        × 3
```

**Step 2**: Count pairs again. The most frequent pair is now (es, t), which
appears 6 + 3 = 9 times.

```
l o w </w>              × 5
l o w es r </w>         × 2
n es w est </w>         × 6
w i d est </w>          × 3
```

**Step 3**: The most frequent pair is now (l, o) at 7 times. Merge it.

```
lo w </w>               × 5
lo w es r </w>          × 2
n es w est </w>         × 6
w i d est </w>          × 3
```

**Step 4**: (lo, w) appears 7 times. Merge.

```
low </w>                × 5
low es r </w>           × 2
n es w est </w>         × 6
w i d est </w>          × 3
```

And so on. After K merge steps, we have a vocabulary of the original characters
plus K new tokens, and an ordered list of K merge rules.

### 2.2 Encoding: Applying the Merge Rules

At inference time, we apply the learned merge rules to new text. The algorithm
is:

1. Split the input word into individual characters.
2. Find the pair of adjacent tokens whose merge has the **lowest rank**
   (highest priority, i.e., was learned earliest).
3. Merge all occurrences of that pair.
4. Repeat until no more merges can be applied.

The key insight: we always apply the **highest-priority** merge first, not the
most frequent one. During training, frequency determined the order. During
encoding, we simply follow that order.

Let us trace through encoding the word "lowest":

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

The word "lowest" is tokenized into two subwords: "low" and "est". Both carry
meaning — "low" is a recognizable root and "est" is a common superlative
suffix. The model can learn representations for both pieces independently and
combine them.

### 2.3 Why This Works So Well

BPE naturally produces subwords that reflect the statistical structure of the
training corpus. In English:

- Common words like "the", "and", "is" get their own tokens because they are
  formed by early merges.
- Less common words like "unbelievable" get split into meaningful subwords:
  "un", "believ", "able".
- Very rare words like "supercalifragilistic" get split into individual
  characters or very small subwords.

The vocabulary size is a knob you can turn: more merges = larger vocabulary =
shorter sequences = more per-token meaning, at the cost of a bigger embedding
table.

---

## 3. Byte-Level BPE

Standard BPE operates on Unicode characters, but this creates practical
problems. The GPT-2 paper (Radford et al., 2019) introduced **byte-level BPE**
to solve them.

### 3.1 The Unicode Problem

Standard BPE builds its base vocabulary from the characters in the training
data. For English, that is about 70 characters (a-z, A-Z, 0-9, punctuation).
For Chinese, it is tens of thousands of characters. For emoji and special
symbols, even more.

This creates several issues:

1. **Inconsistent vocabulary size**: A model trained on English might have a
   70-character base vocabulary, while one trained on Chinese has 30,000+.
2. **Cross-lingual interference**: If a rare Chinese character appears in an
   English training corpus, it takes up a vocabulary slot but is almost never
   useful.
3. **Out-of-vocabulary characters**: Any character not seen during training
   cannot be represented at all.

### 3.2 The Byte-Level Solution

Byte-level BPE sidesteps these problems entirely by working on **bytes**
instead of characters. Any text, in any language, can be represented as a
sequence of bytes using UTF-8 encoding. There are only 256 possible byte
values, so the base vocabulary is always exactly 256 — small, fixed, and
universal.

But there is a catch: raw bytes include control characters (byte 0 = null,
byte 10 = newline, byte 32 = space), which are problematic for BPE. BPE
operates by looking at pairs of tokens, and if some "tokens" are invisible
whitespace or control characters, it is harder to reason about the merges.

The solution is a **byte-to-unicode mapping**. We map each of the 256 byte
values to a unique, visible Unicode character:

- **Direct mapping** (188 bytes): Printable ASCII bytes 33-126 (`!` through
  `~`), plus Latin-1 supplement bytes 161-172 and 174-255, map to their
  corresponding Unicode code points. These bytes already have visible character
  representations, so we keep them as-is.

- **Shifted mapping** (68 bytes): The remaining byte values — control
  characters (0-32), delete and C1 controls (127-160), and soft hyphen (173) —
  are mapped to Unicode code points starting at U+0100. So byte 0 maps to
  `Ā` (U+0100), byte 1 maps to `ā` (U+0101), and so on.

Here is a partial view of the mapping for the shifted bytes that are most
commonly encountered:

| Byte value | Unicode char | Code point | Common meaning |
|-----------|-------------|-----------|---------------|
| 0         | Ā | U+0100 | Null byte |
| 9         | Ĩ | U+0128 | Tab |
| 10        | ĩ | U+0129 | Newline |
| 13        | ļ | U+013C | Carriage return |
| 32        | Ġ | U+0120 | **Space** |
| 127       | ł | U+0142 | Delete |

The most important mapping to remember: **space (byte 32) maps to Ġ**. When
you see `Ġ` in a tokenizer's vocabulary, it represents a space character. A
token like `Ġthe` means "the word 'the' preceded by a space" — i.e., an
intra-sentence occurrence rather than a sentence-starting one.

The 188 directly-mapped bytes include all the characters you would normally
type on a keyboard: letters, digits, and common punctuation. They map to
themselves:

| Byte range | Characters | Example |
|-----------|-----------|---------|
| 33-126 | `!` through `~` | `A` = byte 65 = char 'A' |
| 161-172 | `¡` through `¬` | `©` = byte 169 = char '©' |
| 174-255 | `®` through `ÿ` | `ü` = byte 252 = char 'ü' |

### 3.3 How Byte-Level BPE Works

The full pipeline is:

1. Convert the input text to UTF-8 bytes.
2. Map each byte to its byte-level Unicode character using the table above.
3. Apply BPE merges on the resulting sequence of byte-level characters.
4. Look up each resulting token in the vocabulary to get its ID.

For decoding:

1. Look up each token ID to get the token string.
2. Concatenate all token strings.
3. Map each byte-level character back to its original byte.
4. Decode the resulting bytes as UTF-8.

### 3.4 Why This Handles All Languages

UTF-8 can represent any Unicode character. A Chinese character like `你` is
encoded as three bytes: `0xE4 0xBD 0xA0`. In byte-level BPE, this becomes three
byte-level characters, which can then be merged by BPE into larger subword
units.

For Chinese text, BPE will quickly learn to merge the common three-byte
sequences that correspond to frequent characters into single tokens. Less
common characters remain as multi-token sequences. This means:

- The tokenizer works for **any language** without special configuration.
- The vocabulary size stays bounded, regardless of how many languages are in
  the training data.
- Cross-lingual transfer is possible: if the model learns that `Ġun` means
  "un-" in English, it can apply that knowledge even in a mixed-language
  context.

For code, the same principle applies. Python keywords like `def`, `class`,
`return` become single tokens. Less common identifiers like `quantize` become
`["quant", "ize"]`. Special characters like `{`, `}`, `=`, `==` are all single
tokens. This is why modern LLMs can write code reasonably well — the tokenizer
understands the "vocabulary" of programming languages.

---

## 4. The Qwen3 Tokenizer

Qwen3 uses a byte-level BPE tokenizer based on the tiktoken/cl100k_base
family, similar to GPT-4's tokenizer. It is distributed as a HuggingFace
`tokenizer.json` file.

### 4.1 Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Vocabulary size | 151,936 | Total number of tokens (base 256 + merges + special) |
| Tokenizer type | Byte-level BPE | Operates on UTF-8 bytes mapped to Unicode chars |
| Special tokens | 2+ | EOS and other control tokens |
| Pre-tokenizer | GPT-2 style regex | Splits text into words before BPE |
| Decoder | Byte-level | Converts byte-level chars back to bytes |

### 4.2 Special Tokens

Qwen3 defines several special tokens that serve specific roles in the model's
input and output format:

| Token | ID | Purpose |
|-------|----|---------|
| `<\|endoftext\|>` | 151643 | End-of-sequence (EOS) marker |
| `<\|im_start\|>` | 151644 | Start of a chat message |
| `<\|im_end\|>` | 151645 | End of a chat message |

The EOS token is the most important one at inference time: it signals that the
model has finished generating. During autoregressive generation, we stop as soon
as the model outputs the EOS token ID (151643).

The `im_start` and `im_end` tokens are used to format conversations in the
ChatML format. A typical conversation looks like:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
2+2 equals 4.<|im_end|>
```

Each message is wrapped in `im_start` and `im_end` markers, with the role
(system, user, assistant) following `im_start`. This format was introduced by
OpenAI and adopted by many models including Qwen.

### 4.3 The tokenizer.json File

HuggingFace distributes the tokenizer as a JSON file with this structure:

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

The key sections are:

- **`added_tokens`**: Special tokens with their IDs. These are added to the
  vocabulary before BPE processing and are matched literally in the input text.
- **`model.vocab`**: The complete vocabulary mapping from token strings to IDs.
  This includes the 256 base byte-level characters, all BPE merge results, and
  any added tokens.
- **`model.merges`**: The ordered list of BPE merge rules. Each entry is
  `"token_a token_b"`, where the rank is the position in the list.
- **`pre_tokenizer`**: Configuration for splitting text into words before BPE.
  The GPT-2 regex pattern ensures that BPE merges never cross word boundaries.
- **`decoder`**: Tells us how to convert token strings back to text. For
  byte-level BPE, this is `"ByteLevel"`, meaning we reverse the byte-to-unicode
  mapping.

### 4.4 Multilingual Support

The Qwen3 tokenizer handles Chinese text efficiently. Common Chinese
characters are single tokens, while less common ones are split into 2-3 tokens
(the UTF-8 byte sequences). This is a significant improvement over earlier
tokenizers where Chinese text would produce 2-3x more tokens than equivalent
English text, making the model slower and more expensive for Chinese users.

With 151,936 tokens in the vocabulary, Qwen3 can dedicate a large fraction
to Chinese, code, and other specialized domains while still covering English
well. The result is a tokenizer that produces roughly similar token counts for
English and Chinese text of the same semantic content.

---

## 5. Encoding: Text to Token IDs

Now let us walk through the full encoding process step by step, using the
example text `"Hello world"`.

### 5.1 Step 1: Pre-tokenization

Pre-tokenization splits the input text into "words" so that BPE merges never
cross word boundaries. The GPT-2 pre-tokenizer uses a regex pattern that
captures:

- Contractions: `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`
- Letter sequences (with optional leading space): `Hello`, ` world`
- Digit sequences (with optional leading space): `42`, ` 123`
- Punctuation sequences (with optional leading space): `!`, ` .`
- Whitespace: newlines, trailing spaces

For our example:

```
Input:  "Hello world"
Split:  ["Hello", " world"]
```

Notice that the space before "world" is attached to the word as a leading
space. This is the GPT-2 convention: spaces are part of the following word,
not separate tokens.

### 5.2 Step 2: Convert to Byte-Level Characters

Each word is converted to UTF-8 bytes, and each byte is mapped to its
byte-level Unicode character.

For "Hello":
```
H → byte 72 → 'H' (direct mapping)
e → byte 101 → 'e' (direct mapping)
l → byte 108 → 'l' (direct mapping)
l → byte 108 → 'l' (direct mapping)
o → byte 111 → 'o' (direct mapping)
Result: ["H", "e", "l", "l", "o"]
```

For " world" (with the leading space):
```
(space) → byte 32 → 'Ġ' (shifted mapping)
w → byte 119 → 'w' (direct mapping)
o → byte 111 → 'o' (direct mapping)
r → byte 114 → 'r' (direct mapping)
l → byte 108 → 'l' (direct mapping)
d → byte 100 → 'd' (direct mapping)
Result: ["Ġ", "w", "o", "r", "l", "d"]
```

Since all the characters in "Hello world" are printable ASCII, the byte-level
mapping is almost trivial — only the space character gets transformed into `Ġ`.

For Chinese text like "你好" (hello in Chinese), the conversion is more
interesting:
```
你 → bytes [0xE4, 0xBD, 0xA0] → ['ä', '½', ' ']
好 → bytes [0xE5, 0xA5, 0xBD] → ['å', '¥', '½']
```

Wait — those do not look right. That is because bytes 0xE4, 0xBD, 0xA0 fall in
the "direct mapping" range (161-255), so they map to their Latin-1
correspondents: `ä` (0xE4), `½` (0xBD), etc. These look odd, but they are just
intermediate representations — the BPE algorithm will quickly merge common
sequences like these into proper Chinese character tokens.

### 5.3 Step 3: Apply BPE Merges

For each word, we apply the BPE merge algorithm:

**"Hello"** — starting from `["H", "e", "l", "l", "o"]`:

The BPE algorithm looks at all adjacent pairs and finds the one with the lowest
rank (highest priority) in the merge list. It merges that pair and repeats.

In the actual Qwen3 tokenizer, "Hello" (capital H) might be tokenized as
something like `["Hello"]` if it is a common enough word in the training data.
If it is not a single token, it might be split as `["H", "ello"]` or
`["He", "llo"]`, depending on which merges have been learned.

**" world"** — starting from `["Ġ", "w", "o", "r", "l", "d"]`:

The merge sequence for " world" in the actual Qwen3 tokenizer might go:
1. `Ġ` + `w` → `Ġw` (space + w is a very common sequence)
2. `o` + `r` → `or` (common in English)
3. `Ġw` + `or` → `Ġwor`
4. `l` + `d` → `ld` (common ending)
5. `Ġwor` + `ld` → `Ġworld`

Result: `["Ġworld"]` — a single token meaning " world" (space + world).

### 5.4 Step 4: Vocabulary Lookup

Each resulting token string is looked up in the vocabulary to get its integer
ID:

```
"Hello" → ID 15496 (example; actual ID depends on the tokenizer)
"Ġworld" → ID 995 (example)
```

The final encoded output is: `[15496, 995]`

These are the integer IDs that get fed into the embedding table as the first
step of the transformer forward pass.

---

## 6. Decoding: Token IDs to Text

Decoding is the reverse process. Given a sequence of token IDs, we reconstruct
the original text.

### 6.1 Step 1: Reverse Vocabulary Lookup

Each token ID is looked up in the reverse vocabulary (ID → token string):

```
[15496, 995] → ["Hello", "Ġworld"]
```

### 6.2 Step 2: Concatenate Token Strings

The token strings are simply concatenated:

```
"Hello" + "Ġworld" = "HelloĠworld"
```

### 6.3 Step 3: Convert Byte-Level Characters Back to Bytes

Each character in the concatenated string is mapped back to its byte value
using the inverse of the byte-to-unicode mapping:

```
'H' → byte 72
'e' → byte 101
'l' → byte 108
'l' → byte 108
'o' → byte 111
'Ġ' → byte 32 (space!)
'w' → byte 119
'o' → byte 111
'r' → byte 114
'l' → byte 108
'd' → byte 100
```

The key step: `Ġ` maps back to byte 32, which is the space character.

### 6.4 Step 4: Decode Bytes as UTF-8

The byte sequence `[72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]` is
decoded as UTF-8:

```
"Hello world"
```

We have recovered the original text.

### 6.5 Edge Cases in Decoding

**Partial tokens**: Sometimes a token might end in the middle of a multi-byte
UTF-8 sequence. For example, the Chinese character `你` is three bytes
(0xE4, 0xBD, 0xA0). If the tokenizer splits this across two tokens, the first
token might end with 0xE4 (an incomplete UTF-8 sequence) and the second starts
with 0xBD, 0xA0. The decoder must concatenate all token strings first, convert
them all to bytes, and then decode the complete byte sequence as UTF-8. Decoding
token by token would fail.

**Special tokens**: Special tokens like `<|endoftext|>` are not part of the
byte-level encoding. They are stored as-is in the vocabulary and decoded as
literal strings. In our implementation, characters that are not in the byte
decoder (because they are part of special token strings) are handled by
encoding them as their UTF-8 representation directly.

---

## 7. Implementation Details

Our Rust implementation in `src/tokenizer.rs` follows the algorithm described
above. Here we highlight the key design decisions.

### 7.1 The Tokenizer Struct

```rust
pub struct Tokenizer {
    vocab: HashMap<String, usize>,           // token string → ID
    id_to_token: HashMap<usize, String>,      // ID → token string
    merges: Vec<(String, String)>,            // merge rules (ordered)
    merge_ranks: HashMap<(String, String), usize>,  // merge pair → rank
    special_tokens: HashMap<String, usize>,   // special token → ID
    byte_encoder: [char; 256],                // byte → unicode char
    byte_decoder: HashMap<char, u8>,          // unicode char → byte
}
```

The `merge_ranks` HashMap is derived from `merges` at construction time. It
allows the BPE algorithm to check the priority of any adjacent pair in O(1)
time, rather than linearly scanning the merge list.

### 7.2 The Byte Encoder

The `build_byte_encoder` function constructs the byte-to-unicode mapping:

```rust
fn build_byte_encoder() -> ([char; 256], HashMap<char, u8>) {
    let mut encoder = ['\0'; 256];
    let mut decoder = HashMap::new();

    // Direct-mapping bytes (188 total)
    let direct_bytes: Vec<u8> = (33..=126)
        .chain(161..=172)
        .chain(174..=255)
        .collect();

    for &b in &direct_bytes {
        let c = char::from_u32(b as u32).unwrap();
        encoder[b as usize] = c;
        decoder.insert(c, b);
    }

    // Shifted bytes (68 remaining) → U+0100 and up
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

This produces exactly the GPT-2 byte encoder. The result is deterministic and
matches what HuggingFace's tokenizers library and OpenAI's tiktoken use.

### 7.3 The BPE Merge Algorithm

The `apply_bpe` method implements the standard BPE merge algorithm:

```rust
fn apply_bpe(&self, tokens: &[String]) -> Vec<String> {
    let mut tokens = tokens.to_vec();

    loop {
        // Find the pair with the lowest merge rank.
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
            None => break,  // No more merges possible
        };

        // Merge all occurrences of the best pair.
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

Each iteration of the loop takes O(n) time to scan for the best pair and O(n)
to perform the merge, where n is the current number of tokens. In the worst
case, we do O(m) iterations (where m is the number of applicable merges),
giving O(m * n) total. For a typical word, n is small (5-15 characters) and m
is at most n-1, so this is fast enough for our educational purposes.

Production tokenizers (like tiktoken) optimize this further by using more
sophisticated data structures, but the algorithm is the same.

### 7.4 Pre-tokenization Simplification

Our implementation uses a simplified pre-tokenizer that splits on character
categories (letters, digits, punctuation, whitespace) rather than implementing
the full GPT-2 regex. This means:

- **Correct for most English text**: Words with leading spaces, punctuation,
  and numbers are handled correctly.
- **Slightly different for contractions**: "don't" might be split differently
  than the reference tokenizer (which handles `'s`, `'t`, `'re`, etc.
  specially).
- **Slightly different for multi-character punctuation**: Sequences like
  `...` or `==` may be split differently.

For educational purposes, this is fine. The BPE algorithm itself is correct,
and the pre-tokenizer can be improved later by adding the `regex` crate as a
dependency.

### 7.5 Limitations

Our implementation has several deliberate simplifications:

1. **No regex pre-tokenizer**: We use a character-category-based splitter
   instead of the full GPT-2 regex pattern. This produces slightly different
   tokenization for some inputs.

2. **No normalization**: Some tokenizers apply Unicode normalization (NFC, NFD)
   before encoding. We skip this step, which means visually identical strings
   with different Unicode representations will tokenize differently.

3. **No truncation/padding**: The tokenizer does not handle truncating long
   inputs or padding short ones to a fixed length. These are typically handled
   by the calling code, not the tokenizer itself.

4. **Performance**: Our implementation is not optimized for speed. For
   production use, you would want to cache the pre-tokenization results, use a
   more efficient merge algorithm, and possibly implement parallel encoding.

Despite these simplifications, the implementation correctly demonstrates all
the core concepts of byte-level BPE tokenization. If you feed it the Qwen3
`tokenizer.json` file, it will tokenize most English and Chinese text
correctly, and decode the results back to the original text.

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Tokenization | Converts text to integers for the model |
| Subword tokenization | Balances vocabulary size and sequence length |
| BPE | Iteratively merges the most frequent pair of tokens |
| Byte-level BPE | Operates on bytes, not characters; handles all languages |
| Byte-to-unicode mapping | Makes all bytes "visible" for BPE; space = Ġ |
| Pre-tokenization | Splits text into words so BPE does not cross boundaries |
| Encoding | Pre-tokenize → byte-level convert → BPE merge → vocab lookup |
| Decoding | Vocab reverse lookup → concatenate → byte-level decode → UTF-8 |

---

## Further Reading

- Sennrich, R., Haddow, B., and Birch, A. "Neural Machine Translation of Rare
  Words with Subword Units." ACL 2016. The paper that introduced BPE for NLP.

- Radford, A., et al. "Language Models are Unsupervised Multitask Learners."
  2019. The GPT-2 paper that introduced byte-level BPE.

- Kudo, T. "Subword Regularization: Improving Neural Network Translation
  Models with Multiple Subword Candidates." ACL 2018. Introduces the Unigram
  tokenizer used by SentencePiece.

- HuggingFace Tokenizers Documentation:
  https://huggingface.co/docs/tokenizers/ -- the library whose format we read.

- OpenAI Tiktoken:
  https://github.com/openai/tiktoken -- the fast BPE tokenizer used by GPT-4.