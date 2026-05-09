//! BPE (Byte Pair Encoding) tokenizer implementation.
//!
//! This module implements a byte-level BPE tokenizer that can read HuggingFace's
//! `tokenizer.json` format. It is the bridge between human-readable text and the
//! integer token IDs that the transformer model operates on.
//!
//! # What Is BPE?
//!
//! Byte Pair Encoding is a subword tokenization algorithm. It starts with a
//! vocabulary of individual characters and iteratively merges the most frequent
//! adjacent pair of tokens into a new token, growing the vocabulary until it
//! reaches the desired size. At inference time, we apply the *learned* merge
//! rules in priority order to segment text into tokens.
//!
//! # Byte-Level BPE
//!
//! Standard BPE operates on Unicode characters, which creates problems for
//! multilingual text: the character vocabulary is unbounded, and rare characters
//! cannot be handled gracefully. **Byte-level BPE** (introduced by GPT-2) solves
//! this by first converting text to raw bytes (UTF-8), then mapping each byte to
//! a fixed Unicode character via a deterministic table. BPE merges are then
//! learned and applied on these byte-level characters.
//!
//! The byte-to-unicode mapping works as follows:
//! - Printable ASCII bytes (33-126), plus Latin-1 supplement bytes (161-172,
//!   174-255), map to their corresponding Unicode code points.
//! - All remaining bytes (control characters, space, delete, etc.) map to
//!   Unicode code points starting at U+0100 (Ā).
//!
//! This ensures every possible byte has a unique visible character representation,
//! and the vocabulary is always exactly 256 base tokens plus any merges.
//!
//! # Encoding Pipeline
//!
//! Encoding text to token IDs proceeds in four steps:
//! 1. **Pre-tokenization**: Split the input text into "words" using a heuristic
//!    that separates letters, digits, punctuation, and whitespace.
//! 2. **Byte-level conversion**: For each word, convert to UTF-8 bytes and map
//!    each byte to its byte-level Unicode character.
//! 3. **BPE merging**: For each word (as a sequence of single-character tokens),
//!    repeatedly find the highest-priority merge pair and combine them, until no
//!    more merges apply.
//! 4. **Vocabulary lookup**: Look up each resulting token string in the
//!    vocabulary to get its integer ID.
//!
//! # Decoding Pipeline
//!
//! Decoding token IDs back to text is simpler:
//! 1. Look up each ID in the reverse vocabulary to get the token string.
//! 2. Concatenate all token strings.
//! 3. Convert byte-level Unicode characters back to bytes using the inverse
//!    mapping.
//! 4. Decode the resulting bytes as UTF-8.
//!
//! # Simplifications
//!
//! This is an educational implementation. It does not perfectly replicate the
//! HuggingFace tokenizer. The main simplification is in pre-tokenization: we
//! use a simple character-category-based splitter instead of the full GPT-2
//! regex. This means that for some inputs, the tokenization may differ from
//! the reference implementation. However, the BPE merge algorithm and
//! byte-level encoding are faithful to the standard.

use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

// ─────────────────────────────────────────────────────────────────────────────
// JSON schema for tokenizer.json
// ─────────────────────────────────────────────────────────────────────────────

/// Top-level structure of a HuggingFace `tokenizer.json` file.
///
/// Only the fields we actually use are deserialized; everything else is
/// silently ignored by serde.
#[derive(Deserialize)]
struct TokenizerFile {
    added_tokens: Option<Vec<AddedToken>>,
    model: BpeModel,
}

/// An "added token" entry — special tokens like `<|endoftext|>`.
#[derive(Deserialize)]
struct AddedToken {
    id: usize,
    content: String,
    special: Option<bool>,
}

/// The BPE model section of `tokenizer.json`, containing the vocabulary
/// and merge rules.
#[derive(Deserialize)]
struct BpeModel {
    vocab: HashMap<String, usize>,
    /// Merge rules. Can be either:
    /// - `["a b", "c d", ...]` (string format, used by most tokenizers)
    /// - `[["a", "b"], ["c", "d"], ...]` (array format, used by Qwen3)
    #[serde(default, deserialize_with = "deserialize_merges")]
    merges: Option<Vec<(String, String)>>,
}

/// Custom deserializer that handles both merge formats:
/// - String format: `"a b"` → `("a", "b")`
/// - Array format: `["a", "b"]` → `("a", "b")`
fn deserialize_merges<'de, D>(de: D) -> Result<Option<Vec<(String, String)>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value: Option<serde_json::Value> = Option::deserialize(de)?;

    let Some(value) = value else {
        return Ok(None);
    };

    let serde_json::Value::Array(arr) = value else {
        return Ok(None);
    };

    let mut merges = Vec::new();
    for item in arr {
        match item {
            serde_json::Value::String(s) => {
                // String format: "a b" → split on first space
                if let Some(space_pos) = s.find(' ') {
                    merges.push((s[..space_pos].to_string(), s[space_pos + 1..].to_string()));
                }
            }
            serde_json::Value::Array(inner) if inner.len() == 2 => {
                // Array format: ["a", "b"]
                if let (Some(serde_json::Value::String(a)), Some(serde_json::Value::String(b))) =
                    (inner.first(), inner.get(1))
                {
                    merges.push((a.clone(), b.clone()));
                }
            }
            _ => {}
        }
    }

    Ok(Some(merges))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tokenizer
// ─────────────────────────────────────────────────────────────────────────────

/// A BPE tokenizer that reads HuggingFace's `tokenizer.json` format.
///
/// Use [`Tokenizer::from_file`] to load from disk, then [`Tokenizer::encode`]
/// to convert text to token IDs and [`Tokenizer::decode`] to convert back.
///
/// # Example
///
/// ```no_run
/// use qwen3_5_rs::tokenizer::Tokenizer;
/// use std::path::Path;
///
/// let tokenizer = Tokenizer::from_file(Path::new("model_dir/tokenizer.json")).unwrap();
/// let ids = tokenizer.encode("Hello world");
/// let text = tokenizer.decode(&ids);
/// ```
pub struct Tokenizer {
    /// Mapping from token string to token ID.
    vocab: HashMap<String, usize>,
    /// Mapping from token ID to token string (for decoding).
    id_to_token: HashMap<usize, String>,
    /// BPE merge rules: list of `(token_a, token_b)` pairs in priority order.
    ///
    /// The index in this vector is the merge rank — lower indices have higher
    /// priority and should be applied first.
    ///
    /// Note: This field is kept for introspection and debugging, but the BPE
    /// algorithm uses [`Self::merge_ranks`] for fast lookup instead.
    #[allow(dead_code)]
    merges: Vec<(String, String)>,
    /// Merge ranks for fast lookup: `(token_a, token_b) -> rank`.
    ///
    /// This is derived from [`Self::merges`] at construction time so that the
    /// BPE algorithm can quickly find the best merge for any adjacent pair.
    merge_ranks: HashMap<(String, String), usize>,
    /// Special tokens (e.g., `<|endoftext|>`) mapped to their IDs.
    special_tokens: HashMap<String, usize>,
    /// Byte-to-unicode mapping for byte-level BPE.
    ///
    /// `byte_encoder[b]` is the Unicode character that represents byte value `b`
    /// in the tokenized text. For example, space (0x20) maps to `Ġ` (U+0120).
    byte_encoder: [char; 256],
    /// Unicode-to-byte mapping for decoding byte-level tokens.
    ///
    /// This is the inverse of [`Self::byte_encoder`].
    byte_decoder: HashMap<char, u8>,
}

impl Tokenizer {
    /// Load a tokenizer from a `tokenizer.json` file on disk.
    ///
    /// The file is expected to follow the HuggingFace tokenizers library format.
    /// It must contain a `"model"` key with `"type": "BPE"`, a `"vocab"` object,
    /// and optionally a `"merges"` array.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, the JSON is malformed, or
    /// the required fields (`model.vocab`) are missing.
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let json = fs::read_to_string(path)?;
        Self::from_json(&json)
    }

    /// Parse a tokenizer from a JSON string.
    ///
    /// This is useful when the tokenizer JSON has already been loaded into
    /// memory, or in tests.
    ///
    /// # Errors
    ///
    /// Returns an error if the string is not valid JSON or is missing the
    /// required `model.vocab` field.
    pub fn from_json(json: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let tf: TokenizerFile = serde_json::from_str(json)?;

        // Build the vocabulary mapping.
        let vocab = tf.model.vocab;

        // Build the reverse mapping: ID -> token string.
        let id_to_token: HashMap<usize, String> = vocab
            .iter()
            .map(|(token, &id)| (id, token.clone()))
            .collect();

        // Merge rules are already parsed as (token_a, token_b) tuples
        // by the custom deserializer (handles both string and array formats).
        let merges: Vec<(String, String)> = tf
            .model
            .merges
            .unwrap_or_default();

        // Build merge rank lookup for O(1) priority comparison.
        let merge_ranks: HashMap<(String, String), usize> = merges
            .iter()
            .enumerate()
            .map(|(rank, pair)| (pair.clone(), rank))
            .collect();

        // Extract special tokens from the added_tokens list.
        let special_tokens: HashMap<String, usize> = tf
            .added_tokens
            .unwrap_or_default()
            .into_iter()
            .filter(|t| t.special.unwrap_or(false))
            .map(|t| (t.content, t.id))
            .collect();

        // Build the byte-level encoder and decoder.
        let (byte_encoder, byte_decoder) = build_byte_encoder();

        Ok(Tokenizer {
            vocab,
            id_to_token,
            merges,
            merge_ranks,
            special_tokens,
            byte_encoder,
            byte_decoder,
        })
    }

    /// Encode text into a sequence of token IDs.
    ///
    /// This performs the full encoding pipeline:
    /// 1. Pre-tokenize the text into words.
    /// 2. Convert each word to byte-level characters.
    /// 3. Apply BPE merges in priority order.
    /// 4. Look up each resulting token in the vocabulary.
    ///
    /// Any token that is not found in the vocabulary is encoded as ID 0
    /// (typically the unknown/padding token). Special tokens in the input
    /// text are detected and encoded directly by their assigned IDs.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();

        // Check for special tokens in the input text first.
        // If the text is exactly a special token, encode it directly.
        if let Some(&id) = self.special_tokens.get(text) {
            ids.push(id);
            return ids;
        }

        // Pre-tokenize the text into words.
        let words = pretokenize(text, &self.special_tokens);

        for word in words {
            // Check if this word is a special token.
            if let Some(&id) = self.special_tokens.get(&word) {
                ids.push(id);
                continue;
            }

            // Convert the word to byte-level characters.
            let byte_level: Vec<String> = word
                .bytes()
                .map(|b| self.byte_encoder[b as usize].to_string())
                .collect();

            // Apply BPE merges.
            let tokens = self.apply_bpe(&byte_level);

            // Look up each token in the vocabulary.
            for token in tokens {
                if let Some(&id) = self.vocab.get(&token) {
                    ids.push(id);
                } else {
                    // Fallback: encode unknown tokens character by character.
                    for ch in token.chars() {
                        let ch_str = ch.to_string();
                        if let Some(&id) = self.vocab.get(&ch_str) {
                            ids.push(id);
                        }
                        // If even single characters are unknown, skip them.
                        // This should not happen with a well-formed vocabulary.
                    }
                }
            }
        }

        ids
    }

    /// Decode a sequence of token IDs back into text.
    ///
    /// This performs the decoding pipeline:
    /// 1. Look up each ID in the reverse vocabulary.
    /// 2. Concatenate all token strings.
    /// 3. Convert byte-level Unicode characters back to bytes.
    /// 4. Decode the bytes as UTF-8 (replacing invalid sequences with the
    ///    Unicode replacement character).
    pub fn decode(&self, ids: &[usize]) -> String {
        // Look up each ID and concatenate.
        let mut byte_level_chars = String::new();
        for &id in ids {
            if let Some(token) = self.id_to_token.get(&id) {
                byte_level_chars.push_str(token);
            }
            // Skip unknown IDs silently.
        }

        // Convert byte-level characters back to bytes.
        let mut bytes = Vec::new();
        for ch in byte_level_chars.chars() {
            if let Some(&b) = self.byte_decoder.get(&ch) {
                bytes.push(b);
            } else {
                // The character is not in the byte decoder — it might be a
                // special token or multi-byte UTF-8 character stored directly.
                // Push its UTF-8 representation.
                let mut buf = [0u8; 4];
                let s = ch.encode_utf8(&mut buf);
                bytes.extend_from_slice(s.as_bytes());
            }
        }

        // Decode bytes as UTF-8.
        String::from_utf8(bytes).unwrap_or_else(|err| {
            // Replace invalid byte sequences with the Unicode replacement character.
            let bytes = err.into_bytes();
            String::from_utf8_lossy(&bytes).into_owned()
        })
    }

    /// Get the vocabulary size (number of entries in the vocabulary).
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get the EOS (End of Sequence) token ID.
    ///
    /// For Qwen3 models, the EOS token is `<|endoftext|>` with ID 151643.
    /// If that token is not in the vocabulary, this falls back to searching
    /// for common EOS token names, and finally returns 0 if none is found.
    pub fn eos_token_id(&self) -> usize {
        // Qwen3 specific: <|endoftext|> = 151643
        if let Some(&id) = self.vocab.get("<|endoftext|>") {
            return id;
        }
        if let Some(&id) = self.special_tokens.get("<|endoftext|>") {
            return id;
        }
        // Try other common EOS token names.
        for name in &["</s>", "<|end|>", "<eos>", "<|eos|>"] {
            if let Some(&id) = self.vocab.get(*name) {
                return id;
            }
            if let Some(&id) = self.special_tokens.get(*name) {
                return id;
            }
        }
        // Fallback: return 0. This should not happen with a well-formed
        // tokenizer, but prevents a panic.
        0
    }

    /// Apply BPE merges to a sequence of tokens representing a single word.
    ///
    /// The algorithm:
    /// 1. Find the pair of adjacent tokens whose merge has the lowest rank
    ///    (i.e., highest priority) in the merge list.
    /// 2. Merge that pair into a single token.
    /// 3. Repeat until no more merges can be applied.
    ///
    /// This is a faithful implementation of the standard BPE merge algorithm
    /// used by GPT-2, tiktoken, and HuggingFace tokenizers.
    fn apply_bpe(&self, tokens: &[String]) -> Vec<String> {
        if tokens.len() < 2 {
            return tokens.to_vec();
        }

        let mut tokens: Vec<String> = tokens.to_vec();

        loop {
            // Find the pair with the lowest merge rank.
            let mut best_pair: Option<(String, String)> = None;
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

            // If no mergeable pair was found, we're done.
            let best_pair = match best_pair {
                Some(pair) => pair,
                None => break,
            };

            // Merge all occurrences of the best pair.
            let merged = format!("{}{}", best_pair.0, best_pair.1);
            let mut new_tokens = Vec::with_capacity(tokens.len());
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

            // If we've merged down to a single token, no more merges possible.
            if tokens.len() < 2 {
                break;
            }
        }

        tokens
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Byte-level encoding
// ─────────────────────────────────────────────────────────────────────────────

/// Build the byte-to-unicode mapping used by GPT-2 / Qwen byte-level BPE.
///
/// The mapping assigns each of the 256 possible byte values to a unique Unicode
/// character. The goal is to make every byte "visible" so that BPE can operate
/// on a clean character sequence without control characters or whitespace
/// causing issues.
///
/// # How it works
///
/// Bytes that already correspond to visible, non-space ASCII characters or
/// Latin-1 supplement characters map to themselves. Specifically:
/// - Bytes 33-126 (printable ASCII, excluding space): `!` through `~`
/// - Bytes 161-172 (Latin-1 supplement): `¡` through `¬`
/// - Bytes 174-255 (Latin-1 supplement): `®` through `ÿ`
///
/// These 188 byte values are directly representable as visible Unicode
/// characters.
///
/// The remaining 68 byte values (0-32, 127-160, 173) — which include control
/// characters, the space character, and soft hyphen — are mapped to Unicode
/// code points starting at U+0100 (Ā). So byte 0 maps to Ā, byte 1 maps to ā,
/// and so on.
///
/// # Returns
///
/// A tuple of `(encoder, decoder)`:
/// - `encoder`: an array of 256 chars where `encoder[byte_value]` gives the
///   Unicode character for that byte.
/// - `decoder`: a HashMap from Unicode character back to byte value.
fn build_byte_encoder() -> ([char; 256], HashMap<char, u8>) {
    let mut encoder = ['\0'; 256];
    let mut decoder = HashMap::new();

    // Byte values that map directly to their Unicode code points.
    // These are the "visible" bytes that don't need remapping.
    let direct_bytes: Vec<u8> = (33..=126) // printable ASCII (excl. space)
        .chain(161..=172) // Latin-1 supplement (¡ through ¬)
        .chain(174..=255) // Latin-1 supplement (® through ÿ)
        .collect();

    // Map direct bytes to their corresponding Unicode characters.
    for &b in &direct_bytes {
        let c = char::from_u32(b as u32).unwrap();
        encoder[b as usize] = c;
        decoder.insert(c, b);
    }

    // Map remaining bytes to Unicode characters starting at U+0100.
    // These are the "invisible" bytes: control chars, space, delete, etc.
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

// ─────────────────────────────────────────────────────────────────────────────
// Pre-tokenization
// ─────────────────────────────────────────────────────────────────────────────

/// Split input text into "words" for BPE processing.
///
/// Pre-tokenization ensures that BPE merges never cross word boundaries.
/// For example, in "Hello world", the merges for "Hello" and " world" are
/// computed independently.
///
/// This simplified pre-tokenizer splits text into chunks that are:
/// - Runs of alphabetic characters (with optional leading space), which
///   capture whole words like "Hello" or " world"
/// - Runs of digits (with optional leading space)
/// - Individual punctuation or special characters (with optional leading space)
/// - Whitespace sequences that end the line
///
/// The key rule: a space before a word/digit/punctuation is attached to that
/// token as a leading space. In byte-level BPE, this leading space becomes the
/// `Ġ` character (the byte-level encoding of byte 0x20 = space).
///
/// This is a simplification of the full GPT-2 regex pattern. The full pattern
/// would require the `regex` crate and is:
/// ```text
/// '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
/// ```
///
/// Our simpler version produces slightly different splits for some edge cases
/// (particularly apostrophe contractions and multi-character punctuation
/// sequences), but works correctly for the vast majority of text.
fn pretokenize(text: &str, special_tokens: &HashMap<String, usize>) -> Vec<String> {
    let mut words = Vec::new();
    let mut chars = text.chars().peekable();

    while let Some(&ch) = chars.peek() {
        // Check if we're at a special token.
        let remaining: String = chars.clone().collect();
        let mut found_special = false;
        for token_str in special_tokens.keys() {
            if remaining.starts_with(token_str) {
                words.push(token_str.clone());
                for _ in token_str.chars() {
                    chars.next();
                }
                found_special = true;
                break;
            }
        }
        if found_special {
            continue;
        }

        if ch.is_whitespace() {
            // Check if this whitespace is a leading space before a word,
            // digit, or punctuation. If so, attach it to the next token.
            if ch == ' ' {
                // Peek at the next non-space character.
                let mut lookahead = chars.clone();
                lookahead.next(); // skip the space
                if let Some(&next_ch) = lookahead.peek() {
                    if next_ch.is_alphabetic() {
                        // Leading space + letters.
                        chars.next(); // consume the space
                        let mut word = String::from(" ");
                        word.push_str(&consume_while(&mut chars, |c| c.is_alphabetic()));
                        words.push(word);
                        continue;
                    } else if next_ch.is_numeric() {
                        // Leading space + digits.
                        chars.next(); // consume the space
                        let mut word = String::from(" ");
                        word.push_str(&consume_while(&mut chars, |c| c.is_numeric()));
                        words.push(word);
                        continue;
                    } else if !next_ch.is_whitespace() {
                        // Leading space + punctuation/symbol.
                        chars.next(); // consume the space
                        let mut word = String::from(" ");
                        word.push(chars.next().unwrap());
                        words.push(word);
                        continue;
                    }
                }
            }
            // Standalone whitespace (not a leading space).
            // Collect consecutive whitespace.
            let ws: String = consume_while(&mut chars, |c| c.is_whitespace());
            words.push(ws);
        } else if ch.is_alphabetic() {
            // Word without leading space.
            let word: String = consume_while(&mut chars, |c| c.is_alphabetic());
            words.push(word);
        } else if ch.is_numeric() {
            // Number without leading space.
            let num: String = consume_while(&mut chars, |c| c.is_numeric());
            words.push(num);
        } else {
            // Individual punctuation or symbol.
            chars.next();
            words.push(ch.to_string());
        }
    }

    words
}

/// Consume characters from the iterator while `predicate` returns true,
/// collecting them into a String.
fn consume_while<I, P>(iter: &mut std::iter::Peekable<I>, predicate: P) -> String
where
    I: Iterator<Item = char>,
    P: Fn(char) -> bool,
{
    let mut result = String::new();
    while let Some(&ch) = iter.peek() {
        if predicate(ch) {
            result.push(ch);
            iter.next();
        } else {
            break;
        }
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal tokenizer.json for testing.
    ///
    /// This creates a small vocabulary with enough tokens and merges to
    /// demonstrate BPE encoding and decoding of simple English text.
    fn make_test_tokenizer_json() -> String {
        r#"{
            "version": "1.0",
            "added_tokens": [
                {"id": 0, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "model": {
                "type": "BPE",
                "vocab": {
                    "<|endoftext|>": 0,
                    "!": 1,
                    "a": 2,
                    "b": 3,
                    "c": 4,
                    "d": 5,
                    "e": 6,
                    "h": 7,
                    "l": 8,
                    "o": 9,
                    "r": 10,
                    "w": 11,
                    "Ġ": 12,
                    "Ġw": 13,
                    "he": 14,
                    "lo": 15,
                    "or": 16,
                    "hel": 17,
                    "hello": 18,
                    "Ġwor": 19,
                    "ld": 20,
                    "Ġworld": 21,
                    "ll": 22,
                    "Ġworl": 23,
                    "Ġworld!": 24
                },
                "merges": [
                    "Ġ w",
                    "h e",
                    "l o",
                    "he l",
                    "hel lo",
                    "o r",
                    "Ġw or",
                    "l d",
                    "Ġwor ld",
                    "l l",
                    "Ġworl d"
                ]
            }
        }"#.to_string()
    }

    #[test]
    fn test_byte_encoder_decoder_roundtrip() {
        let (encoder, decoder) = build_byte_encoder();

        // Every byte value should have a mapping.
        for b in 0u8..=255 {
            let c = encoder[b as usize];
            assert_ne!(c, '\0', "byte {} should have a non-null mapping", b);

            // Roundtrip: byte -> char -> byte.
            let roundtrip = decoder.get(&c).copied();
            assert_eq!(
                roundtrip,
                Some(b),
                "roundtrip failed for byte {}: char {:?} decoded to {:?}",
                b,
                c,
                roundtrip
            );
        }

        // All mapped characters should be unique.
        let chars: Vec<char> = (0..=255).map(|b| encoder[b as usize]).collect();
        let unique: std::collections::HashSet<char> = chars.iter().copied().collect();
        assert_eq!(
            chars.len(),
            unique.len(),
            "byte encoder should map to 256 unique characters"
        );
    }

    #[test]
    fn test_byte_encoder_specific_mappings() {
        let (encoder, _) = build_byte_encoder();

        // Printable ASCII maps to itself.
        assert_eq!(encoder[b'!' as usize], '!');
        assert_eq!(encoder[b'A' as usize], 'A');
        assert_eq!(encoder[b'z' as usize], 'z');
        assert_eq!(encoder[b'~' as usize], '~');

        // Space (0x20) is NOT in the direct-mapping range, so it maps to a
        // character >= U+0100. The specific character depends on how many
        // non-direct bytes come before it.
        assert_ne!(
            encoder[b' ' as usize],
            ' ',
            "space should not map to itself"
        );

        // The byte-level character for space is Ġ (U+0120) in GPT-2 encoding.
        // This is a widely-used convention in the tokenizer world.
        assert_eq!(encoder[b' ' as usize], 'Ġ', "space should map to Ġ");
    }

    #[test]
    fn test_vocab_loading() {
        let json = make_test_tokenizer_json();
        let tokenizer = Tokenizer::from_json(&json).unwrap();

        assert_eq!(tokenizer.vocab_size(), 25);
        assert_eq!(tokenizer.vocab.get("!"), Some(&1));
        assert_eq!(tokenizer.vocab.get("hello"), Some(&18));
        assert_eq!(tokenizer.vocab.get("Ġworld"), Some(&21));
    }

    #[test]
    fn test_special_tokens() {
        let json = make_test_tokenizer_json();
        let tokenizer = Tokenizer::from_json(&json).unwrap();

        assert_eq!(
            tokenizer.special_tokens.get("<|endoftext|>"),
            Some(&0)
        );
        assert_eq!(tokenizer.eos_token_id(), 0);
    }

    #[test]
    fn test_eos_token_id_qwen35() {
        // Simulate the Qwen3 tokenizer with its actual EOS token ID.
        let json = r#"{
            "added_tokens": [
                {"id": 151643, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "model": {
                "type": "BPE",
                "vocab": {
                    "<|endoftext|>": 151643,
                    "a": 1
                },
                "merges": []
            }
        }"#;
        let tokenizer = Tokenizer::from_json(json).unwrap();
        assert_eq!(tokenizer.eos_token_id(), 151643);
    }

    #[test]
    fn test_encode_hello() {
        let json = make_test_tokenizer_json();
        let tokenizer = Tokenizer::from_json(&json).unwrap();

        // "hello" should encode through BPE merges:
        // h, e, l, l, o
        //   Merge h+e (rank 1): he, l, l, o
        //   Merge l+o (rank 2): he, l, lo
        //   Merge he+l (rank 3): hel, lo
        //   Merge hel+lo (rank 4): hello
        let ids = tokenizer.encode("hello");
        assert!(!ids.is_empty(), "encode should produce some token IDs");
        assert_eq!(
            ids.len(),
            1,
            "fully merged 'hello' should be a single token"
        );
        assert_eq!(ids[0], 18, "hello should have ID 18");
    }

    #[test]
    fn test_encode_hello_world() {
        let json = make_test_tokenizer_json();
        let tokenizer = Tokenizer::from_json(&json).unwrap();

        // "Hello world" -> pre-tokenize into ["Hello", " world"]
        // Note: "Hello" starts with uppercase H, but our vocab only has
        // lowercase. So "Hello" would be split into individual chars that
        // are in vocab. However, "H" is NOT in our test vocab, so let's
        // test with lowercase instead.
        let ids = tokenizer.encode("hello world");

        // "hello" -> token 18
        // " world" -> Ġ, w, o, r, l, d
        //   Merge Ġ+w (rank 0): Ġw, o, r, l, d
        //   Merge o+r (rank 5): Ġw, or, l, d
        //   Merge Ġw+or (rank 6): Ġwor, l, d
        //   Merge l+d (rank 7): Ġwor, ld
        //   Merge Ġwor+ld (rank 8): Ġworld
        assert!(!ids.is_empty(), "encode should produce token IDs");

        // The result should decode back correctly regardless of exact IDs.
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, "hello world", "roundtrip should recover original text");
    }

    #[test]
    fn test_decode_roundtrip() {
        let json = make_test_tokenizer_json();
        let tokenizer = Tokenizer::from_json(&json).unwrap();

        let texts = vec!["hello", "hello world", "a", "abc"];
        for text in texts {
            let ids = tokenizer.encode(text);
            let decoded = tokenizer.decode(&ids);
            assert_eq!(
                decoded, text,
                "roundtrip failed for '{:?}': encoded={:?}, decoded='{:?}'",
                text, ids, decoded
            );
        }
    }

    #[test]
    fn test_decode_individual_tokens() {
        let json = make_test_tokenizer_json();
        let tokenizer = Tokenizer::from_json(&json).unwrap();

        // Decode specific token IDs.
        assert_eq!(tokenizer.decode(&[1]), "!"); // "!" = ID 1
        assert_eq!(tokenizer.decode(&[2]), "a"); // "a" = ID 2
        assert_eq!(tokenizer.decode(&[18]), "hello"); // "hello" = ID 18
        assert_eq!(tokenizer.decode(&[21]), " world"); // "Ġworld" = ID 21, Ġ decodes to space
    }

    #[test]
    fn test_pretokenize_simple() {
        let special_tokens = HashMap::new();
        let words = pretokenize("hello world", &special_tokens);
        // "hello" (no leading space), " world" (leading space attached)
        assert_eq!(words, vec!["hello", " world"]);
    }

    #[test]
    fn test_pretokenize_with_punctuation() {
        let special_tokens = HashMap::new();
        let words = pretokenize("hello, world!", &special_tokens);
        // "hello", ",", " world", "!"
        assert_eq!(words, vec!["hello", ",", " world", "!"]);
    }

    #[test]
    fn test_pretokenize_with_numbers() {
        let special_tokens = HashMap::new();
        let words = pretokenize("test 123", &special_tokens);
        assert_eq!(words, vec!["test", " 123"]);
    }

    #[test]
    fn test_pretokenize_with_special_token() {
        let mut special_tokens = HashMap::new();
        special_tokens.insert("<|endoftext|>".to_string(), 0);
        let words = pretokenize("hello<|endoftext|>world", &special_tokens);
        assert_eq!(words, vec!["hello", "<|endoftext|>", "world"]);
    }

    #[test]
    fn test_apply_bpe_basic() {
        let json = make_test_tokenizer_json();
        let tokenizer = Tokenizer::from_json(&json).unwrap();

        // Test BPE on "hello": h, e, l, l, o
        let input = vec![
            "h".to_string(),
            "e".to_string(),
            "l".to_string(),
            "l".to_string(),
            "o".to_string(),
        ];
        let result = tokenizer.apply_bpe(&input);
        assert_eq!(result, vec!["hello"]);
    }

    #[test]
    fn test_apply_bpe_no_merges() {
        let json = make_test_tokenizer_json();
        let tokenizer = Tokenizer::from_json(&json).unwrap();

        // Tokens with no applicable merges should be returned as-is.
        let input = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let result = tokenizer.apply_bpe(&input);
        assert_eq!(result, vec!["x", "y", "z"]);
    }

    #[test]
    fn test_apply_bpe_single_token() {
        let json = make_test_tokenizer_json();
        let tokenizer = Tokenizer::from_json(&json).unwrap();

        // Single token cannot be merged.
        let input = vec!["hello".to_string()];
        let result = tokenizer.apply_bpe(&input);
        assert_eq!(result, vec!["hello"]);
    }

    #[test]
    fn test_encode_special_token() {
        let json = make_test_tokenizer_json();
        let tokenizer = Tokenizer::from_json(&json).unwrap();

        // Encoding the EOS token string directly should produce its ID.
        let ids = tokenizer.encode("<|endoftext|>");
        assert_eq!(ids, vec![0]);
    }

    #[test]
    fn test_from_file() {
        let dir = std::env::temp_dir().join("qwen35_rs_tokenizer_test");
        fs::create_dir_all(&dir).expect("should create temp dir");
        let path = dir.join("tokenizer.json");

        fs::write(&path, make_test_tokenizer_json()).expect("should write test file");

        let tokenizer = Tokenizer::from_file(&path)
            .expect("from_file should succeed with a valid tokenizer.json");
        assert_eq!(tokenizer.vocab_size(), 25);

        // Clean up.
        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir(&dir);
    }

    #[test]
    fn test_merge_ranks_populated() {
        let json = make_test_tokenizer_json();
        let tokenizer = Tokenizer::from_json(&json).unwrap();

        // Verify merge ranks are populated correctly.
        assert_eq!(tokenizer.merge_ranks.get(&("h".to_string(), "e".to_string())), Some(&1));
        assert_eq!(tokenizer.merge_ranks.get(&("l".to_string(), "o".to_string())), Some(&2));
        assert_eq!(tokenizer.merge_ranks.get(&("Ġ".to_string(), "w".to_string())), Some(&0));
    }

    #[test]
    fn test_empty_input() {
        let json = make_test_tokenizer_json();
        let tokenizer = Tokenizer::from_json(&json).unwrap();

        let ids = tokenizer.encode("");
        assert!(ids.is_empty(), "encoding empty string should produce no tokens");

        let decoded = tokenizer.decode(&[]);
        assert_eq!(decoded, "", "decoding empty IDs should produce empty string");
    }
}
