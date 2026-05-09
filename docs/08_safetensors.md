# 08 — Safetensors: How Model Weights Live on Disk

A trained language model is, at its core, a collection of large matrices and
vectors — billions of floating-point numbers that encode everything the model
learned during training. Before we can run inference, we need to load these
numbers from disk into memory. The format we use to store them matters more
than you might think.

This chapter explains the safetensors format: why it exists, how it works at
the byte level, and how our Rust code reads it.

---

## 1. How Are Model Weights Stored?

### The Scale of the Problem

Qwen3-0.6B has roughly 596 million parameters. At 4 bytes per `f32`
parameter, that is about 2.3 GB of raw numerical data. Larger models are even
more imposing: a 7B model is ~28 GB, and a 70B model is ~280 GB.

These numbers must be saved to disk after training and loaded back before
inference. The file format determines:

- **How fast** we can load the model (seconds vs. minutes).
- **How much memory** we need (can we avoid loading everything at once?).
- **Whether it is safe** to open a stranger's model file.

### Traditional Formats and Their Problems

**PyTorch `.pt` / `.pth` files** use Python's `pickle` module for
serialization. Pickle can serialize arbitrary Python objects — which means
it can also deserialize and **execute arbitrary code** when loading a file.
Opening a malicious `.pt` file can give an attacker full control over your
machine. This is not a theoretical risk; proof-of-concept exploits exist.

Additionally, `.pt` files are slow to load because pickle must reconstruct
the entire Python object graph, including all the framework metadata that
PyTorch stores alongside the raw tensors.

**NumPy `.npz` files** are safer (no code execution) but are designed for
general numerical data, not specifically for ML model weights. They use ZIP
compression internally, which adds overhead. They also lack a standard way
to attach metadata like tensor names and shapes in a way that ML frameworks
can agree on.

**HDF5** is efficient and supports memory mapping, but it is a complex
format with a large C library dependency. The specification is over 150
pages long. For the specific task of "store a dictionary of named tensors,"
this is overkill.

What the ML community needed was a format that was:

1. **Safe**: No code execution, ever.
2. **Fast**: Minimal overhead to load.
3. **Simple**: Easy to implement in any language.
4. **Memory-mappable**: Optionally load tensors on demand without reading
   the whole file.

---

## 2. What Is Safetensors?

[Safetensors](https://github.com/huggingface/safetensors) is a file format
designed by HuggingFace specifically for storing ML model weights. It
satisfies all four requirements above.

### Key Properties

**Safe.** The format contains only a JSON header (metadata) and raw binary
data (tensor values). There is no scripting language, no bytecode, no
serialization framework, and no way to embed executable code. Loading a
safetensors file never executes anything — it just reads bytes.

**Fast.** The JSON header is small (typically a few KB) and parsed once.
The tensor data is stored as contiguous raw bytes, exactly matching the
in-memory layout. Loading is essentially a file read plus a few pointer
adjustments — no deserialization, no object reconstruction, no
decompression.

**Simple.** The entire specification fits in a few paragraphs (see Section
3). A basic reader can be implemented in under 100 lines of code in any
language. There is no complex framing, no compression scheme, no optional
features to negotiate.

**Memory-mappable.** Because the tensor data is stored as contiguous raw
bytes at known offsets, you can `mmap` the file and access individual
tensors on demand without loading the entire file into RAM. This is
critical for large models: a 70B model in F16 is ~140 GB, and you may not
have that much RAM. With mmap, only the pages you actually access are
loaded into physical memory.

### Industry Adoption

Safetensors is now the **default format** on the HuggingFace Model Hub.
When you download a model like `Qwen/Qwen3-0.6B`, the weight files you
get are `.safetensors` files. The PyTorch `.bin` format is being phased out
for security and performance reasons.

---

## 3. File Format Specification

### Binary Layout

A safetensors file consists of three consecutive regions:

```
┌─────────────────────────────────┐
│ 8 bytes: header_size (u64 LE)   │  ← length of the JSON header
├─────────────────────────────────┤
│ header_size bytes: JSON header  │  ← tensor metadata (padded to 8-byte alignment)
├─────────────────────────────────┤
│ remaining bytes: tensor data    │  ← raw tensor values, concatenated
└─────────────────────────────────┘
```

**Region 1 — Header size (8 bytes).** A single `u64` in little-endian byte
order. This tells us how many bytes to read for the JSON header. The total
of `8 + header_size` is always a multiple of 8 (the header is padded with
spaces if necessary), ensuring the data section starts at an 8-byte-aligned
offset.

**Region 2 — JSON header.** A UTF-8 JSON object that maps tensor names to
their metadata. This is the "table of contents" — it tells us where each
tensor's data lives in the data section and how to interpret it.

**Region 3 — Data section.** Raw bytes containing the actual tensor values,
concatenated back-to-back. The order and offsets are determined by the
header. Each tensor's data is stored contiguously in row-major (C) order
using little-endian byte ordering.

### The JSON Header Structure

Each tensor entry in the header has three fields:

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

- **`dtype`** (string): The data type of each element. Common values:
  - `"F32"` — 32-bit IEEE 754 floating point (4 bytes per element)
  - `"F16"` — 16-bit IEEE 754 floating point (2 bytes per element)
  - `"BF16"` — BFloat16 (2 bytes per element, wider exponent range than F16)
  - Others: `"F64"`, `"I8"`, `"I32"`, `"I64"`, `"U8"`, etc.

- **`shape`** (array of integers): The dimensions of the tensor. This is
  exactly what you would pass to `Tensor::new`. For a weight matrix in a
  linear layer, this is `[out_features, in_features]`.

- **`data_offsets`** (array of two integers): Byte offsets `[start, end]`
  into the data section. The tensor's raw data occupies bytes `start` through
  `end` (exclusive) of the data section. Note: these offsets are relative to
  the start of the data section, not the start of the file.

The header may also contain a special `"__metadata__"` key with arbitrary
key-value pairs (e.g., `{"format": "pt"}`). This is not a tensor — readers
must skip it.

### A Real Example: Qwen3-0.6B Tensor Names and Shapes

The Qwen3-0.6B model has the following weight tensors (shapes shown for
the F32 version):

```
model.embed_tokens.weight              [151936, 1024]    ← token embedding
model.layers.0.self_attn.q_proj.weight [2048, 1024]      ← query projection (16 heads * 128)
model.layers.0.self_attn.k_proj.weight [1024, 1024]      ← key projection (GQA: 8 heads * 128)
model.layers.0.self_attn.v_proj.weight [1024, 1024]      ← value projection (GQA: 8 heads * 128)
model.layers.0.self_attn.o_proj.weight [1024, 2048]      ← output projection
model.layers.0.mlp.gate_proj.weight    [3072, 1024]      ← SwiGLU gate
model.layers.0.mlp.up_proj.weight      [3072, 1024]      ← SwiGLU up
model.layers.0.mlp.down_proj.weight    [1024, 3072]      ← SwiGLU down
model.layers.0.input_layernorm.weight  [1024]            ← RMSNorm before attention
model.layers.0.post_attention_layernorm.weight [1024]    ← RMSNorm before FFN
...
model.norm.weight                      [1024]            ← final RMSNorm
lm_head.weight                         [151936, 1024]    ← output projection (tied with embedding)
```

Each of the 28 transformer layers has the same set of 10 weight tensors,
for a total of 28 × 10 + 3 = 283 tensors. The embedding matrix and
`lm_head` share the same weights (`tie_word_embeddings = true`),
so the combined size is 151936 × 1024 × 4 = ~590 MB in F32.

### How data_offsets Work

The `data_offsets` field specifies where each tensor's bytes live within the
data section. Tensors are typically laid out contiguously, so the end offset
of one tensor equals the start offset of the next:

```
Data section:
┌──────────────────┬──────────────────┬──────────────────┬─────
│ embed_tokens      │ q_proj (layer 0)  │ k_proj (layer 0)  │ ...
│ [0 .. 155705344)  │ [155705344 ..      │ [156754304 ..      │
│                   │  156754304)       │  157287168)       │
└──────────────────┴──────────────────┴──────────────────┴─────
```

To read the query projection weight for layer 0:
1. Look up `"model.layers.0.self_attn.q_proj.weight"` in the header.
2. Get `data_offsets = [155705344, 156754304]`.
3. Seek to byte `8 + header_size + 155705344` in the file.
4. Read `156754304 - 155705344 = 1048960` bytes (which equals
   1024 × 1024 × 4 for a `[1024, 1024]` F32 tensor).
5. Interpret those bytes as 1,048,576 little-endian `f32` values.

The offsets are byte-level, not element-level, which makes the format
dtype-agnostic — the same offset scheme works for F32, F16, BF16, or any
other type.

---

## 4. Supported Data Types

### F32 (float32) — What We Use

F32 is the IEEE 754 single-precision floating-point format: 1 sign bit, 8
exponent bits, and 23 mantissa bits. Each value occupies 4 bytes. The
representable range is roughly ±3.4 × 10^38 with about 7 decimal digits of
precision.

For our educational project, F32 is the natural choice:
- It matches Rust's `f32` type exactly — no conversion needed.
- It provides enough precision for inference (training often requires
  higher precision for gradient accumulation, but inference does not).
- It is the simplest format to read: just interpret 4 bytes as a
  little-endian `f32`.

### F16 (float16) — Half Precision

F16 is the IEEE 754 half-precision format: 1 sign bit, 5 exponent bits, and
10 mantissa bits. Each value is only 2 bytes. The representable range is
roughly ±65,504 with about 3 decimal digits of precision.

F16 is popular for inference on GPUs because it halves memory usage and
bandwidth compared to F32. However, the limited range causes numerical
issues (overflow to infinity, underflow to zero) that require careful
scaling. Modern GPUs have native F16 arithmetic units, making it very fast.

### BF16 (bfloat16) — Brain Float

BF16 is Google's alternative to F16: 1 sign bit, 8 exponent bits, and 7
mantissa bits. It has the same exponent range as F32 (so no overflow
issues) but only about 2-3 decimal digits of precision. Like F16, each
value is 2 bytes.

BF16 was designed specifically for deep learning. The insight is that neural
networks are far more sensitive to exponent range (to avoid overflow) than
to mantissa precision (the extra bits of F16 precision are often wasted).
Modern training frameworks (PyTorch, JAX) use BF16 as the default
mixed-precision format, and many models are distributed in BF16.

### Why We Only Support F32

Our implementation rejects F16 and BF16 tensors with a clear error message:

```
Unsupported dtype BF16 for tensor model.embed_tokens.weight -- only F32 is supported
```

This is a deliberate choice for an educational project:
- Supporting F16/BF16 would require a conversion step (dequantize to F32
  after reading), adding complexity without teaching new concepts.
- The Qwen3-0.6B model is available in F32 on HuggingFace, so there is
  a working weight file for our code.
- For a production inference engine, F16/BF16 support is essential to halve
  memory usage. This would be a natural extension exercise.

### Size Comparison

| Dtype | Bytes per element | Qwen3-0.6B total | Typical use |
|-------|-------------------|---------------------|-------------|
| F32   | 4                 | ~2.3 GB             | Training, educational inference |
| F16   | 2                 | ~1.7 GB             | GPU inference |
| BF16  | 2                 | ~1.7 GB             | Training & GPU inference |
| Q8    | 1                 | ~0.85 GB            | Quantized CPU inference |

---

## 5. Reading Safetensors in Our Code

Let us walk through the parsing steps in our `read_safetensors` function,
explaining what each step does and why.

### Step 1: Read the 8-byte header size

```rust
let header_size = file.read_u64::<LittleEndian>()? as usize;
```

We use the `byteorder` crate's `ReadU64` trait to read a `u64` in
little-endian byte order. This gives us the number of bytes in the JSON
header. The `as usize` cast is safe because header sizes are typically a
few kilobytes — far below the `usize` limit on any platform.

### Step 2: Read and parse the JSON header

```rust
let mut header_bytes = vec![0u8; header_size];
file.read_exact(&mut header_bytes)?;
let header_str = std::str::from_utf8(&header_bytes)?;
let header_map: HashMap<String, serde_json::Value> =
    serde_json::from_str(header_str)?;
```

We read exactly `header_size` bytes, validate that they are valid UTF-8,
and parse the JSON into a `HashMap<String, Value>`. We use `serde_json::Value`
for the top level because the header contains heterogeneous entries: tensor
metadata objects and the special `__metadata__` object.

### Step 3: Compute the data section start

```rust
let data_offset_start = 8 + header_size;
```

The data section begins right after the 8-byte size prefix and the header
bytes. Because `header_size` includes any padding (the safetensors spec
requires 8-byte alignment of `8 + header_size`), the data section is
always aligned.

### Step 4: Iterate over tensor entries

For each entry in the header, we skip `__metadata__` and deserialize the
value into our `TensorHeader` struct:

```rust
if name == "__metadata__" {
    continue;
}
let tensor_header: TensorHeader = serde_json::from_value(value.clone())?;
```

Using a typed struct (`TensorHeader`) gives us automatic field extraction
and type checking — if a tensor entry is missing `dtype` or `shape`, serde
returns a clear error.

### Step 5: Validate the dtype

```rust
if tensor_header.dtype != "F32" {
    return Err(format!(
        "Unsupported dtype {} for tensor {} -- only F32 is supported",
        tensor_header.dtype, name
    ).into());
}
```

We check the dtype before doing any I/O. If the tensor uses an unsupported
format, we fail fast with a clear message that includes both the dtype and
the tensor name, so the user knows exactly which tensor caused the problem.

### Step 6: Verify data size consistency

```rust
let num_elements: usize = tensor_header.shape.iter().product();
let expected_bytes = num_elements * 4;
let [start, end] = tensor_header.data_offsets;
let actual_bytes = end - start;
if actual_bytes != expected_bytes {
    return Err(...);
}
```

We compute the expected byte count from the shape (product of dimensions ×
4 bytes per F32 element) and verify it matches the `data_offsets` range.
This catches malformed files where the header is inconsistent — a useful
sanity check that produces a better error message than a buffer overrun.

### Step 7: Read the raw bytes

```rust
file.seek(SeekFrom::Start((data_offset_start + start) as u64))?;
let mut raw_bytes = vec![0u8; actual_bytes];
file.read_exact(&mut raw_bytes)?;
```

We seek to the correct position in the file (data section start + the
tensor's byte offset within the data section) and read exactly the number
of bytes needed. Using `read_exact` ensures we get all the data or a clear
error — no short reads.

### Step 8: Convert bytes to Vec<f32>

```rust
let mut data = Vec::with_capacity(num_elements);
let mut cursor = std::io::Cursor::new(&raw_bytes);
for _ in 0..num_elements {
    data.push(cursor.read_f32::<LittleEndian>()?);
}
```

This is where the `byteorder` crate earns its keep. Each F32 value in a
safetensors file is stored in little-endian byte order (the standard for
modern x86 and ARM processors). The `read_f32::<LittleEndian>()` method
reads 4 bytes and interprets them as a little-endian IEEE 754 float,
returning a Rust `f32`.

Why use byteorder instead of `f32::from_le_bytes`? Both work, but
byteorder provides a convenient `Read`-based interface that handles the
byte-by-byte iteration for us. It is also consistent with how we read the
header size (step 1). An alternative approach using `from_le_bytes` would
look like:

```rust
// Alternative: using from_le_bytes (chunk-based)
let data: Vec<f32> = raw_bytes
    .chunks_exact(4)
    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
    .collect();
```

Both are correct; the byteorder approach is slightly more idiomatic when
you are already using the crate for other reads.

### Step 9: Create the entry and return

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

We store each tensor as a `SafeTensorEntry` — a simple struct with the
tensor's name, shape, dtype string, and f32 data. The `read_safetensors_as_tensors`
convenience function then converts each entry into our `Tensor` type by
calling `Tensor::new(entry.shape, entry.data)`.

---

## 6. Memory Considerations

### How Much Data Are We Loading?

Qwen3-0.6B at F32 has approximately 596 million parameters. Each F32
parameter is 4 bytes, so the total tensor data is:

```
596M × 4 bytes ≈ 2.3 GB
```

The JSON header for this model is tiny by comparison — a few tens of
kilobytes. So the file is almost entirely raw tensor data.

### Loading All at Once

Our `read_safetensors` function reads every tensor into a `Vec<f32>` and
stores them all in a `HashMap`. For Qwen3-0.6B, this means:

- ~2.3 GB for the `Vec<f32>` buffers (the tensor data itself).
- Some overhead for the `HashMap` structure and `String` keys (negligible).
- During reading, we also allocate temporary `raw_bytes` buffers (freed
  after conversion to `Vec<f32>`).

The peak memory usage is roughly 2× the data size during reading (one copy
in `raw_bytes`, one in the final `Vec<f32>`), settling to ~2.3 GB after
reading is complete. This is fine for a model that fits in RAM.

### The Tensor Lifecycle

After loading, the `HashMap<String, Tensor>` is typically consumed by the
model construction code, which extracts each tensor by name and assigns it
to the corresponding model component. The HashMap is then dropped, and only
the model's fields hold references to the data. The peak memory during model
construction is about the same as the loaded data — the tensors are moved,
not copied.

### Memory Mapping for Larger Models

For models that are too large to fit in RAM (or when you want to avoid the
upfront loading cost), the safetensors format supports **memory mapping**
(mmap). The idea is simple:

1. Call `mmap` on the file to map it into the process's virtual address
   space. This does not actually read the data — it just sets up page
   table entries.
2. When the code accesses a tensor's data, the operating system loads the
   corresponding pages from disk on demand (a "page fault").
3. Pages that have not been accessed recently can be evicted from physical
   RAM by the OS, automatically freeing memory.

With mmap, you can "load" a 70 GB model on a machine with only 16 GB of
RAM — the OS will keep the most recently used tensors in memory and page
the rest out to disk. This is slower than having everything in RAM, but it
makes large models accessible on modest hardware.

Our implementation does not use mmap (it would add significant complexity
for an educational project), but the safetensors format was designed to
support it. The key design choices that enable mmap are:

- **Contiguous data section**: All tensor data is in one contiguous region
  of the file, making it easy to map the entire data section at once.
- **Known offsets**: The header tells you exactly where each tensor's data
  lives, so you can construct a "view" into the mmaped region without
  copying.
- **No compression**: The bytes on disk are exactly the bytes in memory,
  so the mmaped data can be used directly without decompression.

### A Rough Memory Budget for Qwen3-0.6B Inference

| Component | Memory (F32) |
|-----------|-------------|
| Model weights | ~2.3 GB |
| KV cache (28 layers, seq_len=2048) | ~224 MB |
| Activations (per forward pass) | ~10 MB |
| Tokenizer, config, misc. | ~10 MB |
| **Total** | **~3.7 GB** |

This fits comfortably on any modern laptop with 8+ GB of RAM. For a 7B
model in F32, the weights alone would be ~28 GB, which requires a machine
with 32+ GB of RAM or the use of F16/quantized weights.

---

## Summary

| Concept | Key Point |
|---------|-----------|
| Why safetensors? | Safe (no code execution), fast (raw bytes), simple (JSON + binary) |
| File layout | 8-byte header size, then JSON header, then raw tensor data |
| JSON header | Maps tensor names to `{dtype, shape, data_offsets}` |
| data_offsets | Byte ranges `[start, end]` into the data section |
| Our dtype support | F32 only; F16/BF16 produce clear error messages |
| byteorder crate | Reads multi-byte values (u64, f32) in specified endianness |
| Memory for 0.8B | ~2.3 GB in F32 — fits in RAM, no mmap needed |
| Memory mapping | Enables loading models larger than RAM by paging data on demand |

The safetensors format is a beautiful example of good design: it solves a
real problem (safe, fast model loading) with the minimum possible
complexity. A reader needs only three operations — read 8 bytes, parse
JSON, read byte ranges — and can be implemented in an afternoon. Yet this
simple format now powers the distribution of hundreds of thousands of
models on the HuggingFace Hub.
