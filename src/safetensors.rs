//! Safetensors file format reader.
//!
//! The [safetensors format](https://github.com/huggingface/safetensors) is a
//! binary file format designed by HuggingFace for storing machine learning model
//! weights. It was created to address the security and performance problems of
//! traditional serialization formats like PyTorch's `.pt` files (which use
//! Python's `pickle` and can execute arbitrary code on load).
//!
//! # File layout
//!
//! A safetensors file consists of three consecutive regions:
//!
//! ```text
//! ┌──────────────────────────────┐
//! │ 8 bytes: header_size (u64 LE)│  ← length of the JSON header in bytes
//! ├──────────────────────────────┤
//! │ header_size bytes: JSON      │  ← tensor names → metadata mapping
//! ├──────────────────────────────┤
//! │ remaining bytes: data        │  ← raw tensor data, concatenated
//! └──────────────────────────────┘
//! ```
//!
//! The JSON header maps each tensor name to a metadata object:
//!
//! ```json
//! {
//!   "model.embed_tokens.weight": {
//!     "dtype": "F32",
//!     "shape": [151936, 1024],
//!     "data_offsets": [0, 155705344]
//!   },
//!   "__metadata__": { "format": "pt" }
//! }
//! ```
//!
//! - `dtype` is a string like `"F32"`, `"F16"`, or `"BF16"`.
//! - `shape` is an array of dimension sizes.
//! - `data_offsets` is `[start, end]` — byte offsets into the data section
//!   (the region *after* the header).
//!
//! # Supported dtypes
//!
//! `F32`, `BF16`, and `F16` tensors are supported. BF16 and F16 data is
//! automatically converted to `f32` on load. This is necessary because real
//! model weights (e.g. Qwen3-0.6B) are stored in BF16 format.

use crate::tensor::Tensor;
use byteorder::{LittleEndian, ReadBytesExt};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read as _, Seek, SeekFrom};
use std::path::Path;

// ─────────────────────────────────────────────────────────────────────────────
// JSON header types
// ─────────────────────────────────────────────────────────────────────────────

/// Deserialized metadata for a single tensor from the JSON header.
///
/// Each tensor entry in the safetensors header has a `dtype`, `shape`, and
/// `data_offsets` field. We deserialize into this struct so we don't have
/// to do untyped JSON value lookups.
#[derive(Debug, Deserialize)]
struct TensorHeader {
    /// Data type string, e.g. `"F32"`, `"F16"`, `"BF16"`.
    dtype: String,
    /// Shape of the tensor as a list of dimension sizes.
    shape: Vec<usize>,
    /// Byte offsets `[start, end]` into the data section of the file.
    data_offsets: [usize; 2],
}

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A single tensor entry read from a safetensors file.
///
/// Contains the tensor's name, shape, dtype string, and the actual data
/// converted to `Vec<f32>`. The dtype is preserved as a string for
/// informational purposes even though the data is always stored as f32.
#[derive(Debug, Clone)]
pub struct SafeTensorEntry {
    /// The name of the tensor (e.g. `"model.layers.0.self_attn.q_proj.weight"`).
    pub name: String,
    /// The shape of the tensor (e.g. `[1024, 1024]` for a square weight matrix).
    pub shape: Vec<usize>,
    /// The dtype string from the header (e.g. `"F32"`).
    pub dtype: String,
    /// The tensor data, always stored as `f32` after conversion from raw bytes.
    pub data: Vec<f32>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Reader functions
// ─────────────────────────────────────────────────────────────────────────────

/// Read all tensors from a safetensors file.
///
/// Opens the file at `path`, parses the binary header, and extracts every
/// tensor into a [`SafeTensorEntry`]. Returns a `HashMap` keyed by tensor
/// name.
///
/// # Errors
///
/// - Returns an error if the file cannot be opened or read.
/// - Returns an error if the 8-byte header size cannot be read.
/// - Returns an error if the JSON header cannot be parsed.
/// - Returns an error if any tensor has an unsupported dtype (not F32, BF16,
///   or F16), with a message like `"Unsupported dtype U8 for tensor X -- only
///   F32, BF16, and F16 are supported"`.
/// - Returns an error if the byte range for a tensor is invalid or extends
///   beyond the file.
///
/// # Example
///
/// ```ignore
/// use qwen3_5_rs::safetensors::read_safetensors;
/// use std::path::Path;
///
/// let tensors = read_safetensors(Path::new("model.safetensors"))?;
/// let embed = &tensors["model.embed_tokens.weight"];
/// println!("embed shape: {:?}", embed.shape);
/// ```
pub fn read_safetensors(
    path: &Path,
) -> Result<HashMap<String, SafeTensorEntry>, Box<dyn std::error::Error>> {
    // Step 1: Open the file for reading.
    let mut file = File::open(path)?;

    // Step 2: Read the 8-byte header size (u64 little-endian).
    // This tells us how many bytes the JSON header occupies.
    let header_size = file.read_u64::<LittleEndian>()? as usize;

    // Step 3: Read the JSON header.
    let mut header_bytes = vec![0u8; header_size];
    file.read_exact(&mut header_bytes)?;
    let header_str = std::str::from_utf8(&header_bytes)?;

    // Step 4: Parse the JSON header into a map of tensor name → metadata.
    // We use serde_json::Value for the top level because the header contains
    // both tensor entries and a special "__metadata__" key that has a
    // different structure.
    let header_map: HashMap<String, serde_json::Value> = serde_json::from_str(header_str)?;

    // Step 5: Determine where the data section starts.
    // It begins after the 8-byte size prefix plus the *padded* header.
    // The safetensors spec requires that the total of (8 + header_size) is
    // aligned to an 8-byte boundary. The padding bytes (if any) come after
    // the JSON content but are included in header_size.
    let data_offset_start = 8 + header_size;

    // Step 6: Iterate over each entry in the header, skipping __metadata__.
    let mut result = HashMap::new();

    for (name, value) in &header_map {
        // Skip the special __metadata__ key — it's not a tensor.
        if name == "__metadata__" {
            continue;
        }

        // Deserialize the value into our TensorHeader struct.
        let tensor_header: TensorHeader = serde_json::from_value(value.clone())?;

        // Determine bytes per element and conversion method based on dtype.
        let (bytes_per_element, dtype_label) = match tensor_header.dtype.as_str() {
            "F32" => (4usize, "F32"),
            "BF16" => (2usize, "BF16"),
            "F16" => (2usize, "F16"),
            _ => {
                return Err(format!(
                    "Unsupported dtype {} for tensor {} -- only F32, BF16, and F16 are supported",
                    tensor_header.dtype, name
                )
                .into());
            }
        };

        // Calculate the expected number of elements from the shape.
        let num_elements: usize = tensor_header.shape.iter().product();

        // The expected byte size for this dtype.
        let expected_bytes = num_elements * bytes_per_element;

        // Verify that the data_offsets span matches the expected byte count.
        let [start, end] = tensor_header.data_offsets;
        let actual_bytes = end - start;
        if actual_bytes != expected_bytes {
            return Err(format!(
                "Data size mismatch for tensor {}: expected {} bytes (shape {:?} with {}), got {} bytes from data_offsets",
                name, expected_bytes, tensor_header.shape, dtype_label, actual_bytes
            )
            .into());
        }

        // Step 7: Seek to the correct position in the data section and read raw bytes.
        // The data_offsets are relative to the start of the data section,
        // so we add the data_offset_start (8 + header_size).
        file.seek(SeekFrom::Start((data_offset_start + start) as u64))?;

        let mut raw_bytes = vec![0u8; actual_bytes];
        file.read_exact(&mut raw_bytes)?;

        // Step 8: Convert raw bytes to Vec<f32>.
        // The safetensors format stores data in little-endian byte order.
        let data = match tensor_header.dtype.as_str() {
            "F32" => {
                let mut data = Vec::with_capacity(num_elements);
                let mut cursor = std::io::Cursor::new(&raw_bytes);
                for _ in 0..num_elements {
                    data.push(cursor.read_f32::<LittleEndian>()?);
                }
                data
            }
            "BF16" => {
                // BF16 has the same 8-bit exponent as F32 but only 8 mantissa
                // bits (vs 23 for F32). To convert BF16 → F32: read 2 bytes as
                // u16, shift left by 16 bits, reinterpret as f32.
                let mut data = Vec::with_capacity(num_elements);
                let mut cursor = std::io::Cursor::new(&raw_bytes);
                for _ in 0..num_elements {
                    let bf16_bits = cursor.read_u16::<LittleEndian>()?;
                    let f32_bits = (bf16_bits as u32) << 16;
                    data.push(f32::from_bits(f32_bits));
                }
                data
            }
            "F16" => {
                // F16 (IEEE 754 half-precision): 1 sign bit, 5 exponent bits
                // (bias 15), 10 mantissa bits.
                let mut data = Vec::with_capacity(num_elements);
                let mut cursor = std::io::Cursor::new(&raw_bytes);
                for _ in 0..num_elements {
                    let f16_bits = cursor.read_u16::<LittleEndian>()?;
                    data.push(f16_to_f32(f16_bits));
                }
                data
            }
            // We already validated the dtype above, so this is unreachable.
            _ => unreachable!(),
        };

        // Step 9: Create the SafeTensorEntry and insert into the result map.
        result.insert(
            name.clone(),
            SafeTensorEntry {
                name: name.clone(),
                shape: tensor_header.shape,
                dtype: tensor_header.dtype,
                data,
            },
        );
    }

    Ok(result)
}

/// Read safetensors and return tensors as our [`Tensor`] type.
///
/// This is a convenience function that calls [`read_safetensors`] and converts
/// each [`SafeTensorEntry`] into a [`Tensor`]. The shape and data are passed
/// directly to [`Tensor::new`].
///
/// # Errors
///
/// Propagates any error from [`read_safetensors`], including unsupported
/// dtypes or file I/O errors.
///
/// # Example
///
/// ```ignore
/// use qwen3_5_rs::safetensors::read_safetensors_as_tensors;
/// use std::path::Path;
///
/// let tensors = read_safetensors_as_tensors(Path::new("model.safetensors"))?;
/// let embed = &tensors["model.embed_tokens.weight"];
/// println!("embed shape: {:?}", embed.shape());
/// ```
pub fn read_safetensors_as_tensors(
    path: &Path,
) -> Result<HashMap<String, Tensor>, Box<dyn std::error::Error>> {
    let entries = read_safetensors(path)?;
    let mut tensors = HashMap::new();

    for (name, entry) in entries {
        tensors.insert(name, Tensor::new(entry.shape, entry.data));
    }

    Ok(tensors)
}

// ─────────────────────────────────────────────────────────────────────────────
// F16 → F32 conversion
// ─────────────────────────────────────────────────────────────────────────────

/// Convert an IEEE 754 half-precision (F16) value to `f32`.
///
/// F16 layout (16 bits):
/// - Bit 15: sign
/// - Bits 14–10: exponent (bias 15)
/// - Bits 9–0: mantissa
///
/// F32 layout (32 bits):
/// - Bit 31: sign
/// - Bits 30–23: exponent (bias 127)
/// - Bits 22–0: mantissa
fn f16_to_f32(half: u16) -> f32 {
    let sign = (half >> 15) & 0x1;
    let exp = (half >> 10) & 0x1F;
    let mant = half & 0x3FF;

    if exp == 0 {
        if mant == 0 {
            // Zero (positive or negative)
            f32::from_bits((sign as u32) << 31)
        } else {
            // Subnormal F16 → normalized F32
            // Adjust the mantissa until the leading bit is 1, decrementing the
            // exponent for each shift. The F16 exponent bias is 15, and the
            // denormalized exponent is 1 - 15 = -14. The F32 exponent bias is
            // 127, so the F32 exponent is -14 + 127 = 113.
            let mut m = mant;
            let mut e = 0u32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            m &= 0x3FF; // Remove the implicit leading 1
            let f32_exp = 113 - e; // 1 - 15 + 127 - e
            f32::from_bits((sign as u32) << 31 | f32_exp << 23 | (m as u32) << 13)
        }
    } else if exp == 31 {
        if mant == 0 {
            // Infinity
            f32::from_bits((sign as u32) << 31 | 0x7F800000)
        } else {
            // NaN — preserve the mantissa bits
            f32::from_bits((sign as u32) << 31 | 0x7F800000 | (mant as u32) << 13)
        }
    } else {
        // Normalized number
        let f32_exp = (exp as u32) + (127 - 15); // re-bias from 15 to 127
        f32::from_bits((sign as u32) << 31 | f32_exp << 23 | (mant as u32) << 13)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Helper: write a safetensors file from a JSON header string and raw data.
    ///
    /// Builds the binary format: [8-byte header_size][header][data].
    /// Returns the path to the temporary file.
    fn write_test_safetensors(header_json: &str, data: &[u8]) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let dir = std::env::temp_dir().join("qwen35_rs_safetensors_test");
        std::fs::create_dir_all(&dir).expect("should create temp dir");
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = dir.join(format!("test_{}_{}.safetensors", std::process::id(), id));

        let header_bytes = header_json.as_bytes();

        // The safetensors spec requires that (8 + header_size) is aligned
        // to 8 bytes. We pad the JSON with spaces to achieve this.
        let json_len = header_bytes.len();
        let aligned_len = ((json_len + 7) / 8) * 8; // round up to next 8-byte boundary
        let padding = aligned_len - json_len;

        let header_size = aligned_len as u64;

        let mut file_data = Vec::new();
        // 8-byte header size (u64 little-endian).
        file_data.extend_from_slice(&header_size.to_le_bytes());
        // The JSON header.
        file_data.extend_from_slice(header_bytes);
        // Padding bytes (spaces) to align to 8-byte boundary.
        file_data.extend_from_slice(&vec![b' '; padding]);
        // The raw tensor data.
        file_data.extend_from_slice(data);

        let mut file = File::create(&path).expect("should create temp file");
        file.write_all(&file_data).expect("should write test file");

        path
    }

    /// Helper: clean up a temporary safetensors file.
    fn cleanup(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_read_single_f32_tensor() {
        // Build a safetensors file with one 2x3 F32 tensor.
        // 6 f32 values = 24 bytes of data.
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut data_bytes = Vec::new();
        for v in &values {
            data_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let header_json = r#"{"test_tensor":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let path = write_test_safetensors(header_json, &data_bytes);

        let result = read_safetensors(&path).expect("reading safetensors should succeed");
        let entry = &result["test_tensor"];

        assert_eq!(entry.shape, vec![2, 3]);
        assert_eq!(entry.dtype, "F32");
        assert_eq!(entry.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        cleanup(&path);
    }

    #[test]
    fn test_read_multiple_tensors() {
        // Build a file with two tensors.
        // Tensor A: shape [2], 2 f32 values = 8 bytes, offsets [0, 8].
        // Tensor B: shape [3], 3 f32 values = 12 bytes, offsets [8, 20].
        let a_values = [10.0f32, 20.0];
        let b_values = [1.0f32, 2.0, 3.0];

        let mut data_bytes = Vec::new();
        for v in &a_values {
            data_bytes.extend_from_slice(&v.to_le_bytes());
        }
        for v in &b_values {
            data_bytes.extend_from_slice(&v.to_le_bytes());
        }

        // Note: JSON keys are sorted in the header for determinism, but
        // safetensors does not require sorted keys. We use explicit ordering.
        let header_json = r#"{"tensor_a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"tensor_b":{"dtype":"F32","shape":[3],"data_offsets":[8,20]}}"#;
        let path = write_test_safetensors(header_json, &data_bytes);

        let result = read_safetensors(&path).expect("reading safetensors should succeed");

        let a = &result["tensor_a"];
        assert_eq!(a.shape, vec![2]);
        assert_eq!(a.data, vec![10.0, 20.0]);

        let b = &result["tensor_b"];
        assert_eq!(b.shape, vec![3]);
        assert_eq!(b.data, vec![1.0, 2.0, 3.0]);

        cleanup(&path);
    }

    #[test]
    fn test_skip_metadata_key() {
        // The __metadata__ key should be silently skipped.
        let values = [42.0f32];
        let mut data_bytes = Vec::new();
        for v in &values {
            data_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let header_json = r#"{"__metadata__":{"format":"pt"},"my_tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let path = write_test_safetensors(header_json, &data_bytes);

        let result = read_safetensors(&path).expect("reading safetensors should succeed");

        // Only the real tensor should appear, not __metadata__.
        assert_eq!(result.len(), 1);
        assert!(result.contains_key("my_tensor"));
        assert!(!result.contains_key("__metadata__"));

        let entry = &result["my_tensor"];
        assert_eq!(entry.data, vec![42.0]);

        cleanup(&path);
    }

    #[test]
    fn test_read_f16_tensor() {
        // F16 tensor with known values.
        // F16 representation of 1.0: sign=0, exp=15 (0x0F), mant=0
        // bits = 0_01111_0000000000 = 0x3C00
        // F16 representation of 2.0: sign=0, exp=16 (0x10), mant=0
        // bits = 0_10000_0000000000 = 0x4000
        let f16_1_0 = 0x3C00u16; // 1.0 in F16
        let f16_2_0 = 0x4000u16; // 2.0 in F16

        let mut data_bytes = Vec::new();
        data_bytes.extend_from_slice(&f16_1_0.to_le_bytes());
        data_bytes.extend_from_slice(&f16_2_0.to_le_bytes());

        let header_json = r#"{"f16_tensor":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#;
        let path = write_test_safetensors(header_json, &data_bytes);

        let result = read_safetensors(&path).expect("reading F16 tensor should succeed");
        let entry = &result["f16_tensor"];

        assert_eq!(entry.shape, vec![2]);
        assert_eq!(entry.dtype, "F16");
        assert!((entry.data[0] - 1.0f32).abs() < 1e-6, "F16 1.0 should convert to f32 1.0, got {}", entry.data[0]);
        assert!((entry.data[1] - 2.0f32).abs() < 1e-6, "F16 2.0 should convert to f32 2.0, got {}", entry.data[1]);

        cleanup(&path);
    }

    #[test]
    fn test_read_bf16_tensor() {
        // BF16 tensor with known values.
        // BF16 representation of 1.0: same exponent as F32 (0x7F), mantissa=0
        // F32 bits for 1.0 = 0x3F800000
        // BF16 = top 16 bits = 0x3F80
        // BF16 representation of -2.0:
        // F32 bits for -2.0 = 0xC0000000
        // BF16 = top 16 bits = 0xC000
        let bf16_1_0 = 0x3F80u16; // 1.0 in BF16
        let bf16_neg_2_0 = 0xC000u16; // -2.0 in BF16

        let mut data_bytes = Vec::new();
        data_bytes.extend_from_slice(&bf16_1_0.to_le_bytes());
        data_bytes.extend_from_slice(&bf16_neg_2_0.to_le_bytes());

        let header_json = r#"{"bf16_tensor":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
        let path = write_test_safetensors(header_json, &data_bytes);

        let result = read_safetensors(&path).expect("reading BF16 tensor should succeed");
        let entry = &result["bf16_tensor"];

        assert_eq!(entry.shape, vec![2]);
        assert_eq!(entry.dtype, "BF16");
        assert!((entry.data[0] - 1.0f32).abs() < 1e-6, "BF16 1.0 should convert to f32 1.0, got {}", entry.data[0]);
        assert!((entry.data[1] - (-2.0f32)).abs() < 1e-6, "BF16 -2.0 should convert to f32 -2.0, got {}", entry.data[1]);

        cleanup(&path);
    }

    #[test]
    fn test_unsupported_dtype() {
        // A tensor with an unsupported dtype should produce a clear error.
        let header_json = r#"{"bad_tensor":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#;
        let path = write_test_safetensors(header_json, &[0u8; 2]);

        let result = read_safetensors(&path);
        let err_msg = result.unwrap_err().to_string();

        assert!(
            err_msg.contains("Unsupported dtype U8 for tensor bad_tensor"),
            "error message should mention the dtype and tensor name, got: {}",
            err_msg,
        );
        assert!(
            err_msg.contains("only F32, BF16, and F16 are supported"),
            "error message should list supported dtypes, got: {}",
            err_msg,
        );

        cleanup(&path);
    }

    #[test]
    fn test_data_size_mismatch() {
        // A tensor whose data_offsets span doesn't match the shape * sizeof(f32)
        // should produce an error.
        // Shape [2, 3] with F32 should be 24 bytes, but data_offsets say [0, 10].
        let header_json = r#"{"mismatch":{"dtype":"F32","shape":[2,3],"data_offsets":[0,10]}}"#;
        let path = write_test_safetensors(header_json, &[0u8; 10]);

        let result = read_safetensors(&path);
        let err_msg = result.unwrap_err().to_string();

        assert!(
            err_msg.contains("Data size mismatch for tensor mismatch"),
            "error should mention data size mismatch, got: {}",
            err_msg,
        );

        cleanup(&path);
    }

    #[test]
    fn test_read_as_tensors() {
        // Verify that read_safetensors_as_tensors correctly converts entries
        // to our Tensor type.
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut data_bytes = Vec::new();
        for v in &values {
            data_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let header_json = r#"{"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let path = write_test_safetensors(header_json, &data_bytes);

        let tensors = read_safetensors_as_tensors(&path)
            .expect("reading safetensors as tensors should succeed");

        let t = &tensors["weight"];
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        cleanup(&path);
    }

    #[test]
    fn test_1d_tensor() {
        // A 1-D tensor (vector).
        let values = [0.5f32, -1.0, 2.5];
        let mut data_bytes = Vec::new();
        for v in &values {
            data_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let header_json = r#"{"bias":{"dtype":"F32","shape":[3],"data_offsets":[0,12]}}"#;
        let path = write_test_safetensors(header_json, &data_bytes);

        let result = read_safetensors(&path).expect("reading 1-D tensor should succeed");
        let entry = &result["bias"];

        assert_eq!(entry.shape, vec![3]);
        assert_eq!(entry.data, vec![0.5, -1.0, 2.5]);

        cleanup(&path);
    }

    #[test]
    fn test_zero_element_tensor() {
        // A tensor with a zero dimension (0 elements, 0 bytes of data).
        let header_json = r#"{"empty":{"dtype":"F32","shape":[0,4],"data_offsets":[0,0]}}"#;
        let path = write_test_safetensors(header_json, &[]);

        let result = read_safetensors(&path).expect("reading zero-element tensor should succeed");
        let entry = &result["empty"];

        assert_eq!(entry.shape, vec![0, 4]);
        assert!(entry.data.is_empty());

        cleanup(&path);
    }

    #[test]
    fn test_negative_f32_values() {
        // Ensure negative and special f32 values round-trip correctly.
        let values = [-0.0f32, -1.5, f32::MIN_POSITIVE, 0.0];
        let mut data_bytes = Vec::new();
        for v in &values {
            data_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let header_json = r#"{"negatives":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
        let path = write_test_safetensors(header_json, &data_bytes);

        let result = read_safetensors(&path).expect("reading negative values should succeed");
        let entry = &result["negatives"];

        assert_eq!(entry.data.len(), 4);
        assert_eq!(entry.data[0].to_bits(), (-0.0f32).to_bits()); // -0.0 preserved
        assert!((entry.data[1] - (-1.5)).abs() < 1e-6);
        assert!((entry.data[2] - f32::MIN_POSITIVE).abs() < 1e-30);

        cleanup(&path);
    }
}
