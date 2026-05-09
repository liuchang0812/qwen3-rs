//! Rotary Position Embedding (RoPE)
//!
//! This module implements Rotary Position Embedding (RoPE), the position
//! encoding scheme used in Qwen3 and most modern large language models
//! (LLaMA, Mistral, Qwen, etc.).
//!
//! # What RoPE Does
//!
//! RoPE encodes the **position** of each token by **rotating** pairs of
//! dimensions in the query (Q) and key (K) vectors used by attention.  Unlike
//! earlier approaches that *add* a position vector to the embedding, RoPE
//! *multiplies* by a rotation matrix.  The key mathematical property is that
//! the dot product of two RoPE-encoded vectors depends only on their
//! **relative** position, which is exactly what attention needs.
//!
//! # How It Works
//!
//! For a pair of values `(x_i, x_{i+1})` at position `p`, the rotation is:
//!
//! ```text
//! x_i'     = x_i * cos(p * theta_i) - x_{i+1} * sin(p * theta_i)
//! x_{i+1}' = x_i * sin(p * theta_i) + x_{i+1} * cos(p * theta_i)
//! ```
//!
//! This is equivalent to multiplying by a 2x2 rotation matrix:
//!
//! ```text
//! | x_i'     |   | cos(p*theta_i)  -sin(p*theta_i) | | x_i     |
//! | x_{i+1}' | = | sin(p*theta_i)   cos(p*theta_i) | | x_{i+1} |
//! ```
//!
//! The rotation angle depends on the position `p` and a frequency `theta_i`
//! that decreases geometrically across dimension pairs:
//!
//! ```text
//! theta_i = 1 / (base ^ (2i / head_dim))
//! ```
//!
//! where `base = 1000000.0` for Qwen3 and `i` ranges over
//! `0, 1, ..., head_dim/2 - 1`.
//!
//! # Implementation Strategy
//!
//! We precompute cos and sin tables at initialization for all positions up
//! to `max_seq_len`.  During the forward pass, we simply look up the values
//! for the current position range.  The rotation is applied by splitting
//! each head's dimension into a first half and a second half (as opposed to
//! interleaving adjacent pairs), which is equivalent but more efficient for
//! row-major tensor storage.

use crate::tensor::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// Precomputation
// ─────────────────────────────────────────────────────────────────────────────

/// Precompute the cosine and sine tables for Rotary Position Embedding.
///
/// This function computes the rotation angles for every (position,
/// dimension-pair) combination and stores their cosines and sines in two
/// 2-D tensors.  Because the rotation angles depend only on the position
/// and the dimension-pair index (not on the actual data), we can compute
/// these tables once at model initialization and reuse them for every
/// forward pass.
///
/// # Arguments
///
/// * `head_dim`     - The dimension of each attention head (e.g. 128 for
///                    Qwen3-0.6B).  Must be even.
/// * `max_seq_len`  - The maximum sequence length to precompute for.  The
///                    tables will have `max_seq_len` rows.
/// * `theta_base`   - The base of the geometric frequency schedule.
///                    Qwen3 uses 1000000.0.
///
/// # Returns
///
/// A tuple `(cos_table, sin_table)` where each tensor has shape
/// `[max_seq_len, head_dim / 2]`.
///
/// # Precomputation Steps
///
/// 1. For each dimension pair `i` (where `i = 0, 1, ..., head_dim/2 - 1`),
///    compute the frequency:
///
///    ```text
///    freq_i = 1.0 / (theta_base ^ (2i / head_dim))
///    ```
///
///    Lower-indexed pairs rotate faster (higher frequency), and
///    higher-indexed pairs rotate slower (lower frequency).  This gives
///    the model a "clock" with many hands: some rotate quickly to
///    distinguish nearby positions, while others rotate slowly to
///    distinguish far-apart positions.
///
/// 2. For each position `p` and each dimension pair `i`, compute the
///    rotation angle:
///
///    ```text
///    angle[p][i] = p * freq_i
///    ```
///
/// 3. Compute cos and sin of each angle.
///
/// # Example
///
/// ```ignore
/// let (cos_table, sin_table) = precompute_freqs(128, 40960, 1000000.0);
/// // cos_table.shape() == [40960, 64]
/// // sin_table.shape() == [40960, 64]
/// ```
pub fn precompute_freqs(
    head_dim: usize,
    max_seq_len: usize,
    theta_base: f32,
) -> (Tensor, Tensor) {
    assert!(
        head_dim % 2 == 0,
        "precompute_freqs: head_dim must be even, got {}",
        head_dim,
    );

    let half_dim = head_dim / 2;

    // Step 1: Compute the frequency for each dimension pair.
    //
    // freq_i = 1.0 / (theta_base ^ (2i / head_dim))
    //
    // For head_dim = 128 and theta_base = 1000000.0:
    //   freq_0 = 1 / 1000000^(0/128) = 1 / 1         = 1.0
    //   freq_1 = 1 / 1000000^(2/128) = 1 / 5.179     ≈ 0.1931
    //   freq_2 = 1 / 1000000^(4/128) = 1 / 26.827    ≈ 0.0373
    //   ...
    //   freq_63 = 1 / 1000000^(126/128) ≈ 0.000003
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| {
            let exponent = (2.0 * i as f32) / (head_dim as f32);
            1.0 / theta_base.powf(exponent)
        })
        .collect();

    // Step 2: Compute the angle for each (position, dimension-pair).
    //
    // angle[p][i] = p * freq_i
    //
    // This gives us a 2D table of shape [max_seq_len, half_dim].
    let mut cos_data = vec![0.0f32; max_seq_len * half_dim];
    let mut sin_data = vec![0.0f32; max_seq_len * half_dim];

    for p in 0..max_seq_len {
        for i in 0..half_dim {
            let angle = (p as f32) * freqs[i];
            let flat_idx = p * half_dim + i;
            cos_data[flat_idx] = angle.cos();
            sin_data[flat_idx] = angle.sin();
        }
    }

    // Step 3: Package into tensors of shape [max_seq_len, half_dim].
    let cos_table = Tensor::new(vec![max_seq_len, half_dim], cos_data);
    let sin_table = Tensor::new(vec![max_seq_len, half_dim], sin_data);

    (cos_table, sin_table)
}

// ─────────────────────────────────────────────────────────────────────────────
// Application
// ─────────────────────────────────────────────────────────────────────────────

/// Apply Rotary Position Embedding to a query or key tensor.
///
/// This function takes a tensor of Q or K vectors (shape
/// `[seq_len, num_heads, head_dim]`) and applies the position-dependent
/// rotation using precomputed cosine and sine tables.
///
/// # Arguments
///
/// * `x`         - The input tensor of shape `[seq_len, num_heads, head_dim]`.
///                 This is typically the Q or K projection output.
/// * `cos_table` - Precomputed cosine table of shape `[seq_len, head_dim/2]`.
///                 This should be a slice of the full precomputed table
///                 corresponding to positions `start_pos..start_pos+seq_len`.
/// * `sin_table` - Precomputed sine table of shape `[seq_len, head_dim/2]`.
///                 Same slicing as `cos_table`.
///
/// # Returns
///
/// A tensor of the same shape as `x` with RoPE applied.
///
/// # Rotation Algorithm
///
/// The head dimension is split into two halves (not interleaved pairs).
/// This is mathematically equivalent to the paired rotation but is more
/// cache-friendly for row-major storage:
///
/// ```text
/// x1 = x[..., :head_dim/2]        (first half of each head)
/// x2 = x[..., head_dim/2:]        (second half of each head)
///
/// out1 = x1 * cos - x2 * sin      (rotate first half)
/// out2 = x1 * sin + x2 * cos      (rotate second half)
///
/// out = concat(out1, out2)         (reassemble along last dim)
/// ```
///
/// # Why Split into Halves Instead of Interleaving?
///
/// The original RoPE paper describes rotating *adjacent* pairs:
/// `(x_0, x_1), (x_2, x_3), ...`.  An equivalent formulation is to group
/// all the "first elements" of each pair into one half and all the "second
/// elements" into another half.  With head_dim = 128:
///
/// ```text
/// Interleaved pairs:  (x_0,x_1), (x_2,x_3), ..., (x_126,x_127)
/// Half-split:         x_0,x_2,...,x_126  |  x_1,x_3,...,x_127
///                     ──── first half ────    ──── second half ────
/// ```
///
/// The half-split approach lets us do simple slice operations on contiguous
/// memory regions, which is much more efficient than gathering/scattering
/// individual elements.  The two approaches produce the same mathematical
/// result when the cos/sin tables are arranged to match.
///
/// # Panics
///
/// Panics if the shapes are incompatible (wrong number of dimensions,
/// `head_dim` is odd, or the cos/sin table length does not match
/// `head_dim/2`).
pub fn apply_rope(
    x: &Tensor,
    cos_table: &Tensor,
    sin_table: &Tensor,
) -> Tensor {
    // --- Validate input shapes ---
    assert_eq!(
        x.ndim(),
        3,
        "apply_rope: x must be 3-D [seq_len, num_heads, head_dim], got shape {:?}",
        x.shape(),
    );
    let seq_len = x.shape()[0];
    let num_heads = x.shape()[1];
    let head_dim = x.shape()[2];

    assert!(
        head_dim % 2 == 0,
        "apply_rope: head_dim must be even, got {}",
        head_dim,
    );

    let half_dim = head_dim / 2;

    // Validate cos/sin table shapes.
    assert_eq!(
        cos_table.ndim(),
        2,
        "apply_rope: cos_table must be 2-D, got shape {:?}",
        cos_table.shape(),
    );
    assert_eq!(
        sin_table.ndim(),
        2,
        "apply_rope: sin_table must be 2-D, got shape {:?}",
        sin_table.shape(),
    );
    assert_eq!(
        cos_table.shape()[0], seq_len,
        "apply_rope: cos_table has {} rows but x has seq_len={}",
        cos_table.shape()[0], seq_len,
    );
    assert_eq!(
        sin_table.shape()[0], seq_len,
        "apply_rope: sin_table has {} rows but x has seq_len={}",
        sin_table.shape()[0], seq_len,
    );
    assert_eq!(
        cos_table.shape()[1], half_dim,
        "apply_rope: cos_table has {} cols but head_dim/2={}",
        cos_table.shape()[1], half_dim,
    );
    assert_eq!(
        sin_table.shape()[1], half_dim,
        "apply_rope: sin_table has {} cols but head_dim/2={}",
        sin_table.shape()[1], half_dim,
    );

    // --- Step 1: Split x into first half and second half along head_dim ---
    //
    // For each position p and each head h:
    //   x1[p][h] = x[p][h][0..half_dim]      (first half)
    //   x2[p][h] = x[p][h][half_dim..head_dim] (second half)
    //
    // In row-major storage, element x[p][h][d] is at flat index:
    //   p * (num_heads * head_dim) + h * head_dim + d

    let mut out_data = vec![0.0f32; x.len()];

    for p in 0..seq_len {
        for h in 0..num_heads {
            // Base flat index for x[p][h][..]
            let x_base = p * num_heads * head_dim + h * head_dim;

            // Base flat index for cos_table[p][..] and sin_table[p][..]
            let cs_base = p * half_dim;

            // --- Step 2: Apply the rotation ---
            //
            // For each dimension-pair index i (0..half_dim):
            //
            //   x1_i = x[p][h][i]               (first half element)
            //   x2_i = x[p][h][half_dim + i]     (second half element)
            //   cos_i = cos_table[p][i]
            //   sin_i = sin_table[p][i]
            //
            //   out1_i = x1_i * cos_i - x2_i * sin_i    (rotated first half)
            //   out2_i = x1_i * sin_i + x2_i * cos_i    (rotated second half)

            for i in 0..half_dim {
                let x1_i = x.data()[x_base + i];
                let x2_i = x.data()[x_base + half_dim + i];
                let cos_i = cos_table.data()[cs_base + i];
                let sin_i = sin_table.data()[cs_base + i];

                // --- Step 3: Compute rotated values ---
                //
                // This is the 2x2 rotation matrix applied to the pair
                // (x1_i, x2_i) with angle (p * freq_i):
                //
                // | out1_i |   | cos  -sin | | x1_i |
                // | out2_i | = | sin   cos | | x2_i |

                let out1_i = x1_i * cos_i - x2_i * sin_i;
                let out2_i = x1_i * sin_i + x2_i * cos_i;

                // --- Step 4: Write outputs back in the same half-split layout ---
                out_data[x_base + i] = out1_i;
                out_data[x_base + half_dim + i] = out2_i;
            }
        }
    }

    Tensor::new(vec![seq_len, num_heads, head_dim], out_data)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that at position 0, cos = 1 and sin = 0 for all frequencies.
    ///
    /// Since angle = position * freq, position 0 gives angle = 0 for all
    /// frequencies.  Therefore cos(0) = 1 and sin(0) = 0.  This means
    /// applying RoPE at position 0 should not change the vector at all
    /// (rotation by zero angle is the identity).
    #[test]
    fn test_precompute_freqs_position_zero() {
        let (cos_table, sin_table) = precompute_freqs(8, 4, 10000.0);

        // Check all entries at position 0 (row 0).
        for i in 0..4 {
            let cos_val = cos_table.get(&[0, i]);
            let sin_val = sin_table.get(&[0, i]);
            assert!(
                (cos_val - 1.0).abs() < 1e-6,
                "cos(0) should be 1.0, got {} at pair {}",
                cos_val,
                i,
            );
            assert!(
                sin_val.abs() < 1e-6,
                "sin(0) should be 0.0, got {} at pair {}",
                sin_val,
                i,
            );
        }
    }

    /// Verify that values at position 1 match the expected formula.
    ///
    /// For pair index i: angle = 1.0 * freq_i, where
    /// freq_i = 1.0 / (10000.0 ^ (2i / head_dim)).
    ///
    /// With head_dim = 8, the four frequencies are:
    ///   freq_0 = 1 / 10000^(0/8) = 1.0
    ///   freq_1 = 1 / 10000^(2/8) = 1 / 10000^0.25 ≈ 0.1
    ///   freq_2 = 1 / 10000^(4/8) = 1 / 10000^0.5  = 0.01
    ///   freq_3 = 1 / 10000^(6/8) = 1 / 10000^0.75 ≈ 0.001
    #[test]
    fn test_precompute_freqs_position_one() {
        let head_dim = 8usize;
        let theta_base = 10000.0f32;
        let (cos_table, sin_table) = precompute_freqs(head_dim, 4, theta_base);

        for i in 0..(head_dim / 2) {
            let exponent = (2.0 * i as f32) / (head_dim as f32);
            let freq = 1.0 / theta_base.powf(exponent);
            let expected_angle = freq; // position 1 * freq
            let expected_cos = expected_angle.cos();
            let expected_sin = expected_angle.sin();

            let cos_val = cos_table.get(&[1, i]);
            let sin_val = sin_table.get(&[1, i]);

            assert!(
                (cos_val - expected_cos).abs() < 1e-5,
                "cos at position 1, pair {}: expected {}, got {}",
                i,
                expected_cos,
                cos_val,
            );
            assert!(
                (sin_val - expected_sin).abs() < 1e-5,
                "sin at position 1, pair {}: expected {}, got {}",
                i,
                expected_sin,
                sin_val,
            );
        }
    }

    /// Verify that applying RoPE at position 0 does not change the vector.
    ///
    /// Since cos(0) = 1 and sin(0) = 0 for all pairs, the rotation matrix
    /// at position 0 is the identity.  Therefore the output should equal
    /// the input.
    #[test]
    fn test_apply_rope_position_zero_is_identity() {
        // x: shape [1, 2, 4] — one position, two heads, head_dim = 4
        let x = Tensor::new(
            vec![1, 2, 4],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );

        let (cos_table, sin_table) = precompute_freqs(4, 4, 10000.0);

        // Slice position 0 from the full tables.
        let cos_slice = cos_table.rows(0, 1);
        let sin_slice = sin_table.rows(0, 1);

        let out = apply_rope(&x, &cos_slice, &sin_slice);

        // Output should be identical to input (rotation by zero angle).
        for i in 0..x.len() {
            assert!(
                (out.data()[i] - x.data()[i]).abs() < 1e-5,
                "position 0 should be identity: x[{}] = {}, out[{}] = {}",
                i,
                x.data()[i],
                i,
                out.data()[i],
            );
        }
    }

    /// Verify that the rotation is applied correctly with a concrete example.
    ///
    /// We use a small tensor with known values and manually compute the
    /// expected output.
    ///
    /// Setup: head_dim = 4, theta_base = 10000.0, position = 3.
    ///
    /// Frequencies:
    ///   freq_0 = 1 / 10000^(0/4) = 1.0
    ///   freq_1 = 1 / 10000^(2/4) = 0.01
    ///
    /// Angles at position 3:
    ///   angle_0 = 3 * 1.0  = 3.0
    ///   angle_1 = 3 * 0.01 = 0.03
    ///
    /// Input x (1 position, 1 head, head_dim = 4): [1.0, 0.0, 0.0, 1.0]
    ///   x1 (first half)  = [1.0, 0.0]
    ///   x2 (second half) = [0.0, 1.0]
    ///
    /// Rotation:
    ///   out1[0] = x1[0]*cos(3.0) - x2[0]*sin(3.0) = 1.0*cos(3.0) - 0.0*sin(3.0) = cos(3.0)
    ///   out1[1] = x1[1]*cos(0.03) - x2[1]*sin(0.03) = 0.0*cos(0.03) - 1.0*sin(0.03) = -sin(0.03)
    ///   out2[0] = x1[0]*sin(3.0) + x2[0]*cos(3.0) = 1.0*sin(3.0) + 0.0*cos(3.0) = sin(3.0)
    ///   out2[1] = x1[1]*sin(0.03) + x2[1]*cos(0.03) = 0.0*sin(0.03) + 1.0*cos(0.03) = cos(0.03)
    #[test]
    fn test_apply_rope_rotation_values() {
        // head_dim = 4, position = 3
        let (cos_table, sin_table) = precompute_freqs(4, 10, 10000.0);
        let cos_slice = cos_table.rows(3, 4);
        let sin_slice = sin_table.rows(3, 4);

        // x: shape [1, 1, 4] — one position, one head
        let x = Tensor::new(vec![1, 1, 4], vec![1.0, 0.0, 0.0, 1.0]);

        let out = apply_rope(&x, &cos_slice, &sin_slice);

        // Manually compute expected values.
        let angle_0 = 3.0f32 * 1.0; // position 3 * freq_0 = 3.0
        let angle_1 = 3.0f32 * 0.01; // position 3 * freq_1 = 0.03

        let expected = vec![
            angle_0.cos(),   // out1[0] = 1.0 * cos(3.0) - 0.0 * sin(3.0)
            -angle_1.sin(),  // out1[1] = 0.0 * cos(0.03) - 1.0 * sin(0.03)
            angle_0.sin(),   // out2[0] = 1.0 * sin(3.0) + 0.0 * cos(3.0)
            angle_1.cos(),   // out2[1] = 0.0 * sin(0.03) + 1.0 * cos(0.03)
        ];

        for i in 0..4 {
            assert!(
                (out.data()[i] - expected[i]).abs() < 1e-5,
                "out[{}] = {}, expected {}",
                i,
                out.data()[i],
                expected[i],
            );
        }
    }

    /// Verify that the output shape matches the input shape.
    #[test]
    fn test_apply_rope_output_shape() {
        let seq_len = 5;
        let num_heads = 8;
        let head_dim = 16;

        let x = Tensor::ones(vec![seq_len, num_heads, head_dim]);
        let (cos_table, sin_table) = precompute_freqs(head_dim, seq_len, 10000.0);

        let out = apply_rope(&x, &cos_table, &sin_table);

        assert_eq!(
            out.shape(),
            &[seq_len, num_heads, head_dim],
            "output shape should match input shape",
        );
    }

    /// Verify that RoPE is applied consistently across multiple positions.
    ///
    /// A vector at position 0 should be unchanged, and at position 1 it
    /// should be rotated.  We check that the two positions produce
    /// different outputs (otherwise RoPE would be doing nothing useful).
    #[test]
    fn test_apply_rope_different_positions_differ() {
        let head_dim = 8;
        let (cos_table, sin_table) = precompute_freqs(head_dim, 10, 10000.0);

        // Same input vector at two different positions.
        let x = Tensor::new(
            vec![2, 1, head_dim],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // position 0
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // position 1
            ],
        );

        // Use positions 0 and 1 from the precomputed tables.
        let cos_slice = cos_table.rows(0, 2);
        let sin_slice = sin_table.rows(0, 2);

        let out = apply_rope(&x, &cos_slice, &sin_slice);

        // Position 0 should equal the input (cos=1, sin=0).
        for d in 0..head_dim {
            assert!(
                (out.data()[d] - x.data()[d]).abs() < 1e-5,
                "position 0 should be identity: dim {} = {} vs {}",
                d,
                out.data()[d],
                x.data()[d],
            );
        }

        // Position 1 should differ from the input.
        let mut any_different = false;
        for d in 0..head_dim {
            if (out.data()[head_dim + d] - x.data()[head_dim + d]).abs() > 1e-5 {
                any_different = true;
                break;
            }
        }
        assert!(
            any_different,
            "position 1 should produce different output than input",
        );
    }

    /// Verify that the precomputed table shapes are correct.
    #[test]
    fn test_precompute_freqs_shapes() {
        let head_dim = 128;
        let max_seq_len = 40960;
        let (cos_table, sin_table) = precompute_freqs(head_dim, max_seq_len, 1000000.0);

        assert_eq!(cos_table.shape(), &[max_seq_len, head_dim / 2]);
        assert_eq!(sin_table.shape(), &[max_seq_len, head_dim / 2]);
    }
}
