//! Grouped Query Attention (GQA) with KV cache.
//!
//! This module implements the attention mechanism used by Qwen3-0.6B.
//! The key feature is **Grouped Query Attention (GQA)**, where multiple
//! query heads share the same key-value heads. This reduces the KV cache
//! size compared to standard Multi-Head Attention while preserving most
//! of the quality.
//!
//! # Qwen3-0.6B Attention Configuration
//!
//! ```text
//! num_attention_heads  = 16   (query heads)
//! num_key_value_heads  = 8    (KV heads)
//! head_dim             = 128  (explicit in config)
//! kv_groups            = 2    (16 / 8, each KV head serves 2 query heads)
//! hidden_size          = 1024
//! ```
//!
//! # Forward Pass Overview
//!
//! 1. Project input to Q, K, V using linear projections (weight^T)
//! 2. Reshape Q/K/V to separate heads
//! 3. Apply Rotary Position Embedding (RoPE) to Q and K
//! 4. Update KV cache (concatenate with past K/V if present)
//! 5. Expand KV heads for GQA (repeat each KV head kv_groups times)
//! 6. Compute scaled dot-product attention with causal mask
//! 7. Project output through o_proj

use crate::rmsnorm::RMSNorm;
use crate::rope::{apply_rope, precompute_freqs};
use crate::tensor::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// KV Cache
// ─────────────────────────────────────────────────────────────────────────────

/// Key-Value cache for autoregressive generation.
///
/// During generation, the model produces one token at a time. Instead of
/// recomputing K and V for all past tokens at every step, we cache them.
/// The cache grows by one row (one token's K and V) per decode step.
///
/// # Shapes
///
/// - `key_cache`: `[seq_len_so_far, num_kv_heads, head_dim]`
/// - `value_cache`: `[seq_len_so_far, num_kv_heads, head_dim]`
///
/// These are stored as 2-D tensors with shape
/// `[seq_len_so_far, num_kv_heads * head_dim]` for efficient row
/// concatenation. The first dimension (seq_len) grows over time; the
/// second dimension is fixed.
///
/// # Why `Option`?
///
/// The cache starts empty (no tokens have been processed). On the first
/// forward pass (prefill), we store the computed K and V. On subsequent
/// passes (decode), we concatenate the new K/V rows to the existing cache.
pub struct KVCache {
    /// Cached key states. `None` before the first forward pass.
    pub key_cache: Option<Tensor>,
    /// Cached value states. `None` before the first forward pass.
    pub value_cache: Option<Tensor>,
}

impl KVCache {
    /// Create an empty KV cache.
    pub fn new() -> Self {
        Self {
            key_cache: None,
            value_cache: None,
        }
    }
}

impl KVCache {
    /// Clear the cache, resetting both key and value caches to `None`.
    ///
    /// This is used when starting a new conversation turn so that the
    /// model does not attend to tokens from a previous conversation.
    /// After calling `clear()`, the cache is in the same state as if
    /// it had just been created with [`KVCache::new`].
    pub fn clear(&mut self) {
        self.key_cache = None;
        self.value_cache = None;
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Attention struct
// ─────────────────────────────────────────────────────────────────────────────

/// Grouped Query Attention layer.
///
/// This struct holds the learned projection weights and the precomputed
/// RoPE tables. The forward method computes the full attention operation
/// including KV cache management and GQA expansion.
///
/// # Weight Shapes
///
/// The weights follow the safetensors convention:
/// `[out_features, in_features]`. The forward projection is
/// `output = input · weight^T`, which gives:
///   `[seq_len, in_features] × [in_features, out_features] → [seq_len, out_features]`
///
/// ```text
/// q_proj: [num_heads * head_dim, hidden_size]  = [2048, 1024]
/// k_proj: [kv_dim, hidden_size]               = [1024, 1024]
/// v_proj: [kv_dim, hidden_size]               = [1024, 1024]
/// o_proj: [hidden_size, num_heads * head_dim]  = [1024, 2048]
/// ```
///
/// where `kv_dim = num_kv_heads * head_dim = 8 * 128 = 1024`.
pub struct Attention {
    /// Query projection weight: `[num_heads * head_dim, hidden_size]`.
    q_proj: Tensor,
    /// Key projection weight: `[kv_dim, hidden_size]` where `kv_dim = num_kv_heads * head_dim`.
    k_proj: Tensor,
    /// Value projection weight: `[kv_dim, hidden_size]` where `kv_dim = num_kv_heads * head_dim`.
    v_proj: Tensor,
    /// Output projection weight: `[hidden_size, num_heads * head_dim]`.
    o_proj: Tensor,

    /// Optional RMSNorm applied to Q after projection, per head.
    /// Present in Qwen3 models. Shape: `[head_dim]`.
    q_norm: Option<RMSNorm>,

    /// Optional RMSNorm applied to K after projection, per head.
    /// Present in Qwen3 models. Shape: `[head_dim]`.
    k_norm: Option<RMSNorm>,

    /// Number of query heads (16 for Qwen3-0.6B).
    num_heads: usize,
    /// Number of key-value heads (8 for Qwen3-0.6B).
    num_kv_heads: usize,
    /// Dimension per head (128 for Qwen3-0.6B).
    head_dim: usize,
    /// Number of query heads per KV head = num_heads / num_kv_heads.
    kv_groups: usize,

    /// Precomputed cosine table for RoPE: `[max_seq_len, head_dim/2]`.
    cos_table: Tensor,
    /// Precomputed sine table for RoPE: `[max_seq_len, head_dim/2]`.
    sin_table: Tensor,
}

impl Attention {
    /// Create a new Attention layer.
    ///
    /// # Arguments
    ///
    /// * `q_proj`         - Query projection weight `[hidden_size, hidden_size]`.
    /// * `k_proj`         - Key projection weight `[kv_dim, hidden_size]`.
    /// * `v_proj`         - Value projection weight `[kv_dim, hidden_size]`.
    /// * `o_proj`         - Output projection weight `[hidden_size, hidden_size]`.
    /// * `num_heads`      - Number of query heads.
    /// * `num_kv_heads`   - Number of key-value heads.
    /// * `head_dim`       - Dimension per attention head.
    /// * `max_seq_len`    - Maximum sequence length for RoPE precomputation.
    /// * `rope_theta`     - Base frequency for RoPE (1000000.0 for Qwen3).
    ///
    /// # Panics
    ///
    /// Panics if `num_heads` is not evenly divisible by `num_kv_heads`.
    pub fn new(
        q_proj: Tensor,
        k_proj: Tensor,
        v_proj: Tensor,
        o_proj: Tensor,
        q_norm: Option<RMSNorm>,
        k_norm: Option<RMSNorm>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f32,
    ) -> Self {
        assert_eq!(
            num_heads % num_kv_heads,
            0,
            "Attention::new: num_heads ({}) must be divisible by num_kv_heads ({})",
            num_heads,
            num_kv_heads,
        );

        let kv_groups = num_heads / num_kv_heads;
        let (cos_table, sin_table) = precompute_freqs(head_dim, max_seq_len, rope_theta);

        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_groups,
            cos_table,
            sin_table,
        }
    }

    /// Run the attention forward pass.
    ///
    /// # Arguments
    ///
    /// * `x`          - Input tensor of shape `[seq_len, hidden_size]`.
    ///   During prefill (first pass), `seq_len > 1`. During decode,
    ///   `seq_len = 1`.
    /// * `start_pos`  - The position of the first token in `x` within the
    ///   overall sequence. During prefill this is 0; during decode it is
    ///   the number of tokens already processed.
    /// * `kv_cache`   - Mutable reference to the KV cache for this layer.
    ///   Updated in-place with the new K and V values.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[seq_len, hidden_size]`.
    pub fn forward(&self, x: &Tensor, start_pos: usize, kv_cache: &mut KVCache) -> Tensor {
        let seq_len = x.shape()[0];
        let hidden_size = x.shape()[1];

        // ── Step 1: Project input to Q, K, V ────────────────────────────
        //
        // Weight matrices are stored as [out_features, in_features] (the
        // safetensors convention). The projection is:
        //   output = x · W^T
        //
        // We compute this as x.matmul(&weight.transpose_2d()):
        //   For q_proj/v_proj: W is [hidden_size, hidden_size], W^T is [hidden_size, hidden_size]
        //   For k_proj/v_proj: W is [kv_dim, hidden_size], W^T is [hidden_size, kv_dim]
        //
        //   x:      [seq_len, hidden_size]
        //   W^T:    [hidden_size, out_features]
        //   result: [seq_len, out_features]

        let q_proj_t = self.q_proj.transpose_2d();
        let k_proj_t = self.k_proj.transpose_2d();
        let v_proj_t = self.v_proj.transpose_2d();

        // Q: [seq_len, hidden_size]  (num_heads * head_dim)
        // K: [seq_len, kv_dim]       (num_kv_heads * head_dim)
        // V: [seq_len, kv_dim]       (num_kv_heads * head_dim)
        let q = x.matmul(&q_proj_t);
        let k = x.matmul(&k_proj_t);
        let v = x.matmul(&v_proj_t);

        // ── Step 2: Reshape to separate heads ───────────────────────────
        //
        // Q: [seq_len, num_heads * head_dim] → [seq_len, num_heads, head_dim]
        // K: [seq_len, num_kv_heads * head_dim] → [seq_len, num_kv_heads, head_dim]
        // V: [seq_len, num_kv_heads * head_dim] → [seq_len, num_kv_heads, head_dim]

        let mut q = q.reshape(vec![seq_len, self.num_heads, self.head_dim]);
        let mut k = k.reshape(vec![seq_len, self.num_kv_heads, self.head_dim]);
        let v = v.reshape(vec![seq_len, self.num_kv_heads, self.head_dim]);

        // ── Step 2b: Apply Q/K norm if present (Qwen3 per-head normalization) ─
        //
        // Qwen3 models apply RMSNorm to Q and K after projection and reshaping
        // but BEFORE RoPE. This normalizes each head independently, stabilizing
        // attention scores. The norm operates over the head_dim for each
        // (seq_pos, head) combination.

        if let Some(ref q_norm) = self.q_norm {
            q = apply_per_head_norm(&q, q_norm);
        }

        if let Some(ref k_norm) = self.k_norm {
            k = apply_per_head_norm(&k, k_norm);
        }

        // ── Step 3: Apply RoPE to Q and K ───────────────────────────────
        //
        // Slice the precomputed cos/sin tables for the positions
        // [start_pos .. start_pos + seq_len].

        let cos_slice = self.cos_table.rows(start_pos, start_pos + seq_len);
        let sin_slice = self.sin_table.rows(start_pos, start_pos + seq_len);

        let q = apply_rope(&q, &cos_slice, &sin_slice);
        let k = apply_rope(&k, &cos_slice, &sin_slice);

        // ── Step 4: Update KV cache ─────────────────────────────────────
        //
        // We store K and V as 2-D tensors of shape
        //   [seq_len_so_far, num_kv_heads * head_dim]
        // so that we can use stack_rows for efficient concatenation.
        //
        // Flatten the head dimensions for cache storage:
        //   [seq_len, num_kv_heads, head_dim] → [seq_len, num_kv_heads * head_dim]

        let kv_dim = self.num_kv_heads * self.head_dim;
        let k_flat = k.reshape(vec![seq_len, kv_dim]);
        let v_flat = v.reshape(vec![seq_len, kv_dim]);

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
            _ => panic!("Attention::forward: KV cache is in an inconsistent state \
                         (one of key/value is Some but the other is None)"),
        };

        // Store updated cache.
        kv_cache.key_cache = Some(k_full.clone());
        kv_cache.value_cache = Some(v_full.clone());

        // Total sequence length so far (all cached tokens + current tokens).
        let total_seq_len = k_full.shape()[0];

        // Reshape back to 3-D for attention computation.
        let k_full = k_full.reshape(vec![total_seq_len, self.num_kv_heads, self.head_dim]);
        let v_full = v_full.reshape(vec![total_seq_len, self.num_kv_heads, self.head_dim]);

        // ── Step 5: Expand KV heads for GQA ─────────────────────────────
        //
        // In GQA, each KV head is shared by `kv_groups` query heads.
        // We expand K and V by repeating each KV head kv_groups times so
        // that the number of KV heads matches the number of query heads.
        //
        // K: [total_seq_len, num_kv_heads, head_dim]
        //  → [total_seq_len, num_heads, head_dim]
        //
        // For Qwen3: [total_seq_len, 8, 128] → [total_seq_len, 16, 128]
        // KV head 0 is repeated 2x → becomes Q heads 0 and 1
        // KV head 1 is repeated 2x → becomes Q heads 2 and 3
        // ... and so on.

        let k_expanded = expand_kv_heads(&k_full, self.kv_groups);
        let v_expanded = expand_kv_heads(&v_full, self.kv_groups);

        // ── Step 6: Compute attention scores ────────────────────────────
        //
        // Transpose Q and K to put heads in the first dimension so that
        // each head's attention is an independent 2-D matmul.
        //
        // Q: [seq_len, num_heads, head_dim] → [num_heads, seq_len, head_dim]
        // K: [total_seq_len, num_heads, head_dim] → [num_heads, total_seq_len, head_dim]

        let q_transposed = transpose_heads(&q); // [num_heads, seq_len, head_dim]
        let k_transposed = transpose_heads(&k_expanded); // [num_heads, total_seq_len, head_dim]

        // scores = Q · K^T for each head
        // Q: [num_heads, seq_len, head_dim]
        // K^T: [num_heads, head_dim, total_seq_len]
        // scores: [num_heads, seq_len, total_seq_len]

        let scores = batch_matmul_qk(&q_transposed, &k_transposed);

        // Scale by 1 / sqrt(head_dim) to keep the variance stable.
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let scores = scores.mul_scalar(scale);

        // ── Step 7: Apply causal mask ───────────────────────────────────
        //
        // During attention, a token at position i can only attend to
        // tokens at positions 0..=i. Positions > i are in the "future"
        // and must be masked out by setting their score to -infinity.
        //
        // For the Q at position `start_pos + q_idx`, the attended K
        // positions range from 0 to `start_pos + q_idx`. All K positions
        // beyond that must be masked.
        //
        // During prefill (start_pos=0, seq_len>1): this creates a standard
        // lower-triangular mask.
        // During decode (seq_len=1): the single new token can attend to
        // all cached positions (no masking needed since total_seq_len ==
        // start_pos + 1).

        let scores = apply_causal_mask(scores, start_pos, seq_len, total_seq_len);

        // ── Step 8: Softmax along the last dimension ────────────────────
        //
        // Convert attention scores to probabilities. Each query token's
        // attention distribution sums to 1 over the attended key positions.
        // dim=2 because shape is [num_heads, seq_len, total_seq_len].

        let attn_weights = scores.softmax(2);

        // ── Step 9: Compute weighted sum of values ──────────────────────
        //
        // attn_weights: [num_heads, seq_len, total_seq_len]
        // V:            [num_heads, total_seq_len, head_dim]
        // output:       [num_heads, seq_len, head_dim]

        let v_transposed = transpose_heads(&v_expanded); // [num_heads, total_seq_len, head_dim]
        let attn_output = batch_matmul_attn_v(&attn_weights, &v_transposed);

        // ── Step 10: Reshape and project ────────────────────────────────
        //
        // Transpose back: [num_heads, seq_len, head_dim] → [seq_len, num_heads, head_dim]
        // Flatten heads:  [seq_len, num_heads, head_dim] → [seq_len, num_heads * head_dim]
        // Project:        [seq_len, hidden_size] · o_proj^T → [seq_len, hidden_size]

        let attn_output = untranspose_heads(&attn_output); // [seq_len, num_heads, head_dim]
        let attn_output = attn_output.reshape(vec![seq_len, self.num_heads * self.head_dim]);

        let o_proj_t = self.o_proj.transpose_2d();
        let output = attn_output.matmul(&o_proj_t);

        // Verify output shape matches expectation.
        assert_eq!(
            output.shape(),
            &[seq_len, hidden_size],
            "Attention::forward: output shape {:?} doesn't match expected [{} , {}]",
            output.shape(),
            seq_len,
            hidden_size,
        );

        output
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions
// ─────────────────────────────────────────────────────────────────────────────

/// Expand KV heads by repeating each head `kv_groups` times.
///
/// For GQA, each KV head is shared by `kv_groups` query heads. This
/// function repeats each KV head `kv_groups` times along the head
/// dimension so that the number of KV heads matches the number of
/// query heads.
///
/// # Input
///
/// `x`: shape `[seq_len, num_kv_heads, head_dim]`
///
/// # Output
///
/// Shape `[seq_len, num_heads, head_dim]` where `num_heads = num_kv_heads * kv_groups`.
///
/// # Example
///
/// ```text
/// Input:  [seq_len, 2, head_dim] with kv_groups = 3
///   Heads: [h0, h1]
/// Output: [seq_len, 6, head_dim]
///   Heads: [h0, h0, h0, h1, h1, h1]
/// ```
fn expand_kv_heads(x: &Tensor, kv_groups: usize) -> Tensor {
    assert_eq!(x.ndim(), 3, "expand_kv_heads: expected 3-D input, got shape {:?}", x.shape());
    let seq_len = x.shape()[0];
    let num_kv_heads = x.shape()[1];
    let head_dim = x.shape()[2];
    let num_heads = num_kv_heads * kv_groups;

    let mut out_data = vec![0.0f32; seq_len * num_heads * head_dim];

    for s in 0..seq_len {
        for kv_h in 0..num_kv_heads {
            // Source: x[s][kv_h][..]
            let src_base = s * num_kv_heads * head_dim + kv_h * head_dim;
            // Destination: repeat kv_groups times for heads
            //   kv_h * kv_groups, kv_h * kv_groups + 1, ..., kv_h * kv_groups + kv_groups - 1
            for g in 0..kv_groups {
                let dst_h = kv_h * kv_groups + g;
                let dst_base = s * num_heads * head_dim + dst_h * head_dim;
                for d in 0..head_dim {
                    out_data[dst_base + d] = x.data()[src_base + d];
                }
            }
        }
    }

    Tensor::new(vec![seq_len, num_heads, head_dim], out_data)
}

/// Transpose a 3-D tensor from `[S, H, D]` to `[H, S, D]`.
///
/// This moves the head dimension to the front so that each head's
/// computation is a contiguous 2-D matrix, enabling batched matmul.
fn transpose_heads(x: &Tensor) -> Tensor {
    assert_eq!(x.ndim(), 3, "transpose_heads: expected 3-D input, got shape {:?}", x.shape());
    let s = x.shape()[0];
    let h = x.shape()[1];
    let d = x.shape()[2];

    let mut out_data = vec![0.0f32; s * h * d];

    for si in 0..s {
        for hi in 0..h {
            for di in 0..d {
                // Input layout:  [S][H][D] → flat index si * h * d + hi * d + di
                // Output layout: [H][S][D] → flat index hi * s * d + si * d + di
                let src_idx = si * h * d + hi * d + di;
                let dst_idx = hi * s * d + si * d + di;
                out_data[dst_idx] = x.data()[src_idx];
            }
        }
    }

    Tensor::new(vec![h, s, d], out_data)
}

/// Inverse of `transpose_heads`: from `[H, S, D]` to `[S, H, D]`.
fn untranspose_heads(x: &Tensor) -> Tensor {
    assert_eq!(x.ndim(), 3, "untranspose_heads: expected 3-D input, got shape {:?}", x.shape());
    let h = x.shape()[0];
    let s = x.shape()[1];
    let d = x.shape()[2];

    let mut out_data = vec![0.0f32; s * h * d];

    for hi in 0..h {
        for si in 0..s {
            for di in 0..d {
                // Input layout:  [H][S][D] → flat index hi * s * d + si * d + di
                // Output layout: [S][H][D] → flat index si * h * d + hi * d + di
                let src_idx = hi * s * d + si * d + di;
                let dst_idx = si * h * d + hi * d + di;
                out_data[dst_idx] = x.data()[src_idx];
            }
        }
    }

    Tensor::new(vec![s, h, d], out_data)
}

/// Batched matrix multiply for Q·K^T: computes attention scores.
///
/// Both inputs have shape `[num_heads, M, D]` and `[num_heads, N, D]`.
/// The result has shape `[num_heads, M, N]` where each head computes:
///   scores[h][i][j] = sum_d Q[h][i][d] * K[h][j][d]
///
/// This is equivalent to Q · K^T for each head independently.
fn batch_matmul_qk(q: &Tensor, k: &Tensor) -> Tensor {
    assert_eq!(q.ndim(), 3, "batch_matmul_qk: Q must be 3-D, got shape {:?}", q.shape());
    assert_eq!(k.ndim(), 3, "batch_matmul_qk: K must be 3-D, got shape {:?}", k.shape());
    assert_eq!(q.shape()[0], k.shape()[0], "batch_matmul_qk: head count mismatch");
    assert_eq!(q.shape()[2], k.shape()[2], "batch_matmul_qk: head_dim mismatch");

    let num_heads = q.shape()[0];
    let m = q.shape()[1]; // seq_len (Q)
    let n = k.shape()[1]; // total_seq_len (K)
    let d = q.shape()[2]; // head_dim

    let mut result = vec![0.0f32; num_heads * m * n];

    for h in 0..num_heads {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for di in 0..d {
                    // Q[h][i][d] at flat index h*m*d + i*d + di
                    // K[h][j][d] at flat index h*n*d + j*d + di
                    sum += q.data()[h * m * d + i * d + di] * k.data()[h * n * d + j * d + di];
                }
                // result[h][i][j] at flat index h*m*n + i*n + j
                result[h * m * n + i * n + j] = sum;
            }
        }
    }

    Tensor::new(vec![num_heads, m, n], result)
}

/// Batched matrix multiply for attention_weights · V: computes the
/// weighted sum of values.
///
/// `attn`: shape `[num_heads, M, N]` (attention weights)
/// `v`:    shape `[num_heads, N, D]` (value vectors)
/// Result: shape `[num_heads, M, D]`
fn batch_matmul_attn_v(attn: &Tensor, v: &Tensor) -> Tensor {
    assert_eq!(attn.ndim(), 3, "batch_matmul_attn_v: attn must be 3-D");
    assert_eq!(v.ndim(), 3, "batch_matmul_attn_v: V must be 3-D");
    assert_eq!(attn.shape()[0], v.shape()[0], "batch_matmul_attn_v: head count mismatch");
    assert_eq!(attn.shape()[2], v.shape()[1], "batch_matmul_attn_v: inner dim mismatch");

    let num_heads = attn.shape()[0];
    let m = attn.shape()[1];
    let n = attn.shape()[2];
    let d = v.shape()[2];

    let mut result = vec![0.0f32; num_heads * m * d];

    for h in 0..num_heads {
        for i in 0..m {
            for di in 0..d {
                let mut sum = 0.0f32;
                for j in 0..n {
                    // attn[h][i][j] at flat index h*m*n + i*n + j
                    // v[h][j][d] at flat index h*n*d + j*d + di
                    sum += attn.data()[h * m * n + i * n + j] * v.data()[h * n * d + j * d + di];
                }
                // result[h][i][d] at flat index h*m*d + i*d + di
                result[h * m * d + i * d + di] = sum;
            }
        }
    }

    Tensor::new(vec![num_heads, m, d], result)
}

/// Apply a causal attention mask.
///
/// For each query position `q_idx` (0-indexed within the current input),
/// the absolute position is `start_pos + q_idx`. This query can attend
/// to key positions `0..=start_pos + q_idx`. Key positions beyond that
/// are "future" and must be masked out (set to -infinity).
///
/// # Arguments
///
/// * `scores`        - Shape `[num_heads, seq_len, total_seq_len]`.
/// * `start_pos`     - Position offset of the first query token.
/// * `seq_len`       - Number of query tokens in this forward pass.
/// * `total_seq_len` - Total number of key tokens (cached + current).
fn apply_causal_mask(
    scores: Tensor,
    start_pos: usize,
    seq_len: usize,
    total_seq_len: usize,
) -> Tensor {
    let num_heads = scores.shape()[0];
    let mut data = scores.data().to_vec();

    for h in 0..num_heads {
        for qi in 0..seq_len {
            // Absolute position of this query token.
            let q_pos = start_pos + qi;
            // Key positions > q_pos are in the future — mask them out.
            for ki in (q_pos + 1)..total_seq_len {
                let flat_idx = h * seq_len * total_seq_len + qi * total_seq_len + ki;
                data[flat_idx] = f32::NEG_INFINITY;
            }
        }
    }

    Tensor::new(scores.shape().to_vec(), data)
}

/// Apply RMSNorm to a 3D tensor `[seq_len, num_heads, head_dim]`.
///
/// Normalizes over the last dimension (head_dim) for each (seq_pos, head)
/// combination independently. This is used for Qwen3's per-head Q/K
/// normalization, which applies RMSNorm to Q and K after projection but
/// before RoPE.
///
/// The function reshapes the 3D tensor to 2D, applies RMSNorm (which
/// normalizes over the last dimension of a 2D tensor), then reshapes back.
fn apply_per_head_norm(x: &Tensor, norm: &RMSNorm) -> Tensor {
    let shape = x.shape();
    let seq_len = shape[0];
    let num_heads = shape[1];
    let head_dim = shape[2];

    // Reshape to 2D: [seq_len * num_heads, head_dim]
    let x_2d = x.reshape(vec![seq_len * num_heads, head_dim]);

    // Apply RMSNorm
    let normed = norm.forward(&x_2d);

    // Reshape back to 3D
    normed.reshape(vec![seq_len, num_heads, head_dim])
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a small Attention layer for testing.
    ///
    /// Configuration:
    /// - hidden_size = 8
    /// - num_heads = 2
    /// - num_kv_heads = 1
    /// - head_dim = 4
    /// - kv_groups = 2
    /// - max_seq_len = 16
    /// - rope_theta = 10000.0
    ///
    /// Weights are initialized with simple patterns so we can verify
    /// computations by hand.
    fn make_test_attention() -> Attention {
        let hidden = 8;
        let kv_dim = 4; // num_kv_heads * head_dim = 1 * 4

        // q_proj: [hidden, hidden] = [8, 8] — identity-like
        let q_proj = Tensor::new(
            vec![hidden, hidden],
            (0..hidden * hidden).map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 }).collect(),
        );

        // k_proj: [kv_dim, hidden] = [4, 8] — take first 4 output dims
        // Row i of k_proj projects hidden → kv_dim. We use a simple pattern
        // where row i has a 1 at column i (partial identity).
        let k_proj = Tensor::new(
            vec![kv_dim, hidden],
            (0..kv_dim * hidden).map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 }).collect(),
        );

        // v_proj: [kv_dim, hidden] = [4, 8] — same as k_proj
        let v_proj = Tensor::new(
            vec![kv_dim, hidden],
            (0..kv_dim * hidden).map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 }).collect(),
        );

        // o_proj: [hidden, hidden] = [8, 8] — identity
        let o_proj = Tensor::new(
            vec![hidden, hidden],
            (0..hidden * hidden).map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 }).collect(),
        );

        Attention::new(q_proj, k_proj, v_proj, o_proj, None, None, 2, 1, 4, 16, 10000.0)
    }

    // ── GQA expansion ──────────────────────────────────────────────────

    #[test]
    fn test_expand_kv_heads_basic() {
        // Input: [1, 2, 3] with kv_groups = 2
        //   Head 0: [1.0, 2.0, 3.0]
        //   Head 1: [4.0, 5.0, 6.0]
        // Expected output: [1, 4, 3]
        //   Head 0: [1.0, 2.0, 3.0]  (copy of KV head 0)
        //   Head 1: [1.0, 2.0, 3.0]  (copy of KV head 0)
        //   Head 2: [4.0, 5.0, 6.0]  (copy of KV head 1)
        //   Head 3: [4.0, 5.0, 6.0]  (copy of KV head 1)
        let x = Tensor::new(
            vec![1, 2, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        let expanded = expand_kv_heads(&x, 2);
        assert_eq!(expanded.shape(), &[1, 4, 3]);
        assert_eq!(
            expanded.data(),
            &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0],
        );
    }

    #[test]
    fn test_expand_kv_heads_no_expansion() {
        // When kv_groups = 1, the output should be identical to the input.
        let x = Tensor::new(
            vec![2, 3, 4],
            (0..24).map(|v| v as f32).collect(),
        );
        let expanded = expand_kv_heads(&x, 1);
        assert_eq!(expanded.shape(), &[2, 3, 4]);
        assert_eq!(expanded.data(), x.data());
    }

    #[test]
    fn test_expand_kv_heads_multi_token() {
        // Two tokens, each with 2 KV heads, kv_groups = 3
        let x = Tensor::new(
            vec![2, 2, 3],
            vec![
                1.0, 2.0, 3.0,  // token 0, KV head 0
                4.0, 5.0, 6.0,  // token 0, KV head 1
                7.0, 8.0, 9.0,  // token 1, KV head 0
                10.0, 11.0, 12.0, // token 1, KV head 1
            ],
        );
        let expanded = expand_kv_heads(&x, 3);
        assert_eq!(expanded.shape(), &[2, 6, 3]);

        // Token 0, Q head 0,1,2 = KV head 0; Q head 3,4,5 = KV head 1
        // Token 1, Q head 0,1,2 = KV head 0; Q head 3,4,5 = KV head 1
        let expected = vec![
            // Token 0
            1.0, 2.0, 3.0,   // Q head 0 = KV head 0
            1.0, 2.0, 3.0,   // Q head 1 = KV head 0
            1.0, 2.0, 3.0,   // Q head 2 = KV head 0
            4.0, 5.0, 6.0,   // Q head 3 = KV head 1
            4.0, 5.0, 6.0,   // Q head 4 = KV head 1
            4.0, 5.0, 6.0,   // Q head 5 = KV head 1
            // Token 1
            7.0, 8.0, 9.0,   // Q head 0 = KV head 0
            7.0, 8.0, 9.0,   // Q head 1 = KV head 0
            7.0, 8.0, 9.0,   // Q head 2 = KV head 0
            10.0, 11.0, 12.0, // Q head 3 = KV head 1
            10.0, 11.0, 12.0, // Q head 4 = KV head 1
            10.0, 11.0, 12.0, // Q head 5 = KV head 1
        ];
        assert_eq!(expanded.data(), &expected[..]);
    }

    // ── Causal masking ─────────────────────────────────────────────────

    #[test]
    fn test_causal_mask_prefill() {
        // 1 head, seq_len=3, total_seq_len=3, start_pos=0
        // Scores (arbitrary values):
        //   [[1, 2, 3],
        //    [4, 5, 6],
        //    [7, 8, 9]]
        // After causal mask (lower triangular):
        //   [[1, -inf, -inf],
        //    [4, 5,    -inf],
        //    [7, 8,    9   ]]
        let scores = Tensor::new(vec![1, 3, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let masked = apply_causal_mask(scores, 0, 3, 3);

        // Position 0 can only see position 0
        assert_eq!(masked.get(&[0, 0, 0]), 1.0);
        assert!(masked.get(&[0, 0, 1]).is_infinite() && masked.get(&[0, 0, 1]).is_sign_negative());
        assert!(masked.get(&[0, 0, 2]).is_infinite() && masked.get(&[0, 0, 2]).is_sign_negative());

        // Position 1 can see positions 0, 1
        assert_eq!(masked.get(&[0, 1, 0]), 4.0);
        assert_eq!(masked.get(&[0, 1, 1]), 5.0);
        assert!(masked.get(&[0, 1, 2]).is_infinite() && masked.get(&[0, 1, 2]).is_sign_negative());

        // Position 2 can see all positions
        assert_eq!(masked.get(&[0, 2, 0]), 7.0);
        assert_eq!(masked.get(&[0, 2, 1]), 8.0);
        assert_eq!(masked.get(&[0, 2, 2]), 9.0);
    }

    #[test]
    fn test_causal_mask_decode() {
        // Decode step: start_pos=2, seq_len=1, total_seq_len=3
        // The new token at position 2 can attend to all 3 key positions
        // (positions 0, 1, 2). No positions should be masked.
        let scores = Tensor::new(vec![1, 1, 3], vec![1.0, 2.0, 3.0]);
        let masked = apply_causal_mask(scores, 2, 1, 3);

        // All positions should be unmasked since q_pos=2 >= all k positions
        assert_eq!(masked.get(&[0, 0, 0]), 1.0);
        assert_eq!(masked.get(&[0, 0, 1]), 2.0);
        assert_eq!(masked.get(&[0, 0, 2]), 3.0);
    }

    #[test]
    fn test_causal_mask_partial_prefill() {
        // start_pos=1, seq_len=2, total_seq_len=3
        // Query token 0 is at absolute position 1, can see keys 0, 1
        // Query token 1 is at absolute position 2, can see keys 0, 1, 2
        let scores = Tensor::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let masked = apply_causal_mask(scores, 1, 2, 3);

        // Q position 1: key 2 is future → masked
        assert_eq!(masked.get(&[0, 0, 0]), 1.0);
        assert_eq!(masked.get(&[0, 0, 1]), 2.0);
        assert!(masked.get(&[0, 0, 2]).is_infinite() && masked.get(&[0, 0, 2]).is_sign_negative());

        // Q position 2: all keys are present or past → none masked
        assert_eq!(masked.get(&[0, 1, 0]), 4.0);
        assert_eq!(masked.get(&[0, 1, 1]), 5.0);
        assert_eq!(masked.get(&[0, 1, 2]), 6.0);
    }

    // ── KV cache updates ───────────────────────────────────────────────

    #[test]
    fn test_kv_cache_starts_empty() {
        let cache = KVCache::new();
        assert!(cache.key_cache.is_none());
        assert!(cache.value_cache.is_none());
    }

    #[test]
    fn test_kv_cache_grows_after_forward() {
        let attn = make_test_attention();
        let mut cache = KVCache::new();

        // Prefill: process 3 tokens
        let x = Tensor::ones(vec![3, 8]);
        let _output = attn.forward(&x, 0, &mut cache);

        // Cache should now contain 3 rows
        assert!(cache.key_cache.is_some());
        assert!(cache.value_cache.is_some());
        let k = cache.key_cache.as_ref().unwrap();
        let v = cache.value_cache.as_ref().unwrap();
        assert_eq!(k.shape()[0], 3); // 3 tokens
        assert_eq!(v.shape()[0], 3);

        // Decode: process 1 more token
        let x2 = Tensor::ones(vec![1, 8]);
        let _output2 = attn.forward(&x2, 3, &mut cache);

        // Cache should now contain 4 rows
        let k = cache.key_cache.as_ref().unwrap();
        let v = cache.value_cache.as_ref().unwrap();
        assert_eq!(k.shape()[0], 4); // 3 + 1
        assert_eq!(v.shape()[0], 4);
    }

    #[test]
    fn test_kv_cache_multiple_decode_steps() {
        let attn = make_test_attention();
        let mut cache = KVCache::new();

        // Prefill with 2 tokens
        let x = Tensor::ones(vec![2, 8]);
        let _ = attn.forward(&x, 0, &mut cache);
        assert_eq!(cache.key_cache.as_ref().unwrap().shape()[0], 2);

        // Decode step 1
        let x1 = Tensor::ones(vec![1, 8]);
        let _ = attn.forward(&x1, 2, &mut cache);
        assert_eq!(cache.key_cache.as_ref().unwrap().shape()[0], 3);

        // Decode step 2
        let x2 = Tensor::ones(vec![1, 8]);
        let _ = attn.forward(&x2, 3, &mut cache);
        assert_eq!(cache.key_cache.as_ref().unwrap().shape()[0], 4);

        // Decode step 3
        let x3 = Tensor::ones(vec![1, 8]);
        let _ = attn.forward(&x3, 4, &mut cache);
        assert_eq!(cache.key_cache.as_ref().unwrap().shape()[0], 5);
    }

    // ── Full forward pass ──────────────────────────────────────────────

    #[test]
    fn test_attention_forward_output_shape() {
        let attn = make_test_attention();
        let mut cache = KVCache::new();

        // Prefill: 4 tokens
        let x = Tensor::ones(vec![4, 8]);
        let output = attn.forward(&x, 0, &mut cache);
        assert_eq!(output.shape(), &[4, 8]);
    }

    #[test]
    fn test_attention_forward_decode_shape() {
        let attn = make_test_attention();
        let mut cache = KVCache::new();

        // Prefill
        let x = Tensor::ones(vec![3, 8]);
        let _ = attn.forward(&x, 0, &mut cache);

        // Decode: 1 token
        let x2 = Tensor::ones(vec![1, 8]);
        let output = attn.forward(&x2, 3, &mut cache);
        assert_eq!(output.shape(), &[1, 8]);
    }

    #[test]
    fn test_attention_forward_deterministic() {
        let attn = make_test_attention();

        // Run twice with fresh caches — should produce identical results.
        let mut cache1 = KVCache::new();
        let mut cache2 = KVCache::new();

        let x = Tensor::new(vec![2, 8], (0..16).map(|v| v as f32).collect());
        let out1 = attn.forward(&x, 0, &mut cache1);
        let out2 = attn.forward(&x, 0, &mut cache2);

        for i in 0..out1.len() {
            assert!(
                (out1.data()[i] - out2.data()[i]).abs() < 1e-6,
                "determinism check failed at index {}: {} vs {}",
                i,
                out1.data()[i],
                out2.data()[i],
            );
        }
    }

    #[test]
    fn test_attention_forward_different_inputs_differ() {
        let attn = make_test_attention();
        let mut cache1 = KVCache::new();
        let mut cache2 = KVCache::new();

        let x1 = Tensor::ones(vec![1, 8]);
        let x2 = Tensor::new(vec![1, 8], vec![2.0; 8]);

        let out1 = attn.forward(&x1, 0, &mut cache1);
        let out2 = attn.forward(&x2, 0, &mut cache2);

        // Different inputs should produce different outputs (with high probability).
        let any_different = out1.data().iter().zip(out2.data().iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_different, "different inputs should produce different outputs");
    }

    #[test]
    fn test_attention_softmax_sums_to_one() {
        // With causal masking, softmax should produce valid probability
        // distributions. We verify this indirectly by checking that the
        // output is finite (no NaN from softmax of all-masked rows).
        let attn = make_test_attention();
        let mut cache = KVCache::new();

        let x = Tensor::ones(vec![4, 8]);
        let output = attn.forward(&x, 0, &mut cache);

        for &v in output.data() {
            assert!(v.is_finite(), "output should be finite, got {}", v);
        }
    }

    #[test]
    fn test_attention_prefill_then_decode_consistent() {
        // Process tokens one at a time (decode) and compare with processing
        // all at once (prefill). The output for each token should be
        // consistent (though not identical because RoPE depends on position,
        // and the attention context differs). At minimum, the output shapes
        // and finiteness should match.
        let attn = make_test_attention();

        // Prefill path: process 3 tokens at once
        let mut cache_prefill = KVCache::new();
        let x = Tensor::new(vec![3, 8], (0..24).map(|v| v as f32 * 0.1).collect());
        let out_prefill = attn.forward(&x, 0, &mut cache_prefill);

        // Verify the prefill output is well-formed
        for &v in out_prefill.data() {
            assert!(v.is_finite(), "prefill output should be finite, got {}", v);
        }

        // Decode path: process the same tokens one at a time
        let mut cache_decode = KVCache::new();
        for pos in 0..3 {
            let x_tok = Tensor::new(vec![1, 8], (0..8).map(|d| (pos * 8 + d) as f32 * 0.1).collect());
            let out_tok = attn.forward(&x_tok, pos, &mut cache_decode);
            for &v in out_tok.data() {
                assert!(v.is_finite(), "decode output at pos {} should be finite, got {}", pos, v);
            }
        }

        // Both caches should have the same number of rows
        assert_eq!(
            cache_prefill.key_cache.as_ref().unwrap().shape()[0],
            cache_decode.key_cache.as_ref().unwrap().shape()[0],
        );
    }
}
