//! Simple N-dimensional tensor with math operations.
//!
//! This module implements a `Tensor` struct that stores f32 data in a flat
//! `Vec<f32>` buffer using **row-major** (C-order) layout. In row-major
//! storage, the last index varies fastest: for a shape [M, N], element
//! [i, j] is stored at position `i * N + j`.
//!
//! All operations use plain Rust loops — no external math crates — making
//! the code easy to read and understand, which is the goal of this
//! educational project.

// ─────────────────────────────────────────────────────────────────────────────
// Tensor struct
// ─────────────────────────────────────────────────────────────────────────────

/// An N-dimensional tensor of `f32` values stored in row-major order.
///
/// # Storage layout
///
/// Data is kept in a single flat `Vec<f32>`.  For a tensor with shape
/// `[d0, d1, …, d_{n-1}]`, the element at indices `[i0, i1, …, i_{n-1}]`
/// lives at flat offset:
///
/// ```text
/// offset = i0 * (d1 * d2 * … * d_{n-1})
///        + i1 * (d2 * … * d_{n-1})
///        + …
///        + i_{n-1}
/// ```
///
/// This is standard C / row-major ordering.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// The size of each dimension, e.g. `[3, 4]` for a 3×4 matrix.
    shape: Vec<usize>,
    /// All elements in row-major (C) order.
    data: Vec<f32>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Constructors
// ─────────────────────────────────────────────────────────────────────────────

impl Tensor {
    /// Create a new tensor from a shape and a flat data vector.
    ///
    /// The length of `data` must equal the product of all dimensions in
    /// `shape`.  Data is assumed to already be in row-major order.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != shape.iter().product()`.
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "Tensor::new: data length ({}) does not match shape {:?} (expected {} elements)",
            data.len(),
            shape,
            expected,
        );
        Self { shape, data }
    }

    /// Create a tensor filled with zeros.
    ///
    /// ```ignore
    /// let t = Tensor::zeros(vec![2, 3]); // 2×3 matrix of 0.0
    /// ```
    pub fn zeros(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; len],
        }
    }

    /// Create a tensor filled with ones.
    ///
    /// ```ignore
    /// let t = Tensor::ones(vec![3]); // vector of 1.0
    /// ```
    pub fn ones(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        Self {
            shape,
            data: vec![1.0; len],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Accessors
// ─────────────────────────────────────────────────────────────────────────────

impl Tensor {
    /// Return the shape as a slice, e.g. `&[3, 4]`.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return a slice over the raw flat data.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Return a mutable reference to the flat data vector.
    ///
    /// Use this when you need to modify elements in-place by flat index.
    pub fn data_mut(&mut self) -> &mut Vec<f32> {
        &mut self.data
    }

    /// Number of dimensions (rank). A scalar would be 0, a vector 1,
    /// a matrix 2, etc.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements (product of all shape dimensions).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True when the tensor has zero elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shape manipulation
// ─────────────────────────────────────────────────────────────────────────────

impl Tensor {
    /// Return a new tensor with the same data but a different shape.
    ///
    /// This is a "view" operation — the data buffer is cloned, not
    /// reinterpreted in place, so it is always safe.  The total number
    /// of elements must remain the same.
    ///
    /// # Panics
    ///
    /// Panics if the product of `new_shape` differs from the current
    /// number of elements.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let expected: usize = new_shape.iter().product();
        assert_eq!(
            expected,
            self.data.len(),
            "reshape: new shape {:?} has {} elements but tensor has {}",
            new_shape,
            expected,
            self.data.len(),
        );
        Self {
            shape: new_shape,
            data: self.data.clone(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Indexing helpers
// ─────────────────────────────────────────────────────────────────────────────

impl Tensor {
    /// Convert a multi-dimensional index to a flat offset.
    ///
    /// For shape `[d0, d1, …, d_{n-1}]` and indices `[i0, i1, …, i_{n-1}]`:
    ///
    /// ```text
    /// offset = i0 * (d1 * d2 * … * d_{n-1})
    ///        + i1 * (d2 * … * d_{n-1})
    ///        + …
    ///        + i_{n-1}
    /// ```
    fn flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(
            indices.len(),
            self.shape.len(),
            "index dimension mismatch: got {} indices for {:?} tensor",
            indices.len(),
            self.shape,
        );
        let mut offset = 0usize;
        let mut stride = 1usize;
        // Walk dimensions from right to left accumulating the stride.
        for dim in (0..self.shape.len()).rev() {
            assert!(
                indices[dim] < self.shape[dim],
                "index {:?} out of bounds for shape {:?}",
                indices,
                self.shape,
            );
            offset += indices[dim] * stride;
            stride *= self.shape[dim];
        }
        offset
    }

    /// Get the value at a multi-dimensional index.
    ///
    /// ```ignore
    /// let val = tensor.get(&[row, col]);
    /// ```
    pub fn get(&self, indices: &[usize]) -> f32 {
        self.data[self.flat_index(indices)]
    }

    /// Set the value at a multi-dimensional index.
    ///
    /// ```ignore
    /// tensor.set(&[row, col], 3.14);
    /// ```
    pub fn set(&mut self, indices: &[usize], value: f32) {
        let idx = self.flat_index(indices);
        self.data[idx] = value;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Row operations (2-D focused — what the transformer needs)
// ─────────────────────────────────────────────────────────────────────────────

impl Tensor {
    /// Extract a single row from a 2-D tensor as an owned 1-D tensor.
    ///
    /// For a tensor with shape `[M, N]`, `row(i)` returns a 1-D tensor
    /// of shape `[N]` containing a copy of row `i`.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not 2-D or `row` is out of bounds.
    pub fn row(&self, row: usize) -> Tensor {
        assert_eq!(
            self.ndim(),
            2,
            "row() requires a 2-D tensor, got {:?}",
            self.shape,
        );
        let cols = self.shape[1];
        assert!(row < self.shape[0], "row {} out of bounds for shape {:?}", row, self.shape);
        let start = row * cols;
        Tensor::new(vec![cols], self.data[start..start + cols].to_vec())
    }

    /// Extract rows `[start..end)` from a 2-D tensor as an owned 2-D tensor.
    ///
    /// For a tensor with shape `[M, N]`, `rows(a, b)` returns shape
    /// `[b - a, N]`.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not 2-D or the range is out of bounds.
    pub fn rows(&self, start: usize, end: usize) -> Tensor {
        assert_eq!(
            self.ndim(),
            2,
            "rows() requires a 2-D tensor, got {:?}",
            self.shape,
        );
        assert!(start <= end, "rows: start ({}) > end ({})", start, end);
        assert!(end <= self.shape[0], "rows: end ({}) out of bounds for shape {:?}", end, self.shape);
        let cols = self.shape[1];
        let flat_start = start * cols;
        let flat_end = end * cols;
        Tensor::new(
            vec![end - start, cols],
            self.data[flat_start..flat_end].to_vec(),
        )
    }

    /// Transpose a 2-D tensor, swapping rows and columns.
    ///
    /// For a tensor of shape `[M, N]`, the result has shape `[N, M]` where
    /// element `[i, j]` of the output equals element `[j, i]` of the input.
    ///
    /// This is essential for implementing linear projections when weights
    /// are stored as `[out_features, in_features]` (the safetensors
    /// convention): the forward pass is `x · W^T`, which we compute as
    /// `x.matmul(&weight.transpose_2d())`.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not 2-D.
    pub fn transpose_2d(&self) -> Tensor {
        assert_eq!(
            self.ndim(),
            2,
            "transpose_2d: requires a 2-D tensor, got shape {:?}",
            self.shape,
        );
        let m = self.shape[0];
        let n = self.shape[1];
        let mut result = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                // output[j][i] = input[i][j]
                result[j * m + i] = self.data[i * n + j];
            }
        }
        Tensor::new(vec![n, m], result)
    }

    /// Concatenate two 2-D tensors along the row dimension (dim 0).
    ///
    /// Both tensors must have the same number of columns. The result has
    /// shape `[a.rows + b.rows, cols]`.
    ///
    /// This is used for KV cache updates: when we have cached K/V tensors
    /// from previous steps and want to append the new K/V rows.
    ///
    /// # Panics
    ///
    /// Panics if either tensor is not 2-D or their column counts differ.
    pub fn stack_rows(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.ndim(),
            2,
            "stack_rows: left operand must be 2-D, got shape {:?}",
            self.shape,
        );
        assert_eq!(
            other.ndim(),
            2,
            "stack_rows: right operand must be 2-D, got shape {:?}",
            other.shape,
        );
        assert_eq!(
            self.shape[1],
            other.shape[1],
            "stack_rows: column count mismatch: {} vs {}",
            self.shape[1],
            other.shape[1],
        );
        let cols = self.shape[1];
        let new_rows = self.shape[0] + other.shape[0];
        let mut result = Vec::with_capacity(new_rows * cols);
        result.extend_from_slice(&self.data);
        result.extend_from_slice(&other.data);
        Tensor::new(vec![new_rows, cols], result)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Math operations
// ─────────────────────────────────────────────────────────────────────────────

impl Tensor {
    /// 2-D matrix multiply: `[M, K] × [K, N] → [M, N]`.
    ///
    /// This is the single most important operation in a transformer.
    /// Every linear projection (Q, K, V, output) and every attention
    /// weight computation is a matrix multiply.
    ///
    /// # Algorithm (triple-loop)
    ///
    /// For each output element `C[i][j]` we compute the dot product of
    /// row `i` from the left matrix and column `j` from the right
    /// matrix:
    ///
    /// ```text
    /// C[i][j] = Σ_k  A[i][k] * B[k][j]
    /// ```
    ///
    /// We iterate `i` over rows of `A`, `j` over columns of `B`, and
    /// `k` over the shared dimension.  This is the straightforward
    /// O(MKN) textbook algorithm — not the fastest, but the clearest.
    ///
    /// # Panics
    ///
    /// Panics if either argument is not 2-D or the inner dimensions
    /// don't match (`K` must agree).
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        // --- Validate dimensions ---
        assert_eq!(self.ndim(), 2, "matmul: left operand must be 2-D, got {:?}", self.shape);
        assert_eq!(other.ndim(), 2, "matmul: right operand must be 2-D, got {:?}", other.shape);

        let m = self.shape[0];  // rows of A
        let k = self.shape[1];  // cols of A == rows of B
        let n = other.shape[1]; // cols of B

        assert_eq!(
            other.shape[0], k,
            "matmul: inner dimensions mismatch: [{}x{}] × [{}x{}]",
            m, k, other.shape[0], n,
        );

        // --- Allocate output ---
        let mut result = vec![0.0f32; m * n];

        // --- Triple loop ---
        // For every row i of the left matrix …
        for i in 0..m {
            // … and every column j of the right matrix …
            for j in 0..n {
                // … compute the dot product over the shared dimension k.
                let mut sum = 0.0f32;
                for ki in 0..k {
                    // A[i][k] is at flat index i*k + ki  (row-major)
                    // B[k][j] is at flat index ki*n + j
                    sum += self.data[i * k + ki] * other.data[ki * n + j];
                }
                // C[i][j] is at flat index i*n + j
                result[i * n + j] = sum;
            }
        }

        Tensor::new(vec![m, n], result)
    }

    /// Element-wise addition with broadcasting support.
    ///
    /// Three cases are handled:
    ///
    /// 1. **Same shape** — straightforward element-wise add.
    /// 2. **Scalar + Tensor** — the left operand has shape `[]` or `[1]`
    ///    (a single number); it is added to every element.
    /// 3. **Row + Matrix** — the left operand has shape `[1, N]` or `[N]`
    ///    and the right has shape `[M, N]`; the row is added to every
    ///    row of the matrix.
    ///
    /// # Panics
    ///
    /// Panics if the shapes are incompatible.
    pub fn add(&self, other: &Tensor) -> Tensor {
        // Case 1: identical shapes — just zip and add.
        if self.shape == other.shape {
            let data: Vec<f32> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect();
            return Tensor::new(self.shape.clone(), data);
        }

        // Case 2: scalar + tensor (scalar is the left operand).
        // A scalar is a 0-D or 1-element tensor.
        if self.data.len() == 1 {
            let s = self.data[0];
            let data: Vec<f32> = other.data.iter().map(|v| s + v).collect();
            return Tensor::new(other.shape.clone(), data);
        }

        // Case 2b: tensor + scalar (scalar is the right operand).
        if other.data.len() == 1 {
            let s = other.data[0];
            let data: Vec<f32> = self.data.iter().map(|v| v + s).collect();
            return Tensor::new(self.shape.clone(), data);
        }

        // Case 3: row + matrix broadcasting.
        // Left operand [1, N] or [N] + right operand [M, N].
        if other.ndim() == 2 {
            let m = other.shape[0];
            let n = other.shape[1];

            // Left is a 1-D vector of length N.
            if self.ndim() == 1 && self.shape[0] == n {
                let mut result = vec![0.0f32; m * n];
                for i in 0..m {
                    for j in 0..n {
                        result[i * n + j] = self.data[j] + other.data[i * n + j];
                    }
                }
                return Tensor::new(vec![m, n], result);
            }

            // Left is a [1, N] matrix.
            if self.ndim() == 2 && self.shape[0] == 1 && self.shape[1] == n {
                let mut result = vec![0.0f32; m * n];
                for i in 0..m {
                    for j in 0..n {
                        result[i * n + j] = self.data[j] + other.data[i * n + j];
                    }
                }
                return Tensor::new(vec![m, n], result);
            }
        }

        // If we reach here, the broadcast is not one we support.
        panic!(
            "add: incompatible shapes {:?} and {:?}",
            self.shape, other.shape
        );
    }

    /// Multiply every element by a scalar.
    ///
    /// ```ignore
    /// let doubled = tensor.mul_scalar(2.0);
    /// ```
    pub fn mul_scalar(&self, scalar: f32) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|v| v * scalar).collect();
        Tensor::new(self.shape.clone(), data)
    }

    /// Element-wise multiplication (Hadamard product).
    ///
    /// Both tensors must have the **same shape**.  Each output element
    /// is `self[i] * other[i]`.
    ///
    /// # Panics
    ///
    /// Panics if the shapes differ.
    pub fn mul_elementwise(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape, other.shape,
            "mul_elementwise: shape mismatch {:?} vs {:?}",
            self.shape, other.shape,
        );
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Tensor::new(self.shape.clone(), data)
    }

    /// SiLU (Sigmoid Linear Unit) activation: `x * sigmoid(x)`.
    ///
    /// SiLU is defined as:
    ///
    /// ```text
    /// SiLU(x) = x * σ(x) = x / (1 + e^(-x))
    /// ```
    ///
    /// This is the activation used in the SwiGLU feed-forward block of
    /// the Qwen3 model.
    pub fn silu(&self) -> Tensor {
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| {
                // σ(x) = 1 / (1 + exp(-x))
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                x * sigmoid
            })
            .collect();
        Tensor::new(self.shape.clone(), data)
    }

    /// Numerically stable softmax along a given dimension.
    ///
    /// Softmax converts a vector of real numbers into a probability
    /// distribution (all values in [0, 1] and sum to 1):
    ///
    /// ```text
    /// softmax(x_i) = exp(x_i - max) / Σ_j exp(x_j - max)
    /// ```
    ///
    /// We subtract the maximum value first to prevent overflow in
    /// `exp()`.  This is the standard "numerically stable" trick.
    ///
    /// For a 2-D tensor, `dim=0` means softmax down each column and
    /// `dim=1` means softmax across each row.  The transformer uses
    /// `dim=1` (softmax over the attention scores in each row).
    ///
    /// # Panics
    ///
    /// Panics if `dim` is out of bounds.
    pub fn softmax(&self, dim: usize) -> Tensor {
        assert!(dim < self.shape.len(), "softmax: dim {} out of bounds for shape {:?}", dim, self.shape);

        let mut result = self.data.clone();

        if self.ndim() == 1 {
            // --- 1-D: single vector ---
            // Step 1: find the max for numerical stability.
            let max = self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            // Step 2: subtract max and exponentiate.
            let exps: Vec<f32> = self.data.iter().map(|&x| (x - max).exp()).collect();
            // Step 3: sum the exponentials.
            let sum: f32 = exps.iter().sum();
            // Step 4: normalize so each element is exp / sum.
            for (i, &e) in exps.iter().enumerate() {
                result[i] = e / sum;
            }
        } else if self.ndim() == 2 {
            let rows = self.shape[0];
            let cols = self.shape[1];

            if dim == 1 {
                // Softmax across each row (the common case for attention).
                for i in 0..rows {
                    let row_start = i * cols;

                    // Step 1: max of the row.
                    let mut max = f32::NEG_INFINITY;
                    for j in 0..cols {
                        let v = self.data[row_start + j];
                        if v > max {
                            max = v;
                        }
                    }

                    // Step 2: exp(x - max) for each element in the row.
                    let mut exps = vec![0.0f32; cols];
                    let mut sum = 0.0f32;
                    for j in 0..cols {
                        let e = (self.data[row_start + j] - max).exp();
                        exps[j] = e;
                        sum += e;
                    }

                    // Step 3: normalize.
                    for j in 0..cols {
                        result[row_start + j] = exps[j] / sum;
                    }
                }
            } else {
                // dim == 0: softmax down each column.
                for j in 0..cols {
                    // Step 1: max of the column.
                    let mut max = f32::NEG_INFINITY;
                    for i in 0..rows {
                        let v = self.data[i * cols + j];
                        if v > max {
                            max = v;
                        }
                    }

                    // Step 2: exp(x - max).
                    let mut exps = vec![0.0f32; rows];
                    let mut sum = 0.0f32;
                    for i in 0..rows {
                        let e = (self.data[i * cols + j] - max).exp();
                        exps[i] = e;
                        sum += e;
                    }

                    // Step 3: normalize.
                    for i in 0..rows {
                        result[i * cols + j] = exps[i] / sum;
                    }
                }
            }
        } else {
            // General N-D case: iterate over all "lanes" along `dim`.
            // This is rarely needed in the transformer but provided for
            // completeness.
            //
            // The outer dimensions (before `dim`) and inner dimensions
            // (after `dim`) define how many independent softmax lanes
            // there are and how elements are strided.
            let outer: usize = if dim == 0 { 1 } else { self.shape[..dim].iter().product() };
            let inner: usize = if dim + 1 == self.ndim() { 1 } else { self.shape[dim + 1..].iter().product() };
            let dim_size = self.shape[dim];
            // Stride: number of flat elements to skip to move one step
            // along `dim`.
            let stride = inner;

            for o in 0..outer {
                for i in 0..inner {
                    // Base flat index for this lane.
                    let base = o * dim_size * stride + i;

                    // Step 1: find max in this lane.
                    let mut max = f32::NEG_INFINITY;
                    for d in 0..dim_size {
                        let v = self.data[base + d * stride];
                        if v > max {
                            max = v;
                        }
                    }

                    // Step 2: exponentiate and accumulate.
                    let mut exps = vec![0.0f32; dim_size];
                    let mut sum = 0.0f32;
                    for d in 0..dim_size {
                        let e = (self.data[base + d * stride] - max).exp();
                        exps[d] = e;
                        sum += e;
                    }

                    // Step 3: normalize.
                    for d in 0..dim_size {
                        result[base + d * stride] = exps[d] / sum;
                    }
                }
            }
        }

        Tensor::new(self.shape.clone(), result)
    }

    /// RMSNorm (Root Mean Square Normalization) along the last dimension.
    ///
    /// RMSNorm normalizes each row by its root-mean-square value and
    /// then scales by a learned weight:
    ///
    /// ```text
    /// RMSNorm(x) = weight * (x / sqrt(mean(x^2) + eps))
    /// ```
    ///
    /// This is the normalization used by Qwen3 (and most modern LLMs)
    /// instead of LayerNorm, because it is slightly cheaper to compute
    /// (no mean subtraction needed, only the square mean).
    ///
    /// The `weight` tensor must be 1-D with length equal to the last
    /// dimension of `self`.  `eps` is a small constant (e.g. 1e-6) to
    /// avoid division by zero.
    ///
    /// # Panics
    ///
    /// Panics if `weight` is not 1-D or its length doesn't match the
    /// last dimension of `self`.
    pub fn rms_norm(&self, weight: &Tensor, eps: f32) -> Tensor {
        assert_eq!(weight.ndim(), 1, "rms_norm: weight must be 1-D, got {:?}", weight.shape);

        let last_dim = self.shape[self.ndim() - 1];
        assert_eq!(
            weight.shape[0], last_dim,
            "rms_norm: weight length {} must match last dim {}",
            weight.shape[0], last_dim,
        );

        // Number of "rows" — everything except the last dimension.
        let num_rows: usize = if self.ndim() > 1 {
            self.shape[..self.ndim() - 1].iter().product()
        } else {
            1
        };

        let mut result = vec![0.0f32; self.data.len()];

        for r in 0..num_rows {
            let row_start = r * last_dim;

            // Step 1: compute mean of squares for this row.
            let mut sum_sq = 0.0f32;
            for j in 0..last_dim {
                let v = self.data[row_start + j];
                sum_sq += v * v;
            }
            let mean_sq = sum_sq / (last_dim as f32);

            // Step 2: reciprocal of the root mean square.
            //         rms = sqrt(mean_sq + eps)
            //         1/rms = 1 / sqrt(mean_sq + eps)
            let rms_inv = 1.0 / (mean_sq + eps).sqrt();

            // Step 3: normalize and scale by weight.
            for j in 0..last_dim {
                result[row_start + j] = self.data[row_start + j] * rms_inv * weight.data[j];
            }
        }

        Tensor::new(self.shape.clone(), result)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────────────────────────────────────

impl Tensor {
    /// Convert a 1-D or 2-D tensor to a nested `Vec` for debugging.
    ///
    /// - 1-D → `Vec<Vec<f32>>` with a single inner vec.
    /// - 2-D → `Vec<Vec<f32>>` where each inner vec is a row.
    ///
    /// # Panics
    ///
    /// Panics if the tensor has more than 2 dimensions.
    pub fn to_vec2d(&self) -> Vec<Vec<f32>> {
        if self.ndim() == 1 {
            // Wrap the flat data in a single row.
            vec![self.data.clone()]
        } else if self.ndim() == 2 {
            let rows = self.shape[0];
            let cols = self.shape[1];
            let mut out = Vec::with_capacity(rows);
            for i in 0..rows {
                let start = i * cols;
                out.push(self.data[start..start + cols].to_vec());
            }
            out
        } else {
            panic!("to_vec2d: only 1-D and 2-D tensors supported, got shape {:?}", self.shape);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Creation ──────────────────────────────────────────────────────────

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(vec![2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.len(), 6);
        assert!(t.data().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_ones() {
        let t = Tensor::ones(vec![3]);
        assert_eq!(t.shape(), &[3]);
        assert!(t.data().iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_new() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "does not match shape")]
    fn test_new_wrong_length() {
        Tensor::new(vec![2, 3], vec![1.0, 2.0]); // only 2 elements, need 6
    }

    // ── Shape / reshape ───────────────────────────────────────────────────

    #[test]
    fn test_reshape() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = t.reshape(vec![3, 2]);
        assert_eq!(r.shape(), &[3, 2]);
        // Data is the same (just reinterpreted).
        assert_eq!(r.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_ndim_and_len() {
        let t = Tensor::zeros(vec![4, 5, 6]);
        assert_eq!(t.ndim(), 3);
        assert_eq!(t.len(), 120);
    }

    #[test]
    fn test_is_empty() {
        let t = Tensor::zeros(vec![0]);
        assert!(t.is_empty());
        let t = Tensor::zeros(vec![2, 3]);
        assert!(!t.is_empty());
    }

    // ── 2-D indexing and row extraction ───────────────────────────────────

    #[test]
    fn test_get_set_2d() {
        let mut t = Tensor::zeros(vec![3, 4]);
        t.set(&[1, 2], 7.5);
        assert_eq!(t.get(&[1, 2]), 7.5);
        // Other elements should still be 0.
        assert_eq!(t.get(&[0, 0]), 0.0);
    }

    #[test]
    fn test_row() {
        let t = Tensor::new(vec![2, 3], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        let r = t.row(1);
        assert_eq!(r.shape(), &[3]);
        assert_eq!(r.data(), &[40.0, 50.0, 60.0]);
    }

    #[test]
    fn test_rows() {
        let t = Tensor::new(vec![4, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let r = t.rows(1, 3);
        assert_eq!(r.shape(), &[2, 2]);
        assert_eq!(r.data(), &[3.0, 4.0, 5.0, 6.0]);
    }

    // ── Matrix multiply ───────────────────────────────────────────────────

    #[test]
    fn test_matmul_simple() {
        // A = [[1, 2, 3],
        //      [4, 5, 6]]   shape [2, 3]
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // B = [[7,  8],
        //      [9, 10],
        //      [11, 12]]   shape [3, 2]
        let b = Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

        // C = A × B
        // C[0][0] = 1*7 + 2*9  + 3*11 = 7+18+33  = 58
        // C[0][1] = 1*8 + 2*10 + 3*12 = 8+20+36  = 64
        // C[1][0] = 4*7 + 5*9  + 6*11 = 28+45+66 = 139
        // C[1][1] = 4*8 + 5*10 + 6*12 = 32+50+72 = 154
        let c = a.matmul(&b);
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_matmul_identity() {
        // 2×2 identity
        let i = Tensor::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);
        let a = Tensor::new(vec![2, 2], vec![3.0, 7.0, 1.0, 2.0]);
        let c = a.matmul(&i);
        assert_eq!(c.data(), &[3.0, 7.0, 1.0, 2.0]);
    }

    // ── Softmax ───────────────────────────────────────────────────────────

    #[test]
    fn test_softmax_1d_sums_to_one() {
        let t = Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let s = t.softmax(0);
        let sum: f32 = s.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax should sum to 1.0, got {}", sum);
    }

    #[test]
    fn test_softmax_2d_row() {
        // Each row should sum to 1.
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
        let s = t.softmax(1);
        for i in 0..2 {
            let row_sum: f32 = s.data()[i * 3..(i + 1) * 3].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-5, "row {} sums to {}", i, row_sum);
        }
        // Larger values should get larger probabilities.
        assert!(s.get(&[1, 2]) > s.get(&[1, 1]));
        assert!(s.get(&[1, 1]) > s.get(&[1, 0]));
    }

    #[test]
    fn test_softmax_stability() {
        // Very large values should not overflow.
        let t = Tensor::new(vec![3], vec![1000.0, 1001.0, 1002.0]);
        let s = t.softmax(0);
        let sum: f32 = s.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "numerically stable softmax should sum to 1.0, got {}", sum);
    }

    // ── SiLU ──────────────────────────────────────────────────────────────

    #[test]
    fn test_silu() {
        // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        let t = Tensor::new(vec![3], vec![0.0, 1.0, -1.0]);
        let s = t.silu();
        assert!((s.data()[0] - 0.0).abs() < 1e-6, "SiLU(0) = 0");
        // SiLU(1) = 1 * sigmoid(1) ≈ 1 * 0.7311 ≈ 0.7311
        assert!((s.data()[1] - 0.7311).abs() < 0.01, "SiLU(1) ≈ 0.7311");
        // SiLU(-1) = -1 * sigmoid(-1) ≈ -1 * 0.2689 ≈ -0.2689
        assert!((s.data()[2] - (-0.2689)).abs() < 0.01, "SiLU(-1) ≈ -0.2689");
    }

    // ── Add with broadcasting ─────────────────────────────────────────────

    #[test]
    fn test_add_same_shape() {
        let a = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = Tensor::new(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0]);
        let c = a.add(&b);
        assert_eq!(c.data(), &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_add_scalar_broadcast() {
        let scalar = Tensor::new(vec![1], vec![5.0]);
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let c = scalar.add(&t);
        assert_eq!(c.data(), &[6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn test_add_tensor_scalar() {
        let t = Tensor::new(vec![2], vec![1.0, 2.0]);
        let scalar = Tensor::new(vec![1], vec![10.0]);
        let c = t.add(&scalar);
        assert_eq!(c.data(), &[11.0, 12.0]);
    }

    #[test]
    fn test_add_row_to_matrix() {
        // row [1, 2, 3] broadcast to a 2×3 matrix
        let row = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]);
        let mat = Tensor::new(vec![2, 3], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        let c = row.add(&mat);
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.data(), &[11.0, 22.0, 33.0, 41.0, 52.0, 63.0]);
    }

    // ── mul_scalar and mul_elementwise ─────────────────────────────────────

    #[test]
    fn test_mul_scalar() {
        let t = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let c = t.mul_scalar(3.0);
        assert_eq!(c.data(), &[3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_mul_elementwise() {
        let a = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = Tensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        let c = a.mul_elementwise(&b);
        assert_eq!(c.data(), &[5.0, 12.0, 21.0, 32.0]);
    }

    // ── RMSNorm ───────────────────────────────────────────────────────────

    #[test]
    fn test_rms_norm() {
        // Single row: [3, 4, 0, 0]
        // mean_sq = (9 + 16 + 0 + 0) / 4 = 6.25
        // rms = sqrt(6.25 + 1e-6) ≈ 2.5
        // result = [3/2.5, 4/2.5, 0, 0] * weight
        let t = Tensor::new(vec![4], vec![3.0, 4.0, 0.0, 0.0]);
        let w = Tensor::new(vec![4], vec![1.0, 1.0, 1.0, 1.0]);
        let normed = t.rms_norm(&w, 1e-6);
        assert!((normed.data()[0] - 1.2).abs() < 1e-4, "got {}", normed.data()[0]);
        assert!((normed.data()[1] - 1.6).abs() < 1e-4, "got {}", normed.data()[1]);
        assert!((normed.data()[2]).abs() < 1e-4);
        assert!((normed.data()[3]).abs() < 1e-4);
    }

    // ── transpose_2d ──────────────────────────────────────────────────────

    #[test]
    fn test_transpose_2d_basic() {
        // [[1, 2, 3],
        //  [4, 5, 6]]  shape [2, 3]
        // Transpose:
        // [[1, 4],
        //  [2, 5],
        //  [3, 6]]  shape [3, 2]
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let tt = t.transpose_2d();
        assert_eq!(tt.shape(), &[3, 2]);
        assert_eq!(tt.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_2d_square() {
        let t = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let tt = t.transpose_2d();
        assert_eq!(tt.shape(), &[2, 2]);
        assert_eq!(tt.data(), &[1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_transpose_2d_double_is_identity() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let tt = t.transpose_2d().transpose_2d();
        assert_eq!(tt.shape(), t.shape());
        assert_eq!(tt.data(), t.data());
    }

    // ── stack_rows ──────────────────────────────────────────────────────

    #[test]
    fn test_stack_rows_basic() {
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::new(vec![1, 3], vec![7.0, 8.0, 9.0]);
        let c = a.stack_rows(&b);
        assert_eq!(c.shape(), &[3, 3]);
        assert_eq!(c.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_stack_rows_empty_left() {
        let a = Tensor::new(vec![0, 2], vec![]);
        let b = Tensor::new(vec![1, 2], vec![3.0, 4.0]);
        let c = a.stack_rows(&b);
        assert_eq!(c.shape(), &[1, 2]);
        assert_eq!(c.data(), &[3.0, 4.0]);
    }

    #[test]
    fn test_stack_rows_preserves_order() {
        let a = Tensor::new(vec![1, 2], vec![10.0, 20.0]);
        let b = Tensor::new(vec![1, 2], vec![30.0, 40.0]);
        let c = a.stack_rows(&b);
        assert_eq!(c.shape(), &[2, 2]);
        // First row from a, second row from b
        assert_eq!(c.data(), &[10.0, 20.0, 30.0, 40.0]);
    }

    // ── to_vec2d ──────────────────────────────────────────────────────────

    #[test]
    fn test_to_vec2d() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let v = t.to_vec2d();
        assert_eq!(v, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    }
}
