use cubecl::prelude::*;
use cubecl_cuda::CudaRuntime;
use std::sync::OnceLock;

type Rt = CudaRuntime;

/// Simple 1D element-wise addition kernel.
#[cube(launch_unchecked)]
fn add_one_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] + 1.0_f32;
    }
}

/// Naïve 2-D matrix multiply: C[M, N] = A[M, K] @ B[K, N].
/// Each unit computes one output element.
#[cube(launch_unchecked)]
fn matmul_naive_kernel(
    a: &Array<f32>,
    b: &Array<f32>,
    c: &mut Array<f32>,
    m: u32,
    n: u32,
    k: u32,
) {
    let row = UNIT_POS_X + CUBE_POS_X * CUBE_DIM_X;
    let col = UNIT_POS_Y + CUBE_POS_Y * CUBE_DIM_Y;
    if row < m && col < n {
        let mut sum = 0.0_f32;
        for i in 0..k {
            sum = sum + a[(row * k + i) as usize] * b[(i * n + col) as usize];
        }
        c[(row * n + col) as usize] = sum;
    }
}

pub struct GpuContext {
    client: ComputeClient<Rt>,
}

impl GpuContext {
    pub fn new() -> Self {
        let device = Default::default();
        let client = Rt::client(&device);
        Self { client }
    }

    pub fn client(&self) -> &ComputeClient<Rt> {
        &self.client
    }

    pub fn upload(&self, data: &[f32]) -> cubecl::server::Handle {
        self.client.create_from_slice(f32::as_bytes(data))
    }

    pub fn alloc_f32s(&self, n: usize) -> cubecl::server::Handle {
        self.client.empty(n * core::mem::size_of::<f32>())
    }

    pub fn download(&self, handle: cubecl::server::Handle) -> Vec<f32> {
        let bytes = self.client.read_one(handle).unwrap();
        f32::from_bytes(&bytes).to_vec()
    }

    pub fn add_one(&self, input: &[f32]) -> Vec<f32> {
        let input_h = self.upload(input);
        let output_h = self.alloc_f32s(input.len());

        unsafe {
            add_one_kernel::launch_unchecked::<Rt>(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(input.len() as u32),
                ArrayArg::from_raw_parts(input_h, input.len()),
                ArrayArg::from_raw_parts(output_h.clone(), input.len()),
            )
        };

        self.download(output_h)
    }

    /// Returns the global `GpuContext` singleton, initializing it on first access.
    pub fn global() -> &'static Self {
        static GPU: OnceLock<GpuContext> = OnceLock::new();
        GPU.get_or_init(|| GpuContext::new())
    }

    /// Convenience: upload A[m,k], B[k,n], launch matmul, download C[m,n].
    pub fn matmul_generic(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        Self::global().matmul(a, b, m, n, k)
    }

    pub fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);

        let a_h = self.upload(a);
        let b_h = self.upload(b);
        let c_h = self.alloc_f32s(m * n);

        let cube_dim = CubeDim::new_2d(16, 16);
        let cube_count_x = cubecl::prelude::div_ceil(m as u32, 16);
        let cube_count_y = cubecl::prelude::div_ceil(n as u32, 16);

        unsafe {
            matmul_naive_kernel::launch_unchecked::<Rt>(
                &self.client,
                CubeCount::Static(cube_count_x, cube_count_y, 1),
                cube_dim,
                ArrayArg::from_raw_parts(a_h, a.len()),
                ArrayArg::from_raw_parts(b_h, b.len()),
                ArrayArg::from_raw_parts(c_h.clone(), m * n),
                m as u32,
                n as u32,
                k as u32,
            )
        };

        self.download(c_h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_smoke_add_one() {
        let ctx = GpuContext::new();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output = ctx.add_one(&input);
        assert_eq!(output, vec![2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_gpu_matmul_2x2() {
        let ctx = GpuContext::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = ctx.matmul(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_gpu_matmul_non_square() {
        let ctx = GpuContext::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let c = ctx.matmul(&a, &b, 3, 4, 2);
        let expected = vec![
            11.0, 14.0, 17.0, 20.0, 23.0, 30.0, 37.0, 44.0, 35.0, 46.0, 57.0, 68.0,
        ];
        assert_eq!(c, expected);
    }
}
