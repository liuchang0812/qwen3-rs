# CUDA Setup

This project can optionally use a GPU via `cubecl` + `cubecl-cuda`.

## Prerequisites

- NVIDIA GPU with CUDA 12+ driver (tested on GTX 1060 with driver 580.142)
- CUDA toolkit headers and NVRTC libraries at runtime

## Local CUDA Setup

CUDA toolkit headers + NVRTC libraries must be extracted into `cuda_local/` (gitignored).

1. Install CUDA debs via apt (requires sudo) or download header/libs packages:
   ```
   apt-get download cuda-cudart-dev-13-1 libnvrtc12 libnvrtc-builtins12.4
   ```
2. Extract them and copy into `cuda_local/`:
   ```
   mkdir -p cuda_local/include cuda_local/lib
   # Extract headers from cuda-cudart-dev deb to cuda_local/include/
   # Extract libnvrtc.so.*, libnvrtc-builtins.so.* from debs to cuda_local/lib/
   ```

## Running with GPU

Source the environment script before any `cargo` command:

```sh
source ./cuda_env.sh
cargo build --features gpu
cargo test --features gpu
```

## Compilation Flow

- `gpu` feature flag adds `cubecl 0.10` + `cubecl-cuda 0.10` as optional deps
- At runtime, cubecl-cuda JIT-compiles CUDA C++ kernels via NVRTC
- CUDA headers found via `CUDA_PATH` env var (set by `cuda_env.sh`)
- NVRTC library loaded via `libloading` (set `LD_LIBRARY_PATH`)

## GPU Matmul Integration

`Tensor::matmul` in `tensor.rs` automatically routes to GPU when `gpu` feature is enabled.
The naive kernel uses one thread per output element (no shared memory tiling).
Performance is on par with the CPU triple-loop for small matrices.

## Tests

```sh
# CPU-only tests
cargo test

# GPU tests (requires source cuda_env.sh first)
source ./cuda_env.sh && cargo test --features gpu
```

Key tests in `src/gpu.rs`:
- `test_gpu_smoke_add_one` — simple element-wise GPU kernel
- `test_gpu_matmul_2x2` — 2x2 matrix multiply
- `test_gpu_matmul_non_square` — 3x4 @ 4x2 matrix multiply
