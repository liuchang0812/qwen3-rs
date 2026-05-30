use qwen3_5_rs::gpu::GpuContext;
use qwen3_5_rs::tensor::Tensor;
use std::time::Instant;

fn main() {
    let shapes = [
        (1, 1536, 1536),
        (1, 1536, 4608),
        (1, 1536, 8960),
        (1, 8960, 1536),
        (16, 4096, 4096),
        (64, 1024, 1024),
    ];

    println!("Benchmark: CPU vs GPU (naive cubecl kernel)");
    println!("{:-<78}", "");
    println!("{:>8} {:>8} {:>8} {:>12} {:>12} {:>10}", "M", "K", "N", "CPU (ms)", "GPU (ms)", "Speedup");
    println!("{:-<78}", "");

    let ctx = GpuContext::new();

    for &(m, k, n) in &shapes {
        let a = Tensor::new(vec![m, k], (0..m*k).map(|i| i as f32 * 0.001).collect());
        let b = Tensor::new(vec![k, n], (0..k*n).map(|i| i as f32 * 0.001).collect());

        // CPU
        let n_iter = if m * k * n < 1_000_000 { 50 } else { 5 };
        let _ = a.matmul(&b);
        let cpu_start = Instant::now();
        for _ in 0..n_iter { let _ = a.matmul(&b); }
        let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0 / n_iter as f64;

        // GPU (upload each time to be fair)
        let _ = ctx.matmul(a.data(), b.data(), m, n, k);
        let gpu_start = Instant::now();
        for _ in 0..n_iter { let _ = ctx.matmul(a.data(), b.data(), m, n, k); }
        let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1000.0 / n_iter as f64;

        let speedup = cpu_ms / gpu_ms;
        println!("{:>8} {:>8} {:>8} {:>12.3} {:>12.3} {:>9.2}x",
                 m, k, n, cpu_ms, gpu_ms, speedup);
    }
}
