//! Benchmark: the Metal elementwise family and the explicit transfers against the CPU on
//! the same shared buffers, on the M5 Max this unit was developed on.
//!
//! Run with `cargo bench -p tenferro-gpu --features metal --bench metal_engine`. The numbers
//! are a recorded observation (plan §2.11): the benchmark prints them, and each PR records
//! them; there is no tuning loop in this slice.
//!
//! The CPU side runs on the very same shared allocations, so the comparison is kernel time,
//! not transfer time: nothing is copied between the two measurements.

use std::time::{Duration, Instant};

use tenferro_gpu::metal::{metal_available, MetalContext};
use tenferro_tensor::{Tensor, TensorElementwise, TensorRead};

fn host_f32(len: usize) -> Tensor {
    let values: Vec<f32> = (0..len)
        .map(|index| ((index % 17) as f32) * 0.125 - 1.0)
        .collect();
    Tensor::from_vec_col_major(vec![len], values).expect("a 1-D host tensor")
}

fn host_c32(len: usize) -> Tensor {
    let values: Vec<num_complex::Complex32> = (0..len)
        .map(|index| {
            let scale = index as f32 * 0.25;
            num_complex::Complex32::new(scale, -scale * 0.5)
        })
        .collect();
    Tensor::from_vec_col_major(vec![len], values).expect("a 1-D host tensor")
}

/// Fastest of `repeats` timings of `body`.
fn best(mut body: impl FnMut(), repeats: usize) -> Duration {
    let mut best = Duration::MAX;
    for _ in 0..repeats {
        let start = Instant::now();
        body();
        best = best.min(start.elapsed());
    }
    best
}

fn report(label: &str, len: usize, elapsed: Duration, flops_per_element: f64) {
    let seconds = elapsed.as_secs_f64();
    let gflops = (len as f64 * flops_per_element) / seconds / 1e9;
    println!("{label:<28} n={len:<9} {seconds:>9.4e} s  {gflops:>8.2} GFLOP/s");
}

fn main() {
    if !metal_available() {
        println!("no Metal device on this host; nothing to measure");
        return;
    }
    let context = MetalContext::open(Default::default()).expect("the probe reported a device");
    println!("device: {}", context.device().name);
    let mut engine = context.engine();
    let mut cpu =
        tenferro_cpu::CpuEngine::serial().with_allocation_domain(context.domain().clone());

    for len in [1_000usize, 1_000_000, 10_000_000] {
        let host = host_f32(len);
        let upload = best(
            || {
                let shared = context
                    .upload(TensorRead::from_tensor(&host))
                    .expect("upload");
                std::hint::black_box(&shared);
            },
            5,
        );
        let shared = context
            .upload(TensorRead::from_tensor(&host))
            .expect("upload");
        let download = best(
            || {
                let back = context
                    .download(TensorRead::from_tensor(&shared))
                    .expect("download");
                std::hint::black_box(&back);
            },
            5,
        );
        let bytes = len as f64 * 4.0;
        println!(
            "upload f32 n={len:<10} {:>9.4} GB/s   download {:>9.4} GB/s",
            bytes / upload.as_secs_f64() / 1e9,
            bytes / download.as_secs_f64() / 1e9
        );

        let left = context
            .upload(TensorRead::from_tensor(&host))
            .expect("upload");
        let right = context
            .upload(TensorRead::from_tensor(&host))
            .expect("upload");

        // The GPU result must be consumed, or a compiler could elide the measurement; the
        // last iteration of each call is kept and dropped after the timer stops.
        let metal_add = best(
            || {
                let sum = engine
                    .add_read(
                        TensorRead::from_tensor(&left),
                        TensorRead::from_tensor(&right),
                    )
                    .expect("add");
                context.synchronize().expect("synchronize");
                std::hint::black_box(&sum);
            },
            5,
        );
        report("metal add f32", len, metal_add, 1.0);

        // Ten submissions then one wait: the difference from the line above separates the
        // per-submission CPU cost from the queue-latency of a single waiter.
        let batched = best(
            || {
                for _ in 0..10 {
                    let sum = engine
                        .add_read(
                            TensorRead::from_tensor(&left),
                            TensorRead::from_tensor(&right),
                        )
                        .expect("add");
                    std::hint::black_box(&sum);
                }
                context.synchronize().expect("synchronize");
            },
            5,
        );
        report("metal add f32 x10", len, batched, 10.0);

        let cpu_add = best(
            || {
                let sum = cpu
                    .add_read(
                        TensorRead::from_tensor(&left),
                        TensorRead::from_tensor(&right),
                    )
                    .expect("add");
                std::hint::black_box(&sum);
            },
            5,
        );
        report("cpu add f32 (same buffers)", len, cpu_add, 1.0);

        let metal_mul = best(
            || {
                let product = engine
                    .mul_read(
                        TensorRead::from_tensor(&left),
                        TensorRead::from_tensor(&right),
                    )
                    .expect("mul");
                context.synchronize().expect("synchronize");
                std::hint::black_box(&product);
            },
            5,
        );
        report("metal mul f32", len, metal_mul, 1.0);

        let cpu_mul = best(
            || {
                let product = cpu
                    .mul_read(
                        TensorRead::from_tensor(&left),
                        TensorRead::from_tensor(&right),
                    )
                    .expect("mul");
                std::hint::black_box(&product);
            },
            5,
        );
        report("cpu mul f32 (same buffers)", len, cpu_mul, 1.0);

        let complex_host = host_c32(len);
        let complex_left = context
            .upload(TensorRead::from_tensor(&complex_host))
            .expect("upload");
        let complex_right = context
            .upload(TensorRead::from_tensor(&complex_host))
            .expect("upload");
        let metal_c32_add = best(
            || {
                let sum = engine
                    .add_read(
                        TensorRead::from_tensor(&complex_left),
                        TensorRead::from_tensor(&complex_right),
                    )
                    .expect("add");
                context.synchronize().expect("synchronize");
                std::hint::black_box(&sum);
            },
            5,
        );
        report("metal add c32", len, metal_c32_add, 2.0);

        let cpu_c32_add = best(
            || {
                let sum = cpu
                    .add_read(
                        TensorRead::from_tensor(&complex_left),
                        TensorRead::from_tensor(&complex_right),
                    )
                    .expect("add");
                std::hint::black_box(&sum);
            },
            5,
        );
        report("cpu add c32 (same buffers)", len, cpu_c32_add, 2.0);
    }

    let stats = context.stats();
    println!(
        "submissions {:<10} allocations {:<10} uploads {:<10} downloads {:<10} waits {}",
        stats.submissions, stats.allocations, stats.uploads, stats.downloads, stats.waits
    );
}
