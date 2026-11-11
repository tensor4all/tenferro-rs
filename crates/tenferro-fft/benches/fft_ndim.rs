//! Compare one n-dimensional FFT entry point against a caller-written
//! per-axis loop on the same input, engine, and plan cache.
//!
//! Run with an explicit single-thread backend and a pinned CPU set, for
//! example `taskset -c 0-3 cargo bench -p tenferro-fft --bench fft_ndim`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::FftExecutor;
use tenferro_fft::{FftOptions, FftnOptions};
use tenferro_tensor::{Engine, Tensor};

fn c64_grid(cube: usize) -> Tensor {
    let len = cube * cube * cube;
    let values = (0..len)
        .map(|index| {
            let real = ((index * 17 + 3) % 257) as f64 / 257.0 - 0.5;
            let imag = ((index * 29 + 5) % 263) as f64 / 263.0 - 0.5;
            Complex64::new(real, imag)
        })
        .collect::<Vec<_>>();
    Tensor::from_vec_col_major(vec![cube, cube, cube], values).unwrap()
}

fn bench_fft_ndim(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_ndim/c64_forward");
    group.sample_size(10);
    for cube in [64usize, 256] {
        let input = c64_grid(cube);
        let id = format!("{cube}x{cube}x{cube}");
        let mut engine = CpuEngine::serial();
        let mut executor = FftExecutor::default();
        // Warm the plans both arms share, so neither pays planning cost.
        executor
            .fftn_with(
                &input,
                &FftnOptions::default().axes(vec![0, 1, 2]),
                &mut engine,
            )
            .unwrap();

        group.bench_function(format!("caller_one_axis_loop/{id}"), |bench| {
            let mut engine = CpuEngine::serial();
            bench.iter(|| {
                let mut current = {
                    let session: &mut dyn Engine = &mut engine;
                    executor
                        .fft_with(black_box(&input), &FftOptions::default().axis(0), session)
                        .unwrap()
                };
                for axis in 1..3 {
                    let session: &mut dyn Engine = &mut engine;
                    current = executor
                        .fft_with(&current, &FftOptions::default().axis(axis), session)
                        .unwrap();
                }
                black_box(current);
            });
        });

        group.bench_function(format!("fftn/{id}"), |bench| {
            let mut engine = CpuEngine::serial();
            bench.iter(|| {
                let session: &mut dyn Engine = &mut engine;
                black_box(
                    executor
                        .fftn_with(
                            black_box(&input),
                            &FftnOptions::default().axes(vec![0, 1, 2]),
                            session,
                        )
                        .unwrap(),
                );
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_fft_ndim);
criterion_main!(benches);
