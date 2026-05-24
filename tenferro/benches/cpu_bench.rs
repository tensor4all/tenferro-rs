use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, Bencher, BenchmarkId, Criterion};
use num_complex::Complex64;
use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};

const SMALL_MATMUL_SIZES: &[usize] = &[2, 4, 8, 16, 32];
const LARGE_MATMUL_SIZES: &[usize] = &[128, 256, 512];

fn bench_threads() -> usize {
    env::var("TENFERRO_BENCH_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&threads| threads > 0)
        .unwrap_or(1)
}

fn cpu_ctx(threads: usize) -> Arc<EagerRuntime> {
    EagerRuntime::with_cpu_backend(CpuBackend::with_threads(threads))
}

fn f64_tensor(shape: Vec<usize>, seed: usize) -> Tensor {
    let len = shape.iter().product();
    let data = (0..len)
        .map(|idx| ((idx * 17 + seed * 31 + 7) % 997) as f64 / 997.0 - 0.5)
        .collect();
    Tensor::from_vec_col_major(shape, data)
}

fn c64_tensor(shape: Vec<usize>, seed: usize) -> Tensor {
    let len = shape.iter().product();
    let data = (0..len)
        .map(|idx| {
            let re = ((idx * 17 + seed * 31 + 7) % 997) as f64 / 997.0 - 0.5;
            let im = ((idx * 23 + seed * 19 + 11) % 991) as f64 / 991.0 - 0.5;
            Complex64::new(re, im)
        })
        .collect();
    Tensor::from_vec_col_major(shape, data)
}

fn eager(ctx: &Arc<EagerRuntime>, tensor: Tensor) -> EagerTensor {
    EagerTensor::from_tensor_in(tensor, Arc::clone(ctx))
}

fn tracked(ctx: &Arc<EagerRuntime>, tensor: Tensor) -> EagerTensor {
    EagerTensor::requires_grad_in(tensor, Arc::clone(ctx))
}

fn iter_excluding_setup_and_input_drop<I>(
    bench: &mut Bencher<'_>,
    mut setup: impl FnMut() -> I,
    mut routine: impl FnMut(&I),
) {
    bench.iter_custom(|iters| {
        let mut total = Duration::ZERO;
        for _ in 0..iters {
            let input = setup();
            let started = Instant::now();
            routine(&input);
            total += started.elapsed();
            black_box(&input);
            drop(input);
        }
        total
    });
}

fn consume_f64(tensor: &EagerTensor) {
    let data = tensor.data();
    black_box(data.shape());
    black_box(data.as_slice::<f64>().expect("f64 tensor")[0]);
}

fn consume_c64(tensor: &EagerTensor) {
    let data = tensor.data();
    black_box(data.shape());
    black_box(data.as_slice::<Complex64>().expect("c64 tensor")[0]);
}

fn bench_matmul(c: &mut Criterion) {
    let threads = bench_threads();
    let ctx = cpu_ctx(threads);
    let mut group = c.benchmark_group(format!("tenferro_cpu/matmul/threads_{threads}"));

    for &n in SMALL_MATMUL_SIZES.iter().chain(LARGE_MATMUL_SIZES) {
        let a = eager(&ctx, f64_tensor(vec![n, n], 1));
        let b = eager(&ctx, f64_tensor(vec![n, n], 2));
        group.bench_function(BenchmarkId::new("f64_square", n), |bench| {
            bench.iter(|| {
                let out = black_box(&a)
                    .matmul(black_box(&b))
                    .expect("f64 matmul should succeed");
                consume_f64(&out);
            });
        });

        if n <= 128 {
            let a = eager(&ctx, c64_tensor(vec![n, n], 3));
            let b = eager(&ctx, c64_tensor(vec![n, n], 4));
            group.bench_function(BenchmarkId::new("c64_square", n), |bench| {
                bench.iter(|| {
                    let out = black_box(&a)
                        .matmul(black_box(&b))
                        .expect("c64 matmul should succeed");
                    consume_c64(&out);
                });
            });
        }
    }

    group.finish();
}

fn bench_ad(c: &mut Criterion) {
    let threads = bench_threads();
    let mut group = c.benchmark_group(format!("tenferro_cpu/ad/threads_{threads}"));

    for &n in &[4, 16, 64] {
        group.bench_function(BenchmarkId::new("f64_grad_sum_matmul", n), |bench| {
            iter_excluding_setup_and_input_drop(
                bench,
                || {
                    let ctx = cpu_ctx(threads);
                    let a = tracked(&ctx, f64_tensor(vec![n, n], 1));
                    let b = tracked(&ctx, f64_tensor(vec![n, n], 2));
                    (a, b)
                },
                |(a, b)| {
                    let out = a.matmul(&b).expect("matmul should succeed");
                    let loss = out.reduce_sum(&[0, 1]).expect("sum should succeed");
                    black_box(loss.backward().expect("backward should succeed"));
                    black_box(a.grad());
                    black_box(b.grad());
                },
            );
        });
    }

    let n = 64;
    group.bench_function("f64_forward_matmul_sum_untracked/64", |bench| {
        let ctx = cpu_ctx(threads);
        let a = eager(&ctx, f64_tensor(vec![n, n], 1));
        let b = eager(&ctx, f64_tensor(vec![n, n], 2));
        bench.iter(|| {
            let out = black_box(&a)
                .matmul(black_box(&b))
                .expect("matmul should succeed");
            let loss = out.reduce_sum(&[0, 1]).expect("sum should succeed");
            consume_f64(&loss);
        });
    });

    group.bench_function("f64_forward_matmul_sum_tracked/64", |bench| {
        iter_excluding_setup_and_input_drop(
            bench,
            || {
                let ctx = cpu_ctx(threads);
                let a = tracked(&ctx, f64_tensor(vec![n, n], 1));
                let b = tracked(&ctx, f64_tensor(vec![n, n], 2));
                (a, b)
            },
            |(a, b)| {
                let out = black_box(&a)
                    .matmul(black_box(&b))
                    .expect("matmul should succeed");
                let loss = out.reduce_sum(&[0, 1]).expect("sum should succeed");
                consume_f64(&loss);
            },
        );
    });

    group.bench_function("f64_backward_only_sum_matmul/64", |bench| {
        iter_excluding_setup_and_input_drop(
            bench,
            || {
                let ctx = cpu_ctx(threads);
                let a = tracked(&ctx, f64_tensor(vec![n, n], 1));
                let b = tracked(&ctx, f64_tensor(vec![n, n], 2));
                let out = a.matmul(&b).expect("matmul should succeed");
                let loss = out.reduce_sum(&[0, 1]).expect("sum should succeed");
                (a, b, loss)
            },
            |(a, b, loss)| {
                black_box(loss.backward().expect("backward should succeed"));
                black_box(a.grad());
                black_box(b.grad());
            },
        );
    });

    group.bench_function("f64_backward_only_reduce_sum/64", |bench| {
        iter_excluding_setup_and_input_drop(
            bench,
            || {
                let ctx = cpu_ctx(threads);
                let a = tracked(&ctx, f64_tensor(vec![n, n], 1));
                let loss = a.reduce_sum(&[0, 1]).expect("sum should succeed");
                (a, loss)
            },
            |(a, loss)| {
                black_box(loss.backward().expect("backward should succeed"));
                black_box(a.grad());
            },
        );
    });

    group.bench_function("f64_manual_grad_sum_matmul_math/64", |bench| {
        let ctx = cpu_ctx(threads);
        let a = eager(&ctx, f64_tensor(vec![n, n], 1));
        let b = eager(&ctx, f64_tensor(vec![n, n], 2));
        let ct = eager(
            &ctx,
            Tensor::from_vec_col_major(vec![n, n], vec![1.0_f64; n * n]),
        );
        let grad_a_config = tenferro::DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![1],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        };
        let grad_b_config = tenferro::DotGeneralConfig {
            lhs_contracting_dims: vec![0],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        };
        bench.iter(|| {
            let grad_a = black_box(&ct)
                .dot_general(black_box(&b), grad_a_config.clone())
                .expect("grad A dot_general should succeed");
            let grad_b = black_box(&a)
                .dot_general(black_box(&ct), grad_b_config.clone())
                .expect("grad B dot_general should succeed");
            consume_f64(&grad_a);
            consume_f64(&grad_b);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_matmul, bench_ad);
criterion_main!(benches);
