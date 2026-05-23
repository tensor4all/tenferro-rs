use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, Bencher, BenchmarkId, Criterion};
use num_complex::{Complex32, Complex64};
use tenferro::eager_tensor::einsum;
use tenferro::{CpuBackend, DType, EagerRuntime, EagerTensor, Tensor};

const SMALL_MATMUL_SIZES: &[usize] = &[2, 4, 8, 16, 32];
const LARGE_MATMUL_SIZES: &[usize] = &[128, 256, 512];
const SMALL_LINALG_SIZES: &[usize] = &[4, 8, 16, 32];
const MEDIUM_LINALG_SIZES: &[usize] = &[64, 128];
const BATCHES: &[usize] = &[16, 64, 256];
const BATCHED_SMALL_SIZES: &[usize] = &[2, 4, 8, 16];

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

fn f64_spd_tensor(n: usize, seed: usize) -> Tensor {
    let mut data = vec![0.0; n * n];
    for col in 0..n {
        for row in 0..n {
            let value = if row == col {
                n as f64 + 2.0 + (row + seed) as f64 * 0.001
            } else {
                ((row + col + seed) % 7) as f64 * 0.01
            };
            data[row + col * n] = value;
        }
    }
    Tensor::from_vec_col_major(vec![n, n], data)
}

fn c64_hermitian_tensor(n: usize, seed: usize) -> Tensor {
    let mut data = vec![Complex64::new(0.0, 0.0); n * n];
    for col in 0..n {
        for row in 0..=col {
            let value = if row == col {
                Complex64::new(n as f64 + 2.0 + (row + seed) as f64 * 0.001, 0.0)
            } else {
                let re = ((row + col + seed) % 7) as f64 * 0.01;
                let im = ((row * 3 + col + seed) % 5) as f64 * 0.01;
                Complex64::new(re, im)
            };
            data[row + col * n] = value;
            data[col + row * n] = value.conj();
        }
    }
    Tensor::from_vec_col_major(vec![n, n], data)
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

fn consume_numeric(tensor: &EagerTensor) {
    let data = tensor.data();
    black_box(data.shape());
    match data.dtype() {
        DType::F32 => {
            black_box(data.as_slice::<f32>().expect("f32 tensor")[0]);
        }
        DType::F64 => {
            black_box(data.as_slice::<f64>().expect("f64 tensor")[0]);
        }
        DType::I64 => {
            black_box(data.as_slice::<i64>().expect("i64 tensor")[0]);
        }
        DType::C32 => {
            black_box(data.as_slice::<Complex32>().expect("c32 tensor")[0]);
        }
        DType::C64 => {
            black_box(data.as_slice::<Complex64>().expect("c64 tensor")[0]);
        }
    }
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

fn bench_linalg(c: &mut Criterion) {
    let threads = bench_threads();
    let ctx = cpu_ctx(threads);
    let mut group = c.benchmark_group(format!("tenferro_cpu/linalg/threads_{threads}"));

    for &n in SMALL_LINALG_SIZES.iter().chain(MEDIUM_LINALG_SIZES) {
        let a = eager(&ctx, f64_spd_tensor(n, 1));
        let b_col = eager(&ctx, f64_tensor(vec![n, 1], 2));
        let b_mat = eager(&ctx, f64_tensor(vec![n, 4], 3));

        group.bench_function(BenchmarkId::new("f64_svd", n), |bench| {
            bench.iter(|| {
                let (_, s, _) = black_box(&a).svd().expect("svd should succeed");
                consume_f64(&s);
            });
        });
        group.bench_function(BenchmarkId::new("f64_qr", n), |bench| {
            bench.iter(|| {
                let (_, r) = black_box(&a).qr().expect("qr should succeed");
                consume_f64(&r);
            });
        });
        group.bench_function(BenchmarkId::new("f64_eigh", n), |bench| {
            bench.iter(|| {
                let (values, _) = black_box(&a).eigh().expect("eigh should succeed");
                consume_f64(&values);
            });
        });
        group.bench_function(BenchmarkId::new("f64_solve_column_rhs1", n), |bench| {
            bench.iter(|| {
                let out = black_box(&a)
                    .solve(black_box(&b_col))
                    .expect("solve column RHS should succeed");
                consume_f64(&out);
            });
        });
        group.bench_function(BenchmarkId::new("f64_solve_matrix_rhs4", n), |bench| {
            bench.iter(|| {
                let out = black_box(&a)
                    .solve(black_box(&b_mat))
                    .expect("solve matrix RHS should succeed");
                consume_f64(&out);
            });
        });

        if n <= 32 {
            let a = eager(&ctx, c64_hermitian_tensor(n, 5));
            group.bench_function(BenchmarkId::new("c64_eigh", n), |bench| {
                bench.iter(|| {
                    let (values, _) = black_box(&a).eigh().expect("c64 eigh should succeed");
                    consume_numeric(&values);
                });
            });
        }
    }

    group.finish();
}

fn bench_batched_einsum(c: &mut Criterion) {
    let threads = bench_threads();
    let ctx = cpu_ctx(threads);
    let mut group = c.benchmark_group(format!(
        "tenferro_cpu/batched_einsum_rightmost_batch/threads_{threads}"
    ));

    for &batch in BATCHES {
        for &n in BATCHED_SMALL_SIZES {
            let a = eager(&ctx, f64_tensor(vec![n, n, batch], 1));
            let b = eager(&ctx, f64_tensor(vec![n, n, batch], 2));
            let params = format!("n_{n}_batch_{batch}");
            group.bench_function(BenchmarkId::new("f64_ikb_knb_to_inb", params), |bench| {
                bench.iter(|| {
                    let out = einsum(&[black_box(&a), black_box(&b)], "ikb,knb->inb")
                        .expect("batched einsum should succeed");
                    consume_f64(&out);
                });
            });
        }
    }

    group.finish();
}

fn bench_einsum_patterns(c: &mut Criterion) {
    let threads = bench_threads();
    let ctx = cpu_ctx(threads);
    let mut group = c.benchmark_group(format!("tenferro_cpu/einsum_patterns/threads_{threads}"));

    let a = eager(&ctx, f64_tensor(vec![64, 64], 1));
    let b = eager(&ctx, f64_tensor(vec![64, 64], 2));
    let c_tensor = eager(&ctx, f64_tensor(vec![64, 64], 3));
    group.bench_function("f64_binary_ij_jk_to_ik", |bench| {
        bench.iter(|| {
            let out = einsum(&[black_box(&a), black_box(&b)], "ij,jk->ik")
                .expect("binary einsum should succeed");
            consume_f64(&out);
        });
    });
    group.bench_function("f64_chain_ij_jk_kl_to_il", |bench| {
        bench.iter(|| {
            let out = einsum(
                &[black_box(&a), black_box(&b), black_box(&c_tensor)],
                "ij,jk,kl->il",
            )
            .expect("chain einsum should succeed");
            consume_f64(&out);
        });
    });

    let x = eager(&ctx, f64_tensor(vec![8, 16, 8], 4));
    let y = eager(&ctx, f64_tensor(vec![16, 8, 8], 5));
    group.bench_function("f64_multiedge_ijk_jkl_to_il", |bench| {
        bench.iter(|| {
            let out = einsum(&[black_box(&x), black_box(&y)], "ijk,jkl->il")
                .expect("multi-edge einsum should succeed");
            consume_f64(&out);
        });
    });

    let a = eager(&ctx, c64_tensor(vec![32, 32], 6));
    let b = eager(&ctx, c64_tensor(vec![32, 32], 7));
    group.bench_function("c64_binary_ij_jk_to_ik", |bench| {
        bench.iter(|| {
            let out = einsum(&[black_box(&a), black_box(&b)], "ij,jk->ik")
                .expect("c64 binary einsum should succeed");
            consume_c64(&out);
        });
    });

    group.finish();
}

fn bench_ad(c: &mut Criterion) {
    let threads = bench_threads();
    let mut group = c.benchmark_group(format!("tenferro_cpu/ad/threads_{threads}"));

    for &n in SMALL_LINALG_SIZES.iter().chain(MEDIUM_LINALG_SIZES) {
        group.bench_function(BenchmarkId::new("f64_grad_sum_svd_values", n), |bench| {
            iter_excluding_setup_and_input_drop(
                bench,
                || {
                    let ctx = cpu_ctx(threads);
                    tracked(&ctx, f64_spd_tensor(n, 3))
                },
                |a| {
                    let (_, s, _) = a.svd().expect("svd should succeed");
                    let loss = s.reduce_sum(&[0]).expect("sum should succeed");
                    black_box(loss.backward().expect("backward should succeed"));
                    black_box(a.grad());
                },
            );
        });
    }

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

        if n <= 16 {
            group.bench_function(BenchmarkId::new("f64_grad_sum_solve", n), |bench| {
                iter_excluding_setup_and_input_drop(
                    bench,
                    || {
                        let ctx = cpu_ctx(threads);
                        let a = tracked(&ctx, f64_spd_tensor(n, 4));
                        let b = tracked(&ctx, f64_tensor(vec![n, 1], 5));
                        (a, b)
                    },
                    |(a, b)| {
                        let x = a.solve(&b).expect("solve should succeed");
                        let loss = x.reduce_sum(&[0, 1]).expect("sum should succeed");
                        black_box(loss.backward().expect("backward should succeed"));
                        black_box(a.grad());
                        black_box(b.grad());
                    },
                );
            });
        }
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

criterion_group!(
    benches,
    bench_matmul,
    bench_linalg,
    bench_batched_einsum,
    bench_einsum_patterns,
    bench_ad
);
criterion_main!(benches);
