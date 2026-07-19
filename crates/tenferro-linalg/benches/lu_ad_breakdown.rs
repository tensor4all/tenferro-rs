use std::env;
use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tenferro_ad::{AdContext, EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;
use tenferro_linalg::EagerTensorLinalgExt;
use tenferro_runtime::DotGeneralConfig;

const DEFAULT_SIZES: &[usize] = &[256, 512];

struct Fixture {
    ctx: Arc<EagerRuntime>,
    matrix: EagerTensor,
    tangent: EagerTensor,
    lu_loss: EagerTensor,
    lower_unit: EagerTensor,
    upper: EagerTensor,
    rhs: EagerTensor,
    scalar_one: EagerTensor,
}

fn bench_threads() -> usize {
    env::var("TENFERRO_BENCH_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&threads| threads > 0)
        .unwrap_or(1)
}

fn bench_sizes() -> Vec<usize> {
    env::var("TENFERRO_LINALG_BENCH_SIZES")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .filter(|&size| size > 0)
                .collect::<Vec<_>>()
        })
        .filter(|sizes| !sizes.is_empty())
        .unwrap_or_else(|| DEFAULT_SIZES.to_vec())
}

fn ad_ctx(threads: usize) -> Arc<EagerRuntime> {
    let ad = AdContext::builder()
        .with_extension_rules(tenferro_linalg::ad_rules().unwrap())
        .build()
        .unwrap();
    EagerRuntime::with_cpu_backend_and_ad_context(CpuBackend::with_threads(threads).unwrap(), &ad)
}

fn tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::from_vec_col_major(shape, data).unwrap()
}

fn eager(ctx: &Arc<EagerRuntime>, shape: Vec<usize>, data: Vec<f64>) -> EagerTensor {
    EagerTensor::from_tensor_in(tensor(shape, data), Arc::clone(ctx)).unwrap()
}

fn variable(ctx: &Arc<EagerRuntime>, shape: Vec<usize>, data: Vec<f64>) -> EagerTensor {
    EagerTensor::requires_grad_in(tensor(shape, data), Arc::clone(ctx)).unwrap()
}

fn dense_matrix(n: usize, seed: u64) -> Vec<f64> {
    let mut data = Vec::with_capacity(n * n);
    for col in 0..n {
        for row in 0..n {
            let mixed = (row as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add((col as u64).wrapping_mul(1442695040888963407))
                .wrapping_add(seed);
            let mut value = (mixed % 2048) as f64 / 2048.0 - 0.5;
            value *= 0.01;
            if row == col {
                value += 2.0 + row as f64 / n as f64;
            }
            data.push(value);
        }
    }
    data
}

fn lower_unit_matrix(n: usize) -> Vec<f64> {
    let mut data = vec![0.0; n * n];
    for col in 0..n {
        for row in col..n {
            data[row + n * col] = if row == col {
                1.0
            } else {
                ((row + col * 7 + 3) % 31) as f64 / 500.0
            };
        }
    }
    data
}

fn upper_matrix(n: usize) -> Vec<f64> {
    let mut data = vec![0.0; n * n];
    for col in 0..n {
        for row in 0..=col {
            data[row + n * col] = if row == col {
                2.0 + row as f64 / n as f64
            } else {
                ((row * 11 + col + 5) % 37) as f64 / 400.0
            };
        }
    }
    data
}

fn reduce_all(tensor: &EagerTensor) -> EagerTensor {
    let axes: Vec<usize> = (0..tensor.shape().len()).collect();
    tensor.reduce_sum(Some(&axes)).unwrap()
}

fn fixture(n: usize, threads: usize) -> Fixture {
    let ctx = ad_ctx(threads);
    let matrix = variable(&ctx, vec![n, n], dense_matrix(n, 1));
    let tangent = eager(&ctx, vec![n, n], dense_matrix(n, 2));
    let lower_unit = eager(&ctx, vec![n, n], lower_unit_matrix(n));
    let upper = eager(&ctx, vec![n, n], upper_matrix(n));
    let rhs = eager(&ctx, vec![n, n], dense_matrix(n, 3));
    let scalar_one = eager(&ctx, vec![], vec![1.0]);

    let (_p, l, u, _parity) = matrix.lu().unwrap();
    let lu_loss = reduce_all(&l).add(&reduce_all(&u)).unwrap();
    let warm = ctx.jvp(&lu_loss, &matrix, &tangent).unwrap();
    consume_f64(&warm);

    Fixture {
        ctx,
        matrix,
        tangent,
        lu_loss,
        lower_unit,
        upper,
        rhs,
        scalar_one,
    }
}

fn consume_f64(tensor: &EagerTensor) {
    let materialized = tensor.materialized().unwrap();
    black_box(materialized.shape());
    black_box(materialized.as_slice::<f64>().unwrap()[0]);
}

fn consume_many(outputs: &[EagerTensor]) {
    for output in outputs {
        consume_f64(output);
    }
}

fn bench_lu_ad_breakdown(c: &mut Criterion) {
    let threads = bench_threads();
    let mut group = c.benchmark_group(format!("tenferro_linalg/lu_ad_breakdown/threads_{threads}"));

    for n in bench_sizes() {
        let fixture = fixture(n, threads);

        group.bench_function(BenchmarkId::new("lu_jvp_sum_lu_outputs", n), |bench| {
            bench.iter(|| {
                let out = fixture
                    .ctx
                    .jvp(
                        black_box(&fixture.lu_loss),
                        black_box(&fixture.matrix),
                        black_box(&fixture.tangent),
                    )
                    .unwrap();
                consume_f64(&out);
            });
        });

        group.bench_function(BenchmarkId::new("lu_forward_unpack", n), |bench| {
            bench.iter(|| {
                let (p, l, u, parity) = black_box(&fixture.matrix).lu().unwrap();
                consume_many(&[p, l, u, parity]);
            });
        });

        group.bench_function(
            BenchmarkId::new("triangular_solve_left_unit_lower", n),
            |bench| {
                bench.iter(|| {
                    let out = black_box(&fixture.lower_unit)
                        .triangular_solve(black_box(&fixture.rhs), true, true, false, true)
                        .unwrap();
                    consume_f64(&out);
                });
            },
        );

        group.bench_function(
            BenchmarkId::new("triangular_solve_right_upper", n),
            |bench| {
                bench.iter(|| {
                    let out = black_box(&fixture.upper)
                        .triangular_solve(black_box(&fixture.rhs), false, false, false, false)
                        .unwrap();
                    consume_f64(&out);
                });
            },
        );

        group.bench_function(
            BenchmarkId::new("structural_lower_plus_identity", n),
            |bench| {
                bench.iter(|| {
                    let strict_lower = black_box(&fixture.lower_unit).tril(-1).unwrap();
                    let diagonal = fixture
                        .scalar_one
                        .broadcast_in_dim(black_box(&[n]), black_box(&[]))
                        .unwrap();
                    let eye = diagonal.embed_diag(0, 1).unwrap();
                    let out = strict_lower.add(&eye).unwrap();
                    consume_f64(&out);
                });
            },
        );

        group.bench_function(BenchmarkId::new("structural_upper_mask", n), |bench| {
            bench.iter(|| {
                let out = black_box(&fixture.upper).triu(0).unwrap();
                consume_f64(&out);
            });
        });

        group.bench_function(BenchmarkId::new("dot_general_square", n), |bench| {
            let config = DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            };
            bench.iter(|| {
                let out = black_box(&fixture.matrix)
                    .dot_general(black_box(&fixture.rhs), black_box(config.clone()))
                    .unwrap();
                consume_f64(&out);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_lu_ad_breakdown);
criterion_main!(benches);
