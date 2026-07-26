use std::env;
use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tenferro_ad::{AdContext, EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;
use tenferro_linalg::EagerTensorLinalgExt;

const DEFAULT_SIZES: &[usize] = &[8, 16];

struct Fixture {
    ctx: Arc<EagerRuntime>,
    input: EagerTensor,
    loss: EagerTensor,
}

fn bench_threads() -> usize {
    env::var("TENFERRO_BENCH_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&threads| threads > 0)
        .unwrap_or(1)
}

fn bench_sizes() -> Vec<usize> {
    env::var("TENFERRO_LINALG_VJP_BENCH_SIZES")
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

#[cfg(feature = "__bench_unification_semantic_ad_api")]
fn ad_ctx(threads: usize) -> Arc<EagerRuntime> {
    let ad = AdContext::builder()
        .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules().unwrap())
        .unwrap()
        .build()
        .unwrap();

    EagerRuntime::with_cpu_backend_and_ad_context(CpuBackend::with_threads(threads).unwrap(), &ad)
}

#[cfg(not(feature = "__bench_unification_semantic_ad_api"))]
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
            let mut value = (mixed % 2048) as f64 / 4096.0 - 0.25;
            value *= 0.01;
            if row == col {
                value += 2.0 + row as f64 / n as f64;
            }
            data.push(value);
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

fn triangular_solve_fixture(n: usize, threads: usize) -> Fixture {
    let ctx = ad_ctx(threads);
    let matrix = variable(&ctx, vec![n, n], upper_matrix(n));
    let rhs = eager(&ctx, vec![n, 2], dense_matrix(n, 3)[0..(n * 2)].to_vec());
    let solution = matrix
        .triangular_solve(&rhs, true, false, false, false)
        .unwrap();
    let loss = reduce_all(&solution);
    Fixture {
        ctx,
        input: matrix,
        loss,
    }
}

fn svd_values_fixture(n: usize, threads: usize) -> Fixture {
    let ctx = ad_ctx(threads);
    let matrix = variable(&ctx, vec![n, n], dense_matrix(n, 5));
    let (_u, singular_values, _vt) = matrix.svd().unwrap();
    let loss = singular_values.reduce_sum(Some(&[0])).unwrap();
    Fixture {
        ctx,
        input: matrix,
        loss,
    }
}

fn run_vjp(fixture: &Fixture) {
    fixture.ctx.clear_grads().unwrap();
    fixture.loss.backward().unwrap();
    let grad = fixture.input.grad().unwrap().unwrap();
    black_box(grad.shape());
    black_box(grad.as_slice::<f64>().unwrap()[0]);
}

fn bench_linalg_vjp_gate(c: &mut Criterion) {
    let threads = bench_threads();
    let mut group = c.benchmark_group(format!("tenferro_linalg/linalg_vjp_gate/threads_{threads}"));

    for n in bench_sizes() {
        let triangular = triangular_solve_fixture(n, threads);
        run_vjp(&triangular);
        group.bench_function(BenchmarkId::new("triangular_solve_vjp", n), |bench| {
            bench.iter(|| run_vjp(black_box(&triangular)));
        });

        let svd = svd_values_fixture(n, threads);
        run_vjp(&svd);
        group.bench_function(BenchmarkId::new("svd_values_vjp", n), |bench| {
            bench.iter(|| run_vjp(black_box(&svd)));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_linalg_vjp_gate);
criterion_main!(benches);
