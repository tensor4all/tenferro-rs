use std::env;
use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;

const BOND_DIMS: &[(usize, usize)] = &[
    (8, 10),
    (10, 12),
    (12, 14),
    (14, 16),
    (16, 18),
    (18, 20),
    (20, 22),
    (22, 24),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
    (32, 34),
    (34, 36),
    (36, 38),
    (38, 40),
];
const PHYSICAL_DIM: usize = 4;

struct Fixture {
    ctx: Arc<EagerRuntime>,
    x: EagerTensor,
    loss: EagerTensor,
}

fn bench_threads() -> usize {
    env::var("TENFERRO_BENCH_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&threads| threads > 0)
        .unwrap_or(1)
}

fn tensor(shape: Vec<usize>, seed: usize) -> Tensor {
    let len = shape.iter().product();
    let data = (0..len)
        .map(|index| ((index * 23 + seed * 41 + 17) % 997) as f64 / 997.0 - 0.5)
        .collect();
    Tensor::from_vec_col_major(shape, data).unwrap()
}

fn reduce_all(tensor: &EagerTensor) -> EagerTensor {
    let axes: Vec<usize> = (0..tensor.shape().len()).collect();
    tensor.reduce_sum(Some(&axes)).unwrap()
}

fn fixture(ctx: &Arc<EagerRuntime>, left_bond: usize, right_bond: usize, seed: usize) -> Fixture {
    let shape = vec![left_bond, PHYSICAL_DIM, right_bond];
    let x = EagerTensor::requires_grad_in(tensor(shape.clone(), seed), Arc::clone(ctx)).unwrap();
    let weight = EagerTensor::from_tensor_in(tensor(shape, seed + 1000), Arc::clone(ctx)).unwrap();
    let weighted = x.mul(&weight).unwrap();
    let quadratic = weighted.mul(&x).unwrap();
    let loss = reduce_all(&quadratic);
    Fixture {
        ctx: Arc::clone(ctx),
        x,
        loss,
    }
}

fn run_backward(fixture: &Fixture) {
    fixture.ctx.clear_grads().unwrap();
    fixture.loss.backward().unwrap();
    let grad = fixture.x.grad().unwrap().unwrap();
    black_box(grad.shape());
    black_box(grad.as_slice::<f64>().unwrap()[0]);
}

fn bench_eager_backward_shape_churn(c: &mut Criterion) {
    let threads = bench_threads();
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(threads).unwrap()).unwrap();
    let fixtures = BOND_DIMS
        .iter()
        .enumerate()
        .map(|(index, &(left, right))| fixture(&ctx, left, right, index))
        .collect::<Vec<_>>();

    for fixture in &fixtures {
        run_backward(fixture);
    }

    let mut group = c.benchmark_group(format!(
        "tenferro_ad/eager_backward_shape_churn/threads_{threads}"
    ));
    group.bench_function(
        BenchmarkId::new("bond_dimension_sequence", fixtures.len()),
        |bench| {
            bench.iter(|| {
                for fixture in black_box(&fixtures) {
                    run_backward(fixture);
                }
            });
        },
    );
    group.finish();
}

criterion_group!(benches, bench_eager_backward_shape_churn);
criterion_main!(benches);
