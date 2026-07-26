use std::env;
use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_einsum::EagerEinsumExt;
use tenferro_runtime::Tensor;

const SHAPE_COUNT: usize = 129;
const PREPARED_PLAN_DEFAULT_ENTRY_LIMIT: usize = 128;
const _: () = assert!(SHAPE_COUNT > PREPARED_PLAN_DEFAULT_ENTRY_LIMIT);

struct ShapeCase {
    lhs: EagerTensor,
    mid: EagerTensor,
    rhs: EagerTensor,
}

fn bench_threads() -> usize {
    env::var("TENFERRO_BENCH_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&threads| threads > 0)
        .unwrap_or(1)
}

fn cpu_ctx(threads: usize) -> Arc<EagerRuntime> {
    EagerRuntime::with_cpu_backend(CpuBackend::with_threads(threads).unwrap())
}

fn f64_tensor(shape: Vec<usize>, seed: usize) -> Tensor {
    let len = shape.iter().product();
    let data = (0..len)
        .map(|index| ((index * 19 + seed * 37 + 11) % 1009) as f64 / 1009.0 - 0.5)
        .collect();
    Tensor::from_vec_col_major(shape, data).unwrap()
}

fn eager(ctx: &Arc<EagerRuntime>, tensor: Tensor) -> EagerTensor {
    EagerTensor::from_tensor_in(tensor, Arc::clone(ctx)).unwrap()
}

fn consume_f64(tensor: &EagerTensor) {
    let materialized = tensor.materialized().unwrap();
    let data = materialized.as_ref();
    black_box(data.shape());
    black_box(data.as_slice::<f64>().unwrap()[0]);
}

fn shape_cases(ctx: &Arc<EagerRuntime>) -> Vec<ShapeCase> {
    (0..SHAPE_COUNT)
        .map(|index| {
            let a = 2 + index % 5;
            let b = 2 + (index * 3) % 4;
            let c = 2 + (index * 5) % 7;
            let d = 2 + (index * 7) % 5;
            let e = 2 + (index * 11) % 4;
            let f = 2 + (index * 13) % 3;
            ShapeCase {
                lhs: eager(ctx, f64_tensor(vec![a, b, c], index * 3 + 1)),
                mid: eager(ctx, f64_tensor(vec![c, d, e], index * 3 + 2)),
                rhs: eager(ctx, f64_tensor(vec![e, f], index * 3 + 3)),
            }
        })
        .collect()
}

fn run_shape_sequence(cases: &[ShapeCase]) {
    for case in cases {
        let out = [
            black_box(&case.lhs),
            black_box(&case.mid),
            black_box(&case.rhs),
        ]
        .einsum("abc,cde,ef->abdf")
        .unwrap();
        consume_f64(&out);
    }
}

fn bench_changing_shape_prepare(c: &mut Criterion) {
    let threads = bench_threads();
    let ctx = cpu_ctx(threads);
    let cases = shape_cases(&ctx);

    let mut group = c.benchmark_group(format!(
        "tenferro_einsum/changing_shape_prepare/threads_{threads}"
    ));
    group.bench_function(
        BenchmarkId::new("complete_call_sequence_shapes", cases.len()),
        |bench| {
            bench.iter(|| run_shape_sequence(black_box(&cases)));
        },
    );
    group.finish();
}

criterion_group!(benches, bench_changing_shape_prepare);
criterion_main!(benches);
