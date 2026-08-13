//! Eager standard-op dispatch baseline (lazy vs materialized consume).
//!
//! The per-op floor includes the CPU session-open cost that issue #1667 tracks:
//! for a single worker the `enter_managed_session` wrapper (~5-8 us) plus the
//! eager view-read materialization are currently hard to avoid. See
//! docs/design/cpu-session-open-cost.md.

use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{DotGeneralConfig, SliceConfig, Tensor};

const ELEMENT_COUNTS: &[usize] = &[1, 8, 64];
const MATRIX_SIZES: &[usize] = &[1, 2];

fn runtime() -> Arc<EagerRuntime> {
    EagerRuntime::with_cpu_backend(
        CpuBackend::with_threads(1).expect("one-thread faer backend should construct"),
    )
    .unwrap()
}

fn eager(runtime: &Arc<EagerRuntime>, shape: Vec<usize>) -> EagerTensor {
    let len = shape.iter().product();
    let tensor = Tensor::from_vec_col_major(shape, vec![1.0_f64; len])
        .expect("benchmark tensor should be valid");
    EagerTensor::from_tensor_in(tensor, Arc::clone(runtime))
        .expect("benchmark eager tensor should be valid")
}

fn consume_lazy(output: EagerTensor) {
    black_box(output.shape());
    black_box(output.tensor_read());
}

fn consume_materialized(output: EagerTensor) {
    let output = output.to_tensor().expect("output should materialize");
    black_box(output.shape());
    black_box(output.as_slice::<f64>().expect("f64 output")[0]);
}

fn matmul_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    }
}

fn bench_lazy(c: &mut Criterion) {
    let runtime = runtime();
    let mut group = c.benchmark_group("eager_dispatch_baseline/lazy");

    for &len in ELEMENT_COUNTS {
        let lhs = eager(&runtime, vec![len]);
        let rhs = eager(&runtime, vec![len]);
        let axes = [0_usize];

        group.bench_with_input(BenchmarkId::new("neg_f64", len), &len, |b, _| {
            b.iter(|| consume_lazy(black_box(&lhs).neg().expect("neg should succeed")));
        });
        group.bench_with_input(BenchmarkId::new("add_f64", len), &len, |b, _| {
            b.iter(|| {
                consume_lazy(
                    black_box(&lhs)
                        .add(black_box(&rhs))
                        .expect("add should succeed"),
                )
            });
        });
        group.bench_with_input(BenchmarkId::new("reduce_sum_f64", len), &len, |b, _| {
            b.iter(|| {
                consume_lazy(
                    black_box(&lhs)
                        .reduce_sum(Some(&axes))
                        .expect("reduce_sum should succeed"),
                )
            });
        });
        let slice = SliceConfig {
            starts: vec![0],
            limits: vec![len],
            strides: vec![1],
        };
        group.bench_with_input(BenchmarkId::new("slice_f64", len), &len, |b, _| {
            b.iter(|| {
                consume_lazy(
                    black_box(&lhs)
                        .slice(black_box(slice.clone()))
                        .expect("slice should succeed"),
                )
            });
        });
    }

    for &size in MATRIX_SIZES {
        let lhs = eager(&runtime, vec![size, size]);
        let rhs = eager(&runtime, vec![size, size]);
        let config = matmul_config();
        group.bench_with_input(BenchmarkId::new("dot_general_f64", size), &size, |b, _| {
            b.iter(|| {
                consume_lazy(
                    black_box(&lhs)
                        .dot_general(black_box(&rhs), black_box(config.clone()))
                        .expect("dot_general should succeed"),
                )
            });
        });
    }
    group.finish();
}

fn bench_materialized(c: &mut Criterion) {
    let runtime = runtime();
    let mut group = c.benchmark_group("eager_dispatch_baseline/materialized");

    for &len in ELEMENT_COUNTS {
        let lhs = eager(&runtime, vec![len]);
        let rhs = eager(&runtime, vec![len]);
        let axes = [0_usize];

        group.bench_with_input(BenchmarkId::new("neg_f64", len), &len, |b, _| {
            b.iter(|| consume_materialized(black_box(&lhs).neg().expect("neg should succeed")));
        });
        group.bench_with_input(BenchmarkId::new("add_f64", len), &len, |b, _| {
            b.iter(|| {
                consume_materialized(
                    black_box(&lhs)
                        .add(black_box(&rhs))
                        .expect("add should succeed"),
                )
            });
        });
        group.bench_with_input(BenchmarkId::new("reduce_sum_f64", len), &len, |b, _| {
            b.iter(|| {
                consume_materialized(
                    black_box(&lhs)
                        .reduce_sum(Some(&axes))
                        .expect("reduce_sum should succeed"),
                )
            });
        });
        let slice = SliceConfig {
            starts: vec![0],
            limits: vec![len],
            strides: vec![1],
        };
        group.bench_with_input(BenchmarkId::new("slice_f64", len), &len, |b, _| {
            b.iter(|| {
                consume_materialized(
                    black_box(&lhs)
                        .slice(black_box(slice.clone()))
                        .expect("slice should succeed"),
                )
            });
        });
    }

    for &size in MATRIX_SIZES {
        let lhs = eager(&runtime, vec![size, size]);
        let rhs = eager(&runtime, vec![size, size]);
        let config = matmul_config();
        group.bench_with_input(BenchmarkId::new("dot_general_f64", size), &size, |b, _| {
            b.iter(|| {
                consume_materialized(
                    black_box(&lhs)
                        .dot_general(black_box(&rhs), black_box(config.clone()))
                        .expect("dot_general should succeed"),
                )
            });
        });
    }
    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5))
        .sample_size(100)
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_lazy, bench_materialized
}
criterion_main!(benches);
