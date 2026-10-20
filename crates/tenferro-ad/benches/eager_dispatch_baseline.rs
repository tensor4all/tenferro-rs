//! Eager standard-op dispatch baseline (lazy vs materialized consume).
//!
//! Each benchmark group reuses one caller-owned [`CpuEngine`], the supported
//! caller pattern. Do not construct an engine inside `b.iter`: an engine owns
//! its contraction workspace and plan cache, so re-creating one per iteration
//! measures cold workspace/plan turnover, not eager dispatch. (The per-provider
//! worker-handle registry that used to grow with every dropped engine was fixed
//! in tcontract-rs#134, so the cold cost no longer accumulates; it is still
//! cold.)
//!
//! [`CpuEngine`]: tenferro_cpu::CpuEngine

use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_tensor::{DotGeneralConfig, SliceConfig, Tensor};

const ELEMENT_COUNTS: &[usize] = &[1, 8, 64];
const MATRIX_SIZES: &[usize] = &[1, 2];

fn runtime() -> Arc<EagerRuntime> {
    EagerRuntime::new().unwrap()
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

fn consume_materialized(output: EagerTensor, engine: &mut tenferro_cpu::CpuEngine) {
    let output = output.to_tensor(engine).expect("output should materialize");
    black_box(output.shape());
    black_box(output.as_slice::<f64>().expect("f64 output")[0]);
}

fn matmul_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    }
}

fn bench_lazy(c: &mut Criterion) {
    let runtime = runtime();
    let mut engine = tenferro_cpu::CpuEngine::serial();
    let mut group = c.benchmark_group("eager_dispatch_baseline/lazy");

    for &len in ELEMENT_COUNTS {
        let lhs = eager(&runtime, vec![len]);
        let rhs = eager(&runtime, vec![len]);
        let axes = [0_usize];

        group.bench_with_input(BenchmarkId::new("neg_f64", len), &len, |b, _| {
            b.iter(|| {
                consume_lazy(
                    runtime
                        .with_eager_session(&mut engine, |session| session.neg(black_box(&lhs)))
                        .expect("neg should succeed"),
                )
            });
        });
        group.bench_with_input(BenchmarkId::new("add_f64", len), &len, |b, _| {
            b.iter(|| {
                consume_lazy(
                    runtime
                        .with_eager_session(&mut engine, |session| {
                            session.add(black_box(&lhs), black_box(&rhs))
                        })
                        .expect("add should succeed"),
                )
            });
        });
        group.bench_with_input(BenchmarkId::new("reduce_sum_f64", len), &len, |b, _| {
            b.iter(|| {
                consume_lazy(
                    runtime
                        .with_eager_session(&mut engine, |session| {
                            session.reduce_sum(black_box(&lhs), Some(&axes))
                        })
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
                    runtime
                        .with_eager_session(&mut engine, |session| {
                            session.slice(black_box(&lhs), black_box(slice.clone()))
                        })
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
                    runtime
                        .with_eager_session(&mut engine, |session| {
                            session.dot_general(
                                black_box(&lhs),
                                black_box(&rhs),
                                black_box(config.clone()),
                            )
                        })
                        .expect("dot_general should succeed"),
                )
            });
        });
    }
    group.finish();
}

fn bench_materialized(c: &mut Criterion) {
    let runtime = runtime();
    let mut engine = tenferro_cpu::CpuEngine::serial();
    let mut group = c.benchmark_group("eager_dispatch_baseline/materialized");

    for &len in ELEMENT_COUNTS {
        let lhs = eager(&runtime, vec![len]);
        let rhs = eager(&runtime, vec![len]);
        let axes = [0_usize];

        group.bench_with_input(BenchmarkId::new("neg_f64", len), &len, |b, _| {
            b.iter(|| {
                consume_materialized(
                    runtime
                        .with_eager_session(&mut engine, |session| session.neg(black_box(&lhs)))
                        .expect("neg should succeed"),
                    &mut engine,
                )
            });
        });
        group.bench_with_input(BenchmarkId::new("add_f64", len), &len, |b, _| {
            b.iter(|| {
                consume_materialized(
                    runtime
                        .with_eager_session(&mut engine, |session| {
                            session.add(black_box(&lhs), black_box(&rhs))
                        })
                        .expect("add should succeed"),
                    &mut engine,
                )
            });
        });
        group.bench_with_input(BenchmarkId::new("reduce_sum_f64", len), &len, |b, _| {
            b.iter(|| {
                consume_materialized(
                    runtime
                        .with_eager_session(&mut engine, |session| {
                            session.reduce_sum(black_box(&lhs), Some(&axes))
                        })
                        .expect("reduce_sum should succeed"),
                    &mut engine,
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
                    runtime
                        .with_eager_session(&mut engine, |session| {
                            session.slice(black_box(&lhs), black_box(slice.clone()))
                        })
                        .expect("slice should succeed"),
                    &mut engine,
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
                    runtime
                        .with_eager_session(&mut engine, |session| {
                            session.dot_general(
                                black_box(&lhs),
                                black_box(&rhs),
                                black_box(config.clone()),
                            )
                        })
                        .expect("dot_general should succeed"),
                    &mut engine,
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
