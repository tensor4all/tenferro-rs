//! CPU per-operation cost matrix: the same logical operations called directly
//! on a `CpuEngine` (#2206).
//!
//! Operations take the engine as `&mut`; there is no session entry to amortize.
//! The `single` arms measure one operation per iteration and the `marginal16`
//! arm a chain of sixteen, which together show the fixed per-operation cost.
//!
//! All arms run on a serial engine (`CpuEngine::serial()`), per the repository
//! rule for dispatch/overhead measurement. Every arm validates its result
//! outside the timed region.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tenferro_cpu::CpuEngine;
use tenferro_tensor::config::SliceConfig;
use tenferro_tensor::{
    DType, DotGeneralConfig, Tensor, TensorDot, TensorElementwise, TensorIndexing, TensorRead,
    TensorReduction, TensorStructural,
};

/// Operations per iteration for the `marginal` arms.
const CHAIN_LEN: usize = 16;

/// Elementwise sizes: two overhead-scale and two throughput-scale.
const ELEMENTWISE_LENS: &[usize] = &[1, 64, 4096, 65536];
/// Contraction sizes.
const DOT_SIZES: &[usize] = &[2, 64, 256];
/// Reduction and indexing sizes.
const STRUCTURAL_LENS: &[usize] = &[64, 65536];

fn full_tensor(shape: Vec<usize>) -> Tensor {
    let len: usize = shape.iter().product();
    Tensor::from_vec_col_major(shape, vec![1.0_f64; len]).expect("benchmark tensor should be valid")
}

fn dot_config() -> DotGeneralConfig {
    const NONE: &[usize] = &[];
    DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: NONE.iter().copied().collect(),
        rhs_batch_dims: NONE.iter().copied().collect(),
    }
}

fn slice_config(len: usize) -> SliceConfig {
    SliceConfig {
        starts: vec![0],
        limits: vec![len],
        strides: vec![1],
    }
}

fn add(engine: &mut CpuEngine, a: &Tensor, b: &Tensor) -> Tensor {
    engine
        .add_read(TensorRead::from_tensor(a), TensorRead::from_tensor(b))
        .expect("add")
}

fn add_chain(engine: &mut CpuEngine, a: &Tensor, b: &Tensor) -> Tensor {
    let mut last = add(engine, a, b);
    for _ in 1..CHAIN_LEN {
        last = add(engine, a, b);
    }
    last
}

fn bench_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("route_matrix/elementwise_add_f64");
    for &len in ELEMENTWISE_LENS {
        let a = full_tensor(vec![len]);
        let b = full_tensor(vec![len]);
        let mut engine = CpuEngine::serial();
        let value = add(&mut engine, &a, &b);
        assert_eq!(value.shape(), &[len]);
        assert_eq!(value.as_slice::<f64>().unwrap()[0], 2.0);

        group.bench_with_input(BenchmarkId::new("engine/single", len), &len, |bench, _| {
            bench.iter(|| black_box(add(&mut engine, black_box(&a), black_box(&b))));
        });
        group.bench_with_input(
            BenchmarkId::new("engine/marginal16", len),
            &len,
            |bench, _| {
                bench.iter(|| black_box(add_chain(&mut engine, black_box(&a), black_box(&b))));
            },
        );
    }
    group.finish();
}

fn bench_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("route_matrix/dot_general_f64");
    let config = dot_config();
    for &size in DOT_SIZES {
        let a = full_tensor(vec![size, size]);
        let b = full_tensor(vec![size, size]);
        let mut engine = CpuEngine::serial();
        let dot = |engine: &mut CpuEngine, a: &Tensor, b: &Tensor| {
            engine
                .dot_general_read(
                    TensorRead::from_tensor(a),
                    TensorRead::from_tensor(b),
                    &config,
                )
                .expect("dot_general")
        };
        let value = dot(&mut engine, &a, &b);
        assert_eq!(value.shape(), &[size, size]);
        assert_eq!(value.as_slice::<f64>().unwrap()[0], size as f64);

        group.bench_with_input(
            BenchmarkId::new("engine/single", size),
            &size,
            |bench, _| {
                bench.iter(|| black_box(dot(&mut engine, black_box(&a), black_box(&b))));
            },
        );
    }
    group.finish();
}

fn bench_reduce(c: &mut Criterion) {
    let mut group = c.benchmark_group("route_matrix/reduce_sum_f64");
    for &len in STRUCTURAL_LENS {
        let a = full_tensor(vec![len]);
        let mut engine = CpuEngine::serial();
        let value = engine
            .reduce_sum_read(TensorRead::from_tensor(&a), &[0])
            .expect("reduce_sum");
        assert!(value.shape().is_empty());
        assert_eq!(value.as_slice::<f64>().unwrap()[0], len as f64);

        group.bench_with_input(BenchmarkId::new("engine/single", len), &len, |bench, _| {
            bench.iter(|| {
                black_box(
                    engine
                        .reduce_sum_read(TensorRead::from_tensor(black_box(&a)), &[0])
                        .expect("reduce_sum"),
                )
            });
        });
    }
    group.finish();
}

fn bench_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("route_matrix/slice_f64");
    for &len in STRUCTURAL_LENS {
        let a = full_tensor(vec![len]);
        let config = slice_config(len);
        let mut engine = CpuEngine::serial();
        let value = TensorIndexing::slice(&mut engine, &a, &config).expect("slice");
        assert_eq!(value.shape(), &[len]);
        assert_eq!(value.as_slice::<f64>().unwrap()[0], 1.0);

        group.bench_with_input(BenchmarkId::new("engine/single", len), &len, |bench, _| {
            bench.iter(|| {
                black_box(
                    TensorIndexing::slice(&mut engine, black_box(&a), black_box(&config))
                        .expect("slice"),
                )
            });
        });
    }
    group.finish();
}

/// Cast: a dtype-conversion kernel provided on the engine surface rather than
/// a dedicated `_read` method.
fn bench_cast(c: &mut Criterion) {
    let mut group = c.benchmark_group("route_matrix/cast_f64_f32");
    let len = 4096;
    let a = full_tensor(vec![len]);
    let mut engine = CpuEngine::serial();
    let out = engine.cast(&a, DType::F32).expect("cast");
    assert_eq!(out.as_slice::<f32>().unwrap()[0], 1.0);

    group.bench_function("engine/single", |bench| {
        bench.iter(|| black_box(engine.cast(black_box(&a), DType::F32).expect("cast")));
    });
    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(5))
        .sample_size(100)
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_elementwise, bench_dot, bench_reduce, bench_indexing, bench_cast
}
criterion_main!(benches);
