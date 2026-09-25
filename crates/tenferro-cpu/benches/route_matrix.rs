//! CPU execution-route matrix: the coexisting entry mechanisms measured on the
//! same logical operation.
//!
//! Issue #1926 / umbrella #1929. The CPU backend currently exposes three
//! distinct ways to reach the same kernel:
//!
//! * `oneshot` — an operation method on `CpuBackend` itself
//!   (`TensorElementwise::add(&mut backend, ..)`), which enters through
//!   `install_with_pool_context` (permit + `CpuOperationEntry`, no session).
//! * `session` — the same kernel through `BackendSessionHost::with_backend_session`
//!   and an `_read` method on the borrowed session, which enters through
//!   `run_backend_session_cached` (permit + `CpuOperationEntry` + session
//!   construction + pool loan).
//! * `scope` — an operation method on a backend clone inside
//!   `CpuBackend::with_execution_scope`, which reuses the scope's permit.
//!
//! The route/API unification deletes the `oneshot` spelling and keeps the
//! session surface. The pair that measures the price of that unification on
//! small operations is `oneshot/single` (before) versus `session/single`
//! (after). The marginal arms expose the per-operation cost once an entry is
//! already open, which is what a caller amortizes; those must not regress.
//!
//! All arms run on an explicit one-worker backend (`CpuBackend::with_threads(1)`),
//! per the repository rule for dispatch/overhead measurement. Every arm
//! validates its result outside the timed region.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::config::SliceConfig;
use tenferro_tensor::{
    BackendSessionHost, DType, DotGeneralConfig, Tensor, TensorDot, TensorElementwise,
    TensorIndexing, TensorRead, TensorReduction, TensorStructural,
};

/// Operations per entry for the `marginal` arms.
const CHAIN_LEN: usize = 16;

/// Elementwise sizes: two overhead-scale and two throughput-scale.
const ELEMENTWISE_LENS: &[usize] = &[1, 64, 4096, 65536];
/// Contraction sizes.
const DOT_SIZES: &[usize] = &[2, 64, 256];
/// Reduction and indexing sizes.
const STRUCTURAL_LENS: &[usize] = &[64, 65536];

fn backend() -> CpuBackend {
    CpuBackend::with_threads(1).expect("one-worker faer backend should construct")
}

fn full_tensor(shape: Vec<usize>) -> Tensor {
    let len: usize = shape.iter().product();
    Tensor::from_vec_col_major(shape, vec![1.0_f64; len]).expect("benchmark tensor should be valid")
}

fn dot_config() -> DotGeneralConfig {
    const NONE: &[usize] = &[];
    DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: NONE.into(),
        rhs_batch_dims: NONE.into(),
    }
}

fn slice_config(len: usize) -> SliceConfig {
    SliceConfig {
        starts: vec![0],
        limits: vec![len],
        strides: vec![1],
    }
}

// ---------------------------------------------------------------------------
// Per-operation entry cost (one operation, one entry).
// ---------------------------------------------------------------------------

fn add_oneshot(ops: &mut CpuBackend, a: &Tensor, b: &Tensor) -> Tensor {
    ops.add(a, b).expect("oneshot add should succeed")
}

fn add_session(owner: &mut CpuBackend, a: &Tensor, b: &Tensor) -> Tensor {
    owner
        .with_backend_session(|session| {
            session.add_read(TensorRead::from_tensor(a), TensorRead::from_tensor(b))
        })
        .expect("session add should succeed")
}

fn add_scope(owner: &CpuBackend, ops: &mut CpuBackend, a: &Tensor, b: &Tensor) -> Tensor {
    owner
        .with_execution_scope(|| ops.add(a, b))
        .expect("scope admission should succeed")
        .expect("scope add should succeed")
}

fn dot_oneshot(ops: &mut CpuBackend, a: &Tensor, b: &Tensor) -> Tensor {
    ops.dot_general(a, b, &dot_config())
        .expect("oneshot dot should succeed")
}

fn dot_session(owner: &mut CpuBackend, a: &Tensor, b: &Tensor) -> Tensor {
    owner
        .with_backend_session(|session| {
            session.dot_general_read(
                TensorRead::from_tensor(a),
                TensorRead::from_tensor(b),
                &dot_config(),
            )
        })
        .expect("session dot should succeed")
}

fn dot_scope(owner: &CpuBackend, ops: &mut CpuBackend, a: &Tensor, b: &Tensor) -> Tensor {
    owner
        .with_execution_scope(|| ops.dot_general(a, b, &dot_config()))
        .expect("scope admission should succeed")
        .expect("scope dot should succeed")
}

fn reduce_oneshot(ops: &mut CpuBackend, a: &Tensor) -> Tensor {
    ops.reduce_sum(a, &[0]).expect("oneshot reduce_sum should succeed")
}

fn reduce_session(owner: &mut CpuBackend, a: &Tensor) -> Tensor {
    owner
        .with_backend_session(|session| session.reduce_sum_read(TensorRead::from_tensor(a), &[0]))
        .expect("session reduce_sum should succeed")
}

fn reduce_scope(owner: &CpuBackend, ops: &mut CpuBackend, a: &Tensor) -> Tensor {
    owner
        .with_execution_scope(|| ops.reduce_sum(a, &[0]))
        .expect("scope admission should succeed")
        .expect("scope reduce_sum should succeed")
}

fn slice_oneshot(ops: &mut CpuBackend, a: &Tensor, config: &SliceConfig) -> Tensor {
    ops.slice(a, config).expect("oneshot slice should succeed")
}

fn slice_session(owner: &mut CpuBackend, a: &Tensor, config: &SliceConfig) -> Tensor {
    owner
        .with_backend_session(|session| {
            // `TensorIndexing` has no `slice_read`; the session path for an
            // indexed operation is the capability dispatch on the session.
            TensorIndexing::slice(session, a, config)
        })
        .expect("session slice should succeed")
}

fn slice_scope(
    owner: &CpuBackend,
    ops: &mut CpuBackend,
    a: &Tensor,
    config: &SliceConfig,
) -> Tensor {
    owner
        .with_execution_scope(|| ops.slice(a, config))
        .expect("scope admission should succeed")
        .expect("scope slice should succeed")
}

// ---------------------------------------------------------------------------
// Marginal per-operation cost (CHAIN_LEN operations inside one entry).
// ---------------------------------------------------------------------------

fn add_marginal_oneshot(ops: &mut CpuBackend, a: &Tensor, b: &Tensor) -> Tensor {
    // CHAIN_LEN operations: the first produces the owned seed, the rest chain.
    let mut x = ops.add(a, b).expect("oneshot add should succeed");
    for _ in 1..CHAIN_LEN {
        x = ops.add(&x, b).expect("oneshot add should succeed");
    }
    x
}

fn add_marginal_session(owner: &mut CpuBackend, a: &Tensor, b: &Tensor) -> Tensor {
    owner.with_backend_session(|session| {
        let mut x = session
            .add_read(TensorRead::from_tensor(a), TensorRead::from_tensor(b))
            .expect("session add should succeed");
        for _ in 1..CHAIN_LEN {
            x = session
                .add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(b))
                .expect("session add should succeed");
        }
        x
    })
}

fn add_marginal_scope(owner: &CpuBackend, ops: &mut CpuBackend, a: &Tensor, b: &Tensor) -> Tensor {
    owner
        .with_execution_scope(|| {
            let mut x = ops.add(a, b).expect("scope add should succeed");
            for _ in 1..CHAIN_LEN {
                x = ops.add(&x, b).expect("scope add should succeed");
            }
            x
        })
        .expect("scope admission should succeed")
}

// ---------------------------------------------------------------------------
// Groups.
// ---------------------------------------------------------------------------

fn bench_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("route_matrix/elementwise_add_f64");

    for &len in ELEMENTWISE_LENS {
        let a = full_tensor(vec![len]);
        let b = full_tensor(vec![len]);
        let mut owner = backend();
        let mut ops = owner.clone();

        // Validation outside the timed region: both operands are all-ones, so
        // the sum is 2 everywhere.
        let expected = 2.0_f64;
        for (name, value) in [
            ("oneshot", add_oneshot(&mut ops, &a, &b)),
            ("session", add_session(&mut owner, &a, &b)),
            ("scope", add_scope(&owner, &mut ops, &a, &b)),
        ] {
            assert_eq!(value.shape(), &[len], "{name} shape");
            assert_eq!(
                value.as_slice::<f64>().unwrap()[0],
                expected,
                "{name} value"
            );
        }

        group.bench_with_input(BenchmarkId::new("oneshot/single", len), &len, |bench, _| {
            bench.iter(|| black_box(add_oneshot(&mut ops, black_box(&a), black_box(&b))));
        });
        group.bench_with_input(BenchmarkId::new("session/single", len), &len, |bench, _| {
            bench.iter(|| black_box(add_session(&mut owner, black_box(&a), black_box(&b))));
        });
        group.bench_with_input(BenchmarkId::new("scope/single", len), &len, |bench, _| {
            bench.iter(|| black_box(add_scope(&owner, &mut ops, black_box(&a), black_box(&b))));
        });
        group.bench_with_input(
            BenchmarkId::new("oneshot/marginal16", len),
            &len,
            |bench, _| {
                bench.iter(|| {
                    black_box(add_marginal_oneshot(&mut ops, black_box(&a), black_box(&b)))
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("session/marginal16", len),
            &len,
            |bench, _| {
                bench.iter(|| {
                    black_box(add_marginal_session(&mut owner, black_box(&a), black_box(&b)))
                });
            },
        );
        group.bench_with_input(BenchmarkId::new("scope/marginal16", len), &len, |bench, _| {
            bench.iter(|| black_box(add_marginal_scope(&owner, &mut ops, black_box(&a), black_box(&b))));
        });
    }
    group.finish();
}

fn bench_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("route_matrix/dot_general_f64");

    for &size in DOT_SIZES {
        let a = full_tensor(vec![size, size]);
        let b = full_tensor(vec![size, size]);
        let mut owner = backend();
        let mut ops = owner.clone();

        let expected = size as f64;
        for (name, value) in [
            ("oneshot", dot_oneshot(&mut ops, &a, &b)),
            ("session", dot_session(&mut owner, &a, &b)),
            ("scope", dot_scope(&owner, &mut ops, &a, &b)),
        ] {
            assert_eq!(value.shape(), &[size, size], "{name} shape");
            assert_eq!(value.as_slice::<f64>().unwrap()[0], expected, "{name} value");
        }

        group.bench_with_input(BenchmarkId::new("oneshot/single", size), &size, |bench, _| {
            bench.iter(|| black_box(dot_oneshot(&mut ops, black_box(&a), black_box(&b))));
        });
        group.bench_with_input(BenchmarkId::new("session/single", size), &size, |bench, _| {
            bench.iter(|| black_box(dot_session(&mut owner, black_box(&a), black_box(&b))));
        });
        group.bench_with_input(BenchmarkId::new("scope/single", size), &size, |bench, _| {
            bench.iter(|| black_box(dot_scope(&owner, &mut ops, black_box(&a), black_box(&b))));
        });
    }
    group.finish();
}

fn bench_reduce(c: &mut Criterion) {
    let mut group = c.benchmark_group("route_matrix/reduce_sum_f64");

    for &len in STRUCTURAL_LENS {
        let a = full_tensor(vec![len]);
        let mut owner = backend();
        let mut ops = owner.clone();

        let expected = len as f64;
        for (name, value) in [
            ("oneshot", reduce_oneshot(&mut ops, &a)),
            ("session", reduce_session(&mut owner, &a)),
            ("scope", reduce_scope(&owner, &mut ops, &a)),
        ] {
            assert!(value.shape().is_empty(), "{name} shape");
            assert_eq!(value.as_slice::<f64>().unwrap()[0], expected, "{name} value");
        }

        group.bench_with_input(BenchmarkId::new("oneshot/single", len), &len, |bench, _| {
            bench.iter(|| black_box(reduce_oneshot(&mut ops, black_box(&a))));
        });
        group.bench_with_input(BenchmarkId::new("session/single", len), &len, |bench, _| {
            bench.iter(|| black_box(reduce_session(&mut owner, black_box(&a))));
        });
        group.bench_with_input(BenchmarkId::new("scope/single", len), &len, |bench, _| {
            bench.iter(|| black_box(reduce_scope(&owner, &mut ops, black_box(&a))));
        });
    }
    group.finish();
}

fn bench_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("route_matrix/slice_f64");

    for &len in STRUCTURAL_LENS {
        let a = full_tensor(vec![len]);
        let config = slice_config(len);
        let mut owner = backend();
        let mut ops = owner.clone();

        for (name, value) in [
            ("oneshot", slice_oneshot(&mut ops, &a, &config)),
            ("session", slice_session(&mut owner, &a, &config)),
            ("scope", slice_scope(&owner, &mut ops, &a, &config)),
        ] {
            assert_eq!(value.shape(), &[len], "{name} shape");
            assert_eq!(value.as_slice::<f64>().unwrap()[0], 1.0, "{name} value");
        }

        group.bench_with_input(BenchmarkId::new("oneshot/single", len), &len, |bench, _| {
            bench.iter(|| black_box(slice_oneshot(&mut ops, black_box(&a), black_box(&config))));
        });
        group.bench_with_input(BenchmarkId::new("session/single", len), &len, |bench, _| {
            bench.iter(|| black_box(slice_session(&mut owner, black_box(&a), black_box(&config))));
        });
        group.bench_with_input(BenchmarkId::new("scope/single", len), &len, |bench, _| {
            bench.iter(|| {
                black_box(slice_scope(&owner, &mut ops, black_box(&a), black_box(&config)))
            });
        });
    }
    group.finish();
}

/// Cast through the same three routes: a dtype-conversion kernel whose
/// session path is a provided method on the session surface rather than a
/// dedicated `_read` override.
fn bench_cast(c: &mut Criterion) {
    let mut group = c.benchmark_group("route_matrix/cast_f64_f32");
    let len = 4096;
    let a = full_tensor(vec![len]);
    let mut owner = backend();
    let mut ops = owner.clone();

    let oneshot = ops.cast(&a, DType::F32).expect("oneshot cast");
    let session_out = owner
        .with_backend_session(|session| session.cast(&a, DType::F32))
        .expect("session cast");
    assert_eq!(oneshot.as_slice::<f32>().unwrap()[0], 1.0);
    assert_eq!(session_out.as_slice::<f32>().unwrap()[0], 1.0);

    group.bench_function("oneshot/single", |bench| {
        bench.iter(|| black_box(ops.cast(black_box(&a), DType::F32)));
    });
    group.bench_function("session/single", |bench| {
        bench.iter(|| {
            black_box(
                owner
                    .with_backend_session(|session| session.cast(black_box(&a), DType::F32))
                    .expect("session cast"),
            )
        });
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
