//! GPU execution-route matrix: the coexisting CUDA entry mechanisms on one
//! logical contraction, reported separately for enqueue and synchronized
//! completion.
//!
//! Issue #1926 / umbrella #1929. `CudaBackend` implements the `Tensor*`
//! operation traits by calling the same functions as its `BackendSession`
//! impl — the backend is effectively its own session — so a one-shot call does
//! not open a session, it runs *unbatched*. Two spellings therefore coexist:
//!
//! * `oneshot` — `TensorDot::dot_general(&mut backend, ..)` on `CudaBackend`
//!   itself.
//! * `session` — `with_backend_session` plus `dot_general_read` on the borrowed
//!   session.
//!
//! The route/API unification deletes the `oneshot` spelling, so its rows are
//! before-only references.
//!
//! Timing modes. The umbrella requires GPU enqueue cost and synchronized
//! completion cost to be reported distinctly, because a per-call "round trip"
//! number mixes launch cost with `synchronize` cost:
//!
//! * `round_trip` — one operation, then `synchronize`, per iteration.
//! * `enqueue_batch16` — 16 operations enqueued, one `synchronize`, timed as a
//!   whole and divided by 16. The batch bounds the outstanding launch queue so
//!   the measurement cannot queue unbounded work.
//!
//! `sync_empty` measures `synchronize` with nothing outstanding, so a reader can
//! subtract it from `round_trip` to recover enqueue-plus-completion latency.
//!
//! Every arm uploads its operands before the timed region and validates the
//! result outside it.

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tenferro_gpu::cuda::{gpu_available, upload_tensor, CudaBackend, CudaDeviceId};
use tenferro_tensor::{
    BackendSessionHost, DotGeneralConfig, Tensor, TensorDot, TensorRead, TensorScalar,
};

const SIZES: &[usize] = &[8, 64, 256];
const BATCH: usize = 16;

fn matmul_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    }
}

fn matrix<T>(rows: usize, cols: usize) -> Tensor
where
    T: TensorScalar + From<f32>,
{
    let len = rows * cols;
    let data = (0..len)
        .map(|index| T::from(((index % 97) as f32 + 1.0) * 0.001))
        .collect();
    Tensor::from_vec_col_major(vec![rows, cols], data).expect("benchmark matrix")
}

/// One-shot route: the operation method on the backend object.
fn oneshot(backend: &mut CudaBackend, lhs: &Tensor, rhs: &Tensor, config: &DotGeneralConfig) {
    let out = backend
        .dot_general(lhs, rhs, config)
        .expect("one-shot dot_general should succeed");
    black_box(out);
}

/// Session route: the same kernel through a borrowed session.
fn session(backend: &mut CudaBackend, lhs: &Tensor, rhs: &Tensor, config: &DotGeneralConfig) {
    backend
        .with_backend_session(|exec| {
            exec.dot_general_read(
                TensorRead::from_tensor(lhs),
                TensorRead::from_tensor(rhs),
                config,
            )
        })
        .expect("session dot_general should succeed");
}

fn bench_size<T>(c: &mut Criterion, label: &str, size: usize)
where
    T: TensorScalar + From<f32> + 'static,
{
    let mut backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).expect("CUDA backend");
    let lhs = upload_tensor(backend.runtime(), &matrix::<T>(size, size)).expect("upload lhs");
    let rhs = upload_tensor(backend.runtime(), &matrix::<T>(size, size)).expect("upload rhs");
    let config = matmul_config();

    // Validation outside the timed region, on both routes.
    let device_out = backend
        .dot_general(&lhs, &rhs, &config)
        .expect("validation dot_general");
    backend.runtime().synchronize().expect("validation sync");
    assert_eq!(device_out.shape(), &[size, size], "{label}: output shape");
    session(&mut backend, &lhs, &rhs, &config);
    backend.runtime().synchronize().expect("validation sync");

    let mut group = c.benchmark_group(format!("route_matrix_gpu/dot_general_{label}"));
    group.bench_with_input(
        BenchmarkId::new("oneshot/round_trip", size),
        &size,
        |bench, _| {
            bench.iter(|| {
                oneshot(&mut backend, black_box(&lhs), black_box(&rhs), &config);
                backend.runtime().synchronize().expect("sync");
            });
        },
    );
    group.bench_with_input(
        BenchmarkId::new("session/round_trip", size),
        &size,
        |bench, _| {
            bench.iter(|| {
                session(&mut backend, black_box(&lhs), black_box(&rhs), &config);
                backend.runtime().synchronize().expect("sync");
            });
        },
    );
    group.bench_with_input(
        BenchmarkId::new("oneshot/enqueue_batch16", size),
        &size,
        |bench, _| {
            bench.iter(|| {
                for _ in 0..BATCH {
                    oneshot(&mut backend, black_box(&lhs), black_box(&rhs), &config);
                }
                backend.runtime().synchronize().expect("sync");
            });
        },
    );
    group.bench_with_input(
        BenchmarkId::new("session/enqueue_batch16", size),
        &size,
        |bench, _| {
            bench.iter(|| {
                for _ in 0..BATCH {
                    session(&mut backend, black_box(&lhs), black_box(&rhs), &config);
                }
                backend.runtime().synchronize().expect("sync");
            });
        },
    );
    group.finish();
}

fn bench_sync_empty(c: &mut Criterion) {
    let backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).expect("CUDA backend");
    let mut group = c.benchmark_group("route_matrix_gpu/sync_empty");
    group.bench_function("synchronize", |bench| {
        bench.iter(|| {
            backend.runtime().synchronize().expect("sync");
        });
    });
    group.finish();
}

fn route_matrix_gpu(c: &mut Criterion) {
    if !gpu_available() {
        eprintln!("skipping GPU route matrix: no CUDA device available");
        return;
    }
    for &size in SIZES {
        bench_size::<f32>(c, "f32", size);
        bench_size::<f64>(c, "f64", size);
    }
    bench_sync_empty(c);
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
    targets = route_matrix_gpu
}
criterion_main!(benches);
