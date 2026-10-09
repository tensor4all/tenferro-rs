//! Held concrete CPU session: entry and per-operation cost against the scoped
//! entry.
//!
//! Issue #1945 work package U2-concrete. The held session (#2053) owns the CPU
//! admission reservation, the engine's reusable resources and the caller's
//! affinity for as long as the caller keeps it open, so a sequence of operations
//! pays one entry instead of one per operation. This harness measures
//! *dispatch* — reaching the same kernel through the two surviving CPU entry
//! mechanisms — on an explicit one-worker backend, per the repository rule for
//! overhead measurement.
//!
//! Arms:
//!
//! * `scoped/*` — `BackendSessionHost::with_backend_session`, i.e. one admission,
//!   one session construction and one buffer-pool loan per operation.
//! * `held/*` — `CpuBackend::open_session` opened outside the timed region, one
//!   `CpuHeldSession::with_session` view per operation, closed outside it.
//! * `held/chain/*` — one held session, `CHAIN_LEN` operations per iteration. The
//!   reported per-iteration time covers the whole chain, so divide by `CHAIN_LEN`
//!   for the marginal per-operation cost.
//!
//! Not measured here: untracked eager and AD execution, which live in other
//! crates, and the bridge-inclusive cost a downstream adapter pays around these
//! numbers. Every arm validates its result outside the timed region.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{BackendSessionHost, DotGeneralConfig, Tensor, TensorRead, TensorWrite};

/// Operations per entry for the `held/chain/*` arms.
const CHAIN_LEN: usize = 16;

/// Contraction sizes: an overhead-scale GEMM in the shape of the small-GEMM
/// fixed-cost target, a small square case, and a throughput-scale control.
const DOT_SIZES: &[usize] = &[2, 95, 256];

fn backend() -> CpuBackend {
    CpuBackend::with_threads(1).expect("one-worker CPU backend should construct")
}

fn operand(rows: usize, cols: usize) -> Tensor {
    // Column-major, so `k` (the contracted dimension of the RHS) is the leading
    // axis of a `k x 1` operand, matching the small-GEMM fixed-cost case.
    Tensor::from_vec_col_major(vec![rows, cols], vec![1.0_f64; rows * cols])
        .expect("benchmark tensor should be valid")
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

fn dot_scoped(backend: &mut CpuBackend, lhs: &Tensor, rhs: &Tensor) -> Tensor {
    backend
        .with_backend_session(|session| {
            session.dot_general_read(
                TensorRead::from_tensor(lhs),
                TensorRead::from_tensor(rhs),
                &dot_config(),
            )
        })
        .expect("scoped admission should succeed")
        .expect("scoped dot should succeed")
}

fn dot_held(session: &mut tenferro_cpu::CpuHeldSession<'_>, lhs: &Tensor, rhs: &Tensor) -> Tensor {
    session
        .with_session(|view| {
            view.dot_general_read(
                TensorRead::from_tensor(lhs),
                TensorRead::from_tensor(rhs),
                &dot_config(),
            )
        })
        .expect("held dot should succeed")
}

/// The same small GEMM into a caller-owned output buffer: no result allocation
/// and no zero-fill in the timed region.
fn dot_into_scoped(backend: &mut CpuBackend, lhs: &Tensor, rhs: &Tensor, out: &mut Tensor) {
    backend
        .with_backend_session(|session| {
            session.dot_general_read_into(
                TensorRead::from_tensor(lhs),
                TensorRead::from_tensor(rhs),
                &dot_config(),
                TensorWrite::Tensor(out),
            )
        })
        .expect("scoped admission should succeed")
        .expect("scoped dot should succeed");
}

fn dot_into_held(
    session: &mut tenferro_cpu::CpuHeldSession<'_>,
    lhs: &Tensor,
    rhs: &Tensor,
    out: &mut Tensor,
) {
    session
        .with_session(|view| {
            view.dot_general_read_into(
                TensorRead::from_tensor(lhs),
                TensorRead::from_tensor(rhs),
                &dot_config(),
                TensorWrite::Tensor(out),
            )
        })
        .expect("held dot should succeed");
}

/// Cost of reaching an execution once: scoped entry with an empty callback
/// against opening and closing a held session.
fn bench_entry(c: &mut Criterion) {
    let mut group = c.benchmark_group("held_session_entry");

    group.bench_function("scoped/empty", |bench| {
        let mut backend = backend();
        bench.iter(|| {
            backend
                .with_backend_session(|_| black_box(()))
                .expect("scoped admission should succeed");
        });
    });

    group.bench_function("held/open_close", |bench| {
        let backend = backend();
        bench.iter(|| {
            let session = backend.open_session().expect("held entry should succeed");
            session.close().expect("affinity restores");
        });
    });

    group.finish();
}

fn bench_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("held_session_dot");

    for &size in DOT_SIZES {
        let lhs = operand(size, size);
        let rhs = operand(size, 1);

        group.bench_with_input(
            BenchmarkId::new("scoped/single", size),
            &size,
            |bench, _| {
                let mut backend = backend();
                // Validate outside the timed region.
                black_box(dot_scoped(&mut backend, &lhs, &rhs));
                bench.iter(|| black_box(dot_scoped(&mut backend, &lhs, &rhs)));
            },
        );

        group.bench_with_input(BenchmarkId::new("held/single", size), &size, |bench, _| {
            let backend = backend();
            let mut session = backend.open_session().expect("held entry should succeed");
            black_box(dot_held(&mut session, &lhs, &rhs));
            bench.iter(|| black_box(dot_held(&mut session, &lhs, &rhs)));
            session.close().expect("affinity restores");
        });

        group.bench_with_input(BenchmarkId::new("scoped/into", size), &size, |bench, _| {
            let mut backend = backend();
            let mut out = operand(size, 1);
            dot_into_scoped(&mut backend, &lhs, &rhs, &mut out);
            bench.iter(|| dot_into_scoped(&mut backend, &lhs, &rhs, &mut out));
        });

        group.bench_with_input(BenchmarkId::new("held/into", size), &size, |bench, _| {
            let backend = backend();
            let mut session = backend.open_session().expect("held entry should succeed");
            let mut out = operand(size, 1);
            dot_into_held(&mut session, &lhs, &rhs, &mut out);
            bench.iter(|| dot_into_held(&mut session, &lhs, &rhs, &mut out));
            session.close().expect("affinity restores");
        });

        group.bench_with_input(BenchmarkId::new("held/chain16", size), &size, |bench, _| {
            let backend = backend();
            let mut session = backend.open_session().expect("held entry should succeed");
            bench.iter(|| {
                for _ in 0..CHAIN_LEN {
                    black_box(dot_held(&mut session, &lhs, &rhs));
                }
            });
            session.close().expect("affinity restores");
        });
    }

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
    targets = bench_entry, bench_dot
}
criterion_main!(benches);
