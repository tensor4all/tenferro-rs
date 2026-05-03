//! Tropical matmul composition-vs-fusion benchmark.
//!
//! Compares forward-only evaluation cost of two semantically identical
//! tropical (max-plus) matmul paths:
//!
//! - **Composition**: `tropical_dot_general` lowering to
//!   `BroadcastInDim + Add + ReduceMax` over the core op vocabulary.
//! - **Fused `ExtensionOp`**: `tropical_dot_general_fused`
//!   routing through the `FusedTropicalDotGeneralOp` `ExtensionOp` whose
//!   `eager_execute` is a direct triple-loop max-plus GEMM over host data.
//!
//! Each bench iteration rebuilds the traced graph and evaluates once so
//! the measurement reflects end-to-end cost from a user call site. The
//! `Engine` is reused across iterations so `engine.get_or_compile` hits
//! cache on all iterations past the first — the first iteration pays
//! compile cost, subsequent iterations pay only execution cost.
//! Criterion's warm-up discards the first iterations, so the measured
//! median is the steady-state per-call cost.
//!
//! `f64` only. Square matrices `N x N` for sizes in `MATRIX_SIZES`.
//!
//! Output layout in criterion reports (`target/criterion/...`) groups by
//! size, with two paths per size (`composition`, `fused_ext`).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use tenferro::{CpuBackend, Engine, TracedTensor};
use tenferro_ext_tropical::traced::{tropical_dot_general, tropical_dot_general_fused};

/// Matrix sizes to cover — see the bench task specification.
///
/// - 16: tiny, startup / graph-build overhead should dominate for both paths.
/// - 64: small; fused eager_execute should be cheap.
/// - 128: medium-small.
/// - 256: medium; composition path allocates an `M*K*N` 3D tensor
///   (≈ 16.7M f64 = 128 MiB at 256 — big enough for memory bandwidth to bite).
/// - 384: larger; keeps total bench time bounded.
const MATRIX_SIZES: &[usize] = &[16, 64, 128, 256, 384];

/// Column-major deterministic test matrix of size `n x n`.
fn build_matrix(n: usize, seed: f64) -> Vec<f64> {
    let mut v = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            // Spread the values so ties are rare in the max-plus reduction.
            v.push(seed + (i as f64) * 0.25 - (j as f64) * 0.125);
        }
    }
    v
}

fn bench_tropical_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("tropical_matmul");

    // Reuse a single engine across all sizes and paths so compile cache hits
    // on structurally identical ExecIR.
    let mut engine = Engine::new(CpuBackend::new());

    for &n in MATRIX_SIZES {
        let a_data = build_matrix(n, 1.0);
        let b_data = build_matrix(n, 7.5);
        let shape = vec![n, n];

        // --- Composition path ---
        group.bench_with_input(BenchmarkId::new("composition", n), &n, |bencher, _| {
            bencher.iter(|| {
                let a = TracedTensor::from_vec(shape.clone(), a_data.clone());
                let b = TracedTensor::from_vec(shape.clone(), b_data.clone());
                let mut c_traced = tropical_dot_general(black_box(&a), black_box(&b));
                let out = c_traced.eval(&mut engine).expect("composition eval");
                // Touch the output to keep the compiler from DCE'ing.
                black_box(out.shape().len())
            });
        });

        // --- Fused ExtensionOp path ---
        group.bench_with_input(BenchmarkId::new("fused_ext", n), &n, |bencher, _| {
            bencher.iter(|| {
                let a = TracedTensor::from_vec(shape.clone(), a_data.clone());
                let b = TracedTensor::from_vec(shape.clone(), b_data.clone());
                let mut c_traced = tropical_dot_general_fused(black_box(&a), black_box(&b));
                let out = c_traced.eval(&mut engine).expect("fused eval");
                black_box(out.shape().len())
            });
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    // Short warm-up / modest sample count to keep total bench time bounded
    // on a single-host evidence-gathering run. The `measurement_time` and
    // `sample_size` can still be overridden from the command line via
    // `-- --measurement-time 5 --sample-size 20`.
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(3))
        .sample_size(10);
    targets = bench_tropical_matmul
}

criterion_main!(benches);
