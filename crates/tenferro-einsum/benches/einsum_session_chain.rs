//! Session-entry cost benchmark for a prepared [`ConcreteEinsumPlan`]:
//! repeated executes inside a single borrowed session, and a mixed chain
//! (einsum + exp + reduce_sum) in the same session.
//!
//! Workload (design doc "Prototype specification (PR C)"): a prepared plan
//! for `"ij,jk->ik"` on 8×8 f64 column-major inputs, executed 10 times per
//! iteration, on a one-worker `CpuBackend::with_threads(1)`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tenferro_cpu::CpuBackend;
use tenferro_einsum::ConcreteEinsumPlan;
use tenferro_runtime::{Tensor, TensorSessionOpsExt};
use tenferro_tensor::BackendSessionHost;

const N: usize = 8;
const CALLS: usize = 10;

fn f64_square(seed: usize) -> Tensor {
    let data = (0..N * N)
        .map(|idx| ((idx * 17 + seed * 31 + 7) % 997) as f64 / 997.0 - 0.5)
        .collect();
    Tensor::from_vec_col_major(vec![N, N], data).unwrap()
}

fn bench_einsum_session_chain(c: &mut Criterion) {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let lhs = f64_square(1);
    let rhs = f64_square(2);
    let plan = ConcreteEinsumPlan::prepare([&lhs, &rhs], "ij,jk->ik").unwrap();

    // Correctness validation outside the timed region: shape [8, 8], finite.
    let check = backend
        .with_backend_session(|session| plan.execute([&lhs, &rhs], session))
        .unwrap();
    assert_eq!(check.shape(), &[8, 8]);
    assert!(check
        .as_slice::<f64>()
        .unwrap()
        .iter()
        .all(|value| value.is_finite()));

    let mut group = c.benchmark_group("einsum_session_chain/f64_8x8_10calls/threads1");

    group.bench_function("B_one_session", |b| {
        b.iter(|| {
            backend.with_backend_session(|session| {
                let inputs = black_box([&lhs, &rhs]);
                for _ in 0..CALLS {
                    let out = plan.execute(inputs, session).unwrap();
                    black_box(out);
                }
            });
        });
    });

    group.bench_function("C_mixed_one_session", |b| {
        b.iter(|| {
            backend.with_backend_session(|session| {
                let inputs = black_box([&lhs, &rhs]);
                for _ in 0..CALLS {
                    let x = plan.execute(inputs, session).unwrap();
                    let x = x.exp(session).unwrap();
                    let out = x.reduce_sum(&[0], session).unwrap();
                    black_box(out);
                }
            });
        });
    });

    group.finish();
}

criterion_group!(benches, bench_einsum_session_chain);
criterion_main!(benches);
