//! Session-entry cost benchmark for the `_in` concrete-ops surface.
//!
//! Exact 10-op chain: three repetitions of `add -> exp -> mul` (9 ops)
//! followed by a final `reduce_sum([0])` (10th). Two shape arms (no-broadcast
//! 1x8 duplicate path vs 1x1/1x8 real reshape+broadcast) and two execution
//! arms (one-shot `TensorOpsExt`, one session entry per op = 10 entries, vs
//! `TensorSessionOpsExt` inside one `with_backend_session` = 1 entry).
//!
//! Constants are chosen so the chain does not overflow (`exp` of large
//! values); the result is validated outside the timed region.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{Tensor, TensorOpsExt, TensorSessionOpsExt};
use tenferro_tensor::BackendSessionHost;

/// `a` is a 1x8 constant row in the no-broadcast arm and a 1x1 row (the
/// singleton broadcast source) in the broadcast arm.
fn operand_a(broadcast: bool) -> Tensor {
    let value = vec![0.5_f64];
    if broadcast {
        Tensor::from_vec_col_major(vec![1], value).expect("benchmark tensor")
    } else {
        Tensor::from_vec_col_major(vec![8], vec![0.5_f64; 8]).expect("benchmark tensor")
    }
}

fn operand_b() -> Tensor {
    Tensor::from_vec_col_major(vec![8], vec![1.0_f64; 8]).expect("benchmark tensor")
}

/// 10 ops, each entering its own backend session.
fn run_chain_one_shot(a: &Tensor, b: &Tensor, backend: &mut CpuBackend) -> Tensor {
    let x = a.add(b, backend).expect("add 1");
    let x = x.exp(backend).expect("exp 1");
    let x = x.mul(a, backend).expect("mul 1");
    let x = x.add(b, backend).expect("add 2");
    let x = x.exp(backend).expect("exp 2");
    let x = x.mul(a, backend).expect("mul 2");
    let x = x.add(b, backend).expect("add 3");
    let x = x.exp(backend).expect("exp 3");
    let x = x.mul(a, backend).expect("mul 3");
    x.reduce_sum(&[0], backend).expect("reduce_sum")
}

/// The same 10 ops inside one backend session (1 entry).
fn run_chain_one_session(a: &Tensor, b: &Tensor, backend: &mut CpuBackend) -> Tensor {
    backend.with_backend_session(|session| {
        let x = a.add_in(b, session).expect("add 1");
        let x = x.exp_in(session).expect("exp 1");
        let x = x.mul_in(a, session).expect("mul 1");
        let x = x.add_in(b, session).expect("add 2");
        let x = x.exp_in(session).expect("exp 2");
        let x = x.mul_in(a, session).expect("mul 2");
        let x = x.add_in(b, session).expect("add 3");
        let x = x.exp_in(session).expect("exp 3");
        let x = x.mul_in(a, session).expect("mul 3");
        x.reduce_sum_in(&[0], session).expect("reduce_sum")
    })
}

fn bench_session_chain(c: &mut Criterion) {
    for (arm, broadcast) in [("no_broadcast", false), ("broadcast", true)] {
        let mut group = c.benchmark_group(format!("session_chain/{arm}"));
        let a = operand_a(broadcast);
        let b = operand_b();
        let mut backend = CpuBackend::new();

        // Validation outside the timed region: finite scalar result, and the
        // two execution arms agree.
        let one_shot = run_chain_one_shot(&a, &b, &mut backend);
        assert!(one_shot.shape().is_empty(), "chain must reduce to a scalar");
        assert!(one_shot.as_slice::<f64>().unwrap()[0].is_finite());
        let one_session = run_chain_one_session(&a, &b, &mut backend);
        assert!(one_session.as_slice::<f64>().unwrap()[0].is_finite());
        assert_eq!(
            one_shot.as_slice::<f64>().unwrap(),
            one_session.as_slice::<f64>().unwrap()
        );

        group.bench_function("one_shot", |bench| {
            bench.iter(|| {
                let out = run_chain_one_shot(black_box(&a), black_box(&b), &mut backend);
                black_box(out);
            });
        });
        group.bench_function("one_session", |bench| {
            bench.iter(|| {
                let out = run_chain_one_session(black_box(&a), black_box(&b), &mut backend);
                black_box(out);
            });
        });
        group.finish();
    }
}

criterion_group!(benches, bench_session_chain);
criterion_main!(benches);
