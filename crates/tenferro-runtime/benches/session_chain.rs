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

/// Phase-1 (issue #1680) chain operands.
///
/// Exact 10-op chain: `[2,2]` f64 through
/// sub -> log -> pow -> maximum -> neg (elementwise, operands stay positive
/// through `log`), reshape to `[4,1]`, transpose to `[1,4]`, scalar-broadcast
/// clamp, matmul by a `[4,1]` rhs to `[1,1]`, then cast F64 -> F32.
///
/// After `neg` the elements are `[-1, -1, -ln(3)^2, -ln(4)^2]`, all of which
/// lie above the clamp upper bound `-2`, so the clamp pulls every element to
/// `-2` and the matmul sums them to the known scalar result `-8.0`.
fn phase1_operand_a() -> Tensor {
    Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 3.0, 4.0, 5.0]).expect("benchmark tensor")
}

fn phase1_operand_b() -> Tensor {
    Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).expect("benchmark tensor")
}

fn phase1_operand_power() -> Tensor {
    Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64; 4]).expect("benchmark tensor")
}

fn phase1_operand_max() -> Tensor {
    Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).expect("benchmark tensor")
}

fn phase1_operand_clamp_bounds() -> (Tensor, Tensor) {
    (
        Tensor::from_vec_col_major(vec![], vec![-100.0_f64]).expect("benchmark tensor"),
        Tensor::from_vec_col_major(vec![], vec![-2.0_f64]).expect("benchmark tensor"),
    )
}

fn phase1_operand_rhs() -> Tensor {
    Tensor::from_vec_col_major(vec![4, 1], vec![1.0_f64; 4]).expect("benchmark tensor")
}

/// Pre-built phase-1 chain operands, shared by both execution arms.
struct Phase1Operands {
    a: Tensor,
    b: Tensor,
    power: Tensor,
    max: Tensor,
    rhs: Tensor,
    lower: Tensor,
    upper: Tensor,
}

impl Phase1Operands {
    fn new() -> Self {
        let (lower, upper) = phase1_operand_clamp_bounds();
        Self {
            a: phase1_operand_a(),
            b: phase1_operand_b(),
            power: phase1_operand_power(),
            max: phase1_operand_max(),
            rhs: phase1_operand_rhs(),
            lower,
            upper,
        }
    }
}

/// 10 ops, each entering its own backend session.
fn run_phase1_chain_one_shot(ops: &Phase1Operands, backend: &mut CpuBackend) -> Tensor {
    let x = ops.a.sub(&ops.b, backend).expect("sub");
    let x = x.log(backend).expect("log");
    let x = x.pow(&ops.power, backend).expect("pow");
    let x = x.maximum(&ops.max, backend).expect("maximum");
    let x = x.neg(backend).expect("neg");
    let x = x.reshape(&[4, 1], backend).expect("reshape");
    let x = x.transpose(&[1, 0], backend).expect("transpose");
    let x = x.clamp(&ops.lower, &ops.upper, backend).expect("clamp");
    let x = x.matmul(&ops.rhs, backend).expect("matmul");
    x.cast(tenferro_runtime::DType::F32, backend).expect("cast")
}

/// The same 10 ops inside one backend session (1 entry).
fn run_phase1_chain_one_session(ops: &Phase1Operands, backend: &mut CpuBackend) -> Tensor {
    backend.with_backend_session(|session| {
        let x = ops.a.sub_in(&ops.b, session).expect("sub");
        let x = x.log_in(session).expect("log");
        let x = x.pow_in(&ops.power, session).expect("pow");
        let x = x.maximum_in(&ops.max, session).expect("maximum");
        let x = x.neg_in(session).expect("neg");
        let x = x.reshape_in(&[4, 1], session).expect("reshape");
        let x = x.transpose_in(&[1, 0], session).expect("transpose");
        let x = x.clamp_in(&ops.lower, &ops.upper, session).expect("clamp");
        let x = x.matmul_in(&ops.rhs, session).expect("matmul");
        x.cast_in(tenferro_runtime::DType::F32, session)
            .expect("cast")
    })
}

fn bench_session_chain_phase1(c: &mut Criterion) {
    let mut group = c.benchmark_group("session_chain/phase1");
    let ops = Phase1Operands::new();
    let mut backend = CpuBackend::new();

    // Validation outside the timed region: the known scalar result -8.0
    // (see the chain comment above), and the two execution arms agree.
    let one_shot = run_phase1_chain_one_shot(&ops, &mut backend);
    assert_eq!(one_shot.shape(), &[1, 1], "chain must reduce to [1,1]");
    assert_eq!(one_shot.as_slice::<f32>().unwrap(), &[-8.0]);
    let one_session = run_phase1_chain_one_session(&ops, &mut backend);
    assert_eq!(one_session.as_slice::<f32>().unwrap(), &[-8.0]);

    group.bench_function("one_shot", |bench| {
        bench.iter(|| {
            let out = run_phase1_chain_one_shot(black_box(&ops), &mut backend);
            black_box(out);
        });
    });
    group.bench_function("one_session", |bench| {
        bench.iter(|| {
            let out = run_phase1_chain_one_session(black_box(&ops), &mut backend);
            black_box(out);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_session_chain, bench_session_chain_phase1);
criterion_main!(benches);
