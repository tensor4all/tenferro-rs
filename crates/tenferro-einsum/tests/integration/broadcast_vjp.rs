#![cfg(feature = "autodiff")]
//! Numerical VJP of a singleton-broadcast einsum (#2557).
//!
//! An operand axis whose primal extent is 1 but whose label has a larger
//! extent elsewhere is broadcast by the forward einsum, so its cotangent must
//! be summed over that label and keep the primal extent 1. Each case feeds an
//! all-ones cotangent and compares every element of both operand gradients.

use tenferro_ad::{AdContext, EagerRuntime};
use tenferro_cpu::CpuEngine;
use tenferro_einsum::{EagerEinsumOps, TracedEinsumOps};
use tenferro_runtime::{GraphCompiler, TraceContext};
use tenferro_tensor::Tensor;

use super::support;

/// One broadcast einsum with all-ones operands and its exact gradients.
struct BroadcastCase {
    subscripts: &'static str,
    lhs_shape: &'static [usize],
    rhs_shape: &'static [usize],
    output_shape: &'static [usize],
    /// Every element of `d sum(out) / d lhs`.
    lhs_grad: f64,
    /// Every element of `d sum(out) / d rhs`.
    rhs_grad: f64,
}

/// `out[p, q, i] = lhs[i, 0] * sum_j rhs[p, q, j]`, `j` broadcast from 1 to 3:
/// `d/d lhs[i, 0] = sum_{p,q,j} rhs = 4 * 2 * 3 = 24` and
/// `d/d rhs[p, q, j] = sum_i lhs[i, 0] = 2`.
const EXPLICIT: BroadcastCase = BroadcastCase {
    subscripts: "ij,pqj->pqi",
    lhs_shape: &[2, 1],
    rhs_shape: &[4, 2, 3],
    output_shape: &[4, 2, 2],
    lhs_grad: 24.0,
    rhs_grad: 2.0,
};

/// The batch axis of lhs broadcasts from 1 to 3:
/// `out[b, i, k] = sum_j lhs[0, i, j] * rhs[b, j, k]`, so
/// `d/d lhs[0, i, j] = sum_{b,k} rhs[b, j, k] = 3 * 2 = 6` and
/// `d/d rhs[b, j, k] = sum_i lhs[0, i, j] = 2`.
const ELLIPSIS: BroadcastCase = BroadcastCase {
    subscripts: "...ij,...jk->...ik",
    lhs_shape: &[1, 2, 4],
    rhs_shape: &[3, 4, 2],
    output_shape: &[3, 2, 2],
    lhs_grad: 6.0,
    rhs_grad: 2.0,
};

fn ones(shape: &[usize]) -> Tensor {
    Tensor::from_vec_col_major(shape.to_vec(), vec![1.0_f64; shape.iter().product()]).unwrap()
}

fn einsum_ad() -> AdContext {
    AdContext::builder()
        .with_semantic_extension_rules(tenferro_einsum::semantic_ad_rules().unwrap())
        .unwrap()
        .build()
        .unwrap()
}

/// Shape and every element in one comparison, so a failure shows both.
fn assert_filled(label: &str, actual: &Tensor, shape: &[usize], value: f64) {
    let expected = vec![value; shape.iter().product()];
    assert_eq!(
        (actual.shape(), actual.as_slice::<f64>().unwrap()),
        (shape, expected.as_slice()),
        "{label}"
    );
}

/// Traced surface: record the einsum, take `AdContext::vjp` with an all-ones
/// cotangent for each operand, compile and run.
fn assert_traced_vjp(case: &BroadcastCase) {
    let ad = einsum_ad();
    let trace = TraceContext::new();
    let lhs = trace
        .from_tensor_concrete_shape(ones(case.lhs_shape))
        .unwrap();
    let rhs = trace
        .from_tensor_concrete_shape(ones(case.rhs_shape))
        .unwrap();
    let output = trace.einsum(&[&lhs, &rhs], case.subscripts).unwrap();
    let cotangent = trace
        .from_tensor_concrete_shape(ones(case.output_shape))
        .unwrap();

    let forward = support::run_one(&GraphCompiler::new().compile(&output).unwrap(), &[]).unwrap();
    assert_eq!(forward.shape(), case.output_shape, "traced forward shape");

    let lhs_grad = ad.vjp(&output, &lhs, &cotangent).unwrap();
    let lhs_grad =
        support::run_one(&GraphCompiler::new().compile(&lhs_grad).unwrap(), &[]).unwrap();
    assert_filled(
        "traced lhs gradient",
        &lhs_grad,
        case.lhs_shape,
        case.lhs_grad,
    );

    let rhs_grad = ad.vjp(&output, &rhs, &cotangent).unwrap();
    let rhs_grad =
        support::run_one(&GraphCompiler::new().compile(&rhs_grad).unwrap(), &[]).unwrap();
    assert_filled(
        "traced rhs gradient",
        &rhs_grad,
        case.rhs_shape,
        case.rhs_grad,
    );
}

/// Eager surface: `backward` of `sum(out)` seeds the einsum with an all-ones
/// cotangent.
fn assert_eager_vjp(case: &BroadcastCase) {
    let ad = einsum_ad();
    let ctx = EagerRuntime::with_ad_context(&ad).unwrap();
    let lhs = ctx.variable_from(ones(case.lhs_shape)).unwrap();
    let rhs = ctx.variable_from(ones(case.rhs_shape)).unwrap();
    let mut engine = CpuEngine::serial();
    let output = ctx
        .with_eager_session(&mut engine, |s| s.einsum(&[&lhs, &rhs], case.subscripts))
        .unwrap();
    assert_eq!(output.shape(), case.output_shape, "eager forward shape");
    let axes: Vec<usize> = (0..case.output_shape.len()).collect();
    let loss = ctx
        .with_eager_session(&mut engine, |s| {
            s.reduce_sum(
                &output,
                Some(&axes.iter().map(|&a| a as isize).collect::<Vec<_>>()),
            )
        })
        .unwrap();
    let _ = loss
        .runtime()
        .with_eager_session(&mut engine, |s| s.backward(&loss))
        .unwrap();

    let lhs_grad = lhs.grad().unwrap().unwrap().to_tensor(&mut engine).unwrap();
    assert_filled(
        "eager lhs gradient",
        &lhs_grad,
        case.lhs_shape,
        case.lhs_grad,
    );
    let rhs_grad = rhs.grad().unwrap().unwrap().to_tensor(&mut engine).unwrap();
    assert_filled(
        "eager rhs gradient",
        &rhs_grad,
        case.rhs_shape,
        case.rhs_grad,
    );
}

#[test]
fn traced_einsum_vjp_sums_a_singleton_broadcast_explicit_label() {
    assert_traced_vjp(&EXPLICIT);
}

#[test]
fn traced_einsum_vjp_sums_a_singleton_broadcast_ellipsis_axis() {
    assert_traced_vjp(&ELLIPSIS);
}

#[test]
fn eager_einsum_vjp_sums_a_singleton_broadcast_explicit_label() {
    assert_eager_vjp(&EXPLICIT);
}

#[test]
fn eager_einsum_vjp_sums_a_singleton_broadcast_ellipsis_axis() {
    assert_eager_vjp(&ELLIPSIS);
}
