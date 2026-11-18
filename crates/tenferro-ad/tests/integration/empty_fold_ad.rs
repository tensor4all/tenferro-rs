//! Empty-fold AD acceptance tests for #2453.
//!
//! `sum` and `prod` over a zero-length reduced axis must differentiate to
//! correctly shaped empty derivatives on both the eager and the traced
//! surfaces, and the two surfaces must agree bitwise.

use std::sync::Arc;

use tenferro_ad::{EagerRuntime, EagerTensor, TracedTensorAdExt};
use tenferro_runtime::{GraphCompiler, Tensor, TraceContext, TracedTensor};

use crate::support::{cpu_runtime, run_compiled_one};

fn empty_tensor(shape: &[usize]) -> Tensor {
    Tensor::from_vec_col_major(shape.to_vec(), Vec::<f64>::new()).unwrap()
}

fn eager_eval(tensor: &EagerTensor) -> Tensor {
    tensor
        .runtime()
        .with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |session| {
            session.duplicate_value(tensor)
        })
        .unwrap()
}

fn traced_eval(tensor: &TracedTensor) -> Tensor {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(tensor).unwrap();
    run_compiled_one(&cpu_runtime(), &program, &[]).unwrap()
}

fn eager_reduce(
    ctx: &Arc<EagerRuntime>,
    x: &EagerTensor,
    axes: &[usize],
    product: bool,
) -> EagerTensor {
    let axes: Vec<isize> = axes.iter().map(|&axis| axis as isize).collect();
    ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |session| {
        if product {
            session.reduce_prod(x, Some(&axes))
        } else {
            session.reduce_sum(x, Some(&axes))
        }
    })
    .unwrap()
}

fn traced_reduce(x: &TracedTensor, axes: &[usize], product: bool) -> TracedTensor {
    let axes: Vec<isize> = axes.iter().map(|&axis| axis as isize).collect();
    if product {
        x.reduce_prod(Some(&axes)).unwrap()
    } else {
        x.reduce_sum(Some(&axes)).unwrap()
    }
}

fn assert_bitwise_equal(actual: &Tensor, expected: &Tensor) {
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(
        actual.as_slice::<f64>().unwrap(),
        expected.as_slice::<f64>().unwrap()
    );
}

/// Empty fold with a non-empty output, i.e. the reduced axis is empty but the
/// kept axes are not. The output cotangent is non-empty and must be scattered
/// into an empty axis, contributing nothing.
fn check_kept_axis_nonempty(product: bool, expected_primal: &[f64]) {
    let shape = [2_usize, 0];
    let axes = [1_usize];
    let ctx = EagerRuntime::new().unwrap();
    let trace = TraceContext::new();

    let x = ctx.variable_from(empty_tensor(&shape)).unwrap();
    let y = eager_reduce(&ctx, &x, &axes, product);
    let tangent = ctx.constant_from(empty_tensor(&shape)).unwrap();
    let cotangent = ctx
        .constant_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap())
        .unwrap();
    let (dy, dx) = ctx
        .with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |session| {
            Ok::<_, tenferro_ad::Error>((
                session.jvp(&y, &x, &tangent)?,
                session.vjp(&y, &x, &cotangent)?,
            ))
        })
        .unwrap();

    let tx = trace
        .from_vec_col_major(shape.to_vec(), Vec::<f64>::new())
        .unwrap();
    let ty = traced_reduce(&tx, &axes, product);
    let t_tangent = trace
        .from_vec_col_major(shape.to_vec(), Vec::<f64>::new())
        .unwrap();
    let t_cotangent = trace
        .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .unwrap();
    let t_dy = ty.jvp(&tx, &t_tangent).unwrap();
    let t_dx = ty.vjp(&tx, &t_cotangent).unwrap();

    assert_eq!(y.shape(), &[2]);
    let eager_primal = eager_eval(&y);
    assert_eq!(eager_primal.as_slice::<f64>().unwrap(), expected_primal);
    assert_bitwise_equal(&traced_eval(&ty), &eager_primal);

    // The JVP of an empty fold is a zero of the output shape on both surfaces.
    let eager_dy = eager_eval(&dy);
    assert_eq!(eager_dy.shape(), &[2]);
    assert_eq!(eager_dy.as_slice::<f64>().unwrap(), &[0.0, 0.0]);
    assert_bitwise_equal(&traced_eval(&t_dy), &eager_dy);

    // The VJP has the input shape and therefore carries no elements; eager and
    // traced must agree on the shape and the (empty) payload bitwise.
    let eager_dx = eager_eval(&dx);
    assert_eq!(eager_dx.shape(), &[2, 0]);
    assert!(eager_dx.as_slice::<f64>().unwrap().is_empty());
    assert_bitwise_equal(&traced_eval(&t_dx), &eager_dx);
}

/// Empty fold that also removes every axis: a length-zero rank-1 input reduces
/// to a scalar, so the output seed is broadcast into an empty axis.
fn check_scalar_output(product: bool, expected_primal: f64) {
    let shape = [0_usize];
    let axes = [0_usize];
    let ctx = EagerRuntime::new().unwrap();
    let trace = TraceContext::new();

    let x = ctx.variable_from(empty_tensor(&shape)).unwrap();
    let y = eager_reduce(&ctx, &x, &axes, product);
    let tangent = ctx.constant_from(empty_tensor(&shape)).unwrap();
    let cotangent = ctx
        .constant_from(Tensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap())
        .unwrap();
    let (dy, dx) = ctx
        .with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |session| {
            Ok::<_, tenferro_ad::Error>((
                session.jvp(&y, &x, &tangent)?,
                session.vjp(&y, &x, &cotangent)?,
            ))
        })
        .unwrap();

    let tx = trace
        .from_vec_col_major(shape.to_vec(), Vec::<f64>::new())
        .unwrap();
    let ty = traced_reduce(&tx, &axes, product);
    let t_tangent = trace
        .from_vec_col_major(shape.to_vec(), Vec::<f64>::new())
        .unwrap();
    let t_cotangent = trace.from_vec_col_major(vec![], vec![1.0_f64]).unwrap();
    let t_dy = ty.jvp(&tx, &t_tangent).unwrap();
    let t_dx = ty.vjp(&tx, &t_cotangent).unwrap();

    assert_eq!(y.shape(), &[] as &[usize]);
    let eager_primal = eager_eval(&y);
    assert_eq!(eager_primal.as_slice::<f64>().unwrap(), &[expected_primal]);
    assert_bitwise_equal(&traced_eval(&ty), &eager_primal);

    let eager_dy = eager_eval(&dy);
    assert_eq!(eager_dy.shape(), &[] as &[usize]);
    assert_eq!(eager_dy.as_slice::<f64>().unwrap(), &[0.0]);
    assert_bitwise_equal(&traced_eval(&t_dy), &eager_dy);

    let eager_dx = eager_eval(&dx);
    assert_eq!(eager_dx.shape(), &[0]);
    assert!(eager_dx.as_slice::<f64>().unwrap().is_empty());
    assert_bitwise_equal(&traced_eval(&t_dx), &eager_dx);
}

#[test]
fn eager_and_traced_sum_over_empty_axis_agree() {
    check_kept_axis_nonempty(false, &[0.0, 0.0]);
    check_scalar_output(false, 0.0);
}

#[test]
fn eager_and_traced_prod_over_empty_axis_agree() {
    check_kept_axis_nonempty(true, &[1.0, 1.0]);
    check_scalar_output(true, 1.0);
}
