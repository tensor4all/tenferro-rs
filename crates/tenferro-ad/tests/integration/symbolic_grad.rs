//! AD tests over placeholder-input graphs.
//!
//! Demonstrates that `grad` built over an `input_concrete_shape` placeholder
//! as well as `input_symbolic_shape` placeholders produces graphs that can be
//! evaluated via `run_with_inputs`.
//!
//! Previously the `input_symbolic_shape` variants were ignored because the
//! VJP code materialised zero cotangents with a concrete `vec![0; rank]`
//! shape when `wrt.shape_hint` was `None`. That has been replaced with a
//! deferred zero-tangent approach: the zero tensor is synthesised at
//! `run_with_inputs` time once the caller supplies the concrete binding.
//! See the deferred zero-tangent policy in `docs/spec/extension-op.md` and the
//! symbolic-shape tests in this file.

use crate::support;
use support::RunTraced;
use tenferro_ad::TracedTensorAdExt;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphExecutor, Tensor, TracedTensor};
use tenferro_tensor::DType;

fn f64_data(tensor: &Tensor) -> &[f64] {
    tensor.as_slice::<f64>().expect("expected f64 tensor")
}

#[test]
fn grad_of_sum_of_squares_against_symbolic_input() {
    // f(x) = sum(x * x), df/dx = 2 * x
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let sq = (&x * &x).unwrap();
    let loss = sq.reduce_sum(&[0]).unwrap();

    let g = loss.grad(&x).expect("grad build");

    let bound = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let grad_out = g
        .run_with_inputs_auto(&mut engine, &[(&x, &bound)])
        .expect("grad eval");

    assert_eq!(grad_out.shape(), &[4]);
    assert_eq!(f64_data(&grad_out), &[2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn grad_evaluates_with_different_shapes_from_same_symbolic_graph() {
    // Same symbolic grad graph, eval with shape [3] then shape [5].
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let sq = (&x * &x).unwrap();
    let loss = sq.reduce_sum(&[0]).unwrap();
    let g_template = loss.grad(&x).expect("grad build");

    let mut engine = GraphExecutor::new(CpuBackend::new());

    let g1 = g_template.clone();
    let b1 = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let out1 = g1
        .run_with_inputs_auto(&mut engine, &[(&x, &b1)])
        .expect("eval1");
    assert_eq!(f64_data(&out1), &[2.0, 4.0, 6.0]);

    let g2 = g_template.clone();
    let b2 = Tensor::from_vec_col_major(vec![5], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let out2 = g2
        .run_with_inputs_auto(&mut engine, &[(&x, &b2)])
        .expect("eval2");
    assert_eq!(f64_data(&out2), &[2.0, 4.0, 6.0, 8.0, 10.0]);
}

#[test]
fn grad_of_dot_product_against_two_symbolic_inputs() {
    // f(a, b) = sum(a * b)
    // df/da = b, df/db = a
    let a = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let b = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let loss = (&a * &b).unwrap().reduce_sum(&[0]).unwrap();

    let grad_a = loss.grad(&a).expect("grad a");
    let grad_b = loss.grad(&b).expect("grad b");

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let tb = Tensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0]).unwrap();

    let out_a = grad_a
        .run_with_inputs_auto(&mut engine, &[(&a, &ta), (&b, &tb)])
        .expect("eval grad a");
    assert_eq!(f64_data(&out_a), &[10.0, 20.0, 30.0]);

    let out_b = grad_b
        .run_with_inputs_auto(&mut engine, &[(&a, &ta), (&b, &tb)])
        .expect("eval grad b");
    assert_eq!(f64_data(&out_b), &[1.0, 2.0, 3.0]);
}

#[test]
fn grad_with_concrete_shape_placeholder() {
    // input_concrete_shape placeholder works for AD too.
    let x = TracedTensor::input_concrete_shape(DType::F64, &[3]).unwrap();
    let sq = (&x * &x).unwrap();
    let loss = sq.reduce_sum(&[0]).unwrap();
    let g = loss.grad(&x).expect("grad build");

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let bound = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let out = g
        .run_with_inputs_auto(&mut engine, &[(&x, &bound)])
        .expect("eval grad");

    assert_eq!(f64_data(&out), &[2.0, 4.0, 6.0]);
}
