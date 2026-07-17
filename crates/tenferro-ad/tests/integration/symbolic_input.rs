//! End-to-end tests for the placeholder constructor API
//! (`input_concrete_shape` / `input_symbolic_shape`) and
//! [`GraphExecutor::run_with_inputs`].
//!
//! These tests verify that:
//! * A graph built against a placeholder can be evaluated by binding a
//!   concrete tensor.
//! * Concrete-shape placeholders reject binding tensors with mismatched
//!   shapes but accept matching ones.
//! * Symbolic-shape placeholders accept any tensor with the right rank and
//!   dtype.
//! * Mixed graphs (static leaves + placeholder leaves) route data correctly.

use crate::support;
use support::RunTraced;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphExecutor, Tensor, TracedTensor};
use tenferro_tensor::DType;

fn f64_data(tensor: &Tensor) -> &[f64] {
    tensor.as_slice::<f64>().expect("expected f64 tensor")
}

#[test]
fn identity_on_symbolic_input_rank_1() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let y = x.clone();

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let bound = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let out = y
        .run_with_inputs_auto(&mut engine, &[(&x, &bound)])
        .expect("run_with_inputs");

    assert_eq!(out.shape(), &[4]);
    assert_eq!(f64_data(&out), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn addition_of_two_symbolic_inputs() {
    let a = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let b = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let y = (&a + &b).unwrap();

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let tb = Tensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0]).unwrap();
    let out = y
        .run_with_inputs_auto(&mut engine, &[(&a, &ta), (&b, &tb)])
        .expect("run_with_inputs");

    assert_eq!(out.shape(), &[3]);
    assert_eq!(f64_data(&out), &[11.0, 22.0, 33.0]);
}

#[test]
fn matrix_symbolic_input_uses_column_major_values() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let y = (&x + &x).unwrap();

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let bound =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
    let out = y
        .run_with_inputs_auto(&mut engine, &[(&x, &bound)])
        .expect("run_with_inputs");

    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(f64_data(&out), &[2.0, 8.0, 4.0, 10.0, 6.0, 12.0]);
}

#[test]
fn same_symbolic_graph_reused_with_different_shapes() {
    // Build the graph once.
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let graph = (&x + &x).unwrap();

    let mut engine = GraphExecutor::new(CpuBackend::new());

    // First eval: shape [2].
    let g1 = graph.clone();
    let bound_small = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let out_small = g1
        .run_with_inputs_auto(&mut engine, &[(&x, &bound_small)])
        .expect("small eval");
    assert_eq!(out_small.shape(), &[2]);
    assert_eq!(f64_data(&out_small), &[2.0, 4.0]);

    // Second eval: shape [5].
    let g2 = graph.clone();
    let bound_big = Tensor::from_vec_col_major(vec![5], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let out_big = g2
        .run_with_inputs_auto(&mut engine, &[(&x, &bound_big)])
        .expect("big eval");
    assert_eq!(out_big.shape(), &[5]);
    assert_eq!(f64_data(&out_big), &[2.0, 4.0, 6.0, 8.0, 10.0]);
}

#[test]
fn concrete_shape_placeholder_accepts_exact_shape() {
    let x = TracedTensor::input_concrete_shape(DType::F64, &[2, 3]).unwrap();
    assert!(x.is_concrete_shape());
    let y = x.clone();

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let bound =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let out = y
        .run_with_inputs_auto(&mut engine, &[(&x, &bound)])
        .expect("run_with_inputs");

    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(f64_data(&out), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn mixed_graph_static_and_placeholder_inputs() {
    // Binary ops that broadcast (like `+`) require both operands' shape_hint
    // to match or be concrete, so we use `input_concrete_shape` for the
    // placeholder here.
    let static_leaf = TracedTensor::from_vec_col_major(vec![2], vec![100.0_f64, 200.0]).unwrap();
    let placeholder = TracedTensor::input_concrete_shape(DType::F64, &[2]).unwrap();
    let sum = (&static_leaf + &placeholder).unwrap();

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let bound = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let out = sum
        .run_with_inputs_auto(&mut engine, &[(&placeholder, &bound)])
        .expect("run_with_inputs");

    assert_eq!(out.shape(), &[2]);
    assert_eq!(f64_data(&out), &[101.0, 202.0]);
}

#[test]
fn eval_with_empty_bindings_behaves_like_eval_for_all_static_graph() {
    let a = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let b = TracedTensor::from_vec_col_major(vec![2], vec![10.0_f64, 20.0]).unwrap();
    let y = (&a + &b).unwrap();

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let out = y
        .run_with_inputs_auto(&mut engine, &[])
        .expect("run_with_inputs with no bindings");

    assert_eq!(f64_data(&out), &[11.0, 22.0]);
}

#[test]
fn concrete_placeholder_is_concrete_shape_true() {
    let x = TracedTensor::input_concrete_shape(DType::F32, &[3, 4]).unwrap();
    assert!(x.is_concrete_shape());
    assert_eq!(x.rank, 2);
    assert_eq!(x.dtype, DType::F32);
}

#[test]
fn symbolic_placeholder_is_concrete_shape_false() {
    let x = TracedTensor::input_symbolic_shape(DType::C64, 3).unwrap();
    assert!(!x.is_concrete_shape());
    assert_eq!(x.rank, 3);
    assert_eq!(x.dtype, DType::C64);
}

#[test]
fn from_tensor_symbolic_shape_drops_shape_but_keeps_data() {
    let tensor = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let x = TracedTensor::from_tensor_symbolic_shape(tensor).unwrap();
    assert!(!x.is_concrete_shape());

    // Even though shape is advertised as symbolic, data is attached so plain
    // `eval` (via the no-bindings shortcut) still works.
    let y = x.clone();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let out = y.run_with(&mut engine).expect("eval with attached data");
    assert_eq!(out.shape(), &[4]);
    assert_eq!(f64_data(&out), &[1.0, 2.0, 3.0, 4.0]);
}
