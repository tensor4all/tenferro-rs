use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_runtime::{DType, GraphCompiler, TracedTensor};

use super::{einsum, einsum_with};
use crate::EinsumOptimize;

#[test]
fn concrete_traced_nary_einsum_expands_to_standard_graph() {
    let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    let b = TracedTensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap();
    let c = TracedTensor::from_vec_col_major(vec![4, 5], vec![1.0_f64; 20]).unwrap();
    let mut compiler = GraphCompiler::new();

    let out = einsum(&mut compiler, &[&a, &b, &c], "ij,jk,kl->il").unwrap();

    assert!(out
        .graph()
        .operations()
        .iter()
        .all(|node| { !matches!(node.operation, StdTensorOp::Extension(_)) }));
    assert!(out
        .graph()
        .operations()
        .iter()
        .any(|node| { matches!(node.operation, StdTensorOp::DotGeneral { .. }) }));
}

#[test]
fn symbolic_path_traced_nary_einsum_expands_to_standard_graph() {
    let a = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let b = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let c = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let mut compiler = GraphCompiler::new();

    let out = einsum_with(
        &mut compiler,
        &[&a, &b, &c],
        "ij,jk,kl->il",
        EinsumOptimize::Path(vec![(0, 1), (0, 1)]),
    )
    .unwrap();

    assert!(out
        .graph()
        .operations()
        .iter()
        .all(|node| { !matches!(node.operation, StdTensorOp::Extension(_)) }));
    assert!(out
        .graph()
        .operations()
        .iter()
        .any(|node| { matches!(node.operation, StdTensorOp::DotGeneral { .. }) }));
}

#[test]
fn symbolic_auto_traced_nary_einsum_remains_extension() {
    let a = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let b = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let c = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let mut compiler = GraphCompiler::new();

    let out = einsum(&mut compiler, &[&a, &b, &c], "ij,jk,kl->il").unwrap();

    assert!(out
        .graph()
        .operations()
        .iter()
        .any(|node| { matches!(node.operation, StdTensorOp::Extension(_)) }));
}
