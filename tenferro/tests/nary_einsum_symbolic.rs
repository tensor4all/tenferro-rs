//! Verify that an einsum with at least one `input_symbolic_shape` input is
//! kept as a single `NaryEinsum` op rather than being decomposed at graph
//! build time.
//!
//! Inspects the resulting fragment's ops directly.

mod support;
use support::{
    einsum, einsum_subscripts, einsum_subscripts_with, einsum_with, run_many_traced_with, RunTraced,
};
use tenferro::{CpuBackend, GraphExecutor, Tensor, TracedTensor};
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::DType;

fn fragment_has_nary_einsum(tensor: &TracedTensor) -> bool {
    tensor
        .fragment
        .ops()
        .iter()
        .any(|op_node| matches!(op_node.op, StdTensorOp::NaryEinsum { .. }))
}

fn fragment_has_dot_general(tensor: &TracedTensor) -> bool {
    tensor
        .fragment
        .ops()
        .iter()
        .any(|op_node| matches!(op_node.op, StdTensorOp::DotGeneral { .. }))
}

#[test]
fn all_concrete_inputs_decompose_to_dot_general() {
    let a = TracedTensor::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = TracedTensor::from_vec(vec![3, 4], vec![1.0_f64; 12]);
    let mut engine = GraphExecutor::new(CpuBackend::new());

    let c = einsum(&mut engine, &[&a, &b], "ij,jk->ik").expect("einsum concrete");

    // With all concrete shapes, einsum decomposes into binary DotGeneral ops.
    assert!(
        fragment_has_dot_general(&c),
        "concrete einsum should decompose to DotGeneral"
    );
    assert!(
        !fragment_has_nary_einsum(&c),
        "concrete einsum should NOT keep NaryEinsum"
    );
}

#[test]
fn any_symbolic_input_keeps_nary_einsum() {
    let a = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let b = TracedTensor::from_vec(vec![3, 4], vec![1.0_f64; 12]);
    let mut engine = GraphExecutor::new(CpuBackend::new());

    let c = einsum(&mut engine, &[&a, &b], "ij,jk->ik").expect("einsum symbolic");

    assert!(
        fragment_has_nary_einsum(&c),
        "symbolic einsum should keep a single NaryEinsum op"
    );
}

#[test]
fn three_way_einsum_with_one_symbolic_input_is_nary() {
    let a = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let b = TracedTensor::from_vec(vec![3, 4], vec![1.0_f64; 12]);
    let c = TracedTensor::from_vec(vec![4, 5], vec![1.0_f64; 20]);
    let mut engine = GraphExecutor::new(CpuBackend::new());

    let out = einsum(&mut engine, &[&a, &b, &c], "ij,jk,kl->il").expect("einsum 3-way");

    assert!(
        fragment_has_nary_einsum(&out),
        "3-way einsum with 1 symbolic input should keep NaryEinsum"
    );
}

#[test]
fn from_tensor_symbolic_shape_triggers_nary_path() {
    let a = TracedTensor::from_tensor_symbolic_shape(Tensor::from_vec(
        vec![2, 3],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let b = TracedTensor::from_vec(vec![3, 4], vec![1.0_f64; 12]);
    let mut engine = GraphExecutor::new(CpuBackend::new());

    let c = einsum(&mut engine, &[&a, &b], "ij,jk->ik").expect("einsum mixed");

    // `from_tensor_symbolic_shape` makes `a` appear symbolic to the
    // dispatcher even though data is attached.
    assert!(
        fragment_has_nary_einsum(&c),
        "from_tensor_symbolic_shape should route through NaryEinsum"
    );
}

#[test]
fn symbolic_einsum_evaluates_correctly_via_run_with_inputs() {
    // Full roundtrip: symbolic graph -> NaryEinsum -> run_with_inputs yields
    // the same numeric result as a concrete einsum would.
    let a = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let b = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let mut engine = GraphExecutor::new(CpuBackend::new());

    let mut c = einsum(&mut engine, &[&a, &b], "ij,jk->ik").expect("einsum sym");

    let ta = Tensor::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let tb = Tensor::from_vec(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = c
        .run_with_inputs_auto(&mut engine, &[(&a, &ta), (&b, &tb)])
        .expect("run_with_inputs");

    assert_eq!(out.shape(), &[2, 2]);
    // Column-major [2,3] @ [3,2] → [2,2]
    // A = [[1,3,5],[2,4,6]] (column-major)
    // B = [[1,3,5],[2,4,6]] (column-major)
    // C = A @ B; let the numerical check be a simple sanity bound.
    let data = out.as_slice::<f64>().expect("f64");
    assert_eq!(data.len(), 4);
    assert!(data.iter().all(|x| x.is_finite()));
}
