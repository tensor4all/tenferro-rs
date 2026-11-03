use std::sync::Arc;

use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_runtime::ad_support::RetainedValue;
use tenferro_runtime::DType;
use tenferro_tensor::Tensor;

use super::{reachable_leaf_nodes, replay, SemanticValue};

fn assert_send_sync<T: Send + Sync>() {}

fn leaf(values: Vec<f64>) -> SemanticValue {
    let data = Arc::new(RetainedValue::from_tensor(
        Tensor::from_vec_col_major(vec![values.len()], values).unwrap(),
    ));
    SemanticValue::leaf(
        TensorInputKey::User { id: 7 },
        DType::F64,
        data.shape().len(),
        data,
    )
}

#[test]
fn semantic_values_and_nodes_are_send_and_sync() {
    // `EagerTensor` embeds a `SemanticValue`, so the record must not introduce a
    // thread-bound handle; the downstream assertion is on `EagerTensor` itself.
    assert_send_sync::<SemanticValue>();
}

#[test]
fn one_node_is_recorded_per_applied_op() {
    let input = leaf(vec![1.0, 2.0]);
    let mul = SemanticValue::op(
        StdTensorOp::Mul,
        vec![input.clone(), input.clone()].into_boxed_slice(),
        1,
    );
    let sum = SemanticValue::op(
        StdTensorOp::ReduceSum { axes: vec![0] },
        vec![mul.clone()].into_boxed_slice(),
        1,
    );

    // Two ops, two nodes: the record does not re-wrap or duplicate intermediates.
    assert_eq!(reachable_leaf_nodes(&sum).len(), 1);
    let replayed = replay(&[sum]).unwrap();
    assert_eq!(replayed.leaves().len(), 1);
    assert_eq!(replayed.outputs(), &[0]);
    assert_eq!(replayed.frozen().program.operations().count(), 2);
    assert_eq!(replayed.leaf_data().len(), 1);
}

#[test]
fn multi_output_values_share_a_node_and_keep_distinct_identities() {
    // The recorded output count is a fact of the forward pass: a multi-output
    // extension records three values over one node, and residual identity and
    // edges must distinguish them (SVD has three outputs).
    let input = leaf(vec![1.0, 2.0, 3.0, 4.0]);
    let svd = SemanticValue::op(StdTensorOp::Mul, vec![input].into_boxed_slice(), 3);
    let outputs: Vec<SemanticValue> = (0..3).map(|index| svd.at_output(index)).collect();

    assert_eq!(outputs[0].node_id(), outputs[1].node_id());
    assert_ne!(outputs[0].identity(), outputs[1].identity());
    assert_eq!(outputs[0].identity(), (outputs[0].node_id(), 0));
    assert_eq!(outputs[2].identity(), (outputs[2].node_id(), 2));
}

#[test]
fn replay_rejects_a_recorded_output_count_the_op_cannot_produce() {
    let input = leaf(vec![1.0, 2.0]);

    // `Mul` produces one value, so a record claiming three is malformed; the
    // replay reports it instead of silently mis-indexing the outputs.
    let broken = SemanticValue::op(
        StdTensorOp::Mul,
        vec![input.clone(), input].into_boxed_slice(),
        3,
    );
    let error = match replay(&[broken]) {
        Ok(_) => panic!("a malformed record must not replay"),
        Err(error) => error,
    };
    assert!(
        matches!(error, crate::Error::Internal(_)),
        "expected an internal replay error, got {error:?}"
    );
}

#[test]
fn replayed_program_binds_each_leaf_once() {
    let input = leaf(vec![3.0]);
    let squared = SemanticValue::op(
        StdTensorOp::Mul,
        vec![input.clone(), input].into_boxed_slice(),
        1,
    );
    let replayed = replay(&[squared]).unwrap();

    // One leaf value feeds both operands, so the program has one input.
    assert_eq!(replayed.leaves().len(), 1);
    assert_eq!(replayed.frozen().bindings.len(), 1);
    assert_eq!(replayed.leaf_data()[0].shape(), &[1]);
}
