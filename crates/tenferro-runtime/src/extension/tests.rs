use computegraph::types::OperationRole;
use tenferro_tensor::DType;

use super::*;

#[test]
fn apply_expanded_graph_builds_standard_op_without_extension() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let y = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]);

    let outputs = apply_expanded_graph(
        &[&x, &y],
        vec![(DType::F64, vec![SymDim::from(2)])],
        |builder, inputs| {
            Ok(builder.add_operation(StdTensorOp::Add, inputs.to_vec(), OperationRole::Primary))
        },
    )
    .expect("expanded graph should build");

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].rank, 1);
    assert_eq!(outputs[0].dtype, DType::F64);
    assert!(outputs[0]
        .graph
        .operations()
        .iter()
        .all(|node| !matches!(node.operation, StdTensorOp::Extension(_))));
}

#[test]
fn apply_expanded_graph_rejects_output_metadata_count_mismatch() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);

    let result = apply_expanded_graph(&[&x], vec![], |builder, inputs| {
        Ok(builder.add_operation(StdTensorOp::Neg, inputs.to_vec(), OperationRole::Primary))
    });
    let Err(err) = result else {
        panic!("expanded graph output metadata mismatch should error");
    };

    let message = err.to_string();
    assert!(message.contains("expanded graph"), "{message}");
    assert!(message.contains("returned 1 outputs"), "{message}");
    assert!(message.contains("0 output metadata entries"), "{message}");
}
