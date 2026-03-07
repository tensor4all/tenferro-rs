use super::*;
use tenferro_tensor::{MemoryOrder, Tensor};

fn f64_vec(values: &[f64]) -> Tensor<f64> {
    Tensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn pullback_wrt_scalars_seeds_output_when_only_scalar_bridge_is_registered() {
    let tape = TapeId(701);
    let output_node = NodeId(11);
    let scalar_node = NodeId(12);

    register_scalar_bridge_rule::<f64, f64>(
        tape,
        output_node,
        Box::new(move |cotangent| {
            let sum = cotangent.buffer().as_slice().unwrap().iter().copied().sum();
            Ok(vec![(scalar_node, sum)])
        }),
    )
    .unwrap();

    let grads = pullback_wrt_scalars::<f64, f64>(
        tape,
        output_node,
        &f64_vec(&[0.5, 1.25]),
        &[Some(scalar_node)],
    )
    .unwrap();
    assert_eq!(grads, vec![Some(1.75)]);
}

#[test]
fn pullback_wrt_scalars_propagates_registered_scalar_rule_chain() {
    let tape = TapeId(702);
    let output_node = NodeId(21);
    let intermediate = NodeId(22);
    let leaf_a = NodeId(23);
    let leaf_b = NodeId(24);

    register_scalar_bridge_rule::<f64, f64>(
        tape,
        output_node,
        Box::new(move |_| Ok(vec![(intermediate, 2.0_f64)])),
    )
    .unwrap();
    register_scalar_rule::<f64>(
        tape,
        intermediate,
        Box::new(move |cotangent| Ok(vec![(leaf_a, *cotangent), (leaf_b, *cotangent * 3.0)])),
    )
    .unwrap();

    let grads = pullback_wrt_scalars::<f64, f64>(
        tape,
        output_node,
        &f64_vec(&[1.0]),
        &[Some(leaf_a), Some(leaf_b), Some(NodeId(999))],
    )
    .unwrap();
    assert_eq!(grads, vec![Some(2.0), Some(6.0), None]);
}

#[test]
fn pullback_scalar_reports_missing_registry() {
    let err = pullback_scalar::<f64>(TapeId(703), NodeId(31), &1.0_f64).unwrap_err();
    assert!(
        matches!(err, Error::InvalidAdScalar { message } if message.contains("no reverse scalar rules registered"))
    );
}
