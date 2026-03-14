mod organization;

use chainrules::Tape;
use chainrules_core::AutodiffError;
use tenferro_tensor::{MemoryOrder, Tensor};

use super::*;
use crate::{AdTensor, StructuredTensor};

fn f64_vec(values: &[f64]) -> Tensor<f64> {
    Tensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn f64_scalar(value: f64) -> Tensor<f64> {
    Tensor::<f64>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn tensor_pullback_routes_through_registered_rule_chain() {
    let tape = Tape::<StructuredTensor<f64>>::new();
    let x = AdTensor::new_reverse_leaf(f64_vec(&[1.0, 2.0]), &tape).unwrap();
    let y = AdTensor::new_reverse_output(f64_scalar(3.0), &tape, None).unwrap();
    let x_node = x.node_id().unwrap();
    let y_node = y.node_id().unwrap();

    register_rule::<f64>(
        &tape,
        y_node,
        Box::new(move |cotangent| {
            let seed = cotangent.payload().buffer().as_slice().unwrap()[0];
            Ok(vec![(
                x_node,
                StructuredTensor::from_dense(
                    Tensor::<f64>::from_slice(&[seed, seed * 2.0], &[2], MemoryOrder::ColumnMajor)
                        .unwrap(),
                ),
            )])
        }),
    );

    let grads = pullback(&y, &f64_scalar(1.5)).unwrap();
    assert_eq!(
        grads.get(&x_node).unwrap().buffer().as_slice().unwrap(),
        &[1.5, 3.0]
    );
    assert!(
        grads.get(&y_node).is_none(),
        "pullback returns leaf gradients only; output nodes must stay absent"
    );
}

#[test]
fn tensor_pullback_rule_rejects_hvp_when_only_vjp_is_registered() {
    let tape = Tape::<StructuredTensor<f64>>::new();
    let x = tape
        .leaf_with_tangent(
            StructuredTensor::from_dense(f64_vec(&[1.0, 2.0])),
            StructuredTensor::from_dense(f64_vec(&[0.5, -0.5])),
        )
        .unwrap();
    let y = AdTensor::new_reverse_output(f64_scalar(3.0), &tape, None).unwrap();
    let x_node = x.node_id().unwrap();
    let y_node = y.node_id().unwrap();

    register_rule::<f64>(
        &tape,
        y_node,
        Box::new(move |cotangent| {
            let seed = cotangent.payload().buffer().as_slice().unwrap()[0];
            Ok(vec![(
                x_node,
                StructuredTensor::from_dense(
                    Tensor::<f64>::from_slice(&[seed, seed], &[2], MemoryOrder::ColumnMajor)
                        .unwrap(),
                ),
            )])
        }),
    );

    match tape.hvp(
        y.as_tracked()
            .expect("reverse output should expose tracked value"),
    ) {
        Err(AutodiffError::HvpNotSupported) => {}
        Err(err) => panic!("unexpected hvp error: {err}"),
        Ok(_) => panic!("expected hvp to reject a VJP-only rule"),
    }
}

#[test]
fn tensor_pullback_rule_maps_rule_errors_to_invalid_argument() {
    let tape = Tape::<StructuredTensor<f64>>::new();
    let y = AdTensor::new_reverse_output(f64_scalar(3.0), &tape, None).unwrap();
    let y_node = y.node_id().unwrap();

    register_rule::<f64>(
        &tape,
        y_node,
        Box::new(|_| {
            Err(crate::Error::InvalidAdTensor {
                message: "synthetic pullback failure".to_string(),
            })
        }),
    );

    match tape.pullback_with_seed(
        y.as_tracked()
            .expect("reverse output should expose tracked value"),
        StructuredTensor::from_dense(f64_scalar(1.0)),
    ) {
        Err(AutodiffError::InvalidArgument(message)) => {
            assert!(message.contains("synthetic pullback failure"));
        }
        Err(err) => panic!("unexpected pullback error: {err}"),
        Ok(_) => panic!("expected registered rule failure to surface as InvalidArgument"),
    }
}
