mod organization;

use chainrules_core::AutodiffError;
use num_complex::Complex64;
use tenferro_device::Error as DeviceError;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};
use tidu::Tape;

use super::*;
use crate::structured::StructuredTensor;
use crate::AdTensor;

fn f64_vec(values: &[f64]) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn f64_scalar(value: f64) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

fn dense_structured(tensor: DenseTensor<f64>) -> StructuredTensor<f64> {
    StructuredTensor(tenferro_tensor::StructuredTensor::from_dense(tensor))
}

#[test]
fn tensor_pullback_routes_through_registered_rule_chain() {
    let tape = Tape::<crate::DynTensor>::new();
    let x = AdTensor::new_reverse_leaf(f64_vec(&[1.0, 2.0]), &tape).unwrap();
    let y = AdTensor::new_reverse_output(f64_scalar(3.0), &tape, None).unwrap();
    let x_node = x.node_id().unwrap();
    let y_node = y.node_id().unwrap();

    register_closure_rule::<f64>(
        &tape,
        y_node,
        vec![x_node],
        Box::new(move |cotangent| {
            let seed = cotangent.payload().buffer().as_slice().unwrap()[0];
            Ok(vec![(
                x_node,
                dense_structured(
                    DenseTensor::<f64>::from_slice(
                        &[seed, seed * 2.0],
                        &[2],
                        MemoryOrder::ColumnMajor,
                    )
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
    let tape = Tape::<crate::DynTensor>::new();
    let x = tape
        .leaf_with_tangent(
            crate::DynTensor::from(dense_structured(f64_vec(&[1.0, 2.0]))),
            crate::DynTensor::from(dense_structured(f64_vec(&[0.5, -0.5]))),
        )
        .unwrap();
    let y = AdTensor::new_reverse_output(f64_scalar(3.0), &tape, None).unwrap();
    let x_node = x.node_id().unwrap();
    let y_node = y.node_id().unwrap();

    register_closure_rule::<f64>(
        &tape,
        y_node,
        vec![x_node],
        Box::new(move |cotangent| {
            let seed = cotangent.payload().buffer().as_slice().unwrap()[0];
            Ok(vec![(
                x_node,
                dense_structured(
                    DenseTensor::<f64>::from_slice(&[seed, seed], &[2], MemoryOrder::ColumnMajor)
                        .unwrap(),
                ),
            )])
        }),
    );

    let mut leaf_tangents = std::collections::HashMap::new();
    leaf_tangents.insert(
        x.node_id().unwrap(),
        crate::DynTensor::from(dense_structured(f64_vec(&[0.5, -0.5]))),
    );
    match tape.hvp(
        &y.as_tracked()
            .expect("reverse output should expose tracked value"),
        &leaf_tangents,
    ) {
        Err(AutodiffError::HvpNotSupported) => {}
        Err(err) => panic!("unexpected hvp error: {err}"),
        Ok(_) => panic!("expected hvp to reject a VJP-only rule"),
    }
}

#[test]
fn tensor_pullback_rule_maps_rule_errors_to_invalid_argument() {
    let tape = Tape::<crate::DynTensor>::new();
    let y = AdTensor::new_reverse_output(f64_scalar(3.0), &tape, None).unwrap();
    let y_node = y.node_id().unwrap();

    register_closure_rule::<f64>(
        &tape,
        y_node,
        vec![],
        Box::new(|_| {
            Err(crate::Error::InvalidAdTensor {
                message: "synthetic pullback failure".to_string(),
            })
        }),
    );

    match tape.pullback_with_seed(
        &y.as_tracked()
            .expect("reverse output should expose tracked value"),
        crate::DynTensor::from(dense_structured(f64_scalar(1.0))),
    ) {
        Err(AutodiffError::InvalidArgument(message)) => {
            assert!(message.contains("synthetic pullback failure"));
        }
        Err(err) => panic!("unexpected pullback error: {err}"),
        Ok(_) => panic!("expected registered rule failure to surface as InvalidArgument"),
    }
}

#[test]
fn tensor_pullback_rejects_primal_output_directly() {
    let primal = AdTensor::new_primal(f64_scalar(3.0));
    match pullback(&primal, &f64_scalar(1.0)) {
        Err(crate::Error::InvalidAdTensor { message }) => {
            assert!(message.contains("reverse-mode output tensor"));
        }
        Err(err) => panic!("unexpected primal pullback error: {err}"),
        Ok(_) => panic!("primal outputs should not support reverse pullback"),
    }
}

#[test]
fn tensor_pullback_rejects_mismatched_registered_rule_dtype() {
    let tape = Tape::<crate::DynTensor>::new();
    let y = AdTensor::new_reverse_output(f64_scalar(3.0), &tape, None).unwrap();
    let y_node = y.node_id().unwrap();

    register_closure_rule::<Complex64>(&tape, y_node, vec![], Box::new(|_| Ok(Vec::new())));

    match pullback(&y, &f64_scalar(1.0)) {
        Err(crate::Error::Autodiff(AutodiffError::InvalidArgument(message))) => {
            assert!(message.contains("cotangent dtype did not match"));
        }
        Err(err) => panic!("unexpected pullback error: {err}"),
        Ok(_) => panic!("mismatched rule dtype should be rejected"),
    }
}

#[test]
fn tensor_pullback_rejects_cotangent_shape_mismatch() {
    let tape = Tape::<crate::DynTensor>::new();
    let y = AdTensor::new_reverse_output(f64_scalar(3.0), &tape, None).unwrap();
    let cotangent =
        DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();

    match pullback(&y, &cotangent) {
        Err(crate::Error::Backend(DeviceError::InvalidArgument(message))) => {
            assert!(message.contains("payload rank"));
        }
        Err(err) => panic!("unexpected cotangent mismatch error: {err}"),
        Ok(_) => panic!("cotangent shape mismatch should be rejected"),
    }
}
