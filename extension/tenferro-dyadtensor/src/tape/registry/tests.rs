use chainrules_core::AutodiffError;
use num_complex::Complex64;
use tenferro_tensor::{MemoryOrder, Tensor};

use super::*;
use crate::StructuredTensor;

fn f64_scalar(value: f64) -> DynTensor {
    DynTensor::from(StructuredTensor::from_dense(
        Tensor::<f64>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap(),
    ))
}

#[test]
fn tensor_rule_adapter_rejects_cotangent_dtype_mismatch_and_reports_empty_inputs() {
    let adapter = TensorRuleAdapter::<f64> {
        rule: Box::new(|_| Ok(Vec::new())),
    };
    let cotangent = DynTensor::from(StructuredTensor::from_dense(
        Tensor::<Complex64>::from_slice(
            &[Complex64::new(1.0, -2.0)],
            &[],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
    ));

    match adapter.pullback(&cotangent) {
        Err(AutodiffError::InvalidArgument(message)) => {
            assert!(message.contains("cotangent dtype did not match"));
        }
        Err(err) => panic!("unexpected adapter error: {err}"),
        Ok(_) => panic!("dtype mismatch should be rejected"),
    }
    assert!(adapter.inputs().is_empty());
}

#[test]
fn tensor_rule_adapter_pullback_with_tangents_reports_hvp_not_supported() {
    let adapter = TensorRuleAdapter::<f64> {
        rule: Box::new(|_| Ok(Vec::new())),
    };

    match adapter.pullback_with_tangents(&f64_scalar(1.0), &f64_scalar(0.5)) {
        Err(AutodiffError::HvpNotSupported) => {}
        Err(err) => panic!("unexpected HVP error: {err}"),
        Ok(_) => panic!("VJP-only rule should not claim HVP support"),
    }
}

#[test]
fn mixed_tensor_rule_adapter_rejects_cotangent_dtype_mismatch_and_reports_empty_inputs() {
    let adapter = MixedTensorRuleAdapter::<f64, Complex64> {
        rule: Box::new(|_| Ok(Vec::new())),
    };
    let cotangent = DynTensor::from(StructuredTensor::from_dense(
        Tensor::<Complex64>::from_slice(
            &[Complex64::new(1.0, -2.0)],
            &[],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
    ));

    match adapter.pullback(&cotangent) {
        Err(AutodiffError::InvalidArgument(message)) => {
            assert!(message.contains("cotangent dtype did not match"));
        }
        Err(err) => panic!("unexpected adapter error: {err}"),
        Ok(_) => panic!("dtype mismatch should be rejected"),
    }
    assert!(adapter.inputs().is_empty());
}

#[test]
fn mixed_tensor_rule_adapter_pullback_converts_gradient_dtype() {
    let adapter = MixedTensorRuleAdapter::<f64, Complex64> {
        rule: Box::new(|cotangent| {
            let seed = cotangent.payload().buffer().as_slice().unwrap()[0];
            Ok(vec![(
                NodeId::new(7),
                StructuredTensor::from_dense(
                    Tensor::<Complex64>::from_slice(
                        &[Complex64::new(seed, -seed)],
                        &[],
                        MemoryOrder::ColumnMajor,
                    )
                    .unwrap(),
                ),
            )])
        }),
    };

    let cotangent = f64_scalar(2.5);
    let grads = adapter.pullback(&cotangent).unwrap();
    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].0, NodeId::new(7));
    assert_eq!(grads[0].1.scalar_type(), crate::ScalarType::C64);
    assert_eq!(
        grads[0]
            .1
            .as_c64()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[Complex64::new(2.5, -2.5)]
    );
}
