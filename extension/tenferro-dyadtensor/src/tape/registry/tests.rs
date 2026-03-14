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
