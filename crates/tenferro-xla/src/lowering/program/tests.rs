use tenferro_tensor::{DType, DotGeneralConfig};

use super::*;

fn f64_type(shape: &[usize]) -> TensorType {
    TensorType::new(shape.to_vec(), DType::F64, "test tensor").unwrap()
}

fn f64_value(name: &str, shape: &[usize]) -> Value {
    Value {
        name: name.to_string(),
        ty: f64_type(shape),
    }
}

#[test]
fn helper_errors_preserve_diagnostic_context() {
    let err = require_input_count("stablehlo.test", &[], 1).unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidProgram { ref message }
            if message == "stablehlo.test expected 1 inputs, got 0"
    ));

    let empty_slots: Vec<Option<Value>> = Vec::new();
    let err = slot_value(&empty_slots, 0).unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidProgram { ref message }
            if message == "slot 0 is outside slot table length 0"
    ));

    let missing_slots: Vec<Option<Value>> = vec![None];
    let err = slot_value(&missing_slots, 0).unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidProgram { ref message } if message == "slot 0 has no value"
    ));
}

#[test]
fn lowering_helpers_reject_invalid_internal_shapes() {
    let mut emitter = Emitter::default();
    let non_scalar = f64_type(&[1]);
    let err = lower_constant(
        DType::F64,
        &1.0_f64.to_le_bytes(),
        &non_scalar,
        &mut emitter,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidProgram { ref message }
            if message == "ExecOp::Constant must lower as a scalar tensor"
    ));

    let input = f64_value("%arg0", &[2, 3]);
    let output_ty = f64_type(&[2]);
    let err = emit_transpose(&input, &[1, 0], &output_ty, &mut emitter).unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidProgram { ref message }
            if message.contains("transpose permutation length 2")
                && message.contains("output rank 1")
    ));
}

#[test]
fn constant_lowering_rejects_malformed_literal_bytes() {
    let mut emitter = Emitter::default();
    let scalar_f32 = TensorType::scalar(DType::F32, "test scalar").unwrap();
    let err = lower_constant(DType::F32, &[0, 1, 2], &scalar_f32, &mut emitter).unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidProgram { ref message } if message == "F32 constant expected 4 bytes, got 3"
    ));

    let scalar_f64 = TensorType::scalar(DType::F64, "test scalar").unwrap();
    let err = lower_constant(
        DType::F64,
        &[0, 1, 2, 3, 4, 5, 6],
        &scalar_f64,
        &mut emitter,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidProgram { ref message } if message == "F64 constant expected 8 bytes, got 7"
    ));
}

#[test]
fn dot_shape_rejects_invalid_dimension_config() {
    let err = stablehlo_dot_shape(
        &[2, 3],
        &[3, 4],
        &DotGeneralConfig {
            lhs_contracting_dims: vec![2],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
    )
    .unwrap_err();

    assert!(matches!(
        err,
        Error::InvalidProgram { ref message } if message.contains("lhs_contracting_dim")
    ));
}
