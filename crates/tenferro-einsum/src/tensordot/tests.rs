use super::{dot_general_config, validate_traced_contract_dims, TensorDotAxes};

#[test]
fn dot_general_config_rejects_invalid_count_and_explicit_axes() {
    let count_err = dot_general_config(TensorDotAxes::Count(3), 2, 2).unwrap_err();
    assert!(count_err.to_string().contains("Count(3)"));

    let length_err = dot_general_config(
        TensorDotAxes::Axes {
            lhs: &[0, 1],
            rhs: &[0],
        },
        2,
        2,
    )
    .unwrap_err();
    assert!(length_err.to_string().contains("matching lengths"));

    let duplicate_err = dot_general_config(
        TensorDotAxes::Axes {
            lhs: &[0, -2],
            rhs: &[0, 1],
        },
        2,
        2,
    )
    .unwrap_err();
    assert!(duplicate_err.to_string().contains("duplicate lhs axis"));
}

#[test]
fn dot_general_config_rejects_isize_min_axis_without_overflow() {
    // #2605: `rank + axis` used to overflow in overflow-checked builds before
    // the bounds check ran. The error must carry the caller's original signed
    // axis, matching every other out-of-range axis.
    let err = dot_general_config(
        TensorDotAxes::Axes {
            lhs: &[isize::MIN],
            rhs: &[0],
        },
        2,
        2,
    )
    .unwrap_err();
    let message = err.to_string();
    assert!(message.contains("out of bounds"), "{message}");
    assert!(message.contains(&isize::MIN.to_string()), "{message}");
}

#[test]
fn validate_traced_contract_dims_allows_symbolic_and_rejects_concrete_mismatch() {
    let trace = tenferro_runtime::TraceContext::new();
    let lhs = trace
        .input_symbolic_shape(tenferro_runtime::DType::F64, 2)
        .unwrap();
    let rhs = trace
        .input_symbolic_shape(tenferro_runtime::DType::F64, 2)
        .unwrap();
    let config = dot_general_config(
        TensorDotAxes::Axes {
            lhs: &[1],
            rhs: &[0],
        },
        lhs.rank(),
        rhs.rank(),
    )
    .unwrap();
    validate_traced_contract_dims(&lhs, &rhs, &config).unwrap();

    let lhs = trace
        .from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])
        .unwrap();
    let rhs = trace
        .from_vec_col_major(vec![4, 2], vec![1.0_f64; 8])
        .unwrap();
    let err = validate_traced_contract_dims(&lhs, &rhs, &config).unwrap_err();
    assert!(err.to_string().contains("contracted dimensions differ"));
}
