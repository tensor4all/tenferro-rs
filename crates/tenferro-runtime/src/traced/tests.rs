use crate::{DType, DotGeneralConfig, TracedTensor};

#[test]
fn dot_general_returns_error_for_invalid_config() {
    let lhs = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    let rhs = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![2],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let Err(err) = lhs.dot_general(&rhs, config) else {
        panic!("invalid dim config should be a typed runtime error");
    };

    let message = err.to_string();
    assert!(
        message.contains("lhs_contracting_dim 2 out of bounds"),
        "{message}"
    );
}

#[test]
fn integer_scale_real_rejects_non_finite_factors() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1_i64, 2]).unwrap();

    for factor in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let err = x.scale_real(factor).unwrap_err();
        assert!(
            err.to_string().contains("finite"),
            "expected finite-value error, got {err:?}"
        );
    }
}

#[test]
fn dot_general_keeps_existing_success_metadata() {
    let lhs = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    let rhs = TracedTensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let out = lhs
        .dot_general(&rhs, config)
        .expect("valid dot_general config should build a traced tensor");

    assert_eq!(out.rank, 2);
    assert_eq!(out.try_concrete_shape().unwrap(), &[2, 4]);
}

#[test]
fn reductions_reject_invalid_axes_instead_of_saturating_rank() {
    let x = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();

    let out_of_bounds = x.reduce_max(&[2]).unwrap_err();
    assert!(
        out_of_bounds
            .to_string()
            .contains("axis 2 out of bounds for rank 2"),
        "{out_of_bounds}"
    );

    let duplicate = x.reduce_min(&[0, 0]).unwrap_err();
    assert!(
        duplicate.to_string().contains("duplicate reduction axis 0"),
        "{duplicate}"
    );

    let y = x.reduce_sum(&[1]).unwrap();
    assert_eq!(y.rank, 1);
    assert_eq!(y.try_concrete_shape().unwrap(), &[2]);
}

#[test]
fn symbolic_axis_accessors_reject_invalid_axes() {
    let x = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();

    let sym_size_err = x.sym_size(2).unwrap_err();
    assert!(
        sym_size_err
            .to_string()
            .contains("axis 2 out of bounds for rank 2"),
        "{sym_size_err}"
    );

    let axis_err = x.axis_sym_dim(2).unwrap_err();
    assert!(
        axis_err
            .to_string()
            .contains("axis 2 out of bounds for rank 2"),
        "{axis_err}"
    );

    assert_eq!(x.axis_sym_dim(0).unwrap().constant_value(), Some(2));
}

#[test]
fn broadcast_in_dim_sym_rejects_missing_shape_reference() {
    let lhs = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let rhs = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let target_shape = [lhs.axis_sym_dim(0).unwrap(), rhs.axis_sym_dim(0).unwrap()];

    let err = lhs
        .broadcast_in_dim_sym(&target_shape, &[0], &[])
        .unwrap_err();

    let message = err.to_string();
    assert!(
        message.contains("broadcast_in_dim_sym")
            && message.contains("unresolved symbolic dimension")
            && message.contains("shape_refs"),
        "{message}"
    );
}

#[test]
fn broadcast_in_dim_rejects_invalid_dimension_mappings() {
    let x = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();

    let rank_mismatch = x.broadcast_in_dim(&[2, 3, 4], &[0]).unwrap_err();
    assert!(
        rank_mismatch
            .to_string()
            .contains("dims length 1 must match input rank 2"),
        "{rank_mismatch}"
    );

    let out_of_bounds = x.broadcast_in_dim(&[2, 3, 4], &[0, 3]).unwrap_err();
    assert!(
        out_of_bounds
            .to_string()
            .contains("broadcast dim 3 out of bounds for output rank 3"),
        "{out_of_bounds}"
    );

    let duplicate = x.broadcast_in_dim(&[2, 3, 4], &[1, 1]).unwrap_err();
    assert!(
        duplicate.to_string().contains("duplicate broadcast dim 1"),
        "{duplicate}"
    );

    let valid = x.broadcast_in_dim(&[2, 3, 4], &[0, 1]).unwrap();
    assert_eq!(valid.rank, 3);
    assert_eq!(valid.try_concrete_shape().unwrap(), &[2, 3, 4]);
}
