use crate::{DotGeneralConfig, TracedTensor};

#[test]
fn try_dot_general_returns_error_for_invalid_config() {
    let lhs = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
    let rhs = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![2],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let Err(err) = lhs.try_dot_general(&rhs, config) else {
        panic!("invalid dim config should be a typed runtime error");
    };

    let message = err.to_string();
    assert!(
        message.contains("lhs_contracting_dim 2 out of bounds"),
        "{message}"
    );
}

#[test]
fn try_dot_general_keeps_existing_success_metadata() {
    let lhs = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
    let rhs = TracedTensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let out = lhs
        .try_dot_general(&rhs, config)
        .expect("valid dot_general config should build a traced tensor");

    assert_eq!(out.rank, 2);
    assert_eq!(out.try_concrete_shape().unwrap(), &[2, 4]);
}
