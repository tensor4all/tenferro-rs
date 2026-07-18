use crate::{DotGeneralConfig, Error, ValidationError};

#[test]
fn validate_dims_with_explicit_ranks_rejects_out_of_range_contract() {
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![2],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let err = config
        .validate_dims_with_ranks(2, 2)
        .expect_err("dim index 2 is out of range for rank 2");
    assert!(matches!(
        err,
        Error::Validation {
            op: "dot_general",
            source: ValidationError::AxisOutOfBounds { .. },
            ..
        }
    ));
    assert!(err.to_string().contains("out of bounds"));
}

#[test]
fn validate_dims_with_explicit_ranks_accepts_valid_config() {
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    config.validate_dims_with_ranks(2, 2).unwrap();
}
