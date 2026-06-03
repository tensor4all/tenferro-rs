use crate::kernels::reduce::{
    axis_reduce_len, keepdims_output_shape, reduced_output_len, supports_dtype,
    validate_keepdims_output_shape, ReduceDType, ReduceOp,
};
use crate::kernels::CubeclKernelError;

#[test]
fn keepdims_output_shape_sets_only_reduced_axis_to_one() {
    assert_eq!(keepdims_output_shape(&[2, 3, 4], 1).unwrap(), vec![2, 1, 4]);
}

#[test]
fn keepdims_output_shape_rejects_axis_equal_to_rank() {
    let err = keepdims_output_shape(&[2, 3], 2).unwrap_err();

    assert_eq!(err, CubeclKernelError::InvalidAxis { axis: 2, rank: 2 });
}

#[test]
fn validate_keepdims_output_shape_accepts_expected_shape() {
    validate_keepdims_output_shape(&[2, 3, 4], &[2, 1, 4], 1).unwrap();
}

#[test]
fn validate_keepdims_output_shape_reports_expected_shape() {
    let err = validate_keepdims_output_shape(&[2, 3, 4], &[2, 3, 1], 1).unwrap_err();

    assert_eq!(
        err,
        CubeclKernelError::MismatchOutputShape {
            expected: vec![2, 1, 4],
            actual: vec![2, 3, 1],
        }
    );
}

#[test]
fn axis_reduce_len_rejects_invalid_axis() {
    let err = axis_reduce_len(&[2, 3], 2).unwrap_err();

    assert_eq!(err, CubeclKernelError::InvalidAxis { axis: 2, rank: 2 });
}

#[test]
fn reduction_lengths_match_keepdims_primitive_contract() {
    assert_eq!(axis_reduce_len(&[2, 3, 4], 1).unwrap(), 3);
    assert_eq!(reduced_output_len(&[2, 3, 4], 1).unwrap(), 8);
}

#[test]
fn support_table_matches_first_split_scope() {
    for dtype in [
        ReduceDType::F32,
        ReduceDType::F64,
        ReduceDType::I64,
        ReduceDType::Complex32,
        ReduceDType::Complex64,
    ] {
        assert!(supports_dtype(ReduceOp::Sum, dtype));
        assert!(supports_dtype(ReduceOp::Prod, dtype));
    }

    assert!(supports_dtype(ReduceOp::Max, ReduceDType::F32));
    assert!(supports_dtype(ReduceOp::Max, ReduceDType::F64));
    assert!(supports_dtype(ReduceOp::Min, ReduceDType::F32));
    assert!(supports_dtype(ReduceOp::Min, ReduceDType::F64));
    assert!(!supports_dtype(ReduceOp::Max, ReduceDType::I64));
    assert!(!supports_dtype(ReduceOp::Min, ReduceDType::Complex32));
    assert!(!supports_dtype(ReduceOp::Max, ReduceDType::Complex64));
}
