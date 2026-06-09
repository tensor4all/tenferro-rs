use crate::kernels::CubeclKernelError;

use super::super::definition::{keepdims_output_shape, validate_keepdims_output_shape};

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
fn reduction_lengths_match_keepdims_primitive_contract() {
    let output_shape = keepdims_output_shape(&[2, 3, 4], 1).unwrap();

    assert_eq!(output_shape[1], 1);
    assert_eq!(output_shape.iter().product::<usize>(), 8);
}
