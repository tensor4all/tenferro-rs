use crate::CubeclKernelError;

#[test]
fn validate_reduce_problem_computes_single_axis_keepdims_contract() {
    let problem = super::validate_reduce_problem(&[2, 3, 4], &[2, 1, 4], 1).unwrap();

    assert_eq!(problem.reduce_len, 3);
    assert_eq!(problem.reduce_count, 8);
    assert_eq!(problem.axis, 1);
}

#[test]
fn validate_reduce_problem_rejects_zero_length_reduced_axis() {
    let err = super::validate_reduce_problem(&[2, 0, 4], &[2, 1, 4], 1).unwrap_err();

    assert_eq!(
        err,
        CubeclKernelError::InvalidStrategy {
            reason: "cannot reduce zero-length axis 1".to_owned(),
        }
    );
}

#[test]
fn validate_reduce_problem_rejects_non_keepdims_output_shape() {
    let err = super::validate_reduce_problem(&[2, 3, 4], &[2, 3, 1], 1).unwrap_err();

    assert_eq!(
        err,
        CubeclKernelError::MismatchOutputShape {
            expected: vec![2, 1, 4],
            actual: vec![2, 3, 1],
        }
    );
}
