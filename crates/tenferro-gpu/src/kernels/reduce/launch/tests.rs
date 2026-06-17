use crate::kernels::CubeclKernelError;

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

#[test]
fn validate_reduce_problem_rejects_input_shape_product_overflow() {
    let input_shape = [usize::MAX, 2];
    let err = super::validate_reduce_problem(&input_shape, &[usize::MAX, 1], 1).unwrap_err();

    assert_eq!(
        err,
        CubeclKernelError::InvalidStrategy {
            reason: format!(
                "reduction input element count overflows usize for shape {input_shape:?}"
            ),
        }
    );
}

#[test]
fn auto_strategy_uses_unit_only_within_bounded_axis_limit() {
    assert_eq!(
        super::auto_reduce_strategy_for_capabilities(32, 32, false).unwrap(),
        super::ResolvedReduceStrategy::Unit
    );
    assert_eq!(
        super::auto_reduce_strategy_for_capabilities(33, 32, true).unwrap(),
        super::ResolvedReduceStrategy::Plane
    );
}

#[test]
fn auto_strategy_rejects_large_axis_without_plane_ops() {
    let err = super::auto_reduce_strategy_for_capabilities(33, 32, false).unwrap_err();

    assert_eq!(
        err,
        CubeclKernelError::InvalidStrategy {
            reason: "Auto reduction cannot reduce axis length 33 without plane operations"
                .to_owned(),
        }
    );
}

#[test]
fn explicit_plane_strategy_requires_full_plane_and_plane_ops() {
    let problem = super::ReduceProblem {
        reduce_len: 31,
        reduce_count: 4,
        axis: 0,
    };

    assert_eq!(
        super::validate_plane_strategy(problem, 32, false).unwrap_err(),
        CubeclKernelError::InvalidStrategy {
            reason: "plane reduction requires backend plane operations".to_owned(),
        }
    );
    assert_eq!(
        super::validate_plane_strategy(problem, 32, true).unwrap_err(),
        CubeclKernelError::InvalidStrategy {
            reason: "plane reduction requires reduce axis length 31 to be at least plane width 32"
                .to_owned(),
        }
    );
}
