use std::error::Error as _;

use tenferro_einsum::{Error, PlanningError, Result};
use tenferro_tensor::{Error as TensorError, ErrorKind, ShapeMismatch, ShapeVec, ValidationKind};

#[test]
fn invalid_subscripts_are_a_typed_local_validation_error() {
    let result: Result<()> = Err(Error::invalid_subscripts("bad labels"));

    let Err(err) = result else {
        panic!("expected invalid subscripts error");
    };

    assert_eq!(
        err.kind(),
        ErrorKind::Validation(ValidationKind::InvalidArgument)
    );
    assert!(err.to_string().contains("bad labels"));
}

#[test]
fn shared_einsum_validation_promotes_directly_to_tensor_validation() {
    let err = Error::validation(
        "einsum",
        ShapeMismatch::ExpectedActual {
            expected: ShapeVec::from_vec(vec![2, 3]),
            actual: ShapeVec::from_vec(vec![2, 4]),
        }
        .into(),
    );

    let tensor_err = err.into_tensor_error("einsum_extension");
    assert!(matches!(tensor_err, TensorError::Validation { .. }));
}

#[test]
fn einsum_planning_error_remains_a_typed_extension_source() {
    let err = Error::planning("no valid contraction path");
    let tensor_err = err.into_tensor_error("einsum_extension");

    assert_eq!(
        tensor_err.kind(),
        ErrorKind::Validation(ValidationKind::InvalidArgument)
    );
    assert!(matches!(tensor_err, TensorError::Extension { .. }));
    assert!(tensor_err.source().is_some());
}

#[test]
fn einsum_planning_runtime_state_is_classified_separately() {
    let err = Error::planning_runtime_state("planner state is unavailable");

    assert_eq!(err.kind(), ErrorKind::RuntimeState);
    assert!(matches!(
        err,
        Error::Planning {
            source: PlanningError::RuntimeState { .. }
        }
    ));
}
