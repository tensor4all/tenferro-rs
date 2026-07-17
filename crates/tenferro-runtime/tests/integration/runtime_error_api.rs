use tenferro_runtime::{Error, ErrorPhase};
use tenferro_tensor::{ErrorKind, ShapeMismatch, ValidationKind};

#[test]
fn runtime_validation_separates_kind_from_phase() {
    let err = Error::validation(
        "reshape",
        ErrorPhase::GraphBuild,
        ShapeMismatch::ReshapeElementCount { from: 2, to: 3 }.into(),
    );

    assert_eq!(
        err.kind(),
        ErrorKind::Validation(ValidationKind::ShapeMismatch)
    );
    assert_eq!(err.phase(), Some(ErrorPhase::GraphBuild));
}
