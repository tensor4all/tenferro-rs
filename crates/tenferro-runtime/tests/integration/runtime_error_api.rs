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

#[test]
fn runtime_state_preserves_kind_phase_and_source() {
    use std::error::Error as _;

    let source = std::io::Error::other("registry lock poisoned");
    let err = Error::runtime_state_source("metadata", ErrorPhase::Compile, source);

    assert_eq!(err.kind(), ErrorKind::RuntimeState);
    assert_eq!(err.phase(), Some(ErrorPhase::Compile));
    assert!(err.source().is_some());
}
