//! Lower-provider error mapping for the contraction boundary.
//!
//! The new provider errors (`Workspace`, declined strategies) must reach the
//! tensor boundary as typed sources, not as Internal or string-only failures.

use std::error::Error as _;

use super::LowerContractionError;
use tenferro_tensor::{Error as TensorError, ErrorKind};

fn has_source<T: std::error::Error + 'static>(error: &TensorError) -> bool {
    error
        .source()
        .and_then(|source| source.downcast_ref::<T>())
        .is_some()
}

#[test]
fn cpueinsum_workspace_is_a_typed_runtime_state_failure() {
    let error = cpueinsum::Error::Workspace {
        needed: 16,
        have: 8,
    }
    .into_crate_error("dot_general");
    assert_eq!(error.kind(), ErrorKind::RuntimeState);
    assert!(has_source::<cpueinsum::Error>(&error));
}

#[test]
fn cpueinsum_declined_route_keeps_its_typed_source() {
    let declined = cpueinsum::Error::Contract {
        step: 0,
        source: cpueinsum::tcontract::Error::Unsupported(
            cpueinsum::tcontract::api::Unsupported::Reason("test decline"),
        ),
    };
    let error = declined.into_crate_error("dot_general");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(has_source::<cpueinsum::Error>(&error));
}

#[test]
fn bare_tcontract_decline_keeps_its_typed_source() {
    let error = cpueinsum::tcontract::Error::Unsupported(
        cpueinsum::tcontract::api::Unsupported::Reason("test decline"),
    )
    .into_crate_error("dot_general");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(has_source::<cpueinsum::tcontract::Error>(&error));
}

#[cfg(feature = "blas")]
#[test]
fn blas_workspace_is_a_typed_runtime_state_failure() {
    let error =
        cpueinsum_blas::BlasError::Work { need: 16, got: 8 }.into_crate_error("dot_general");
    assert_eq!(error.kind(), ErrorKind::RuntimeState);
    assert!(has_source::<cpueinsum_blas::BlasError>(&error));
}

#[cfg(feature = "blas")]
#[test]
fn blas_strategy_unavailable_is_typed_unsupported() {
    let error = cpueinsum_blas::BlasError::StrategyUnavailable.into_crate_error("dot_general");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(has_source::<cpueinsum_blas::BlasError>(&error));
}
