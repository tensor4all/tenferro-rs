use super::*;

#[test]
fn compact_host_accumulation_slice_selects_only_compact_host_views() {
    let mut compact_data = [0.0_f64; 4];
    let mut compact = TypedTensorViewMut::from_slice([2, 2], [1, 2], 0, &mut compact_data).unwrap();
    assert_eq!(
        compact_host_accumulation_slice(&mut compact, 4)
            .unwrap()
            .unwrap()
            .len(),
        4
    );

    let mut strided_data = [0.0_f64; 3];
    let mut strided = TypedTensorViewMut::from_slice([2], [2], 0, &mut strided_data).unwrap();
    assert!(compact_host_accumulation_slice(&mut strided, 2)
        .unwrap()
        .is_none());
}

#[test]
fn contraction_scalar_identity_errors_name_the_public_constructor() {
    let one_error = ContractionScalar::one(DType::I32).unwrap_err();
    assert!(matches!(
        one_error,
        Error::Validation {
            op: "ContractionScalar::one",
            source: ValidationError::DTypeMismatch { .. },
        }
    ));

    let zero_error = ContractionScalar::zero(DType::Bool).unwrap_err();
    assert!(matches!(
        zero_error,
        Error::Validation {
            op: "ContractionScalar::zero",
            source: ValidationError::DTypeMismatch { .. },
        }
    ));

    let overwrite_error = DotGeneralAccumulation::overwrite(DType::I32).unwrap_err();
    assert!(matches!(
        overwrite_error,
        Error::Validation {
            op: "DotGeneralAccumulation::overwrite",
            source: ValidationError::DTypeMismatch { .. },
        }
    ));

    let add_to_error = DotGeneralAccumulation::add_to(DType::Bool).unwrap_err();
    assert!(matches!(
        add_to_error,
        Error::Validation {
            op: "DotGeneralAccumulation::add_to",
            source: ValidationError::DTypeMismatch { .. },
        }
    ));

    let scaled_error =
        DotGeneralAccumulation::scaled(ContractionScalar::F32(1.0), ContractionScalar::F64(1.0))
            .unwrap_err();
    assert!(matches!(
        scaled_error,
        Error::Validation {
            op: "DotGeneralAccumulation::scaled",
            source: ValidationError::DTypeMismatch { .. },
        }
    ));
}

#[test]
fn nested_backend_session_entry_is_rejected_before_its_callback_runs() {
    use crate::tests::backend_default_read_tests::DefaultReadBackend;

    // The portable guard is per thread, so a second backend value entered from
    // inside a session closure is rejected like a re-entry of the same one.
    let mut backend = DefaultReadBackend::default();
    let mut other = DefaultReadBackend::default();
    let mut inner_ran = false;
    let nested = backend
        .with_backend_session(|_outer| other.with_backend_session(|_inner| inner_ran = true))
        .unwrap();
    assert!(matches!(
        nested,
        Err(SessionEntryError::Reentered {
            backend: "test backend"
        })
    ));
    assert!(!inner_ran);
    // The rejected nested entry leaves the outer guard's flag handling intact.
    assert!(!IN_SESSION.get());
    assert_eq!(backend.with_backend_session(|_| 4usize).unwrap(), 4);
}

#[test]
fn backend_session_entry_runs_and_clears_the_in_session_flag() {
    let mut backend = crate::tests::backend_default_read_tests::DefaultReadBackend::default();

    let first = backend.with_backend_session(|_| 1usize).unwrap();
    assert_eq!(first, 1);
    assert!(!IN_SESSION.get());

    // A second sequential session proves the first guard restored the flag.
    let second = backend.with_backend_session(|_| 2usize).unwrap();
    assert_eq!(second, 2);
    assert!(!IN_SESSION.get());
}

#[test]
fn backend_session_entry_clears_the_in_session_flag_after_panic() {
    // Panicking inside `f` must still restore the thread-local flag (the
    // guard is Drop-based), so a later session on the same thread succeeds.
    let mut backend = crate::tests::backend_default_read_tests::DefaultReadBackend::default();

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        backend
            .with_backend_session(|_| {
                assert!(IN_SESSION.get());
                panic!("boom");
            })
            .unwrap()
    }));
    assert!(outcome.is_err());
    assert!(!IN_SESSION.get());

    // The flag is usable again on the same thread.
    let again = backend.with_backend_session(|_| 3usize).unwrap();
    assert_eq!(again, 3);
    assert!(!IN_SESSION.get());
}

#[test]
fn with_session_entry_guard_rejects_nested_entry_in_every_build() {
    let mut inner_ran = false;
    let nested = with_session_entry_guard("test backend", || {
        with_session_entry_guard("test backend", || inner_ran = true)
    })
    .unwrap();
    assert!(matches!(
        nested,
        Err(SessionEntryError::Reentered {
            backend: "test backend"
        })
    ));
    assert!(!inner_ran);
    assert_eq!(
        nested.unwrap_err().kind(),
        tenferro_tensor_core::ErrorKind::RuntimeState
    );
}

#[test]
fn with_session_entry_guard_sets_and_restores_the_flag() {
    let value = with_session_entry_guard("test backend", || 1usize).unwrap();
    assert_eq!(value, 1);
    assert!(!IN_SESSION.get());

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_session_entry_guard("test backend", || panic!("boom"))
    }));
    assert!(outcome.is_err());
    assert!(!IN_SESSION.get());

    let again = with_session_entry_guard("test backend", || 2usize).unwrap();
    assert_eq!(again, 2);
}

#[test]
fn held_session_marker_stays_visible_for_its_whole_lifetime() {
    assert!(!has_held_backend_session());
    let marker = HeldSessionMarker::enter("held backend").unwrap();
    // Unlike the callback-scoped guard, the marker does not depend on any closure being
    // active: owners that serialize callers must see it between operations too.
    assert!(has_held_backend_session());
    assert_eq!(marker.backend(), "held backend");
    drop(marker);
    assert!(!has_held_backend_session());
}

#[test]
fn held_session_marker_rejects_a_second_marker_and_names_the_holder() {
    let _held = HeldSessionMarker::enter("first backend").unwrap();
    let second = HeldSessionMarker::enter("second backend");
    assert!(matches!(
        second,
        Err(SessionEntryError::Reentered {
            backend: "first backend"
        })
    ));
    // The rejected entry must not replace the live holder.
    assert!(has_held_backend_session());
}

#[test]
fn held_session_marker_is_cleared_on_unwind() {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _held = HeldSessionMarker::enter("held backend").unwrap();
        panic!("boom");
    }));
    assert!(outcome.is_err());
    assert!(!has_held_backend_session());
    assert!(HeldSessionMarker::enter("next backend").is_ok());
}
