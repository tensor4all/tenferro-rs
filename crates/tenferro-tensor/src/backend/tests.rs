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
#[cfg(debug_assertions)]
#[should_panic(expected = "nested backend session entry")]
fn nested_default_backend_session_is_rejected_in_debug_builds() {
    use crate::tests::backend_default_read_tests::DefaultReadBackend;

    let mut backend = DefaultReadBackend::default();
    default_backend_session(&mut backend, |outer| {
        // Recover the concrete backend from the session through the
        // documented capability bridge so the nested call re-enters
        // `default_backend_session` on the same backend, same thread.
        let concrete: &mut DefaultReadBackend =
            unsafe { &mut *outer.session_data_mut().cast::<DefaultReadBackend>() };
        default_backend_session(concrete, |_inner| ())
    });
}

#[test]
fn default_backend_session_runs_and_clears_the_in_session_flag() {
    let mut backend = crate::tests::backend_default_read_tests::DefaultReadBackend::default();

    let first = default_backend_session(&mut backend, |_| 1usize);
    assert_eq!(first, 1);
    assert!(!IN_SESSION.get());

    // A second sequential session proves the first guard restored the flag.
    let second = default_backend_session(&mut backend, |_| 2usize);
    assert_eq!(second, 2);
    assert!(!IN_SESSION.get());
}

#[test]
fn default_backend_session_clears_the_in_session_flag_after_panic() {
    // Panicking inside `f` must still restore the thread-local flag (the
    // guard is Drop-based), so a later session on the same thread succeeds.
    let mut backend = crate::tests::backend_default_read_tests::DefaultReadBackend::default();

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        default_backend_session(&mut backend, |_| {
            assert!(IN_SESSION.get());
            panic!("boom");
        })
    }));
    assert!(outcome.is_err());
    assert!(!IN_SESSION.get());

    // The flag is usable again on the same thread.
    let again = default_backend_session(&mut backend, |_| 3usize);
    assert_eq!(again, 3);
    assert!(!IN_SESSION.get());
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "nested backend session entry")]
fn with_session_entry_guard_rejects_nested_entry_in_debug_builds() {
    with_session_entry_guard(|| with_session_entry_guard(|| ()))
}

#[test]
fn with_session_entry_guard_sets_and_restores_the_flag() {
    let value = with_session_entry_guard(|| 1usize);
    assert_eq!(value, 1);
    assert!(!IN_SESSION.get());

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_session_entry_guard(|| panic!("boom"))
    }));
    assert!(outcome.is_err());
    assert!(!IN_SESSION.get());

    let again = with_session_entry_guard(|| 2usize);
    assert_eq!(again, 2);
}
