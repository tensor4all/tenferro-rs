use std::time::Duration;

use super::*;

struct ProfileOverrideGuard;

impl ProfileOverrideGuard {
    fn set(profile_enabled: bool, trace_enabled: bool, print_every: Option<usize>) -> Self {
        EAGER_EINSUM_PROFILE_ENABLED_OVERRIDE.with(|state| {
            *state.borrow_mut() = Some(profile_enabled);
        });
        EAGER_EINSUM_TRACE_ENABLED_OVERRIDE.with(|state| {
            *state.borrow_mut() = Some(trace_enabled);
        });
        EAGER_EINSUM_PRINT_EVERY_OVERRIDE.with(|state| {
            *state.borrow_mut() = Some(print_every);
        });
        EAGER_EINSUM_PROFILE_STATE.with(|state| {
            state.borrow_mut().clear();
        });
        Self
    }
}

impl Drop for ProfileOverrideGuard {
    fn drop(&mut self) {
        EAGER_EINSUM_PROFILE_ENABLED_OVERRIDE.with(|state| {
            *state.borrow_mut() = None;
        });
        EAGER_EINSUM_TRACE_ENABLED_OVERRIDE.with(|state| {
            *state.borrow_mut() = None;
        });
        EAGER_EINSUM_PRINT_EVERY_OVERRIDE.with(|state| {
            *state.borrow_mut() = None;
        });
        EAGER_EINSUM_PROFILE_STATE.with(|state| {
            state.borrow_mut().clear();
        });
    }
}

#[test]
fn disabled_profile_bypasses_recording() {
    let _guard = ProfileOverrideGuard::set(false, false, Some(1));

    record_eager_einsum_profile("total", Duration::from_micros(1));
    let value = profile_eager_einsum_section("phase", || 17);
    maybe_print_eager_einsum_profile();

    assert_eq!(value, 17);
    assert!(!eager_einsum_profile_enabled());
    assert!(!eager_einsum_trace_enabled());
    EAGER_EINSUM_PROFILE_STATE.with(|state| {
        assert!(state.borrow().is_empty());
    });
}

#[test]
fn enabled_profile_records_sections_and_print_reset_clears_state() {
    let _guard = ProfileOverrideGuard::set(true, true, Some(1));

    let value = profile_eager_einsum_section("phase", || 23);
    record_eager_einsum_profile("total", Duration::from_micros(2));

    assert_eq!(value, 23);
    assert!(eager_einsum_profile_enabled());
    assert!(eager_einsum_trace_enabled());
    EAGER_EINSUM_PROFILE_STATE.with(|state| {
        let state = state.borrow();
        assert_eq!(state.get("phase").map(|entry| entry.calls), Some(1));
        assert_eq!(state.get("total").map(|entry| entry.calls), Some(1));
    });

    maybe_print_eager_einsum_profile();

    EAGER_EINSUM_PROFILE_STATE.with(|state| {
        assert!(state.borrow().is_empty());
    });
}

#[test]
fn print_every_zero_keeps_recorded_state() {
    let _guard = ProfileOverrideGuard::set(true, false, Some(0));

    record_eager_einsum_profile("total", Duration::from_micros(1));
    maybe_print_eager_einsum_profile();

    EAGER_EINSUM_PROFILE_STATE.with(|state| {
        assert_eq!(
            state.borrow().get("total").map(|entry| entry.calls),
            Some(1)
        );
    });
}
