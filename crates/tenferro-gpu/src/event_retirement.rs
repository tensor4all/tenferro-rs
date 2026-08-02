use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(Debug)]
pub(crate) enum EventDomainRunState {
    Pending,
    Retired,
    Failed,
}

pub(crate) fn take_pending_retirement(state: &mut EventDomainRunState) -> bool {
    match std::mem::replace(state, EventDomainRunState::Failed) {
        EventDomainRunState::Pending => true,
        EventDomainRunState::Retired => {
            *state = EventDomainRunState::Retired;
            false
        }
        EventDomainRunState::Failed => false,
    }
}

pub(crate) fn retire_pending(state: &mut EventDomainRunState, retirement: impl FnOnce()) {
    if take_pending_retirement(state) {
        retirement();
    }
}

/// Execute implicit provider retirement without allowing backend cleanup or
/// diagnostic formatting to unwind from a Drop implementation.
pub(crate) fn best_effort_retirement(retirement: impl FnOnce()) {
    if let Err(payload) = catch_unwind(AssertUnwindSafe(retirement)) {
        // Provider-controlled panic payloads can have arbitrary Drop behavior.
        // The retirement body has already been consumed, so retain no payload
        // ownership that could unwind after this containment boundary.
        std::mem::forget(payload);
    }
}

#[cfg(test)]
mod tests {
    use super::{best_effort_retirement, retire_pending, EventDomainRunState};

    #[derive(Debug)]
    struct PanickingPanicPayload;

    impl Drop for PanickingPanicPayload {
        fn drop(&mut self) {
            panic!("panic payload destructor escaped containment");
        }
    }

    #[test]
    fn implicit_retirement_contains_provider_panic() {
        best_effort_retirement(|| panic!("injected provider retirement panic"));
    }

    #[test]
    fn implicit_retirement_forgets_a_panicking_panic_payload() {
        best_effort_retirement(|| std::panic::panic_any(PanickingPanicPayload));
    }

    #[test]
    fn explicit_retirement_then_drop_invokes_retirement_once() {
        let mut state = EventDomainRunState::Pending;
        let mut retirements = 0;

        retire_pending(&mut state, || retirements += 1);
        state = EventDomainRunState::Retired;
        retire_pending(&mut state, || retirements += 1);

        assert_eq!(retirements, 1);
        assert!(matches!(state, EventDomainRunState::Retired));
    }

    #[test]
    fn implicit_drop_invokes_retirement_once() {
        let mut state = EventDomainRunState::Pending;
        let mut retirements = 0;

        retire_pending(&mut state, || retirements += 1);
        retire_pending(&mut state, || retirements += 1);

        assert_eq!(retirements, 1);
        assert!(matches!(state, EventDomainRunState::Failed));
    }
}
