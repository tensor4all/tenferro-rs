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
    if catch_unwind(AssertUnwindSafe(retirement)).is_err() {
        // The retirement body has been consumed; its cleanup panic is contained.
    }
}

#[cfg(test)]
mod tests {
    use super::{best_effort_retirement, retire_pending, EventDomainRunState};

    #[test]
    fn implicit_retirement_contains_provider_panic() {
        best_effort_retirement(|| panic!("injected provider retirement panic"));
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
