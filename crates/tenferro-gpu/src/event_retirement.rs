use std::panic::{catch_unwind, AssertUnwindSafe};

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
    use super::best_effort_retirement;

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
}
