use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::HashMap;
use std::env;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

#[derive(Debug, Default, Clone)]
struct EagerEinsumProfileEntry {
    calls: usize,
    total_time: Duration,
}

thread_local! {
    static EAGER_EINSUM_PROFILE_STATE: RefCell<HashMap<&'static str, EagerEinsumProfileEntry>> =
        RefCell::new(HashMap::new());
    #[cfg(test)]
    static EAGER_EINSUM_PROFILE_ENABLED_OVERRIDE: RefCell<Option<bool>> = const { RefCell::new(None) };
    #[cfg(test)]
    static EAGER_EINSUM_TRACE_ENABLED_OVERRIDE: RefCell<Option<bool>> = const { RefCell::new(None) };
    #[cfg(test)]
    static EAGER_EINSUM_PRINT_EVERY_OVERRIDE: RefCell<Option<Option<usize>>> = const { RefCell::new(None) };
}

pub(super) fn eager_einsum_profile_enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = EAGER_EINSUM_PROFILE_ENABLED_OVERRIDE.with(|state| *state.borrow()) {
        return value;
    }

    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env::var("TENFERRO_PROFILE_EAGER_EINSUM_AGG").is_ok())
}

pub(super) fn eager_einsum_trace_enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = EAGER_EINSUM_TRACE_ENABLED_OVERRIDE.with(|state| *state.borrow()) {
        return value;
    }

    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env::var("TENFERRO_PROFILE_EAGER_EINSUM").is_ok())
}

pub(super) fn record_eager_einsum_profile(section: &'static str, elapsed: Duration) {
    if !eager_einsum_profile_enabled() {
        return;
    }
    EAGER_EINSUM_PROFILE_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let entry = state.entry(section).or_default();
        entry.calls += 1;
        entry.total_time += elapsed;
    });
}

pub(super) fn profile_eager_einsum_section<T>(section: &'static str, f: impl FnOnce() -> T) -> T {
    if !eager_einsum_profile_enabled() {
        return f();
    }
    let started = Instant::now();
    let result = f();
    record_eager_einsum_profile(section, started.elapsed());
    result
}

pub(super) fn maybe_print_eager_einsum_profile() {
    if !eager_einsum_profile_enabled() {
        return;
    }
    let Some(print_every) = eager_einsum_profile_print_every() else {
        return;
    };
    if print_every == 0 {
        return;
    }

    let should_print = EAGER_EINSUM_PROFILE_STATE.with(|state| {
        state
            .borrow()
            .get("total")
            .is_some_and(|entry| entry.calls % print_every == 0)
    });
    if should_print {
        print_and_reset_eager_einsum_profile();
    }
}

fn eager_einsum_profile_print_every() -> Option<usize> {
    #[cfg(test)]
    if let Some(value) = EAGER_EINSUM_PRINT_EVERY_OVERRIDE.with(|state| *state.borrow()) {
        return value;
    }

    env::var("TENFERRO_PROFILE_EAGER_EINSUM_PRINT_EVERY")
        .ok()?
        .parse()
        .ok()
}

fn print_and_reset_eager_einsum_profile() {
    EAGER_EINSUM_PROFILE_STATE.with(|state| {
        let mut entries: Vec<_> = state
            .borrow()
            .iter()
            .map(|(section, entry)| (*section, entry.clone()))
            .collect();
        state.borrow_mut().clear();
        entries.sort_by_key(|(_, entry)| Reverse(entry.total_time));

        eprintln!("=== tenferro eager einsum profile ===");
        for (section, entry) in entries {
            eprintln!(
                "{section}: calls={} total={:.6}ms per_call={:.3}us",
                entry.calls,
                entry.total_time.as_secs_f64() * 1.0e3,
                entry.total_time.as_secs_f64() * 1.0e6 / entry.calls as f64,
            );
        }
    });
}

#[cfg(test)]
mod tests;
