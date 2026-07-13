use std::cell::Cell;

thread_local! {
    static CONSTRAINT_SCOPE_CLONES: Cell<Option<usize>> = const { Cell::new(None) };
}

pub(super) fn with_constraint_scope_clone_count<T>(f: impl FnOnce() -> T) -> (T, usize) {
    CONSTRAINT_SCOPE_CLONES.with(|count| count.set(Some(0)));
    let result = f();
    let clones = CONSTRAINT_SCOPE_CLONES.with(|count| count.replace(None).unwrap_or_default());
    (result, clones)
}

pub(super) fn record_constraint_scope_clones(clones: usize) {
    CONSTRAINT_SCOPE_CLONES.with(|count| {
        if let Some(current) = count.get() {
            count.set(Some(current + clones));
        }
    });
}
