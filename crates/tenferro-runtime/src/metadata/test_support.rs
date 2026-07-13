use std::cell::Cell;

thread_local! {
    static GRAPH_ANALYSIS_VISITS: Cell<Option<(usize, usize)>> = const { Cell::new(None) };
}

pub(crate) fn with_visit_count<T>(f: impl FnOnce() -> T) -> (T, usize, usize) {
    GRAPH_ANALYSIS_VISITS.with(|visits| visits.set(Some((0, 0))));
    let result = f();
    let (graph_visits, operation_visits) =
        GRAPH_ANALYSIS_VISITS.with(|visits| visits.replace(None).unwrap_or_default());
    (result, graph_visits, operation_visits)
}

pub(super) fn record_graph_visit() {
    GRAPH_ANALYSIS_VISITS.with(|visits| {
        if let Some((graph_visits, operation_visits)) = visits.get() {
            visits.set(Some((graph_visits + 1, operation_visits)));
        }
    });
}

pub(super) fn record_operation_visit() {
    GRAPH_ANALYSIS_VISITS.with(|visits| {
        if let Some((graph_visits, operation_visits)) = visits.get() {
            visits.set(Some((graph_visits, operation_visits + 1)));
        }
    });
}
