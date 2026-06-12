use tenferro_cpu::CpuBackend;
use tenferro_tensor::{
    BackendSession, BackendSessionHost, TensorAnalytic, TensorBackend, TensorBuffer,
    TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing,
    TensorReduction, TensorStructural,
};

fn accepts_backend_capabilities<B>()
where
    B: TensorElementwise
        + TensorAnalytic
        + TensorStructural
        + TensorReduction
        + TensorIndexing
        + TensorDot
        + TensorFusion
        + TensorBuffer
        + TensorDeviceTransfer
        + BackendSessionHost
        + TensorBackend,
{
}

fn accepts_session_capabilities<S>(_: &mut S)
where
    S: TensorElementwise
        + TensorAnalytic
        + TensorStructural
        + TensorReduction
        + TensorIndexing
        + TensorDot
        + TensorFusion
        + TensorBuffer
        + BackendSession
        + ?Sized,
{
}

#[test]
fn cpu_backend_exposes_narrow_capability_bounds() {
    accepts_backend_capabilities::<CpuBackend>();
}

#[test]
fn backend_session_exposes_narrow_capability_bounds() {
    let mut backend = CpuBackend::new();
    backend.with_backend_session(|session| {
        accepts_session_capabilities(session);
    });
}

#[test]
fn backend_surface_no_longer_uses_forwarding_macro() {
    let backend_source = include_str!("../src/backend.rs");
    assert!(!backend_source.contains("forward_exec_to_backend"));
}

#[test]
fn read_elementwise_and_analytic_paths_do_not_materialize_views() {
    let elementwise_source = include_str!("../src/elementwise.rs");
    let analytic_source = include_str!("../src/analytic.rs");

    assert!(
        !elementwise_source.contains("materialize_tensor_read"),
        "elementwise read paths must dispatch over TensorRead views directly"
    );
    assert!(
        !analytic_source.contains("materialize_tensor_read"),
        "analytic read paths must dispatch over TensorRead views directly"
    );
}

#[test]
fn indexing_hot_loops_do_not_recompute_multi_indices_from_flat_offsets() {
    let indexing_source = include_str!("../src/indexing.rs");

    assert!(
        !indexing_source.contains("flat_to_multi"),
        "indexing kernels should carry column-major indices incrementally after validation"
    );
}

#[test]
fn concatenate_hot_loop_does_not_linearly_scan_input_segments() {
    let indexing_source = include_str!("../src/indexing.rs");

    assert!(
        !indexing_source.contains(".position(|&end| concat_idx < end)"),
        "concatenate should not linearly scan all input segment ends for each output element"
    );
    assert!(
        indexing_source.contains("partition_point"),
        "concatenate should use precomputed ordered segment boundaries for logarithmic lookup"
    );
}

#[test]
fn gather_scatter_index_component_reuses_index_scratch() {
    let indexing_source = include_str!("../src/indexing.rs");

    assert!(
        !indexing_source.contains("let mut full_idx = vec![0usize; indices.shape.len()];"),
        "gather/scatter should not allocate index vectors for every index component"
    );
    assert!(
        indexing_source.contains("index_scratch"),
        "gather/scatter should carry reusable index scratch through index_component"
    );
}
