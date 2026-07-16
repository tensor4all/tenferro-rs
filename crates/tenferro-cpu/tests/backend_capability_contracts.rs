use tenferro_cpu::CpuBackend;
use tenferro_tensor::{
    BackendSession, BackendSessionHost, TensorAnalytic, TensorBackend, TensorBuffer,
    TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing,
    TensorReduction, TensorStructural,
};

fn rust_function_body<'a>(source: &'a str, function: &str) -> Option<&'a str> {
    let signature = format!("fn {function}(");
    let function_start = source.find(&signature)?;
    let body_start = function_start + source[function_start..].find('{')?;
    let mut depth = 0usize;
    for (offset, character) in source[body_start..].char_indices() {
        match character {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&source[body_start..=body_start + offset]);
                }
            }
            _ => {}
        }
    }
    None
}

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
fn structural_read_paths_dispatch_directly_to_typed_view_helpers() {
    let backend_source = include_str!("../src/backend.rs");
    let session_source = include_str!("../src/exec_session.rs");
    let structural_source = include_str!("../src/structural.rs");

    for (surface, source) in [
        ("CpuBackend", backend_source),
        ("CpuExecSession", session_source),
    ] {
        let structural_impl = source
            .split_once(&format!("impl TensorStructural for {surface}"))
            .expect("TensorStructural implementation must exist")
            .1;
        for (operation, helper) in [
            ("transpose_read", "transpose_read_with_pool"),
            ("reshape_read", "reshape_read_with_pool"),
            ("broadcast_in_dim_read", "broadcast_in_dim_read_with_pool"),
        ] {
            let implementation = rust_function_body(structural_impl, operation)
                .unwrap_or_else(|| panic!("{surface}::{operation} must be implemented"));
            assert!(
                !implementation.contains("materialize_tensor_read"),
                "{surface}::{operation} must not materialize an intermediate input"
            );
            assert!(
                implementation.contains(&format!("structural::{helper}")),
                "{surface}::{operation} must dispatch to structural::{helper}"
            );
        }
    }

    for (read_helper, typed_view_helper) in [
        ("transpose_read_with_pool", "typed_transpose_view_with_pool"),
        ("reshape_read_with_pool", "typed_reshape_view_with_pool"),
        (
            "broadcast_in_dim_read_with_pool",
            "typed_broadcast_in_dim_view_with_pool",
        ),
    ] {
        let implementation = rust_function_body(structural_source, read_helper)
            .unwrap_or_else(|| panic!("structural::{read_helper} must own dtype dispatch"));
        assert!(
            implementation.contains(typed_view_helper),
            "structural::{read_helper} must dispatch to {typed_view_helper}"
        );
    }
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

#[test]
fn cpu_public_ops_require_backend_owner() {
    let lib_source = include_str!("../src/lib.rs");
    for reexport in [
        "pub use analytic::pow;",
        "pub use elementwise::",
        "pub use indexing::",
        "pub use reduction::",
        "pub use structural::",
    ] {
        assert!(
            !lib_source.contains(reexport),
            "resource-bypassing reexport remains: {reexport}"
        );
    }

    for (module, source) in [
        ("analytic", include_str!("../src/analytic.rs")),
        ("elementwise", include_str!("../src/elementwise.rs")),
        ("indexing", include_str!("../src/indexing.rs")),
        ("structural", include_str!("../src/structural.rs")),
    ] {
        assert!(
            !source.contains("fn with_local_pool"),
            "{module} still constructs a throwaway BufferPool"
        );
    }
}

#[test]
fn install_pool_has_no_placeholder_construction_or_gemm_descriptor_clones() {
    let backend_source = include_str!("../src/backend.rs");
    let buffer_pool_source = include_str!("../src/buffer_pool.rs");
    let gemm_source = include_str!("../src/gemm/mod.rs");
    let exec_session_source = include_str!("../src/exec_session.rs");

    assert!(!backend_source.contains("std::mem::take(target)"));
    assert!(backend_source.contains("buffers: &'a mut BufferPool"));
    assert!(buffer_pool_source.contains("OnceLock"));
    assert!(buffer_pool_source.contains("parse_default_max_retained_capacity_bytes"));
    assert!(gemm_source.contains("lhs: &TensorRead<'_>"));
    assert!(gemm_source.contains("rhs: &TensorRead<'_>"));
    assert!(!backend_source.contains("lhs.clone()"));
    assert!(!backend_source.contains("rhs.clone()"));
    assert!(!exec_session_source.contains("lhs.clone()"));
    assert!(!exec_session_source.contains("rhs.clone()"));
}
