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
