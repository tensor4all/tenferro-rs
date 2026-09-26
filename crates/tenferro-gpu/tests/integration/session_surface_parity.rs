//! One-interface contract across the shipped backends (issues #1926 / #1929).
//!
//! The unification's claim is that a caller writes the same operation code
//! against any backend's session. The compiler already enforces this per type;
//! what it does not state anywhere is that every session type the workspace
//! ships exposes the *same* operation surface. This contract asserts exactly
//! that, so a backend cannot quietly expose a smaller surface, and the CPU half
//! runs in every configuration while the CUDA and WebGPU halves follow their
//! features.
//!
//! It complements the source-text contracts: `backend_read_contract` checks the
//! CUDA bodies and `webgpu_backend_contract` names individual methods, whereas
//! this one compares the three session types against one bound.

use tenferro_tensor::backend::{
    BackendSession, SessionCachedDot, TensorAnalytic, TensorDot, TensorElementwise, TensorIndexing,
    TensorReduction, TensorStructural,
};

/// The operation surface every backend session must expose.
///
/// `SessionCachedDot: TensorDot`, so the cached contraction entry is part of the
/// same bound rather than a backend-specific extra.
fn assert_common_session_surface<S>()
where
    S: BackendSession
        + SessionCachedDot
        + TensorElementwise
        + TensorAnalytic
        + TensorStructural
        + TensorReduction
        + TensorIndexing
        + TensorDot,
{
}

#[test]
fn cpu_session_exposes_the_common_surface() {
    assert_common_session_surface::<tenferro_cpu::CpuExecSession<'static>>();
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_session_exposes_the_common_surface() {
    assert_common_session_surface::<tenferro_gpu::cuda::CudaExecSession<'static>>();
}

#[cfg(feature = "webgpu")]
#[test]
fn webgpu_session_exposes_the_common_surface() {
    assert_common_session_surface::<tenferro_gpu::webgpu::WebGpuExecSession<'static>>();
}
