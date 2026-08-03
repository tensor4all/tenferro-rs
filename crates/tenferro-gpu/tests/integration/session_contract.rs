#[cfg(any(feature = "cuda", feature = "webgpu"))]
use std::any::type_name;

#[cfg(feature = "webgpu")]
#[test]
fn webgpu_session_type_is_distinct_from_owner() {
    use tenferro_gpu::{WebGpuBackend, WebGpuExecSession};

    assert_ne!(
        type_name::<WebGpuBackend>(),
        type_name::<WebGpuExecSession<'static>>()
    );
}

#[cfg(feature = "webgpu")]
#[test]
fn webgpu_session_exposes_backend_session_operations() {
    use tenferro_gpu::WebGpuExecSession;
    use tenferro_tensor::{BackendSession, Tensor, TensorElementwise};

    fn assert_backend_session<S: BackendSession + ?Sized>() {}

    assert_backend_session::<WebGpuExecSession<'static>>();
    let _add: fn(
        &mut WebGpuExecSession<'static>,
        &Tensor,
        &Tensor,
    ) -> tenferro_tensor::Result<Tensor> = TensorElementwise::add;
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_session_type_is_distinct_from_owner() {
    use tenferro_gpu::{CudaBackend, CudaExecSession};

    assert_ne!(
        type_name::<CudaBackend>(),
        type_name::<CudaExecSession<'static>>()
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_session_exposes_backend_session_operations() {
    use tenferro_gpu::CudaExecSession;
    use tenferro_tensor::{BackendSession, Tensor, TensorElementwise};

    fn assert_backend_session<S: BackendSession + ?Sized>() {}

    assert_backend_session::<CudaExecSession<'static>>();
    let _add: fn(
        &mut CudaExecSession<'static>,
        &Tensor,
        &Tensor,
    ) -> tenferro_tensor::Result<Tensor> = TensorElementwise::add;
}

#[cfg(any(feature = "cuda", feature = "webgpu"))]
#[test]
fn execution_session_capability_cannot_project_or_escape_owner_borrow() {
    if std::env::var_os("NEXTEST").is_some()
        && std::env::var("CARGO_NET_OFFLINE").is_ok_and(|value| value == "true" || value == "1")
    {
        eprintln!("skipping compile-only trybuild contract in an offline nextest archive");
        return;
    }

    let tests = trybuild::TestCases::new();

    #[cfg(feature = "webgpu")]
    {
        tests.compile_fail("tests/ui/webgpu_session_borrow_escape.rs");
        tests.compile_fail("tests/ui/webgpu_session_owner_projection.rs");
    }

    #[cfg(feature = "cuda")]
    {
        tests.compile_fail("tests/ui/cuda_session_borrow_escape.rs");
        tests.compile_fail("tests/ui/cuda_session_owner_projection.rs");
        tests.compile_fail("tests/ui/cuda_removed_runtime_engine_id.rs");
        tests.compile_fail("tests/ui/cuda_removed_runtime_engine_registration_with_id.rs");
        tests.compile_fail("tests/ui/cuda_backend_rejects_bare_integer.rs");
    }
}
