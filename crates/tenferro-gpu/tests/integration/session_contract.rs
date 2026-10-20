#[cfg(feature = "webgpu")]
#[test]
fn webgpu_engine_exposes_engine_operations() {
    use tenferro_gpu::webgpu::WebGpuBackend;
    use tenferro_tensor::{Engine, Tensor, TensorElementwise, TensorRead};

    fn assert_backend_session<S: Engine + ?Sized>() {}

    assert_backend_session::<WebGpuBackend>();
    let _add: fn(
        &mut WebGpuBackend,
        TensorRead<'_>,
        TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> = TensorElementwise::add_read;
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_engine_exposes_engine_operations() {
    use tenferro_gpu::cuda::CudaEngine;
    use tenferro_tensor::{Engine, Tensor, TensorElementwise, TensorRead};

    fn assert_backend_session<S: Engine + ?Sized>() {}

    assert_backend_session::<CudaEngine>();
    let _add: fn(
        &mut CudaEngine,
        TensorRead<'_>,
        TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> = TensorElementwise::add_read;
}

#[cfg(feature = "cuda")]
#[test]
fn execution_session_capability_cannot_project_or_escape_owner_borrow() {
    if std::env::var_os("NEXTEST").is_some() {
        eprintln!("skipping compile-only trybuild contract under nextest");
        return;
    }

    let tests = trybuild::TestCases::new();

    tests.compile_fail("tests/ui/cuda_engine_is_not_clone.rs");
    tests.compile_fail("tests/ui/cuda_removed_runtime_engine_id.rs");
    tests.compile_fail("tests/ui/cuda_removed_runtime_engine_registration_with_id.rs");
    tests.compile_fail("tests/ui/cuda_backend_rejects_bare_integer.rs");
    tests.compile_fail("tests/ui/cuda_pending_upload_moves_the_destination.rs");
    tests.compile_fail("tests/ui/cuda_pending_download_moves_the_source.rs");
    tests.compile_fail("tests/ui/cuda_source_publication_moves_the_value.rs");
    tests.compile_fail("tests/ui/cuda_publication_copy_consumes_the_publication.rs");
}
