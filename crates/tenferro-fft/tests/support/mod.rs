use tenferro_cpu::CpuBackend;
#[cfg(feature = "cuda")]
use tenferro_gpu::cuda::{
    download_tensor, upload_tensor, with_cuda_exec_session, CudaBackend, CudaExecSession,
};
use tenferro_runtime::{CompiledGraph, Runtime, RuntimeConfigError};
#[cfg(feature = "cuda")]
use tenferro_tensor::BackendSessionHost;
use tenferro_tensor::Tensor;

/// Build a CPU runtime with the FFT extension module installed.
///
/// # Errors
///
/// Returns [`RuntimeConfigError`] when CPU engine registration, FFT module
/// construction, module installation, or final runtime construction fails.
pub fn cpu_runtime_with_fft(backend: &CpuBackend) -> Result<Runtime, RuntimeConfigError> {
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(backend)?)?;
    builder.install_extension_module(tenferro_fft::extension_module::<CpuBackend>(
        tenferro_cpu::runtime_engine_id()?,
    )?)?;
    builder.build()
}

/// Execute a compiled FFT test program and return its single output.
///
/// # Errors
///
/// Returns [`tenferro_runtime::Error::RuntimeState`] when constructing the test
/// runtime fails or when the compiled program returns any output count other
/// than one. Propagates typed input binding, validation, and backend execution
/// errors from
/// [`Runtime::run_compiled`].
pub fn run_one(program: &CompiledGraph, inputs: &[&Tensor]) -> tenferro_runtime::Result<Tensor> {
    let backend = CpuBackend::new();
    let runtime = cpu_runtime_with_fft(&backend).map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "fft_test_runtime",
            tenferro_runtime::ErrorPhase::Execution,
            source,
        )
    })?;
    let mut outputs = runtime.run_compiled(program, inputs)?;
    let actual = outputs.len();
    if actual != 1 {
        return Err(tenferro_runtime::Error::runtime_state(
            "fft_test_runtime",
            tenferro_runtime::ErrorPhase::Execution,
            format!("expected one runtime output, got {actual}"),
        ));
    }
    outputs.pop().ok_or_else(|| {
        tenferro_runtime::Error::runtime_state(
            "fft_test_runtime",
            tenferro_runtime::ErrorPhase::Execution,
            "runtime returned no output after successful output-count validation",
        )
    })
}

#[cfg(feature = "cuda")]
pub fn with_cuda_fft_session<R>(
    backend: &mut CudaBackend,
    f: impl for<'a> FnOnce(&'a mut CudaExecSession<'a>) -> R + Send,
) -> R
where
    R: Send,
{
    backend.with_backend_session(|session| {
        with_cuda_exec_session(session, f).expect("CUDA backend session should be available")
    })
}

#[cfg(feature = "cuda")]
pub fn cuda_backend() -> CudaBackend {
    CudaBackend::new(tenferro_gpu::cuda::CudaDeviceId::from_ordinal(0))
        .expect("CUDA device 0 should initialize for an ignored CUDA test")
}

#[cfg(feature = "cuda")]
pub fn upload_cuda(runtime: &tenferro_gpu::cuda::CudaRuntime, tensor: &Tensor) -> Tensor {
    upload_tensor(runtime, tensor).expect("explicit CUDA upload")
}

#[cfg(feature = "cuda")]
pub fn download_cuda(
    runtime: &tenferro_gpu::cuda::CudaRuntime,
    tensor: &Tensor,
) -> tenferro_tensor::Result<Tensor> {
    runtime.synchronize()?;
    download_tensor(runtime, tensor)
}

#[cfg(feature = "cuda")]
pub fn assert_cuda_resident(tensor: &Tensor, runtime_domain: tenferro_tensor::AllocationDomainId) {
    let read = tenferro_tensor::TensorRead::from_tensor(tensor);
    assert_eq!(read.backend_family(), Some("cuda"));
    assert_eq!(
        read.placement().memory_kind,
        tenferro_tensor::MemoryKind::Device
    );
    assert_eq!(
        read.placement().device,
        Some(tenferro_tensor::DeviceId {
            kind: tenferro_tensor::DeviceKind::Gpu(tenferro_tensor::GpuBackendKind::Cuda),
            ordinal: 0,
        })
    );
    assert_eq!(read.allocation_domain(), Some(runtime_domain));
}
