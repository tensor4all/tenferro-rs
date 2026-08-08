use tenferro_gpu::cuda::{
    download_tensor, upload_tensor, with_cuda_exec_session, CudaBackend, CudaExecSession,
};
use tenferro_runtime::Tensor;
use tenferro_tensor::{BackendSessionHost, TensorRead};

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

pub fn cuda_backend() -> CudaBackend {
    CudaBackend::new(tenferro_gpu::cuda::CudaDeviceId::from_ordinal(0))
        .expect("CUDA device 0 should initialize for an ignored CUDA test")
}

pub fn upload_cuda(runtime: &tenferro_gpu::cuda::CudaRuntime, tensor: &Tensor) -> Tensor {
    upload_tensor(runtime, tensor).expect("explicit CUDA upload")
}

pub fn download_cuda(
    runtime: &tenferro_gpu::cuda::CudaRuntime,
    tensor: &Tensor,
) -> tenferro_tensor::Result<Tensor> {
    runtime.synchronize()?;
    download_tensor(runtime, tensor)
}

pub fn assert_cuda_resident(tensor: &Tensor, runtime_domain: tenferro_tensor::AllocationDomainId) {
    let read = TensorRead::from_tensor(tensor);
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
