use tenferro_gpu::cuda_next::{download_tensor, upload_tensor, CudaDeviceId, CudaEngine};
use tenferro_runtime::{ExecutionEngine, ExtensionCacheStore, Tensor};
use tenferro_tensor::TensorRead;

/// Open the CUDA engine every ignored CUDA FFT test runs on.
///
/// The caller holds the engine for the whole test: cuFFT plans live in the
/// engine's own extension caches and are released with it (#2206).
pub fn cuda_engine() -> CudaEngine {
    CudaEngine::open(CudaDeviceId::from_ordinal(0))
        .expect("CUDA device 0 should initialize for an ignored CUDA test")
}

pub fn upload_cuda(engine: &CudaEngine, tensor: &Tensor) -> Tensor {
    upload_tensor(engine.runtime(), tensor).expect("explicit CUDA upload")
}

/// Download a CUDA tensor; `download_tensor` performs the one required stream
/// synchronization before reading the payload.
///
/// # Errors
///
/// Returns [`tenferro_gpu::Error::RuntimeState`] for host-backed or foreign
/// tensors, [`tenferro_gpu::Error::BackendSource`] when synchronization or
/// readback fails, or a typed validation error when device data cannot be
/// decoded.
pub fn download_cuda(engine: &CudaEngine, tensor: &Tensor) -> tenferro_tensor::Result<Tensor> {
    download_tensor(engine.runtime(), tensor)
}

/// Run `f` on the engine's extension-cache store and give the store back.
///
/// The engine owns its device-bound plans, so a caller-held `FftExecutor` cache
/// stays empty for CUDA and these entries are the observable plan cache.
pub fn with_engine_caches<R>(
    engine: &mut CudaEngine,
    f: impl FnOnce(&mut ExtensionCacheStore) -> R,
) -> R {
    let mut caches = engine.take_extension_caches();
    let output = f(&mut caches);
    engine.restore_extension_caches(caches);
    output
}

/// Aggregate stats of every entry the engine retains.
pub fn engine_cache_stats(engine: &mut CudaEngine) -> tenferro_tensor::CacheStats {
    with_engine_caches(engine, |caches| {
        caches.stats(tenferro_runtime::ExtensionCacheSelector::All)
    })
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
