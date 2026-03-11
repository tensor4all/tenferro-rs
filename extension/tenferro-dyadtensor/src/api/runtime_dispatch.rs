use tenferro_algebra::{Scalar, Standard};
use tenferro_prims::{
    CpuBackend, CpuContext, CudaBackend, CudaContext, Extension, RocmBackend, RocmContext,
    TensorPrims,
};

use crate::runtime::RuntimeContext;
use crate::{Error, Result};

use super::with_default_runtime;

pub(crate) fn unsupported_runtime_capability(op: &'static str, runtime: &'static str) -> Error {
    Error::UnsupportedRuntimeOp { op, runtime }
}

pub(crate) fn with_runtime<R>(
    cpu: impl FnOnce(&mut CpuContext) -> Result<R>,
    cuda: impl FnOnce(&mut CudaContext) -> Result<R>,
    rocm: impl FnOnce(&mut RocmContext) -> Result<R>,
) -> Result<R> {
    with_default_runtime(|runtime| match runtime {
        RuntimeContext::Cpu(ctx) => cpu(ctx),
        RuntimeContext::Cuda(ctx) => cuda(ctx),
        RuntimeContext::Rocm(ctx) => rocm(ctx),
    })
}

pub(crate) fn with_runtime_cpu_only<R>(
    op: &'static str,
    cpu: impl FnOnce(&mut CpuContext) -> Result<R>,
) -> Result<R> {
    with_runtime(
        cpu,
        |_ctx| Err(unsupported_runtime_capability(op, "cuda")),
        |_ctx| Err(unsupported_runtime_capability(op, "rocm")),
    )
}

pub(crate) fn with_einsum_runtime<T: Scalar, R>(
    op: &'static str,
    cpu: impl FnOnce(&mut CpuContext) -> Result<R>,
    cuda: impl FnOnce(&mut CudaContext) -> Result<R>,
    rocm: impl FnOnce(&mut RocmContext) -> Result<R>,
) -> Result<R>
where
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
    CudaBackend: TensorPrims<Standard<T>, Context = CudaContext>,
    RocmBackend: TensorPrims<Standard<T>, Context = RocmContext>,
{
    with_runtime(
        cpu,
        |ctx| {
            if !<CudaBackend as TensorPrims<Standard<T>>>::has_extension_for(Extension::Contract) {
                return Err(unsupported_runtime_capability(op, "cuda"));
            }
            cuda(ctx)
        },
        |ctx| {
            if !<RocmBackend as TensorPrims<Standard<T>>>::has_extension_for(Extension::Contract) {
                return Err(unsupported_runtime_capability(op, "rocm"));
            }
            rocm(ctx)
        },
    )
}
