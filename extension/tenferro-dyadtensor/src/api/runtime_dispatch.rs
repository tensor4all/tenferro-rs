use tenferro_algebra::Standard;
use tenferro_linalg::backend::{LinalgCapabilityOp, TensorLinalgBackend, TensorLinalgContextFor};
use tenferro_prims::CpuContext;
use tenferro_prims::{
    CudaBackend, CudaContext, RocmBackend, RocmContext, SemiringFastPathDescriptor,
    TensorSemiringFastPath,
};

use crate::runtime::RuntimeContext;
use crate::{Error, Result};

use super::contracts::{EinsumRuntimeValue, LinalgRuntimeValue};
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

pub(crate) fn with_einsum_runtime<T: EinsumRuntimeValue, R>(
    op: &'static str,
    cpu: impl FnOnce(&mut CpuContext) -> Result<R>,
    cuda: impl FnOnce(&mut CudaContext) -> Result<R>,
    rocm: impl FnOnce(&mut RocmContext) -> Result<R>,
) -> Result<R> {
    with_runtime(
        cpu,
        |ctx| {
            if !<CudaBackend as TensorSemiringFastPath<Standard<T>>>::has_fast_path(
                SemiringFastPathDescriptor::Contract {
                    modes_a: Vec::new(),
                    modes_b: Vec::new(),
                    modes_c: Vec::new(),
                },
            ) {
                return Err(unsupported_runtime_capability(op, "cuda"));
            }
            cuda(ctx)
        },
        |ctx| {
            if !<RocmBackend as TensorSemiringFastPath<Standard<T>>>::has_fast_path(
                SemiringFastPathDescriptor::Contract {
                    modes_a: Vec::new(),
                    modes_b: Vec::new(),
                    modes_c: Vec::new(),
                },
            ) {
                return Err(unsupported_runtime_capability(op, "rocm"));
            }
            rocm(ctx)
        },
    )
}

pub(crate) fn with_linalg_runtime<T: LinalgRuntimeValue, R>(
    op: &'static str,
    capability: LinalgCapabilityOp,
    cpu: impl FnOnce(&mut CpuContext) -> Result<R>,
    cuda: impl FnOnce(&mut CudaContext) -> Result<R>,
    rocm: impl FnOnce(&mut RocmContext) -> Result<R>,
) -> Result<R> {
    with_runtime(
        |ctx| {
            if !<<CpuContext as TensorLinalgContextFor<T>>::Backend as TensorLinalgBackend<T>>::has_linalg_support(capability)
            {
                return Err(unsupported_runtime_capability(op, "cpu"));
            }
            cpu(ctx)
        },
        |ctx| {
            if !<<CudaContext as TensorLinalgContextFor<T>>::Backend as TensorLinalgBackend<T>>::has_linalg_support(capability)
            {
                return Err(unsupported_runtime_capability(op, "cuda"));
            }
            cuda(ctx)
        },
        |ctx| {
            if !<<RocmContext as TensorLinalgContextFor<T>>::Backend as TensorLinalgBackend<T>>::has_linalg_support(capability)
            {
                return Err(unsupported_runtime_capability(op, "rocm"));
            }
            rocm(ctx)
        },
    )
}

macro_rules! dispatch_einsum_runtime {
    ($ty:ty, $op:expr, |$ctx:ident, $backend:ident| $body:expr) => {{
        dispatch_einsum_runtime!($ty, $op, |$ctx, $backend, _runtime| $body)
    }};
    ($ty:ty, $op:expr, |$ctx:ident, $backend:ident, $runtime:ident| $body:expr) => {{
        crate::api::with_einsum_runtime::<$ty, _>(
            $op,
            |$ctx| {
                type $backend = tenferro_prims::CpuBackend;
                let $runtime = "cpu";
                $body
            },
            |$ctx| {
                type $backend = tenferro_prims::CudaBackend;
                let $runtime = "cuda";
                $body
            },
            |$ctx| {
                type $backend = tenferro_prims::RocmBackend;
                let $runtime = "rocm";
                $body
            },
        )
    }};
}

pub(crate) use dispatch_einsum_runtime;

macro_rules! dispatch_standard_runtime {
    ($op:expr, |$ctx:ident, $backend:ident, $runtime:ident| $body:expr) => {{
        crate::api::with_runtime(
            |$ctx| {
                type $backend = tenferro_prims::CpuBackend;
                let $runtime = "cpu";
                $body
            },
            |$ctx| {
                type $backend = tenferro_prims::CudaBackend;
                let $runtime = "cuda";
                $body
            },
            |$ctx| {
                type $backend = tenferro_prims::RocmBackend;
                let $runtime = "rocm";
                $body
            },
        )
    }};
}

pub(crate) use dispatch_standard_runtime;
