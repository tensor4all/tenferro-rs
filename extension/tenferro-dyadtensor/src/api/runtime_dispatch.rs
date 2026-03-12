use tenferro_algebra::Standard;
use tenferro_einsum::EinsumBackend;
use tenferro_linalg::backend::{LinalgCapabilityOp, TensorLinalgBackend, TensorLinalgContextFor};
use tenferro_prims::{
    CpuBackend, CpuContext, CudaBackend, CudaContext, RocmBackend, RocmContext,
    SemiringFastPathDescriptor, TensorSemiringCore, TensorSemiringFastPath,
};

use crate::runtime::RuntimeContext;
use crate::{Error, Result};

use super::contracts::{EinsumRuntimeValue, LinalgRuntimeValue, StandardRuntimeValue};
use super::with_default_runtime;

pub(crate) trait DenseEinsumBackend<T, C>:
    EinsumBackend<Standard<T>>
    + TensorSemiringCore<Standard<T>, Context = C>
    + TensorSemiringFastPath<
        Standard<T>,
        Context = C,
        Plan = <Self as TensorSemiringCore<Standard<T>>>::Plan,
    >
where
    T: StandardRuntimeValue,
{
}

impl<T, C, B> DenseEinsumBackend<T, C> for B
where
    T: StandardRuntimeValue,
    B: EinsumBackend<Standard<T>>
        + TensorSemiringCore<Standard<T>, Context = C>
        + TensorSemiringFastPath<
            Standard<T>,
            Context = C,
            Plan = <B as TensorSemiringCore<Standard<T>>>::Plan,
        >,
{
}

pub(crate) trait RuntimeSlot {
    type Context;
    type SemiringBackend;

    const NAME: &'static str;
}

pub(crate) struct CpuRuntimeSlot;
pub(crate) struct CudaRuntimeSlot;
pub(crate) struct RocmRuntimeSlot;

impl RuntimeSlot for CpuRuntimeSlot {
    type Context = CpuContext;
    type SemiringBackend = CpuBackend;

    const NAME: &'static str = "cpu";
}

impl RuntimeSlot for CudaRuntimeSlot {
    type Context = CudaContext;
    type SemiringBackend = CudaBackend;

    const NAME: &'static str = "cuda";
}

impl RuntimeSlot for RocmRuntimeSlot {
    type Context = RocmContext;
    type SemiringBackend = RocmBackend;

    const NAME: &'static str = "rocm";
}

fn contract_capability_marker() -> SemiringFastPathDescriptor {
    SemiringFastPathDescriptor::Contract {
        modes_a: vec![0],
        modes_b: vec![0],
        modes_c: vec![0],
    }
}

fn ensure_einsum_runtime_capability<T, Slot>(op: &'static str) -> Result<()>
where
    T: EinsumRuntimeValue,
    Slot: RuntimeSlot,
    Slot::SemiringBackend: DenseEinsumBackend<T, Slot::Context>,
{
    if !<Slot::SemiringBackend as TensorSemiringFastPath<Standard<T>>>::has_fast_path(
        contract_capability_marker(),
    ) {
        return Err(unsupported_runtime_capability(op, Slot::NAME));
    }
    Ok(())
}

fn ensure_linalg_runtime_capability<T, Slot>(
    op: &'static str,
    capability: LinalgCapabilityOp,
) -> Result<()>
where
    T: LinalgRuntimeValue,
    Slot: RuntimeSlot,
    Slot::Context: TensorLinalgContextFor<T>,
    <Slot::Context as TensorLinalgContextFor<T>>::Backend:
        TensorLinalgBackend<T, Context = Slot::Context>,
{
    if !<<Slot::Context as TensorLinalgContextFor<T>>::Backend as TensorLinalgBackend<T>>::has_linalg_support(capability)
    {
        return Err(unsupported_runtime_capability(op, Slot::NAME));
    }
    Ok(())
}

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
            ensure_einsum_runtime_capability::<T, CudaRuntimeSlot>(op)?;
            cuda(ctx)
        },
        |ctx| {
            ensure_einsum_runtime_capability::<T, RocmRuntimeSlot>(op)?;
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
            ensure_linalg_runtime_capability::<T, CpuRuntimeSlot>(op, capability)?;
            cpu(ctx)
        },
        |ctx| {
            ensure_linalg_runtime_capability::<T, CudaRuntimeSlot>(op, capability)?;
            cuda(ctx)
        },
        |ctx| {
            ensure_linalg_runtime_capability::<T, RocmRuntimeSlot>(op, capability)?;
            rocm(ctx)
        },
    )
}

macro_rules! runtime_slot_closure {
    ($slot:ty, |$ctx:ident, $backend:ident, $runtime:ident| $body:expr) => {{
        |$ctx| {
            type $backend = <$slot as crate::api::RuntimeSlot>::SemiringBackend;
            let $runtime = <$slot as crate::api::RuntimeSlot>::NAME;
            $body
        }
    }};
}

pub(crate) use runtime_slot_closure;

macro_rules! dispatch_einsum_runtime {
    ($ty:ty, $op:expr, |$ctx:ident, $backend:ident| $body:expr) => {{
        dispatch_einsum_runtime!($ty, $op, |$ctx, $backend, _runtime| $body)
    }};
    ($ty:ty, $op:expr, |$ctx:ident, $backend:ident, $runtime:ident| $body:expr) => {{
        crate::api::with_einsum_runtime::<$ty, _>(
            $op,
            crate::api::runtime_slot_closure!(
                crate::api::CpuRuntimeSlot,
                |$ctx, $backend, $runtime| $body
            ),
            crate::api::runtime_slot_closure!(
                crate::api::CudaRuntimeSlot,
                |$ctx, $backend, $runtime| $body
            ),
            crate::api::runtime_slot_closure!(
                crate::api::RocmRuntimeSlot,
                |$ctx, $backend, $runtime| $body
            ),
        )
    }};
}

pub(crate) use dispatch_einsum_runtime;

macro_rules! dispatch_standard_runtime {
    ($op:expr, |$ctx:ident, $backend:ident, $runtime:ident| $body:expr) => {{
        crate::api::with_runtime(
            crate::api::runtime_slot_closure!(
                crate::api::CpuRuntimeSlot,
                |$ctx, $backend, $runtime| $body
            ),
            crate::api::runtime_slot_closure!(
                crate::api::CudaRuntimeSlot,
                |$ctx, $backend, $runtime| $body
            ),
            crate::api::runtime_slot_closure!(
                crate::api::RocmRuntimeSlot,
                |$ctx, $backend, $runtime| $body
            ),
        )
    }};
}

pub(crate) use dispatch_standard_runtime;
