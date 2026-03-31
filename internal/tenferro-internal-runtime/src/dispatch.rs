use tenferro_algebra::Standard;
use tenferro_einsum::EinsumBackend;
use tenferro_internal_error::{Error, Result};
use tenferro_linalg::backend::{LinalgCapabilityOp, TensorLinalgBackend, TensorLinalgContextFor};
use tenferro_linalg::LiftPermutationMatrixTensor;
use tenferro_prims::{
    CpuBackend, CpuContext, CudaBackend, CudaContext, RocmBackend, RocmContext,
    SemiringFastPathDescriptor, TensorSemiringCore, TensorSemiringFastPath,
};

use crate::contracts::{EinsumRuntimeValue, LinalgRuntimeValue, StandardRuntimeValue};
use crate::{with_default_runtime, RuntimeContext};

pub trait DenseEinsumBackend<T, C>:
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

pub trait RuntimeSlot {
    type Context;
    type SemiringBackend;

    const NAME: &'static str;
}

pub struct CpuRuntimeSlot;
pub struct CudaRuntimeSlot;
pub struct RocmRuntimeSlot;

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

pub trait ScaledRealLinalgDispatchValue:
    crate::contracts::RealLinalgRuntimeValue
    + tenferro_linalg::ScaleTensorByRealSameShape<CpuContext>
    + tenferro_linalg::ScaleTensorByRealSameShape<CudaContext>
    + tenferro_linalg::ScaleTensorByRealSameShape<RocmContext>
{
}

impl<T> ScaledRealLinalgDispatchValue for T where
    T: crate::contracts::RealLinalgRuntimeValue
        + tenferro_linalg::ScaleTensorByRealSameShape<CpuContext>
        + tenferro_linalg::ScaleTensorByRealSameShape<CudaContext>
        + tenferro_linalg::ScaleTensorByRealSameShape<RocmContext>
{
}

pub trait NormLinalgDispatchValue:
    crate::contracts::RealLinalgRuntimeValue
    + tenferro_linalg::NormPrimal<CpuContext>
    + tenferro_linalg::NormPrimal<CudaContext>
    + tenferro_linalg::NormPrimal<RocmContext>
{
}

impl<T> NormLinalgDispatchValue for T where
    T: crate::contracts::RealLinalgRuntimeValue
        + tenferro_linalg::NormPrimal<CpuContext>
        + tenferro_linalg::NormPrimal<CudaContext>
        + tenferro_linalg::NormPrimal<RocmContext>
{
}

pub trait SlogdetLinalgDispatchValue:
    crate::contracts::RealLinalgRuntimeValue
    + tenferro_linalg::SlogdetDispatch<CpuContext>
    + tenferro_linalg::SlogdetDispatch<CudaContext>
    + tenferro_linalg::SlogdetDispatch<RocmContext>
{
}

impl<T> SlogdetLinalgDispatchValue for T where
    T: crate::contracts::RealLinalgRuntimeValue
        + tenferro_linalg::SlogdetDispatch<CpuContext>
        + tenferro_linalg::SlogdetDispatch<CudaContext>
        + tenferro_linalg::SlogdetDispatch<RocmContext>
{
}

pub trait MatrixExpLinalgDispatchValue:
    crate::contracts::LinalgRuntimeValue
    + tenferro_linalg::ScaleTensorByRealSameShape<CpuContext>
    + tenferro_linalg::ScaleTensorByRealSameShape<CudaContext>
    + tenferro_linalg::ScaleTensorByRealSameShape<RocmContext>
    + tenferro_linalg::MatrixExpAbsTensor<CpuContext>
    + tenferro_linalg::MatrixExpAbsTensor<CudaContext>
    + tenferro_linalg::MatrixExpAbsTensor<RocmContext>
{
}

impl<T> MatrixExpLinalgDispatchValue for T where
    T: crate::contracts::LinalgRuntimeValue
        + tenferro_linalg::ScaleTensorByRealSameShape<CpuContext>
        + tenferro_linalg::ScaleTensorByRealSameShape<CudaContext>
        + tenferro_linalg::ScaleTensorByRealSameShape<RocmContext>
        + tenferro_linalg::MatrixExpAbsTensor<CpuContext>
        + tenferro_linalg::MatrixExpAbsTensor<CudaContext>
        + tenferro_linalg::MatrixExpAbsTensor<RocmContext>
{
}

pub trait RealMatrixExpLinalgDispatchValue:
    crate::contracts::RealLinalgRuntimeValue + MatrixExpLinalgDispatchValue
{
}

impl<T> RealMatrixExpLinalgDispatchValue for T where
    T: crate::contracts::RealLinalgRuntimeValue + MatrixExpLinalgDispatchValue
{
}

pub trait LuLinalgDispatchValue:
    crate::contracts::LinalgRuntimeValue
    + LiftPermutationMatrixTensor<CpuContext>
    + LiftPermutationMatrixTensor<CudaContext>
    + LiftPermutationMatrixTensor<RocmContext>
{
}

impl<T> LuLinalgDispatchValue for T where
    T: crate::contracts::LinalgRuntimeValue
        + LiftPermutationMatrixTensor<CpuContext>
        + LiftPermutationMatrixTensor<CudaContext>
        + LiftPermutationMatrixTensor<RocmContext>
{
}

pub trait RealLuLinalgDispatchValue:
    crate::contracts::RealLinalgRuntimeValue
    + LiftPermutationMatrixTensor<CpuContext>
    + LiftPermutationMatrixTensor<CudaContext>
    + LiftPermutationMatrixTensor<RocmContext>
{
}

impl<T> RealLuLinalgDispatchValue for T where
    T: crate::contracts::RealLinalgRuntimeValue
        + LiftPermutationMatrixTensor<CpuContext>
        + LiftPermutationMatrixTensor<CudaContext>
        + LiftPermutationMatrixTensor<RocmContext>
{
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

pub fn unsupported_runtime_capability(op: &'static str, runtime: &'static str) -> Error {
    Error::UnsupportedRuntimeOp { op, runtime }
}

pub fn with_runtime<R>(
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

pub fn with_einsum_runtime<T: EinsumRuntimeValue, R>(
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

pub fn with_linalg_runtime<T: LinalgRuntimeValue, R>(
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
