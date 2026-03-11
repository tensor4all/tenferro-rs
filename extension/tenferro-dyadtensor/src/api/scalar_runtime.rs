use std::convert::TryFrom;

use tenferro_algebra::{HasAlgebra, Scalar, Standard};
use tenferro_prims::{
    AnalyticPrimsDescriptor, AnalyticUnaryOp, CpuBackend, CudaBackend, RocmBackend, ScalarBinaryOp,
    ScalarPrimsDescriptor, ScalarReductionOp, ScalarUnaryOp, TensorAnalyticPrims, TensorPrims,
    TensorScalarPrims,
};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{AdTensor, Error, Result};

use super::runtime::{dense_input_snapshot_in_ctx, zero_like};
use super::{unsupported_runtime_capability, with_runtime};

pub(crate) fn dense_input_snapshot_in_runtime<T>(
    op_name: &'static str,
    input: &AdTensor<T>,
    needs_tangent: bool,
) -> Result<(Tensor<T>, Option<Tensor<T>>)>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
{
    with_runtime(
        |ctx| dense_input_snapshot_in_ctx(ctx, input, needs_tangent),
        |_ctx| {
            if !input.is_dense() {
                return Err(unsupported_runtime_capability(op_name, "cuda"));
            }
            let primal = input.primal().clone().contiguous(MemoryOrder::ColumnMajor);
            let tangent = if needs_tangent {
                Some(
                    input
                        .tangent()
                        .cloned()
                        .unwrap_or_else(|| zero_like(&primal))
                        .contiguous(MemoryOrder::ColumnMajor),
                )
            } else {
                None
            };
            Ok((primal, tangent))
        },
        |_ctx| {
            if !input.is_dense() {
                return Err(unsupported_runtime_capability(op_name, "rocm"));
            }
            let primal = input.primal().clone().contiguous(MemoryOrder::ColumnMajor);
            let tangent = if needs_tangent {
                Some(
                    input
                        .tangent()
                        .cloned()
                        .unwrap_or_else(|| zero_like(&primal))
                        .contiguous(MemoryOrder::ColumnMajor),
                )
            } else {
                None
            };
            Ok((primal, tangent))
        },
    )
}

fn run_scalar_unary_backend<B, T>(
    ctx: &mut <B as TensorScalarPrims<Standard<T>>>::Context,
    runtime: &'static str,
    op_name: &'static str,
    op: ScalarUnaryOp,
    input: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    B: TensorScalarPrims<Standard<T>>,
{
    let desc = ScalarPrimsDescriptor::PointwiseUnary { op };
    if !B::has_scalar_support(desc.clone()) {
        return Err(Error::UnsupportedRuntimeOp {
            op: op_name,
            runtime,
        });
    }
    let mut output = Tensor::zeros(
        input.dims(),
        input.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    );
    let plan = B::plan(ctx, &desc, &[input.dims(), output.dims()]).map_err(Error::from)?;
    B::execute(ctx, &plan, T::one(), &[input], T::zero(), &mut output).map_err(Error::from)?;
    Ok(output)
}

fn run_scalar_binary_backend<B, T>(
    ctx: &mut <B as TensorScalarPrims<Standard<T>>>::Context,
    runtime: &'static str,
    op_name: &'static str,
    op: ScalarBinaryOp,
    lhs: &Tensor<T>,
    rhs: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    B: TensorScalarPrims<Standard<T>>,
{
    let desc = ScalarPrimsDescriptor::PointwiseBinary { op };
    if !B::has_scalar_support(desc.clone()) {
        return Err(Error::UnsupportedRuntimeOp {
            op: op_name,
            runtime,
        });
    }
    let mut output = Tensor::zeros(
        lhs.dims(),
        lhs.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    );
    let plan =
        B::plan(ctx, &desc, &[lhs.dims(), rhs.dims(), output.dims()]).map_err(Error::from)?;
    B::execute(ctx, &plan, T::one(), &[lhs, rhs], T::zero(), &mut output).map_err(Error::from)?;
    Ok(output)
}

fn run_scalar_full_reduction_backend<B, T>(
    ctx: &mut <B as TensorScalarPrims<Standard<T>>>::Context,
    runtime: &'static str,
    op_name: &'static str,
    op: ScalarReductionOp,
    input: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    B: TensorScalarPrims<Standard<T>>,
{
    let modes_a: Vec<u32> = (0..input.dims().len())
        .map(|idx| {
            u32::try_from(idx).map_err(|_| Error::InvalidAdTensor {
                message: format!(
                    "{op_name} rank {} exceeds u32 label space",
                    input.dims().len()
                ),
            })
        })
        .collect::<Result<_>>()?;
    let desc = ScalarPrimsDescriptor::Reduction {
        modes_a,
        modes_c: Vec::new(),
        op,
    };
    if !B::has_scalar_support(desc.clone()) {
        return Err(Error::UnsupportedRuntimeOp {
            op: op_name,
            runtime,
        });
    }
    let mut output = Tensor::zeros(&[], input.logical_memory_space(), MemoryOrder::ColumnMajor);
    let plan = B::plan(ctx, &desc, &[input.dims(), output.dims()]).map_err(Error::from)?;
    B::execute(ctx, &plan, T::one(), &[input], T::zero(), &mut output).map_err(Error::from)?;
    Ok(output)
}

fn run_analytic_unary_backend<B, T>(
    ctx: &mut <B as TensorAnalyticPrims<Standard<T>>>::Context,
    runtime: &'static str,
    op_name: &'static str,
    op: AnalyticUnaryOp,
    input: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    B: TensorAnalyticPrims<Standard<T>>,
{
    let desc = AnalyticPrimsDescriptor::PointwiseUnary { op };
    if !B::has_analytic_support(desc.clone()) {
        return Err(Error::UnsupportedRuntimeOp {
            op: op_name,
            runtime,
        });
    }
    let mut output = Tensor::zeros(
        input.dims(),
        input.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    );
    let plan = B::plan(ctx, &desc, &[input.dims(), output.dims()]).map_err(Error::from)?;
    B::execute(ctx, &plan, T::one(), &[input], T::zero(), &mut output).map_err(Error::from)?;
    Ok(output)
}

pub(crate) fn scalar_unary_primal<T>(
    op_name: &'static str,
    op: ScalarUnaryOp,
    input: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CudaBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    RocmBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
{
    with_runtime(
        |ctx| run_scalar_unary_backend::<CpuBackend, T>(ctx, "cpu", op_name, op, input),
        |ctx| run_scalar_unary_backend::<CudaBackend, T>(ctx, "cuda", op_name, op, input),
        |ctx| run_scalar_unary_backend::<RocmBackend, T>(ctx, "rocm", op_name, op, input),
    )
}

pub(crate) fn scalar_binary_primal<T>(
    op_name: &'static str,
    op: ScalarBinaryOp,
    lhs: &Tensor<T>,
    rhs: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CudaBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    RocmBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
{
    with_runtime(
        |ctx| run_scalar_binary_backend::<CpuBackend, T>(ctx, "cpu", op_name, op, lhs, rhs),
        |ctx| run_scalar_binary_backend::<CudaBackend, T>(ctx, "cuda", op_name, op, lhs, rhs),
        |ctx| run_scalar_binary_backend::<RocmBackend, T>(ctx, "rocm", op_name, op, lhs, rhs),
    )
}

pub(crate) fn scalar_full_reduction_primal<T>(
    op_name: &'static str,
    op: ScalarReductionOp,
    input: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CudaBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    RocmBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
{
    with_runtime(
        |ctx| run_scalar_full_reduction_backend::<CpuBackend, T>(ctx, "cpu", op_name, op, input),
        |ctx| run_scalar_full_reduction_backend::<CudaBackend, T>(ctx, "cuda", op_name, op, input),
        |ctx| run_scalar_full_reduction_backend::<RocmBackend, T>(ctx, "rocm", op_name, op, input),
    )
}

pub(crate) fn analytic_unary_primal<T>(
    op_name: &'static str,
    op: AnalyticUnaryOp,
    input: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CudaBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    RocmBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
{
    with_runtime(
        |ctx| run_analytic_unary_backend::<CpuBackend, T>(ctx, "cpu", op_name, op, input),
        |ctx| run_analytic_unary_backend::<CudaBackend, T>(ctx, "cuda", op_name, op, input),
        |ctx| run_analytic_unary_backend::<RocmBackend, T>(ctx, "rocm", op_name, op, input),
    )
}
