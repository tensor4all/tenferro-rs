use std::convert::TryFrom;

use tenferro_algebra::Standard;
use tenferro_prims::{
    AnalyticBinaryOp, AnalyticPrimsDescriptor, AnalyticReductionOp, AnalyticUnaryOp,
    ScalarBinaryOp, ScalarPrimsDescriptor, ScalarReductionOp, ScalarUnaryOp, TensorAnalyticPrims,
    TensorScalarPrims,
};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{Error, Result};

use crate::runtime::contracts::{AnalyticRuntimeValue, ScalarRuntimeValue, StandardRuntimeValue};
use crate::runtime::dispatch::dispatch_standard_runtime;

fn run_scalar_unary_backend<B, T>(
    ctx: &mut <B as TensorScalarPrims<Standard<T>>>::Context,
    runtime: &'static str,
    op_name: &'static str,
    op: ScalarUnaryOp,
    input: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: StandardRuntimeValue,
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
    )?;
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
    T: StandardRuntimeValue,
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
    )?;
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
    T: StandardRuntimeValue,
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
    let mut output = Tensor::zeros(&[], input.logical_memory_space(), MemoryOrder::ColumnMajor)?;
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
    T: StandardRuntimeValue,
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
    )?;
    let plan = B::plan(ctx, &desc, &[input.dims(), output.dims()]).map_err(Error::from)?;
    B::execute(ctx, &plan, T::one(), &[input], T::zero(), &mut output).map_err(Error::from)?;
    Ok(output)
}

fn run_analytic_binary_backend<B, T>(
    ctx: &mut <B as TensorAnalyticPrims<Standard<T>>>::Context,
    runtime: &'static str,
    op_name: &'static str,
    op: AnalyticBinaryOp,
    lhs: &Tensor<T>,
    rhs: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: StandardRuntimeValue,
    B: TensorAnalyticPrims<Standard<T>>,
{
    let desc = AnalyticPrimsDescriptor::PointwiseBinary { op };
    if !B::has_analytic_support(desc.clone()) {
        return Err(Error::UnsupportedRuntimeOp {
            op: op_name,
            runtime,
        });
    }
    let mut output = Tensor::zeros(
        lhs.dims(),
        lhs.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    )?;
    let plan =
        B::plan(ctx, &desc, &[lhs.dims(), rhs.dims(), output.dims()]).map_err(Error::from)?;
    B::execute(ctx, &plan, T::one(), &[lhs, rhs], T::zero(), &mut output).map_err(Error::from)?;
    Ok(output)
}

fn run_analytic_full_reduction_backend<B, T>(
    ctx: &mut <B as TensorAnalyticPrims<Standard<T>>>::Context,
    runtime: &'static str,
    op_name: &'static str,
    op: AnalyticReductionOp,
    input: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: StandardRuntimeValue,
    B: TensorAnalyticPrims<Standard<T>>,
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
    let desc = AnalyticPrimsDescriptor::Reduction {
        modes_a,
        modes_c: Vec::new(),
        op,
    };
    if !B::has_analytic_support(desc.clone()) {
        return Err(Error::UnsupportedRuntimeOp {
            op: op_name,
            runtime,
        });
    }
    let mut output = Tensor::zeros(&[], input.logical_memory_space(), MemoryOrder::ColumnMajor)?;
    let plan = B::plan(ctx, &desc, &[input.dims(), output.dims()]).map_err(Error::from)?;
    B::execute(ctx, &plan, T::one(), &[input], T::zero(), &mut output).map_err(Error::from)?;
    Ok(output)
}

pub fn scalar_unary_primal<T>(
    op_name: &'static str,
    op: ScalarUnaryOp,
    input: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: ScalarRuntimeValue,
{
    dispatch_standard_runtime!(op_name, |ctx, Backend, runtime| {
        run_scalar_unary_backend::<Backend, T>(ctx, runtime, op_name, op, input)
    })
}

pub fn scalar_binary_primal<T>(
    op_name: &'static str,
    op: ScalarBinaryOp,
    lhs: &Tensor<T>,
    rhs: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: ScalarRuntimeValue,
{
    dispatch_standard_runtime!(op_name, |ctx, Backend, runtime| {
        run_scalar_binary_backend::<Backend, T>(ctx, runtime, op_name, op, lhs, rhs)
    })
}

pub fn scalar_full_reduction_primal<T>(
    op_name: &'static str,
    op: ScalarReductionOp,
    input: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: ScalarRuntimeValue,
{
    dispatch_standard_runtime!(op_name, |ctx, Backend, runtime| {
        run_scalar_full_reduction_backend::<Backend, T>(ctx, runtime, op_name, op, input)
    })
}

pub fn analytic_unary_primal<T>(
    op_name: &'static str,
    op: AnalyticUnaryOp,
    input: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: AnalyticRuntimeValue,
{
    dispatch_standard_runtime!(op_name, |ctx, Backend, runtime| {
        run_analytic_unary_backend::<Backend, T>(ctx, runtime, op_name, op, input)
    })
}

pub fn analytic_binary_primal<T>(
    op_name: &'static str,
    op: AnalyticBinaryOp,
    lhs: &Tensor<T>,
    rhs: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: AnalyticRuntimeValue,
{
    dispatch_standard_runtime!(op_name, |ctx, Backend, runtime| {
        run_analytic_binary_backend::<Backend, T>(ctx, runtime, op_name, op, lhs, rhs)
    })
}

pub fn analytic_full_reduction_primal<T>(
    op_name: &'static str,
    op: AnalyticReductionOp,
    input: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: AnalyticRuntimeValue,
{
    dispatch_standard_runtime!(op_name, |ctx, Backend, runtime| {
        run_analytic_full_reduction_backend::<Backend, T>(ctx, runtime, op_name, op, input)
    })
}
