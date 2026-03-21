use tenferro_algebra::Standard;
use tenferro_device::{Error, LogicalMemorySpace, Result};
use tenferro_prims::{
    AnalyticPrimsDescriptor, AnalyticUnaryOp, ScalarBinaryOp, ScalarPrimsDescriptor,
    ScalarReductionOp, ScalarUnaryOp, SemiringCoreDescriptor, TensorAnalyticPrims,
    TensorScalarContextFor, TensorScalarPrims, TensorSemiringContextFor, TensorSemiringCore,
};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::LinalgScalar;

pub(crate) fn batched_gemm_with_semiring_core<T, Backend>(
    ctx: &mut <Backend as TensorSemiringCore<Standard<T>>>::Context,
    a: &[T],
    m: usize,
    k: usize,
    b: &[T],
    n: usize,
) -> Result<Vec<T>>
where
    T: LinalgScalar,
    Backend: TensorSemiringCore<Standard<T>>,
{
    let a_shape = [m, k];
    let b_shape = [k, n];
    let c_shape = [m, n];

    let a_strides = [1isize, m as isize];
    let b_strides = [1isize, k as isize];
    let a_tensor = Tensor::from_vec(a.to_vec(), &a_shape, &a_strides, 0)?;
    let b_tensor = Tensor::from_vec(b.to_vec(), &b_shape, &b_strides, 0)?;
    let mut c_tensor = Tensor::zeros(
        &c_shape,
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );

    let desc = SemiringCoreDescriptor::BatchedGemm {
        batch_dims: vec![],
        m,
        n,
        k,
    };
    let plan = <Backend as TensorSemiringCore<Standard<T>>>::plan(
        ctx,
        &desc,
        &[&a_shape, &b_shape, &c_shape],
    )?;
    <Backend as TensorSemiringCore<Standard<T>>>::execute(
        ctx,
        &plan,
        T::one(),
        &[&a_tensor, &b_tensor],
        T::zero(),
        &mut c_tensor,
    )?;

    c_tensor
        .try_into_data_vec()
        .ok_or_else(|| Error::DeviceError("expected owned CPU output tensor".into()))
}

pub(crate) fn batched_gemm_with_semiring_context<T, C>(
    ctx: &mut C,
    a: &[T],
    m: usize,
    k: usize,
    b: &[T],
    n: usize,
) -> Result<Vec<T>>
where
    T: LinalgScalar,
    C: TensorSemiringContextFor<Standard<T>>,
{
    batched_gemm_with_semiring_core::<
        T,
        <C as TensorSemiringContextFor<Standard<T>>>::SemiringBackend,
    >(ctx, a, m, k, b, n)
}

pub(crate) fn full_like_constant<T: LinalgScalar>(
    value: T,
    dims: &[usize],
    memory_space: LogicalMemorySpace,
) -> Result<Tensor<T>> {
    let host = Tensor::from_slice(
        &vec![value; dims.iter().product()],
        dims,
        MemoryOrder::ColumnMajor,
    )?;
    if memory_space == LogicalMemorySpace::MainMemory {
        Ok(host)
    } else {
        host.to_memory_space_async(memory_space)
    }
}

pub(crate) fn scalar_binary_same_shape<T, C>(
    ctx: &mut C,
    lhs: &Tensor<T>,
    rhs: &Tensor<T>,
    op: ScalarBinaryOp,
) -> Result<Tensor<T>>
where
    T: LinalgScalar,
    C: TensorScalarContextFor<Standard<T>>,
{
    let desc = ScalarPrimsDescriptor::PointwiseBinary { op };
    let mut output = Tensor::zeros(
        lhs.dims(),
        lhs.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    );
    let plan = <C::ScalarBackend as TensorScalarPrims<Standard<T>>>::plan(
        ctx,
        &desc,
        &[lhs.dims(), rhs.dims(), output.dims()],
    )?;
    <C::ScalarBackend as TensorScalarPrims<Standard<T>>>::execute(
        ctx,
        &plan,
        T::one(),
        &[lhs, rhs],
        T::zero(),
        &mut output,
    )?;
    Ok(output)
}

pub(crate) fn scalar_unary_same_shape<T, C>(
    ctx: &mut C,
    input: &Tensor<T>,
    op: ScalarUnaryOp,
) -> Result<Tensor<T>>
where
    T: LinalgScalar,
    C: TensorScalarContextFor<Standard<T>>,
{
    let desc = ScalarPrimsDescriptor::PointwiseUnary { op };
    let mut output = Tensor::zeros(
        input.dims(),
        input.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    );
    let plan = <C::ScalarBackend as TensorScalarPrims<Standard<T>>>::plan(
        ctx,
        &desc,
        &[input.dims(), output.dims()],
    )?;
    <C::ScalarBackend as TensorScalarPrims<Standard<T>>>::execute(
        ctx,
        &plan,
        T::one(),
        &[input],
        T::zero(),
        &mut output,
    )?;
    Ok(output)
}

pub(crate) fn analytic_unary_same_shape<T, C>(
    ctx: &mut C,
    input: &Tensor<T>,
    op: AnalyticUnaryOp,
) -> Result<Tensor<T>>
where
    T: LinalgScalar + crate::KernelLinalgScalar,
    C: TensorScalarContextFor<Standard<T>>,
    <C as TensorScalarContextFor<Standard<T>>>::ScalarBackend:
        TensorAnalyticPrims<Standard<T>, Context = C>,
{
    let desc = AnalyticPrimsDescriptor::PointwiseUnary { op };
    let mut output = Tensor::zeros(
        input.dims(),
        input.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    );
    let plan =
        <<C as TensorScalarContextFor<Standard<T>>>::ScalarBackend as TensorAnalyticPrims<
            Standard<T>,
        >>::plan(ctx, &desc, &[input.dims(), output.dims()])?;
    <<C as TensorScalarContextFor<Standard<T>>>::ScalarBackend as TensorAnalyticPrims<
        Standard<T>,
    >>::execute(ctx, &plan, T::one(), &[input], T::zero(), &mut output)?;
    Ok(output)
}

pub(crate) fn scalar_reduce_keep_axes<T, C>(
    ctx: &mut C,
    input: &Tensor<T>,
    kept_axes: &[usize],
    op: ScalarReductionOp,
) -> Result<Tensor<T>>
where
    T: LinalgScalar,
    C: TensorScalarContextFor<Standard<T>>,
{
    let modes_a: Vec<u32> = (0..input.ndim())
        .map(|axis| {
            u32::try_from(axis)
                .map_err(|_| Error::InvalidArgument(format!("axis {axis} exceeds u32 range")))
        })
        .collect::<Result<_>>()?;
    let modes_c: Vec<u32> = kept_axes
        .iter()
        .map(|&axis| {
            u32::try_from(axis)
                .map_err(|_| Error::InvalidArgument(format!("axis {axis} exceeds u32 range")))
        })
        .collect::<Result<_>>()?;
    let output_dims: Vec<usize> = kept_axes.iter().map(|&axis| input.dims()[axis]).collect();
    let desc = ScalarPrimsDescriptor::Reduction {
        modes_a,
        modes_c,
        op,
    };
    let mut output = Tensor::zeros(
        &output_dims,
        input.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    );
    let plan = <C::ScalarBackend as TensorScalarPrims<Standard<T>>>::plan(
        ctx,
        &desc,
        &[input.dims(), output.dims()],
    )?;
    <C::ScalarBackend as TensorScalarPrims<Standard<T>>>::execute(
        ctx,
        &plan,
        T::one(),
        &[input],
        T::zero(),
        &mut output,
    )?;
    Ok(output)
}

pub(crate) fn scalar_sum_keep_axes<T, C>(
    ctx: &mut C,
    input: &Tensor<T>,
    kept_axes: &[usize],
) -> Result<Tensor<T>>
where
    T: LinalgScalar,
    C: TensorScalarContextFor<Standard<T>>,
{
    scalar_reduce_keep_axes(ctx, input, kept_axes, ScalarReductionOp::Sum)
}

#[cfg(test)]
mod tests;
