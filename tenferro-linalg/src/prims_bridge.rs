use num_complex::{Complex32, Complex64, ComplexFloat};
use num_traits::{One, Zero};
use tenferro_algebra::Standard;
use tenferro_device::{Error, LogicalMemorySpace, Result};
use tenferro_prims::{
    AnalyticBinaryOp, AnalyticPrimsDescriptor, AnalyticUnaryOp, ComplexRealPrimsDescriptor,
    ComplexRealUnaryOp, ComplexScalePrimsDescriptor, ScalarBinaryOp, ScalarPrimsDescriptor,
    ScalarReductionOp, ScalarTernaryOp, ScalarUnaryOp, SemiringCoreDescriptor, TensorAnalyticPrims,
    TensorComplexRealContextFor, TensorComplexRealPrims, TensorComplexScaleContextFor,
    TensorComplexScalePrims, TensorResolveConjContextFor, TensorScalarContextFor,
    TensorScalarPrims, TensorSemiringContextFor, TensorSemiringCore,
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
    )?;

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

pub(crate) fn batched_gemm_with_semiring_tensors<T, C>(
    ctx: &mut C,
    lhs: &Tensor<T>,
    rhs: &Tensor<T>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Tensor<T>>
where
    T: LinalgScalar,
    C: TensorSemiringContextFor<Standard<T>>,
{
    let lhs = lhs.contiguous(MemoryOrder::ColumnMajor);
    let rhs = rhs.contiguous(MemoryOrder::ColumnMajor);

    if lhs.ndim() != rhs.ndim() || lhs.ndim() < 2 {
        return Err(Error::InvalidArgument(format!(
            "batched_gemm expects tensors with matching rank >= 2, got lhs {:?}, rhs {:?}",
            lhs.dims(),
            rhs.dims()
        )));
    }
    if lhs.dims()[0] != m || lhs.dims()[1] != k {
        return Err(Error::ShapeMismatch {
            expected: vec![m, k],
            got: lhs.dims()[..2].to_vec(),
        });
    }
    if rhs.dims()[0] != k || rhs.dims()[1] != n {
        return Err(Error::ShapeMismatch {
            expected: vec![k, n],
            got: rhs.dims()[..2].to_vec(),
        });
    }
    if lhs.dims()[2..] != rhs.dims()[2..] {
        return Err(Error::ShapeMismatch {
            expected: lhs.dims()[2..].to_vec(),
            got: rhs.dims()[2..].to_vec(),
        });
    }
    if lhs.logical_memory_space() != rhs.logical_memory_space() {
        return Err(Error::InvalidArgument(
            "batched_gemm expects inputs in the same logical memory space".into(),
        ));
    }

    let batch_dims = lhs.dims()[2..].to_vec();
    let output_dims = {
        let mut dims = Vec::with_capacity(lhs.ndim());
        dims.push(m);
        dims.push(n);
        dims.extend_from_slice(&batch_dims);
        dims
    };
    let mut output = Tensor::zeros(
        &output_dims,
        lhs.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    )?;
    let desc = SemiringCoreDescriptor::BatchedGemm {
        batch_dims,
        m,
        n,
        k,
    };
    let plan = <C::SemiringBackend as TensorSemiringCore<Standard<T>>>::plan(
        ctx,
        &desc,
        &[lhs.dims(), rhs.dims(), output.dims()],
    )?;
    <C::SemiringBackend as TensorSemiringCore<Standard<T>>>::execute(
        ctx,
        &plan,
        T::one(),
        &[&lhs, &rhs],
        T::zero(),
        &mut output,
    )?;

    Ok(output)
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

pub(crate) fn identity_matrix<T: LinalgScalar>(
    n: usize,
    memory_space: LogicalMemorySpace,
) -> Result<Tensor<T>> {
    let numel = n
        .checked_mul(n)
        .ok_or_else(|| Error::InvalidArgument(format!("identity_matrix overflow for n={n}")))?;
    let mut data = std::iter::repeat_with(T::zero)
        .take(numel)
        .collect::<Vec<_>>();
    for i in 0..n {
        data[i + i * n] = T::one();
    }
    let n_isize = isize::try_from(n)
        .map_err(|_| Error::StrideError(format!("identity_matrix stride overflow for n={n}")))?;
    let tensor = Tensor::from_vec(data, &[n, n], &[1, n_isize], 0)?;
    if memory_space == LogicalMemorySpace::MainMemory {
        Ok(tensor)
    } else {
        tensor.to_memory_space_async(memory_space)
    }
}

pub(crate) fn resolve_conj<T, C>(ctx: &mut C, input: &Tensor<T>) -> Tensor<T>
where
    T: LinalgScalar + tenferro_algebra::Conjugate,
    C: TensorResolveConjContextFor<T>,
{
    <C as TensorResolveConjContextFor<T>>::resolve_conj(ctx, input)
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
    let mut output = Tensor::zeros(
        lhs.dims(),
        lhs.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    )?;
    scalar_binary_same_shape_into(ctx, lhs, rhs, op, &mut output)?;
    Ok(output)
}

pub(crate) fn scalar_binary_same_shape_into<T, C>(
    ctx: &mut C,
    lhs: &Tensor<T>,
    rhs: &Tensor<T>,
    op: ScalarBinaryOp,
    output: &mut Tensor<T>,
) -> Result<()>
where
    T: LinalgScalar,
    C: TensorScalarContextFor<Standard<T>>,
{
    if lhs.dims() != rhs.dims() {
        return Err(Error::ShapeMismatch {
            expected: lhs.dims().to_vec(),
            got: rhs.dims().to_vec(),
        });
    }
    if lhs.dims() != output.dims() {
        return Err(Error::ShapeMismatch {
            expected: lhs.dims().to_vec(),
            got: output.dims().to_vec(),
        });
    }
    if lhs.logical_memory_space() != rhs.logical_memory_space()
        || lhs.logical_memory_space() != output.logical_memory_space()
    {
        return Err(Error::InvalidArgument(format!(
            "scalar_binary_same_shape_into requires matching memory spaces, got lhs {:?}, rhs {:?}, output {:?}",
            lhs.logical_memory_space(),
            rhs.logical_memory_space(),
            output.logical_memory_space()
        )));
    }

    let desc = ScalarPrimsDescriptor::PointwiseBinary { op };
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
        output,
    )?;
    Ok(())
}

pub(crate) fn scalar_where_same_shape<T, C>(
    ctx: &mut C,
    cond: &Tensor<T>,
    on_true: &Tensor<T>,
    on_false: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: LinalgScalar<Real = T> + num_traits::Float,
    C: TensorScalarContextFor<Standard<T>>,
{
    let desc = ScalarPrimsDescriptor::PointwiseTernary {
        op: ScalarTernaryOp::Where,
    };
    let mut output = Tensor::zeros(
        cond.dims(),
        cond.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    )?;
    let plan = <C::ScalarBackend as TensorScalarPrims<Standard<T>>>::plan(
        ctx,
        &desc,
        &[cond.dims(), on_true.dims(), on_false.dims(), output.dims()],
    )?;
    <C::ScalarBackend as TensorScalarPrims<Standard<T>>>::execute(
        ctx,
        &plan,
        T::one(),
        &[cond, on_true, on_false],
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
    )?;
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

pub(crate) fn complex_real_unary_same_shape<T, C>(
    ctx: &mut C,
    input: &Tensor<T>,
    op: ComplexRealUnaryOp,
) -> Result<Tensor<<T as LinalgScalar>::Real>>
where
    T: LinalgScalar + ComplexFloat<Real = <T as LinalgScalar>::Real>,
    C: TensorComplexRealContextFor<T>,
    C::ComplexRealBackend: TensorComplexRealPrims<T, Context = C, Real = <T as LinalgScalar>::Real>,
{
    let desc = ComplexRealPrimsDescriptor::PointwiseUnary { op };
    let mut output = Tensor::zeros(
        input.dims(),
        input.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    )?;
    let plan = <C::ComplexRealBackend as TensorComplexRealPrims<T>>::plan(
        ctx,
        &desc,
        &[input.dims(), output.dims()],
    )?;
    <C::ComplexRealBackend as TensorComplexRealPrims<T>>::execute(
        ctx,
        &plan,
        <T as LinalgScalar>::Real::one(),
        &[input],
        <T as LinalgScalar>::Real::zero(),
        &mut output,
    )?;
    Ok(output)
}

#[doc(hidden)]
/// Hidden dispatch trait for scaling a tensor by a same-shape real tensor.
///
/// This keeps public linalg entrypoints generic across real and complex scalar
/// families while routing to the appropriate backend family underneath.
///
/// # Examples
///
/// ```ignore
/// fn require_scale_by_real<T, C>()
/// where
///     T: tenferro_linalg::ScaleTensorByRealSameShape<C>,
/// {
/// }
/// ```
pub trait ScaleTensorByRealSameShape<C>: LinalgScalar {
    fn scale_tensor_by_real_same_shape(
        ctx: &mut C,
        lhs: &Tensor<Self>,
        rhs: &Tensor<Self::Real>,
    ) -> Result<Tensor<Self>>;
}

impl<C> ScaleTensorByRealSameShape<C> for f32
where
    C: TensorScalarContextFor<Standard<f32>>,
{
    fn scale_tensor_by_real_same_shape(
        ctx: &mut C,
        lhs: &Tensor<Self>,
        rhs: &Tensor<Self::Real>,
    ) -> Result<Tensor<Self>> {
        scalar_binary_same_shape(ctx, lhs, rhs, ScalarBinaryOp::Mul)
    }
}

impl<C> ScaleTensorByRealSameShape<C> for f64
where
    C: TensorScalarContextFor<Standard<f64>>,
{
    fn scale_tensor_by_real_same_shape(
        ctx: &mut C,
        lhs: &Tensor<Self>,
        rhs: &Tensor<Self::Real>,
    ) -> Result<Tensor<Self>> {
        scalar_binary_same_shape(ctx, lhs, rhs, ScalarBinaryOp::Mul)
    }
}

impl<C> ScaleTensorByRealSameShape<C> for Complex32
where
    C: TensorComplexScaleContextFor<Complex32>,
    C::ComplexScaleBackend: TensorComplexScalePrims<Complex32, Context = C>,
{
    fn scale_tensor_by_real_same_shape(
        ctx: &mut C,
        lhs: &Tensor<Self>,
        rhs: &Tensor<Self::Real>,
    ) -> Result<Tensor<Self>> {
        let desc = ComplexScalePrimsDescriptor::PointwiseMul;
        let mut output = Tensor::zeros(
            lhs.dims(),
            lhs.logical_memory_space(),
            MemoryOrder::ColumnMajor,
        )?;
        let plan = <C::ComplexScaleBackend as TensorComplexScalePrims<Complex32>>::plan(
            ctx,
            &desc,
            &[lhs.dims(), rhs.dims(), output.dims()],
        )?;
        <C::ComplexScaleBackend as TensorComplexScalePrims<Complex32>>::execute(
            ctx,
            &plan,
            Complex32::new(1.0, 0.0),
            lhs,
            rhs,
            Complex32::new(0.0, 0.0),
            &mut output,
        )?;
        Ok(output)
    }
}

impl<C> ScaleTensorByRealSameShape<C> for Complex64
where
    C: TensorComplexScaleContextFor<Complex64>,
    C::ComplexScaleBackend: TensorComplexScalePrims<Complex64, Context = C>,
{
    fn scale_tensor_by_real_same_shape(
        ctx: &mut C,
        lhs: &Tensor<Self>,
        rhs: &Tensor<Self::Real>,
    ) -> Result<Tensor<Self>> {
        let desc = ComplexScalePrimsDescriptor::PointwiseMul;
        let mut output = Tensor::zeros(
            lhs.dims(),
            lhs.logical_memory_space(),
            MemoryOrder::ColumnMajor,
        )?;
        let plan = <C::ComplexScaleBackend as TensorComplexScalePrims<Complex64>>::plan(
            ctx,
            &desc,
            &[lhs.dims(), rhs.dims(), output.dims()],
        )?;
        <C::ComplexScaleBackend as TensorComplexScalePrims<Complex64>>::execute(
            ctx,
            &plan,
            Complex64::new(1.0, 0.0),
            lhs,
            rhs,
            Complex64::new(0.0, 0.0),
            &mut output,
        )?;
        Ok(output)
    }
}

pub(crate) fn complex_scale_same_shape<T, C>(
    ctx: &mut C,
    lhs: &Tensor<T>,
    rhs: &Tensor<T::Real>,
) -> Result<Tensor<T>>
where
    T: LinalgScalar + ScaleTensorByRealSameShape<C>,
{
    T::scale_tensor_by_real_same_shape(ctx, lhs, rhs)
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
    )?;
    let plan =
        <<C as TensorScalarContextFor<Standard<T>>>::ScalarBackend as TensorAnalyticPrims<
            Standard<T>,
        >>::plan(ctx, &desc, &[input.dims(), output.dims()])?;
    <<C as TensorScalarContextFor<Standard<T>>>::ScalarBackend as TensorAnalyticPrims<
        Standard<T>,
    >>::execute(ctx, &plan, T::one(), &[input], T::zero(), &mut output)?;
    Ok(output)
}

pub(crate) fn analytic_binary_same_shape<T, C>(
    ctx: &mut C,
    lhs: &Tensor<T>,
    rhs: &Tensor<T>,
    op: AnalyticBinaryOp,
) -> Result<Tensor<T>>
where
    T: LinalgScalar + crate::KernelLinalgScalar,
    C: TensorScalarContextFor<Standard<T>>,
    <C as TensorScalarContextFor<Standard<T>>>::ScalarBackend:
        TensorAnalyticPrims<Standard<T>, Context = C>,
{
    let desc = AnalyticPrimsDescriptor::PointwiseBinary { op };
    let mut output = Tensor::zeros(
        lhs.dims(),
        lhs.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    )?;
    let plan =
        <<C as TensorScalarContextFor<Standard<T>>>::ScalarBackend as TensorAnalyticPrims<
            Standard<T>,
        >>::plan(ctx, &desc, &[lhs.dims(), rhs.dims(), output.dims()])?;
    <<C as TensorScalarContextFor<Standard<T>>>::ScalarBackend as TensorAnalyticPrims<
        Standard<T>,
    >>::execute(ctx, &plan, T::one(), &[lhs, rhs], T::zero(), &mut output)?;
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
    )?;
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

pub(crate) fn complex_real_reduce_keep_axes<T, C>(
    ctx: &mut C,
    input: &Tensor<T>,
    unary_op: ComplexRealUnaryOp,
    kept_axes: &[usize],
    reduction_op: ScalarReductionOp,
) -> Result<Tensor<<T as LinalgScalar>::Real>>
where
    T: LinalgScalar + ComplexFloat<Real = <T as LinalgScalar>::Real>,
    C: TensorComplexRealContextFor<T>,
    C::ComplexRealBackend: TensorComplexRealPrims<T, Context = C, Real = <T as LinalgScalar>::Real>,
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
    let desc = ComplexRealPrimsDescriptor::Reduction {
        modes_a,
        modes_c,
        unary_op,
        reduction_op,
    };
    let output_dims: Vec<usize> = kept_axes.iter().map(|&axis| input.dims()[axis]).collect();
    let mut output = Tensor::zeros(
        &output_dims,
        input.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    )?;
    let plan = <C::ComplexRealBackend as TensorComplexRealPrims<T>>::plan(
        ctx,
        &desc,
        &[input.dims(), output.dims()],
    )?;
    <C::ComplexRealBackend as TensorComplexRealPrims<T>>::execute(
        ctx,
        &plan,
        <T as LinalgScalar>::Real::one(),
        &[input],
        <T as LinalgScalar>::Real::zero(),
        &mut output,
    )?;
    Ok(output)
}

#[cfg(test)]
mod tests;
