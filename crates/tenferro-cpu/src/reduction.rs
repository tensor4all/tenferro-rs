use std::mem::size_of_val;

use num_traits::Float;
use strided_kernel::{
    col_major_strides, reduce_axis, ErasedRawStridedMut, ErasedRawStridedRef, ErasedReducePlan,
    ExecContext, KernelDType, ReduceOp,
};

use super::{typed_host_data, typed_view, typed_view_from_view};
use crate::buffer_pool::BufferPool;
use crate::materialize_tensor_read;
use tenferro_tensor::{
    DType, Tensor, TensorRank, TensorRead, TensorScalar, TensorView, TypedTensor, TypedTensorView,
};

fn validate_axes(op: &'static str, axes: &[usize], rank: usize) -> crate::Result<()> {
    let mut seen = vec![false; rank];
    for &axis in axes {
        if axis >= rank {
            return Err(crate::Error::axis_out_of_bounds(op, axis, rank));
        }
        if seen[axis] {
            return Err(crate::Error::duplicate_axis(op, axis, "axes"));
        }
        seen[axis] = true;
    }
    Ok(())
}

fn ensure_host_tensor(op: &'static str, input: &Tensor) -> crate::Result<()> {
    macro_rules! ensure {
        ($tensor:expr) => {{
            typed_host_data(op, $tensor)?;
            Ok(())
        }};
    }

    match input {
        Tensor::F32(t) => ensure!(t),
        Tensor::F64(t) => ensure!(t),
        Tensor::I32(t) => ensure!(t),
        Tensor::I64(t) => ensure!(t),
        Tensor::Bool(t) => ensure!(t),
        Tensor::C32(t) => ensure!(t),
        Tensor::C64(t) => ensure!(t),
    }
}

fn validate_reduced_axes_nonempty(
    op: &'static str,
    shape: &[usize],
    axes: &[usize],
) -> crate::Result<()> {
    validate_axes(op, axes, shape.len())?;
    for &axis in axes {
        if shape[axis] == 0 {
            return Err(crate::Error::invalid_argument(
                op,
                "configuration",
                format!("cannot reduce over zero-length axis {axis}"),
            ));
        }
    }
    Ok(())
}

fn reduction_empty_axes_noop(
    op: &'static str,
    input: &Tensor,
    axes: &[usize],
) -> crate::Result<Option<Tensor>> {
    validate_axes(op, axes, input.shape().len())?;
    // INVARIANT: empty-axis reduction is semantic identity, but this public
    // owned-output API must return an independently owned tensor.
    Ok(axes.is_empty().then(|| input.clone()))
}

fn reduction_read_empty_axes_noop(
    buffers: &mut BufferPool,
    op: &'static str,
    input: &TensorRead<'_>,
    axes: &[usize],
) -> crate::Result<Option<Tensor>> {
    validate_axes(op, axes, input.shape().len())?;
    if !axes.is_empty() {
        return Ok(None);
    }

    materialize_tensor_read(buffers, op, input.clone()).map(Some)
}

fn nan_propagating_max<T: Float>(a: T, b: T) -> T {
    if a.is_nan() || b.is_nan() {
        T::nan()
    } else {
        a.max(b)
    }
}

fn nan_propagating_min<T: Float>(a: T, b: T) -> T {
    if a.is_nan() || b.is_nan() {
        T::nan()
    } else {
        a.min(b)
    }
}

fn kernel_dtype(dtype: DType) -> KernelDType {
    match dtype {
        DType::F32 => KernelDType::F32,
        DType::F64 => KernelDType::F64,
        DType::I32 => KernelDType::I32,
        DType::I64 => KernelDType::I64,
        DType::Bool => KernelDType::Bool,
        DType::C32 => KernelDType::C32,
        DType::C64 => KernelDType::C64,
    }
}

fn typed_bytes<T>(data: &[T]) -> &[u8] {
    // SAFETY: `data` is an aligned typed slice. The returned byte slice has
    // the same lifetime and exact byte length, and is read-only.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), size_of_val(data)) }
}

fn typed_bytes_mut<T>(data: &mut [T]) -> &mut [u8] {
    // SAFETY: `data` is an aligned typed slice. The erased reduction writes
    // valid scalar values before the buffer is read through its typed view.
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<u8>(), size_of_val(data)) }
}

fn reduction_output_shape(input_shape: &[usize], axes: &[usize]) -> Vec<usize> {
    input_shape
        .iter()
        .enumerate()
        .filter(|&(axis, _)| !axes.contains(&axis))
        .map(|(_, &dim)| dim)
        .collect()
}

trait WrappingReductionElem: Copy + Clone + Send + Sync + 'static {
    fn min_value_elem() -> Self;
    fn max_value_elem() -> Self;
    fn max_elem(self, other: Self) -> Self;
    fn min_elem(self, other: Self) -> Self;
}

macro_rules! impl_wrapping_reduction_elem {
    ($ty:ty) => {
        impl WrappingReductionElem for $ty {
            fn min_value_elem() -> Self {
                <$ty>::MIN
            }

            fn max_value_elem() -> Self {
                <$ty>::MAX
            }

            fn max_elem(self, other: Self) -> Self {
                self.max(other)
            }

            fn min_elem(self, other: Self) -> Self {
                self.min(other)
            }
        }
    };
}

impl_wrapping_reduction_elem!(i32);
impl_wrapping_reduction_elem!(i64);

/// # Errors
///
/// Returns [`crate::Error::Validation`] with `AxisOutOfBounds`,
/// `DuplicateAxis`, or `InvalidArgument` for invalid axes or zero-length
/// reductions, [`crate::Error::Unsupported`] for `Bool`, or a typed backend
/// error when the input storage cannot be read.
pub(crate) fn reduce_sum(
    input: &Tensor,
    axes: &[usize],
    exec_context: &ExecContext,
) -> crate::Result<Tensor> {
    if let Some(output) = reduction_empty_axes_noop("reduce_sum", input, axes)? {
        return Ok(output);
    }

    match input {
        Tensor::F32(t) => Ok(Tensor::F32(typed_reduce_sum(t, axes, exec_context)?)),
        Tensor::F64(t) => Ok(Tensor::F64(typed_reduce_sum(t, axes, exec_context)?)),
        Tensor::I32(t) => Ok(Tensor::I32(typed_reduce_sum_wrapping(
            t,
            axes,
            exec_context,
        )?)),
        Tensor::I64(t) => Ok(Tensor::I64(typed_reduce_sum_wrapping(
            t,
            axes,
            exec_context,
        )?)),
        Tensor::Bool(_) => Err(crate::Error::unsupported(
            "reduce_sum",
            "unsupported dtype Bool",
        )),
        Tensor::C32(t) => Ok(Tensor::C32(typed_reduce_sum(t, axes, exec_context)?)),
        Tensor::C64(t) => Ok(Tensor::C64(typed_reduce_sum(t, axes, exec_context)?)),
    }
}

pub(crate) fn reduce_sum_read(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
    axes: &[usize],
    exec_context: &ExecContext,
) -> crate::Result<Tensor> {
    if let Some(output) = reduction_read_empty_axes_noop(buffers, "reduce_sum", &input, axes)? {
        return Ok(output);
    }

    match input {
        TensorRead::Tensor(input) => {
            ensure_host_tensor("reduce_sum", input)?;
            reduce_sum(input, axes, exec_context)
        }
        TensorRead::View(TensorView::F32(t)) => Ok(Tensor::F32(typed_reduce_view_erased(
            buffers,
            &t,
            axes,
            ReduceOp::Sum,
            "reduce_sum",
            exec_context,
        )?)),
        TensorRead::View(TensorView::F64(t)) => Ok(Tensor::F64(typed_reduce_view_erased(
            buffers,
            &t,
            axes,
            ReduceOp::Sum,
            "reduce_sum",
            exec_context,
        )?)),
        TensorRead::View(TensorView::I32(t)) => Ok(Tensor::I32(typed_reduce_view_erased(
            buffers,
            &t,
            axes,
            ReduceOp::Sum,
            "reduce_sum",
            exec_context,
        )?)),
        TensorRead::View(TensorView::I64(t)) => Ok(Tensor::I64(typed_reduce_view_erased(
            buffers,
            &t,
            axes,
            ReduceOp::Sum,
            "reduce_sum",
            exec_context,
        )?)),
        TensorRead::View(TensorView::Bool(_)) => Err(crate::Error::unsupported(
            "reduce_sum",
            "unsupported dtype Bool",
        )),
        TensorRead::View(TensorView::C32(t)) => Ok(Tensor::C32(typed_reduce_view_erased(
            buffers,
            &t,
            axes,
            ReduceOp::Sum,
            "reduce_sum",
            exec_context,
        )?)),
        TensorRead::View(TensorView::C64(t)) => Ok(Tensor::C64(typed_reduce_view_erased(
            buffers,
            &t,
            axes,
            ReduceOp::Sum,
            "reduce_sum",
            exec_context,
        )?)),
    }
}

/// # Errors
///
/// Returns [`crate::Error::Validation`] with `AxisOutOfBounds`,
/// `DuplicateAxis`, or `InvalidArgument` for invalid axes or zero-length
/// reductions, [`crate::Error::Unsupported`] for `Bool`, or a typed backend
/// error when the input storage cannot be read.
pub(crate) fn reduce_prod(
    input: &Tensor,
    axes: &[usize],
    exec_context: &ExecContext,
) -> crate::Result<Tensor> {
    if let Some(output) = reduction_empty_axes_noop("reduce_prod", input, axes)? {
        return Ok(output);
    }

    match input {
        Tensor::F32(t) => Ok(Tensor::F32(typed_reduce_prod(t, axes, exec_context)?)),
        Tensor::F64(t) => Ok(Tensor::F64(typed_reduce_prod(t, axes, exec_context)?)),
        Tensor::I32(t) => Ok(Tensor::I32(typed_reduce_prod_wrapping(
            t,
            axes,
            exec_context,
        )?)),
        Tensor::I64(t) => Ok(Tensor::I64(typed_reduce_prod_wrapping(
            t,
            axes,
            exec_context,
        )?)),
        Tensor::Bool(_) => Err(crate::Error::unsupported(
            "reduce_prod",
            "unsupported dtype Bool",
        )),
        Tensor::C32(t) => Ok(Tensor::C32(typed_reduce_prod(t, axes, exec_context)?)),
        Tensor::C64(t) => Ok(Tensor::C64(typed_reduce_prod(t, axes, exec_context)?)),
    }
}

pub(crate) fn reduce_prod_read(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
    axes: &[usize],
    exec_context: &ExecContext,
) -> crate::Result<Tensor> {
    if let Some(output) = reduction_read_empty_axes_noop(buffers, "reduce_prod", &input, axes)? {
        return Ok(output);
    }

    match input {
        TensorRead::Tensor(input) => {
            ensure_host_tensor("reduce_prod", input)?;
            reduce_prod(input, axes, exec_context)
        }
        TensorRead::View(TensorView::F32(t)) => Ok(Tensor::F32(typed_reduce_view_erased(
            buffers,
            &t,
            axes,
            ReduceOp::Product,
            "reduce_prod",
            exec_context,
        )?)),
        TensorRead::View(TensorView::F64(t)) => Ok(Tensor::F64(typed_reduce_view_erased(
            buffers,
            &t,
            axes,
            ReduceOp::Product,
            "reduce_prod",
            exec_context,
        )?)),
        TensorRead::View(TensorView::I32(t)) => Ok(Tensor::I32(typed_reduce_view_erased(
            buffers,
            &t,
            axes,
            ReduceOp::Product,
            "reduce_prod",
            exec_context,
        )?)),
        TensorRead::View(TensorView::I64(t)) => Ok(Tensor::I64(typed_reduce_view_erased(
            buffers,
            &t,
            axes,
            ReduceOp::Product,
            "reduce_prod",
            exec_context,
        )?)),
        TensorRead::View(TensorView::Bool(_)) => Err(crate::Error::unsupported(
            "reduce_prod",
            "unsupported dtype Bool",
        )),
        TensorRead::View(TensorView::C32(t)) => Ok(Tensor::C32(typed_reduce_view_erased(
            buffers,
            &t,
            axes,
            ReduceOp::Product,
            "reduce_prod",
            exec_context,
        )?)),
        TensorRead::View(TensorView::C64(t)) => Ok(Tensor::C64(typed_reduce_view_erased(
            buffers,
            &t,
            axes,
            ReduceOp::Product,
            "reduce_prod",
            exec_context,
        )?)),
    }
}

/// # Errors
///
/// Returns [`crate::Error::Validation`] with `AxisOutOfBounds`,
/// `DuplicateAxis`, or `InvalidArgument` for invalid axes or zero-length
/// reductions, [`crate::Error::Unsupported`] for `Bool` and complex dtypes, or
/// a typed backend error when the input storage cannot be read.
pub fn reduce_max(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
    if let Some(output) = reduction_empty_axes_noop("reduce_max", input, axes)? {
        return Ok(output);
    }

    match input {
        Tensor::F32(tensor) => Ok(Tensor::F32(typed_reduce_max(tensor, axes)?)),
        Tensor::F64(tensor) => Ok(Tensor::F64(typed_reduce_max(tensor, axes)?)),
        Tensor::I32(tensor) => Ok(Tensor::I32(typed_reduce_max_integer(tensor, axes)?)),
        Tensor::I64(tensor) => Ok(Tensor::I64(typed_reduce_max_integer(tensor, axes)?)),
        Tensor::Bool(_) | Tensor::C32(_) | Tensor::C64(_) => Err(crate::Error::unsupported(
            "reduce_max",
            format!("unsupported dtype {:?}", input.dtype()),
        )),
    }
}

pub(crate) fn reduce_max_read(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
    axes: &[usize],
) -> crate::Result<Tensor> {
    if let Some(output) = reduction_read_empty_axes_noop(buffers, "reduce_max", &input, axes)? {
        return Ok(output);
    }

    match input {
        TensorRead::Tensor(input) => {
            ensure_host_tensor("reduce_max", input)?;
            reduce_max(input, axes)
        }
        TensorRead::View(TensorView::F32(t)) => {
            validate_reduced_axes_nonempty("reduce_max", t.shape(), axes)?;
            Ok(Tensor::F32(typed_reduce_view(
                &t,
                axes,
                |x| x,
                nan_propagating_max,
                f32::neg_infinity(),
                "reduce_max",
            )?))
        }
        TensorRead::View(TensorView::F64(t)) => {
            validate_reduced_axes_nonempty("reduce_max", t.shape(), axes)?;
            Ok(Tensor::F64(typed_reduce_view(
                &t,
                axes,
                |x| x,
                nan_propagating_max,
                f64::neg_infinity(),
                "reduce_max",
            )?))
        }
        TensorRead::View(TensorView::I32(t)) => {
            validate_reduced_axes_nonempty("reduce_max", t.shape(), axes)?;
            Ok(Tensor::I32(typed_reduce_view(
                &t,
                axes,
                |x| x,
                |a, b| a.max_elem(b),
                i32::min_value_elem(),
                "reduce_max",
            )?))
        }
        TensorRead::View(TensorView::I64(t)) => {
            validate_reduced_axes_nonempty("reduce_max", t.shape(), axes)?;
            Ok(Tensor::I64(typed_reduce_view(
                &t,
                axes,
                |x| x,
                |a, b| a.max_elem(b),
                i64::min_value_elem(),
                "reduce_max",
            )?))
        }
        view => Err(crate::Error::unsupported(
            "reduce_max",
            format!("unsupported dtype {:?}", view.dtype()),
        )),
    }
}

/// # Errors
///
/// Returns [`crate::Error::Validation`] with `AxisOutOfBounds`,
/// `DuplicateAxis`, or `InvalidArgument` for invalid axes or zero-length
/// reductions, [`crate::Error::Unsupported`] for `Bool` and complex dtypes, or
/// a typed backend error when the input storage cannot be read.
pub fn reduce_min(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
    if let Some(output) = reduction_empty_axes_noop("reduce_min", input, axes)? {
        return Ok(output);
    }

    match input {
        Tensor::F32(tensor) => Ok(Tensor::F32(typed_reduce_min(tensor, axes)?)),
        Tensor::F64(tensor) => Ok(Tensor::F64(typed_reduce_min(tensor, axes)?)),
        Tensor::I32(tensor) => Ok(Tensor::I32(typed_reduce_min_integer(tensor, axes)?)),
        Tensor::I64(tensor) => Ok(Tensor::I64(typed_reduce_min_integer(tensor, axes)?)),
        Tensor::Bool(_) | Tensor::C32(_) | Tensor::C64(_) => Err(crate::Error::unsupported(
            "reduce_min",
            format!("unsupported dtype {:?}", input.dtype()),
        )),
    }
}

pub(crate) fn reduce_min_read(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
    axes: &[usize],
) -> crate::Result<Tensor> {
    if let Some(output) = reduction_read_empty_axes_noop(buffers, "reduce_min", &input, axes)? {
        return Ok(output);
    }

    match input {
        TensorRead::Tensor(input) => {
            ensure_host_tensor("reduce_min", input)?;
            reduce_min(input, axes)
        }
        TensorRead::View(TensorView::F32(t)) => {
            validate_reduced_axes_nonempty("reduce_min", t.shape(), axes)?;
            Ok(Tensor::F32(typed_reduce_view(
                &t,
                axes,
                |x| x,
                nan_propagating_min,
                f32::infinity(),
                "reduce_min",
            )?))
        }
        TensorRead::View(TensorView::F64(t)) => {
            validate_reduced_axes_nonempty("reduce_min", t.shape(), axes)?;
            Ok(Tensor::F64(typed_reduce_view(
                &t,
                axes,
                |x| x,
                nan_propagating_min,
                f64::infinity(),
                "reduce_min",
            )?))
        }
        TensorRead::View(TensorView::I32(t)) => {
            validate_reduced_axes_nonempty("reduce_min", t.shape(), axes)?;
            Ok(Tensor::I32(typed_reduce_view(
                &t,
                axes,
                |x| x,
                |a, b| a.min_elem(b),
                i32::max_value_elem(),
                "reduce_min",
            )?))
        }
        TensorRead::View(TensorView::I64(t)) => {
            validate_reduced_axes_nonempty("reduce_min", t.shape(), axes)?;
            Ok(Tensor::I64(typed_reduce_view(
                &t,
                axes,
                |x| x,
                |a, b| a.min_elem(b),
                i64::max_value_elem(),
                "reduce_min",
            )?))
        }
        view => Err(crate::Error::unsupported(
            "reduce_min",
            format!("unsupported dtype {:?}", view.dtype()),
        )),
    }
}

fn typed_reduce<T, M, R>(
    input: &TypedTensor<T>,
    axes: &[usize],
    map_fn: M,
    reduce_fn: R,
    init: T,
    label: &'static str,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Send + Sync,
    M: Fn(T) -> T + Copy + Sync,
    R: Fn(T, T) -> T + Copy + Sync,
{
    validate_reduced_axes_nonempty(label, input.shape(), axes)?;
    if axes.is_empty() {
        // INVARIANT: empty-axis typed reductions preserve values exactly while
        // satisfying the owned-output contract.
        return Ok(input.clone());
    }

    let output_shape: Vec<usize> = input
        .shape()
        .iter()
        .enumerate()
        .filter(|&(axis, _)| !axes.contains(&axis))
        .map(|(_, &dim)| dim)
        .collect();

    let mut sorted_axes = axes.to_vec();
    sorted_axes.sort_unstable_by(|a, b| b.cmp(a));
    let Some((&first_axis, remaining_axes)) = sorted_axes.split_first() else {
        // INVARIANT: this is the same empty-axis owned identity case handled
        // above; it remains here to keep split-first control flow total.
        return Ok(input.clone());
    };

    let input_view = typed_view(label, input)?;
    let mut current = reduce_axis(&input_view, first_axis, map_fn, reduce_fn, init)
        .map_err(|err| crate::Error::backend_source(label, err))?;

    for &axis in remaining_axes {
        current = reduce_axis(&current.view(), axis, map_fn, reduce_fn, init)
            .map_err(|err| crate::Error::backend_source(label, err))?;
    }

    TypedTensor::from_vec_col_major(output_shape, current.into_data())
}

fn typed_reduce_erased<T>(
    input: &TypedTensor<T>,
    axes: &[usize],
    op: ReduceOp,
    label: &'static str,
    exec_context: &ExecContext,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + TensorScalar,
{
    validate_reduced_axes_nonempty(label, input.shape(), axes)?;
    if axes.is_empty() {
        // INVARIANT: empty-axis typed reductions preserve values exactly while
        // satisfying the owned-output contract.
        return Ok(input.clone());
    }

    let output_shape = reduction_output_shape(input.shape(), axes);
    let output_len =
        tenferro_tensor::validate::checked_shape_product(label, "output shape", &output_shape)?;
    let output_strides = col_major_strides(&output_shape);
    let dtype = kernel_dtype(T::dtype());
    let input_view = typed_view(label, input)?;
    let plan = ErasedReducePlan::compile_axes(
        dtype,
        op,
        input_view.dims(),
        input_view.strides(),
        &output_shape,
        &output_strides,
        axes,
    )
    .map_err(|err| crate::Error::backend_source(label, err))?;
    let source = ErasedRawStridedRef::new(
        dtype,
        typed_bytes(input_view.data()),
        input_view.dims(),
        input_view.strides(),
        input_view.offset(),
    )
    .map_err(|err| crate::Error::backend_source(label, err))?;
    // SAFETY: ErasedReducePlan writes every destination element.
    let mut output = unsafe { uninit_full_overwrite_vec(output_len) };
    let mut dest = ErasedRawStridedMut::new(
        dtype,
        typed_bytes_mut(&mut output),
        &output_shape,
        &output_strides,
        0,
    )
    .map_err(|err| crate::Error::backend_source(label, err))?;
    plan.execute(exec_context, &mut dest, &source)
        .map_err(|err| crate::Error::backend_source(label, err))?;

    TypedTensor::from_vec_col_major(output_shape, output)
}

#[allow(clippy::uninit_vec)]
unsafe fn uninit_full_overwrite_vec<T>(len: usize) -> Vec<T> {
    let mut output = Vec::with_capacity(len);
    // SAFETY: the caller promises every element is overwritten before any read.
    unsafe { output.set_len(len) };
    output
}

pub(crate) fn typed_reduce_view<T, M, R, TR>(
    input: &TypedTensorView<'_, T, TR>,
    axes: &[usize],
    map_fn: M,
    reduce_fn: R,
    init: T,
    label: &'static str,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Send + Sync + 'static,
    M: Fn(T) -> T + Copy + Sync,
    R: Fn(T, T) -> T + Copy + Sync,
    TR: TensorRank,
{
    validate_reduced_axes_nonempty(label, input.shape(), axes)?;
    if axes.is_empty() {
        return Err(crate::Error::unsupported(
            label,
            "empty-axis view reductions require backend-owned materialization",
        ));
    }

    let output_shape: Vec<usize> = input
        .shape()
        .iter()
        .enumerate()
        .filter(|&(axis, _)| !axes.contains(&axis))
        .map(|(_, &dim)| dim)
        .collect();

    let mut sorted_axes = axes.to_vec();
    sorted_axes.sort_unstable_by(|a, b| b.cmp(a));
    let Some((&first_axis, remaining_axes)) = sorted_axes.split_first() else {
        return Err(crate::Error::unsupported(
            label,
            "empty-axis view reductions require backend-owned materialization",
        ));
    };

    let input_view = typed_view_from_view(label, input)?;
    let mut current = reduce_axis(&input_view, first_axis, map_fn, reduce_fn, init)
        .map_err(|err| crate::Error::backend_source(label, err))?;

    for &axis in remaining_axes {
        current = reduce_axis(&current.view(), axis, map_fn, reduce_fn, init)
            .map_err(|err| crate::Error::backend_source(label, err))?;
    }

    TypedTensor::from_vec_col_major(output_shape, current.into_data())
}

fn typed_reduce_view_erased<T, TR>(
    buffers: &mut BufferPool,
    input: &TypedTensorView<'_, T, TR>,
    axes: &[usize],
    op: ReduceOp,
    label: &'static str,
    exec_context: &ExecContext,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + TensorScalar + crate::buffer_pool::PoolScalar + 'static,
    TR: TensorRank,
{
    validate_reduced_axes_nonempty(label, input.shape(), axes)?;
    if axes.is_empty() {
        return Err(crate::Error::unsupported(
            label,
            "empty-axis view reductions require backend-owned materialization",
        ));
    }

    let output_shape = reduction_output_shape(input.shape(), axes);
    let output_strides = col_major_strides(&output_shape);
    let dtype = kernel_dtype(T::dtype());
    let input_view = typed_view_from_view(label, input)?;
    let plan = ErasedReducePlan::compile_axes(
        dtype,
        op,
        input_view.dims(),
        input_view.strides(),
        &output_shape,
        &output_strides,
        axes,
    )
    .map_err(|err| crate::Error::backend_source(label, err))?;
    let source = ErasedRawStridedRef::new(
        dtype,
        typed_bytes(input_view.data()),
        input_view.dims(),
        input_view.strides(),
        input_view.offset(),
    )
    .map_err(|err| crate::Error::backend_source(label, err))?;
    // SAFETY: ErasedReducePlan writes every destination element.
    let mut output = unsafe { crate::typed_array_uninit_from_pool(buffers, &output_shape) }?;
    let mut dest = ErasedRawStridedMut::new(
        dtype,
        typed_bytes_mut(output.data_mut()),
        &output_shape,
        &output_strides,
        0,
    )
    .map_err(|err| crate::Error::backend_source(label, err))?;
    plan.execute(exec_context, &mut dest, &source)
        .map_err(|err| crate::Error::backend_source(label, err))?;

    Ok(crate::tensor_from_array(output))
}

/// # Errors
///
/// Returns [`crate::Error::Validation`] with `AxisOutOfBounds`,
/// `DuplicateAxis`, or `InvalidArgument` for invalid axes or zero-length
/// reductions, or a typed backend error while materializing the result.
pub(crate) fn typed_reduce_sum<T>(
    input: &TypedTensor<T>,
    axes: &[usize],
    exec_context: &ExecContext,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Send + Sync + TensorScalar,
{
    typed_reduce_erased(input, axes, ReduceOp::Sum, "reduce_sum", exec_context)
}

fn typed_reduce_sum_wrapping<T>(
    input: &TypedTensor<T>,
    axes: &[usize],
    exec_context: &ExecContext,
) -> crate::Result<TypedTensor<T>>
where
    T: WrappingReductionElem + TensorScalar,
{
    typed_reduce_erased(input, axes, ReduceOp::Sum, "reduce_sum", exec_context)
}

/// # Errors
///
/// Returns [`crate::Error::Validation`] with `AxisOutOfBounds`,
/// `DuplicateAxis`, or `InvalidArgument` for invalid axes or zero-length
/// reductions, or a typed backend error while materializing the result.
pub(crate) fn typed_reduce_prod<T>(
    input: &TypedTensor<T>,
    axes: &[usize],
    exec_context: &ExecContext,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Send + Sync + TensorScalar,
{
    typed_reduce_erased(input, axes, ReduceOp::Product, "reduce_prod", exec_context)
}

fn typed_reduce_prod_wrapping<T>(
    input: &TypedTensor<T>,
    axes: &[usize],
    exec_context: &ExecContext,
) -> crate::Result<TypedTensor<T>>
where
    T: WrappingReductionElem + TensorScalar,
{
    typed_reduce_erased(input, axes, ReduceOp::Product, "reduce_prod", exec_context)
}

/// # Errors
///
/// Returns [`crate::Error::Validation`] with `AxisOutOfBounds`,
/// `DuplicateAxis`, or `InvalidArgument` for invalid axes or zero-length
/// reductions, or a typed backend error while materializing the result.
pub fn typed_reduce_max<T>(input: &TypedTensor<T>, axes: &[usize]) -> crate::Result<TypedTensor<T>>
where
    T: Float + Send + Sync,
{
    validate_reduced_axes_nonempty("reduce_max", input.shape(), axes)?;
    typed_reduce(
        input,
        axes,
        |x| x,
        nan_propagating_max,
        T::neg_infinity(),
        "reduce_max",
    )
}

fn typed_reduce_max_integer<T>(
    input: &TypedTensor<T>,
    axes: &[usize],
) -> crate::Result<TypedTensor<T>>
where
    T: WrappingReductionElem,
{
    validate_reduced_axes_nonempty("reduce_max", input.shape(), axes)?;
    typed_reduce(
        input,
        axes,
        |x| x,
        |a, b| a.max_elem(b),
        T::min_value_elem(),
        "reduce_max",
    )
}

/// # Errors
///
/// Returns [`crate::Error::Validation`] with `AxisOutOfBounds`,
/// `DuplicateAxis`, or `InvalidArgument` for invalid axes or zero-length
/// reductions, or a typed backend error while materializing the result.
pub fn typed_reduce_min<T>(input: &TypedTensor<T>, axes: &[usize]) -> crate::Result<TypedTensor<T>>
where
    T: Float + Send + Sync,
{
    validate_reduced_axes_nonempty("reduce_min", input.shape(), axes)?;
    typed_reduce(
        input,
        axes,
        |x| x,
        nan_propagating_min,
        T::infinity(),
        "reduce_min",
    )
}

fn typed_reduce_min_integer<T>(
    input: &TypedTensor<T>,
    axes: &[usize],
) -> crate::Result<TypedTensor<T>>
where
    T: WrappingReductionElem,
{
    validate_reduced_axes_nonempty("reduce_min", input.shape(), axes)?;
    typed_reduce(
        input,
        axes,
        |x| x,
        |a, b| a.min_elem(b),
        T::max_value_elem(),
        "reduce_min",
    )
}
