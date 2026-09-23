use std::mem::size_of_val;

use num_complex::{Complex32, Complex64};
use strided_kernel::{
    col_major_strides, reduce, ErasedReducePlan, ExecContext, KernelDType, ReduceOp,
};

use super::{typed_host_data, typed_view, typed_view_from_view, PooledUninitOutput};
use crate::buffer_pool::BufferPool;
use crate::elementwise;
use crate::materialize_tensor_read;
use tenferro_tensor::{
    DType, Tensor, TensorRank, TensorRead, TensorScalar, TensorView, TypedTensor, TypedTensorView,
};

fn unsupported_dtype_with_supported(
    op: &'static str,
    dtype: DType,
    supported: &'static str,
) -> crate::Error {
    crate::Error::unsupported(
        op,
        format!("unsupported dtype {dtype:?}; supported dtypes: {supported}"),
    )
}

fn unsupported_sum_squares_dtype(op: &'static str, dtype: DType) -> crate::Error {
    let remedy =
        matches!(dtype, DType::I32 | DType::I64).then_some("; convert to F64 before reduction");
    crate::Error::unsupported(
        op,
        format!(
            "unsupported dtype {dtype:?}; supported dtypes: F32/F64{}",
            remedy.unwrap_or_default()
        ),
    )
}

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
/// The typed host tensor behind `input`, or a typed refusal.
///
/// Callers reach this from a match on `input.dtype()`, so `None` means the tag table
/// and the runtime dtype disagree rather than a caller mistake.
fn typed_input<'a, T: TensorScalar>(
    op: &'static str,
    input: &'a Tensor,
) -> crate::Result<&'a TypedTensor<T>> {
    input.as_typed::<T>().ok_or_else(|| {
        crate::Error::unsupported_dtype(
            op,
            input.dtype(),
            "the CPU reduction requires a preset scalar",
        )
    })
}

fn ensure_host_tensor(op: &'static str, input: &Tensor) -> crate::Result<()> {
    macro_rules! ensure {
        ($tensor:expr) => {{
            typed_host_data(op, $tensor)?;
            Ok(())
        }};
    }

    match input.dtype() {
        DType::F32 => ensure!(typed_input::<f32>("ensure_host_tensor", input)?),
        DType::F64 => ensure!(typed_input::<f64>("ensure_host_tensor", input)?),
        DType::I32 => ensure!(typed_input::<i32>("ensure_host_tensor", input)?),
        DType::I64 => ensure!(typed_input::<i64>("ensure_host_tensor", input)?),
        DType::Bool => ensure!(typed_input::<bool>("ensure_host_tensor", input)?),
        DType::C32 => ensure!(typed_input::<Complex32>("ensure_host_tensor", input)?),
        DType::C64 => ensure!(typed_input::<Complex64>("ensure_host_tensor", input)?),
        // A caller-owned payload has no CPU implementation for this operation.
        DType::External(type_id) => Err(crate::Error::unsupported_dtype(
            "ensure_host_tensor",
            tenferro_tensor::DType::External(type_id),
            "an externally defined payload is not supported by this CPU operation",
        )),
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
    axes.is_empty().then(|| input.duplicate()).transpose()
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

fn kernel_dtype(dtype: DType) -> KernelDType {
    match dtype {
        DType::F32 => KernelDType::F32,
        DType::F64 => KernelDType::F64,
        DType::I32 => KernelDType::I32,
        DType::I64 => KernelDType::I64,
        DType::Bool => KernelDType::Bool,
        DType::C32 => KernelDType::C32,
        DType::C64 => KernelDType::C64,
        // INVARIANT: `KernelDType` is the fixed compiled-kernel vocabulary, and
        // callers reach this after rejecting unsupported dtypes. An externally
        // defined scalar has no entry in it.
        DType::External(_) => unreachable!("KernelDType covers the preset scalars"),
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

/// Integer dtypes whose erased sum and product reductions wrap on overflow.
trait WrappingReductionElem: Copy + Clone + Send + Sync + 'static {}

impl WrappingReductionElem for i32 {}
impl WrappingReductionElem for i64 {}

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

    match input.dtype() {
        DType::F32 => Ok(Tensor::from_typed::<f32>(typed_reduce_sum(
            typed_input::<f32>("reduce_sum", input)?,
            axes,
            exec_context,
        )?)),
        DType::F64 => Ok(Tensor::from_typed::<f64>(typed_reduce_sum(
            typed_input::<f64>("reduce_sum", input)?,
            axes,
            exec_context,
        )?)),
        DType::I32 => Ok(Tensor::from_typed::<i32>(typed_reduce_sum_wrapping(
            typed_input::<i32>("reduce_sum", input)?,
            axes,
            exec_context,
        )?)),
        DType::I64 => Ok(Tensor::from_typed::<i64>(typed_reduce_sum_wrapping(
            typed_input::<i64>("reduce_sum", input)?,
            axes,
            exec_context,
        )?)),
        DType::Bool => Err(unsupported_dtype_with_supported(
            "reduce_sum",
            DType::Bool,
            "F32/F64/I32/I64/C32/C64",
        )),
        DType::C32 => Ok(Tensor::from_typed::<Complex32>(typed_reduce_sum(
            typed_input::<Complex32>("reduce_sum", input)?,
            axes,
            exec_context,
        )?)),
        DType::C64 => Ok(Tensor::from_typed::<Complex64>(typed_reduce_sum(
            typed_input::<Complex64>("reduce_sum", input)?,
            axes,
            exec_context,
        )?)),
        // A caller-owned payload has no CPU implementation for this operation.
        DType::External(type_id) => Err(crate::Error::unsupported_dtype(
            "reduce_sum",
            tenferro_tensor::DType::External(type_id),
            "an externally defined payload is not supported by this CPU operation",
        )),
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
        TensorRead::View(TensorView::F32(t)) => Ok(Tensor::from_typed::<f32>(
            typed_reduce_view_erased(buffers, &t, axes, ReduceOp::Sum, "reduce_sum", exec_context)?,
        )),
        TensorRead::View(TensorView::F64(t)) => Ok(Tensor::from_typed::<f64>(
            typed_reduce_view_erased(buffers, &t, axes, ReduceOp::Sum, "reduce_sum", exec_context)?,
        )),
        TensorRead::View(TensorView::I32(t)) => Ok(Tensor::from_typed::<i32>(
            typed_reduce_view_erased(buffers, &t, axes, ReduceOp::Sum, "reduce_sum", exec_context)?,
        )),
        TensorRead::View(TensorView::I64(t)) => Ok(Tensor::from_typed::<i64>(
            typed_reduce_view_erased(buffers, &t, axes, ReduceOp::Sum, "reduce_sum", exec_context)?,
        )),
        TensorRead::View(TensorView::Bool(_)) => Err(unsupported_dtype_with_supported(
            "reduce_sum",
            DType::Bool,
            "F32/F64/I32/I64/C32/C64",
        )),
        TensorRead::View(TensorView::C32(t)) => Ok(Tensor::from_typed::<Complex32>(
            typed_reduce_view_erased(buffers, &t, axes, ReduceOp::Sum, "reduce_sum", exec_context)?,
        )),
        TensorRead::View(TensorView::C64(t)) => Ok(Tensor::from_typed::<Complex64>(
            typed_reduce_view_erased(buffers, &t, axes, ReduceOp::Sum, "reduce_sum", exec_context)?,
        )),
    }
}

/// # Errors
///
/// Returns a typed validation error for invalid axes or zero-length reduced
/// dimensions, `Unsupported` for non-real floating dtypes, or a typed backend
/// error while reading storage or executing the strided kernel.
pub(crate) fn reduce_sum_squares(
    buffers: &mut BufferPool,
    input: &Tensor,
    axes: &[usize],
    exec_context: &ExecContext,
) -> crate::Result<Tensor> {
    if !matches!(input.dtype(), DType::F32 | DType::F64) {
        return Err(unsupported_sum_squares_dtype(
            "reduce_sum_squares",
            input.dtype(),
        ));
    }
    validate_axes("reduce_sum_squares", axes, input.shape().len())?;
    if axes.is_empty() {
        return elementwise::mul_with_pool(buffers, exec_context, input, input);
    }

    match input.dtype() {
        DType::F32 => Ok(Tensor::from_typed::<f32>(typed_reduce_erased(
            input.as_typed::<f32>().ok_or_else(|| {
                unsupported_sum_squares_dtype("reduce_sum_squares", input.dtype())
            })?,
            axes,
            ReduceOp::SumSquares,
            "reduce_sum_squares",
            exec_context,
        )?)),
        DType::F64 => Ok(Tensor::from_typed::<f64>(typed_reduce_erased(
            input.as_typed::<f64>().ok_or_else(|| {
                unsupported_sum_squares_dtype("reduce_sum_squares", input.dtype())
            })?,
            axes,
            ReduceOp::SumSquares,
            "reduce_sum_squares",
            exec_context,
        )?)),
        _ => Err(unsupported_sum_squares_dtype(
            "reduce_sum_squares",
            input.dtype(),
        )),
    }
}

/// The typed tensor behind a read's tensor, or this module's refusal for one.
fn norm_squared_operand<T: tenferro_tensor::TensorScalar>(
    tensor: &Tensor,
) -> crate::Result<&TypedTensor<T>> {
    tensor.as_typed::<T>().ok_or_else(|| {
        unsupported_sum_squares_dtype("BackendSession::norm_squared_read", tensor.dtype())
    })
}

pub(crate) fn norm_squared_read(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
) -> crate::Result<Tensor> {
    match input {
        TensorRead::Tensor(tensor) => match tensor.dtype() {
            DType::F32 => {
                let input = norm_squared_operand::<f32>(tensor)?;
                let view = typed_view("BackendSession::norm_squared_read", input)?;
                let value = norm_squared_scalar(&view, |x| x * x, 0.0_f32)?;
                pooled_scalar_f32(buffers, value)
            }
            DType::F64 => {
                let input = norm_squared_operand::<f64>(tensor)?;
                let view = typed_view("BackendSession::norm_squared_read", input)?;
                let value = norm_squared_scalar(&view, |x| x * x, 0.0_f64)?;
                pooled_scalar_f64(buffers, value)
            }
            DType::C32 => {
                let input = norm_squared_operand::<Complex32>(tensor)?;
                let view = typed_view("BackendSession::norm_squared_read", input)?;
                let value = norm_squared_scalar(&view, |x| x.norm_sqr(), 0.0_f32)?;
                pooled_scalar_f32(buffers, value)
            }
            DType::C64 => {
                let input = norm_squared_operand::<Complex64>(tensor)?;
                let view = typed_view("BackendSession::norm_squared_read", input)?;
                let value = norm_squared_scalar(&view, |x| x.norm_sqr(), 0.0_f64)?;
                pooled_scalar_f64(buffers, value)
            }
            _ => Err(unsupported_sum_squares_dtype(
                "BackendSession::norm_squared_read",
                tensor.dtype(),
            )),
        },
        TensorRead::View(input) => match input {
            TensorView::F32(input) => {
                let view = typed_view_from_view("BackendSession::norm_squared_read", &input)?;
                let value = norm_squared_scalar(&view, |x| x * x, 0.0_f32)?;
                pooled_scalar_f32(buffers, value)
            }
            TensorView::F64(input) => {
                let view = typed_view_from_view("BackendSession::norm_squared_read", &input)?;
                let value = norm_squared_scalar(&view, |x| x * x, 0.0_f64)?;
                pooled_scalar_f64(buffers, value)
            }
            TensorView::C32(input) => {
                let view = typed_view_from_view("BackendSession::norm_squared_read", &input)?;
                let value = norm_squared_scalar(&view, |x| x.norm_sqr(), 0.0_f32)?;
                pooled_scalar_f32(buffers, value)
            }
            TensorView::C64(input) => {
                let view = typed_view_from_view("BackendSession::norm_squared_read", &input)?;
                let value = norm_squared_scalar(&view, |x| x.norm_sqr(), 0.0_f64)?;
                pooled_scalar_f64(buffers, value)
            }
            _ => Err(unsupported_sum_squares_dtype(
                "BackendSession::norm_squared_read",
                input.dtype(),
            )),
        },
    }
}

fn norm_squared_scalar<T, U, M>(
    input: &strided_kernel::StridedView<'_, T>,
    map_fn: M,
    init: U,
) -> crate::Result<U>
where
    T: Copy + Send + Sync,
    U: Clone + Send + Sync + std::ops::Add<Output = U>,
    M: Fn(T) -> U + Copy + Send + Sync,
{
    reduce(input, map_fn, |lhs, rhs| lhs + rhs, init)
        .map_err(|err| crate::Error::backend_source("BackendSession::norm_squared_read", err))
}

fn pooled_scalar_f32(buffers: &mut BufferPool, value: f32) -> crate::Result<Tensor> {
    let mut output = PooledUninitOutput::<f32>::new(buffers, vec![])?;
    output.as_uninit_slice_mut()[0].write(value);
    // SAFETY: the rank-0 output has exactly one element, initialized above.
    unsafe { output.assume_init().map(Tensor::from_typed::<f32>) }
}

fn pooled_scalar_f64(buffers: &mut BufferPool, value: f64) -> crate::Result<Tensor> {
    let mut output = PooledUninitOutput::<f64>::new(buffers, vec![])?;
    output.as_uninit_slice_mut()[0].write(value);
    // SAFETY: the rank-0 output has exactly one element, initialized above.
    unsafe { output.assume_init().map(Tensor::from_typed::<f64>) }
}

pub(crate) fn reduce_sum_squares_read(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
    axes: &[usize],
    exec_context: &ExecContext,
) -> crate::Result<Tensor> {
    if !matches!(input.dtype(), DType::F32 | DType::F64) {
        return Err(unsupported_sum_squares_dtype(
            "reduce_sum_squares",
            input.dtype(),
        ));
    }
    validate_axes("reduce_sum_squares", axes, input.shape().len())?;
    if axes.is_empty() {
        let rhs = input.clone();
        return elementwise::mul_read_with_pool(buffers, exec_context, input, rhs);
    }

    match input {
        TensorRead::Tensor(input) => {
            ensure_host_tensor("reduce_sum_squares", input)?;
            reduce_sum_squares(buffers, input, axes, exec_context)
        }
        TensorRead::View(TensorView::F32(t)) => {
            Ok(Tensor::from_typed::<f32>(typed_reduce_view_erased(
                buffers,
                &t,
                axes,
                ReduceOp::SumSquares,
                "reduce_sum_squares",
                exec_context,
            )?))
        }
        TensorRead::View(TensorView::F64(t)) => {
            Ok(Tensor::from_typed::<f64>(typed_reduce_view_erased(
                buffers,
                &t,
                axes,
                ReduceOp::SumSquares,
                "reduce_sum_squares",
                exec_context,
            )?))
        }
        _ => Err(unsupported_sum_squares_dtype(
            "reduce_sum_squares",
            input.dtype(),
        )),
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

    match input.dtype() {
        DType::F32 => Ok(Tensor::from_typed::<f32>(typed_reduce_prod(
            typed_input::<f32>("reduce_prod", input)?,
            axes,
            exec_context,
        )?)),
        DType::F64 => Ok(Tensor::from_typed::<f64>(typed_reduce_prod(
            typed_input::<f64>("reduce_prod", input)?,
            axes,
            exec_context,
        )?)),
        DType::I32 => Ok(Tensor::from_typed::<i32>(typed_reduce_prod_wrapping(
            typed_input::<i32>("reduce_prod", input)?,
            axes,
            exec_context,
        )?)),
        DType::I64 => Ok(Tensor::from_typed::<i64>(typed_reduce_prod_wrapping(
            typed_input::<i64>("reduce_prod", input)?,
            axes,
            exec_context,
        )?)),
        DType::Bool => Err(unsupported_dtype_with_supported(
            "reduce_prod",
            DType::Bool,
            "F32/F64/I32/I64/C32/C64",
        )),
        DType::C32 => Ok(Tensor::from_typed::<Complex32>(typed_reduce_prod(
            typed_input::<Complex32>("reduce_prod", input)?,
            axes,
            exec_context,
        )?)),
        DType::C64 => Ok(Tensor::from_typed::<Complex64>(typed_reduce_prod(
            typed_input::<Complex64>("reduce_prod", input)?,
            axes,
            exec_context,
        )?)),
        // A caller-owned payload has no CPU implementation for this operation.
        DType::External(type_id) => Err(crate::Error::unsupported_dtype(
            "reduce_prod",
            tenferro_tensor::DType::External(type_id),
            "an externally defined payload is not supported by this CPU operation",
        )),
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
        TensorRead::View(TensorView::F32(t)) => {
            Ok(Tensor::from_typed::<f32>(typed_reduce_view_erased(
                buffers,
                &t,
                axes,
                ReduceOp::Product,
                "reduce_prod",
                exec_context,
            )?))
        }
        TensorRead::View(TensorView::F64(t)) => {
            Ok(Tensor::from_typed::<f64>(typed_reduce_view_erased(
                buffers,
                &t,
                axes,
                ReduceOp::Product,
                "reduce_prod",
                exec_context,
            )?))
        }
        TensorRead::View(TensorView::I32(t)) => {
            Ok(Tensor::from_typed::<i32>(typed_reduce_view_erased(
                buffers,
                &t,
                axes,
                ReduceOp::Product,
                "reduce_prod",
                exec_context,
            )?))
        }
        TensorRead::View(TensorView::I64(t)) => {
            Ok(Tensor::from_typed::<i64>(typed_reduce_view_erased(
                buffers,
                &t,
                axes,
                ReduceOp::Product,
                "reduce_prod",
                exec_context,
            )?))
        }
        TensorRead::View(TensorView::Bool(_)) => Err(unsupported_dtype_with_supported(
            "reduce_prod",
            DType::Bool,
            "F32/F64/I32/I64/C32/C64",
        )),
        TensorRead::View(TensorView::C32(t)) => {
            Ok(Tensor::from_typed::<Complex32>(typed_reduce_view_erased(
                buffers,
                &t,
                axes,
                ReduceOp::Product,
                "reduce_prod",
                exec_context,
            )?))
        }
        TensorRead::View(TensorView::C64(t)) => {
            Ok(Tensor::from_typed::<Complex64>(typed_reduce_view_erased(
                buffers,
                &t,
                axes,
                ReduceOp::Product,
                "reduce_prod",
                exec_context,
            )?))
        }
    }
}

/// # Errors
///
/// Returns [`crate::Error::Validation`] with `AxisOutOfBounds`,
/// `DuplicateAxis`, or `InvalidArgument` for invalid axes or zero-length
/// reductions, [`crate::Error::Unsupported`] for `Bool` and complex dtypes, or
/// a typed backend error when the input storage cannot be read.
pub(crate) fn reduce_max(
    input: &Tensor,
    axes: &[usize],
    exec_context: &ExecContext,
) -> crate::Result<Tensor> {
    if let Some(output) = reduction_empty_axes_noop("reduce_max", input, axes)? {
        return Ok(output);
    }

    match input.dtype() {
        DType::F32 => Ok(Tensor::from_typed::<f32>(typed_reduce_erased(
            typed_input::<f32>("reduce_max", input)?,
            axes,
            ReduceOp::Max,
            "reduce_max",
            exec_context,
        )?)),
        DType::F64 => Ok(Tensor::from_typed::<f64>(typed_reduce_erased(
            typed_input::<f64>("reduce_max", input)?,
            axes,
            ReduceOp::Max,
            "reduce_max",
            exec_context,
        )?)),
        DType::I32 => Ok(Tensor::from_typed::<i32>(typed_reduce_erased(
            typed_input::<i32>("reduce_max", input)?,
            axes,
            ReduceOp::Max,
            "reduce_max",
            exec_context,
        )?)),
        DType::I64 => Ok(Tensor::from_typed::<i64>(typed_reduce_erased(
            typed_input::<i64>("reduce_max", input)?,
            axes,
            ReduceOp::Max,
            "reduce_max",
            exec_context,
        )?)),
        DType::Bool | DType::C32 | DType::C64 => Err(unsupported_dtype_with_supported(
            "reduce_max",
            input.dtype(),
            "F32/F64/I32/I64",
        )),
        // A caller-owned payload has no CPU implementation for this operation.
        DType::External(type_id) => Err(crate::Error::unsupported_dtype(
            "reduce_max",
            tenferro_tensor::DType::External(type_id),
            "an externally defined payload is not supported by this CPU operation",
        )),
    }
}

pub(crate) fn reduce_max_read(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
    axes: &[usize],
    exec_context: &ExecContext,
) -> crate::Result<Tensor> {
    if let Some(output) = reduction_read_empty_axes_noop(buffers, "reduce_max", &input, axes)? {
        return Ok(output);
    }

    match input {
        TensorRead::Tensor(input) => {
            ensure_host_tensor("reduce_max", input)?;
            reduce_max(input, axes, exec_context)
        }
        TensorRead::View(TensorView::F32(t)) => Ok(Tensor::from_typed::<f32>(
            typed_reduce_view_erased(buffers, &t, axes, ReduceOp::Max, "reduce_max", exec_context)?,
        )),
        TensorRead::View(TensorView::F64(t)) => Ok(Tensor::from_typed::<f64>(
            typed_reduce_view_erased(buffers, &t, axes, ReduceOp::Max, "reduce_max", exec_context)?,
        )),
        TensorRead::View(TensorView::I32(t)) => Ok(Tensor::from_typed::<i32>(
            typed_reduce_view_erased(buffers, &t, axes, ReduceOp::Max, "reduce_max", exec_context)?,
        )),
        TensorRead::View(TensorView::I64(t)) => Ok(Tensor::from_typed::<i64>(
            typed_reduce_view_erased(buffers, &t, axes, ReduceOp::Max, "reduce_max", exec_context)?,
        )),
        view => Err(unsupported_dtype_with_supported(
            "reduce_max",
            view.dtype(),
            "F32/F64/I32/I64",
        )),
    }
}

/// # Errors
///
/// Returns [`crate::Error::Validation`] with `AxisOutOfBounds`,
/// `DuplicateAxis`, or `InvalidArgument` for invalid axes or zero-length
/// reductions, [`crate::Error::Unsupported`] for `Bool` and complex dtypes, or
/// a typed backend error when the input storage cannot be read.
pub(crate) fn reduce_min(
    input: &Tensor,
    axes: &[usize],
    exec_context: &ExecContext,
) -> crate::Result<Tensor> {
    if let Some(output) = reduction_empty_axes_noop("reduce_min", input, axes)? {
        return Ok(output);
    }

    match input.dtype() {
        DType::F32 => Ok(Tensor::from_typed::<f32>(typed_reduce_erased(
            typed_input::<f32>("reduce_min", input)?,
            axes,
            ReduceOp::Min,
            "reduce_min",
            exec_context,
        )?)),
        DType::F64 => Ok(Tensor::from_typed::<f64>(typed_reduce_erased(
            typed_input::<f64>("reduce_min", input)?,
            axes,
            ReduceOp::Min,
            "reduce_min",
            exec_context,
        )?)),
        DType::I32 => Ok(Tensor::from_typed::<i32>(typed_reduce_erased(
            typed_input::<i32>("reduce_min", input)?,
            axes,
            ReduceOp::Min,
            "reduce_min",
            exec_context,
        )?)),
        DType::I64 => Ok(Tensor::from_typed::<i64>(typed_reduce_erased(
            typed_input::<i64>("reduce_min", input)?,
            axes,
            ReduceOp::Min,
            "reduce_min",
            exec_context,
        )?)),
        DType::Bool | DType::C32 | DType::C64 => Err(unsupported_dtype_with_supported(
            "reduce_min",
            input.dtype(),
            "F32/F64/I32/I64",
        )),
        // A caller-owned payload has no CPU implementation for this operation.
        DType::External(type_id) => Err(crate::Error::unsupported_dtype(
            "reduce_min",
            tenferro_tensor::DType::External(type_id),
            "an externally defined payload is not supported by this CPU operation",
        )),
    }
}

pub(crate) fn reduce_min_read(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
    axes: &[usize],
    exec_context: &ExecContext,
) -> crate::Result<Tensor> {
    if let Some(output) = reduction_read_empty_axes_noop(buffers, "reduce_min", &input, axes)? {
        return Ok(output);
    }

    match input {
        TensorRead::Tensor(input) => {
            ensure_host_tensor("reduce_min", input)?;
            reduce_min(input, axes, exec_context)
        }
        TensorRead::View(TensorView::F32(t)) => Ok(Tensor::from_typed::<f32>(
            typed_reduce_view_erased(buffers, &t, axes, ReduceOp::Min, "reduce_min", exec_context)?,
        )),
        TensorRead::View(TensorView::F64(t)) => Ok(Tensor::from_typed::<f64>(
            typed_reduce_view_erased(buffers, &t, axes, ReduceOp::Min, "reduce_min", exec_context)?,
        )),
        TensorRead::View(TensorView::I32(t)) => Ok(Tensor::from_typed::<i32>(
            typed_reduce_view_erased(buffers, &t, axes, ReduceOp::Min, "reduce_min", exec_context)?,
        )),
        TensorRead::View(TensorView::I64(t)) => Ok(Tensor::from_typed::<i64>(
            typed_reduce_view_erased(buffers, &t, axes, ReduceOp::Min, "reduce_min", exec_context)?,
        )),
        view => Err(unsupported_dtype_with_supported(
            "reduce_min",
            view.dtype(),
            "F32/F64/I32/I64",
        )),
    }
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
        return input.duplicate();
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
    let source = unsafe {
        // SAFETY: this operation supplies initialized typed bytes with
        // alignment and dtype matching the descriptor; validated dimensions,
        // strides, and offset keep every reachable source element in bounds
        // for the retained borrow.
        crate::erased_raw_strided_ref(
            dtype,
            typed_bytes(input_view.data()),
            input_view.dims(),
            input_view.strides(),
            input_view.offset(),
        )
    }
    .map_err(|err| crate::Error::backend_source(label, err))?;
    // SAFETY: ErasedReducePlan writes every destination element.
    let mut output = unsafe { uninit_full_overwrite_vec(output_len) };
    let mut dest = crate::erased_raw_strided_mut(
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
    let source = unsafe {
        // SAFETY: this operation supplies initialized typed bytes with
        // alignment and dtype matching the descriptor; validated dimensions,
        // strides, and offset keep every reachable source element in bounds
        // for the retained borrow.
        crate::erased_raw_strided_ref(
            dtype,
            typed_bytes(input_view.data()),
            input_view.dims(),
            input_view.strides(),
            input_view.offset(),
        )
    }
    .map_err(|err| crate::Error::backend_source(label, err))?;
    let mut output = PooledUninitOutput::<T>::new(buffers, output_shape.clone())?;
    let mut dest = unsafe {
        // SAFETY: the output guard exclusively owns aligned storage whose
        // byte layout agrees with dtype; validated dimensions, strides, and
        // offset keep every destination in bounds, and the following kernel
        // overwrites every reachable element before typed exposure.
        crate::erased_raw_strided_uninit_mut(
            dtype,
            output.as_uninit_bytes_mut(),
            &output_shape,
            &output_strides,
            0,
        )
    }
    .map_err(|err| crate::Error::backend_source(label, err))?;
    let source_ptr = strided_kernel::ErasedRawStridedPtr::from_ref(&source);
    plan.execute_uninit(exec_context, &mut dest, &source_ptr)
        .map_err(|err| crate::Error::backend_source(label, err))?;

    // SAFETY: the reduction plan writes every logical destination element.
    unsafe { output.assume_init() }
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
    // INVARIANT: the pinned strided-kernel `ErasedReduceScalar` implementation
    // for i32/i64 uses `wrapping_add`; the CPU overflow regression test pins
    // this delegated two's-complement contract.
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
    // INVARIANT: the pinned strided-kernel `ErasedReduceScalar` implementation
    // for i32/i64 uses `wrapping_mul`; the CPU overflow regression test pins
    // this delegated two's-complement contract.
    typed_reduce_erased(input, axes, ReduceOp::Product, "reduce_prod", exec_context)
}

#[cfg(test)]
mod tests;
