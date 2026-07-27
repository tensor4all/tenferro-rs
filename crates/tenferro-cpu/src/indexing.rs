use std::mem::size_of_val;

use num_traits::Zero;
use strided_kernel::{
    col_major_strides, ErasedDynamicSlicePlan, ErasedDynamicUpdateSlicePlan, ErasedGatherPlan,
    ErasedRawStridedMut, ErasedRawStridedRef, ErasedScatterPlan, ExecContext, GatherSpec,
    KernelDType, ScatterSpec,
};

use super::indexing_alloc::pooled_uninit_tensor;
use super::typed_host_data;
use crate::buffer_pool::{BufferPool, PoolScalar};
use tenferro_tensor::TensorScalar;
use tenferro_tensor::{DType, GatherConfig, PadConfig, ScatterConfig, SliceConfig};
use tenferro_tensor::{Tensor, TypedTensor};

// Indexed gather, additive scatter, and fixed-window dynamic slice/update
// delegate bulk traversal to strided-kernel erased plans. Pad/concatenate/
// reverse still own their operation-specific loops here. Backend entrypoints
// run these kernels inside CpuContext::install, so future parallel
// implementations can use the same CPU threading policy.

trait TensorAsTyped<T> {
    fn as_typed(&self) -> Option<&TypedTensor<T>>;
}

macro_rules! impl_tensor_as_typed {
    ($(($ty:ty, $variant:ident)),+ $(,)?) => {
        $(
            impl TensorAsTyped<$ty> for Tensor {
                fn as_typed(&self) -> Option<&TypedTensor<$ty>> {
                    match self {
                        Tensor::$variant(tensor) => Some(tensor),
                        _ => None,
                    }
                }
            }
        )+
    };
}

impl_tensor_as_typed!(
    (f32, F32),
    (f64, F64),
    (i32, I32),
    (i64, I64),
    (bool, Bool),
    (num_complex::Complex<f32>, C32),
    (num_complex::Complex<f64>, C64),
);

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
    // SAFETY: `data` is an aligned typed slice. The returned byte slice has
    // the same lifetime and exact byte length; callers must preserve the
    // dtype's byte validity invariants before typed reads resume.
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<u8>(), size_of_val(data)) }
}

fn gather_spec(config: &GatherConfig) -> GatherSpec {
    GatherSpec {
        offset_dims: config.offset_dims.clone(),
        collapsed_slice_dims: config.collapsed_slice_dims.clone(),
        start_index_map: config.start_index_map.clone(),
        index_vector_dim: config.index_vector_dim,
        slice_sizes: config.slice_sizes.clone(),
    }
}

fn scatter_spec(config: &ScatterConfig) -> ScatterSpec {
    ScatterSpec {
        update_window_dims: config.update_window_dims.clone(),
        inserted_window_dims: config.inserted_window_dims.clone(),
        scatter_dims_to_operand_dims: config.scatter_dims_to_operand_dims.clone(),
        index_vector_dim: config.index_vector_dim,
    }
}

macro_rules! dispatch_tensor_unary_result {
    ($input:expr, |$tensor:ident| $body:expr) => {
        match $input {
            Tensor::F32($tensor) => Ok(Tensor::F32($body?)),
            Tensor::F64($tensor) => Ok(Tensor::F64($body?)),
            Tensor::I32($tensor) => Ok(Tensor::I32($body?)),
            Tensor::I64($tensor) => Ok(Tensor::I64($body?)),
            Tensor::Bool($tensor) => Ok(Tensor::Bool($body?)),
            Tensor::C32($tensor) => Ok(Tensor::C32($body?)),
            Tensor::C64($tensor) => Ok(Tensor::C64($body?)),
        }
    };
}

macro_rules! dispatch_tensor_unary_with_bool_special_result {
    ($input:expr, |$tensor:ident| $body:expr, bool |$bool_tensor:ident| $bool_body:expr) => {
        match $input {
            Tensor::F32($tensor) => Ok(Tensor::F32($body?)),
            Tensor::F64($tensor) => Ok(Tensor::F64($body?)),
            Tensor::I32($tensor) => Ok(Tensor::I32($body?)),
            Tensor::I64($tensor) => Ok(Tensor::I64($body?)),
            Tensor::Bool($bool_tensor) => Ok(Tensor::Bool($bool_body?)),
            Tensor::C32($tensor) => Ok(Tensor::C32($body?)),
            Tensor::C64($tensor) => Ok(Tensor::C64($body?)),
        }
    };
}

macro_rules! dispatch_same_dtype_result {
    ($op:literal, $lhs:expr, $rhs:expr, |$lhs_t:ident, $rhs_t:ident| $body:expr) => {
        match ($lhs, $rhs) {
            (Tensor::F32($lhs_t), Tensor::F32($rhs_t)) => Ok(Tensor::F32($body?)),
            (Tensor::F64($lhs_t), Tensor::F64($rhs_t)) => Ok(Tensor::F64($body?)),
            (Tensor::I32($lhs_t), Tensor::I32($rhs_t)) => Ok(Tensor::I32($body?)),
            (Tensor::I64($lhs_t), Tensor::I64($rhs_t)) => Ok(Tensor::I64($body?)),
            (Tensor::Bool($lhs_t), Tensor::Bool($rhs_t)) => Ok(Tensor::Bool($body?)),
            (Tensor::C32($lhs_t), Tensor::C32($rhs_t)) => Ok(Tensor::C32($body?)),
            (Tensor::C64($lhs_t), Tensor::C64($rhs_t)) => Ok(Tensor::C64($body?)),
            _ => Err(crate::Error::dtype_mismatch(
                $op,
                $lhs.dtype(),
                $rhs.dtype(),
            )),
        }
    };
}

macro_rules! dispatch_same_dtype_without_bool_result {
    ($op:literal, $lhs:expr, $rhs:expr, $bool_message:literal, |$lhs_t:ident, $rhs_t:ident| $body:expr) => {
        match ($lhs, $rhs) {
            (Tensor::F32($lhs_t), Tensor::F32($rhs_t)) => Ok(Tensor::F32($body?)),
            (Tensor::F64($lhs_t), Tensor::F64($rhs_t)) => Ok(Tensor::F64($body?)),
            (Tensor::I32($lhs_t), Tensor::I32($rhs_t)) => Ok(Tensor::I32($body?)),
            (Tensor::I64($lhs_t), Tensor::I64($rhs_t)) => Ok(Tensor::I64($body?)),
            (Tensor::C32($lhs_t), Tensor::C32($rhs_t)) => Ok(Tensor::C32($body?)),
            (Tensor::C64($lhs_t), Tensor::C64($rhs_t)) => Ok(Tensor::C64($body?)),
            (Tensor::Bool(_), Tensor::Bool(_)) => {
                Err(crate::Error::unsupported($op, $bool_message))
            }
            _ => Err(crate::Error::dtype_mismatch(
                $op,
                $lhs.dtype(),
                $rhs.dtype(),
            )),
        }
    };
}

#[cfg(test)]
fn with_test_pool<T>(f: impl FnOnce(&mut BufferPool) -> T) -> T {
    let mut buffers = BufferPool::new();
    f(&mut buffers)
}

fn advance_col_major_index(index: &mut [usize], shape: &[usize]) {
    debug_assert_eq!(index.len(), shape.len());
    for axis in 0..index.len() {
        if shape[axis] == 0 {
            index[axis] = 0;
            continue;
        }
        index[axis] += 1;
        if index[axis] < shape[axis] {
            break;
        }
        index[axis] = 0;
    }
}

fn pooled_filled_tensor<T>(
    buffers: &mut BufferPool,
    shape: Vec<usize>,
    fill: T,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + PoolScalar,
{
    // SAFETY: the following fill writes every pooled output element.
    let mut out = pooled_uninit_tensor(buffers, shape)?;
    out.host_data_mut()?.fill(fill);
    Ok(out)
}

#[cfg(test)]
pub(crate) fn gather(
    operand: &Tensor,
    start_indices: &Tensor,
    config: &GatherConfig,
) -> crate::Result<Tensor> {
    with_test_pool(|buffers| {
        gather_with_pool(
            buffers,
            &ExecContext::serial(),
            operand,
            start_indices,
            config,
        )
    })
}

pub(crate) fn gather_with_pool(
    buffers: &mut BufferPool,
    exec_context: &ExecContext,
    operand: &Tensor,
    start_indices: &Tensor,
    config: &GatherConfig,
) -> crate::Result<Tensor> {
    let start_indices = try_index_tensor(start_indices)?;
    dispatch_tensor_unary_result!(operand, |t| typed_gather(
        buffers,
        exec_context,
        t,
        &start_indices,
        config
    ))
}

#[cfg(test)]
pub(crate) fn scatter(
    operand: &Tensor,
    scatter_indices: &Tensor,
    updates: &Tensor,
    config: &ScatterConfig,
) -> crate::Result<Tensor> {
    with_test_pool(|buffers| {
        scatter_with_pool(
            buffers,
            &ExecContext::serial(),
            operand,
            scatter_indices,
            updates,
            config,
        )
    })
}

pub(crate) fn scatter_with_pool(
    buffers: &mut BufferPool,
    exec_context: &ExecContext,
    operand: &Tensor,
    scatter_indices: &Tensor,
    updates: &Tensor,
    config: &ScatterConfig,
) -> crate::Result<Tensor> {
    let scatter_indices = try_index_tensor(scatter_indices)?;
    dispatch_same_dtype_without_bool_result!(
        "scatter",
        operand,
        updates,
        "Bool data tensors are not supported by additive scatter",
        |op, upd| typed_scatter(buffers, exec_context, op, &scatter_indices, upd, config)
    )
}

pub(crate) fn try_slice_with_pool(
    buffers: &mut BufferPool,
    input: &Tensor,
    config: &SliceConfig,
) -> crate::Result<Tensor> {
    dispatch_tensor_unary_result!(input, |t| typed_slice(buffers, t, config))
}

#[cfg(test)]
pub(crate) fn dynamic_slice(
    input: &Tensor,
    starts: &Tensor,
    slice_sizes: &[usize],
) -> crate::Result<Tensor> {
    with_test_pool(|buffers| {
        dynamic_slice_with_pool(buffers, &ExecContext::serial(), input, starts, slice_sizes)
    })
}

pub(crate) fn dynamic_slice_with_pool(
    buffers: &mut BufferPool,
    exec_context: &ExecContext,
    input: &Tensor,
    starts: &Tensor,
    slice_sizes: &[usize],
) -> crate::Result<Tensor> {
    let starts = try_index_tensor(starts)?;
    dispatch_tensor_unary_result!(input, |t| typed_dynamic_slice(
        buffers,
        exec_context,
        t,
        &starts,
        slice_sizes
    ))
}

/// Return `operand` with `update` written at dynamic `starts`.
///
/// Starts are clamped so the whole update window fits inside the operand,
/// matching `dynamic_slice` behavior.
///
/// # Examples
///
/// ```
/// use tenferro_cpu as cpu;
/// use tenferro_tensor::{Tensor, TypedTensor};
///
/// let operand = Tensor::F64(TypedTensor::from_vec_col_major(vec![5], vec![0.0; 5])?);
/// let update = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![3.0, 4.0])?);
/// let starts = Tensor::I64(TypedTensor::from_vec_col_major(vec![1], vec![4])?);
///
/// let out = cpu::dynamic_update_slice(&operand, &update, &starts).unwrap();
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[0.0, 0.0, 0.0, 3.0, 4.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[cfg(test)]
pub(crate) fn dynamic_update_slice(
    operand: &Tensor,
    update: &Tensor,
    starts: &Tensor,
) -> crate::Result<Tensor> {
    with_test_pool(|buffers| {
        dynamic_update_slice_with_pool(buffers, &ExecContext::serial(), operand, update, starts)
    })
}

pub(crate) fn dynamic_update_slice_with_pool(
    buffers: &mut BufferPool,
    exec_context: &ExecContext,
    operand: &Tensor,
    update: &Tensor,
    starts: &Tensor,
) -> crate::Result<Tensor> {
    let starts = try_index_tensor(starts)?;
    dispatch_same_dtype_result!("dynamic_update_slice", operand, update, |op, upd| {
        typed_dynamic_update_slice(buffers, exec_context, op, upd, &starts)
    })
}

#[cfg(test)]
pub(crate) fn pad(input: &Tensor, config: &PadConfig) -> crate::Result<Tensor> {
    try_pad(input, config)
}

#[cfg(test)]
fn try_pad(input: &Tensor, config: &PadConfig) -> crate::Result<Tensor> {
    with_test_pool(|buffers| try_pad_with_pool(buffers, input, config))
}

pub(crate) fn try_pad_with_pool(
    buffers: &mut BufferPool,
    input: &Tensor,
    config: &PadConfig,
) -> crate::Result<Tensor> {
    dispatch_tensor_unary_with_bool_special_result!(
        input,
        |t| typed_pad(buffers, t, config),
        bool | t | typed_pad_with_fill(buffers, t, config, false)
    )
}

pub(crate) fn try_concatenate_with_pool(
    buffers: &mut BufferPool,
    inputs: &[&Tensor],
    axis: usize,
) -> crate::Result<Tensor> {
    let first = inputs.first().copied().ok_or_else(|| {
        crate::Error::invalid_argument(
            "concatenate",
            "configuration",
            "concatenate requires at least one input",
        )
    })?;
    dispatch_tensor_unary_result!(first, |t| typed_concatenate_from_dyn_inputs(
        buffers, t, inputs, axis
    ))
}

pub(crate) fn reverse_with_pool(
    buffers: &mut BufferPool,
    input: &Tensor,
    axes: &[usize],
) -> crate::Result<Tensor> {
    dispatch_tensor_unary_result!(input, |t| typed_reverse(buffers, t, axes))
}

fn typed_slice<T: Copy + Clone + PoolScalar>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    config: &SliceConfig,
) -> crate::Result<TypedTensor<T>> {
    let input_shape = input.shape();
    let rank = input_shape.len();
    if config.starts.len() != rank {
        return Err(crate::Error::rank_mismatch(
            "slice",
            rank,
            config.starts.len(),
        ));
    }
    if config.limits.len() != rank {
        return Err(crate::Error::rank_mismatch(
            "slice",
            rank,
            config.limits.len(),
        ));
    }
    if config.strides.len() != rank {
        return Err(crate::Error::rank_mismatch(
            "slice",
            rank,
            config.strides.len(),
        ));
    }

    let out_shape: Vec<usize> = input
        .shape()
        .iter()
        .enumerate()
        .map(|(axis, &dim)| {
            let start = config.starts[axis];
            let limit = config.limits[axis];
            let stride = config.strides[axis];
            if start > limit {
                return Err(crate::Error::invalid_argument(
                    "slice",
                    "configuration",
                    format!("start exceeds limit on axis {axis}"),
                ));
            }
            if limit > dim {
                return Err(crate::Error::axis_out_of_bounds("slice", axis, rank));
            }
            if stride == 0 {
                return Err(crate::Error::invalid_argument(
                    "slice",
                    "configuration",
                    format!("stride must be positive on axis {axis}"),
                ));
            }
            let span = limit - start;
            Ok(span.div_ceil(stride))
        })
        .collect::<crate::Result<Vec<_>>>()?;

    // SAFETY: the slice loop below assigns every output coordinate exactly once.
    let mut out = pooled_uninit_tensor(buffers, out_shape.clone())?;
    let mut out_idx = vec![0usize; rank];
    let mut in_idx = vec![0usize; rank];

    for out_value in out.host_data_mut()?.iter_mut() {
        for axis in 0..rank {
            in_idx[axis] = config.starts[axis] + out_idx[axis] * config.strides[axis];
        }
        *out_value = *input.get(&in_idx)?;
        advance_col_major_index(&mut out_idx, &out_shape);
    }

    Ok(out)
}

fn typed_concatenate_from_dyn_inputs<T>(
    buffers: &mut BufferPool,
    _first: &TypedTensor<T>,
    inputs: &[&Tensor],
    axis: usize,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + PoolScalar,
    Tensor: TensorAsTyped<T>,
{
    let first_dtype = inputs
        .first()
        .ok_or_else(|| {
            crate::Error::invalid_argument(
                "concatenate",
                "configuration",
                "concatenate requires at least one input",
            )
        })?
        .dtype();
    let typed_inputs = collect_typed_inputs(first_dtype, inputs)?;
    typed_concatenate(buffers, &typed_inputs, axis)
}

fn collect_typed_inputs<'a, T>(
    first_dtype: crate::DType,
    inputs: &[&'a Tensor],
) -> crate::Result<Vec<&'a TypedTensor<T>>>
where
    Tensor: TensorAsTyped<T>,
{
    inputs
        .iter()
        .map(|tensor| {
            TensorAsTyped::<T>::as_typed(*tensor).ok_or_else(|| {
                crate::Error::dtype_mismatch("concatenate", first_dtype, tensor.dtype())
            })
        })
        .collect()
}

fn typed_concatenate<T: Copy + Clone + PoolScalar>(
    buffers: &mut BufferPool,
    inputs: &[&TypedTensor<T>],
    axis: usize,
) -> crate::Result<TypedTensor<T>> {
    let first = inputs.first().copied().ok_or_else(|| {
        crate::Error::invalid_argument(
            "concatenate",
            "configuration",
            "concatenate requires at least one input",
        )
    })?;
    let first_shape = first.shape();
    let rank = first_shape.len();
    if axis >= rank {
        return Err(crate::Error::axis_out_of_bounds("concatenate", axis, rank));
    }

    let mut out_shape = first_shape.to_vec();
    let mut axis_extent = 0usize;
    for input in inputs {
        let input_shape = input.shape();
        if input_shape.len() != rank {
            return Err(crate::Error::rank_mismatch(
                "concatenate",
                rank,
                input_shape.len(),
            ));
        }
        for dim in 0..rank {
            if dim == axis {
                axis_extent = axis_extent.checked_add(input_shape[dim]).ok_or_else(|| {
                    crate::Error::invalid_argument(
                        "concatenate",
                        "configuration",
                        "concatenate axis extent overflows usize".to_string(),
                    )
                })?;
            } else if input_shape[dim] != first_shape[dim] {
                return Err(crate::Error::shape_mismatch(
                    "concatenate",
                    first_shape.to_vec(),
                    input_shape.to_vec(),
                ));
            }
        }
    }
    out_shape[axis] = axis_extent;

    let mut segment_ends = Vec::with_capacity(inputs.len());
    let mut segment_end = 0usize;
    for input in inputs {
        segment_end = segment_end
            .checked_add(input.shape()[axis])
            .ok_or_else(|| {
                crate::Error::invalid_argument(
                    "concatenate",
                    "configuration",
                    "concatenate segment offset overflows usize".to_string(),
                )
            })?;
        segment_ends.push(segment_end);
    }

    // SAFETY: the concatenate loop below assigns every output coordinate exactly once.
    let mut out = pooled_uninit_tensor(buffers, out_shape.clone())?;
    let mut out_idx = vec![0usize; rank];
    let mut in_idx = vec![0usize; rank];

    for out_value in out.host_data_mut()?.iter_mut() {
        let concat_idx = out_idx[axis];
        let input_pos = segment_ends.partition_point(|&end| concat_idx >= end);
        if input_pos == segment_ends.len() {
            return Err(crate::Error::invalid_argument(
                "concatenate",
                "configuration",
                "output index must map to an input".to_string(),
            ));
        }
        let axis_base = if input_pos == 0 {
            0
        } else {
            segment_ends[input_pos - 1]
        };

        in_idx.copy_from_slice(&out_idx);
        in_idx[axis] -= axis_base;
        *out_value = *inputs[input_pos].get(&in_idx)?;
        advance_col_major_index(&mut out_idx, &out_shape);
    }

    Ok(out)
}

fn typed_reverse<T: Copy + Clone + PoolScalar>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    axes: &[usize],
) -> crate::Result<TypedTensor<T>> {
    let input_shape = input.shape();
    let rank = input_shape.len();
    let mut reverse_axis = vec![false; rank];
    for &axis in axes {
        if axis >= rank {
            return Err(crate::Error::axis_out_of_bounds("reverse", axis, rank));
        }
        reverse_axis[axis] = true;
    }

    // SAFETY: the reverse loop below assigns every output coordinate exactly once.
    let mut out = pooled_uninit_tensor(buffers, input_shape.to_vec())?;
    let mut out_idx = vec![0usize; rank];
    let mut in_idx = vec![0usize; rank];

    for out_value in out.host_data_mut()?.iter_mut() {
        for axis in 0..rank {
            in_idx[axis] = if reverse_axis[axis] {
                input_shape[axis] - 1 - out_idx[axis]
            } else {
                out_idx[axis]
            };
        }
        *out_value = *input.get(&in_idx)?;
        advance_col_major_index(&mut out_idx, input_shape);
    }

    Ok(out)
}

struct IndexTensor {
    shape: Vec<usize>,
    values: Vec<i64>,
}

/// Maximum exact integer representable by f32 (2^24).
const F32_MAX_EXACT_INT: f32 = 16_777_216.0;
/// Maximum exact integer representable by f64 (2^53).
const F64_MAX_EXACT_INT: f64 = 9_007_199_254_740_992.0;

fn f32_index_to_i64(value: f32) -> crate::Result<i64> {
    if !value.is_finite() || value.fract() != 0.0 || value.abs() > F32_MAX_EXACT_INT {
        return Err(crate::Error::invalid_argument(
            "index_tensor",
            "index",
            format!("index value {value} is not an exactly representable i64"),
        ));
    }
    Ok(value as i64)
}

fn f64_index_to_i64(value: f64) -> crate::Result<i64> {
    if !value.is_finite() || value.fract() != 0.0 || value.abs() > F64_MAX_EXACT_INT {
        return Err(crate::Error::invalid_argument(
            "index_tensor",
            "index",
            format!("index value {value} is not an exactly representable i64"),
        ));
    }
    Ok(value as i64)
}

fn try_index_tensor(tensor: &Tensor) -> crate::Result<IndexTensor> {
    match tensor {
        Tensor::I32(t) => Ok(IndexTensor {
            shape: t.shape().to_vec(),
            values: typed_host_data("index_tensor", t)?
                .iter()
                .map(|&value| value as i64)
                .collect(),
        }),
        Tensor::I64(t) => Ok(IndexTensor {
            shape: t.shape().to_vec(),
            values: typed_host_data("index_tensor", t)?.to_vec(),
        }),
        Tensor::F32(t) => {
            let values: crate::Result<Vec<i64>> = typed_host_data("index_tensor", t)?
                .iter()
                .map(|&value| f32_index_to_i64(value))
                .collect();
            Ok(IndexTensor {
                shape: t.shape().to_vec(),
                values: values?,
            })
        }
        Tensor::F64(t) => {
            let values: crate::Result<Vec<i64>> = typed_host_data("index_tensor", t)?
                .iter()
                .map(|&value| f64_index_to_i64(value))
                .collect();
            Ok(IndexTensor {
                shape: t.shape().to_vec(),
                values: values?,
            })
        }
        Tensor::Bool(_) => Err(crate::Error::invalid_argument(
            "index_tensor",
            "configuration",
            "bool index tensors are not supported",
        )),
        Tensor::C32(_) | Tensor::C64(_) => Err(crate::Error::invalid_argument(
            "index_tensor",
            "configuration",
            "complex index tensors are not supported",
        )),
    }
}

fn checked_product(op: &'static str, role: &'static str, shape: &[usize]) -> crate::Result<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            crate::Error::invalid_argument(
                op,
                "configuration",
                format!("{role} element count overflows usize"),
            )
        })
    })
}

fn try_index_vector_size(
    op: &'static str,
    shape: &[usize],
    index_vector_dim: usize,
) -> crate::Result<usize> {
    if index_vector_dim > shape.len() {
        return Err(crate::Error::axis_out_of_bounds(
            op,
            index_vector_dim,
            shape.len(),
        ));
    }
    Ok(if index_vector_dim == shape.len() {
        1
    } else {
        shape[index_vector_dim]
    })
}

fn try_index_batch_shape(
    op: &'static str,
    shape: &[usize],
    index_vector_dim: usize,
) -> crate::Result<Vec<usize>> {
    if index_vector_dim > shape.len() {
        return Err(crate::Error::axis_out_of_bounds(
            op,
            index_vector_dim,
            shape.len(),
        ));
    }
    if index_vector_dim == shape.len() {
        return Ok(shape.to_vec());
    }
    Ok(shape
        .iter()
        .enumerate()
        .filter_map(|(axis, &dim)| (axis != index_vector_dim).then_some(dim))
        .collect())
}

fn clamp_window_start(
    op: &'static str,
    start: i64,
    dim_size: usize,
    window_size: usize,
) -> crate::Result<usize> {
    if window_size > dim_size {
        return Err(crate::Error::invalid_argument(
            op,
            "configuration",
            format!("window size {window_size} exceeds dimension size {dim_size}"),
        ));
    }
    let max_start = dim_size.saturating_sub(window_size) as i64;
    Ok(start.clamp(0, max_start) as usize)
}

fn operand_window_dims(rank: usize, collapsed_or_inserted: &[usize]) -> Vec<usize> {
    (0..rank)
        .filter(|dim| !collapsed_or_inserted.contains(dim))
        .collect()
}

fn typed_gather<T: Copy + Clone + PoolScalar + TensorScalar>(
    buffers: &mut BufferPool,
    exec_context: &ExecContext,
    operand: &TypedTensor<T>,
    start_indices: &IndexTensor,
    config: &GatherConfig,
) -> crate::Result<TypedTensor<T>> {
    let operand_shape = operand.shape();
    let rank = operand_shape.len();
    if config.slice_sizes.len() != rank {
        return Err(crate::Error::rank_mismatch(
            "gather",
            rank,
            config.slice_sizes.len(),
        ));
    }

    for &dim in &config.collapsed_slice_dims {
        if dim >= rank {
            return Err(crate::Error::axis_out_of_bounds("gather", dim, rank));
        }
    }
    {
        let mut seen = vec![false; rank];
        for &dim in &config.collapsed_slice_dims {
            if seen[dim] {
                return Err(crate::Error::duplicate_axis(
                    "gather",
                    dim,
                    "collapsed_slice_dims",
                ));
            }
            seen[dim] = true;
        }
    }
    for &dim in &config.collapsed_slice_dims {
        if config.slice_sizes[dim] != 1 {
            return Err(crate::Error::invalid_argument(
                "gather",
                "configuration",
                format!(
                    "collapsed slice dimension {dim} must have slice_size == 1, got {}",
                    config.slice_sizes[dim]
                ),
            ));
        }
    }

    let index_size =
        try_index_vector_size("gather", &start_indices.shape, config.index_vector_dim)?;
    if index_size != config.start_index_map.len() {
        return Err(crate::Error::invalid_argument(
            "gather",
            "configuration",
            format!(
                "start_index_map length {} does not match index vector size {}",
                config.start_index_map.len(),
                index_size
            ),
        ));
    }
    for &operand_dim in &config.start_index_map {
        if operand_dim >= rank {
            return Err(crate::Error::axis_out_of_bounds(
                "gather",
                operand_dim,
                rank,
            ));
        }
    }
    {
        let mut seen = vec![false; rank];
        for &operand_dim in &config.start_index_map {
            if seen[operand_dim] {
                return Err(crate::Error::duplicate_axis(
                    "gather",
                    operand_dim,
                    "start_index_map",
                ));
            }
            seen[operand_dim] = true;
        }
    }

    let window_dims = operand_window_dims(rank, &config.collapsed_slice_dims);
    if config.offset_dims.len() != window_dims.len() {
        return Err(crate::Error::invalid_argument(
            "gather",
            "configuration",
            format!(
                "offset_dims length {} does not match window dims count {}",
                config.offset_dims.len(),
                window_dims.len()
            ),
        ));
    }

    let batch_shape =
        try_index_batch_shape("gather", &start_indices.shape, config.index_vector_dim)?;
    let out_rank = batch_shape.len() + config.offset_dims.len();
    for &out_axis in &config.offset_dims {
        if out_axis >= out_rank {
            return Err(crate::Error::axis_out_of_bounds(
                "gather", out_axis, out_rank,
            ));
        }
    }
    {
        let mut seen = vec![false; out_rank];
        for &out_axis in &config.offset_dims {
            if seen[out_axis] {
                return Err(crate::Error::duplicate_axis(
                    "gather",
                    out_axis,
                    "offset_dims",
                ));
            }
            seen[out_axis] = true;
        }
    }

    let mut out_axis_to_operand_dim = vec![None; out_rank];
    for (offset_axis, &out_axis) in config.offset_dims.iter().enumerate() {
        out_axis_to_operand_dim[out_axis] = Some(window_dims[offset_axis]);
    }

    let mut out_shape = vec![0usize; out_rank];
    let mut batch_axis = 0usize;
    for out_axis in 0..out_rank {
        if let Some(operand_dim) = out_axis_to_operand_dim[out_axis] {
            out_shape[out_axis] = config.slice_sizes[operand_dim];
        } else {
            out_shape[out_axis] = batch_shape[batch_axis];
            batch_axis += 1;
        }
    }

    for (operand_dim, &dim_size) in operand_shape.iter().enumerate() {
        let _ = clamp_window_start("gather", 0, dim_size, config.slice_sizes[operand_dim])?;
    }

    let mut out = pooled_uninit_tensor(buffers, out_shape.clone())?;
    if T::dtype() == DType::Bool {
        out.host_data_mut()?.fill(T::pool_zero());
    }

    let dtype = kernel_dtype(T::dtype());
    let operand_strides = col_major_strides(operand_shape);
    let index_strides = col_major_strides(&start_indices.shape);
    let out_strides = col_major_strides(&out_shape);
    let plan = ErasedGatherPlan::compile(
        dtype,
        KernelDType::I64,
        operand_shape,
        &operand_strides,
        &start_indices.shape,
        &index_strides,
        &out_shape,
        &out_strides,
        gather_spec(config),
    )
    .map_err(|err| crate::Error::backend_source("gather", err))?;

    let operand_ref = ErasedRawStridedRef::new(
        dtype,
        typed_bytes(typed_host_data("gather", operand)?),
        operand_shape,
        &operand_strides,
        0,
    )
    .map_err(|err| crate::Error::backend_source("gather", err))?;
    let index_ref = ErasedRawStridedRef::new(
        KernelDType::I64,
        typed_bytes(&start_indices.values),
        &start_indices.shape,
        &index_strides,
        0,
    )
    .map_err(|err| crate::Error::backend_source("gather", err))?;
    let mut out_ref = ErasedRawStridedMut::new(
        dtype,
        typed_bytes_mut(out.host_data_mut()?),
        &out_shape,
        &out_strides,
        0,
    )
    .map_err(|err| crate::Error::backend_source("gather", err))?;
    plan.execute(exec_context, &mut out_ref, &operand_ref, &index_ref)
        .map_err(|err| crate::Error::backend_source("gather", err))?;

    Ok(out)
}

fn typed_scatter<T>(
    buffers: &mut BufferPool,
    exec_context: &ExecContext,
    operand: &TypedTensor<T>,
    scatter_indices: &IndexTensor,
    updates: &TypedTensor<T>,
    config: &ScatterConfig,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + PoolScalar + TensorScalar,
{
    let operand_shape = operand.shape();
    let updates_shape = updates.shape();
    let op_rank = operand_shape.len();
    for &dim in &config.inserted_window_dims {
        if dim >= op_rank {
            return Err(crate::Error::axis_out_of_bounds("scatter", dim, op_rank));
        }
    }
    {
        let mut seen = vec![false; op_rank];
        for &dim in &config.inserted_window_dims {
            if seen[dim] {
                return Err(crate::Error::duplicate_axis(
                    "scatter",
                    dim,
                    "inserted_window_dims",
                ));
            }
            seen[dim] = true;
        }
    }

    let index_size =
        try_index_vector_size("scatter", &scatter_indices.shape, config.index_vector_dim)?;
    if index_size != config.scatter_dims_to_operand_dims.len() {
        return Err(crate::Error::invalid_argument(
            "scatter",
            "configuration",
            format!(
                "scatter_dims_to_operand_dims length {} does not match index vector size {}",
                config.scatter_dims_to_operand_dims.len(),
                index_size
            ),
        ));
    }
    for &operand_dim in &config.scatter_dims_to_operand_dims {
        if operand_dim >= op_rank {
            return Err(crate::Error::axis_out_of_bounds(
                "scatter",
                operand_dim,
                op_rank,
            ));
        }
    }
    {
        let mut seen = vec![false; op_rank];
        for &operand_dim in &config.scatter_dims_to_operand_dims {
            if seen[operand_dim] {
                return Err(crate::Error::duplicate_axis(
                    "scatter",
                    operand_dim,
                    "scatter_dims_to_operand_dims",
                ));
            }
            seen[operand_dim] = true;
        }
    }

    let batch_shape =
        try_index_batch_shape("scatter", &scatter_indices.shape, config.index_vector_dim)?;
    let window_dims = operand_window_dims(op_rank, &config.inserted_window_dims);
    if config.update_window_dims.len() != window_dims.len() {
        return Err(crate::Error::invalid_argument(
            "scatter",
            "configuration",
            format!(
                "update_window_dims length {} does not match window dims count {}",
                config.update_window_dims.len(),
                window_dims.len()
            ),
        ));
    }

    let update_rank = updates_shape.len();
    let expected_batch_rank = update_rank
        .checked_sub(config.update_window_dims.len())
        .ok_or_else(|| {
            crate::Error::invalid_argument(
                "scatter",
                "configuration",
                format!(
                    "update_window_dims length {} exceeds update rank {}",
                    config.update_window_dims.len(),
                    update_rank
                ),
            )
        })?;
    if expected_batch_rank != batch_shape.len() {
        return Err(crate::Error::invalid_argument(
            "scatter",
            "configuration",
            format!(
                "updates batch rank {} does not match index batch shape length {}",
                expected_batch_rank,
                batch_shape.len()
            ),
        ));
    }

    for &axis in &config.update_window_dims {
        if axis >= update_rank {
            return Err(crate::Error::axis_out_of_bounds(
                "scatter",
                axis,
                update_rank,
            ));
        }
    }
    {
        let mut seen = vec![false; update_rank];
        for &axis in &config.update_window_dims {
            if seen[axis] {
                return Err(crate::Error::duplicate_axis(
                    "scatter",
                    axis,
                    "update_window_dims",
                ));
            }
            seen[axis] = true;
        }
    }

    let mut is_update_window_dim = vec![false; update_rank];
    for &axis in &config.update_window_dims {
        is_update_window_dim[axis] = true;
    }

    {
        let mut batch_axis = 0usize;
        for axis in 0..update_rank {
            if !is_update_window_dim[axis] {
                if updates_shape[axis] != batch_shape[batch_axis] {
                    return Err(crate::Error::invalid_argument("scatter", "configuration", format!(
                            "updates batch dim {} extent {} does not match index batch dim {} extent {}",
                            axis,
                            updates_shape[axis],
                            batch_axis,
                            batch_shape[batch_axis]
                        )));
                }
                batch_axis += 1;
            }
        }
    }

    let mut window_shape = vec![1usize; op_rank];
    let mut window_shape_updates = vec![0usize; config.update_window_dims.len()];
    for (pos, &update_axis) in config.update_window_dims.iter().enumerate() {
        let dim = updates_shape[update_axis];
        window_shape_updates[pos] = dim;
        window_shape[window_dims[pos]] = dim;
    }
    for axis in 0..op_rank {
        let _ = clamp_window_start("scatter", 0, operand_shape[axis], window_shape[axis])?;
    }

    checked_product("scatter", "batch shape", &batch_shape)?;
    checked_product("scatter", "window update shape", &window_shape_updates)?;

    let dtype = kernel_dtype(T::dtype());
    let index_dtype = KernelDType::I64;
    let operand_strides = col_major_strides(operand_shape);
    let index_strides = col_major_strides(&scatter_indices.shape);
    let update_strides = col_major_strides(updates_shape);
    let out_strides = col_major_strides(operand_shape);
    let plan = ErasedScatterPlan::compile(
        dtype,
        index_dtype,
        operand_shape,
        &operand_strides,
        &scatter_indices.shape,
        &index_strides,
        updates_shape,
        &update_strides,
        operand_shape,
        &out_strides,
        scatter_spec(config),
    )
    .map_err(|err| crate::Error::backend_source("scatter", err))?;

    // INVARIANT: ErasedScatterPlan first copies the full operand into `out`,
    // then applies every additive update.
    let mut out = pooled_uninit_tensor(buffers, operand_shape.to_vec())?;
    let operand_ref = ErasedRawStridedRef::new(
        dtype,
        typed_bytes(typed_host_data("scatter", operand)?),
        operand_shape,
        &operand_strides,
        0,
    )
    .map_err(|err| crate::Error::backend_source("scatter", err))?;
    let index_ref = ErasedRawStridedRef::new(
        index_dtype,
        typed_bytes(&scatter_indices.values),
        &scatter_indices.shape,
        &index_strides,
        0,
    )
    .map_err(|err| crate::Error::backend_source("scatter", err))?;
    let update_ref = ErasedRawStridedRef::new(
        dtype,
        typed_bytes(typed_host_data("scatter", updates)?),
        updates_shape,
        &update_strides,
        0,
    )
    .map_err(|err| crate::Error::backend_source("scatter", err))?;
    let mut out_ref = ErasedRawStridedMut::new(
        dtype,
        typed_bytes_mut(out.host_data_mut()?),
        operand_shape,
        &out_strides,
        0,
    )
    .map_err(|err| crate::Error::backend_source("scatter", err))?;
    plan.execute(
        exec_context,
        &mut out_ref,
        &operand_ref,
        &index_ref,
        &update_ref,
    )
    .map_err(|err| crate::Error::backend_source("scatter", err))?;

    Ok(out)
}

fn typed_dynamic_slice<T: Copy + Clone + PoolScalar + TensorScalar>(
    buffers: &mut BufferPool,
    exec_context: &ExecContext,
    input: &TypedTensor<T>,
    starts: &IndexTensor,
    slice_sizes: &[usize],
) -> crate::Result<TypedTensor<T>> {
    let input_shape = input.shape();
    if slice_sizes.len() != input_shape.len() {
        return Err(crate::Error::rank_mismatch(
            "dynamic_slice",
            input_shape.len(),
            slice_sizes.len(),
        ));
    }
    if starts.shape.len() != 1 {
        return Err(crate::Error::invalid_argument(
            "dynamic_slice",
            "starts",
            "starts must be a rank-1 tensor",
        ));
    }
    if starts.values.len() != input_shape.len() {
        return Err(crate::Error::invalid_argument(
            "dynamic_slice",
            "starts",
            format!(
                "starts length {} must match input rank {}",
                starts.values.len(),
                input_shape.len()
            ),
        ));
    }

    for axis in 0..input_shape.len() {
        let _ = clamp_window_start(
            "dynamic_slice",
            starts.values[axis],
            input_shape[axis],
            slice_sizes[axis],
        )?;
    }

    let out_shape = slice_sizes.to_vec();
    let dtype = kernel_dtype(T::dtype());
    let input_strides = col_major_strides(input_shape);
    let start_strides = col_major_strides(&starts.shape);
    let out_strides = col_major_strides(&out_shape);
    let plan = ErasedDynamicSlicePlan::compile(
        dtype,
        KernelDType::I64,
        input_shape,
        &input_strides,
        &starts.shape,
        &start_strides,
        &out_shape,
        &out_strides,
        slice_sizes,
    )
    .map_err(|err| crate::Error::backend_source("dynamic_slice", err))?;
    // INVARIANT: ErasedDynamicSlicePlan writes every output coordinate exactly once.
    let mut out = pooled_uninit_tensor(buffers, out_shape.clone())?;
    if T::dtype() == DType::Bool {
        out.host_data_mut()?.fill(T::pool_zero());
    }
    let input_ref = ErasedRawStridedRef::new(
        dtype,
        typed_bytes(typed_host_data("dynamic_slice", input)?),
        input_shape,
        &input_strides,
        0,
    )
    .map_err(|err| crate::Error::backend_source("dynamic_slice", err))?;
    let start_ref = ErasedRawStridedRef::new(
        KernelDType::I64,
        typed_bytes(&starts.values),
        &starts.shape,
        &start_strides,
        0,
    )
    .map_err(|err| crate::Error::backend_source("dynamic_slice", err))?;
    let mut out_ref = ErasedRawStridedMut::new(
        dtype,
        typed_bytes_mut(out.host_data_mut()?),
        &out_shape,
        &out_strides,
        0,
    )
    .map_err(|err| crate::Error::backend_source("dynamic_slice", err))?;
    plan.execute(exec_context, &mut out_ref, &input_ref, &start_ref)
        .map_err(|err| crate::Error::backend_source("dynamic_slice", err))?;

    Ok(out)
}

fn typed_dynamic_update_slice<T: Copy + Clone + PoolScalar + TensorScalar>(
    buffers: &mut BufferPool,
    exec_context: &ExecContext,
    operand: &TypedTensor<T>,
    update: &TypedTensor<T>,
    starts: &IndexTensor,
) -> crate::Result<TypedTensor<T>> {
    let operand_shape = operand.shape();
    let update_shape = update.shape();
    if update_shape.len() != operand_shape.len() {
        return Err(crate::Error::rank_mismatch(
            "dynamic_update_slice",
            operand_shape.len(),
            update_shape.len(),
        ));
    }
    if starts.shape.len() != 1 {
        return Err(crate::Error::invalid_argument(
            "dynamic_update_slice",
            "configuration",
            "starts must be a rank-1 tensor",
        ));
    }
    if starts.values.len() != operand_shape.len() {
        return Err(crate::Error::invalid_argument(
            "dynamic_update_slice",
            "configuration",
            format!(
                "starts length {} must match operand rank {}",
                starts.values.len(),
                operand_shape.len()
            ),
        ));
    }

    for axis in 0..operand_shape.len() {
        let _ = clamp_window_start(
            "dynamic_update_slice",
            starts.values[axis],
            operand_shape[axis],
            update_shape[axis],
        )?;
    }

    let dtype = kernel_dtype(T::dtype());
    let operand_strides = col_major_strides(operand_shape);
    let start_strides = col_major_strides(&starts.shape);
    let update_strides = col_major_strides(update_shape);
    let out_strides = col_major_strides(operand_shape);
    let plan = ErasedDynamicUpdateSlicePlan::compile(
        dtype,
        KernelDType::I64,
        operand_shape,
        &operand_strides,
        &starts.shape,
        &start_strides,
        update_shape,
        &update_strides,
        operand_shape,
        &out_strides,
    )
    .map_err(|err| crate::Error::backend_source("dynamic_update_slice", err))?;
    // INVARIANT: ErasedDynamicUpdateSlicePlan copies the full operand into
    // `out` before overwriting the update window.
    let mut out = pooled_uninit_tensor(buffers, operand_shape.to_vec())?;
    if T::dtype() == DType::Bool {
        out.host_data_mut()?.fill(T::pool_zero());
    }
    let operand_ref = ErasedRawStridedRef::new(
        dtype,
        typed_bytes(typed_host_data("dynamic_update_slice", operand)?),
        operand_shape,
        &operand_strides,
        0,
    )
    .map_err(|err| crate::Error::backend_source("dynamic_update_slice", err))?;
    let update_ref = ErasedRawStridedRef::new(
        dtype,
        typed_bytes(typed_host_data("dynamic_update_slice", update)?),
        update_shape,
        &update_strides,
        0,
    )
    .map_err(|err| crate::Error::backend_source("dynamic_update_slice", err))?;
    let start_ref = ErasedRawStridedRef::new(
        KernelDType::I64,
        typed_bytes(&starts.values),
        &starts.shape,
        &start_strides,
        0,
    )
    .map_err(|err| crate::Error::backend_source("dynamic_update_slice", err))?;
    let mut out_ref = ErasedRawStridedMut::new(
        dtype,
        typed_bytes_mut(out.host_data_mut()?),
        operand_shape,
        &out_strides,
        0,
    )
    .map_err(|err| crate::Error::backend_source("dynamic_update_slice", err))?;
    plan.execute(
        exec_context,
        &mut out_ref,
        &operand_ref,
        &update_ref,
        &start_ref,
    )
    .map_err(|err| crate::Error::backend_source("dynamic_update_slice", err))?;

    Ok(out)
}

fn typed_pad<T: Copy + Clone + Zero + PoolScalar>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    config: &PadConfig,
) -> crate::Result<TypedTensor<T>> {
    typed_pad_with_fill(buffers, input, config, T::zero())
}

fn typed_pad_with_fill<T: Copy + Clone + PoolScalar>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    config: &PadConfig,
    fill: T,
) -> crate::Result<TypedTensor<T>> {
    let input_shape = input.shape();
    let rank = input_shape.len();
    if config.edge_padding_low.len() != rank {
        return Err(crate::Error::rank_mismatch(
            "pad",
            rank,
            config.edge_padding_low.len(),
        ));
    }
    if config.edge_padding_high.len() != rank {
        return Err(crate::Error::rank_mismatch(
            "pad",
            rank,
            config.edge_padding_high.len(),
        ));
    }
    if config.interior_padding.len() != rank {
        return Err(crate::Error::rank_mismatch(
            "pad",
            rank,
            config.interior_padding.len(),
        ));
    }

    let mut out_shape = Vec::with_capacity(input_shape.len());
    for (axis, &input_extent) in input_shape.iter().enumerate() {
        if config.interior_padding[axis] < 0 {
            return Err(crate::Error::invalid_argument(
                "pad",
                "configuration",
                format!("interior padding must be non-negative on axis {axis}"),
            ));
        }
        let input_extent_i64 = i64::try_from(input_extent).map_err(|_| {
            crate::Error::invalid_argument(
                "pad",
                "configuration",
                format!("input extent on axis {axis} does not fit in i64"),
            )
        })?;
        let spacing = config.interior_padding[axis]
            .checked_add(1)
            .ok_or_else(|| {
                crate::Error::invalid_argument(
                    "pad",
                    "configuration",
                    format!("interior padding overflow on axis {axis}"),
                )
            })?;
        let base = if input_extent == 0 {
            0
        } else {
            input_extent_i64
                .checked_sub(1)
                .and_then(|extent| extent.checked_mul(spacing))
                .and_then(|extent| extent.checked_add(1))
                .ok_or_else(|| {
                    crate::Error::invalid_argument(
                        "pad",
                        "configuration",
                        format!("padded input extent overflow on axis {axis}"),
                    )
                })?
        };
        let dim = config.edge_padding_low[axis]
            .checked_add(config.edge_padding_high[axis])
            .and_then(|edge| edge.checked_add(base))
            .ok_or_else(|| {
                crate::Error::invalid_argument(
                    "pad",
                    "configuration",
                    format!("output dimension overflow on axis {axis}"),
                )
            })?;
        out_shape.push(usize::try_from(dim).map_err(|_| {
            crate::Error::invalid_argument(
                "pad",
                "configuration",
                format!("negative output dimension on axis {axis}"),
            )
        })?);
    }

    let mut out = pooled_filled_tensor(buffers, out_shape.clone(), fill)?;
    let mut input_idx = vec![0usize; input_shape.len()];
    let mut out_idx = vec![0usize; input_shape.len()];

    for input_value in input.as_slice()? {
        let mut in_bounds = true;
        for axis in 0..input_shape.len() {
            let out_pos = i128::from(config.edge_padding_low[axis])
                + input_idx[axis] as i128 * i128::from(config.interior_padding[axis] + 1);
            if !(0..out_shape[axis] as i128).contains(&out_pos) {
                in_bounds = false;
                break;
            }
            out_idx[axis] = out_pos as usize;
        }
        if in_bounds {
            *out.get_mut(&out_idx)? = *input_value;
        }
        advance_col_major_index(&mut input_idx, input_shape);
    }

    Ok(out)
}

#[cfg(test)]
mod tests;
