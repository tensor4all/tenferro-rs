use std::ops::Add;

use num_traits::Zero;

use super::indexing_alloc::pooled_uninit_tensor;
use super::typed_host_data;
use crate::buffer_pool::{BufferPool, PoolScalar};
use tenferro_tensor::{GatherConfig, PadConfig, ScatterConfig, SliceConfig};
use tenferro_tensor::{Tensor, TypedTensor};

// Indexing-family kernels stay as dedicated sequential loops for now. Their
// per-output gather/scatter/slice/pad/concatenate/reverse index transforms do
// not currently map to a strided-kernel or backend-native parallel primitive.
// Backend entrypoints still run these loops inside CpuContext::install, so a
// future parallel implementation can use the same CPU threading policy.

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
            _ => Err(crate::Error::DTypeMismatch {
                op: $op,
                lhs: $lhs.dtype(),
                rhs: $rhs.dtype(),
            }),
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
                Err(crate::Error::backend_failure($op, $bool_message))
            }
            _ => Err(crate::Error::DTypeMismatch {
                op: $op,
                lhs: $lhs.dtype(),
                rhs: $rhs.dtype(),
            }),
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

fn clone_host_tensor_from_pool<T>(
    buffers: &mut BufferPool,
    op: &'static str,
    tensor: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + PoolScalar,
{
    // SAFETY: copy_from_slice writes every pooled output element.
    let mut out = pooled_uninit_tensor(buffers, tensor.shape().to_vec())?;
    out.host_data_mut()?
        .copy_from_slice(typed_host_data(op, tensor)?);
    Ok(out)
}

#[cfg(test)]
pub(crate) fn gather(
    operand: &Tensor,
    start_indices: &Tensor,
    config: &GatherConfig,
) -> crate::Result<Tensor> {
    with_test_pool(|buffers| gather_with_pool(buffers, operand, start_indices, config))
}

pub(crate) fn gather_with_pool(
    buffers: &mut BufferPool,
    operand: &Tensor,
    start_indices: &Tensor,
    config: &GatherConfig,
) -> crate::Result<Tensor> {
    let start_indices = try_index_tensor(start_indices)?;
    dispatch_tensor_unary_result!(operand, |t| typed_gather(
        buffers,
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
    with_test_pool(|buffers| scatter_with_pool(buffers, operand, scatter_indices, updates, config))
}

pub(crate) fn scatter_with_pool(
    buffers: &mut BufferPool,
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
        |op, upd| typed_scatter(buffers, op, &scatter_indices, upd, config)
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
    with_test_pool(|buffers| dynamic_slice_with_pool(buffers, input, starts, slice_sizes))
}

pub(crate) fn dynamic_slice_with_pool(
    buffers: &mut BufferPool,
    input: &Tensor,
    starts: &Tensor,
    slice_sizes: &[usize],
) -> crate::Result<Tensor> {
    let starts = try_index_tensor(starts)?;
    dispatch_tensor_unary_result!(input, |t| typed_dynamic_slice(
        buffers,
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
    with_test_pool(|buffers| dynamic_update_slice_with_pool(buffers, operand, update, starts))
}

pub(crate) fn dynamic_update_slice_with_pool(
    buffers: &mut BufferPool,
    operand: &Tensor,
    update: &Tensor,
    starts: &Tensor,
) -> crate::Result<Tensor> {
    let starts = try_index_tensor(starts)?;
    dispatch_same_dtype_result!("dynamic_update_slice", operand, update, |op, upd| {
        typed_dynamic_update_slice(buffers, op, upd, &starts)
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
    let first = inputs
        .first()
        .copied()
        .ok_or_else(|| crate::Error::InvalidConfig {
            op: "concatenate",
            message: "concatenate requires at least one input".into(),
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
        return Err(crate::Error::RankMismatch {
            op: "slice",
            expected: rank,
            actual: config.starts.len(),
        });
    }
    if config.limits.len() != rank {
        return Err(crate::Error::RankMismatch {
            op: "slice",
            expected: rank,
            actual: config.limits.len(),
        });
    }
    if config.strides.len() != rank {
        return Err(crate::Error::RankMismatch {
            op: "slice",
            expected: rank,
            actual: config.strides.len(),
        });
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
                return Err(crate::Error::InvalidConfig {
                    op: "slice",
                    message: format!("start exceeds limit on axis {axis}"),
                });
            }
            if limit > dim {
                return Err(crate::Error::AxisOutOfBounds {
                    op: "slice",
                    axis,
                    rank,
                });
            }
            if stride == 0 {
                return Err(crate::Error::InvalidConfig {
                    op: "slice",
                    message: format!("stride must be positive on axis {axis}"),
                });
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
        .ok_or_else(|| crate::Error::InvalidConfig {
            op: "concatenate",
            message: "concatenate requires at least one input".into(),
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
            TensorAsTyped::<T>::as_typed(*tensor).ok_or_else(|| crate::Error::DTypeMismatch {
                op: "concatenate",
                lhs: first_dtype,
                rhs: tensor.dtype(),
            })
        })
        .collect()
}

fn typed_concatenate<T: Copy + Clone + PoolScalar>(
    buffers: &mut BufferPool,
    inputs: &[&TypedTensor<T>],
    axis: usize,
) -> crate::Result<TypedTensor<T>> {
    let first = inputs
        .first()
        .copied()
        .ok_or_else(|| crate::Error::InvalidConfig {
            op: "concatenate",
            message: "concatenate requires at least one input".into(),
        })?;
    let first_shape = first.shape();
    let rank = first_shape.len();
    if axis >= rank {
        return Err(crate::Error::AxisOutOfBounds {
            op: "concatenate",
            axis,
            rank,
        });
    }

    let mut out_shape = first_shape.to_vec();
    let mut axis_extent = 0usize;
    for input in inputs {
        let input_shape = input.shape();
        if input_shape.len() != rank {
            return Err(crate::Error::RankMismatch {
                op: "concatenate",
                expected: rank,
                actual: input_shape.len(),
            });
        }
        for dim in 0..rank {
            if dim == axis {
                axis_extent = axis_extent.checked_add(input_shape[dim]).ok_or_else(|| {
                    crate::Error::InvalidConfig {
                        op: "concatenate",
                        message: "concatenate axis extent overflows usize".to_string(),
                    }
                })?;
            } else if input_shape[dim] != first_shape[dim] {
                return Err(crate::Error::ShapeMismatch {
                    op: "concatenate",
                    lhs: first_shape.to_vec(),
                    rhs: input_shape.to_vec(),
                });
            }
        }
    }
    out_shape[axis] = axis_extent;

    let mut segment_ends = Vec::with_capacity(inputs.len());
    let mut segment_end = 0usize;
    for input in inputs {
        segment_end = segment_end
            .checked_add(input.shape()[axis])
            .ok_or_else(|| crate::Error::InvalidConfig {
                op: "concatenate",
                message: "concatenate segment offset overflows usize".to_string(),
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
            return Err(crate::Error::InvalidConfig {
                op: "concatenate",
                message: "output index must map to an input".to_string(),
            });
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
            return Err(crate::Error::AxisOutOfBounds {
                op: "reverse",
                axis,
                rank,
            });
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
        return Err(crate::Error::InvalidConfig {
            op: "index_tensor",
            message: format!("index value {value} is not an exactly representable i64"),
        });
    }
    Ok(value as i64)
}

fn f64_index_to_i64(value: f64) -> crate::Result<i64> {
    if !value.is_finite() || value.fract() != 0.0 || value.abs() > F64_MAX_EXACT_INT {
        return Err(crate::Error::InvalidConfig {
            op: "index_tensor",
            message: format!("index value {value} is not an exactly representable i64"),
        });
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
        Tensor::Bool(_) => Err(crate::Error::InvalidConfig {
            op: "index_tensor",
            message: "bool index tensors are not supported".into(),
        }),
        Tensor::C32(_) | Tensor::C64(_) => Err(crate::Error::InvalidConfig {
            op: "index_tensor",
            message: "complex index tensors are not supported".into(),
        }),
    }
}

fn checked_product(op: &'static str, role: &'static str, shape: &[usize]) -> crate::Result<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op,
                message: format!("{role} element count overflows usize"),
            })
    })
}

fn linear_offset(op: &'static str, shape: &[usize], indices: &[usize]) -> crate::Result<usize> {
    if indices.len() != shape.len() {
        return Err(crate::Error::RankMismatch {
            op,
            expected: shape.len(),
            actual: indices.len(),
        });
    }
    let mut offset = 0usize;
    let mut stride = 1usize;
    for (axis, &index) in indices.iter().enumerate() {
        if index >= shape[axis] {
            return Err(crate::Error::AxisOutOfBounds {
                op,
                axis,
                rank: shape.len(),
            });
        }
        let scaled = index
            .checked_mul(stride)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op,
                message: format!("linear index component overflows usize on axis {axis}"),
            })?;
        offset = offset
            .checked_add(scaled)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op,
                message: format!("linear offset overflows usize on axis {axis}"),
            })?;
        stride = stride
            .checked_mul(shape[axis])
            .ok_or_else(|| crate::Error::InvalidConfig {
                op,
                message: format!("linear stride overflows usize after axis {axis}"),
            })?;
    }
    Ok(offset)
}

fn try_index_vector_size(
    op: &'static str,
    shape: &[usize],
    index_vector_dim: usize,
) -> crate::Result<usize> {
    if index_vector_dim > shape.len() {
        return Err(crate::Error::AxisOutOfBounds {
            op,
            axis: index_vector_dim,
            rank: shape.len(),
        });
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
        return Err(crate::Error::AxisOutOfBounds {
            op,
            axis: index_vector_dim,
            rank: shape.len(),
        });
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

fn index_component(
    op: &'static str,
    indices: &IndexTensor,
    batch_idx: &[usize],
    index_vector_dim: usize,
    component: usize,
    index_scratch: &mut [usize],
) -> crate::Result<i64> {
    if index_vector_dim == indices.shape.len() {
        if component != 0 {
            return Err(crate::Error::InvalidConfig {
                op,
                message: "implicit index_vector_dim only supports scalar index vectors".into(),
            });
        }
        return Ok(indices.values[linear_offset(op, &indices.shape, batch_idx)?]);
    }

    if index_scratch.len() != indices.shape.len() {
        return Err(crate::Error::InvalidConfig {
            op,
            message: format!(
                "index scratch length {} must match index tensor rank {}",
                index_scratch.len(),
                indices.shape.len()
            ),
        });
    }
    if batch_idx.len() + 1 != indices.shape.len() {
        return Err(crate::Error::InvalidConfig {
            op,
            message: format!(
                "batch index rank {} must be one less than index tensor rank {}",
                batch_idx.len(),
                indices.shape.len()
            ),
        });
    }
    let mut batch_axis = 0usize;
    for (axis, slot) in index_scratch.iter_mut().enumerate() {
        if axis == index_vector_dim {
            *slot = component;
        } else {
            *slot = batch_idx[batch_axis];
            batch_axis += 1;
        }
    }
    Ok(indices.values[linear_offset(op, &indices.shape, index_scratch)?])
}

fn clamp_window_start(
    op: &'static str,
    start: i64,
    dim_size: usize,
    window_size: usize,
) -> crate::Result<usize> {
    if window_size > dim_size {
        return Err(crate::Error::InvalidConfig {
            op,
            message: format!("window size {window_size} exceeds dimension size {dim_size}"),
        });
    }
    let max_start = dim_size.saturating_sub(window_size) as i64;
    Ok(start.clamp(0, max_start) as usize)
}

fn operand_window_dims(rank: usize, collapsed_or_inserted: &[usize]) -> Vec<usize> {
    (0..rank)
        .filter(|dim| !collapsed_or_inserted.contains(dim))
        .collect()
}

fn typed_gather<T: Copy + Clone + PoolScalar>(
    buffers: &mut BufferPool,
    operand: &TypedTensor<T>,
    start_indices: &IndexTensor,
    config: &GatherConfig,
) -> crate::Result<TypedTensor<T>> {
    let operand_shape = operand.shape();
    let rank = operand_shape.len();
    if config.slice_sizes.len() != rank {
        return Err(crate::Error::RankMismatch {
            op: "gather",
            expected: rank,
            actual: config.slice_sizes.len(),
        });
    }

    for &dim in &config.collapsed_slice_dims {
        if dim >= rank {
            return Err(crate::Error::AxisOutOfBounds {
                op: "gather",
                axis: dim,
                rank,
            });
        }
    }
    {
        let mut seen = vec![false; rank];
        for &dim in &config.collapsed_slice_dims {
            if seen[dim] {
                return Err(crate::Error::DuplicateAxis {
                    op: "gather",
                    axis: dim,
                    role: "collapsed_slice_dims",
                });
            }
            seen[dim] = true;
        }
    }
    for &dim in &config.collapsed_slice_dims {
        if config.slice_sizes[dim] != 1 {
            return Err(crate::Error::InvalidConfig {
                op: "gather",
                message: format!(
                    "collapsed slice dimension {dim} must have slice_size == 1, got {}",
                    config.slice_sizes[dim]
                ),
            });
        }
    }

    let index_size =
        try_index_vector_size("gather", &start_indices.shape, config.index_vector_dim)?;
    if index_size != config.start_index_map.len() {
        return Err(crate::Error::InvalidConfig {
            op: "gather",
            message: format!(
                "start_index_map length {} does not match index vector size {}",
                config.start_index_map.len(),
                index_size
            ),
        });
    }
    for &operand_dim in &config.start_index_map {
        if operand_dim >= rank {
            return Err(crate::Error::AxisOutOfBounds {
                op: "gather",
                axis: operand_dim,
                rank,
            });
        }
    }
    {
        let mut seen = vec![false; rank];
        for &operand_dim in &config.start_index_map {
            if seen[operand_dim] {
                return Err(crate::Error::DuplicateAxis {
                    op: "gather",
                    axis: operand_dim,
                    role: "start_index_map",
                });
            }
            seen[operand_dim] = true;
        }
    }

    let window_dims = operand_window_dims(rank, &config.collapsed_slice_dims);
    if config.offset_dims.len() != window_dims.len() {
        return Err(crate::Error::InvalidConfig {
            op: "gather",
            message: format!(
                "offset_dims length {} does not match window dims count {}",
                config.offset_dims.len(),
                window_dims.len()
            ),
        });
    }

    let batch_shape =
        try_index_batch_shape("gather", &start_indices.shape, config.index_vector_dim)?;
    let out_rank = batch_shape.len() + config.offset_dims.len();
    for &out_axis in &config.offset_dims {
        if out_axis >= out_rank {
            return Err(crate::Error::AxisOutOfBounds {
                op: "gather",
                axis: out_axis,
                rank: out_rank,
            });
        }
    }
    {
        let mut seen = vec![false; out_rank];
        for &out_axis in &config.offset_dims {
            if seen[out_axis] {
                return Err(crate::Error::DuplicateAxis {
                    op: "gather",
                    axis: out_axis,
                    role: "offset_dims",
                });
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

    for &operand_dim in &config.start_index_map {
        let _ = clamp_window_start(
            "gather",
            0,
            operand_shape[operand_dim],
            config.slice_sizes[operand_dim],
        )?;
    }

    // SAFETY: the gather loop below assigns every output coordinate exactly once.
    let mut out = pooled_uninit_tensor(buffers, out_shape.clone())?;
    let mut out_idx = vec![0usize; out_rank];
    let mut batch_idx = vec![0usize; batch_shape.len()];
    let mut operand_idx = vec![0usize; rank];
    let mut window_offsets = vec![0usize; rank];
    let mut index_scratch = vec![0usize; start_indices.shape.len()];

    for out_value in out.host_data_mut()?.iter_mut() {
        batch_axis = 0;
        window_offsets.fill(0);
        for out_axis in 0..out_rank {
            if let Some(operand_dim) = out_axis_to_operand_dim[out_axis] {
                window_offsets[operand_dim] = out_idx[out_axis];
            } else {
                batch_idx[batch_axis] = out_idx[out_axis];
                batch_axis += 1;
            }
        }

        operand_idx.fill(0);
        for (component, &operand_dim) in config.start_index_map.iter().enumerate() {
            let start = index_component(
                "gather",
                start_indices,
                &batch_idx,
                config.index_vector_dim,
                component,
                &mut index_scratch,
            )?;
            operand_idx[operand_dim] = clamp_window_start(
                "gather",
                start,
                operand_shape[operand_dim],
                config.slice_sizes[operand_dim],
            )?;
        }

        for axis in 0..operand_idx.len() {
            operand_idx[axis] += window_offsets[axis];
        }

        *out_value = *operand.get(&operand_idx)?;
        advance_col_major_index(&mut out_idx, &out_shape);
    }

    Ok(out)
}

fn typed_scatter<T>(
    buffers: &mut BufferPool,
    operand: &TypedTensor<T>,
    scatter_indices: &IndexTensor,
    updates: &TypedTensor<T>,
    config: &ScatterConfig,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Add<Output = T> + PoolScalar,
{
    let operand_shape = operand.shape();
    let updates_shape = updates.shape();
    let op_rank = operand_shape.len();
    for &dim in &config.inserted_window_dims {
        if dim >= op_rank {
            return Err(crate::Error::AxisOutOfBounds {
                op: "scatter",
                axis: dim,
                rank: op_rank,
            });
        }
    }
    {
        let mut seen = vec![false; op_rank];
        for &dim in &config.inserted_window_dims {
            if seen[dim] {
                return Err(crate::Error::DuplicateAxis {
                    op: "scatter",
                    axis: dim,
                    role: "inserted_window_dims",
                });
            }
            seen[dim] = true;
        }
    }

    let index_size =
        try_index_vector_size("scatter", &scatter_indices.shape, config.index_vector_dim)?;
    if index_size != config.scatter_dims_to_operand_dims.len() {
        return Err(crate::Error::InvalidConfig {
            op: "scatter",
            message: format!(
                "scatter_dims_to_operand_dims length {} does not match index vector size {}",
                config.scatter_dims_to_operand_dims.len(),
                index_size
            ),
        });
    }
    for &operand_dim in &config.scatter_dims_to_operand_dims {
        if operand_dim >= op_rank {
            return Err(crate::Error::AxisOutOfBounds {
                op: "scatter",
                axis: operand_dim,
                rank: op_rank,
            });
        }
    }
    {
        let mut seen = vec![false; op_rank];
        for &operand_dim in &config.scatter_dims_to_operand_dims {
            if seen[operand_dim] {
                return Err(crate::Error::DuplicateAxis {
                    op: "scatter",
                    axis: operand_dim,
                    role: "scatter_dims_to_operand_dims",
                });
            }
            seen[operand_dim] = true;
        }
    }

    let batch_shape =
        try_index_batch_shape("scatter", &scatter_indices.shape, config.index_vector_dim)?;
    let window_dims = operand_window_dims(op_rank, &config.inserted_window_dims);
    if config.update_window_dims.len() != window_dims.len() {
        return Err(crate::Error::InvalidConfig {
            op: "scatter",
            message: format!(
                "update_window_dims length {} does not match window dims count {}",
                config.update_window_dims.len(),
                window_dims.len()
            ),
        });
    }

    let update_rank = updates_shape.len();
    let expected_batch_rank = update_rank
        .checked_sub(config.update_window_dims.len())
        .ok_or_else(|| crate::Error::InvalidConfig {
            op: "scatter",
            message: format!(
                "update_window_dims length {} exceeds update rank {}",
                config.update_window_dims.len(),
                update_rank
            ),
        })?;
    if expected_batch_rank != batch_shape.len() {
        return Err(crate::Error::InvalidConfig {
            op: "scatter",
            message: format!(
                "updates batch rank {} does not match index batch shape length {}",
                expected_batch_rank,
                batch_shape.len()
            ),
        });
    }

    for &axis in &config.update_window_dims {
        if axis >= update_rank {
            return Err(crate::Error::AxisOutOfBounds {
                op: "scatter",
                axis,
                rank: update_rank,
            });
        }
    }
    {
        let mut seen = vec![false; update_rank];
        for &axis in &config.update_window_dims {
            if seen[axis] {
                return Err(crate::Error::DuplicateAxis {
                    op: "scatter",
                    axis,
                    role: "update_window_dims",
                });
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
                    return Err(crate::Error::InvalidConfig {
                        op: "scatter",
                        message: format!(
                            "updates batch dim {} extent {} does not match index batch dim {} extent {}",
                            axis,
                            updates_shape[axis],
                            batch_axis,
                            batch_shape[batch_axis]
                        ),
                    });
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

    let batch_elems = checked_product("scatter", "batch shape", &batch_shape)?;
    let window_elems = checked_product("scatter", "window update shape", &window_shape_updates)?;
    let mut out = clone_host_tensor_from_pool(buffers, "scatter", operand)?;

    let mut batch_idx = vec![0usize; batch_shape.len()];
    let mut window_idx = vec![0usize; window_shape_updates.len()];
    let mut update_idx = vec![0usize; update_rank];
    let mut operand_base = vec![0usize; op_rank];
    let mut operand_idx = vec![0usize; op_rank];
    let mut index_scratch = vec![0usize; scatter_indices.shape.len()];

    for _ in 0..batch_elems {
        operand_base.fill(0);
        for (component, &operand_dim) in config.scatter_dims_to_operand_dims.iter().enumerate() {
            let start = index_component(
                "scatter",
                scatter_indices,
                &batch_idx,
                config.index_vector_dim,
                component,
                &mut index_scratch,
            )?;
            operand_base[operand_dim] = clamp_window_start(
                "scatter",
                start,
                operand_shape[operand_dim],
                window_shape[operand_dim],
            )?;
        }

        window_idx.fill(0);
        for _ in 0..window_elems {
            let mut batch_axis = 0usize;
            let mut window_axis = 0usize;
            for axis in 0..update_rank {
                if is_update_window_dim[axis] {
                    update_idx[axis] = window_idx[window_axis];
                    window_axis += 1;
                } else {
                    update_idx[axis] = batch_idx[batch_axis];
                    batch_axis += 1;
                }
            }

            operand_idx.copy_from_slice(&operand_base);
            for (window_axis, &operand_axis) in window_dims.iter().enumerate() {
                operand_idx[operand_axis] += window_idx[window_axis];
            }

            let value = *updates.get(&update_idx)?;
            let slot = out.get_mut(&operand_idx)?;
            *slot = *slot + value;
            advance_col_major_index(&mut window_idx, &window_shape_updates);
        }
        advance_col_major_index(&mut batch_idx, &batch_shape);
    }

    Ok(out)
}

fn typed_dynamic_slice<T: Copy + Clone + PoolScalar>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    starts: &IndexTensor,
    slice_sizes: &[usize],
) -> crate::Result<TypedTensor<T>> {
    let input_shape = input.shape();
    if slice_sizes.len() != input_shape.len() {
        return Err(crate::Error::RankMismatch {
            op: "dynamic_slice",
            expected: input_shape.len(),
            actual: slice_sizes.len(),
        });
    }
    if starts.shape.len() != 1 {
        return Err(crate::Error::InvalidConfig {
            op: "dynamic_slice",
            message: "starts must be a rank-1 tensor".into(),
        });
    }
    if starts.values.len() != input_shape.len() {
        return Err(crate::Error::InvalidConfig {
            op: "dynamic_slice",
            message: format!(
                "starts length {} must match input rank {}",
                starts.values.len(),
                input_shape.len()
            ),
        });
    }

    let mut clamped_starts = vec![0usize; input_shape.len()];
    for axis in 0..input_shape.len() {
        clamped_starts[axis] = clamp_window_start(
            "dynamic_slice",
            starts.values[axis],
            input_shape[axis],
            slice_sizes[axis],
        )?;
    }

    let out_shape = slice_sizes.to_vec();
    // SAFETY: the dynamic-slice loop below assigns every output coordinate exactly once.
    let mut out = pooled_uninit_tensor(buffers, out_shape.clone())?;
    let mut out_idx = vec![0usize; out_shape.len()];
    let mut input_idx = vec![0usize; out_shape.len()];

    for out_value in out.host_data_mut()?.iter_mut() {
        for axis in 0..out_shape.len() {
            input_idx[axis] = clamped_starts[axis] + out_idx[axis];
        }
        *out_value = *input.get(&input_idx)?;
        advance_col_major_index(&mut out_idx, &out_shape);
    }

    Ok(out)
}

fn typed_dynamic_update_slice<T: Copy + Clone + PoolScalar>(
    buffers: &mut BufferPool,
    operand: &TypedTensor<T>,
    update: &TypedTensor<T>,
    starts: &IndexTensor,
) -> crate::Result<TypedTensor<T>> {
    let operand_shape = operand.shape();
    let update_shape = update.shape();
    if update_shape.len() != operand_shape.len() {
        return Err(crate::Error::RankMismatch {
            op: "dynamic_update_slice",
            expected: operand_shape.len(),
            actual: update_shape.len(),
        });
    }
    if starts.shape.len() != 1 {
        return Err(crate::Error::InvalidConfig {
            op: "dynamic_update_slice",
            message: "starts must be a rank-1 tensor".into(),
        });
    }
    if starts.values.len() != operand_shape.len() {
        return Err(crate::Error::InvalidConfig {
            op: "dynamic_update_slice",
            message: format!(
                "starts length {} must match operand rank {}",
                starts.values.len(),
                operand_shape.len()
            ),
        });
    }

    let mut clamped_starts = vec![0usize; operand_shape.len()];
    for axis in 0..operand_shape.len() {
        clamped_starts[axis] = clamp_window_start(
            "dynamic_update_slice",
            starts.values[axis],
            operand_shape[axis],
            update_shape[axis],
        )?;
    }

    let mut out = clone_host_tensor_from_pool(buffers, "dynamic_update_slice", operand)?;
    let mut update_idx = vec![0usize; update_shape.len()];
    let mut operand_idx = vec![0usize; operand_shape.len()];

    for update_value in update.as_slice()? {
        for axis in 0..update_shape.len() {
            operand_idx[axis] = clamped_starts[axis] + update_idx[axis];
        }
        *out.get_mut(&operand_idx)? = *update_value;
        advance_col_major_index(&mut update_idx, update_shape);
    }

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
        return Err(crate::Error::RankMismatch {
            op: "pad",
            expected: rank,
            actual: config.edge_padding_low.len(),
        });
    }
    if config.edge_padding_high.len() != rank {
        return Err(crate::Error::RankMismatch {
            op: "pad",
            expected: rank,
            actual: config.edge_padding_high.len(),
        });
    }
    if config.interior_padding.len() != rank {
        return Err(crate::Error::RankMismatch {
            op: "pad",
            expected: rank,
            actual: config.interior_padding.len(),
        });
    }

    let mut out_shape = Vec::with_capacity(input_shape.len());
    for (axis, &input_extent) in input_shape.iter().enumerate() {
        if config.interior_padding[axis] < 0 {
            return Err(crate::Error::InvalidConfig {
                op: "pad",
                message: format!("interior padding must be non-negative on axis {axis}"),
            });
        }
        let input_extent_i64 =
            i64::try_from(input_extent).map_err(|_| crate::Error::InvalidConfig {
                op: "pad",
                message: format!("input extent on axis {axis} does not fit in i64"),
            })?;
        let spacing = config.interior_padding[axis]
            .checked_add(1)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op: "pad",
                message: format!("interior padding overflow on axis {axis}"),
            })?;
        let base = if input_extent == 0 {
            0
        } else {
            input_extent_i64
                .checked_sub(1)
                .and_then(|extent| extent.checked_mul(spacing))
                .and_then(|extent| extent.checked_add(1))
                .ok_or_else(|| crate::Error::InvalidConfig {
                    op: "pad",
                    message: format!("padded input extent overflow on axis {axis}"),
                })?
        };
        let dim = config.edge_padding_low[axis]
            .checked_add(config.edge_padding_high[axis])
            .and_then(|edge| edge.checked_add(base))
            .ok_or_else(|| crate::Error::InvalidConfig {
                op: "pad",
                message: format!("output dimension overflow on axis {axis}"),
            })?;
        out_shape.push(
            usize::try_from(dim).map_err(|_| crate::Error::InvalidConfig {
                op: "pad",
                message: format!("negative output dimension on axis {axis}"),
            })?,
        );
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
