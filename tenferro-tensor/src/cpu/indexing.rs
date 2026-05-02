use std::ops::Add;

use num_traits::Zero;

use crate::config::{GatherConfig, PadConfig, ScatterConfig, SliceConfig};
use crate::types::{flat_to_multi, Tensor, TypedTensor};

trait TensorAsTyped<T> {
    fn as_typed(&self) -> Option<&TypedTensor<T>>;
}

impl TensorAsTyped<f32> for Tensor {
    fn as_typed(&self) -> Option<&TypedTensor<f32>> {
        match self {
            Tensor::F32(tensor) => Some(tensor),
            _ => None,
        }
    }
}

impl TensorAsTyped<f64> for Tensor {
    fn as_typed(&self) -> Option<&TypedTensor<f64>> {
        match self {
            Tensor::F64(tensor) => Some(tensor),
            _ => None,
        }
    }
}

impl TensorAsTyped<i64> for Tensor {
    fn as_typed(&self) -> Option<&TypedTensor<i64>> {
        match self {
            Tensor::I64(tensor) => Some(tensor),
            _ => None,
        }
    }
}

impl TensorAsTyped<num_complex::Complex<f32>> for Tensor {
    fn as_typed(&self) -> Option<&TypedTensor<num_complex::Complex<f32>>> {
        match self {
            Tensor::C32(tensor) => Some(tensor),
            _ => None,
        }
    }
}

impl TensorAsTyped<num_complex::Complex<f64>> for Tensor {
    fn as_typed(&self) -> Option<&TypedTensor<num_complex::Complex<f64>>> {
        match self {
            Tensor::C64(tensor) => Some(tensor),
            _ => None,
        }
    }
}

pub fn gather(
    operand: &Tensor,
    start_indices: &Tensor,
    config: &GatherConfig,
) -> crate::Result<Tensor> {
    let start_indices = try_index_tensor(start_indices)?;
    match operand {
        Tensor::F32(t) => typed_gather(t, &start_indices, config).map(Tensor::F32),
        Tensor::F64(t) => typed_gather(t, &start_indices, config).map(Tensor::F64),
        Tensor::C32(t) => typed_gather(t, &start_indices, config).map(Tensor::C32),
        Tensor::C64(t) => typed_gather(t, &start_indices, config).map(Tensor::C64),
        Tensor::I64(_) => Err(crate::Error::BackendFailure {
            op: "gather",
            message: "I64 data tensors are not supported by this operation".into(),
        }),
    }
}

pub fn scatter(
    operand: &Tensor,
    scatter_indices: &Tensor,
    updates: &Tensor,
    config: &ScatterConfig,
) -> crate::Result<Tensor> {
    let scatter_indices = try_index_tensor(scatter_indices)?;
    match (operand, updates) {
        (Tensor::F32(op), Tensor::F32(upd)) => {
            typed_scatter(op, &scatter_indices, upd, config).map(Tensor::F32)
        }
        (Tensor::F64(op), Tensor::F64(upd)) => {
            typed_scatter(op, &scatter_indices, upd, config).map(Tensor::F64)
        }
        (Tensor::C32(op), Tensor::C32(upd)) => {
            typed_scatter(op, &scatter_indices, upd, config).map(Tensor::C32)
        }
        (Tensor::C64(op), Tensor::C64(upd)) => {
            typed_scatter(op, &scatter_indices, upd, config).map(Tensor::C64)
        }
        (Tensor::I64(_), Tensor::I64(_)) => Err(crate::Error::BackendFailure {
            op: "scatter",
            message: "I64 data tensors are not supported by this operation".into(),
        }),
        _ => Err(crate::Error::DTypeMismatch {
            op: "scatter",
            lhs: operand.dtype(),
            rhs: updates.dtype(),
        }),
    }
}

pub fn slice(input: &Tensor, config: &SliceConfig) -> crate::Result<Tensor> {
    try_slice(input, config)
}

pub fn try_slice(input: &Tensor, config: &SliceConfig) -> crate::Result<Tensor> {
    match input {
        Tensor::F32(tensor) => Ok(Tensor::F32(typed_slice(tensor, config)?)),
        Tensor::F64(tensor) => Ok(Tensor::F64(typed_slice(tensor, config)?)),
        Tensor::I64(tensor) => Ok(Tensor::I64(typed_slice(tensor, config)?)),
        Tensor::C32(tensor) => Ok(Tensor::C32(typed_slice(tensor, config)?)),
        Tensor::C64(tensor) => Ok(Tensor::C64(typed_slice(tensor, config)?)),
    }
}

pub fn dynamic_slice(
    input: &Tensor,
    starts: &Tensor,
    slice_sizes: &[usize],
) -> crate::Result<Tensor> {
    let starts = try_index_tensor(starts)?;
    match input {
        Tensor::F32(t) => typed_dynamic_slice(t, &starts, slice_sizes).map(Tensor::F32),
        Tensor::F64(t) => typed_dynamic_slice(t, &starts, slice_sizes).map(Tensor::F64),
        Tensor::C32(t) => typed_dynamic_slice(t, &starts, slice_sizes).map(Tensor::C32),
        Tensor::C64(t) => typed_dynamic_slice(t, &starts, slice_sizes).map(Tensor::C64),
        Tensor::I64(_) => Err(crate::Error::BackendFailure {
            op: "dynamic_slice",
            message: "I64 data tensors are not supported by this operation".into(),
        }),
    }
}

pub fn pad(input: &Tensor, config: &PadConfig) -> crate::Result<Tensor> {
    try_pad(input, config)
}

pub fn try_pad(input: &Tensor, config: &PadConfig) -> crate::Result<Tensor> {
    match input {
        Tensor::F32(tensor) => Ok(Tensor::F32(typed_pad(tensor, config)?)),
        Tensor::F64(tensor) => Ok(Tensor::F64(typed_pad(tensor, config)?)),
        Tensor::I64(tensor) => Ok(Tensor::I64(typed_pad(tensor, config)?)),
        Tensor::C32(tensor) => Ok(Tensor::C32(typed_pad(tensor, config)?)),
        Tensor::C64(tensor) => Ok(Tensor::C64(typed_pad(tensor, config)?)),
    }
}

pub fn concatenate(inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor> {
    try_concatenate(inputs, axis)
}

pub fn try_concatenate(inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor> {
    let first = inputs
        .first()
        .copied()
        .ok_or_else(|| crate::Error::InvalidConfig {
            op: "concatenate",
            message: "concatenate requires at least one input".into(),
        })?;
    match first {
        Tensor::F32(t) => Ok(Tensor::F32(typed_concatenate_from_dyn_inputs(
            t, inputs, axis,
        )?)),
        Tensor::F64(t) => Ok(Tensor::F64(typed_concatenate_from_dyn_inputs(
            t, inputs, axis,
        )?)),
        Tensor::I64(t) => Ok(Tensor::I64(typed_concatenate_from_dyn_inputs(
            t, inputs, axis,
        )?)),
        Tensor::C32(t) => Ok(Tensor::C32(typed_concatenate_from_dyn_inputs(
            t, inputs, axis,
        )?)),
        Tensor::C64(t) => Ok(Tensor::C64(typed_concatenate_from_dyn_inputs(
            t, inputs, axis,
        )?)),
    }
}

pub fn reverse(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
    match input {
        Tensor::F32(t) => typed_reverse(t, axes).map(Tensor::F32),
        Tensor::F64(t) => typed_reverse(t, axes).map(Tensor::F64),
        Tensor::I64(t) => typed_reverse(t, axes).map(Tensor::I64),
        Tensor::C32(t) => typed_reverse(t, axes).map(Tensor::C32),
        Tensor::C64(t) => typed_reverse(t, axes).map(Tensor::C64),
    }
}

#[allow(clippy::uninit_vec)]
fn typed_tensor_uninit<T: Clone>(shape: Vec<usize>) -> TypedTensor<T> {
    let n: usize = shape.iter().product();
    let mut data = Vec::with_capacity(n);
    // SAFETY: callers only use this helper for outputs that are fully written
    // before any read.
    unsafe { data.set_len(n) };
    TypedTensor::from_vec(shape, data)
}

fn typed_slice<T: Copy + Clone>(
    input: &TypedTensor<T>,
    config: &SliceConfig,
) -> crate::Result<TypedTensor<T>> {
    let rank = input.shape.len();
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
        .shape
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
            Ok((span + stride - 1) / stride)
        })
        .collect::<crate::Result<Vec<_>>>()?;

    let out_len: usize = out_shape.iter().product();
    let mut out_data = Vec::with_capacity(out_len);
    let mut out_idx = vec![0usize; rank];
    let mut in_idx = vec![0usize; rank];

    for flat in 0..out_len {
        flat_to_multi(flat, &out_shape, &mut out_idx);
        for axis in 0..rank {
            in_idx[axis] = config.starts[axis] + out_idx[axis] * config.strides[axis];
        }
        out_data.push(*input.get(&in_idx));
    }

    Ok(TypedTensor::from_vec(out_shape, out_data))
}

fn typed_concatenate_from_dyn_inputs<T>(
    _first: &TypedTensor<T>,
    inputs: &[&Tensor],
    axis: usize,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone,
    Tensor: TensorAsTyped<T>,
{
    let first_dtype = inputs[0].dtype();
    let typed_inputs = collect_typed_inputs(first_dtype, inputs)?;
    typed_concatenate(&typed_inputs, axis)
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

fn typed_concatenate<T: Copy + Clone>(
    inputs: &[&TypedTensor<T>],
    axis: usize,
) -> crate::Result<TypedTensor<T>> {
    let first = inputs[0];
    let rank = first.shape.len();
    if axis >= rank {
        return Err(crate::Error::AxisOutOfBounds {
            op: "concatenate",
            axis,
            rank,
        });
    }

    let mut out_shape = first.shape.clone();
    let mut axis_extent = 0usize;
    for input in inputs {
        if input.shape.len() != rank {
            return Err(crate::Error::RankMismatch {
                op: "concatenate",
                expected: rank,
                actual: input.shape.len(),
            });
        }
        for dim in 0..rank {
            if dim == axis {
                axis_extent += input.shape[dim];
            } else if input.shape[dim] != first.shape[dim] {
                return Err(crate::Error::ShapeMismatch {
                    op: "concatenate",
                    lhs: first.shape.clone(),
                    rhs: input.shape.clone(),
                });
            }
        }
    }
    out_shape[axis] = axis_extent;

    let segment_ends: Vec<usize> = inputs
        .iter()
        .scan(0usize, |sum, input| {
            *sum += input.shape[axis];
            Some(*sum)
        })
        .collect();

    let out_len: usize = out_shape.iter().product();
    let mut out_data = Vec::with_capacity(out_len);
    let mut out_idx = vec![0usize; rank];
    let mut in_idx = vec![0usize; rank];

    for flat in 0..out_len {
        flat_to_multi(flat, &out_shape, &mut out_idx);
        let concat_idx = out_idx[axis];
        let input_pos = segment_ends
            .iter()
            .position(|&end| concat_idx < end)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op: "concatenate",
                message: "output index must map to an input".to_string(),
            })?;
        let axis_base = if input_pos == 0 {
            0
        } else {
            segment_ends[input_pos - 1]
        };

        in_idx.copy_from_slice(&out_idx);
        in_idx[axis] -= axis_base;
        out_data.push(*inputs[input_pos].get(&in_idx));
    }

    Ok(TypedTensor::from_vec(out_shape, out_data))
}

fn typed_reverse<T: Copy + Clone>(
    input: &TypedTensor<T>,
    axes: &[usize],
) -> crate::Result<TypedTensor<T>> {
    let rank = input.shape.len();
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

    let out_len = input.n_elements();
    let mut out_data = Vec::with_capacity(out_len);
    let mut out_idx = vec![0usize; rank];
    let mut in_idx = vec![0usize; rank];

    for flat in 0..out_len {
        flat_to_multi(flat, &input.shape, &mut out_idx);
        for axis in 0..rank {
            in_idx[axis] = if reverse_axis[axis] {
                input.shape[axis] - 1 - out_idx[axis]
            } else {
                out_idx[axis]
            };
        }
        out_data.push(*input.get(&in_idx));
    }

    Ok(TypedTensor::from_vec(input.shape.clone(), out_data))
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
        Tensor::I64(t) => Ok(IndexTensor {
            shape: t.shape.clone(),
            values: t.host_data().to_vec(),
        }),
        Tensor::F32(t) => {
            let values: crate::Result<Vec<i64>> = t
                .host_data()
                .iter()
                .map(|&value| f32_index_to_i64(value))
                .collect();
            Ok(IndexTensor {
                shape: t.shape.clone(),
                values: values?,
            })
        }
        Tensor::F64(t) => {
            let values: crate::Result<Vec<i64>> = t
                .host_data()
                .iter()
                .map(|&value| f64_index_to_i64(value))
                .collect();
            Ok(IndexTensor {
                shape: t.shape.clone(),
                values: values?,
            })
        }
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
    let mut offset = 0usize;
    let mut stride = 1usize;
    for (axis, &index) in indices.iter().enumerate() {
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

    let mut full_idx = vec![0usize; indices.shape.len()];
    let mut batch_axis = 0usize;
    for (axis, slot) in full_idx.iter_mut().enumerate() {
        if axis == index_vector_dim {
            *slot = component;
        } else {
            *slot = batch_idx[batch_axis];
            batch_axis += 1;
        }
    }
    Ok(indices.values[linear_offset(op, &indices.shape, &full_idx)?])
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

fn typed_gather<T: Copy + Clone + Zero>(
    operand: &TypedTensor<T>,
    start_indices: &IndexTensor,
    config: &GatherConfig,
) -> crate::Result<TypedTensor<T>> {
    let rank = operand.shape.len();
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

    for (component, &operand_dim) in config.start_index_map.iter().enumerate() {
        let _ = clamp_window_start(
            "gather",
            0,
            operand.shape[operand_dim],
            config.slice_sizes[operand_dim],
        )?;
        let _ = component;
    }

    let mut out = typed_tensor_uninit(out_shape.clone());
    let mut out_idx = vec![0usize; out_rank];
    let mut batch_idx = vec![0usize; batch_shape.len()];
    let mut operand_idx = vec![0usize; rank];
    let mut window_offsets = vec![0usize; rank];

    for flat in 0..out.n_elements() {
        flat_to_multi(flat, &out_shape, &mut out_idx);

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
            )?;
            operand_idx[operand_dim] = clamp_window_start(
                "gather",
                start,
                operand.shape[operand_dim],
                config.slice_sizes[operand_dim],
            )?;
        }

        for axis in 0..operand_idx.len() {
            operand_idx[axis] += window_offsets[axis];
        }

        *out.get_mut(&out_idx) = *operand.get(&operand_idx);
    }

    Ok(out)
}

fn typed_scatter<T>(
    operand: &TypedTensor<T>,
    scatter_indices: &IndexTensor,
    updates: &TypedTensor<T>,
    config: &ScatterConfig,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Add<Output = T>,
{
    let op_rank = operand.shape.len();
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

    let update_rank = updates.shape.len();
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
                if updates.shape[axis] != batch_shape[batch_axis] {
                    return Err(crate::Error::InvalidConfig {
                        op: "scatter",
                        message: format!(
                            "updates batch dim {} extent {} does not match index batch dim {} extent {}",
                            axis,
                            updates.shape[axis],
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
        let dim = updates.shape[update_axis];
        window_shape_updates[pos] = dim;
        window_shape[window_dims[pos]] = dim;
    }

    let batch_elems = checked_product("scatter", "batch shape", &batch_shape)?.max(1);
    let window_elems =
        checked_product("scatter", "window update shape", &window_shape_updates)?.max(1);
    let mut out = operand.clone();

    let mut batch_idx = vec![0usize; batch_shape.len()];
    let mut window_idx = vec![0usize; window_shape_updates.len()];
    let mut update_idx = vec![0usize; update_rank];
    let mut operand_base = vec![0usize; op_rank];
    let mut operand_idx = vec![0usize; op_rank];

    for batch_flat in 0..batch_elems {
        if !batch_shape.is_empty() {
            flat_to_multi(batch_flat, &batch_shape, &mut batch_idx);
        }

        let mut window_fits = true;
        operand_base.fill(0);
        for (component, &operand_dim) in config.scatter_dims_to_operand_dims.iter().enumerate() {
            let start = index_component(
                "scatter",
                scatter_indices,
                &batch_idx,
                config.index_vector_dim,
                component,
            )?;
            if start < 0 {
                window_fits = false;
                break;
            }
            operand_base[operand_dim] = start as usize;
        }
        if !window_fits {
            continue;
        }

        for axis in 0..op_rank {
            if operand_base[axis] + window_shape[axis] > operand.shape[axis] {
                window_fits = false;
                break;
            }
        }
        if !window_fits {
            continue;
        }

        for window_flat in 0..window_elems {
            if !window_shape_updates.is_empty() {
                flat_to_multi(window_flat, &window_shape_updates, &mut window_idx);
            }

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

            let value = *updates.get(&update_idx);
            let slot = out.get_mut(&operand_idx);
            *slot = *slot + value;
        }
    }

    Ok(out)
}

fn typed_dynamic_slice<T: Copy + Clone + Zero>(
    input: &TypedTensor<T>,
    starts: &IndexTensor,
    slice_sizes: &[usize],
) -> crate::Result<TypedTensor<T>> {
    if slice_sizes.len() != input.shape.len() {
        return Err(crate::Error::RankMismatch {
            op: "dynamic_slice",
            expected: input.shape.len(),
            actual: slice_sizes.len(),
        });
    }
    if starts.shape.len() != 1 {
        return Err(crate::Error::InvalidConfig {
            op: "dynamic_slice",
            message: "starts must be a rank-1 tensor".into(),
        });
    }
    if starts.values.len() != input.shape.len() {
        return Err(crate::Error::InvalidConfig {
            op: "dynamic_slice",
            message: format!(
                "starts length {} must match input rank {}",
                starts.values.len(),
                input.shape.len()
            ),
        });
    }

    let mut clamped_starts = vec![0usize; input.shape.len()];
    for axis in 0..input.shape.len() {
        clamped_starts[axis] = clamp_window_start(
            "dynamic_slice",
            starts.values[axis],
            input.shape[axis],
            slice_sizes[axis],
        )?;
    }

    let out_shape = slice_sizes.to_vec();
    let mut out = typed_tensor_uninit(out_shape.clone());
    let mut out_idx = vec![0usize; out_shape.len()];
    let mut input_idx = vec![0usize; out_shape.len()];

    for flat in 0..out.n_elements() {
        flat_to_multi(flat, &out_shape, &mut out_idx);
        for axis in 0..out_shape.len() {
            input_idx[axis] = clamped_starts[axis] + out_idx[axis];
        }
        *out.get_mut(&out_idx) = *input.get(&input_idx);
    }

    Ok(out)
}

fn typed_pad<T: Copy + Clone + Zero>(
    input: &TypedTensor<T>,
    config: &PadConfig,
) -> crate::Result<TypedTensor<T>> {
    let rank = input.shape.len();
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

    let mut out_shape = Vec::with_capacity(input.shape.len());
    for axis in 0..input.shape.len() {
        if config.interior_padding[axis] < 0 {
            return Err(crate::Error::InvalidConfig {
                op: "pad",
                message: format!("interior padding must be non-negative on axis {axis}"),
            });
        }
        let base = if input.shape[axis] == 0 {
            0
        } else {
            (input.shape[axis] as i64 - 1) * (config.interior_padding[axis] + 1) + 1
        };
        let dim = config.edge_padding_low[axis] + config.edge_padding_high[axis] + base;
        out_shape.push(
            usize::try_from(dim).map_err(|_| crate::Error::InvalidConfig {
                op: "pad",
                message: format!("negative output dimension on axis {axis}"),
            })?,
        );
    }

    let mut out = TypedTensor::zeros(out_shape.clone());
    let mut input_idx = vec![0usize; input.shape.len()];
    let mut out_idx = vec![0usize; input.shape.len()];

    for flat in 0..input.n_elements() {
        flat_to_multi(flat, &input.shape, &mut input_idx);
        let mut in_bounds = true;
        for axis in 0..input.shape.len() {
            let out_pos = config.edge_padding_low[axis]
                + input_idx[axis] as i64 * (config.interior_padding[axis] + 1);
            if !(0..out_shape[axis] as i64).contains(&out_pos) {
                in_bounds = false;
                break;
            }
            out_idx[axis] = out_pos as usize;
        }
        if in_bounds {
            *out.get_mut(&out_idx) = *input.get(&input_idx);
        }
    }

    Ok(out)
}
