use std::ops::Add;

use num_traits::Zero;

use crate::config::{GatherConfig, PadConfig, ScatterConfig, SliceConfig};
use crate::types::{dispatch_binary, dispatch_tensor, flat_to_multi, Tensor, TypedTensor};

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

pub fn gather(operand: &Tensor, start_indices: &Tensor, config: &GatherConfig) -> Tensor {
    let start_indices = index_tensor(start_indices);
    dispatch_tensor!(operand, t => typed_gather(t, &start_indices, config))
}

pub fn scatter(
    operand: &Tensor,
    scatter_indices: &Tensor,
    updates: &Tensor,
    config: &ScatterConfig,
) -> Tensor {
    let scatter_indices = index_tensor(scatter_indices);
    dispatch_binary!(operand, updates, |op, upd| typed_scatter(
        op,
        &scatter_indices,
        upd,
        config
    ))
}

pub fn slice(input: &Tensor, config: &SliceConfig) -> Tensor {
    try_slice(input, config).expect("slice")
}

pub fn try_slice(input: &Tensor, config: &SliceConfig) -> crate::Result<Tensor> {
    match input {
        Tensor::F32(tensor) => Ok(Tensor::F32(typed_slice(tensor, config)?)),
        Tensor::F64(tensor) => Ok(Tensor::F64(typed_slice(tensor, config)?)),
        Tensor::C32(tensor) => Ok(Tensor::C32(typed_slice(tensor, config)?)),
        Tensor::C64(tensor) => Ok(Tensor::C64(typed_slice(tensor, config)?)),
    }
}

pub fn dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) -> Tensor {
    let starts = index_tensor(starts);
    dispatch_tensor!(input, t => typed_dynamic_slice(t, &starts, slice_sizes))
}

pub fn pad(input: &Tensor, config: &PadConfig) -> Tensor {
    try_pad(input, config).expect("pad")
}

pub fn try_pad(input: &Tensor, config: &PadConfig) -> crate::Result<Tensor> {
    match input {
        Tensor::F32(tensor) => Ok(Tensor::F32(typed_pad(tensor, config)?)),
        Tensor::F64(tensor) => Ok(Tensor::F64(typed_pad(tensor, config)?)),
        Tensor::C32(tensor) => Ok(Tensor::C32(typed_pad(tensor, config)?)),
        Tensor::C64(tensor) => Ok(Tensor::C64(typed_pad(tensor, config)?)),
    }
}

pub fn concatenate(inputs: &[&Tensor], axis: usize) -> Tensor {
    let first = inputs
        .first()
        .copied()
        .expect("concatenate requires at least one input");
    dispatch_tensor!(first, tensor => typed_concatenate_from_dyn_inputs(tensor, inputs, axis))
}

pub fn reverse(input: &Tensor, axes: &[usize]) -> Tensor {
    dispatch_tensor!(input, tensor => typed_reverse(tensor, axes))
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
) -> TypedTensor<T>
where
    T: Copy + Clone,
    Tensor: TensorAsTyped<T>,
{
    let typed_inputs = collect_typed_inputs(inputs);
    typed_concatenate(&typed_inputs, axis)
}

fn collect_typed_inputs<'a, T>(inputs: &[&'a Tensor]) -> Vec<&'a TypedTensor<T>>
where
    Tensor: TensorAsTyped<T>,
{
    inputs
        .iter()
        .map(|tensor| {
            TensorAsTyped::<T>::as_typed(*tensor)
                .expect("concatenate: dtype mismatch across inputs")
        })
        .collect()
}

fn typed_concatenate<T: Copy + Clone>(inputs: &[&TypedTensor<T>], axis: usize) -> TypedTensor<T> {
    let first = inputs[0];
    let rank = first.shape.len();
    assert!(axis < rank, "concatenate: axis out of bounds");

    let mut out_shape = first.shape.clone();
    let mut axis_extent = 0usize;
    for input in inputs {
        assert_eq!(input.shape.len(), rank, "concatenate: rank mismatch");
        for dim in 0..rank {
            if dim == axis {
                axis_extent += input.shape[dim];
            } else {
                assert_eq!(
                    input.shape[dim], first.shape[dim],
                    "concatenate: non-concat dimensions must match"
                );
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
            .expect("concatenate: output index must map to an input");
        let axis_base = if input_pos == 0 {
            0
        } else {
            segment_ends[input_pos - 1]
        };

        in_idx.copy_from_slice(&out_idx);
        in_idx[axis] -= axis_base;
        out_data.push(*inputs[input_pos].get(&in_idx));
    }

    TypedTensor::from_vec(out_shape, out_data)
}

fn typed_reverse<T: Copy + Clone>(input: &TypedTensor<T>, axes: &[usize]) -> TypedTensor<T> {
    let rank = input.shape.len();
    let mut reverse_axis = vec![false; rank];
    for &axis in axes {
        assert!(axis < rank, "reverse: axis out of bounds");
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

    TypedTensor::from_vec(input.shape.clone(), out_data)
}

struct IndexTensor {
    shape: Vec<usize>,
    values: Vec<i64>,
}

fn index_tensor(tensor: &Tensor) -> IndexTensor {
    fn collect_values<T: Copy>(t: &TypedTensor<T>) -> Vec<T> {
        let mut values = Vec::with_capacity(t.n_elements());
        let mut idx = vec![0usize; t.shape.len()];
        for flat in 0..t.n_elements() {
            flat_to_multi(flat, &t.shape, &mut idx);
            values.push(*t.get(&idx));
        }
        values
    }

    match tensor {
        Tensor::F32(t) => IndexTensor {
            shape: t.shape.clone(),
            values: collect_values(t)
                .into_iter()
                .map(|value| value as i64)
                .collect(),
        },
        Tensor::F64(t) => IndexTensor {
            shape: t.shape.clone(),
            values: collect_values(t)
                .into_iter()
                .map(|value| value as i64)
                .collect(),
        },
        Tensor::C32(_) | Tensor::C64(_) => panic!("complex index tensors are not supported"),
    }
}

fn linear_offset(shape: &[usize], indices: &[usize]) -> usize {
    let mut offset = 0usize;
    let mut stride = 1usize;
    for (axis, &index) in indices.iter().enumerate() {
        offset += index * stride;
        stride *= shape[axis];
    }
    offset
}

fn product(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn index_vector_size(shape: &[usize], index_vector_dim: usize) -> usize {
    if index_vector_dim == shape.len() {
        1
    } else {
        shape[index_vector_dim]
    }
}

fn index_batch_shape(shape: &[usize], index_vector_dim: usize) -> Vec<usize> {
    if index_vector_dim == shape.len() {
        return shape.to_vec();
    }
    shape
        .iter()
        .enumerate()
        .filter_map(|(axis, &dim)| (axis != index_vector_dim).then_some(dim))
        .collect()
}

fn index_component(
    indices: &IndexTensor,
    batch_idx: &[usize],
    index_vector_dim: usize,
    component: usize,
) -> i64 {
    if index_vector_dim == indices.shape.len() {
        assert_eq!(
            component, 0,
            "implicit index_vector_dim only supports scalar index vectors"
        );
        return indices.values[linear_offset(&indices.shape, batch_idx)];
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
    indices.values[linear_offset(&indices.shape, &full_idx)]
}

fn clamp_window_start(start: i64, dim_size: usize, window_size: usize) -> usize {
    assert!(
        window_size <= dim_size,
        "window size {window_size} exceeds dimension size {dim_size}"
    );
    let max_start = dim_size.saturating_sub(window_size) as i64;
    start.clamp(0, max_start) as usize
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
) -> TypedTensor<T> {
    assert_eq!(
        config.slice_sizes.len(),
        operand.shape.len(),
        "gather: slice_sizes rank mismatch"
    );

    let index_size = index_vector_size(&start_indices.shape, config.index_vector_dim);
    assert_eq!(
        index_size,
        config.start_index_map.len(),
        "gather: start_index_map length mismatch"
    );

    let window_dims = operand_window_dims(operand.shape.len(), &config.collapsed_slice_dims);
    assert_eq!(
        config.offset_dims.len(),
        window_dims.len(),
        "gather: offset_dims length mismatch"
    );

    let batch_shape = index_batch_shape(&start_indices.shape, config.index_vector_dim);
    let out_rank = batch_shape.len() + config.offset_dims.len();
    let mut out_shape = vec![0usize; out_rank];
    let mut out_axis_to_operand_dim = vec![None; out_rank];
    for (offset_axis, &out_axis) in config.offset_dims.iter().enumerate() {
        out_axis_to_operand_dim[out_axis] = Some(window_dims[offset_axis]);
    }

    let mut batch_axis = 0usize;
    for out_axis in 0..out_rank {
        if let Some(operand_dim) = out_axis_to_operand_dim[out_axis] {
            out_shape[out_axis] = config.slice_sizes[operand_dim];
        } else {
            out_shape[out_axis] = batch_shape[batch_axis];
            batch_axis += 1;
        }
    }

    let mut out = TypedTensor::zeros(out_shape.clone());
    let mut out_idx = vec![0usize; out_rank];
    let mut batch_idx = vec![0usize; batch_shape.len()];
    let mut operand_idx = vec![0usize; operand.shape.len()];
    let mut window_offsets = vec![0usize; operand.shape.len()];

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
                start_indices,
                &batch_idx,
                config.index_vector_dim,
                component,
            );
            operand_idx[operand_dim] = clamp_window_start(
                start,
                operand.shape[operand_dim],
                config.slice_sizes[operand_dim],
            );
        }

        for axis in 0..operand_idx.len() {
            operand_idx[axis] += window_offsets[axis];
        }

        *out.get_mut(&out_idx) = *operand.get(&operand_idx);
    }

    out
}

fn typed_scatter<T>(
    operand: &TypedTensor<T>,
    scatter_indices: &IndexTensor,
    updates: &TypedTensor<T>,
    config: &ScatterConfig,
) -> TypedTensor<T>
where
    T: Copy + Clone + Zero + Add<Output = T>,
{
    let index_size = index_vector_size(&scatter_indices.shape, config.index_vector_dim);
    assert_eq!(
        index_size,
        config.scatter_dims_to_operand_dims.len(),
        "scatter: scatter_dims_to_operand_dims length mismatch"
    );

    let batch_shape = index_batch_shape(&scatter_indices.shape, config.index_vector_dim);
    let window_dims = operand_window_dims(operand.shape.len(), &config.inserted_window_dims);
    assert_eq!(
        config.update_window_dims.len(),
        window_dims.len(),
        "scatter: update_window_dims length mismatch"
    );

    let update_rank = updates.shape.len();
    let mut is_update_window_dim = vec![false; update_rank];
    for &axis in &config.update_window_dims {
        is_update_window_dim[axis] = true;
    }
    assert_eq!(
        update_rank - config.update_window_dims.len(),
        batch_shape.len(),
        "scatter: updates batch rank mismatch"
    );

    let mut window_shape = vec![1usize; operand.shape.len()];
    let mut window_shape_updates = vec![0usize; config.update_window_dims.len()];
    for (pos, &update_axis) in config.update_window_dims.iter().enumerate() {
        let dim = updates.shape[update_axis];
        window_shape_updates[pos] = dim;
        window_shape[window_dims[pos]] = dim;
    }

    let batch_elems = product(&batch_shape).max(1);
    let window_elems = product(&window_shape_updates).max(1);
    let mut out = TypedTensor::zeros(operand.shape.clone());

    let mut batch_idx = vec![0usize; batch_shape.len()];
    let mut window_idx = vec![0usize; window_shape_updates.len()];
    let mut update_idx = vec![0usize; updates.shape.len()];
    let mut operand_base = vec![0usize; operand.shape.len()];
    let mut operand_idx = vec![0usize; operand.shape.len()];

    for batch_flat in 0..batch_elems {
        if !batch_shape.is_empty() {
            flat_to_multi(batch_flat, &batch_shape, &mut batch_idx);
        }

        let mut window_fits = true;
        operand_base.fill(0);
        for (component, &operand_dim) in config.scatter_dims_to_operand_dims.iter().enumerate() {
            let start = index_component(
                scatter_indices,
                &batch_idx,
                config.index_vector_dim,
                component,
            );
            if start < 0 {
                window_fits = false;
                break;
            }
            operand_base[operand_dim] = start as usize;
        }
        if !window_fits {
            continue;
        }

        for axis in 0..operand.shape.len() {
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
            for axis in 0..updates.shape.len() {
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

    out
}

fn typed_dynamic_slice<T: Copy + Clone + Zero>(
    input: &TypedTensor<T>,
    starts: &IndexTensor,
    slice_sizes: &[usize],
) -> TypedTensor<T> {
    assert_eq!(
        slice_sizes.len(),
        input.shape.len(),
        "dynamic_slice: slice_sizes rank mismatch"
    );
    assert_eq!(
        starts.shape.len(),
        1,
        "dynamic_slice: starts must be a rank-1 tensor"
    );
    assert_eq!(
        starts.values.len(),
        input.shape.len(),
        "dynamic_slice: starts length must match input rank"
    );

    let mut clamped_starts = vec![0usize; input.shape.len()];
    for axis in 0..input.shape.len() {
        clamped_starts[axis] =
            clamp_window_start(starts.values[axis], input.shape[axis], slice_sizes[axis]);
    }

    let out_shape = slice_sizes.to_vec();
    let mut out = TypedTensor::zeros(out_shape.clone());
    let mut out_idx = vec![0usize; out_shape.len()];
    let mut input_idx = vec![0usize; out_shape.len()];

    for flat in 0..out.n_elements() {
        flat_to_multi(flat, &out_shape, &mut out_idx);
        for axis in 0..out_shape.len() {
            input_idx[axis] = clamped_starts[axis] + out_idx[axis];
        }
        *out.get_mut(&out_idx) = *input.get(&input_idx);
    }

    out
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
