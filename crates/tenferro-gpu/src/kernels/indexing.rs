// INVARIANT: CubeCL's kernel DSL lowers scalar modulo checks differently
// from host Rust; these expressions are bounded by validated launch shapes.
#![allow(clippy::manual_is_multiple_of)]

use cubecl::prelude::*;

use crate::kernels::helpers::{
    axis_in_sequence, axis_position_in_sequence, flat_to_tensor_index, multi_to_tensor_index,
    zero_value,
};

#[cube(launch_unchecked)]
pub fn init_float_index_validation_flag<F: Float>(
    flag: &mut Array<Atomic<u32>>,
    flag_values: &mut Array<F>,
) {
    // INVARIANT: exactly one worker initializes two scalar validation outputs; this is O(1)
    // setup and does not perform tensor-sized serial work.
    if ABSOLUTE_POS == 0 {
        flag[0].store(u32::MAX);
        flag_values[1] = F::new(0.0_f32);
    }
}

#[cube(launch_unchecked)]
pub fn validate_float_indices_kernel<
    F: Float + CubeElement + CubePrimitive<WithScalar<bool> = bool, WithScalar<F> = F>,
>(
    indices: &Tensor<F>,
    flag: &mut Array<Atomic<u32>>,
    max_exact_integer: F,
) {
    if ABSOLUTE_POS < indices.len() {
        let value = indices[ABSOLUTE_POS];
        if value.is_nan()
            || value.is_inf()
            || value != value.floor()
            || value > max_exact_integer
            || value < -max_exact_integer
        {
            flag[0].fetch_min(ABSOLUTE_POS as u32);
        }
    }
}

#[cube(launch_unchecked)]
pub fn extract_invalid_float_index_kernel<F: Float>(
    indices: &Tensor<F>,
    flag: &Array<Atomic<u32>>,
    flag_values: &mut Array<F>,
) {
    // INVARIANT: exactly one worker reads the scalar flag and, when invalid, extracts one
    // selected index value; this is O(1) work and does not scan the tensor serially.
    if ABSOLUTE_POS == 0 {
        let invalid_index = flag[0].load();
        if invalid_index != u32::MAX {
            flag_values[1] = indices[invalid_index as usize];
        }
    }
}

#[cube]
pub(crate) fn clamp_window_start<I: Numeric + CubePrimitive>(
    start: I,
    dim_size: usize,
    window_size: usize,
) -> usize {
    let max_start = dim_size.saturating_sub(window_size);
    let mut clamped = max_start;
    if start <= I::from_int(0) {
        clamped = 0usize;
    } else {
        let raw = usize::cast_from(start);
        if raw < max_start {
            clamped = raw;
        }
    }
    clamped
}

#[cube]
pub(crate) fn index_component<I: Numeric + CubePrimitive>(
    indices: &Tensor<I>,
    batch_idx: &Array<usize>,
    #[comptime] index_vector_dim: usize,
    #[comptime] component: usize,
    #[comptime] rank: usize,
) -> I {
    if index_vector_dim == rank {
        indices[multi_to_tensor_index(batch_idx, indices, rank)]
    } else {
        let mut full_idx = Array::<usize>::new(rank);
        let mut batch_axis = 0usize;
        #[unroll]
        for axis in 0..rank {
            if axis == index_vector_dim {
                full_idx[axis] = component;
            } else {
                full_idx[axis] = batch_idx[batch_axis];
                batch_axis += 1;
            }
        }
        indices[multi_to_tensor_index(&full_idx, indices, rank)]
    }
}

#[cube]
pub(crate) fn flat_to_index_batch_index<I: CubePrimitive>(
    mut flat: usize,
    indices: &Tensor<I>,
    #[comptime] index_vector_dim: usize,
    #[comptime] indices_rank: usize,
) -> Array<usize> {
    let batch_rank = comptime! {
        if index_vector_dim < indices_rank {
            indices_rank - 1
        } else {
            indices_rank
        }
    };
    let batch_buf_len = comptime! {
        if batch_rank == 0 { 1 } else { batch_rank }
    };
    let mut batch_idx = Array::<usize>::new(batch_buf_len);
    let mut batch_axis = 0usize;
    #[unroll]
    for axis in 0..indices_rank {
        if axis != index_vector_dim {
            let dim = indices.shape(axis);
            batch_idx[batch_axis] = flat % dim;
            flat /= dim;
            batch_axis += 1;
        }
    }
    batch_idx
}

#[cube]
pub(crate) fn flat_to_update_window_index<E: CubePrimitive>(
    mut flat: usize,
    updates: &Tensor<E>,
    #[comptime] update_window_dims: Sequence<usize>,
) -> Array<usize> {
    let window_rank = update_window_dims.len();
    let window_buf_len = comptime! {
        if window_rank == 0 { 1 } else { window_rank }
    };
    let mut window_idx = Array::<usize>::new(window_buf_len);
    #[unroll]
    for pos in 0..window_rank {
        let axis = comptime! { *update_window_dims.index(pos) };
        let dim = updates.shape(axis);
        window_idx[pos] = flat % dim;
        flat /= dim;
    }
    window_idx
}

#[cube]
pub(crate) fn update_window_len<E: CubePrimitive>(
    updates: &Tensor<E>,
    #[comptime] update_window_dims: Sequence<usize>,
) -> usize {
    let mut total = 1usize;
    #[unroll]
    for pos in 0..update_window_dims.len() {
        let axis = comptime! { *update_window_dims.index(pos) };
        total *= updates.shape(axis);
    }
    total
}

#[cube(launch_unchecked)]
pub fn slice_kernel<E: CubePrimitive>(
    out: &mut Tensor<E>,
    input: &Tensor<E>,
    #[comptime] starts: Sequence<usize>,
    #[comptime] strides: Sequence<usize>,
) {
    if ABSOLUTE_POS < out.len() {
        let rank = starts.len();
        let out_idx = flat_to_tensor_index(ABSOLUTE_POS, out, rank);
        let mut input_idx = Array::<usize>::new(rank);
        #[unroll]
        for axis in 0..rank {
            input_idx[axis] = comptime! { *starts.index(axis) }
                + out_idx[axis] * comptime! { *strides.index(axis) };
        }
        out[ABSOLUTE_POS] = input[multi_to_tensor_index(&input_idx, input, rank)];
    }
}

#[cube(launch_unchecked)]
pub fn dynamic_slice_kernel<E: CubePrimitive, I: Numeric + CubePrimitive>(
    out: &mut Tensor<E>,
    input: &Tensor<E>,
    starts: &Tensor<I>,
    #[comptime] slice_sizes: Sequence<usize>,
) {
    if ABSOLUTE_POS < out.len() {
        let rank = slice_sizes.len();
        let out_idx = flat_to_tensor_index(ABSOLUTE_POS, out, rank);
        let mut input_idx = Array::<usize>::new(rank);
        #[unroll]
        for axis in 0..rank {
            let start = starts[axis];
            let dim_size = input.shape(axis);
            let window_size = comptime! { *slice_sizes.index(axis) };
            input_idx[axis] = clamp_window_start::<I>(start, dim_size, window_size) + out_idx[axis];
        }
        out[ABSOLUTE_POS] = input[multi_to_tensor_index(&input_idx, input, rank)];
    }
}

#[cube(launch_unchecked)]
pub fn pad_kernel<E: CubePrimitive>(
    out: &mut Tensor<E>,
    input: &Tensor<E>,
    #[comptime] edge_padding_low: Sequence<i64>,
    #[comptime] interior_padding: Sequence<i64>,
) {
    if ABSOLUTE_POS < out.len() {
        let rank = edge_padding_low.len();
        let out_idx = flat_to_tensor_index(ABSOLUTE_POS, out, rank);
        let mut input_idx = Array::<usize>::new(rank);
        let mut in_bounds = true;
        #[unroll]
        for axis in 0..rank {
            let low = comptime! { *edge_padding_low.index(axis) };
            let low_magnitude = comptime! { low.unsigned_abs() };
            let spacing = comptime! { (*interior_padding.index(axis) + 1) as u64 };
            let out_pos = out_idx[axis] as u64;
            let mut shifted = 0_u64;
            if comptime! { low < 0 } {
                shifted = out_pos + low_magnitude;
            } else if out_pos < low_magnitude {
                in_bounds = false;
            } else {
                shifted = out_pos - low_magnitude;
            }
            if shifted % spacing != 0 {
                in_bounds = false;
            } else {
                let candidate = shifted / spacing;
                if candidate >= input.shape(axis) as u64 {
                    in_bounds = false;
                } else {
                    input_idx[axis] = candidate as usize;
                }
            }
        }
        out[ABSOLUTE_POS] = if in_bounds {
            input[multi_to_tensor_index(&input_idx, input, rank)]
        } else {
            zero_value::<E>()
        };
    }
}

#[cube(launch_unchecked)]
pub fn gather_kernel<E: CubePrimitive, I: Numeric + CubePrimitive>(
    out: &mut Tensor<E>,
    operand: &Tensor<E>,
    start_indices: &Tensor<I>,
    #[comptime] window_dims: Sequence<usize>,
    #[comptime] offset_dims: Sequence<usize>,
    #[comptime] start_index_map: Sequence<usize>,
    #[comptime] slice_sizes: Sequence<usize>,
    #[comptime] index_vector_dim: usize,
    #[comptime] operand_rank: usize,
    #[comptime] out_rank: usize,
    #[comptime] start_indices_rank: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let out_idx = flat_to_tensor_index(ABSOLUTE_POS, out, out_rank);
        let batch_rank = comptime! {
            if index_vector_dim < start_indices_rank {
                start_indices_rank - 1
            } else {
                start_indices_rank
            }
        };
        let batch_buf_len = comptime! {
            if batch_rank == 0 { 1 } else { batch_rank }
        };
        let mut batch_idx = Array::<usize>::new(batch_buf_len);
        let mut window_offsets = Array::<usize>::new(operand_rank);
        #[unroll]
        for axis in 0..operand_rank {
            window_offsets[axis] = 0;
        }

        let mut batch_axis = 0usize;
        #[unroll]
        for out_axis in 0..out_rank {
            let mut mapped = false;
            #[unroll]
            for offset_pos in 0..offset_dims.len() {
                if out_axis == comptime! { *offset_dims.index(offset_pos) } {
                    let operand_dim = comptime! { *window_dims.index(offset_pos) };
                    window_offsets[operand_dim] = out_idx[out_axis];
                    mapped = true;
                }
            }
            if !mapped {
                batch_idx[batch_axis] = out_idx[out_axis];
                batch_axis += 1;
            }
        }

        let mut operand_idx = Array::<usize>::new(operand_rank);
        #[unroll]
        for axis in 0..operand_rank {
            operand_idx[axis] = 0;
        }

        #[unroll]
        for component in 0..start_index_map.len() {
            let operand_dim = comptime! { *start_index_map.index(component) };
            let start = index_component(
                start_indices,
                &batch_idx,
                index_vector_dim,
                component,
                start_indices_rank,
            );
            let dim_size = operand.shape(operand_dim);
            let window_size = comptime! { *slice_sizes.index(operand_dim) };
            operand_idx[operand_dim] = clamp_window_start::<I>(start, dim_size, window_size);
        }

        #[unroll]
        for axis in 0..operand_rank {
            operand_idx[axis] += window_offsets[axis];
        }
        out[ABSOLUTE_POS] = operand[multi_to_tensor_index(&operand_idx, operand, operand_rank)];
    }
}

#[cube(launch_unchecked)]
pub fn scatter_copy_kernel<E: CubePrimitive>(out: &mut Tensor<E>, operand: &Tensor<E>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = operand[ABSOLUTE_POS];
    }
}

#[cube(launch_unchecked)]
pub fn scatter_float_kernel<E: Float, I: Numeric + CubePrimitive>(
    // Atomic<E> is a CubeCL storage view over the same dense output allocation;
    // the host launch path gates it with `ensure_atomic_add_supported`.
    out_parts: &mut Array<Atomic<E>>,
    operand: &Tensor<E>,
    scatter_indices: &Tensor<I>,
    updates: &Tensor<E>,
    #[comptime] window_dims: Sequence<usize>,
    #[comptime] update_window_dims: Sequence<usize>,
    #[comptime] scatter_dims_to_operand_dims: Sequence<usize>,
    #[comptime] index_vector_dim: usize,
    #[comptime] operand_rank: usize,
    #[comptime] updates_rank: usize,
    #[comptime] scatter_indices_rank: usize,
) {
    // INVARIANT: `scatter_update_len` returns the checked batch-window product, including zero;
    // `scatter_float_typed` returns before launch when that checked length is zero.
    let window_iters = update_window_len(updates, update_window_dims.clone());
    let update_iters = updates.len();
    if ABSOLUTE_POS < update_iters {
        let batch_flat = ABSOLUTE_POS / window_iters;
        let window_flat = ABSOLUTE_POS % window_iters;
        let batch_idx = flat_to_index_batch_index(
            batch_flat,
            scatter_indices,
            index_vector_dim,
            scatter_indices_rank,
        );
        let window_idx =
            flat_to_update_window_index(window_flat, updates, update_window_dims.clone());
        let mut update_idx = Array::<usize>::new(updates_rank);
        let mut operand_base = Array::<usize>::new(operand_rank);
        let mut operand_idx = Array::<usize>::new(operand_rank);
        let mut window_shape = Array::<usize>::new(operand_rank);

        #[unroll]
        for axis in 0..operand_rank {
            window_shape[axis] = 1;
            operand_base[axis] = 0;
            operand_idx[axis] = 0;
        }
        #[unroll]
        for pos in 0..window_dims.len() {
            let operand_axis = comptime! { *window_dims.index(pos) };
            let update_axis = comptime! { *update_window_dims.index(pos) };
            window_shape[operand_axis] = updates.shape(update_axis);
        }

        let mut window_fits = true;
        #[unroll]
        for component in 0..scatter_dims_to_operand_dims.len() {
            let operand_dim = comptime! { *scatter_dims_to_operand_dims.index(component) };
            let start = index_component(
                scatter_indices,
                &batch_idx,
                index_vector_dim,
                component,
                scatter_indices_rank,
            );
            operand_base[operand_dim] = clamp_window_start::<I>(
                start,
                operand.shape(operand_dim),
                window_shape[operand_dim],
            );
        }
        if window_fits {
            #[unroll]
            for axis in 0..operand_rank {
                if operand_base[axis] + window_shape[axis] > operand.shape(axis) {
                    window_fits = false;
                }
            }
        }

        if window_fits {
            let mut batch_axis = 0usize;
            #[unroll]
            for axis in 0..updates_rank {
                if axis_in_sequence(update_window_dims.clone(), axis) {
                    let pos = axis_position_in_sequence(update_window_dims.clone(), axis);
                    update_idx[axis] = window_idx[pos];
                } else {
                    update_idx[axis] = batch_idx[batch_axis];
                    batch_axis += 1;
                }
            }

            #[unroll]
            for axis in 0..operand_rank {
                operand_idx[axis] = operand_base[axis];
            }
            #[unroll]
            for pos in 0..window_dims.len() {
                let operand_axis = comptime! { *window_dims.index(pos) };
                operand_idx[operand_axis] += window_idx[pos];
            }

            let dst = multi_to_tensor_index(&operand_idx, operand, operand_rank);
            let src = multi_to_tensor_index(&update_idx, updates, updates_rank);
            out_parts[dst].fetch_add(updates[src]);
        }
    }
}

#[cube(launch_unchecked)]
pub fn scatter_complex_kernel<E: ComplexCore, F: Float, I: Numeric + CubePrimitive>(
    out_parts: &mut Array<Atomic<F>>,
    operand: &Tensor<E>,
    scatter_indices: &Tensor<I>,
    updates: &Tensor<E>,
    update_parts: &Array<F>,
    #[comptime] window_dims: Sequence<usize>,
    #[comptime] update_window_dims: Sequence<usize>,
    #[comptime] scatter_dims_to_operand_dims: Sequence<usize>,
    #[comptime] index_vector_dim: usize,
    #[comptime] operand_rank: usize,
    #[comptime] updates_rank: usize,
    #[comptime] scatter_indices_rank: usize,
) {
    // INVARIANT: `scatter_update_len` returns the checked batch-window product, including zero;
    // `scatter_complex_typed` returns before launch when that checked length is zero.
    let window_iters = update_window_len(updates, update_window_dims.clone());
    let update_iters = updates.len();
    if ABSOLUTE_POS < update_iters {
        let batch_flat = ABSOLUTE_POS / window_iters;
        let window_flat = ABSOLUTE_POS % window_iters;
        let batch_idx = flat_to_index_batch_index(
            batch_flat,
            scatter_indices,
            index_vector_dim,
            scatter_indices_rank,
        );
        let window_idx =
            flat_to_update_window_index(window_flat, updates, update_window_dims.clone());
        let mut update_idx = Array::<usize>::new(updates_rank);
        let mut operand_base = Array::<usize>::new(operand_rank);
        let mut operand_idx = Array::<usize>::new(operand_rank);
        let mut window_shape = Array::<usize>::new(operand_rank);

        #[unroll]
        for axis in 0..operand_rank {
            window_shape[axis] = 1;
            operand_base[axis] = 0;
            operand_idx[axis] = 0;
        }
        #[unroll]
        for pos in 0..window_dims.len() {
            let operand_axis = comptime! { *window_dims.index(pos) };
            let update_axis = comptime! { *update_window_dims.index(pos) };
            window_shape[operand_axis] = updates.shape(update_axis);
        }

        let mut window_fits = true;
        #[unroll]
        for component in 0..scatter_dims_to_operand_dims.len() {
            let operand_dim = comptime! { *scatter_dims_to_operand_dims.index(component) };
            let start = index_component(
                scatter_indices,
                &batch_idx,
                index_vector_dim,
                component,
                scatter_indices_rank,
            );
            operand_base[operand_dim] = clamp_window_start::<I>(
                start,
                operand.shape(operand_dim),
                window_shape[operand_dim],
            );
        }
        if window_fits {
            #[unroll]
            for axis in 0..operand_rank {
                if operand_base[axis] + window_shape[axis] > operand.shape(axis) {
                    window_fits = false;
                }
            }
        }

        if window_fits {
            let mut batch_axis = 0usize;
            #[unroll]
            for axis in 0..updates_rank {
                if axis_in_sequence(update_window_dims.clone(), axis) {
                    let pos = axis_position_in_sequence(update_window_dims.clone(), axis);
                    update_idx[axis] = window_idx[pos];
                } else {
                    update_idx[axis] = batch_idx[batch_axis];
                    batch_axis += 1;
                }
            }

            #[unroll]
            for axis in 0..operand_rank {
                operand_idx[axis] = operand_base[axis];
            }
            #[unroll]
            for pos in 0..window_dims.len() {
                let operand_axis = comptime! { *window_dims.index(pos) };
                operand_idx[operand_axis] += window_idx[pos];
            }

            let dst = multi_to_tensor_index(&operand_idx, operand, operand_rank) * 2;
            let src = multi_to_tensor_index(&update_idx, updates, updates_rank) * 2;
            out_parts[dst].fetch_add(update_parts[src]);
            out_parts[dst + 1].fetch_add(update_parts[src + 1]);
        }
    }
}
