use cubecl::prelude::*;

use crate::helpers::{
    axis_in_sequence, axis_position_in_sequence, flat_to_multi_index, multi_to_flat_index,
    shape_product, zero_value,
};

#[cube]
pub(crate) fn clamp_window_start<I: Float>(start: I, dim_size: usize, window_size: usize) -> usize {
    let max_start = dim_size - window_size;
    let mut clamped = max_start;
    if start <= I::new(0.0) {
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
pub(crate) fn index_component<I: Float>(
    indices: &Array<I>,
    #[comptime] indices_shape: Sequence<usize>,
    batch_idx: &Array<usize>,
    #[comptime] index_vector_dim: usize,
    #[comptime] component: usize,
) -> I {
    let rank = indices_shape.len();
    if index_vector_dim == rank {
        indices[multi_to_flat_index(batch_idx, indices_shape)]
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
        indices[multi_to_flat_index(&full_idx, indices_shape)]
    }
}

#[cube(launch_unchecked)]
pub fn slice_kernel<E: CubePrimitive>(
    out: &mut Array<E>,
    input: &Array<E>,
    #[comptime] input_shape: Sequence<usize>,
    #[comptime] output_shape: Sequence<usize>,
    #[comptime] starts: Sequence<usize>,
    #[comptime] strides: Sequence<usize>,
) {
    if ABSOLUTE_POS < out.len() {
        let out_idx = flat_to_multi_index(ABSOLUTE_POS, output_shape.clone());
        let rank = output_shape.len();
        let mut input_idx = Array::<usize>::new(rank);
        #[unroll]
        for axis in 0..rank {
            input_idx[axis] = comptime! { *starts.index(axis) }
                + out_idx[axis] * comptime! { *strides.index(axis) };
        }
        out[ABSOLUTE_POS] = input[multi_to_flat_index(&input_idx, input_shape)];
    }
}

#[cube(launch_unchecked)]
pub fn dynamic_slice_kernel<E: CubePrimitive, I: Float>(
    out: &mut Array<E>,
    input: &Array<E>,
    starts: &Array<I>,
    #[comptime] input_shape: Sequence<usize>,
    #[comptime] output_shape: Sequence<usize>,
) {
    if ABSOLUTE_POS < out.len() {
        let out_idx = flat_to_multi_index(ABSOLUTE_POS, output_shape.clone());
        let rank = output_shape.len();
        let mut input_idx = Array::<usize>::new(rank);
        #[unroll]
        for axis in 0..rank {
            let start = starts[axis];
            let dim_size = comptime! { *input_shape.index(axis) };
            let window_size = comptime! { *output_shape.index(axis) };
            input_idx[axis] = clamp_window_start::<I>(start, dim_size, window_size) + out_idx[axis];
        }
        out[ABSOLUTE_POS] = input[multi_to_flat_index(&input_idx, input_shape)];
    }
}

#[cube(launch_unchecked)]
pub fn pad_kernel<E: CubePrimitive>(
    out: &mut Array<E>,
    input: &Array<E>,
    #[comptime] input_shape: Sequence<usize>,
    #[comptime] output_shape: Sequence<usize>,
    #[comptime] edge_padding_low: Sequence<i64>,
    #[comptime] interior_padding: Sequence<i64>,
) {
    if ABSOLUTE_POS < out.len() {
        let out_idx = flat_to_multi_index(ABSOLUTE_POS, output_shape.clone());
        let rank = input_shape.len();
        let mut input_idx = Array::<usize>::new(rank);
        let mut in_bounds = true;
        #[unroll]
        for axis in 0..rank {
            let low = comptime! { *edge_padding_low.index(axis) };
            let spacing = comptime! { *interior_padding.index(axis) } + 1;
            let shifted = out_idx[axis] as i64 - low;
            if shifted < 0 || shifted % spacing != 0 {
                in_bounds = false;
            } else {
                let candidate = (shifted / spacing) as usize;
                if candidate >= comptime! { *input_shape.index(axis) } {
                    in_bounds = false;
                } else {
                    input_idx[axis] = candidate;
                }
            }
        }
        out[ABSOLUTE_POS] = if in_bounds {
            input[multi_to_flat_index(&input_idx, input_shape)]
        } else {
            zero_value::<E>()
        };
    }
}

#[cube(launch_unchecked)]
pub fn gather_kernel<E: CubePrimitive, I: Float>(
    out: &mut Array<E>,
    operand: &Array<E>,
    start_indices: &Array<I>,
    #[comptime] operand_shape: Sequence<usize>,
    #[comptime] output_shape: Sequence<usize>,
    #[comptime] batch_shape: Sequence<usize>,
    #[comptime] window_dims: Sequence<usize>,
    #[comptime] offset_dims: Sequence<usize>,
    #[comptime] start_index_map: Sequence<usize>,
    #[comptime] start_indices_shape: Sequence<usize>,
    #[comptime] slice_sizes: Sequence<usize>,
    #[comptime] index_vector_dim: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let out_idx = flat_to_multi_index(ABSOLUTE_POS, output_shape.clone());
        let operand_rank = operand_shape.len();
        let batch_rank = batch_shape.len();
        let out_rank = output_shape.len();
        let mut batch_idx = Array::<usize>::new(batch_rank);
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
                start_indices_shape.clone(),
                &batch_idx,
                index_vector_dim,
                component,
            );
            let dim_size = comptime! { *operand_shape.index(operand_dim) };
            let window_size = comptime! { *slice_sizes.index(operand_dim) };
            operand_idx[operand_dim] = clamp_window_start::<I>(start, dim_size, window_size);
        }

        #[unroll]
        for axis in 0..operand_rank {
            operand_idx[axis] += window_offsets[axis];
        }
        out[ABSOLUTE_POS] = operand[multi_to_flat_index(&operand_idx, operand_shape)];
    }
}

macro_rules! scatter_kernel {
    ($float_name:ident, $complex_name:ident, $ty:ident) => {
        #[cube(launch_unchecked)]
        pub fn $float_name<E: Float, I: Float>(
            out: &mut Array<E>,
            operand: &Array<E>,
            scatter_indices: &Array<I>,
            updates: &Array<E>,
            #[comptime] operand_shape: Sequence<usize>,
            #[comptime] updates_shape: Sequence<usize>,
            #[comptime] scatter_indices_shape: Sequence<usize>,
            #[comptime] batch_shape: Sequence<usize>,
            #[comptime] window_dims: Sequence<usize>,
            #[comptime] update_window_dims: Sequence<usize>,
            #[comptime] scatter_dims_to_operand_dims: Sequence<usize>,
            #[comptime] index_vector_dim: usize,
            #[comptime] window_shape_updates: Sequence<usize>,
        ) {
            if ABSOLUTE_POS == 0 {
                // StableHLO add-scatter: initialise the output from a copy of
                // `operand` so non-scattered slots preserve the operand values.
                for pos in 0..out.len() {
                    out[pos] = operand[pos];
                }

                let batch_iters = shape_product(batch_shape.clone());
                let window_iters = shape_product(window_shape_updates.clone());
                let batch_rank = batch_shape.len();
                let batch_buf_len = if batch_rank == 0 { 1 } else { batch_rank };
                let operand_rank = operand_shape.len();
                let update_rank = updates_shape.len();
                let window_rank = window_shape_updates.len();
                let window_buf_len = if window_rank == 0 { 1 } else { window_rank };
                let mut batch_idx = Array::<usize>::new(batch_buf_len);
                let mut window_idx = Array::<usize>::new(window_buf_len);
                let mut update_idx = Array::<usize>::new(update_rank);
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
                    window_shape[operand_axis] = comptime! { *window_shape_updates.index(pos) };
                }

                for batch_flat in 0..batch_iters {
                    if batch_rank > 0 {
                        let next_batch_idx = flat_to_multi_index(batch_flat, batch_shape.clone());
                        #[unroll]
                        for axis in 0..batch_rank {
                            batch_idx[axis] = next_batch_idx[axis];
                        }
                    }

                    let mut window_fits = true;
                    #[unroll]
                    for axis in 0..operand_rank {
                        operand_base[axis] = 0;
                    }

                    #[unroll]
                    for component in 0..scatter_dims_to_operand_dims.len() {
                        let operand_dim =
                            comptime! { *scatter_dims_to_operand_dims.index(component) };
                        let start = index_component(
                            scatter_indices,
                            scatter_indices_shape.clone(),
                            &batch_idx,
                            index_vector_dim,
                            component,
                        );
                        if start < I::new(0.0) {
                            window_fits = false;
                        } else {
                            operand_base[operand_dim] = usize::cast_from(start);
                        }
                    }
                    if window_fits {
                        #[unroll]
                        for axis in 0..operand_rank {
                            if operand_base[axis] + window_shape[axis]
                                > comptime! { *operand_shape.index(axis) }
                            {
                                window_fits = false;
                            }
                        }
                    }

                    if window_fits {
                        for window_flat in 0..window_iters {
                            if window_rank > 0 {
                                let next_window_idx =
                                    flat_to_multi_index(window_flat, window_shape_updates.clone());
                                #[unroll]
                                for axis in 0..window_rank {
                                    window_idx[axis] = next_window_idx[axis];
                                }
                            }

                            let mut batch_axis = 0usize;
                            #[unroll]
                            for axis in 0..update_rank {
                                if axis_in_sequence(update_window_dims.clone(), axis) {
                                    let pos =
                                        axis_position_in_sequence(update_window_dims.clone(), axis);
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

                            let dst = multi_to_flat_index(&operand_idx, operand_shape.clone());
                            let src = multi_to_flat_index(&update_idx, updates_shape.clone());
                            out[dst] = out[dst] + updates[src];
                        }
                    }
                }
            }
        }

        #[cube(launch_unchecked)]
        pub fn $complex_name<E: Complex, I: Float>(
            out: &mut Array<E>,
            operand: &Array<E>,
            scatter_indices: &Array<I>,
            updates: &Array<E>,
            #[comptime] operand_shape: Sequence<usize>,
            #[comptime] updates_shape: Sequence<usize>,
            #[comptime] scatter_indices_shape: Sequence<usize>,
            #[comptime] batch_shape: Sequence<usize>,
            #[comptime] window_dims: Sequence<usize>,
            #[comptime] update_window_dims: Sequence<usize>,
            #[comptime] scatter_dims_to_operand_dims: Sequence<usize>,
            #[comptime] index_vector_dim: usize,
            #[comptime] window_shape_updates: Sequence<usize>,
        ) {
            if ABSOLUTE_POS == 0 {
                // StableHLO add-scatter: initialise the output from a copy of
                // `operand` so non-scattered slots preserve the operand values.
                for pos in 0..out.len() {
                    out[pos] = operand[pos];
                }

                let batch_iters = shape_product(batch_shape.clone());
                let window_iters = shape_product(window_shape_updates.clone());
                let batch_rank = batch_shape.len();
                let batch_buf_len = if batch_rank == 0 { 1 } else { batch_rank };
                let operand_rank = operand_shape.len();
                let update_rank = updates_shape.len();
                let window_rank = window_shape_updates.len();
                let window_buf_len = if window_rank == 0 { 1 } else { window_rank };
                let mut batch_idx = Array::<usize>::new(batch_buf_len);
                let mut window_idx = Array::<usize>::new(window_buf_len);
                let mut update_idx = Array::<usize>::new(update_rank);
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
                    window_shape[operand_axis] = comptime! { *window_shape_updates.index(pos) };
                }

                for batch_flat in 0..batch_iters {
                    if batch_rank > 0 {
                        let next_batch_idx = flat_to_multi_index(batch_flat, batch_shape.clone());
                        #[unroll]
                        for axis in 0..batch_rank {
                            batch_idx[axis] = next_batch_idx[axis];
                        }
                    }

                    let mut window_fits = true;
                    #[unroll]
                    for axis in 0..operand_rank {
                        operand_base[axis] = 0;
                    }

                    #[unroll]
                    for component in 0..scatter_dims_to_operand_dims.len() {
                        let operand_dim =
                            comptime! { *scatter_dims_to_operand_dims.index(component) };
                        let start = index_component(
                            scatter_indices,
                            scatter_indices_shape.clone(),
                            &batch_idx,
                            index_vector_dim,
                            component,
                        );
                        if start < I::new(0.0) {
                            window_fits = false;
                        } else {
                            operand_base[operand_dim] = usize::cast_from(start);
                        }
                    }
                    if window_fits {
                        #[unroll]
                        for axis in 0..operand_rank {
                            if operand_base[axis] + window_shape[axis]
                                > comptime! { *operand_shape.index(axis) }
                            {
                                window_fits = false;
                            }
                        }
                    }

                    if window_fits {
                        for window_flat in 0..window_iters {
                            if window_rank > 0 {
                                let next_window_idx =
                                    flat_to_multi_index(window_flat, window_shape_updates.clone());
                                #[unroll]
                                for axis in 0..window_rank {
                                    window_idx[axis] = next_window_idx[axis];
                                }
                            }

                            let mut batch_axis = 0usize;
                            #[unroll]
                            for axis in 0..update_rank {
                                if axis_in_sequence(update_window_dims.clone(), axis) {
                                    let pos =
                                        axis_position_in_sequence(update_window_dims.clone(), axis);
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

                            let dst = multi_to_flat_index(&operand_idx, operand_shape.clone());
                            let src = multi_to_flat_index(&update_idx, updates_shape.clone());
                            out[dst] = out[dst] + updates[src];
                        }
                    }
                }
            }
        }
    };
}

scatter_kernel!(scatter_float_kernel, scatter_complex_kernel, Float);
