use cubecl::prelude::*;

use crate::helpers::{flat_to_multi_index, multi_to_flat_index, zero_value};

#[cube(launch_unchecked)]
pub fn extract_diagonal_kernel<E: CubePrimitive>(
    out: &mut Array<E>,
    input: &Array<E>,
    #[comptime] input_shape: Sequence<usize>,
    #[comptime] output_shape: Sequence<usize>,
    #[comptime] axis_a: usize,
    #[comptime] axis_b: usize,
    #[comptime] diag_output_axis: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let out_idx = flat_to_multi_index(ABSOLUTE_POS, output_shape.clone());
        let rank = input_shape.len();
        let mut input_idx = Array::<usize>::new(rank);
        let diag = out_idx[diag_output_axis];
        let mut out_axis = 0usize;
        #[unroll]
        for axis in 0..rank {
            if axis == axis_a || axis == axis_b {
                input_idx[axis] = diag;
            } else {
                input_idx[axis] = out_idx[out_axis];
                out_axis += 1;
            }
        }
        out[ABSOLUTE_POS] = input[multi_to_flat_index(&input_idx, input_shape)];
    }
}

#[cube(launch_unchecked)]
pub fn embed_diagonal_copy_kernel<E: CubePrimitive>(
    out: &mut Array<E>,
    input: &Array<E>,
    #[comptime] input_shape: Sequence<usize>,
    #[comptime] output_shape: Sequence<usize>,
    #[comptime] axis_a: usize,
    #[comptime] axis_b: usize,
) {
    if ABSOLUTE_POS < input.len() {
        let input_idx = flat_to_multi_index(ABSOLUTE_POS, input_shape.clone());
        let out_rank = output_shape.len();
        let mut output_idx = Array::<usize>::new(out_rank);
        let diag_val = input_idx[axis_a];
        let mut src_axis = 0usize;
        #[unroll]
        for out_axis in 0..out_rank {
            if out_axis == axis_b {
                output_idx[out_axis] = diag_val;
            } else {
                output_idx[out_axis] = input_idx[src_axis];
                src_axis += 1;
            }
        }
        let dst = multi_to_flat_index(&output_idx, output_shape);
        out[dst] = input[ABSOLUTE_POS];
    }
}

#[cube(launch_unchecked)]
pub fn tril_kernel<E: CubePrimitive>(
    out: &mut Array<E>,
    input: &Array<E>,
    #[comptime] shape: Sequence<usize>,
    k: i64,
) {
    if ABSOLUTE_POS < out.len() {
        let idx = flat_to_multi_index(ABSOLUTE_POS, shape);
        let row = idx[0] as i64;
        let col = idx[1] as i64;
        let boundary = col - k;
        out[ABSOLUTE_POS] = if row >= boundary {
            input[ABSOLUTE_POS]
        } else {
            zero_value::<E>()
        };
    }
}

#[cube(launch_unchecked)]
pub fn triu_kernel<E: CubePrimitive>(
    out: &mut Array<E>,
    input: &Array<E>,
    #[comptime] shape: Sequence<usize>,
    k: i64,
) {
    if ABSOLUTE_POS < out.len() {
        let idx = flat_to_multi_index(ABSOLUTE_POS, shape);
        let row = idx[0] as i64;
        let col = idx[1] as i64;
        let boundary = col - k;
        out[ABSOLUTE_POS] = if row <= boundary {
            input[ABSOLUTE_POS]
        } else {
            zero_value::<E>()
        };
    }
}
