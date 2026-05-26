use cubecl::prelude::*;

use crate::kernels::helpers::{flat_to_tensor_index, multi_to_tensor_index, zero_value};

#[cube(launch_unchecked)]
pub fn extract_diagonal_kernel<E: CubePrimitive>(
    out: &mut Tensor<E>,
    input: &Tensor<E>,
    #[comptime] axis_a: usize,
    #[comptime] axis_b: usize,
    #[comptime] diag_output_axis: usize,
    #[comptime] input_rank: usize,
    #[comptime] output_rank: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let out_idx = flat_to_tensor_index(ABSOLUTE_POS, out, output_rank);
        let mut input_idx = Array::<usize>::new(input_rank);
        let diag = out_idx[diag_output_axis];
        let mut out_axis = 0usize;
        #[unroll]
        for axis in 0..input_rank {
            if axis == axis_a || axis == axis_b {
                input_idx[axis] = diag;
            } else {
                input_idx[axis] = out_idx[out_axis];
                out_axis += 1;
            }
        }
        out[ABSOLUTE_POS] = input[multi_to_tensor_index(&input_idx, input, input_rank)];
    }
}

#[cube(launch_unchecked)]
pub fn embed_diagonal_copy_kernel<E: CubePrimitive>(
    out: &mut Tensor<E>,
    input: &Tensor<E>,
    #[comptime] axis_a: usize,
    #[comptime] axis_b: usize,
    #[comptime] input_rank: usize,
    #[comptime] output_rank: usize,
) {
    if ABSOLUTE_POS < input.len() {
        let input_idx = flat_to_tensor_index(ABSOLUTE_POS, input, input_rank);
        let mut output_idx = Array::<usize>::new(output_rank);
        let diag_val = input_idx[axis_a];
        let mut src_axis = 0usize;
        #[unroll]
        for out_axis in 0..output_rank {
            if out_axis == axis_b {
                output_idx[out_axis] = diag_val;
            } else {
                output_idx[out_axis] = input_idx[src_axis];
                src_axis += 1;
            }
        }
        let dst = multi_to_tensor_index(&output_idx, out, output_rank);
        out[dst] = input[ABSOLUTE_POS];
    }
}

#[cube(launch_unchecked)]
pub fn tril_kernel<E: CubePrimitive>(out: &mut Tensor<E>, input: &Tensor<E>, k: i64) {
    if ABSOLUTE_POS < out.len() {
        let row = out.coordinate(ABSOLUTE_POS, 0) as i64;
        let col = out.coordinate(ABSOLUTE_POS, 1) as i64;
        let boundary = col - k;
        out[ABSOLUTE_POS] = if row >= boundary {
            input[ABSOLUTE_POS]
        } else {
            zero_value::<E>()
        };
    }
}

#[cube(launch_unchecked)]
pub fn triu_kernel<E: CubePrimitive>(out: &mut Tensor<E>, input: &Tensor<E>, k: i64) {
    if ABSOLUTE_POS < out.len() {
        let row = out.coordinate(ABSOLUTE_POS, 0) as i64;
        let col = out.coordinate(ABSOLUTE_POS, 1) as i64;
        let boundary = col - k;
        out[ABSOLUTE_POS] = if row <= boundary {
            input[ABSOLUTE_POS]
        } else {
            zero_value::<E>()
        };
    }
}
