use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn pack_lhs_dot_general<E: CubePrimitive>(
    out: &mut Tensor<E>,
    input: &Tensor<E>,
    #[comptime] free_axes: Sequence<usize>,
    #[comptime] contract_axes: Sequence<usize>,
    #[comptime] batch_axes: Sequence<usize>,
    #[comptime] input_rank: usize,
    #[comptime] out_rank: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let mut input_idx = Array::<usize>::new(input_rank);

        #[unroll]
        for batch_pos in 0..batch_axes.len() {
            let input_axis = comptime! { *batch_axes.index(batch_pos) };
            input_idx[input_axis] = out.coordinate(ABSOLUTE_POS, batch_pos);
        }

        let mut free_flat = out.coordinate(ABSOLUTE_POS, out_rank - 2);
        #[unroll]
        for pos in 0..free_axes.len() {
            let input_axis = comptime! { *free_axes.index(pos) };
            let dim = input.shape(input_axis);
            input_idx[input_axis] = free_flat % dim;
            free_flat /= dim;
        }

        let mut contract_flat = out.coordinate(ABSOLUTE_POS, out_rank - 1);
        #[unroll]
        for pos in 0..contract_axes.len() {
            let input_axis = comptime! { *contract_axes.index(pos) };
            let dim = input.shape(input_axis);
            input_idx[input_axis] = contract_flat % dim;
            contract_flat /= dim;
        }

        let mut input_offset = 0usize;
        #[unroll]
        for axis in 0..input_rank {
            input_offset += input_idx[axis] * input.stride(axis);
        }
        out[ABSOLUTE_POS] = input[input_offset];
    }
}

#[cube(launch_unchecked)]
pub fn pack_rhs_dot_general<E: CubePrimitive>(
    out: &mut Tensor<E>,
    input: &Tensor<E>,
    #[comptime] contract_axes: Sequence<usize>,
    #[comptime] free_axes: Sequence<usize>,
    #[comptime] batch_axes: Sequence<usize>,
    #[comptime] input_rank: usize,
    #[comptime] out_rank: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let mut input_idx = Array::<usize>::new(input_rank);

        #[unroll]
        for batch_pos in 0..batch_axes.len() {
            let input_axis = comptime! { *batch_axes.index(batch_pos) };
            input_idx[input_axis] = out.coordinate(ABSOLUTE_POS, batch_pos);
        }

        let mut contract_flat = out.coordinate(ABSOLUTE_POS, out_rank - 2);
        #[unroll]
        for pos in 0..contract_axes.len() {
            let input_axis = comptime! { *contract_axes.index(pos) };
            let dim = input.shape(input_axis);
            input_idx[input_axis] = contract_flat % dim;
            contract_flat /= dim;
        }

        let mut free_flat = out.coordinate(ABSOLUTE_POS, out_rank - 1);
        #[unroll]
        for pos in 0..free_axes.len() {
            let input_axis = comptime! { *free_axes.index(pos) };
            let dim = input.shape(input_axis);
            input_idx[input_axis] = free_flat % dim;
            free_flat /= dim;
        }

        let mut input_offset = 0usize;
        #[unroll]
        for axis in 0..input_rank {
            input_offset += input_idx[axis] * input.stride(axis);
        }
        out[ABSOLUTE_POS] = input[input_offset];
    }
}
