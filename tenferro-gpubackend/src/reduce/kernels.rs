// This file implements tenferro-specific CubeCL reduction kernels. The launch
// strategy and reduction split are adapted from cubek-reduce; see
// tenferro-gpubackend/THIRD_PARTY_NOTICES.md for upstream copyright, source paths,
// commit, and license notice text.

//! CubeCL reduction kernel definitions.

use cubecl::prelude::*;

macro_rules! reduce_binary_kernel {
    ($name:ident, $bound:ident, $op:tt) => {
        #[cube(launch_unchecked)]
        pub(crate) fn $name<T: $bound>(
            input: &Tensor<T>,
            output: &mut Tensor<T>,
            axis: usize,
            output_len: usize,
        ) {
            let output_index = ABSOLUTE_POS as usize;
            if output_index < output_len {
                let rank = output.rank();
                let mut remaining = output_index;
                let mut input_base_offset = 0usize;
                let mut output_offset = 0usize;

                for dim in 0..rank {
                    let dim_len = output.shape(dim);
                    let coord = remaining % dim_len;
                    remaining /= dim_len;
                    input_base_offset += coord * input.stride(dim);
                    output_offset += coord * output.stride(dim);
                }

                let axis_stride = input.stride(axis);
                let reduce_len = input.shape(axis);
                let mut acc = input[input_base_offset];

                for reduce_index in 1..reduce_len {
                    let input_offset = input_base_offset + reduce_index * axis_stride;
                    acc = acc $op input[input_offset];
                }

                output[output_offset] = acc;
            }
        }
    };
}

reduce_binary_kernel!(reduce_sum_float, Float, +);
reduce_binary_kernel!(reduce_sum_int, Int, +);
reduce_binary_kernel!(reduce_sum_complex, Complex, +);
reduce_binary_kernel!(reduce_prod_float, Float, *);
reduce_binary_kernel!(reduce_prod_int, Int, *);
reduce_binary_kernel!(reduce_prod_complex, Complex, *);

#[cube(launch_unchecked)]
pub(crate) fn reduce_max_float<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
    axis: usize,
    output_len: usize,
) {
    let output_index = ABSOLUTE_POS as usize;
    if output_index < output_len {
        let rank = output.rank();
        let mut remaining = output_index;
        let mut input_base_offset = 0usize;
        let mut output_offset = 0usize;

        for dim in 0..rank {
            let dim_len = output.shape(dim);
            let coord = remaining % dim_len;
            remaining /= dim_len;
            input_base_offset += coord * input.stride(dim);
            output_offset += coord * output.stride(dim);
        }

        let axis_stride = input.stride(axis);
        let reduce_len = input.shape(axis);
        let mut acc = input[input_base_offset];

        for reduce_index in 1..reduce_len {
            let input_offset = input_base_offset + reduce_index * axis_stride;
            acc = acc.max(input[input_offset]);
        }

        output[output_offset] = acc;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn reduce_min_float<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
    axis: usize,
    output_len: usize,
) {
    let output_index = ABSOLUTE_POS as usize;
    if output_index < output_len {
        let rank = output.rank();
        let mut remaining = output_index;
        let mut input_base_offset = 0usize;
        let mut output_offset = 0usize;

        for dim in 0..rank {
            let dim_len = output.shape(dim);
            let coord = remaining % dim_len;
            remaining /= dim_len;
            input_base_offset += coord * input.stride(dim);
            output_offset += coord * output.stride(dim);
        }

        let axis_stride = input.stride(axis);
        let reduce_len = input.shape(axis);
        let mut acc = input[input_base_offset];

        for reduce_index in 1..reduce_len {
            let input_offset = input_base_offset + reduce_index * axis_stride;
            acc = acc.min(input[input_offset]);
        }

        output[output_offset] = acc;
    }
}
