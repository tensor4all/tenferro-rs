// This file implements tenferro-specific CubeCL reduction kernels. The launch
// strategy and reduction split are adapted from cubek-reduce; see
// tenferro-gpu/THIRD_PARTY_NOTICES.md for upstream copyright, source paths,
// commit, and license notice text.

//! CubeCL reduction kernel definitions.

// CubeCL's kernel DSL uses operator tokens and typed launch indices that
// Clippy cannot model as ordinary host Rust expressions.
#![allow(clippy::assign_op_pattern, clippy::unnecessary_cast)]

use cubecl::prelude::*;

use crate::kernels::helpers::{
    nan_propagating_max, nan_propagating_min, plane_contains_nan, plane_propagate_nan,
    wrapping_add, wrapping_mul, wrapping_plane_prod, wrapping_plane_sum,
};

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
                let axis_len = input.shape(axis);
                let mut acc = input[input_base_offset];

                for reduce_index in 1..axis_len {
                    let input_offset = input_base_offset + reduce_index * axis_stride;
                    acc = acc $op input[input_offset];
                }

                output[output_offset] = acc;
            }
        }
    };
}

reduce_binary_kernel!(reduce_sum_float, Float, +);
reduce_binary_kernel!(reduce_sum_complex, ComplexCore, +);
reduce_binary_kernel!(reduce_prod_float, Float, *);
reduce_binary_kernel!(reduce_prod_complex, ComplexCore, *);

#[cube(launch_unchecked)]
pub(crate) fn reduce_sum_squares_float<F: Float>(
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
        let axis_len = input.shape(axis);
        let first = input[input_base_offset];
        // `fma(x, x, 0)` rounds the square before the later addition and
        // prevents NVRTC from contracting that addition with the multiply.
        let mut acc = fma(first, first, F::new(0.0_f32));

        for reduce_index in 1..axis_len {
            let input_offset = input_base_offset + reduce_index * axis_stride;
            let value = input[input_offset];
            let square = fma(value, value, F::new(0.0_f32));
            acc += square;
        }

        output[output_offset] = acc;
    }
}

macro_rules! reduce_wrapping_int_kernel {
    ($name:ident, $combine:ident) => {
        #[cube(launch_unchecked)]
        pub(crate) fn $name<I: Int>(
            input: &Tensor<I>,
            output: &mut Tensor<I>,
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
                let axis_len = input.shape(axis);
                let mut acc = input[input_base_offset];

                for reduce_index in 1..axis_len {
                    let input_offset = input_base_offset + reduce_index * axis_stride;
                    acc = $combine::<I>(acc, input[input_offset]);
                }

                output[output_offset] = acc;
            }
        }
    };
}

reduce_wrapping_int_kernel!(reduce_sum_int, wrapping_add);
reduce_wrapping_int_kernel!(reduce_prod_int, wrapping_mul);

macro_rules! reduce_binary_plane_kernel {
    ($name:ident, $bound:ident, $op:tt, $plane_reduce:ident) => {
        #[cube(launch_unchecked)]
        pub(crate) fn $name<T: $bound>(
            input: &Tensor<T>,
            output: &mut Tensor<T>,
            axis: usize,
            output_len: usize,
        ) {
            let output_index = CUBE_POS_X as usize;
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
                let axis_len = input.shape(axis);
                let plane_width = PLANE_DIM as usize;
                let mut reduce_index = UNIT_POS as usize;
                let mut acc = input[input_base_offset + reduce_index * axis_stride];

                reduce_index += plane_width;
                while reduce_index < axis_len {
                    let input_offset = input_base_offset + reduce_index * axis_stride;
                    acc = acc $op input[input_offset];
                    reduce_index += plane_width;
                }

                let reduced = $plane_reduce(acc);
                if UNIT_POS == 0 {
                    output[output_offset] = reduced;
                }
            }
        }
    };
}

reduce_binary_plane_kernel!(reduce_sum_float_plane, Float, +, plane_sum);
reduce_binary_plane_kernel!(reduce_sum_complex_plane, ComplexCore, +, plane_sum);
reduce_binary_plane_kernel!(reduce_prod_float_plane, Float, *, plane_prod);
reduce_binary_plane_kernel!(reduce_prod_complex_plane, ComplexCore, *, plane_prod);

#[cube(launch_unchecked)]
pub(crate) fn reduce_sum_squares_float_plane<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
    axis: usize,
    output_len: usize,
) {
    let output_index = CUBE_POS_X as usize;
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
        let axis_len = input.shape(axis);
        let plane_width = PLANE_DIM as usize;
        let mut reduce_index = UNIT_POS as usize;
        let first = input[input_base_offset + reduce_index * axis_stride];
        let mut acc = fma(first, first, F::new(0.0_f32));

        reduce_index += plane_width;
        while reduce_index < axis_len {
            let input_offset = input_base_offset + reduce_index * axis_stride;
            let value = input[input_offset];
            let square = fma(value, value, F::new(0.0_f32));
            acc += square;
            reduce_index += plane_width;
        }

        let reduced = plane_sum(acc);
        if UNIT_POS == 0 {
            output[output_offset] = reduced;
        }
    }
}

macro_rules! reduce_wrapping_int_plane_kernel {
    ($name:ident, $combine:ident, $plane_reduce:ident) => {
        #[cube(launch_unchecked)]
        pub(crate) fn $name<I: Int>(
            input: &Tensor<I>,
            output: &mut Tensor<I>,
            axis: usize,
            output_len: usize,
        ) {
            let output_index = CUBE_POS_X as usize;
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
                let axis_len = input.shape(axis);
                let plane_width = PLANE_DIM as usize;
                let mut reduce_index = UNIT_POS as usize;
                let mut acc = input[input_base_offset + reduce_index * axis_stride];

                reduce_index += plane_width;
                while reduce_index < axis_len {
                    let input_offset = input_base_offset + reduce_index * axis_stride;
                    acc = $combine::<I>(acc, input[input_offset]);
                    reduce_index += plane_width;
                }

                let reduced = $plane_reduce::<I>(acc);
                if UNIT_POS == 0 {
                    output[output_offset] = reduced;
                }
            }
        }
    };
}

reduce_wrapping_int_plane_kernel!(reduce_sum_int_plane, wrapping_add, wrapping_plane_sum);
reduce_wrapping_int_plane_kernel!(reduce_prod_int_plane, wrapping_mul, wrapping_plane_prod);

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
        let axis_len = input.shape(axis);
        let mut acc = input[input_base_offset];

        for reduce_index in 1..axis_len {
            let input_offset = input_base_offset + reduce_index * axis_stride;
            acc = nan_propagating_max::<F>(acc, input[input_offset]);
        }

        output[output_offset] = acc;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn reduce_max_int<I: Int>(
    input: &Tensor<I>,
    output: &mut Tensor<I>,
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
        let axis_len = input.shape(axis);
        let mut acc = input[input_base_offset];

        for reduce_index in 1..axis_len {
            let input_offset = input_base_offset + reduce_index * axis_stride;
            let value = input[input_offset];
            acc = if acc >= value { acc } else { value };
        }

        output[output_offset] = acc;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn reduce_max_float_plane<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
    axis: usize,
    output_len: usize,
) {
    let output_index = CUBE_POS_X as usize;
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
        let axis_len = input.shape(axis);
        let plane_width = PLANE_DIM as usize;
        let mut reduce_index = UNIT_POS as usize;
        let mut acc = input[input_base_offset + reduce_index * axis_stride];

        reduce_index += plane_width;
        while reduce_index < axis_len {
            let input_offset = input_base_offset + reduce_index * axis_stride;
            acc = nan_propagating_max::<F>(acc, input[input_offset]);
            reduce_index += plane_width;
        }

        let contains_nan = plane_contains_nan::<F>(acc);
        let propagated_nan = plane_propagate_nan::<F>(acc);
        let reduced = plane_max(acc);
        if UNIT_POS == 0 {
            output[output_offset] = if contains_nan {
                propagated_nan
            } else {
                reduced
            };
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn reduce_max_int_plane<I: Int>(
    input: &Tensor<I>,
    output: &mut Tensor<I>,
    axis: usize,
    output_len: usize,
) {
    let output_index = CUBE_POS_X as usize;
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
        let axis_len = input.shape(axis);
        let plane_width = PLANE_DIM as usize;
        let mut reduce_index = UNIT_POS as usize;
        let mut acc = input[input_base_offset + reduce_index * axis_stride];

        reduce_index += plane_width;
        while reduce_index < axis_len {
            let input_offset = input_base_offset + reduce_index * axis_stride;
            let value = input[input_offset];
            acc = if acc >= value { acc } else { value };
            reduce_index += plane_width;
        }

        let reduced = plane_max(acc);
        if UNIT_POS == 0 {
            output[output_offset] = reduced;
        }
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
        let axis_len = input.shape(axis);
        let mut acc = input[input_base_offset];

        for reduce_index in 1..axis_len {
            let input_offset = input_base_offset + reduce_index * axis_stride;
            acc = nan_propagating_min::<F>(acc, input[input_offset]);
        }

        output[output_offset] = acc;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn reduce_min_int<I: Int>(
    input: &Tensor<I>,
    output: &mut Tensor<I>,
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
        let axis_len = input.shape(axis);
        let mut acc = input[input_base_offset];

        for reduce_index in 1..axis_len {
            let input_offset = input_base_offset + reduce_index * axis_stride;
            let value = input[input_offset];
            acc = if acc <= value { acc } else { value };
        }

        output[output_offset] = acc;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn reduce_min_float_plane<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
    axis: usize,
    output_len: usize,
) {
    let output_index = CUBE_POS_X as usize;
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
        let axis_len = input.shape(axis);
        let plane_width = PLANE_DIM as usize;
        let mut reduce_index = UNIT_POS as usize;
        let mut acc = input[input_base_offset + reduce_index * axis_stride];

        reduce_index += plane_width;
        while reduce_index < axis_len {
            let input_offset = input_base_offset + reduce_index * axis_stride;
            acc = nan_propagating_min::<F>(acc, input[input_offset]);
            reduce_index += plane_width;
        }

        let contains_nan = plane_contains_nan::<F>(acc);
        let propagated_nan = plane_propagate_nan::<F>(acc);
        let reduced = plane_min(acc);
        if UNIT_POS == 0 {
            output[output_offset] = if contains_nan {
                propagated_nan
            } else {
                reduced
            };
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn reduce_min_int_plane<I: Int>(
    input: &Tensor<I>,
    output: &mut Tensor<I>,
    axis: usize,
    output_len: usize,
) {
    let output_index = CUBE_POS_X as usize;
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
        let axis_len = input.shape(axis);
        let plane_width = PLANE_DIM as usize;
        let mut reduce_index = UNIT_POS as usize;
        let mut acc = input[input_base_offset + reduce_index * axis_stride];

        reduce_index += plane_width;
        while reduce_index < axis_len {
            let input_offset = input_base_offset + reduce_index * axis_stride;
            let value = input[input_offset];
            acc = if acc <= value { acc } else { value };
            reduce_index += plane_width;
        }

        let reduced = plane_min(acc);
        if UNIT_POS == 0 {
            output[output_offset] = reduced;
        }
    }
}
