use cubecl::prelude::*;
use num_complex::{Complex32, Complex64};

use crate::helpers::{axis_in_sequence, flat_to_tensor_index, multi_to_tensor_index, zero_value};

#[cube(launch_unchecked)]
pub fn fill_zero_kernel<E: CubePrimitive>(out: &mut Array<E>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = zero_value::<E>();
    }
}

#[cube(launch_unchecked)]
pub fn transpose_kernel<E: CubePrimitive>(
    out: &mut Tensor<E>,
    input: &Tensor<E>,
    #[comptime] perm: Sequence<usize>,
) {
    if ABSOLUTE_POS < out.len() {
        let rank = perm.len();
        let out_idx = flat_to_tensor_index(ABSOLUTE_POS, out, rank);
        let mut input_idx = Array::<usize>::new(rank);
        #[unroll]
        for axis in 0..rank {
            let src_axis = comptime! { *perm.index(axis) };
            input_idx[src_axis] = out_idx[axis];
        }
        out[ABSOLUTE_POS] = input[multi_to_tensor_index(&input_idx, input, rank)];
    }
}

#[cube(launch_unchecked)]
pub fn broadcast_in_dim_kernel<E: CubePrimitive>(
    out: &mut Tensor<E>,
    input: &Tensor<E>,
    #[comptime] dims: Sequence<usize>,
    #[comptime] output_rank: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let rank = dims.len();
        let out_idx = flat_to_tensor_index(ABSOLUTE_POS, out, output_rank);
        let mut input_idx = Array::<usize>::new(rank);
        #[unroll]
        for src_axis in 0..rank {
            let dst_axis = comptime! { *dims.index(src_axis) };
            let src_dim = input.shape(src_axis);
            input_idx[src_axis] = out_idx[dst_axis];
            if src_dim == 1 {
                input_idx[src_axis] = 0;
            }
        }
        out[ABSOLUTE_POS] = input[multi_to_tensor_index(&input_idx, input, rank)];
    }
}

#[cube(launch_unchecked)]
pub fn convert_float_to_float<Out: Float, In: Float>(out: &mut Array<Out>, input: &Array<In>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = Out::cast_from(input[ABSOLUTE_POS]);
    }
}

#[cube(launch_unchecked)]
pub fn convert_c32_to_f32(out: &mut Array<f32>, input: &Array<Complex32>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = input[ABSOLUTE_POS].real_val();
    }
}

#[cube(launch_unchecked)]
pub fn convert_c32_to_f64(out: &mut Array<f64>, input: &Array<Complex32>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = f64::cast_from(input[ABSOLUTE_POS].real_val());
    }
}

#[cube(launch_unchecked)]
pub fn convert_c64_to_f32(out: &mut Array<f32>, input: &Array<Complex64>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = f32::cast_from(input[ABSOLUTE_POS].real_val());
    }
}

#[cube(launch_unchecked)]
pub fn convert_c64_to_f64(out: &mut Array<f64>, input: &Array<Complex64>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = input[ABSOLUTE_POS].real_val();
    }
}

/// Float-to-complex conversion kernels.
///
/// These write interleaved (re, im) pairs to the output buffer viewed as
/// raw floats. Output array has 2x the length of input (re, 0, re, 0, ...).
#[cube(launch_unchecked)]
pub fn convert_f32_to_c32_raw(out: &mut Array<f32>, input: &Array<f32>) {
    if ABSOLUTE_POS < input.len() {
        let re = input[ABSOLUTE_POS];
        out[ABSOLUTE_POS * 2] = re;
        out[ABSOLUTE_POS * 2 + 1] = 0.0f32;
    }
}

#[cube(launch_unchecked)]
pub fn convert_f32_to_c64_raw(out: &mut Array<f64>, input: &Array<f32>) {
    if ABSOLUTE_POS < input.len() {
        let re = f64::cast_from(input[ABSOLUTE_POS]);
        out[ABSOLUTE_POS * 2] = re;
        out[ABSOLUTE_POS * 2 + 1] = 0.0f64;
    }
}

#[cube(launch_unchecked)]
pub fn convert_f64_to_c32_raw(out: &mut Array<f32>, input: &Array<f64>) {
    if ABSOLUTE_POS < input.len() {
        let re = f32::cast_from(input[ABSOLUTE_POS]);
        out[ABSOLUTE_POS * 2] = re;
        out[ABSOLUTE_POS * 2 + 1] = 0.0f32;
    }
}

#[cube(launch_unchecked)]
pub fn convert_f64_to_c64_raw(out: &mut Array<f64>, input: &Array<f64>) {
    if ABSOLUTE_POS < input.len() {
        let re = input[ABSOLUTE_POS];
        out[ABSOLUTE_POS * 2] = re;
        out[ABSOLUTE_POS * 2 + 1] = 0.0f64;
    }
}

#[cube(launch_unchecked)]
pub fn convert_complex_to_complex<Out: Complex, In: Complex>(
    out: &mut Array<Out>,
    input: &Array<In>,
) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = Out::cast_from(input[ABSOLUTE_POS]);
    }
}

#[cube(launch_unchecked)]
pub fn reverse_kernel<E: CubePrimitive>(
    out: &mut Tensor<E>,
    input: &Tensor<E>,
    #[comptime] axes: Sequence<usize>,
    #[comptime] rank: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let out_idx = flat_to_tensor_index(ABSOLUTE_POS, out, rank);
        let mut input_idx = Array::<usize>::new(rank);
        #[unroll]
        for axis in 0..rank {
            let dim = out.shape(axis);
            input_idx[axis] = if axis_in_sequence(axes.clone(), axis) {
                dim - 1 - out_idx[axis]
            } else {
                out_idx[axis]
            };
        }
        out[ABSOLUTE_POS] = input[multi_to_tensor_index(&input_idx, input, rank)];
    }
}

#[cube(launch_unchecked)]
pub fn concatenate_copy_kernel<E: CubePrimitive>(
    out: &mut Tensor<E>,
    input: &Tensor<E>,
    #[comptime] axis: usize,
    #[comptime] axis_offset: usize,
    #[comptime] rank: usize,
) {
    if ABSOLUTE_POS < input.len() {
        let input_idx = flat_to_tensor_index(ABSOLUTE_POS, input, rank);
        let mut output_idx = Array::<usize>::new(rank);
        #[unroll]
        for dim in 0..rank {
            output_idx[dim] = input_idx[dim];
        }
        output_idx[axis] += axis_offset;
        let dst = multi_to_tensor_index(&output_idx, out, rank);
        out[dst] = input[ABSOLUTE_POS];
    }
}
