use cubecl::prelude::*;
use num_complex::{Complex32, Complex64};

use super::{axis_in_sequence, flat_to_multi_index, multi_to_flat_index, zero_value};

#[cube(launch_unchecked)]
pub(crate) fn fill_zero_kernel<E: CubePrimitive>(out: &mut Array<E>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = zero_value::<E>();
    }
}

#[cube(launch_unchecked)]
pub(crate) fn transpose_kernel<E: CubePrimitive>(
    out: &mut Array<E>,
    input: &Array<E>,
    #[comptime] input_shape: Sequence<usize>,
    #[comptime] output_shape: Sequence<usize>,
    #[comptime] perm: Sequence<usize>,
) {
    if ABSOLUTE_POS < out.len() {
        let out_idx = flat_to_multi_index(ABSOLUTE_POS, output_shape.clone());
        let rank = output_shape.len();
        let mut input_idx = Array::<usize>::new(rank);
        #[unroll]
        for axis in 0..rank {
            let src_axis = comptime! { *perm.index(axis) };
            input_idx[src_axis] = out_idx[axis];
        }
        out[ABSOLUTE_POS] = input[multi_to_flat_index(&input_idx, input_shape)];
    }
}

#[cube(launch_unchecked)]
pub(crate) fn broadcast_in_dim_kernel<E: CubePrimitive>(
    out: &mut Array<E>,
    input: &Array<E>,
    #[comptime] input_shape: Sequence<usize>,
    #[comptime] output_shape: Sequence<usize>,
    #[comptime] dims: Sequence<usize>,
) {
    if ABSOLUTE_POS < out.len() {
        let out_idx = flat_to_multi_index(ABSOLUTE_POS, output_shape.clone());
        let rank = input_shape.len();
        let mut input_idx = Array::<usize>::new(rank);
        #[unroll]
        for src_axis in 0..rank {
            let dst_axis = comptime! { *dims.index(src_axis) };
            let src_dim = comptime! { *input_shape.index(src_axis) };
            input_idx[src_axis] = out_idx[dst_axis];
            if src_dim == 1 {
                input_idx[src_axis] = 0;
            }
        }
        out[ABSOLUTE_POS] = input[multi_to_flat_index(&input_idx, input_shape)];
    }
}

#[cube(launch_unchecked)]
pub(crate) fn convert_float_to_float<Out: Float, In: Float>(
    out: &mut Array<Out>,
    input: &Array<In>,
) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = Out::cast_from(input[ABSOLUTE_POS]);
    }
}

#[cube(launch_unchecked)]
pub(crate) fn convert_c32_to_f32(out: &mut Array<f32>, input: &Array<Complex32>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = input[ABSOLUTE_POS].real_val();
    }
}

#[cube(launch_unchecked)]
pub(crate) fn convert_c32_to_f64(out: &mut Array<f64>, input: &Array<Complex32>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = f64::cast_from(input[ABSOLUTE_POS].real_val());
    }
}

#[cube(launch_unchecked)]
pub(crate) fn convert_c64_to_f32(out: &mut Array<f32>, input: &Array<Complex64>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = f32::cast_from(input[ABSOLUTE_POS].real_val());
    }
}

#[cube(launch_unchecked)]
pub(crate) fn convert_c64_to_f64(out: &mut Array<f64>, input: &Array<Complex64>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = input[ABSOLUTE_POS].real_val();
    }
}

#[cube(launch_unchecked)]
pub(crate) fn convert_complex_to_complex<Out: Complex, In: Complex>(
    out: &mut Array<Out>,
    input: &Array<In>,
) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = Out::cast_from(input[ABSOLUTE_POS]);
    }
}

#[cube(launch_unchecked)]
pub(crate) fn reverse_kernel<E: CubePrimitive>(
    out: &mut Array<E>,
    input: &Array<E>,
    #[comptime] shape: Sequence<usize>,
    #[comptime] axes: Sequence<usize>,
) {
    if ABSOLUTE_POS < out.len() {
        let out_idx = flat_to_multi_index(ABSOLUTE_POS, shape.clone());
        let rank = shape.len();
        let mut input_idx = Array::<usize>::new(rank);
        #[unroll]
        for axis in 0..rank {
            let dim = comptime! { *shape.index(axis) };
            input_idx[axis] = if axis_in_sequence(axes.clone(), axis) {
                dim - 1 - out_idx[axis]
            } else {
                out_idx[axis]
            };
        }
        out[ABSOLUTE_POS] = input[multi_to_flat_index(&input_idx, shape)];
    }
}

#[cube(launch_unchecked)]
pub(crate) fn concatenate_copy_kernel<E: CubePrimitive>(
    out: &mut Array<E>,
    input: &Array<E>,
    #[comptime] input_shape: Sequence<usize>,
    #[comptime] output_shape: Sequence<usize>,
    #[comptime] axis: usize,
    #[comptime] axis_offset: usize,
) {
    if ABSOLUTE_POS < input.len() {
        let input_idx = flat_to_multi_index(ABSOLUTE_POS, input_shape.clone());
        let rank = input_shape.len();
        let mut output_idx = Array::<usize>::new(rank);
        #[unroll]
        for dim in 0..rank {
            output_idx[dim] = input_idx[dim];
        }
        output_idx[axis] += axis_offset;
        let dst = multi_to_flat_index(&output_idx, output_shape);
        out[dst] = input[ABSOLUTE_POS];
    }
}
