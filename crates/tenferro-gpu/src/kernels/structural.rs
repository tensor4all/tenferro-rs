use cubecl::prelude::*;
use num_complex::{Complex32, Complex64};

use crate::kernels::helpers::{
    axis_in_sequence, flat_to_tensor_index, multi_to_tensor_index, zero_value,
};

#[cube]
fn strided_view_offset_from_tensor<E: CubePrimitive>(
    mut flat: usize,
    logical: &Tensor<E>,
    #[comptime] strides: Sequence<i64>,
    base_offset: i64,
    #[comptime] rank: usize,
) -> usize {
    let mut offset = base_offset;
    #[unroll]
    for axis in 0..rank {
        let dim = logical.shape(axis);
        let coordinate = flat % dim;
        flat /= dim;
        let stride = comptime! { *strides.index(axis) };
        offset += (coordinate as i64) * stride;
    }
    usize::cast_from(offset)
}

#[cube(launch_unchecked)]
pub fn fill_zero_kernel<E: CubePrimitive>(out: &mut Array<E>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = zero_value::<E>();
    }
}

/// In-place scale by a device-resident single-element factor:
/// `out[i] *= factor[0]`. Used by the dot-general accumulation path for the
/// degenerate `out = beta * out` case (zero-sized contraction).
#[cube(launch_unchecked)]
pub fn scale_in_place_float_kernel<F: Float>(out: &mut Array<F>, factor: &Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = out[ABSOLUTE_POS] * factor[0];
    }
}

/// Complex twin of [`scale_in_place_float_kernel`].
#[cube(launch_unchecked)]
pub fn scale_in_place_complex_kernel<C: ComplexCore>(out: &mut Array<C>, factor: &Array<C>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = out[ABSOLUTE_POS] * factor[0];
    }
}

#[cube(launch_unchecked)]
pub fn view_to_contiguous_kernel<E: CubePrimitive>(
    out: &mut Tensor<E>,
    input: &Array<E>,
    #[comptime] strides: Sequence<i64>,
    base_offset: i64,
    #[comptime] rank: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let src = strided_view_offset_from_tensor(ABSOLUTE_POS, out, strides, base_offset, rank);
        out[ABSOLUTE_POS] = input[src];
    }
}

#[cube(launch_unchecked)]
pub fn contiguous_to_view_kernel<E: CubePrimitive>(
    dst: &mut Array<E>,
    src: &Tensor<E>,
    #[comptime] strides: Sequence<i64>,
    base_offset: i64,
    #[comptime] rank: usize,
) {
    if ABSOLUTE_POS < src.len() {
        let dst_offset =
            strided_view_offset_from_tensor(ABSOLUTE_POS, src, strides, base_offset, rank);
        dst[dst_offset] = src[ABSOLUTE_POS];
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
pub fn convert_complex_to_complex<Out: ComplexCore, In: ComplexCore>(
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
                dim.saturating_sub(1).saturating_sub(out_idx[axis])
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
