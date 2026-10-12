use cubecl::prelude::*;
use num_complex::{Complex32, Complex64};

use crate::kernels::helpers::{
    axis_in_sequence, flat_to_tensor_index, multi_to_tensor_index, zero_value,
};

#[cube]
fn strided_view_offset_from_tensor<E: CubePrimitive + Scalar>(
    mut flat: usize,
    logical: &Tensor<E>,
    strides: &Sequence<i64>,
    base_offset: i64,
    #[comptime] rank: usize,
) -> usize {
    let mut offset = base_offset;
    #[unroll]
    for axis in 0..rank {
        let dim = logical.shape(axis);
        let coordinate = flat % dim;
        flat /= dim;
        offset += (coordinate as i64) * strides[axis];
    }
    usize::cast_from(offset)
}

#[cube(launch_unchecked)]
pub fn fill_zero_kernel<E: CubePrimitive + Scalar>(out: &mut [E]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = zero_value::<E>();
    }
}

/// Write an exact zero to every logical coordinate of a strided destination
/// region, leaving the elements the region does not address untouched.
///
/// The region's extents, strides, offset and length are runtime scalar
/// arguments: only the rank is compile-time, so one module serves every layout
/// of a given rank.
#[cube(launch_unchecked)]
pub fn fill_zero_view_kernel<E: CubePrimitive + Scalar>(
    out: &mut [E],
    dims: Sequence<usize>,
    strides: Sequence<i64>,
    base_offset: i64,
    len: usize,
    #[comptime] rank: usize,
) {
    if ABSOLUTE_POS < len {
        let mut flat = ABSOLUTE_POS;
        let mut index = base_offset;
        #[unroll]
        for axis in 0..rank {
            let dim = dims[axis];
            let coordinate = flat % dim;
            flat /= dim;
            index += (coordinate as i64) * strides[axis];
        }
        out[usize::cast_from(index)] = zero_value::<E>();
    }
}

#[cube(launch_unchecked)]
pub fn copy_bool_kernel(out: &mut [u8], input: &[u8]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = input[ABSOLUTE_POS];
    }
}

/// In-place scale by a device-resident single-element factor:
/// `out[i] *= factor[0]`. Used by the dot-general accumulation path for the
/// degenerate `out = beta * out` case (zero-sized contraction).
#[cube(launch_unchecked)]
pub fn scale_in_place_float_kernel<F: Float>(out: &mut [F], factor: &[F]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = out[ABSOLUTE_POS] * factor[0];
    }
}

/// Complex twin of [`scale_in_place_float_kernel`].
#[cube(launch_unchecked)]
pub fn scale_in_place_complex_kernel<C: ComplexCore>(out: &mut [C], factor: &[C]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = out[ABSOLUTE_POS] * factor[0];
    }
}

/// Gather a strided source view into a compact destination (CUDA).
///
/// The view's extents, signed strides, offset and length are runtime scalar
/// arguments, so one module serves every layout of a given rank.
#[cube(launch_unchecked)]
pub fn materialize_strided_kernel<E: CubePrimitive + Scalar>(
    dst: &mut [E],
    src: &[E],
    dims: Sequence<usize>,
    src_strides: Sequence<i64>,
    src_offset: i64,
    len: usize,
    #[comptime] rank: usize,
) {
    if ABSOLUTE_POS < len {
        let mut flat = ABSOLUTE_POS;
        let mut src_index = src_offset;
        #[unroll]
        for axis in 0..rank {
            let dim = dims[axis];
            let coordinate = flat % dim;
            flat /= dim;
            src_index += (coordinate as i64) * src_strides[axis];
        }
        dst[ABSOLUTE_POS] = src[usize::cast_from(src_index)];
    }
}

/// WebGPU/Metal twin of [`materialize_strided_kernel`] that bakes the fused
/// native-permutation layout into the module.
///
/// INVARIANT: this compile-time layout is the documented WebGPU/Metal exception
/// in `docs/design/gpu-backend-design.md` (raw source/destination arrays carry no
/// logical layout, and raw-array runtime metadata packing is ambiguous on Metal).
/// CUDA launches [`materialize_strided_kernel`] instead.
#[cube(launch_unchecked)]
pub fn materialize_strided_comptime_layout_kernel<E: CubePrimitive + Scalar>(
    dst: &mut [E],
    src: &[E],
    #[comptime] dims: Sequence<usize>,
    #[comptime] src_strides: Sequence<i64>,
    src_offset: i64,
    #[comptime] len: usize,
    #[comptime] rank: usize,
) {
    if ABSOLUTE_POS < len {
        let mut flat = ABSOLUTE_POS;
        let mut src_index = src_offset;
        #[unroll]
        for axis in 0..rank {
            let dim = comptime! { *dims.index(axis) };
            let coordinate = flat % dim;
            flat /= dim;
            let src_stride = comptime! { *src_strides.index(axis) };
            src_index += (coordinate as i64) * src_stride;
        }
        dst[ABSOLUTE_POS] = src[usize::cast_from(src_index)];
    }
}

/// Shared-memory tiled 2D (optionally batched) transpose of a compact source
/// into a compact destination (CUDA).
///
/// The batch stride and both fast extents are runtime scalar arguments; only
/// the tile shape, block rows, shared-memory padding and vector width are
/// compile-time algorithm configuration.
#[cube(launch_unchecked)]
pub fn tiled_transpose_kernel<E: CubePrimitive + Scalar>(
    dst: &mut [E],
    src: &[E],
    src_offset: usize,
    batch_stride: usize,
    dst_fast_extent: usize,
    src_fast_extent: usize,
    #[comptime] tile: usize,
    #[comptime] block_rows: usize,
    #[comptime] padding: usize,
    #[comptime] vector_width: usize,
) {
    let pitch = tile + padding;
    let mut shared = Shared::new_slice(tile * pitch);
    let unit_x = UNIT_POS_X as usize;
    let unit_y = UNIT_POS_Y as usize;
    // The destination tiles advance along `x`, so a block's neighbours along the
    // fastest-varying grid axis stream the destination instead of striding
    // through it one tile at a time.
    let tile_dst_fast = CUBE_POS_X as usize * tile;
    let tile_src_fast = CUBE_POS_Y as usize * tile;
    let batch_base = CUBE_POS_Z as usize * batch_stride;

    let mut row = unit_y;
    while row < tile {
        let dst_fast = tile_dst_fast + row;
        #[unroll]
        for lane in 0..vector_width {
            let local_src_fast = unit_x * vector_width + lane;
            let src_fast = tile_src_fast + local_src_fast;
            if dst_fast < dst_fast_extent && src_fast < src_fast_extent {
                let src_index = src_offset + batch_base + dst_fast * src_fast_extent + src_fast;
                shared[row * pitch + local_src_fast] = src[src_index];
            }
        }
        row += block_rows;
    }

    sync_cube();

    row = unit_y;
    while row < tile {
        let src_fast = tile_src_fast + row;
        #[unroll]
        for lane in 0..vector_width {
            let local_dst_fast = unit_x * vector_width + lane;
            let dst_fast = tile_dst_fast + local_dst_fast;
            if dst_fast < dst_fast_extent && src_fast < src_fast_extent {
                let dst_index = batch_base + dst_fast + src_fast * dst_fast_extent;
                dst[dst_index] = shared[local_dst_fast * pitch + row];
            }
        }
        row += block_rows;
    }
}

/// WebGPU/Metal twin of [`tiled_transpose_kernel`] with the batch stride and
/// fast extents baked into the module.
///
/// INVARIANT: the documented WebGPU/Metal native-permutation exception (see
/// [`materialize_strided_comptime_layout_kernel`]); CUDA launches
/// [`tiled_transpose_kernel`] instead.
#[cube(launch_unchecked)]
pub fn tiled_transpose_comptime_layout_kernel<E: CubePrimitive + Scalar>(
    dst: &mut [E],
    src: &[E],
    src_offset: usize,
    #[comptime] batch_stride: usize,
    #[comptime] dst_fast_extent: usize,
    #[comptime] src_fast_extent: usize,
    #[comptime] tile: usize,
    #[comptime] block_rows: usize,
    #[comptime] padding: usize,
    #[comptime] vector_width: usize,
) {
    let pitch = tile + padding;
    let mut shared = Shared::new_slice(tile * pitch);
    let unit_x = UNIT_POS_X as usize;
    let unit_y = UNIT_POS_Y as usize;
    // The destination tiles advance along `x`, so a block's neighbours along the
    // fastest-varying grid axis stream the destination instead of striding
    // through it one tile at a time.
    let tile_dst_fast = CUBE_POS_X as usize * tile;
    let tile_src_fast = CUBE_POS_Y as usize * tile;
    let batch_base = CUBE_POS_Z as usize * batch_stride;

    let mut row = unit_y;
    while row < tile {
        let dst_fast = tile_dst_fast + row;
        #[unroll]
        for lane in 0..vector_width {
            let local_src_fast = unit_x * vector_width + lane;
            let src_fast = tile_src_fast + local_src_fast;
            if dst_fast < dst_fast_extent && src_fast < src_fast_extent {
                let src_index = src_offset + batch_base + dst_fast * src_fast_extent + src_fast;
                shared[row * pitch + local_src_fast] = src[src_index];
            }
        }
        row += block_rows;
    }

    sync_cube();

    row = unit_y;
    while row < tile {
        let src_fast = tile_src_fast + row;
        #[unroll]
        for lane in 0..vector_width {
            let local_dst_fast = unit_x * vector_width + lane;
            let dst_fast = tile_dst_fast + local_dst_fast;
            if dst_fast < dst_fast_extent && src_fast < src_fast_extent {
                let dst_index = batch_base + dst_fast + src_fast * dst_fast_extent;
                dst[dst_index] = shared[local_dst_fast * pitch + row];
            }
        }
        row += block_rows;
    }
}

#[cube(launch_unchecked)]
pub fn contiguous_to_view_kernel<E: CubePrimitive + Scalar>(
    dst: &mut [E],
    src: &Tensor<E>,
    strides: Sequence<i64>,
    base_offset: i64,
    #[comptime] rank: usize,
) {
    if ABSOLUTE_POS < src.len() {
        let dst_offset =
            strided_view_offset_from_tensor(ABSOLUTE_POS, src, &strides, base_offset, rank);
        dst[dst_offset] = src[ABSOLUTE_POS];
    }
}

#[cube(launch_unchecked)]
pub fn strided_to_strided_kernel<E: CubePrimitive + Scalar>(
    dst: &mut [E],
    src: &[E],
    dims: Sequence<usize>,
    src_strides: Sequence<i64>,
    dst_strides: Sequence<i64>,
    src_offset: i64,
    dst_offset: i64,
    len: usize,
    #[comptime] rank: usize,
) {
    if ABSOLUTE_POS < len {
        let mut flat = ABSOLUTE_POS;
        let mut src_index = src_offset;
        let mut dst_index = dst_offset;
        #[unroll]
        for axis in 0..rank {
            let dim = dims[axis];
            let coordinate = flat % dim;
            flat /= dim;
            src_index += (coordinate as i64) * src_strides[axis];
            dst_index += (coordinate as i64) * dst_strides[axis];
        }
        dst[usize::cast_from(dst_index)] = src[usize::cast_from(src_index)];
    }
}

#[cube(launch_unchecked)]
pub fn broadcast_in_dim_kernel<E: CubePrimitive + Scalar>(
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
pub fn convert_float_to_float<Out: Float, In: Float>(out: &mut [Out], input: &[In]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = Out::cast_from(input[ABSOLUTE_POS]);
    }
}

#[cube(launch_unchecked)]
pub fn convert_numeric<Out: Numeric, In: Numeric>(out: &mut [Out], input: &[In]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = Out::cast_from(input[ABSOLUTE_POS]);
    }
}

#[cube(launch_unchecked)]
pub fn convert_numeric_to_bool<In: Numeric>(out: &mut [u8], input: &[In]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = if input[ABSOLUTE_POS] != In::from_int(0) {
            1u8
        } else {
            0u8
        };
    }
}

#[cube(launch_unchecked)]
pub fn convert_bool_to_numeric<Out: Numeric>(out: &mut [Out], input: &[u8]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = Out::cast_from(input[ABSOLUTE_POS]);
    }
}

#[cube(launch_unchecked)]
pub fn convert_numeric_to_complex_raw<Out: Float, In: Numeric>(
    out: &mut [Out],
    input: &[In],
) {
    if ABSOLUTE_POS < input.len() {
        out[ABSOLUTE_POS * 2] = Out::cast_from(input[ABSOLUTE_POS]);
        out[ABSOLUTE_POS * 2 + 1] = Out::new(0.0f32);
    }
}

#[cube(launch_unchecked)]
pub fn convert_bool_to_complex_raw<Out: Float>(out: &mut [Out], input: &[u8]) {
    if ABSOLUTE_POS < input.len() {
        out[ABSOLUTE_POS * 2] = Out::cast_from(input[ABSOLUTE_POS]);
        out[ABSOLUTE_POS * 2 + 1] = Out::new(0.0f32);
    }
}

#[cube(launch_unchecked)]
pub fn convert_complex_to_numeric<Out: Numeric, In: ComplexCore>(
    out: &mut [Out],
    input: &[In],
) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = Out::cast_from(input[ABSOLUTE_POS].real_val());
    }
}

#[cube(launch_unchecked)]
pub fn convert_complex_raw_to_bool<F: Float>(out: &mut [u8], input: &[F]) {
    if ABSOLUTE_POS < out.len() {
        let real = input[ABSOLUTE_POS * 2];
        let imag = input[ABSOLUTE_POS * 2 + 1];
        out[ABSOLUTE_POS] = if real != F::new(0.0f32) || imag != F::new(0.0f32) {
            1u8
        } else {
            0u8
        };
    }
}

#[cube(launch_unchecked)]
pub fn validate_real_cast<
    F: Float + CubeElement + CubePrimitive<WithScalar<bool> = bool, WithScalar<F> = F>
        + LaunchArg
        + cubecl::ScalarArgType,
>(
    input: &[F],
    flag: &mut [Atomic<u32>],
    min: F,
    max: F,
    #[comptime] stride: usize,
    #[comptime] max_inclusive: bool,
) {
    if ABSOLUTE_POS * stride < input.len() {
        let value = input[ABSOLUTE_POS * stride];
        let invalid_max = if max_inclusive {
            value > max
        } else {
            value >= max
        };
        if value.is_nan() || value.is_inf() || value < min || invalid_max {
            flag[0].fetch_min(ABSOLUTE_POS as u32);
        }
    }
}

#[cube(launch_unchecked)]
pub fn extract_invalid_real_cast<F: Float>(
    input: &[F],
    flag: &[Atomic<u32>],
    values: &mut [F],
    #[comptime] stride: usize,
) {
    if ABSOLUTE_POS == 0 {
        let index = flag[0].load();
        if index != u32::MAX {
            values[1] = input[index as usize * stride];
        }
    }
}

#[cube(launch_unchecked)]
pub fn convert_c32_to_f32(out: &mut [f32], input: &[Complex32]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = input[ABSOLUTE_POS].real_val();
    }
}

#[cube(launch_unchecked)]
pub fn convert_c32_to_f64(out: &mut [f64], input: &[Complex32]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = f64::cast_from(input[ABSOLUTE_POS].real_val());
    }
}

#[cube(launch_unchecked)]
pub fn convert_c64_to_f32(out: &mut [f32], input: &[Complex64]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = f32::cast_from(input[ABSOLUTE_POS].real_val());
    }
}

#[cube(launch_unchecked)]
pub fn convert_c64_to_f64(out: &mut [f64], input: &[Complex64]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = input[ABSOLUTE_POS].real_val();
    }
}

/// Float-to-complex conversion kernels.
///
/// These write interleaved (re, im) pairs to the output buffer viewed as
/// raw floats. Output array has 2x the length of input (re, 0, re, 0, ...).
#[cube(launch_unchecked)]
pub fn convert_f32_to_c32_raw(out: &mut [f32], input: &[f32]) {
    if ABSOLUTE_POS < input.len() {
        let re = input[ABSOLUTE_POS];
        out[ABSOLUTE_POS * 2] = re;
        out[ABSOLUTE_POS * 2 + 1] = 0.0f32;
    }
}

#[cube(launch_unchecked)]
pub fn convert_f32_to_c64_raw(out: &mut [f64], input: &[f32]) {
    if ABSOLUTE_POS < input.len() {
        let re = f64::cast_from(input[ABSOLUTE_POS]);
        out[ABSOLUTE_POS * 2] = re;
        out[ABSOLUTE_POS * 2 + 1] = 0.0f64;
    }
}

#[cube(launch_unchecked)]
pub fn convert_f64_to_c32_raw(out: &mut [f32], input: &[f64]) {
    if ABSOLUTE_POS < input.len() {
        let re = f32::cast_from(input[ABSOLUTE_POS]);
        out[ABSOLUTE_POS * 2] = re;
        out[ABSOLUTE_POS * 2 + 1] = 0.0f32;
    }
}

#[cube(launch_unchecked)]
pub fn convert_f64_to_c64_raw(out: &mut [f64], input: &[f64]) {
    if ABSOLUTE_POS < input.len() {
        let re = input[ABSOLUTE_POS];
        out[ABSOLUTE_POS * 2] = re;
        out[ABSOLUTE_POS * 2 + 1] = 0.0f64;
    }
}

#[cube(launch_unchecked)]
pub fn convert_complex_to_complex<Out: ComplexCore, In: ComplexCore>(
    out: &mut [Out],
    input: &[In],
) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = Out::cast_from(input[ABSOLUTE_POS]);
    }
}

#[cube(launch_unchecked)]
pub fn convert_complex_raw<Out: Float, In: Float>(out: &mut [Out], input: &[In]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = Out::cast_from(input[ABSOLUTE_POS]);
    }
}

#[cube(launch_unchecked)]
pub fn reverse_kernel<E: CubePrimitive + Scalar>(
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
pub fn concatenate_copy_kernel<E: CubePrimitive + Scalar>(
    out: &mut Tensor<E>,
    input: &Tensor<E>,
    #[comptime] axis: usize,
    axis_offset: usize,
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
