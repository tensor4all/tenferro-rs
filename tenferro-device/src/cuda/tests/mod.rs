use crate as tenferro_device;
use num_complex::Complex64;

fn linear_offset(linear_idx: usize, dims: &[usize], strides: &[isize], offset: isize) -> isize {
    let mut remainder = linear_idx;
    let mut out = offset;
    for axis in 0..dims.len() {
        let coord = remainder % dims[axis];
        remainder /= dims[axis];
        out += (coord as isize) * strides[axis];
    }
    out
}

fn host_strided_copy_reference<T: Copy + Default>(
    src: &[T],
    dims: &[usize],
    src_strides: &[isize],
    src_offset: isize,
    dst_strides: &[isize],
) -> Vec<T> {
    let numel = dims.iter().product::<usize>();
    let mut dst = vec![T::default(); numel];
    if numel == 0 {
        return dst;
    }

    for linear_idx in 0..numel {
        let mut src_index = src_offset;
        let mut dst_index = 0isize;
        let mut remainder = linear_idx;

        for axis in 0..dims.len() {
            let coord = remainder % dims[axis];
            remainder /= dims[axis];
            src_index += (coord as isize) * src_strides[axis];
            dst_index += (coord as isize) * dst_strides[axis];
        }

        dst[dst_index as usize] = src[src_index as usize];
    }

    dst
}

fn host_strided_copy_conj_complex64_reference(
    src: &[Complex64],
    dims: &[usize],
    src_strides: &[isize],
    src_offset: isize,
    dst_strides: &[isize],
) -> Vec<Complex64> {
    let numel = dims.iter().product::<usize>();
    let mut dst = vec![Complex64::default(); numel];
    if numel == 0 {
        return dst;
    }

    for linear_idx in 0..numel {
        let mut remainder = linear_idx;
        let mut src_index = src_offset;
        let mut dst_index = 0isize;

        for axis in 0..dims.len() {
            let coord = remainder % dims[axis];
            remainder /= dims[axis];
            src_index += (coord as isize) * src_strides[axis];
            dst_index += (coord as isize) * dst_strides[axis];
        }

        dst[dst_index as usize] = src[src_index as usize].conj();
    }

    dst
}

fn host_binary_add_reference(
    lhs: &[f64],
    rhs: &[f64],
    dims: &[usize],
    lhs_strides: &[isize],
    rhs_strides: &[isize],
    dst_strides: &[isize],
) -> Vec<f64> {
    let numel = dims.iter().product::<usize>();
    let mut dst = vec![0.0; numel];
    for linear_idx in 0..numel {
        let lhs_idx = linear_offset(linear_idx, dims, lhs_strides, 0) as usize;
        let rhs_idx = linear_offset(linear_idx, dims, rhs_strides, 0) as usize;
        let dst_idx = linear_offset(linear_idx, dims, dst_strides, 0) as usize;
        dst[dst_idx] = lhs[lhs_idx] + rhs[rhs_idx];
    }
    dst
}

fn host_binary_pow_reference(
    lhs: &[f64],
    rhs: &[f64],
    dims: &[usize],
    lhs_strides: &[isize],
    rhs_strides: &[isize],
    dst_strides: &[isize],
) -> Vec<f64> {
    let numel = dims.iter().product::<usize>();
    let mut dst = vec![0.0; numel];
    for linear_idx in 0..numel {
        let lhs_idx = linear_offset(linear_idx, dims, lhs_strides, 0) as usize;
        let rhs_idx = linear_offset(linear_idx, dims, rhs_strides, 0) as usize;
        let dst_idx = linear_offset(linear_idx, dims, dst_strides, 0) as usize;
        dst[dst_idx] = lhs[lhs_idx].powf(rhs[rhs_idx]);
    }
    dst
}

fn host_where_reference(
    mask: &[f64],
    on_true: &[f64],
    on_false: &[f64],
    dims: &[usize],
    mask_strides: &[isize],
    true_strides: &[isize],
    false_strides: &[isize],
    dst_strides: &[isize],
) -> Vec<f64> {
    let numel = dims.iter().product::<usize>();
    let mut dst = vec![0.0; numel];
    for linear_idx in 0..numel {
        let mask_idx = linear_offset(linear_idx, dims, mask_strides, 0) as usize;
        let true_idx = linear_offset(linear_idx, dims, true_strides, 0) as usize;
        let false_idx = linear_offset(linear_idx, dims, false_strides, 0) as usize;
        let dst_idx = linear_offset(linear_idx, dims, dst_strides, 0) as usize;
        dst[dst_idx] = if mask[mask_idx] != 0.0 {
            on_true[true_idx]
        } else {
            on_false[false_idx]
        };
    }
    dst
}

fn host_unary_abs_reference(
    src: &[f64],
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
) -> Vec<f64> {
    let numel = dims.iter().product::<usize>();
    let mut dst = vec![0.0; numel];
    for linear_idx in 0..numel {
        let src_idx = linear_offset(linear_idx, dims, src_strides, 0) as usize;
        let dst_idx = linear_offset(linear_idx, dims, dst_strides, 0) as usize;
        dst[dst_idx] = src[src_idx].abs();
    }
    dst
}

fn host_unary_abs_real_complex64_reference(
    src: &[Complex64],
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
) -> Vec<f64> {
    let numel = dims.iter().product::<usize>();
    let mut dst = vec![0.0; numel];
    for linear_idx in 0..numel {
        let src_idx = linear_offset(linear_idx, dims, src_strides, 0) as usize;
        let dst_idx = linear_offset(linear_idx, dims, dst_strides, 0) as usize;
        dst[dst_idx] = src[src_idx].norm();
    }
    dst
}

fn host_unary_real_complex64_reference(
    src: &[Complex64],
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
) -> Vec<f64> {
    let numel = dims.iter().product::<usize>();
    let mut dst = vec![0.0; numel];
    for linear_idx in 0..numel {
        let src_idx = linear_offset(linear_idx, dims, src_strides, 0) as usize;
        let dst_idx = linear_offset(linear_idx, dims, dst_strides, 0) as usize;
        dst[dst_idx] = src[src_idx].re;
    }
    dst
}

fn host_unary_imag_complex64_reference(
    src: &[Complex64],
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
) -> Vec<f64> {
    let numel = dims.iter().product::<usize>();
    let mut dst = vec![0.0; numel];
    for linear_idx in 0..numel {
        let src_idx = linear_offset(linear_idx, dims, src_strides, 0) as usize;
        let dst_idx = linear_offset(linear_idx, dims, dst_strides, 0) as usize;
        dst[dst_idx] = src[src_idx].im;
    }
    dst
}

fn host_unary_log_reference(
    src: &[f64],
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
) -> Vec<f64> {
    let numel = dims.iter().product::<usize>();
    let mut dst = vec![0.0; numel];
    for linear_idx in 0..numel {
        let src_idx = linear_offset(linear_idx, dims, src_strides, 0) as usize;
        let dst_idx = linear_offset(linear_idx, dims, dst_strides, 0) as usize;
        dst[dst_idx] = src[src_idx].ln();
    }
    dst
}

fn host_unary_sqrt_reference(
    src: &[f64],
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
) -> Vec<f64> {
    let numel = dims.iter().product::<usize>();
    let mut dst = vec![0.0; numel];
    for linear_idx in 0..numel {
        let src_idx = linear_offset(linear_idx, dims, src_strides, 0) as usize;
        let dst_idx = linear_offset(linear_idx, dims, dst_strides, 0) as usize;
        dst[dst_idx] = src[src_idx].sqrt();
    }
    dst
}

fn host_sum_reduction_reference(
    input: &[f64],
    input_dims: &[usize],
    input_strides: &[isize],
    kept_axes: &[usize],
    reduced_axes: &[usize],
) -> Vec<f64> {
    let output_dims: Vec<usize> = kept_axes.iter().map(|&axis| input_dims[axis]).collect();
    let output_numel = output_dims.iter().product::<usize>();
    let reduced_dims: Vec<usize> = reduced_axes.iter().map(|&axis| input_dims[axis]).collect();
    let reduced_numel = reduced_dims.iter().product::<usize>();
    let mut out = vec![0.0; output_numel];

    for out_linear_idx in 0..output_numel {
        let mut base = 0isize;
        let mut remainder = out_linear_idx;
        for &axis in kept_axes {
            let coord = remainder % input_dims[axis];
            remainder /= input_dims[axis];
            base += (coord as isize) * input_strides[axis];
        }

        let mut acc = 0.0;
        for red_linear_idx in 0..reduced_numel {
            let mut input_idx = base;
            let mut red_remainder = red_linear_idx;
            for &axis in reduced_axes {
                let coord = red_remainder % input_dims[axis];
                red_remainder /= input_dims[axis];
                input_idx += (coord as isize) * input_strides[axis];
            }
            acc += input[input_idx as usize];
        }
        out[out_linear_idx] = acc;
    }

    out
}

fn host_prod_reduction_reference(
    input: &[f64],
    input_dims: &[usize],
    input_strides: &[isize],
    kept_axes: &[usize],
    reduced_axes: &[usize],
) -> Vec<f64> {
    let output_dims: Vec<usize> = kept_axes.iter().map(|&axis| input_dims[axis]).collect();
    let output_numel = output_dims.iter().product::<usize>();
    let reduced_dims: Vec<usize> = reduced_axes.iter().map(|&axis| input_dims[axis]).collect();
    let reduced_numel = reduced_dims.iter().product::<usize>();
    let mut out = vec![1.0; output_numel];

    for out_linear_idx in 0..output_numel {
        let mut base = 0isize;
        let mut remainder = out_linear_idx;
        for &axis in kept_axes {
            let coord = remainder % input_dims[axis];
            remainder /= input_dims[axis];
            base += (coord as isize) * input_strides[axis];
        }

        let mut acc = 1.0;
        for red_linear_idx in 0..reduced_numel {
            let mut input_idx = base;
            let mut red_remainder = red_linear_idx;
            for &axis in reduced_axes {
                let coord = red_remainder % input_dims[axis];
                red_remainder /= input_dims[axis];
                input_idx += (coord as isize) * input_strides[axis];
            }
            acc *= input[input_idx as usize];
        }
        out[out_linear_idx] = acc;
    }

    out
}

fn host_zero_trailing_by_counts_reference<T: Copy + Default>(
    src: &[T],
    dims: &[usize],
    src_strides: &[isize],
    src_offset: isize,
    keep_counts: &[usize],
    keep_count_strides: &[isize],
    keep_count_offset: isize,
    axis: usize,
    structural_rank: usize,
) -> Vec<T> {
    let numel = dims.iter().product::<usize>();
    let mut dst = vec![T::default(); numel];
    if numel == 0 {
        return dst;
    }

    for linear_idx in 0..numel {
        let mut remainder = linear_idx;
        let mut src_index = src_offset;
        let mut batch_index = keep_count_offset;
        let mut axis_coord = 0usize;

        for dim_axis in 0..dims.len() {
            let coord = remainder % dims[dim_axis];
            remainder /= dims[dim_axis];
            src_index += (coord as isize) * src_strides[dim_axis];
            if dim_axis == axis {
                axis_coord = coord;
            }
            if dim_axis >= structural_rank {
                batch_index += (coord as isize) * keep_count_strides[dim_axis - structural_rank];
            }
        }

        let keep = keep_counts[batch_index as usize];
        if axis_coord < keep {
            dst[linear_idx] = src[src_index as usize];
        }
    }

    dst
}

fn host_triangular_part_reference<T: Copy + Default>(
    src: &[T],
    dims: &[usize],
    src_strides: &[isize],
    src_offset: isize,
    dst_strides: &[isize],
    diagonal: isize,
    half: tenferro_device::cuda::runtime::TriangularHalf,
) -> Vec<T> {
    let numel = dims.iter().product::<usize>();
    let mut dst = vec![T::default(); numel];
    if numel == 0 {
        return dst;
    }

    for linear_idx in 0..numel {
        let mut remainder = linear_idx;
        let mut src_index = src_offset;
        let mut dst_index = 0isize;
        let mut row = 0usize;
        let mut col = 0usize;

        for axis in 0..dims.len() {
            let coord = remainder % dims[axis];
            remainder /= dims[axis];
            src_index += (coord as isize) * src_strides[axis];
            dst_index += (coord as isize) * dst_strides[axis];
            if axis == 0 {
                row = coord;
            } else if axis == 1 {
                col = coord;
            }
        }

        let keep = match half {
            tenferro_device::cuda::runtime::TriangularHalf::Lower => {
                (col as isize - row as isize) <= diagonal
            }
            tenferro_device::cuda::runtime::TriangularHalf::Upper => {
                (col as isize - row as isize) >= diagonal
            }
        };
        if keep {
            dst[dst_index as usize] = src[src_index as usize];
        }
    }

    dst
}

fn host_triangular_merge_reference<T: Copy + Default>(
    lower_src: &[T],
    upper_src: &[T],
    dims: &[usize],
    lower_strides: &[isize],
    lower_offset: isize,
    upper_strides: &[isize],
    upper_offset: isize,
    dst_strides: &[isize],
) -> Vec<T> {
    let numel = dims.iter().product::<usize>();
    let mut dst = vec![T::default(); numel];
    if numel == 0 {
        return dst;
    }

    for linear_idx in 0..numel {
        let mut remainder = linear_idx;
        let mut lower_index = lower_offset;
        let mut upper_index = upper_offset;
        let mut dst_index = 0isize;
        let mut row = 0usize;
        let mut col = 0usize;

        for axis in 0..dims.len() {
            let coord = remainder % dims[axis];
            remainder /= dims[axis];
            lower_index += (coord as isize) * lower_strides[axis];
            upper_index += (coord as isize) * upper_strides[axis];
            dst_index += (coord as isize) * dst_strides[axis];
            if axis == 0 {
                row = coord;
            } else if axis == 1 {
                col = coord;
            }
        }

        dst[dst_index as usize] = if row > col {
            lower_src[lower_index as usize]
        } else {
            upper_src[upper_index as usize]
        };
    }

    dst
}

fn host_metadata_iota_reference(
    dims: &[usize],
    dst_strides: &[isize],
    dst_offset: isize,
) -> Vec<i32> {
    let numel = dims.iter().product::<usize>();
    let mut dst = vec![0i32; numel];
    for linear_idx in 0..numel {
        let dst_idx = linear_offset(linear_idx, dims, dst_strides, dst_offset) as usize;
        dst[dst_idx] = linear_idx as i32;
    }
    dst
}

fn host_metadata_iota_layout_reference(
    dims: &[usize],
    dst_strides: &[isize],
    dst_offset: isize,
    dst_len: usize,
) -> Vec<i32> {
    let numel = dims.iter().product::<usize>();
    let mut dst = vec![0i32; dst_len];
    for linear_idx in 0..numel {
        let dst_idx = linear_offset(linear_idx, dims, dst_strides, dst_offset) as usize;
        dst[dst_idx] = linear_idx as i32;
    }
    dst
}

fn host_metadata_not_equal_reference(
    lhs: &[i32],
    rhs: &[i32],
    dims: &[usize],
    lhs_strides: &[isize],
    rhs_strides: &[isize],
    dst_strides: &[isize],
) -> Vec<u8> {
    let numel = dims.iter().product::<usize>();
    let mut dst = vec![0u8; numel];
    for linear_idx in 0..numel {
        let lhs_idx = linear_offset(linear_idx, dims, lhs_strides, 0) as usize;
        let rhs_idx = linear_offset(linear_idx, dims, rhs_strides, 0) as usize;
        let dst_idx = linear_offset(linear_idx, dims, dst_strides, 0) as usize;
        dst[dst_idx] = u8::from(lhs[lhs_idx] != rhs[rhs_idx]);
    }
    dst
}

fn host_metadata_not_equal_bool_reference(
    lhs: &[u8],
    rhs: &[u8],
    dims: &[usize],
    lhs_strides: &[isize],
    rhs_strides: &[isize],
    dst_strides: &[isize],
) -> Vec<u8> {
    let numel = dims.iter().product::<usize>();
    let mut dst = vec![0u8; numel];
    for linear_idx in 0..numel {
        let lhs_idx = linear_offset(linear_idx, dims, lhs_strides, 0) as usize;
        let rhs_idx = linear_offset(linear_idx, dims, rhs_strides, 0) as usize;
        let dst_idx = linear_offset(linear_idx, dims, dst_strides, 0) as usize;
        dst[dst_idx] = u8::from(lhs[lhs_idx] != rhs[rhs_idx]);
    }
    dst
}

fn host_metadata_equal_bool_reference(
    lhs: &[u8],
    rhs: &[u8],
    dims: &[usize],
    lhs_strides: &[isize],
    rhs_strides: &[isize],
    dst_strides: &[isize],
) -> Vec<u8> {
    host_metadata_not_equal_bool_reference(lhs, rhs, dims, lhs_strides, rhs_strides, dst_strides)
        .into_iter()
        .map(|value| u8::from(value == 0))
        .collect()
}

fn host_metadata_equal_reference(
    lhs: &[i32],
    rhs: &[i32],
    dims: &[usize],
    lhs_strides: &[isize],
    rhs_strides: &[isize],
    dst_strides: &[isize],
) -> Vec<u8> {
    host_metadata_not_equal_reference(lhs, rhs, dims, lhs_strides, rhs_strides, dst_strides)
        .into_iter()
        .map(|value| u8::from(value == 0))
        .collect()
}

fn host_metadata_where_i32_reference(
    cond: &[u8],
    on_true: &[i32],
    on_false: &[i32],
    dims: &[usize],
    cond_strides: &[isize],
    true_strides: &[isize],
    false_strides: &[isize],
    dst_strides: &[isize],
) -> Vec<i32> {
    let numel = dims.iter().product::<usize>();
    let mut dst = vec![0i32; numel];
    for linear_idx in 0..numel {
        let cond_idx = linear_offset(linear_idx, dims, cond_strides, 0) as usize;
        let true_idx = linear_offset(linear_idx, dims, true_strides, 0) as usize;
        let false_idx = linear_offset(linear_idx, dims, false_strides, 0) as usize;
        let dst_idx = linear_offset(linear_idx, dims, dst_strides, 0) as usize;
        dst[dst_idx] = if cond[cond_idx] != 0 {
            on_true[true_idx]
        } else {
            on_false[false_idx]
        };
    }
    dst
}

fn host_metadata_where_bool_reference(
    cond: &[u8],
    on_true: &[u8],
    on_false: &[u8],
    dims: &[usize],
    cond_strides: &[isize],
    true_strides: &[isize],
    false_strides: &[isize],
    dst_strides: &[isize],
) -> Vec<u8> {
    let numel = dims.iter().product::<usize>();
    let mut dst = vec![0u8; numel];
    for linear_idx in 0..numel {
        let cond_idx = linear_offset(linear_idx, dims, cond_strides, 0) as usize;
        let true_idx = linear_offset(linear_idx, dims, true_strides, 0) as usize;
        let false_idx = linear_offset(linear_idx, dims, false_strides, 0) as usize;
        let dst_idx = linear_offset(linear_idx, dims, dst_strides, 0) as usize;
        dst[dst_idx] = if cond[cond_idx] != 0 {
            on_true[true_idx]
        } else {
            on_false[false_idx]
        };
    }
    dst
}

fn host_metadata_sum_bool_reference(
    input: &[u8],
    input_dims: &[usize],
    input_strides: &[isize],
    kept_axes: &[usize],
    reduced_axes: &[usize],
) -> Vec<i32> {
    let output_dims: Vec<usize> = kept_axes.iter().map(|&axis| input_dims[axis]).collect();
    let output_numel = output_dims.iter().product::<usize>();
    let reduced_dims: Vec<usize> = reduced_axes.iter().map(|&axis| input_dims[axis]).collect();
    let reduced_numel = reduced_dims.iter().product::<usize>();
    let mut out = vec![0i32; output_numel];

    for out_linear_idx in 0..output_numel {
        let mut base = 0isize;
        let mut remainder = out_linear_idx;
        for &axis in kept_axes {
            let coord = remainder % input_dims[axis];
            remainder /= input_dims[axis];
            base += (coord as isize) * input_strides[axis];
        }

        let mut acc = 0i32;
        for red_linear_idx in 0..reduced_numel {
            let mut input_idx = base;
            let mut red_remainder = red_linear_idx;
            for &axis in reduced_axes {
                let coord = red_remainder % input_dims[axis];
                red_remainder /= input_dims[axis];
                input_idx += (coord as isize) * input_strides[axis];
            }
            acc += i32::from(input[input_idx as usize] != 0);
        }
        out[out_linear_idx] = acc;
    }

    out
}

fn host_metadata_all_bool_reference(
    input: &[u8],
    input_dims: &[usize],
    input_strides: &[isize],
    kept_axes: &[usize],
    reduced_axes: &[usize],
) -> Vec<u8> {
    let output_dims: Vec<usize> = kept_axes.iter().map(|&axis| input_dims[axis]).collect();
    let output_numel = output_dims.iter().product::<usize>();
    let reduced_dims: Vec<usize> = reduced_axes.iter().map(|&axis| input_dims[axis]).collect();
    let reduced_numel = reduced_dims.iter().product::<usize>();
    let mut out = vec![1u8; output_numel];

    for out_linear_idx in 0..output_numel {
        let mut base = 0isize;
        let mut remainder = out_linear_idx;
        for &axis in kept_axes {
            let coord = remainder % input_dims[axis];
            remainder /= input_dims[axis];
            base += (coord as isize) * input_strides[axis];
        }

        let mut acc = 1u8;
        for red_linear_idx in 0..reduced_numel {
            let mut input_idx = base;
            let mut red_remainder = red_linear_idx;
            for &axis in reduced_axes {
                let coord = red_remainder % input_dims[axis];
                red_remainder /= input_dims[axis];
                input_idx += (coord as isize) * input_strides[axis];
            }
            acc = if input[input_idx as usize] != 0 {
                acc
            } else {
                0
            };
        }
        out[out_linear_idx] = acc;
    }

    out
}

fn host_metadata_any_bool_reference(
    input: &[u8],
    input_dims: &[usize],
    input_strides: &[isize],
    kept_axes: &[usize],
    reduced_axes: &[usize],
) -> Vec<u8> {
    let output_dims: Vec<usize> = kept_axes.iter().map(|&axis| input_dims[axis]).collect();
    let output_numel = output_dims.iter().product::<usize>();
    let reduced_dims: Vec<usize> = reduced_axes.iter().map(|&axis| input_dims[axis]).collect();
    let reduced_numel = reduced_dims.iter().product::<usize>();
    let mut out = vec![0u8; output_numel];

    for out_linear_idx in 0..output_numel {
        let mut base = 0isize;
        let mut remainder = out_linear_idx;
        for &axis in kept_axes {
            let coord = remainder % input_dims[axis];
            remainder /= input_dims[axis];
            base += (coord as isize) * input_strides[axis];
        }

        let mut acc = 0u8;
        for red_linear_idx in 0..reduced_numel {
            let mut input_idx = base;
            let mut red_remainder = red_linear_idx;
            for &axis in reduced_axes {
                let coord = red_remainder % input_dims[axis];
                red_remainder /= input_dims[axis];
                input_idx += (coord as isize) * input_strides[axis];
            }
            acc = if input[input_idx as usize] != 0 {
                1
            } else {
                acc
            };
        }
        out[out_linear_idx] = acc;
    }

    out
}

#[test]
fn cuda_runtime_can_get_or_create_device_zero_handle() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    let runtime = tenferro_device::cuda::runtime::get_or_init(0).unwrap();
    assert_eq!(runtime.device_id(), 0);
}

#[test]
fn cuda_runtime_dtod_copy_round_trips_small_buffer() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    let runtime = tenferro_device::cuda::runtime::get_or_init(0).unwrap();
    let src = runtime.alloc::<f32>(4).unwrap();
    let dst = runtime.alloc::<f32>(4).unwrap();
    runtime.copy_htod(&[1.0_f32, 2.0, 3.0, 4.0], &src).unwrap();
    runtime.copy_dtod(&src, &dst).unwrap();
    let got = runtime.copy_dtoh(&dst).unwrap();
    assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn cuda_runtime_strided_copy_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    let runtime = tenferro_device::cuda::runtime::get_or_init(0).unwrap();
    let src_data: Vec<f32> = (1..=24).map(|value| value as f32).collect();
    let src = runtime.alloc::<f32>(src_data.len()).unwrap();
    let dst = runtime.alloc::<f32>(src_data.len()).unwrap();
    runtime.copy_htod(&src_data, &src).unwrap();

    let dims = [4usize, 2, 3];
    let src_strides = [6isize, 1, 2];
    let spec = tenferro_device::cuda::runtime::StridedCopySpec::to_contiguous(
        &dims,
        &src_strides,
        0,
        tenferro_device::cuda::runtime::ContiguousOrder::ColumnMajor,
    )
    .unwrap();
    runtime.copy_strided(&src, &dst, &spec).unwrap();

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected =
        host_strided_copy_reference(&src_data, &dims, &src_strides, 0, spec.dst_strides());
    assert_eq!(got, expected);
}

#[test]
fn cuda_runtime_copy_strided_with_conj_transform_matches_host() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{
        self, ContiguousOrder, StridedCopySpec, StridedCopyTransform,
    };

    let runtime = runtime::get_or_init(0).unwrap();
    let dims = [2usize, 2];
    let src_strides = [3isize, 1];
    let src_offset = 1isize;
    let src_data = vec![
        Complex64::new(100.0, -100.0),
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, 4.0),
        Complex64::new(200.0, -200.0),
        Complex64::new(5.0, 6.0),
        Complex64::new(7.0, 8.0),
    ];
    let src = runtime.alloc::<Complex64>(src_data.len()).unwrap();
    let dst = runtime.alloc::<Complex64>(dims.iter().product()).unwrap();
    runtime.copy_htod(&src_data, &src).unwrap();

    let spec = StridedCopySpec::to_contiguous(
        &dims,
        &src_strides,
        src_offset,
        ContiguousOrder::ColumnMajor,
    )
    .unwrap();

    unsafe {
        runtime
            .copy_strided_raw_with_transform(
                src.device_ptr().cast_const(),
                dst.device_ptr(),
                &spec,
                StridedCopyTransform::Conj,
            )
            .unwrap();
    }

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected = host_strided_copy_conj_complex64_reference(
        &src_data,
        &dims,
        &src_strides,
        src_offset,
        spec.dst_strides(),
    );
    assert_eq!(got, expected);
}

#[test]
fn cuda_runtime_copy_strided_with_conj_transform_rejects_non_complex_types() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{
        self, ContiguousOrder, StridedCopySpec, StridedCopyTransform,
    };

    let runtime = runtime::get_or_init(0).unwrap();
    let src_data = vec![1.0_f64, 2.0, 3.0, 4.0];
    let src = runtime.alloc::<f64>(src_data.len()).unwrap();
    let dst = runtime.alloc::<f64>(src_data.len()).unwrap();
    runtime.copy_htod(&src_data, &src).unwrap();

    let spec =
        StridedCopySpec::to_contiguous(&[2usize, 2], &[1isize, 2], 0, ContiguousOrder::ColumnMajor)
            .unwrap();

    let err = unsafe {
        runtime
            .copy_strided_raw_with_transform(
                src.device_ptr().cast_const(),
                dst.device_ptr(),
                &spec,
                StridedCopyTransform::Conj,
            )
            .unwrap_err()
    };

    assert!(
        err.to_string().contains("conj transform requires"),
        "expected conj transform rejection for non-complex element type, got {err}"
    );
}

#[test]
fn cuda_runtime_real_scalar_kernels_match_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    let runtime = tenferro_device::cuda::runtime::get_or_init(0).unwrap();
    let dims = [3usize, 2];
    let view_strides = [2isize, 1];
    let contiguous_strides = [1isize, 3];
    let kept_axes = [1usize];
    let reduced_axes = [0usize];

    let lhs_host = vec![1.0_f64, -2.0, 3.5, -4.5, 5.0, -6.0];
    let rhs_host = vec![-0.5_f64, 1.5, -2.5, 3.5, -4.5, 5.5];
    let lhs = runtime.alloc::<f64>(lhs_host.len()).unwrap();
    let rhs = runtime.alloc::<f64>(rhs_host.len()).unwrap();
    let add_out = runtime.alloc::<f64>(lhs_host.len()).unwrap();
    let abs_out = runtime.alloc::<f64>(lhs_host.len()).unwrap();
    let reduce_out = runtime.alloc::<f64>(2).unwrap();
    runtime.copy_htod(&lhs_host, &lhs).unwrap();
    runtime.copy_htod(&rhs_host, &rhs).unwrap();

    unsafe {
        runtime
            .pointwise_binary_real_f64_raw(
                tenferro_device::cuda::runtime::RealBinaryOp::Add,
                1.0,
                lhs.device_ptr().cast_const(),
                &dims,
                &view_strides,
                0,
                rhs.device_ptr().cast_const(),
                &view_strides,
                0,
                0.0,
                add_out.device_ptr(),
                &contiguous_strides,
                0,
            )
            .unwrap();

        runtime
            .pointwise_unary_real_f64_raw(
                tenferro_device::cuda::runtime::RealUnaryOp::Abs,
                1.0,
                add_out.device_ptr().cast_const(),
                &dims,
                &contiguous_strides,
                0,
                0.0,
                abs_out.device_ptr(),
                &contiguous_strides,
                0,
            )
            .unwrap();

        runtime
            .reduce_real_f64_raw(
                tenferro_device::cuda::runtime::RealReductionOp::Sum,
                1.0,
                add_out.device_ptr().cast_const(),
                &dims,
                &contiguous_strides,
                0,
                0.0,
                reduce_out.device_ptr(),
                &[2],
                &[1],
                0,
                &kept_axes,
                &reduced_axes,
            )
            .unwrap();
    }

    let add_got = runtime.copy_dtoh(&add_out).unwrap();
    let abs_got = runtime.copy_dtoh(&abs_out).unwrap();
    let reduce_got = runtime.copy_dtoh(&reduce_out).unwrap();

    let add_expected = host_binary_add_reference(
        &lhs_host,
        &rhs_host,
        &dims,
        &view_strides,
        &view_strides,
        &contiguous_strides,
    );
    let abs_expected = host_unary_abs_reference(
        &add_expected,
        &dims,
        &contiguous_strides,
        &contiguous_strides,
    );
    let reduce_expected = host_sum_reduction_reference(
        &add_expected,
        &dims,
        &contiguous_strides,
        &kept_axes,
        &reduced_axes,
    );

    assert_eq!(add_got, add_expected);
    assert_eq!(abs_got, abs_expected);
    assert_eq!(reduce_got, reduce_expected);
}

#[test]
fn cuda_runtime_real_unary_log_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{self, ContiguousOrder, RealUnaryOp, StridedCopySpec};

    let runtime = runtime::get_or_init(0).unwrap();
    let src_data = vec![1.0_f64, std::f64::consts::E, 4.0, 16.0, 0.5, 2.0];
    let src = runtime.alloc::<f64>(src_data.len()).unwrap();
    let dst = runtime.alloc::<f64>(src_data.len()).unwrap();
    runtime.copy_htod(&src_data, &src).unwrap();

    let dims = [3usize, 2];
    let src_strides = [2isize, 1];
    let spec = StridedCopySpec::to_contiguous(&dims, &src_strides, 0, ContiguousOrder::ColumnMajor)
        .unwrap();

    unsafe {
        runtime
            .pointwise_unary_real_f64_raw(
                RealUnaryOp::Log,
                1.0,
                src.device_ptr().cast_const(),
                &dims,
                &src_strides,
                0,
                0.0,
                dst.device_ptr(),
                spec.dst_strides(),
                0,
            )
            .unwrap();
    }

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected = host_unary_log_reference(&src_data, &dims, &src_strides, spec.dst_strides());
    assert_eq!(got.len(), expected.len());
    for (lhs, rhs) in got.iter().zip(expected.iter()) {
        assert!((lhs - rhs).abs() < 1.0e-12, "got {lhs}, expected {rhs}");
    }
}

#[test]
fn cuda_runtime_complex64_abs_real_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{self, ComplexRealUnaryOp};

    let runtime = runtime::get_or_init(0).unwrap();
    let dims = [3usize, 2];
    let src_strides = [2isize, 1];
    let dst_strides = [1isize, 3];
    let src_data = vec![
        Complex64::new(3.0, 4.0),
        Complex64::new(5.0, 12.0),
        Complex64::new(8.0, 15.0),
        Complex64::new(7.0, 24.0),
        Complex64::new(9.0, 40.0),
        Complex64::new(12.0, 35.0),
    ];
    let src = runtime.alloc::<Complex64>(src_data.len()).unwrap();
    let dst = runtime.alloc::<f64>(src_data.len()).unwrap();
    runtime.copy_htod(&src_data, &src).unwrap();

    unsafe {
        runtime
            .pointwise_unary_complex64_to_real_f64_raw(
                ComplexRealUnaryOp::Abs,
                1.0,
                src.device_ptr().cast_const(),
                &dims,
                &src_strides,
                0,
                0.0,
                dst.device_ptr(),
                &dst_strides,
                0,
            )
            .unwrap();
    }

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected =
        host_unary_abs_real_complex64_reference(&src_data, &dims, &src_strides, &dst_strides);
    assert_eq!(got, expected);
}

#[test]
fn cuda_runtime_complex64_real_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{self, ComplexRealUnaryOp};

    let runtime = runtime::get_or_init(0).unwrap();
    let dims = [3usize, 2];
    let src_strides = [2isize, 1];
    let dst_strides = [1isize, 3];
    let src_data = vec![
        Complex64::new(3.0, 4.0),
        Complex64::new(5.0, 12.0),
        Complex64::new(8.0, 15.0),
        Complex64::new(7.0, 24.0),
        Complex64::new(9.0, 40.0),
        Complex64::new(12.0, 35.0),
    ];
    let src = runtime.alloc::<Complex64>(src_data.len()).unwrap();
    let dst = runtime.alloc::<f64>(src_data.len()).unwrap();
    runtime.copy_htod(&src_data, &src).unwrap();

    unsafe {
        runtime
            .pointwise_unary_complex64_to_real_f64_raw(
                ComplexRealUnaryOp::Real,
                1.0,
                src.device_ptr().cast_const(),
                &dims,
                &src_strides,
                0,
                0.0,
                dst.device_ptr(),
                &dst_strides,
                0,
            )
            .unwrap();
    }

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected =
        host_unary_real_complex64_reference(&src_data, &dims, &src_strides, &dst_strides);
    assert_eq!(got, expected);
}

#[test]
fn cuda_runtime_complex64_imag_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{self, ComplexRealUnaryOp};

    let runtime = runtime::get_or_init(0).unwrap();
    let dims = [3usize, 2];
    let src_strides = [2isize, 1];
    let dst_strides = [1isize, 3];
    let src_data = vec![
        Complex64::new(3.0, 4.0),
        Complex64::new(5.0, 12.0),
        Complex64::new(8.0, 15.0),
        Complex64::new(7.0, 24.0),
        Complex64::new(9.0, 40.0),
        Complex64::new(12.0, 35.0),
    ];
    let src = runtime.alloc::<Complex64>(src_data.len()).unwrap();
    let dst = runtime.alloc::<f64>(src_data.len()).unwrap();
    runtime.copy_htod(&src_data, &src).unwrap();

    unsafe {
        runtime
            .pointwise_unary_complex64_to_real_f64_raw(
                ComplexRealUnaryOp::Imag,
                1.0,
                src.device_ptr().cast_const(),
                &dims,
                &src_strides,
                0,
                0.0,
                dst.device_ptr(),
                &dst_strides,
                0,
            )
            .unwrap();
    }

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected =
        host_unary_imag_complex64_reference(&src_data, &dims, &src_strides, &dst_strides);
    assert_eq!(got, expected);
}

#[test]
fn cuda_runtime_can_pack_concat_sources() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{self, ContiguousOrder, StridedCopySpec};

    let runtime = runtime::get_or_init(0).unwrap();
    let left_host = vec![0.0_f32, 11.0, 13.0, 0.0, 12.0, 14.0];
    let right_host = vec![0.0_f32, 0.0, 31.0, 0.0, 0.0, 32.0];
    let left = runtime.alloc::<f32>(left_host.len()).unwrap();
    let right = runtime.alloc::<f32>(right_host.len()).unwrap();
    runtime.copy_htod(&left_host, &left).unwrap();
    runtime.copy_htod(&right_host, &right).unwrap();

    let left_dims = [2usize, 2];
    let right_dims = [2usize, 1];
    let left_spec =
        StridedCopySpec::to_contiguous(&left_dims, &[3isize, 1], 1, ContiguousOrder::ColumnMajor)
            .unwrap();
    let right_spec =
        StridedCopySpec::to_contiguous(&right_dims, &[3isize, 1], 2, ContiguousOrder::ColumnMajor)
            .unwrap();

    let packed = runtime
        .pack_concat_sources(
            &left,
            &left_spec,
            &right,
            &right_spec,
            1,
            ContiguousOrder::ColumnMajor,
        )
        .unwrap();

    let got = runtime.copy_dtoh(&packed).unwrap();
    assert_eq!(got, vec![11.0, 12.0, 13.0, 14.0, 31.0, 32.0]);
}

#[test]
fn cuda_runtime_real_where_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    let runtime = tenferro_device::cuda::runtime::get_or_init(0).unwrap();
    let mask_host = vec![1.0_f64, 0.0, -2.0, 0.0, 3.0, 4.0];
    let on_true_host = vec![10.0_f64, 20.0, 30.0, 40.0, 50.0, 60.0];
    let on_false_host = vec![-1.0_f64, -2.0, -3.0, -4.0, -5.0, -6.0];
    let dims = [3usize, 2usize];
    let mask_strides = [1isize, 3isize];
    let true_strides = [2isize, 1isize];
    let false_strides = [1isize, 3isize];
    let dst_strides = [1isize, 3isize];
    let expected = host_where_reference(
        &mask_host,
        &on_true_host,
        &on_false_host,
        &dims,
        &mask_strides,
        &true_strides,
        &false_strides,
        &dst_strides,
    );

    let mask = runtime.alloc::<f64>(mask_host.len()).unwrap();
    let on_true = runtime.alloc::<f64>(on_true_host.len()).unwrap();
    let on_false = runtime.alloc::<f64>(on_false_host.len()).unwrap();
    let dst = runtime.alloc::<f64>(dims.iter().product()).unwrap();
    runtime.copy_htod(&mask_host, &mask).unwrap();
    runtime.copy_htod(&on_true_host, &on_true).unwrap();
    runtime.copy_htod(&on_false_host, &on_false).unwrap();
    unsafe {
        runtime
            .pointwise_ternary_real_f64_raw(
                tenferro_device::cuda::runtime::RealTernaryOp::Where,
                1.0,
                mask.device_ptr().cast_const(),
                &dims,
                &mask_strides,
                0,
                on_true.device_ptr().cast_const(),
                &true_strides,
                0,
                on_false.device_ptr().cast_const(),
                &false_strides,
                0,
                0.0,
                dst.device_ptr(),
                &dst_strides,
                0,
            )
            .unwrap();
    }

    let got = runtime.copy_dtoh(&dst).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn cuda_runtime_metadata_iota_i32_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{
        self, MetadataGenerateOp, MetadataGenerateSpec, MetadataTensorMut,
    };

    let runtime = runtime::get_or_init(0).unwrap();
    let dims = [8usize];
    let dst_strides = [1isize];
    let spec = MetadataGenerateSpec::new(&dims, &dst_strides, 0).unwrap();
    let mut dst = runtime.alloc::<i32>(dims.iter().product()).unwrap();

    runtime
        .metadata_generate(
            MetadataGenerateOp::IotaStartZero,
            MetadataTensorMut::I32(&mut dst),
            &spec,
        )
        .unwrap();

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected = host_metadata_iota_reference(&dims, &dst_strides, 0);
    assert_eq!(got, expected);
}

#[test]
fn cuda_runtime_metadata_reduction_spec_rejects_non_partition_axes() {
    use tenferro_device::cuda::runtime::MetadataReductionSpec;

    let overlap =
        MetadataReductionSpec::new(&[2usize, 3], &[1isize, 2], 0, &[2], &[1], 0, &[0], &[0]);
    assert!(overlap
        .unwrap_err()
        .to_string()
        .contains("appears in both kept_axes and reduced_axes"));

    let missing =
        MetadataReductionSpec::new(&[2usize, 3], &[1isize, 2], 0, &[2], &[1], 0, &[0], &[]);
    assert!(missing
        .unwrap_err()
        .to_string()
        .contains("is missing from kept_axes and reduced_axes"));
}

#[test]
fn cuda_runtime_metadata_iota_i32_supports_offset_noncontiguous_layout() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{
        self, MetadataGenerateOp, MetadataGenerateSpec, MetadataTensorMut,
    };

    let runtime = runtime::get_or_init(0).unwrap();
    let dims = [2usize, 2];
    let dst_strides = [1isize, 3];
    let dst_offset = 1isize;
    let spec = MetadataGenerateSpec::new(&dims, &dst_strides, dst_offset).unwrap();
    let mut dst = runtime.alloc::<i32>(6).unwrap();

    runtime
        .metadata_generate(
            MetadataGenerateOp::IotaStartZero,
            MetadataTensorMut::I32(&mut dst),
            &spec,
        )
        .unwrap();

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected = host_metadata_iota_layout_reference(&dims, &dst_strides, dst_offset, 6);
    assert_eq!(got, expected);
}

#[test]
fn cuda_runtime_metadata_bool_compare_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{
        self, MetadataBinaryOp, MetadataBinarySpec, MetadataTensorMut, MetadataTensorRef,
    };

    let runtime = runtime::get_or_init(0).unwrap();
    let dims = [6usize];
    let strides = [1isize];
    let lhs_data = [0u8, 1, 1, 0, 1, 0];
    let rhs_data = [0u8, 0, 1, 1, 1, 0];
    let lhs = runtime.alloc::<u8>(lhs_data.len()).unwrap();
    let rhs = runtime.alloc::<u8>(rhs_data.len()).unwrap();
    let mut not_equal = runtime.alloc::<u8>(lhs_data.len()).unwrap();
    let mut equal = runtime.alloc::<u8>(lhs_data.len()).unwrap();
    runtime.copy_htod(&lhs_data, &lhs).unwrap();
    runtime.copy_htod(&rhs_data, &rhs).unwrap();

    let spec = MetadataBinarySpec::new(&dims, &strides, 0, &strides, 0, &strides, 0).unwrap();
    runtime
        .metadata_binary(
            MetadataBinaryOp::NotEqual,
            MetadataTensorRef::Bool(&lhs),
            MetadataTensorRef::Bool(&rhs),
            MetadataTensorMut::Bool(&mut not_equal),
            &spec,
        )
        .unwrap();
    runtime
        .metadata_binary(
            MetadataBinaryOp::Equal,
            MetadataTensorRef::Bool(&lhs),
            MetadataTensorRef::Bool(&rhs),
            MetadataTensorMut::Bool(&mut equal),
            &spec,
        )
        .unwrap();

    let got_not_equal = runtime.copy_dtoh(&not_equal).unwrap();
    let got_equal = runtime.copy_dtoh(&equal).unwrap();
    let expected_not_equal = host_metadata_not_equal_bool_reference(
        &lhs_data, &rhs_data, &dims, &strides, &strides, &strides,
    );
    let expected_equal = host_metadata_equal_bool_reference(
        &lhs_data, &rhs_data, &dims, &strides, &strides, &strides,
    );
    assert_eq!(got_not_equal, expected_not_equal);
    assert_eq!(got_equal, expected_equal);
}

#[test]
fn cuda_runtime_metadata_where_bool_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{
        self, MetadataTensorMut, MetadataTensorRef, MetadataTernaryOp, MetadataTernarySpec,
    };

    let runtime = runtime::get_or_init(0).unwrap();
    let dims = [6usize];
    let strides = [1isize];
    let cond_data = [0u8, 1, 0, 0, 1, 0];
    let true_data = [0u8, 11, 0, 0, 22, 0];
    let false_data = [0u8, 101, 0, 0, 202, 0];
    let cond = runtime.alloc::<u8>(cond_data.len()).unwrap();
    let on_true = runtime.alloc::<u8>(true_data.len()).unwrap();
    let on_false = runtime.alloc::<u8>(false_data.len()).unwrap();
    let mut dst = runtime.alloc::<u8>(false_data.len()).unwrap();
    runtime.copy_htod(&cond_data, &cond).unwrap();
    runtime.copy_htod(&true_data, &on_true).unwrap();
    runtime.copy_htod(&false_data, &on_false).unwrap();

    let spec = MetadataTernarySpec::new(&dims, &strides, 0, &strides, 0, &strides, 0, &strides, 0)
        .unwrap();
    runtime
        .metadata_ternary(
            MetadataTernaryOp::Where,
            MetadataTensorRef::Bool(&cond),
            MetadataTensorRef::Bool(&on_true),
            MetadataTensorRef::Bool(&on_false),
            MetadataTensorMut::Bool(&mut dst),
            &spec,
        )
        .unwrap();

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected = host_metadata_where_bool_reference(
        &cond_data,
        &true_data,
        &false_data,
        &dims,
        &strides,
        &strides,
        &strides,
        &strides,
    );
    assert_eq!(got, expected);
}

#[test]
fn cuda_runtime_metadata_not_equal_and_sum_match_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{
        self, MetadataBinaryOp, MetadataBinarySpec, MetadataReductionOp, MetadataReductionSpec,
        MetadataTensorMut, MetadataTensorRef,
    };

    let runtime = runtime::get_or_init(0).unwrap();
    let dims = [6usize];
    let strides = [1isize];
    let lhs_data = [0i32, 1, 2, 3, 4, 5];
    let rhs_data = [0i32, 0, 2, 9, 4, 8];
    let lhs = runtime.alloc::<i32>(lhs_data.len()).unwrap();
    let rhs = runtime.alloc::<i32>(rhs_data.len()).unwrap();
    let mut mask = runtime.alloc::<u8>(lhs_data.len()).unwrap();
    let mut eq_mask = runtime.alloc::<u8>(lhs_data.len()).unwrap();
    let mut sum = runtime.alloc::<i32>(1).unwrap();
    let mut all = runtime.alloc::<u8>(1).unwrap();
    let mut any = runtime.alloc::<u8>(1).unwrap();
    runtime.copy_htod(&lhs_data, &lhs).unwrap();
    runtime.copy_htod(&rhs_data, &rhs).unwrap();

    let binary_spec =
        MetadataBinarySpec::new(&dims, &strides, 0, &strides, 0, &strides, 0).unwrap();
    runtime
        .metadata_binary(
            MetadataBinaryOp::NotEqual,
            MetadataTensorRef::I32(&lhs),
            MetadataTensorRef::I32(&rhs),
            MetadataTensorMut::Bool(&mut mask),
            &binary_spec,
        )
        .unwrap();
    runtime
        .metadata_binary(
            MetadataBinaryOp::Equal,
            MetadataTensorRef::I32(&lhs),
            MetadataTensorRef::I32(&rhs),
            MetadataTensorMut::Bool(&mut eq_mask),
            &binary_spec,
        )
        .unwrap();

    let reduction_spec =
        MetadataReductionSpec::new(&dims, &strides, 0, &[], &[], 0, &[], &[0]).unwrap();
    runtime
        .metadata_reduction(
            MetadataReductionOp::Sum,
            MetadataTensorRef::Bool(&mask),
            MetadataTensorMut::I32(&mut sum),
            &reduction_spec,
        )
        .unwrap();
    runtime
        .metadata_reduction(
            MetadataReductionOp::All,
            MetadataTensorRef::Bool(&mask),
            MetadataTensorMut::Bool(&mut all),
            &reduction_spec,
        )
        .unwrap();
    runtime
        .metadata_reduction(
            MetadataReductionOp::Any,
            MetadataTensorRef::Bool(&mask),
            MetadataTensorMut::Bool(&mut any),
            &reduction_spec,
        )
        .unwrap();

    let got_mask = runtime.copy_dtoh(&mask).unwrap();
    let got_eq_mask = runtime.copy_dtoh(&eq_mask).unwrap();
    let got_sum = runtime.copy_dtoh(&sum).unwrap();
    let got_all = runtime.copy_dtoh(&all).unwrap();
    let got_any = runtime.copy_dtoh(&any).unwrap();
    let expected_mask = host_metadata_not_equal_reference(
        &lhs_data, &rhs_data, &dims, &strides, &strides, &strides,
    );
    let expected_eq_mask =
        host_metadata_equal_reference(&lhs_data, &rhs_data, &dims, &strides, &strides, &strides);
    let expected_sum = host_metadata_sum_bool_reference(&expected_mask, &dims, &strides, &[], &[0]);
    let expected_all = host_metadata_all_bool_reference(&expected_mask, &dims, &strides, &[], &[0]);
    let expected_any = host_metadata_any_bool_reference(&expected_mask, &dims, &strides, &[], &[0]);
    assert_eq!(got_mask, expected_mask);
    assert_eq!(got_eq_mask, expected_eq_mask);
    assert_eq!(got_sum, expected_sum);
    assert_eq!(got_all, expected_all);
    assert_eq!(got_any, expected_any);
}

#[test]
fn cuda_runtime_metadata_where_selects_integer_values() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{
        self, MetadataTensorMut, MetadataTensorRef, MetadataTernaryOp, MetadataTernarySpec,
    };

    let runtime = runtime::get_or_init(0).unwrap();
    let dims = [6usize];
    let strides = [1isize];
    let cond_data = [1u8, 0, 1, 0, 0, 1];
    let true_data = [10i32, 20, 30, 40, 50, 60];
    let false_data = [-1i32, -2, -3, -4, -5, -6];
    let cond = runtime.alloc::<u8>(cond_data.len()).unwrap();
    let on_true = runtime.alloc::<i32>(true_data.len()).unwrap();
    let on_false = runtime.alloc::<i32>(false_data.len()).unwrap();
    let mut dst = runtime.alloc::<i32>(true_data.len()).unwrap();
    runtime.copy_htod(&cond_data, &cond).unwrap();
    runtime.copy_htod(&true_data, &on_true).unwrap();
    runtime.copy_htod(&false_data, &on_false).unwrap();

    let spec = MetadataTernarySpec::new(&dims, &strides, 0, &strides, 0, &strides, 0, &strides, 0)
        .unwrap();
    runtime
        .metadata_ternary(
            MetadataTernaryOp::Where,
            MetadataTensorRef::Bool(&cond),
            MetadataTensorRef::I32(&on_true),
            MetadataTensorRef::I32(&on_false),
            MetadataTensorMut::I32(&mut dst),
            &spec,
        )
        .unwrap();

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected = host_metadata_where_i32_reference(
        &cond_data,
        &true_data,
        &false_data,
        &dims,
        &strides,
        &strides,
        &strides,
        &strides,
    );
    assert_eq!(got, expected);
}

#[test]
fn cuda_runtime_real_binary_pow_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    let runtime = tenferro_device::cuda::runtime::get_or_init(0).unwrap();
    let dims = [4usize];
    let strides = [1isize];
    let lhs_host = vec![2.0_f64, 9.0, 16.0, 27.0];
    let rhs_host = vec![3.0_f64, 0.5, 0.25, 2.0];
    let lhs = runtime.alloc::<f64>(lhs_host.len()).unwrap();
    let rhs = runtime.alloc::<f64>(rhs_host.len()).unwrap();
    let dst = runtime.alloc::<f64>(lhs_host.len()).unwrap();
    runtime.copy_htod(&lhs_host, &lhs).unwrap();
    runtime.copy_htod(&rhs_host, &rhs).unwrap();

    unsafe {
        runtime
            .pointwise_binary_real_f64_raw(
                tenferro_device::cuda::runtime::RealBinaryOp::Pow,
                1.0,
                lhs.device_ptr().cast_const(),
                &dims,
                &strides,
                0,
                rhs.device_ptr().cast_const(),
                &strides,
                0,
                0.0,
                dst.device_ptr(),
                &strides,
                0,
            )
            .unwrap();
    }

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected =
        host_binary_pow_reference(&lhs_host, &rhs_host, &dims, &strides, &strides, &strides);
    for (idx, (got_v, exp_v)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got_v - exp_v).abs() < 1.0e-12,
            "pow mismatch at {idx}: got {got_v}, expected {exp_v}"
        );
    }
}

#[test]
fn cuda_runtime_real_unary_sqrt_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{self, ContiguousOrder, RealUnaryOp, StridedCopySpec};

    let runtime = runtime::get_or_init(0).unwrap();
    let src_data = vec![1.0_f64, 4.0, 9.0, 16.0, 0.25, 2.25];
    let src = runtime.alloc::<f64>(src_data.len()).unwrap();
    let dst = runtime.alloc::<f64>(src_data.len()).unwrap();
    runtime.copy_htod(&src_data, &src).unwrap();

    let dims = [3usize, 2];
    let src_strides = [2isize, 1];
    let spec = StridedCopySpec::to_contiguous(&dims, &src_strides, 0, ContiguousOrder::ColumnMajor)
        .unwrap();

    unsafe {
        runtime
            .pointwise_unary_real_f64_raw(
                RealUnaryOp::Sqrt,
                1.0,
                src.device_ptr().cast_const(),
                &dims,
                &src_strides,
                0,
                0.0,
                dst.device_ptr(),
                spec.dst_strides(),
                0,
            )
            .unwrap();
    }

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected = host_unary_sqrt_reference(&src_data, &dims, &src_strides, spec.dst_strides());
    assert_eq!(got.len(), expected.len());
    for (lhs, rhs) in got.iter().zip(expected.iter()) {
        assert!((lhs - rhs).abs() < 1.0e-12, "got {lhs}, expected {rhs}");
    }
}

#[test]
fn cuda_runtime_real_max_reduction_rejects_empty_domain() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    let runtime = tenferro_device::cuda::runtime::get_or_init(0).unwrap();
    let input = runtime.alloc::<f64>(0).unwrap();
    let output = runtime.alloc::<f64>(2).unwrap();

    let err = unsafe {
        runtime
            .reduce_real_f64_raw(
                tenferro_device::cuda::runtime::RealReductionOp::Max,
                1.0,
                input.device_ptr().cast_const(),
                &[0, 2],
                &[1, 1],
                0,
                0.0,
                output.device_ptr(),
                &[2],
                &[1],
                0,
                &[1],
                &[0],
            )
            .unwrap_err()
    };

    assert!(err.to_string().contains("empty"));
}

#[test]
fn cuda_runtime_real_prod_reduction_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    let runtime = tenferro_device::cuda::runtime::get_or_init(0).unwrap();
    let input_host = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = runtime.alloc::<f64>(input_host.len()).unwrap();
    let output = runtime.alloc::<f64>(2).unwrap();
    runtime.copy_htod(&input_host, &input).unwrap();

    let dims = [3usize, 2usize];
    let input_strides = [1isize, 3isize];
    let kept_axes = [1usize];
    let reduced_axes = [0usize];

    unsafe {
        runtime
            .reduce_real_f64_raw(
                tenferro_device::cuda::runtime::RealReductionOp::Prod,
                1.0,
                input.device_ptr().cast_const(),
                &dims,
                &input_strides,
                0,
                0.0,
                output.device_ptr(),
                &[2],
                &[1],
                0,
                &kept_axes,
                &reduced_axes,
            )
            .unwrap();
    }

    let got = runtime.copy_dtoh(&output).unwrap();
    let expected = host_prod_reduction_reference(
        &input_host,
        &dims,
        &input_strides,
        &kept_axes,
        &reduced_axes,
    );
    assert_eq!(got, expected);
}

#[test]
fn cuda_runtime_zero_trailing_by_counts_f64_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    let runtime = tenferro_device::cuda::runtime::get_or_init(0).unwrap();
    let src_host = vec![1.0_f64, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let keep_counts_host = vec![1.0_f64, 2.0];
    let src = runtime.alloc::<f64>(src_host.len()).unwrap();
    let keep_counts = runtime.alloc::<f64>(keep_counts_host.len()).unwrap();
    let dst = runtime.alloc::<f64>(src_host.len()).unwrap();
    runtime.copy_htod(&src_host, &src).unwrap();
    runtime.copy_htod(&keep_counts_host, &keep_counts).unwrap();

    let spec = tenferro_device::cuda::runtime::ZeroTrailingByCountsSpec::new(
        &[2, 2, 2],
        &[1, 2, 4],
        0,
        &[1, 2, 4],
        0,
        &[1],
        0,
        1,
        2,
    )
    .unwrap();

    runtime
        .zero_trailing_by_counts(&src, &dst, &keep_counts, &spec)
        .unwrap();

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected = host_zero_trailing_by_counts_reference(
        &src_host,
        spec.dims(),
        spec.src_strides(),
        spec.src_offset(),
        &[1, 2],
        spec.keep_count_strides(),
        spec.keep_count_offset(),
        spec.axis(),
        spec.structural_rank(),
    );
    assert_eq!(got, expected);
}

#[test]
fn cuda_runtime_zero_trailing_by_counts_complex64_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    let runtime = tenferro_device::cuda::runtime::get_or_init(0).unwrap();
    let src_host = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, 2.0),
        Complex64::new(3.0, 3.0),
        Complex64::new(4.0, 4.0),
        Complex64::new(5.0, 5.0),
        Complex64::new(6.0, 6.0),
        Complex64::new(7.0, 7.0),
        Complex64::new(8.0, 8.0),
        Complex64::new(9.0, 9.0),
        Complex64::new(10.0, 10.0),
        Complex64::new(11.0, 11.0),
        Complex64::new(12.0, 12.0),
    ];
    let keep_counts_host = vec![2.0_f32, 1.0];
    let src = runtime.alloc::<Complex64>(src_host.len()).unwrap();
    let keep_counts = runtime.alloc::<f32>(keep_counts_host.len()).unwrap();
    let dst = runtime.alloc::<Complex64>(src_host.len()).unwrap();
    runtime.copy_htod(&src_host, &src).unwrap();
    runtime.copy_htod(&keep_counts_host, &keep_counts).unwrap();

    let spec = tenferro_device::cuda::runtime::ZeroTrailingByCountsSpec::new(
        &[3, 2, 2],
        &[1, 3, 6],
        0,
        &[1, 3, 6],
        0,
        &[1],
        0,
        0,
        2,
    )
    .unwrap();

    runtime
        .zero_trailing_by_counts(&src, &dst, &keep_counts, &spec)
        .unwrap();

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected = host_zero_trailing_by_counts_reference(
        &src_host,
        spec.dims(),
        spec.src_strides(),
        spec.src_offset(),
        &[2, 1],
        spec.keep_count_strides(),
        spec.keep_count_offset(),
        spec.axis(),
        spec.structural_rank(),
    );
    assert_eq!(got, expected);
}

#[test]
fn cuda_runtime_zero_trailing_by_counts_rejects_non_integer_keep_counts() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    let runtime = tenferro_device::cuda::runtime::get_or_init(0).unwrap();
    let src_host = vec![1.0_f64, 2.0, 3.0, 4.0];
    let keep_counts_host = vec![1.5_f64];
    let src = runtime.alloc::<f64>(src_host.len()).unwrap();
    let keep_counts = runtime.alloc::<f64>(keep_counts_host.len()).unwrap();
    let dst = runtime.alloc::<f64>(src_host.len()).unwrap();
    runtime.copy_htod(&src_host, &src).unwrap();
    runtime.copy_htod(&keep_counts_host, &keep_counts).unwrap();

    let spec = tenferro_device::cuda::runtime::ZeroTrailingByCountsSpec::new(
        &[2, 2],
        &[1, 2],
        0,
        &[1, 2],
        0,
        &[],
        0,
        1,
        2,
    )
    .unwrap();

    let err = runtime
        .zero_trailing_by_counts(&src, &dst, &keep_counts, &spec)
        .unwrap_err();
    assert!(err.to_string().contains("integer-valued"));
}

#[test]
fn cuda_runtime_triangular_part_f64_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{TriangularHalf, TriangularPartSpec};

    let runtime = tenferro_device::cuda::runtime::get_or_init(0).unwrap();
    let src_host: Vec<f64> = (1..=24).map(|value| value as f64).collect();
    let src = runtime.alloc::<f64>(src_host.len()).unwrap();
    let dst = runtime.alloc::<f64>(src_host.len()).unwrap();
    runtime.copy_htod(&src_host, &src).unwrap();

    let dims = [3usize, 2, 4];
    let src_strides = [1isize, 3, 6];
    let spec = TriangularPartSpec::new(
        &dims,
        &src_strides,
        0,
        &[1isize, 3, 6],
        0,
        -1,
        TriangularHalf::Lower,
    )
    .unwrap();

    runtime.triangular_part(&src, &dst, &spec).unwrap();

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected = host_triangular_part_reference(
        &src_host,
        &dims,
        &src_strides,
        0,
        &[1isize, 3, 6],
        -1,
        TriangularHalf::Lower,
    );
    assert_eq!(got, expected);
}

#[test]
fn cuda_runtime_triangular_part_complex64_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::{TriangularHalf, TriangularPartSpec};

    let runtime = tenferro_device::cuda::runtime::get_or_init(0).unwrap();
    let src_host: Vec<Complex64> = (1..=18)
        .map(|value| Complex64::new(value as f64, -(value as f64)))
        .collect();
    let src = runtime.alloc::<Complex64>(src_host.len()).unwrap();
    let dst = runtime.alloc::<Complex64>(src_host.len()).unwrap();
    runtime.copy_htod(&src_host, &src).unwrap();

    let dims = [3usize, 3, 2];
    let src_strides = [1isize, 3, 9];
    let spec = TriangularPartSpec::new(
        &dims,
        &src_strides,
        0,
        &[1isize, 3, 9],
        0,
        1,
        TriangularHalf::Upper,
    )
    .unwrap();

    runtime.triangular_part(&src, &dst, &spec).unwrap();

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected = host_triangular_part_reference(
        &src_host,
        &dims,
        &src_strides,
        0,
        &[1isize, 3, 9],
        1,
        TriangularHalf::Upper,
    );
    assert_eq!(got, expected);
}

#[test]
fn cuda_runtime_triangular_merge_f64_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::TriangularMergeSpec;

    let runtime = tenferro_device::cuda::runtime::get_or_init(0).unwrap();
    let lower_host: Vec<f64> = (1..=18).map(|value| value as f64).collect();
    let upper_host: Vec<f64> = (101..=124).map(|value| value as f64).collect();
    let lower = runtime.alloc::<f64>(lower_host.len()).unwrap();
    let upper = runtime.alloc::<f64>(upper_host.len()).unwrap();
    let dst = runtime.alloc::<f64>(24).unwrap();
    runtime.copy_htod(&lower_host, &lower).unwrap();
    runtime.copy_htod(&upper_host, &upper).unwrap();

    let dims = [3usize, 4, 2];
    let lower_strides = [1isize, 3, 9];
    let upper_strides = [1isize, 3, 12];
    let dst_strides = [1isize, 3, 12];
    let spec =
        TriangularMergeSpec::new(&dims, &lower_strides, 0, &upper_strides, 0, &dst_strides, 0)
            .unwrap();

    runtime
        .triangular_merge(&lower, &upper, &dst, &spec)
        .unwrap();

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected = host_triangular_merge_reference(
        &lower_host,
        &upper_host,
        &dims,
        &lower_strides,
        0,
        &upper_strides,
        0,
        &dst_strides,
    );
    assert_eq!(got, expected);
}

#[test]
fn cuda_runtime_triangular_merge_complex64_matches_host_reference() {
    if std::env::var_os("TENFERRO_TEST_CUDA").is_none() {
        return;
    }

    use tenferro_device::cuda::runtime::TriangularMergeSpec;

    let runtime = tenferro_device::cuda::runtime::get_or_init(0).unwrap();
    let lower_host: Vec<Complex64> = (1..=8)
        .map(|value| Complex64::new(value as f64, value as f64 + 0.5))
        .collect();
    let upper_host: Vec<Complex64> = (21..=24)
        .map(|value| Complex64::new(value as f64, -(value as f64)))
        .collect();
    let lower = runtime.alloc::<Complex64>(lower_host.len()).unwrap();
    let upper = runtime.alloc::<Complex64>(upper_host.len()).unwrap();
    let dst = runtime.alloc::<Complex64>(8).unwrap();
    runtime.copy_htod(&lower_host, &lower).unwrap();
    runtime.copy_htod(&upper_host, &upper).unwrap();

    let dims = [4usize, 2];
    let lower_strides = [1isize, 4];
    let upper_strides = [1isize, 2];
    let dst_strides = [1isize, 4];
    let spec =
        TriangularMergeSpec::new(&dims, &lower_strides, 0, &upper_strides, 0, &dst_strides, 0)
            .unwrap();

    runtime
        .triangular_merge(&lower, &upper, &dst, &spec)
        .unwrap();

    let got = runtime.copy_dtoh(&dst).unwrap();
    let expected = host_triangular_merge_reference(
        &lower_host,
        &upper_host,
        &dims,
        &lower_strides,
        0,
        &upper_strides,
        0,
        &dst_strides,
    );
    assert_eq!(got, expected);
}
