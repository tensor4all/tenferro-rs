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

fn host_storage_len(dims: &[usize], strides: &[isize], offset: isize) -> usize {
    assert_eq!(dims.len(), strides.len());
    if dims.contains(&0) {
        return 0;
    }

    let numel = dims.iter().product::<usize>();
    let max_offset = (0..numel)
        .map(|linear_idx| linear_offset(linear_idx, dims, strides, offset))
        .max()
        .unwrap_or(offset);
    usize::try_from(max_offset).unwrap().checked_add(1).unwrap()
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
    let mut dst = vec![0u8; host_storage_len(dims, dst_strides, 0)];
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
    let mut dst = vec![0u8; host_storage_len(dims, dst_strides, 0)];
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
    let numel = dims.iter().product::<usize>();
    let mut dst = vec![0u8; host_storage_len(dims, dst_strides, 0)];
    for linear_idx in 0..numel {
        let lhs_idx = linear_offset(linear_idx, dims, lhs_strides, 0) as usize;
        let rhs_idx = linear_offset(linear_idx, dims, rhs_strides, 0) as usize;
        let dst_idx = linear_offset(linear_idx, dims, dst_strides, 0) as usize;
        dst[dst_idx] = u8::from(lhs[lhs_idx] == rhs[rhs_idx]);
    }
    dst
}

fn host_metadata_equal_reference(
    lhs: &[i32],
    rhs: &[i32],
    dims: &[usize],
    lhs_strides: &[isize],
    rhs_strides: &[isize],
    dst_strides: &[isize],
) -> Vec<u8> {
    let numel = dims.iter().product::<usize>();
    let mut dst = vec![0u8; host_storage_len(dims, dst_strides, 0)];
    for linear_idx in 0..numel {
        let lhs_idx = linear_offset(linear_idx, dims, lhs_strides, 0) as usize;
        let rhs_idx = linear_offset(linear_idx, dims, rhs_strides, 0) as usize;
        let dst_idx = linear_offset(linear_idx, dims, dst_strides, 0) as usize;
        dst[dst_idx] = u8::from(lhs[lhs_idx] == rhs[rhs_idx]);
    }
    dst
}

fn host_metadata_bitand_reference(
    lhs: &[i32],
    rhs: &[i32],
    dims: &[usize],
    lhs_strides: &[isize],
    rhs_strides: &[isize],
    dst_strides: &[isize],
) -> Vec<i32> {
    let numel = dims.iter().product::<usize>();
    let mut dst = vec![0i32; host_storage_len(dims, dst_strides, 0)];
    for linear_idx in 0..numel {
        let lhs_idx = linear_offset(linear_idx, dims, lhs_strides, 0) as usize;
        let rhs_idx = linear_offset(linear_idx, dims, rhs_strides, 0) as usize;
        let dst_idx = linear_offset(linear_idx, dims, dst_strides, 0) as usize;
        dst[dst_idx] = lhs[lhs_idx] & rhs[rhs_idx];
    }
    dst
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
    let mut dst = vec![0i32; host_storage_len(dims, dst_strides, 0)];
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
    let mut dst = vec![0u8; host_storage_len(dims, dst_strides, 0)];
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

mod metadata_kernels;
mod numeric_kernels;
mod numeric_reductions;
mod runtime_core;
mod structural_kernels;
