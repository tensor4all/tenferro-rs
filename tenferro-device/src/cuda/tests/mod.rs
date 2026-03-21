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
