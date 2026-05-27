use super::*;

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
