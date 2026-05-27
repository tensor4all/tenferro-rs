use super::*;

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
