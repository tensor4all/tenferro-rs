use super::*;

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
