#![cfg(feature = "webgpu")]

use num_complex::Complex32;
use tenferro_gpu::{webgpu_available, WebGpuBackend};
use tenferro_tensor::{
    Error, ErrorKind, Tensor, TensorDeviceTransfer, TensorRead, TensorStructural, TensorView,
};

#[test]
fn webgpu_transpose_f32_stays_on_device_and_matches_column_major_reference() {
    if !webgpu_available() {
        return;
    }

    let mut backend = WebGpuBackend::new_default().unwrap();
    let host =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let input = backend.upload_host_tensor(&host).unwrap();

    let transposed = backend.transpose(&input, &[1, 0]).unwrap();

    assert_eq!(transposed.placement(), input.placement());
    let actual = backend.download_to_host(&transposed).unwrap();
    assert_eq!(actual.shape(), &[3, 2]);
    assert_eq!(
        actual.as_slice::<f32>().unwrap(),
        &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    );
}

#[test]
fn webgpu_to_contiguous_f32_materializes_a_noncompact_resident_view() {
    if !webgpu_available() {
        return;
    }

    let mut backend = WebGpuBackend::new_default().unwrap();
    let host = Tensor::from_vec_col_major(vec![6], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let input = backend.upload_host_tensor(&host).unwrap();
    let Tensor::F32(input) = &input else {
        unreachable!("uploaded f32 tensor must remain f32");
    };
    let view = input.backend_region_view(vec![3], vec![2], 0).unwrap();

    let materialized = backend
        .to_contiguous_read(TensorRead::from_view(TensorView::F32(view)))
        .unwrap();

    assert_eq!(materialized.placement(), input.placement());
    let actual = backend.download_to_host(&materialized).unwrap();
    assert_eq!(actual.shape(), &[3]);
    assert_eq!(actual.as_slice::<f32>().unwrap(), &[1.0, 3.0, 5.0]);
}

#[test]
fn webgpu_transpose_supports_i32_and_rejects_wgsl_unsupported_complex() {
    if !webgpu_available() {
        return;
    }

    let mut backend = WebGpuBackend::new_default().unwrap();
    let i32_host = Tensor::from_vec_col_major(vec![2, 2], vec![1_i32, 2, 3, 4]).unwrap();
    let i32_input = backend.upload_host_tensor(&i32_host).unwrap();
    let i32_output = backend.transpose(&i32_input, &[1, 0]).unwrap();
    let i32_actual = backend.download_to_host(&i32_output).unwrap();
    assert_eq!(i32_actual.as_slice::<i32>().unwrap(), &[1, 3, 2, 4]);

    let c32_host = Tensor::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex32::new(1.0, -1.0),
            Complex32::new(2.0, -2.0),
            Complex32::new(3.0, -3.0),
            Complex32::new(4.0, -4.0),
        ],
    )
    .unwrap();
    let c32_input = backend.upload_host_tensor(&c32_host).unwrap();
    let error = backend.transpose(&c32_input, &[1, 0]).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
}

#[test]
fn webgpu_transpose_rejects_invalid_permutations_before_launch() {
    if !webgpu_available() {
        return;
    }

    let mut backend = WebGpuBackend::new_default().unwrap();
    let host = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f32; 4]).unwrap();
    let input = backend.upload_host_tensor(&host).unwrap();

    let error = backend.transpose(&input, &[0, 0]).unwrap_err();

    assert!(matches!(error, Error::Validation { .. }));
}

#[test]
fn webgpu_structural_kernels_preserve_zero_length_shapes_without_launching() {
    if !webgpu_available() {
        return;
    }

    let mut backend = WebGpuBackend::new_default().unwrap();
    let host = Tensor::from_vec_col_major(vec![0, 3], Vec::<f32>::new()).unwrap();
    let input = backend.upload_host_tensor(&host).unwrap();

    let output = backend.transpose(&input, &[1, 0]).unwrap();

    assert_eq!(output.shape(), &[3, 0]);
    let actual = backend.download_to_host(&output).unwrap();
    assert!(actual.as_slice::<f32>().unwrap().is_empty());
}
