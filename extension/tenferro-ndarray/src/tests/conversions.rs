use ndarray::{Array2, ArrayD, IxDyn, ShapeBuilder};
use num_complex::Complex64;
use tenferro_device::{Error, LogicalMemorySpace};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{
    can_zero_copy_tensor_to_ndarray, ensure_main_memory, into_owned_data, ndarray_to_tensor,
    shape_error, tensor_to_ndarray, try_ndarray_to_tensor, try_tensor_to_ndarray, usize_strides,
};

#[test]
fn owned_ndarray_to_tensor_preserves_shape_and_values() {
    let array = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let tensor = try_ndarray_to_tensor(array).unwrap();
    assert_eq!(tensor.dims(), &[2, 2]);
    assert_eq!(tensor.buffer().as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn owned_ndarray_to_tensor_reuses_owned_buffer_when_layout_is_representable() {
    let array = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let ptr = array.as_ptr();
    let tensor = try_ndarray_to_tensor(array).unwrap();
    assert_eq!(tensor.buffer().as_slice().unwrap().as_ptr(), ptr);
}

#[test]
fn generic_owned_ndarray_input_reuses_owned_buffer_when_layout_is_representable() {
    let array = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let ptr = array.as_ptr();
    let tensor = try_ndarray_to_tensor(array).unwrap();
    assert_eq!(tensor.buffer().as_slice().unwrap().as_ptr(), ptr);
}

#[test]
fn ndarray_to_tensor_wrapper_matches_checked_conversion() {
    let array = Array2::from_shape_vec((1, 3), vec![1.0_f64, 2.0, 3.0]).unwrap();
    let tensor = ndarray_to_tensor(array);
    assert_eq!(tensor.dims(), &[1, 3]);
    assert_eq!(tensor.buffer().as_slice().unwrap(), &[1.0, 2.0, 3.0]);
}

#[test]
fn borrowed_ndarray_view_falls_back_to_owned_tensor_materialization() {
    let array = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let view = array.view().reversed_axes();
    let tensor = try_ndarray_to_tensor(view).unwrap();
    let roundtrip = tensor_to_ndarray(tensor);
    assert_eq!(roundtrip.shape(), &[2, 2]);
    assert_eq!(roundtrip, view.into_owned().into_dyn());
}

#[test]
fn tensor_to_ndarray_reuses_owned_cpu_buffer_when_layout_is_representable() {
    let tensor = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], &[2, 2], &[2, 1], 0).unwrap();
    let ptr = tensor.buffer().as_slice().unwrap().as_ptr();
    let array = try_tensor_to_ndarray(tensor).unwrap();
    assert_eq!(array.as_ptr(), ptr);
    assert_eq!(array.shape(), &[2, 2]);
}

#[test]
fn tensor_to_ndarray_wrapper_matches_checked_conversion() {
    let tensor = Tensor::from_slice(&[1.0_f64, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();
    let array = tensor_to_ndarray(tensor);
    assert_eq!(array.shape(), &[3]);
    assert_eq!(
        array.iter().copied().collect::<Vec<_>>(),
        vec![1.0, 2.0, 3.0]
    );
}

#[test]
fn tensor_to_ndarray_materializes_permuted_views_when_zero_copy_is_not_possible() {
    let tensor = Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap()
        .narrow(0, 1, 1)
        .unwrap();

    let array = tensor_to_ndarray(tensor);
    assert_eq!(array.shape(), &[1, 2]);
    assert_eq!(array.iter().copied().collect::<Vec<_>>(), vec![2.0, 4.0]);
}

#[test]
fn usize_strides_rejects_negative_values() {
    let err = usize_strides(&[-1]).unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(msg) if msg.contains("negative ndarray/tenferro stride -1"))
    );
}

#[test]
fn shape_error_wraps_ndarray_layout_failures() {
    let source = ArrayD::from_shape_vec(
        IxDyn(&[2, 2]).strides(IxDyn(&[1, 1])),
        vec![1.0_f64, 2.0, 3.0],
    )
    .unwrap_err();
    let err = shape_error(source);
    assert!(
        matches!(err, Error::InvalidArgument(msg) if msg.contains("ndarray layout conversion failed"))
    );
}

#[test]
fn zero_copy_helper_rejects_shared_views_and_conjugated_tensors() {
    let shared = Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap()
        .narrow(0, 1, 1)
        .unwrap();
    assert!(!can_zero_copy_tensor_to_ndarray(&shared));

    let complex = Tensor::from_slice(
        &[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap()
    .conj();
    assert!(!can_zero_copy_tensor_to_ndarray(&complex));
}

#[test]
fn tensor_to_ndarray_rejects_negative_singleton_stride_layout() {
    let tensor = Tensor::from_vec(vec![1.0_f64], &[1], &[-1], 0).unwrap();
    let err = try_tensor_to_ndarray(tensor).unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(msg) if msg.contains("negative ndarray/tenferro stride -1"))
    );
}

#[test]
fn tensor_to_ndarray_reports_layout_errors_for_overlapping_zero_copy_views() {
    let tensor = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], &[2, 2], &[1, 1], 0).unwrap();
    let err = try_tensor_to_ndarray(tensor).unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(msg) if msg.contains("ndarray layout conversion failed"))
    );
}

#[test]
fn zero_copy_helper_requires_main_memory() {
    let tensor = Tensor::<f64>::zeros(
        &[2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );
    assert!(can_zero_copy_tensor_to_ndarray(&tensor));
}

#[test]
fn ensure_main_memory_rejects_gpu_spaces() {
    let err = ensure_main_memory(LogicalMemorySpace::GpuMemory { device_id: 0 }).unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(msg) if msg.contains("CPU/main-memory tensors only"))
    );
}

#[test]
fn into_owned_data_rejects_shared_views() {
    let data = vec![1.0_f64, 2.0, 3.0, 4.0];
    let ptr = data.as_ptr();
    let tensor = unsafe {
        Tensor::from_external_parts(ptr, data.len(), &[2, 2], &[2, 1], 0, move || drop(data))
    }
    .unwrap();
    let err = into_owned_data(tensor, "shared view is not owned").unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("shared view is not owned")));
}

#[cfg(feature = "frontend")]
#[test]
fn frontend_helper_builds_public_tensor() {
    let array = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0_f64, 2.0]).unwrap();
    let tensor = crate::try_ndarray_to_frontend(array).unwrap();
    assert_eq!(tensor.dims(), &[2]);
}
