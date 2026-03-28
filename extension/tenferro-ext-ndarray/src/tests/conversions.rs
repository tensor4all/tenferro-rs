use ndarray::{Array2, ArrayD, IxDyn, ShapeBuilder};
use tenferro_device::{Error, LogicalMemorySpace};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{
    ensure_main_memory, into_owned_data, ndarray_to_tensor, shape_error, tensor_to_ndarray,
    try_ndarray_to_tensor, try_tensor_to_ndarray,
};

#[test]
fn owned_ndarray_to_tensor_normalizes_to_column_major_internal_layout() {
    let array = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let tensor = try_ndarray_to_tensor(array).unwrap();
    assert_eq!(tensor.dims(), &[2, 2]);
    assert_eq!(tensor.strides(), &[1, 2]);
    assert_eq!(tensor.buffer().as_slice().unwrap(), &[1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn generic_owned_ndarray_input_normalizes_to_column_major_internal_layout() {
    let array = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let tensor = try_ndarray_to_tensor(array).unwrap();
    assert_eq!(tensor.strides(), &[1, 2]);
    assert_eq!(tensor.buffer().as_slice().unwrap(), &[1.0, 3.0, 2.0, 4.0]);
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
    assert!(tensor.is_col_major_contiguous());
    let roundtrip = tensor_to_ndarray(tensor);
    assert_eq!(roundtrip.shape(), &[2, 2]);
    assert_eq!(roundtrip, view.into_owned().into_dyn());
}

#[test]
fn tensor_to_ndarray_materializes_row_major_output() {
    let tensor = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], &[2, 2], &[2, 1], 0).unwrap();
    let array = try_tensor_to_ndarray(tensor).unwrap();
    assert_eq!(array.shape(), &[2, 2]);
    assert!(array.is_standard_layout());
    assert_eq!(
        array.iter().copied().collect::<Vec<_>>(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
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
fn tensor_to_ndarray_materializes_permuted_views_as_row_major_owned_arrays() {
    let tensor = Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap()
        .narrow(0, 1, 1)
        .unwrap();

    let array = tensor_to_ndarray(tensor);
    assert_eq!(array.shape(), &[1, 2]);
    assert_eq!(array.iter().copied().collect::<Vec<_>>(), vec![2.0, 4.0]);
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
