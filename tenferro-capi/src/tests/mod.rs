mod organization;

use std::os::raw::c_void;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use tenferro_tensor::{MemoryOrder, Tensor};

use super::*;

struct ImportFixture {
    data: Vec<f64>,
    shape: Box<[i64]>,
    strides: Box<[i64]>,
    calls: Arc<AtomicUsize>,
}

unsafe extern "C" fn import_fixture_deleter(managed: *mut DLManagedTensorVersioned) {
    let managed = unsafe { Box::from_raw(managed) };
    let fixture = unsafe { Box::from_raw(managed.manager_ctx as *mut ImportFixture) };
    fixture.calls.fetch_add(1, Ordering::SeqCst);
}

unsafe extern "C" fn boxed_i64_array_1_deleter(managed: *mut DLManagedTensorVersioned) {
    let managed = unsafe { Box::from_raw(managed) };
    if !managed.manager_ctx.is_null() {
        unsafe {
            drop(Box::from_raw(managed.manager_ctx as *mut [i64; 1]));
        }
    }
}

unsafe extern "C" fn boxed_i64_array_3_deleter(managed: *mut DLManagedTensorVersioned) {
    let managed = unsafe { Box::from_raw(managed) };
    if !managed.manager_ctx.is_null() {
        unsafe {
            drop(Box::from_raw(managed.manager_ctx as *mut [i64; 3]));
        }
    }
}

fn import_fixture_managed(
    device_type: i32,
    device_id: i32,
) -> (*mut DLManagedTensorVersioned, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut fixture = Box::new(ImportFixture {
        data: vec![1.0, 2.0, 3.0, 4.0],
        shape: vec![2, 2].into_boxed_slice(),
        strides: vec![2, 1].into_boxed_slice(),
        calls: calls.clone(),
    });
    let data_ptr = fixture.data.as_mut_ptr();
    let shape_ptr = fixture.shape.as_mut_ptr();
    let strides_ptr = fixture.strides.as_mut_ptr();
    let manager_ctx = Box::into_raw(fixture);

    let managed = Box::into_raw(Box::new(DLManagedTensorVersioned {
        version: DLPackVersion { major: 1, minor: 0 },
        manager_ctx: manager_ctx as *mut c_void,
        deleter: Some(import_fixture_deleter),
        flags: 0,
        dl_tensor: DLTensor {
            data: data_ptr as *mut c_void,
            device: DLDevice {
                device_type,
                device_id,
            },
            ndim: 2,
            dtype: DLDataType {
                code: KDLFLOAT,
                bits: 64,
                lanes: 1,
            },
            shape: shape_ptr,
            strides: strides_ptr,
            byte_offset: 0,
        },
    }));

    (managed, calls)
}

fn import_fixture_with_tensor(
    dl_tensor: DLTensor,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    manager_ctx: *mut c_void,
) -> *mut DLManagedTensorVersioned {
    Box::into_raw(Box::new(DLManagedTensorVersioned {
        version: DLPackVersion { major: 1, minor: 0 },
        manager_ctx,
        deleter,
        flags: 0,
        dl_tensor,
    }))
}

fn read_last_error_message() -> String {
    let mut len = 0usize;
    let status = unsafe { tfe_last_error_message(std::ptr::null_mut(), 0, &mut len) };
    assert_eq!(status, TFE_SUCCESS);
    assert!(len > 0);

    let mut buf = vec![0u8; len];
    let status = unsafe { tfe_last_error_message(buf.as_mut_ptr(), buf.len(), &mut len) };
    assert_eq!(status, TFE_SUCCESS);
    let nul = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8(buf[..nul].to_vec()).unwrap()
}

#[test]
fn dlpack_export_import_roundtrip_preserves_strides_and_values() {
    let base = Tensor::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let view = base.permute(&[1, 0]).unwrap();
    let handle = tensor_to_handle(view);

    let mut status = TFE_INTERNAL_ERROR;
    let dl = unsafe { tfe_tensor_f64_to_dlpack(handle, &mut status) };
    assert_eq!(status, TFE_SUCCESS);
    assert!(!dl.is_null());

    let managed = unsafe { &*dl };
    assert_eq!(managed.version.major, 1);
    assert_eq!(managed.dl_tensor.ndim, 2);
    assert_eq!(
        unsafe { std::slice::from_raw_parts(managed.dl_tensor.shape, 2) },
        &[3, 2]
    );
    assert_eq!(
        unsafe { std::slice::from_raw_parts(managed.dl_tensor.strides, 2) },
        &[2, 1]
    );
    assert_eq!(managed.dl_tensor.byte_offset, 0);

    let imported = unsafe { tfe_tensor_f64_from_dlpack(dl, &mut status) };
    assert_eq!(status, TFE_SUCCESS);
    assert!(!imported.is_null());

    let imported_tensor = unsafe { handle_to_ref(imported) };
    assert_eq!(imported_tensor.dims(), &[3, 2]);
    assert_eq!(imported_tensor.strides(), &[2, 1]);
    let row_major = imported_tensor.contiguous(MemoryOrder::RowMajor);
    assert_eq!(
        row_major.buffer().as_slice().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );

    unsafe { tfe_tensor_f64_release(imported) };
}

#[test]
fn dlpack_import_calls_producer_deleter_on_release() {
    let (managed, calls) = import_fixture_managed(KDLCPU, 0);

    let mut status = TFE_INTERNAL_ERROR;
    let handle = unsafe { tfe_tensor_f64_from_dlpack(managed, &mut status) };
    assert_eq!(status, TFE_SUCCESS);
    assert!(!handle.is_null());
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let tensor = unsafe { handle_to_ref(handle) };
    assert_eq!(tensor.dims(), &[2, 2]);
    assert_eq!(tensor.strides(), &[2, 1]);
    assert_eq!(tensor.buffer().as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

    unsafe { tfe_tensor_f64_release(handle) };
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn dlpack_import_rejects_unsupported_device_and_consumes_input() {
    let (managed, calls) = import_fixture_managed(KDLCUDA, 0);

    let mut status = TFE_INTERNAL_ERROR;
    let handle = unsafe { tfe_tensor_f64_from_dlpack(managed, &mut status) };
    assert!(handle.is_null());
    assert_eq!(status, TFE_INVALID_ARGUMENT);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn dlpack_export_rejects_null_tensor_pointer() {
    let mut status = TFE_INTERNAL_ERROR;
    let managed = unsafe { tfe_tensor_f64_to_dlpack(std::ptr::null_mut(), &mut status) };
    assert!(managed.is_null());
    assert_eq!(status, TFE_INVALID_ARGUMENT);
}

#[test]
fn dlpack_import_rejects_negative_ndim() {
    let managed = import_fixture_with_tensor(
        DLTensor {
            data: std::ptr::dangling_mut::<f64>() as *mut c_void,
            device: DLDevice {
                device_type: KDLCPU,
                device_id: 0,
            },
            ndim: -1,
            dtype: DLDataType {
                code: KDLFLOAT,
                bits: 64,
                lanes: 1,
            },
            shape: std::ptr::null_mut(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        },
        None,
        std::ptr::null_mut(),
    );

    let mut status = TFE_INTERNAL_ERROR;
    let handle = unsafe { tfe_tensor_f64_from_dlpack(managed, &mut status) };
    assert!(handle.is_null());
    assert_eq!(status, TFE_INVALID_ARGUMENT);
}

#[test]
fn dlpack_import_rejects_null_shape_and_null_data_for_non_empty_tensor() {
    let managed_missing_shape = import_fixture_with_tensor(
        DLTensor {
            data: std::ptr::dangling_mut::<f64>() as *mut c_void,
            device: DLDevice {
                device_type: KDLCPU,
                device_id: 0,
            },
            ndim: 1,
            dtype: DLDataType {
                code: KDLFLOAT,
                bits: 64,
                lanes: 1,
            },
            shape: std::ptr::null_mut(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        },
        None,
        std::ptr::null_mut(),
    );

    let mut status = TFE_INTERNAL_ERROR;
    let handle = unsafe { tfe_tensor_f64_from_dlpack(managed_missing_shape, &mut status) };
    assert!(handle.is_null());
    assert_eq!(status, TFE_INVALID_ARGUMENT);

    let mut shape = Box::new([1_i64]);
    let managed_missing_data = import_fixture_with_tensor(
        DLTensor {
            data: std::ptr::null_mut(),
            device: DLDevice {
                device_type: KDLCPU,
                device_id: 0,
            },
            ndim: 1,
            dtype: DLDataType {
                code: KDLFLOAT,
                bits: 64,
                lanes: 1,
            },
            shape: shape.as_mut_ptr(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        },
        Some(boxed_i64_array_1_deleter),
        Box::into_raw(shape) as *mut c_void,
    );

    let handle = unsafe { tfe_tensor_f64_from_dlpack(managed_missing_data, &mut status) };
    assert!(handle.is_null());
    assert_eq!(status, TFE_INVALID_ARGUMENT);
}

#[test]
fn dlpack_import_reports_axis_for_stride_overflow() {
    let mut shape = Box::new([1_i64, i64::MAX, 2_i64]);
    let managed = import_fixture_with_tensor(
        DLTensor {
            data: std::ptr::dangling_mut::<f64>() as *mut c_void,
            device: DLDevice {
                device_type: KDLCPU,
                device_id: 0,
            },
            ndim: 3,
            dtype: DLDataType {
                code: KDLFLOAT,
                bits: 64,
                lanes: 1,
            },
            shape: shape.as_mut_ptr(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        },
        Some(boxed_i64_array_3_deleter),
        Box::into_raw(shape) as *mut c_void,
    );

    let mut status = TFE_INTERNAL_ERROR;
    let handle = unsafe { tfe_tensor_f64_from_dlpack(managed, &mut status) };
    assert!(handle.is_null());
    assert_eq!(status, TFE_INVALID_ARGUMENT);

    let last_error = read_last_error_message();
    assert!(
        last_error.contains("stride overflow in row-major inference at dimension 1 (size="),
        "unexpected error: {last_error}"
    );
}

#[test]
fn dlpack_export_byte_offset_matches_tensor_offset() {
    let data = vec![99.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_vec(data, &[2, 3], &[1, 2], 1).unwrap();
    assert_eq!(tensor.offset(), 1);

    let handle = tensor_to_handle(tensor);
    let mut status = TFE_INTERNAL_ERROR;
    let dl = unsafe { tfe_tensor_f64_to_dlpack(handle, &mut status) };
    assert_eq!(status, TFE_SUCCESS);
    assert!(!dl.is_null());

    let managed = unsafe { &*dl };
    assert_eq!(managed.dl_tensor.byte_offset, 8);

    unsafe {
        if let Some(deleter) = (*dl).deleter {
            deleter(dl);
        }
    }
}

#[test]
fn dlpack_export_positive_offset_zero() {
    let tensor =
        Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    assert_eq!(tensor.offset(), 0);

    let handle = tensor_to_handle(tensor);
    let mut status = TFE_INTERNAL_ERROR;
    let dl = unsafe { tfe_tensor_f64_to_dlpack(handle, &mut status) };
    assert_eq!(status, TFE_SUCCESS);
    assert!(!dl.is_null());

    let managed = unsafe { &*dl };
    assert_eq!(managed.dl_tensor.byte_offset, 0);

    unsafe {
        if let Some(deleter) = (*dl).deleter {
            deleter(dl);
        }
    }
}
