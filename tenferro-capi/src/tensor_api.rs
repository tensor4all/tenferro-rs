use std::panic::{catch_unwind, AssertUnwindSafe};

use tenferro_device::LogicalMemorySpace;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::ffi_utils::read_usize_slice;
use crate::handle::{handle_take, handle_to_ref, tensor_to_handle, TfeTensorF64};
use crate::status::{
    finalize_ptr, map_device_error, panic_message, set_last_error, tfe_status_t,
    TFE_INTERNAL_ERROR, TFE_INVALID_ARGUMENT, TFE_SUCCESS,
};

/// Create a tensor from caller-provided data (copy semantics).
///
/// The data is **copied** into Rust-owned storage. The caller retains
/// ownership of the `data` pointer and may free it after this call.
/// The internal memory layout is implementation-defined.
///
/// For zero-copy tensor exchange with specific memory layouts, use
/// [`crate::tfe_tensor_f64_from_dlpack`] instead.
///
/// # Safety
///
/// - `data` must point to at least `len` valid `f64` values.
/// - `shape` must point to at least `ndim` valid `usize` values.
/// - `status` must be a valid, non-null pointer.
///
/// # Examples (C)
///
/// ```c
/// double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
/// size_t shape[] = {2, 3};
/// tfe_status_t status;
/// tfe_tensor_f64 *t = tfe_tensor_f64_from_data(data, 6, shape, 2, &status);
/// assert(status == TFE_SUCCESS);
/// tfe_tensor_f64_release(t);
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_tensor_f64_from_data(
    data: *const f64,
    len: usize,
    shape: *const usize,
    ndim: usize,
    status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if data.is_null() && len > 0 {
            set_last_error("from_data: data pointer is null");
            return Err(TFE_INVALID_ARGUMENT);
        }
        let dims = read_usize_slice(shape, ndim, "from_data shape")?.to_vec();
        let data_slice = if len > 0 {
            std::slice::from_raw_parts(data, len)
        } else {
            &[]
        };

        Tensor::from_slice(data_slice, &dims, MemoryOrder::ColumnMajor)
            .map(tensor_to_handle)
            .map_err(|e| map_device_error(&e))
    }));

    finalize_ptr(result, status)
}

/// Create a tensor filled with zeros.
///
/// # Safety
///
/// - `shape` must point to at least `ndim` valid `usize` values.
/// - `status` must be a valid, non-null pointer.
///
/// # Examples (C)
///
/// ```c
/// size_t shape[] = {3, 4};
/// tfe_status_t status;
/// tfe_tensor_f64 *t = tfe_tensor_f64_zeros(shape, 2, &status);
/// tfe_tensor_f64_release(t);
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_tensor_f64_zeros(
    shape: *const usize,
    ndim: usize,
    status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let dims = read_usize_slice(shape, ndim, "zeros shape")?.to_vec();
        let t = Tensor::<f64>::zeros(
            &dims,
            LogicalMemorySpace::MainMemory,
            MemoryOrder::ColumnMajor,
        );
        Ok(tensor_to_handle(t))
    }));

    finalize_ptr(result, status)
}

/// Deep-copy a tensor.
///
/// `Tensor::clone()` is a shallow copy (Arc refcount increment).
/// This C API function produces a new tensor with its own independent data buffer.
///
/// # Safety
///
/// - `tensor` must be a valid pointer returned by a `tfe_tensor_f64_*`
///   creation function that has not yet been released.
/// - `status` must be a valid, non-null pointer.
///
/// # Examples (C)
///
/// ```c
/// tfe_tensor_f64 *copy = tfe_tensor_f64_clone(original, &status);
/// tfe_tensor_f64_release(copy);
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_tensor_f64_clone(
    tensor: *const TfeTensorF64,
    status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if tensor.is_null() {
            set_last_error("clone: tensor pointer is null");
            return Err(TFE_INVALID_ARGUMENT);
        }
        let src = handle_to_ref(tensor);
        let materialized = src.contiguous(MemoryOrder::ColumnMajor);
        let src_data = materialized.buffer().as_slice().ok_or_else(|| {
            set_last_error("clone: tensor buffer is not contiguous host memory");
            TFE_INTERNAL_ERROR
        })?;
        let copy = Tensor::from_slice(src_data, materialized.dims(), MemoryOrder::ColumnMajor)
            .map_err(|e| map_device_error(&e))?;
        Ok(tensor_to_handle(copy))
    }));

    finalize_ptr(result, status)
}

/// Release (free) a tensor.
///
/// After this call, `tensor` is invalid and must not be used.
/// Passing a null pointer is a no-op.
///
/// For tensors imported via DLPack, this calls the DLPack deleter
/// to notify the external owner that the data is no longer needed.
///
/// # Safety
///
/// `tensor` must be null or a valid pointer returned by a
/// `tfe_tensor_f64_*` creation function that has not yet been released.
#[no_mangle]
pub unsafe extern "C" fn tfe_tensor_f64_release(tensor: *mut TfeTensorF64) {
    if tensor.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        drop(handle_take(tensor));
    }));
}

/// Return the number of dimensions (rank) of the tensor.
///
/// Returns 0 if `tensor` is null (and sets `status` to `TFE_INVALID_ARGUMENT`).
///
/// # Safety
///
/// - `tensor` must be a valid tensor pointer or null.
/// - `status` must be a valid, non-null pointer.
#[no_mangle]
pub unsafe extern "C" fn tfe_tensor_f64_ndim(
    tensor: *const TfeTensorF64,
    status: *mut tfe_status_t,
) -> usize {
    if tensor.is_null() {
        if !status.is_null() {
            *status = TFE_INVALID_ARGUMENT;
        }
        return 0;
    }
    let result = catch_unwind(AssertUnwindSafe(|| handle_to_ref(tensor).ndim()));
    match result {
        Ok(n) => {
            if !status.is_null() {
                *status = TFE_SUCCESS;
            }
            n
        }
        Err(panic) => {
            set_last_error(&panic_message(&*panic));
            if !status.is_null() {
                *status = TFE_INTERNAL_ERROR;
            }
            0
        }
    }
}

/// Write the shape of the tensor into the caller-provided buffer.
///
/// The caller must allocate `out_shape` with at least
/// `tfe_tensor_f64_ndim(tensor)` elements.
///
/// # Safety
///
/// - `tensor` must be a valid, non-null tensor pointer.
/// - `out_shape` must point to a buffer with at least
///   `tfe_tensor_f64_ndim(tensor)` elements.
/// - `status` must be a valid, non-null pointer.
#[no_mangle]
pub unsafe extern "C" fn tfe_tensor_f64_shape(
    tensor: *const TfeTensorF64,
    out_shape: *mut usize,
    status: *mut tfe_status_t,
) {
    if tensor.is_null() || out_shape.is_null() {
        if !status.is_null() {
            *status = TFE_INVALID_ARGUMENT;
        }
        return;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let t = handle_to_ref(tensor);
        let dims = t.dims();
        std::ptr::copy_nonoverlapping(dims.as_ptr(), out_shape, dims.len());
    }));
    match result {
        Ok(()) => {
            if !status.is_null() {
                *status = TFE_SUCCESS;
            }
        }
        Err(panic) => {
            set_last_error(&panic_message(&*panic));
            if !status.is_null() {
                *status = TFE_INTERNAL_ERROR;
            }
        }
    }
}

/// Return the total number of elements in the tensor.
///
/// Returns 0 if `tensor` is null (and sets `status` to `TFE_INVALID_ARGUMENT`).
///
/// # Safety
///
/// - `tensor` must be a valid tensor pointer or null.
/// - `status` must be a valid, non-null pointer.
#[no_mangle]
pub unsafe extern "C" fn tfe_tensor_f64_len(
    tensor: *const TfeTensorF64,
    status: *mut tfe_status_t,
) -> usize {
    if tensor.is_null() {
        if !status.is_null() {
            *status = TFE_INVALID_ARGUMENT;
        }
        return 0;
    }
    let result = catch_unwind(AssertUnwindSafe(|| handle_to_ref(tensor).len()));
    match result {
        Ok(n) => {
            if !status.is_null() {
                *status = TFE_SUCCESS;
            }
            n
        }
        Err(panic) => {
            set_last_error(&panic_message(&*panic));
            if !status.is_null() {
                *status = TFE_INTERNAL_ERROR;
            }
            0
        }
    }
}

/// Return a pointer to the tensor's raw data buffer.
///
/// The pointer is valid until `tfe_tensor_f64_release` is called on
/// the tensor. Returns null if `tensor` is null.
///
/// # Safety
///
/// - `tensor` must be a valid tensor pointer or null.
/// - `status` must be a valid, non-null pointer.
/// - The returned pointer must not be used after `tfe_tensor_f64_release(tensor)`.
#[no_mangle]
pub unsafe extern "C" fn tfe_tensor_f64_data(
    tensor: *const TfeTensorF64,
    status: *mut tfe_status_t,
) -> *const f64 {
    if tensor.is_null() {
        if !status.is_null() {
            *status = TFE_INVALID_ARGUMENT;
        }
        return std::ptr::null();
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let t = handle_to_ref(tensor);
        let slice = t.buffer().as_slice().ok_or_else(|| {
            set_last_error("data: tensor buffer is not contiguous host memory");
            TFE_INTERNAL_ERROR
        })?;
        Ok(slice.as_ptr().add(t.offset() as usize))
    }));
    match result {
        Ok(Ok(ptr)) => {
            if !status.is_null() {
                *status = TFE_SUCCESS;
            }
            ptr
        }
        Ok(Err(code)) => {
            if !status.is_null() {
                *status = code;
            }
            std::ptr::null()
        }
        Err(panic) => {
            set_last_error(&panic_message(&*panic));
            if !status.is_null() {
                *status = TFE_INTERNAL_ERROR;
            }
            std::ptr::null()
        }
    }
}
