use tenferro_capi::{
    tfe_status_t, TfeTensorF64, TFE_INTERNAL_ERROR, TFE_INVALID_ARGUMENT, TFE_SHAPE_MISMATCH,
    TFE_SUCCESS,
};
use tenferro_device::Error as DeviceError;

pub(crate) unsafe fn finalize_ptr(
    result: std::thread::Result<std::result::Result<*mut TfeTensorF64, tfe_status_t>>,
    status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
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
            std::ptr::null_mut()
        }
        Err(_) => {
            if !status.is_null() {
                *status = TFE_INTERNAL_ERROR;
            }
            std::ptr::null_mut()
        }
    }
}

pub(crate) unsafe fn finalize_void(
    result: std::thread::Result<std::result::Result<(), tfe_status_t>>,
    status: *mut tfe_status_t,
) {
    match result {
        Ok(Ok(())) => {
            if !status.is_null() {
                *status = TFE_SUCCESS;
            }
        }
        Ok(Err(code)) => {
            if !status.is_null() {
                *status = code;
            }
        }
        Err(_) => {
            if !status.is_null() {
                *status = TFE_INTERNAL_ERROR;
            }
        }
    }
}

pub(crate) fn map_device_error(err: &DeviceError) -> tfe_status_t {
    match err {
        DeviceError::ShapeMismatch { .. } | DeviceError::RankMismatch { .. } => TFE_SHAPE_MISMATCH,
        DeviceError::InvalidArgument(_)
        | DeviceError::StrideError(_)
        | DeviceError::CrossMemorySpaceOperation { .. } => TFE_INVALID_ARGUMENT,
        DeviceError::DeviceError(_) | DeviceError::NoCompatibleComputeDevice { .. } => {
            TFE_INTERNAL_ERROR
        }
    }
}
