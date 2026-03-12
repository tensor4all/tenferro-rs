use std::ffi::CStr;
use std::os::raw::c_char;

use tenferro_prims::CpuContext;
use tenferro_tensor::Tensor;

use crate::handle::{handle_to_ref, TfeTensorF64};
use crate::status::{map_device_error, set_last_error, StatusResult, TFE_INVALID_ARGUMENT};

pub(crate) unsafe fn read_c_str<'a>(ptr: *const c_char, arg: &str) -> StatusResult<&'a str> {
    if ptr.is_null() {
        set_last_error(&format!("{arg}: C string pointer is null"));
        return Err(TFE_INVALID_ARGUMENT);
    }
    CStr::from_ptr(ptr).to_str().map_err(|_| {
        set_last_error(&format!("{arg}: string is not valid UTF-8"));
        TFE_INVALID_ARGUMENT
    })
}

pub(crate) unsafe fn read_usize_slice<'a>(
    ptr: *const usize,
    len: usize,
    arg: &str,
) -> StatusResult<&'a [usize]> {
    if ptr.is_null() && len > 0 {
        set_last_error(&format!("{arg}: pointer is null"));
        return Err(TFE_INVALID_ARGUMENT);
    }
    Ok(if len > 0 {
        std::slice::from_raw_parts(ptr, len)
    } else {
        &[]
    })
}

pub(crate) unsafe fn read_tensor_refs<'a>(
    ptrs: *const *const TfeTensorF64,
    len: usize,
    arg: &str,
) -> StatusResult<Vec<&'a Tensor<f64>>> {
    if ptrs.is_null() && len > 0 {
        set_last_error(&format!("{arg}: operand pointer array is null"));
        return Err(TFE_INVALID_ARGUMENT);
    }
    let op_ptrs = std::slice::from_raw_parts(ptrs, len);
    op_ptrs
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            if p.is_null() {
                set_last_error(&format!("{arg}: operand {i} is null"));
                Err(TFE_INVALID_ARGUMENT)
            } else {
                Ok(handle_to_ref(p))
            }
        })
        .collect()
}

pub(crate) unsafe fn read_optional_tensor_refs<'a>(
    ptrs: *const *const TfeTensorF64,
    len: usize,
    arg: &str,
) -> StatusResult<Vec<Option<&'a Tensor<f64>>>> {
    if ptrs.is_null() && len > 0 {
        set_last_error(&format!("{arg}: tensor pointer array is null"));
        return Err(TFE_INVALID_ARGUMENT);
    }
    let tensor_ptrs = std::slice::from_raw_parts(ptrs, len);
    Ok(tensor_ptrs
        .iter()
        .map(|&p| {
            if p.is_null() {
                None
            } else {
                Some(handle_to_ref(p))
            }
        })
        .collect())
}

pub(crate) fn cpu_context() -> StatusResult<CpuContext> {
    CpuContext::try_new(1).map_err(|err| map_device_error(&err))
}
