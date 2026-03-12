use std::os::raw::c_char;

use tenferro_capi::{tfe_status_t, TfeTensorF64, TFE_INVALID_ARGUMENT};
use tenferro_tensor::Tensor;

use crate::handle::handle_to_ref;

pub(crate) unsafe fn parse_subscripts<'a>(
    subscripts: *const c_char,
) -> std::result::Result<&'a str, tfe_status_t> {
    if subscripts.is_null() {
        return Err(TFE_INVALID_ARGUMENT);
    }
    std::ffi::CStr::from_ptr(subscripts)
        .to_str()
        .map_err(|_| TFE_INVALID_ARGUMENT)
}

pub(crate) unsafe fn collect_operand_handles<'a>(
    operands: *const *const TfeTensorF64,
    num_operands: usize,
) -> std::result::Result<Vec<&'a Tensor<f64>>, tfe_status_t> {
    if operands.is_null() {
        return Err(TFE_INVALID_ARGUMENT);
    }
    std::slice::from_raw_parts(operands, num_operands)
        .iter()
        .map(|&ptr| {
            if ptr.is_null() {
                Err(TFE_INVALID_ARGUMENT)
            } else {
                Ok(handle_to_ref(ptr))
            }
        })
        .collect()
}

pub(crate) unsafe fn collect_optional_tangent_handles<'a>(
    tangents: *const *const TfeTensorF64,
    num_operands: usize,
) -> std::result::Result<Vec<Option<&'a Tensor<f64>>>, tfe_status_t> {
    if tangents.is_null() {
        return Err(TFE_INVALID_ARGUMENT);
    }
    Ok(std::slice::from_raw_parts(tangents, num_operands)
        .iter()
        .map(|&ptr| {
            if ptr.is_null() {
                None
            } else {
                Some(handle_to_ref(ptr))
            }
        })
        .collect())
}
