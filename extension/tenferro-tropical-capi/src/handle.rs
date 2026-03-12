use tenferro_capi::TfeTensorF64;
use tenferro_tensor::Tensor;

pub(crate) fn tensor_to_handle(tensor: Tensor<f64>) -> *mut TfeTensorF64 {
    Box::into_raw(Box::new(tensor)) as *mut TfeTensorF64
}

pub(crate) unsafe fn handle_to_ref<'a>(handle: *const TfeTensorF64) -> &'a Tensor<f64> {
    &*(handle as *const Tensor<f64>)
}
