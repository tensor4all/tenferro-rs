mod autodiff;
mod dyn_ad_tensor;
mod dyn_tensor;
mod scalar_type;
mod tensor_ops;

pub use dyn_ad_tensor::{
    EigResult, EigenResult, LstsqResult, LuResult, QrResult, SlogdetResult, SvdResult, Tensor,
};
pub use dyn_tensor::DynTensor;
#[doc(hidden)]
pub use dyn_tensor::DynTensorTyped;
pub use scalar_type::ScalarType;

#[cfg(test)]
mod tests;
