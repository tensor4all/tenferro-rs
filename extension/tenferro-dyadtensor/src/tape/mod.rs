mod registry;
mod scalar_pullback;
mod tensor_pullback;

pub(crate) use registry::{
    register_bridge_rule, register_rule, register_scalar_bridge_rule, register_scalar_mixed_rule,
    register_scalar_rule,
};
#[allow(unused_imports)]
pub(crate) use scalar_pullback::{pullback_scalar, pullback_wrt_scalars};
pub(crate) use tensor_pullback::{pullback, pullback_wrt_mixed};

#[cfg(test)]
mod tests;
