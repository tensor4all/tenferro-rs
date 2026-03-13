mod core;
mod scalar;
mod tensor;

pub use core::{AdMode, AdValue, NodeId, TapeId};
pub use scalar::AdScalar;
pub(crate) use scalar::{map_ad_value_mixed_linear, map_ad_value_same_type_linear};
pub use tensor::AdTensor;

#[cfg(test)]
mod tests;
