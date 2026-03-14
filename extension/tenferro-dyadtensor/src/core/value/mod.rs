mod core;
mod scalar;
mod tensor;

pub use core::{AdMode, AdValue, NodeId};
pub use scalar::AdScalar;
pub use tensor::AdTensor;
pub(crate) use tensor::AdTensorSnapshot;

#[cfg(test)]
mod tests;
