mod core;
mod tensor;

pub use core::{AdMode, AdValue, NodeId};
pub use tensor::AdTensor;
pub(crate) use tensor::AdTensorSnapshot;

#[cfg(test)]
mod tests;
