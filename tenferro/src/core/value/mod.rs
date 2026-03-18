mod core;
mod tensor;

#[cfg(test)]
pub use core::AdValue;
pub use core::{AdMode, NodeId};
pub use tensor::AdTensor;
pub(crate) use tensor::AdTensorSnapshot;

#[cfg(test)]
mod tests;
