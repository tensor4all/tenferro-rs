//! Internal implementation crate. Not a stable public API.
//!
//! # Examples
//!
//! ```text
//! // This crate is wired through tenferro surface crates and is not intended
//! // to be consumed directly.
//! ```

mod core;
pub mod ops;
mod registry;
mod tape;
mod tensor;

#[doc(hidden)]
pub use core::AdValue;
pub use core::{AdMode, NodeId};
#[doc(hidden)]
pub use ops::*;
pub use registry::{register_closure_rule, register_mixed_rule, register_rule};
pub use tape::pullback;
pub use tensor::AdTensor;
#[doc(hidden)]
pub use tensor::AdTensorSnapshot;

#[cfg(test)]
mod tests;
