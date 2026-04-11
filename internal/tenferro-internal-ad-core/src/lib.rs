//! Internal implementation crate. Not a stable public API.
//!
//! # Examples
//!
//! ```text
//! // This crate is wired through tenferro surface crates and is not intended
//! // to be consumed directly.
//! ```

mod core;
mod dyn_ad_tensor;
pub mod linearized;
pub mod ops;
mod registry;
mod tape;
mod tensor;
mod value;

#[doc(hidden)]
pub use core::AdValue;
pub use core::{AdMode, NodeId};
pub use dyn_ad_tensor::{
    DynAdTensor, DynAdTensorBorrowTyped, DynAdTensorMutRef, DynAdTensorRef, DynAdTensorRefTyped,
    DynAdTensorTyped,
};
pub use linearized::{CheckpointHint, LinearizableOp, LinearizedOp};
#[doc(hidden)]
pub use ops::*;
pub use registry::{register_closure_rule, register_mixed_rule, register_rule};
pub use tape::pullback;
pub use tensor::AdTensor;
#[doc(hidden)]
pub use tensor::AdTensorSnapshot;
pub use tidu::{AdResult, AutodiffError, Schema, SlotSchema, Value};
pub use value::{new_dyn_value, new_reverse_leaf, DynValue};

#[cfg(test)]
mod tests;
