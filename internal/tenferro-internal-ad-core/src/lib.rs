//! Internal implementation crate. Not a stable public API.
//!
//! # Examples
//!
//! ```text
//! // This crate is wired through tenferro surface crates and is not intended
//! // to be consumed directly.
//! ```

mod core;
mod value;

pub use core::NodeId;
pub use tidu::{
    AdResult, AutodiffError, CheckpointHint, LinearizableOp, LinearizedOp, Schema, SlotSchema,
    Value,
};
pub use value::{new_dyn_value, new_reverse_leaf, DynValue};
