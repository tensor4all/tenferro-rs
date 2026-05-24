//! Eager tensor operations.
//!
//! Core eager tensor types are re-exported here. Extension crates provide
//! operation-specific helpers outside the core facade.

pub use crate::eager::{EagerRuntime, EagerTensor};
