//! The #2552 call grammar for core tensor operations.
//!
//! [`registry`] declares every core operation once; it generates [`CoreOps`]
//! (the concrete layer) and describes the traced (`TracedTensor`) and eager
//! (`EagerSession`) layers, whose methods stay inherent.

pub(crate) mod axes;
mod body;
pub(crate) mod concrete;
mod registry;
mod repr;

pub use registry::CoreOps;
pub use repr::{CoreRepr, SelectCondition, SelectOperand};

#[cfg(test)]
mod tests;
