//! The #2552 call grammar for einsum and tensordot: `op`, `op_with`, `op_into`.
//!
//! [`registry`] declares every operation once; the traits it generates
//! ([`EinsumOps`], [`TensordotOps`], [`TracedEinsumOps`],
//! [`TracedTensordotOps`], `EagerEinsumOps`) forward to the hand-written
//! bodies in [`concrete`], [`traced`] and `eager`. [`plan`] gives
//! `ConcreteEinsumPlan` the same grammar.

pub(crate) mod concrete;
#[cfg(feature = "autodiff")]
pub(crate) mod eager;
mod options;
mod plan;
mod registry;
pub(crate) mod traced;

#[cfg(feature = "autodiff")]
pub use registry::EagerEinsumOps;
pub use registry::{EinsumOps, TensordotOps, TracedEinsumOps, TracedTensordotOps};

pub use options::{ConjMask, EinsumInputs, EinsumOptions, EinsumSpec, Spec};

#[cfg(test)]
mod tests;
