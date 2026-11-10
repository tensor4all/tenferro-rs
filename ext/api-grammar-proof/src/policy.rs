//! Shared execution-constraint type used by every prototype.
//!
//! Prototype 4 placed the policy next to its error in `tenferro-tensor`, the
//! lowest crate every operation family depends on; this module re-exports it so
//! all prototypes use the one type.

pub use tenferro_tensor::MaterializationPolicy;
