//! Internal implementation crate. Not a stable public API.
//!
//! # Examples
//!
//! ```text
//! // This crate is wired through tenferro surface crates and is not intended
//! // to be consumed directly.
//! ```

mod linearized;
mod math;
mod runtime;

pub use linearized::{
    add_dyn_values, einsum_dyn_values, exp_dyn_value, sum_dyn_value, AddOp, EinsumOp, ExpOp, SumOp,
};
pub use math::{einsum_frule, einsum_rrule, solve_triangular_rrule};
pub use tenferro_internal_error::{Error, Result};
pub use tenferro_internal_frontend_core::{DynTensor, DynTensorTyped, StructuredTensor};

pub mod ad {
    pub use crate::math::{einsum_frule, einsum_rrule, solve_triangular_rrule};
}
