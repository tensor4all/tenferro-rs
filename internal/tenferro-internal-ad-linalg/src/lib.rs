//! Internal implementation crate. Not a stable public API.
//!
//! # Examples
//!
//! ```text
//! // This crate is wired through tenferro surface crates and is not intended
//! // to be consumed directly.
//! ```

pub use tenferro_internal_error::{Error, Result};
#[doc(hidden)]
pub use tenferro_internal_frontend_core::DynTensorTyped;

pub mod results {
    pub use tenferro_linalg::{
        CholeskyExResult, EigResult, EigenResult, InvExResult, LstsqResult, LuFactorExResult,
        LuFactorResult, LuResult, QrResult, SlogdetResult, SolveExResult, SvdResult,
    };
}

pub use results::*;
