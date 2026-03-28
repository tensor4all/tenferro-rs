//! Internal implementation crate. Not a stable public API.
//!
//! # Examples
//!
//! ```text
//! // This crate is wired through tenferro surface crates and is not intended
//! // to be consumed directly.
//! ```

#![allow(clippy::multiple_bound_locations)]

pub use tenferro_internal_ad_core::AdTensor;
pub use tenferro_internal_error::{Error, Result};
#[doc(hidden)]
pub use tenferro_internal_frontend_core::DynTensorTyped;

#[doc(hidden)]
pub mod runtime {
    pub mod contracts {
        pub use tenferro_internal_runtime::contracts::*;
    }

    pub mod dispatch {
        pub use tenferro_internal_runtime::dispatch::*;
    }
}

#[doc(hidden)]
pub mod structured {
    pub use tenferro_internal_frontend_core::StructuredTensor;
}

mod ops;

pub mod eager {
    pub use crate::ops::linalg::ad::eager::*;
}

pub mod results {
    pub use crate::ops::linalg::results::*;
}

pub use ops::linalg::ad::*;
pub use ops::linalg::results::*;
