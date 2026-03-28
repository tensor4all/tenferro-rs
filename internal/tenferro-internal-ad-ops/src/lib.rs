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
mod runtime;
mod structured;
mod tape;

pub use core::{DynTensor, DynTensorTyped, NodeId};
pub use ops::*;
pub use tenferro_internal_ad_core::{AdMode, AdTensor};
pub use tenferro_internal_error::{Error, Result};

#[doc(hidden)]
pub use tenferro_internal_frontend_core::StructuredTensor;

pub mod ad {
    pub use crate::ops::ad::*;
}

pub mod einsum {
    pub mod ad {
        pub use crate::ops::einsum::ad::*;
    }

    pub use crate::ops::einsum::*;
}

pub mod reduction {
    pub mod ad {
        pub use crate::ops::reduction::ad::*;
    }

    pub use crate::ops::reduction::*;
}

pub mod scalar {
    pub mod ad {
        pub use crate::ops::scalar::ad::*;
    }

    pub mod primal {
        pub use crate::ops::scalar::primal::*;
    }

    pub use crate::ops::scalar::*;
}
