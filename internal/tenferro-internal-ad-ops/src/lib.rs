//! Internal implementation crate. Not a stable public API.
//!
//! # Examples
//!
//! ```text
//! // This crate is wired through tenferro surface crates and is not intended
//! // to be consumed directly.
//! ```

mod core;
pub(crate) mod ops;
mod runtime;
mod structured;
mod tape;

pub(crate) use core::{DynTensor, DynTensorTyped, NodeId};
pub use tenferro_internal_ad_core::AdMode;
pub use tenferro_internal_error::{Error, Result};

pub(crate) use tenferro_internal_frontend_core::StructuredTensor;

pub mod ad {
    pub use crate::ops::ad::{
        acos_dyn, acosh_dyn, add_dyn, asin_dyn, asinh_dyn, atan2_dyn, atan_dyn, atanh_dyn, cos_dyn,
        cosh_dyn, einsum_dyn, exp_dyn, expm1_dyn, hypot_dyn, log1p_dyn, log_dyn, mean_dyn,
        normalize_cotangent_payload, pow_dyn, sin_dyn, sinh_dyn, sqrt_dyn, std_dyn, sum_dyn,
        tanh_dyn, var_dyn,
    };
}

#[doc(hidden)]
pub mod __typed_ad {
    pub use crate::ops::ad::*;
}

#[doc(hidden)]
pub mod __typed_einsum {
    pub mod ad {
        pub use crate::ops::einsum::ad::*;
    }

    pub use crate::ops::einsum::*;
}

#[doc(hidden)]
pub mod __typed_reduction {
    pub mod ad {
        pub use crate::ops::reduction::ad::*;
    }

    pub use crate::ops::reduction::*;
}

#[doc(hidden)]
pub mod __typed_scalar {
    pub mod ad {
        pub use crate::ops::scalar::ad::*;
    }

    pub mod primal {
        pub use crate::ops::scalar::primal::*;
    }

    pub use crate::ops::scalar::*;
}
