//! Internal implementation crate. Not a stable public API.
//!
//! # Examples
//!
//! ```text
//! // This crate is wired through tenferro surface crates and is not intended
//! // to be consumed directly.
//! ```

#![allow(clippy::multiple_bound_locations)]

mod autograd_api;
pub mod core;
pub mod forward_ad;
mod ops;

pub use autograd_api::{backward, grad, hvp, BackwardOptions, GradOptions, HvpOptions, HvpResult};
pub use core::dynamic::{
    CholeskyExResult, EigResult, EigenResult, InvExResult, LstsqResult, LuFactorExResult,
    LuFactorResult, LuResult, QrResult, ScalarType, SlogdetResult, SolveExResult, SvdResult,
    Tensor, TensorScalarDowncast, TypedTensorRef,
};
pub use tenferro_device::{ComputeDevice, LogicalMemorySpace};
pub use tenferro_internal_ad_core::AdMode;
pub use tenferro_internal_error::{Error, Result};
pub use tenferro_internal_frontend_core::{
    DynTensor, DynTensorTyped, ScalarValue, StructuredTensor,
};
pub use tenferro_internal_runtime::{
    set_default_runtime, with_default_runtime, with_runtime, DefaultRuntimeGuard, RuntimeContext,
};
pub use tenferro_tensor::MemoryOrder;

pub mod snapshot {
    pub use tenferro_internal_frontend_core::snapshot::*;
}

pub mod runtime {
    pub mod contracts {
        pub use tenferro_internal_runtime::contracts::*;
    }

    pub mod dispatch {
        pub use tenferro_internal_runtime::dispatch::*;
    }
}

pub mod structured {
    pub use tenferro_internal_frontend_core::StructuredTensor;
    pub use tenferro_tensor::structured_tensor::canonicalize_axis_classes;
}

pub mod tape {
    pub use tenferro_internal_ad_core::{
        register_closure_rule, register_mixed_rule, register_rule,
    };
}

#[doc(hidden)]
pub mod __typed_linalg_primal {
    pub use crate::ops::*;
}
