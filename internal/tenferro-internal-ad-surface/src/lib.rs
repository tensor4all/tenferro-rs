//! Internal implementation crate. Not a stable public API.
//!
//! `tenferro` now treats `tidu::Value<DynTensor>` as the only source of truth
//! for reverse-mode metadata. This crate provides the thin dynamic tensor
//! façade and surface-level `grad`/`backward` helpers built on top of that
//! carrier.

#![allow(clippy::multiple_bound_locations)]

mod autograd_api;
pub mod core;

pub use autograd_api::{backward, grad, BackwardOptions, GradOptions};
pub use core::dynamic::{
    CholeskyExResult, EigResult, EigenResult, InvExResult, LstsqResult, LuFactorExResult,
    LuFactorResult, LuResult, QrResult, ScalarType, SlogdetResult, SolveExResult, SvdResult,
    Tensor,
};
pub use tenferro_device::{ComputeDevice, LogicalMemorySpace};
pub use tenferro_internal_ad_core::{
    AdResult, AutodiffError, CheckpointHint, LinearizableOp, LinearizedOp, NodeId, Schema,
    SlotSchema, Value,
};
pub use tenferro_internal_error::{Error, Result};
pub use tenferro_internal_frontend_core::{
    DynTensor, DynTensorTyped, ScalarValue, StructuredTensor,
};
pub use tenferro_internal_runtime::{
    set_default_runtime, with_default_runtime, with_runtime, DefaultRuntimeGuard, RuntimeContext,
};
pub use tenferro_tensor::MemoryOrder;
pub use tidu::{with_ad_policy, AdExecutionPolicy, CheckpointMode};

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
