#![allow(clippy::multiple_bound_locations)]

//! `tenferro`: AD-aware tensor interface layer on top of `tenferro-rs`.
//!
//! # Examples
//!
//! ```rust
//! use tenferro::Tensor;
//!
//! let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
//! let y = x.exp().unwrap().sum().unwrap();
//! assert_eq!(y.dims(), &[] as &[usize]);
//! ```
//!
//! Builder `.run()` execution is configured through [`set_default_runtime`].
//! The primary public frontend is [`Tensor`], backed by `tidu`'s
//! `Value<DynTensor>` carrier. Custom downstream operations should use
//! [`LinearizableOp`] and [`LinearizedOp`] directly.

mod core;
pub mod error;
pub mod runtime;
mod scalar_value;
pub mod snapshot;

pub use core::{
    CholeskyExResult, EigResult, EigenResult, InvExResult, LstsqResult, LuFactorExResult,
    LuFactorResult, LuResult, QrResult, ScalarType, SlogdetResult, SolveExResult, SvdResult,
    Tensor,
};
pub use error::{Error, Result};
pub use runtime::{set_default_runtime, with_default_runtime, DefaultRuntimeGuard, RuntimeContext};
pub use scalar_value::ScalarValue;
pub use tenferro_device::{ComputeDevice, LogicalMemorySpace};
pub use tenferro_internal_ad_surface::{
    backward, grad, with_ad_policy, AdExecutionPolicy, BackwardOptions, CheckpointHint,
    CheckpointMode, GradOptions, LinearizableOp, LinearizedOp, NormKind, Schema, SlotSchema,
    SvdOptions, Value,
};
pub use tenferro_tensor::MemoryOrder;
