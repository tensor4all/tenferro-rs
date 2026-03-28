#![allow(clippy::multiple_bound_locations)]

//! `tenferro`: AD-aware tensor interface layer on top of `tenferro-rs`.
//!
//! # Examples
//!
//! ```rust
//! use tenferro::Tensor;
//!
//! let payload = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
//! let x = Tensor::diag(&payload).unwrap();
//! assert_eq!(x.dims(), &[2, 2]);
//! assert!(x.is_diag());
//! ```
//!
//! Builder `.run()` execution is configured through [`set_default_runtime`].
//! The primary public frontend is [`Tensor`], with rank-0 tensor scalar
//! semantics and direct tensor methods.
//!
//! Module map:
//!
//! - `core` defines the internal typed AD values plus dynamic wrappers.
//! - [`runtime`] owns default-runtime selection and runtime dispatch.
//! - `tape` owns reverse-mode pullback storage.
//! - `ops` is operation-first: `einsum`, `scalar`, `reduction`, and
//!   `linalg/*` each keep primal and AD wiring together.
//! - structured layouts remain available through frontend methods such as
//!   [`Tensor::diag`], [`Tensor::diag_embed`], and
//!   [`Tensor::with_axis_classes`].

mod autograd_api;
mod core;
pub mod error;
pub mod forward_ad;
mod ops;
pub mod runtime;
mod scalar_value;
pub mod snapshot;
mod structured;
mod tape;

pub use autograd_api::{backward, grad, hvp, BackwardOptions, GradOptions, HvpOptions, HvpResult};
#[cfg(test)]
pub(crate) use core::AdValue;
pub(crate) use core::DynTensorTyped;
pub use core::{
    AdMode, CholeskyExResult, EigResult, EigenResult, InvExResult, LstsqResult, LuFactorExResult,
    LuFactorResult, LuResult, QrResult, ScalarType, SlogdetResult, SolveExResult, SvdResult,
    Tensor, TensorScalarDowncast,
};
pub(crate) use core::{AdTensor, DynTensor};
pub use error::{Error, Result};
pub use runtime::{set_default_runtime, with_default_runtime, DefaultRuntimeGuard, RuntimeContext};
pub use scalar_value::ScalarValue;
pub use tenferro_device::{ComputeDevice, LogicalMemorySpace};
pub use tenferro_tensor::MemoryOrder;
