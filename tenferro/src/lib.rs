#![allow(clippy::multiple_bound_locations)]

//! `tenferro`: AD-aware tensor interface layer on top of `tenferro-rs`.
//!
//! The public surface includes reverse-mode helpers plus a narrow JVP
//! transform, [`jvp`], which returns [`JvpResult`] with primal outputs and
//! optional output tangents for the currently wired operations.
//!
//! # Examples
//!
//! ```rust
//! use tenferro::{jvp, Tensor};
//!
//! let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
//! let primals = [x];
//! let tangents = [Some(Tensor::from_slice(&[1.0_f64, 0.0], &[2]).unwrap())];
//!
//! let result = jvp(
//!     |inputs| Ok(vec![inputs[0].add(&inputs[0]).unwrap().exp().unwrap().sum().unwrap()]),
//!     &primals,
//!     &tangents,
//! )
//! .unwrap();
//!
//! assert_eq!(result.outputs.len(), 1);
//! assert_eq!(result.output_tangents.len(), 1);
//! ```
//!
//! Builder `.run()` execution is configured through [`set_default_runtime`].
//! The primary public frontend is [`Tensor`], backed by `tidu`'s
//! `Value<DynTensor>` carrier. Custom downstream operations should use
//! [`LinearizableOp`] and [`LinearizedOp`] directly; `jvp` is not a public
//! dual-builder API and does not imply higher-order forward-mode or HVP
//! support.
//!
//! Runtime-dispatched tensor operations such as [`Tensor::einsum`],
//! [`Tensor::solve`], [`Tensor::solve_triangular`], [`Tensor::det`],
//! [`Tensor::inv`], [`Tensor::slogdet`], [`Tensor::cholesky`],
//! [`Tensor::lstsq`], [`Tensor::lu`], [`Tensor::norm`],
//! [`Tensor::vector_norm`], [`Tensor::matrix_norm`], [`Tensor::qr`],
//! [`Tensor::svd`], [`Tensor::eig`], [`Tensor::eigh`], [`Tensor::pinv`],
//! and [`Tensor::matrix_exp`] require an installed runtime via
//! [`set_default_runtime`] or [`runtime::with_runtime`].

pub mod v2;

mod core;
pub mod error;
#[path = "jvp.rs"]
mod jvp_api;
pub mod runtime;
mod scalar_value;
pub mod snapshot;

pub use core::{
    CholeskyExResult, EigResult, EighResult, InvExResult, LstsqResult, LuFactorExResult,
    LuFactorResult, LuPivot, LuResult, QrResult, ScalarType, SlogdetResult, SolveExResult,
    SvdResult, Tensor,
};
pub use error::{Error, Result};
pub use jvp_api::{jvp, JvpResult};
pub use runtime::{set_default_runtime, with_default_runtime, DefaultRuntimeGuard, RuntimeContext};
pub use scalar_value::ScalarValue;
pub use tenferro_device::{ComputeDevice, LogicalMemorySpace};
pub use tenferro_internal_ad_surface::{
    backward, grad, with_ad_policy, AdExecutionPolicy, BackwardOptions, CheckpointHint,
    CheckpointMode, GradOptions, LinearizableOp, LinearizedOp, MatrixNormOrd, NormKind, Schema,
    SlotSchema, SvdOptions, Value, VectorNormOrd,
};
pub use tenferro_tensor::MemoryOrder;
