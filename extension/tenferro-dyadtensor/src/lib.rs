#![allow(clippy::multiple_bound_locations)]

//! `tenferro-dyadtensor`: AD-aware tensor interface layer on top of `tenferro-rs`.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_dyadtensor::{StructuredTensor, Tensor};
//! use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};
//!
//! let payload =
//!     DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
//! let diag = StructuredTensor::from_diagonal_vector(payload, 2).unwrap();
//! let x = Tensor::from_structured(diag);
//! assert_eq!(x.dims(), &[2, 2]);
//! assert!(x.is_diag());
//! ```
//!
//! Builder `.run()` execution is configured through [`set_default_runtime`].
//! The primary public frontend is [`Tensor`], with rank-0 tensor scalar
//! semantics and eager tensor methods.
//!
//! Module map:
//!
//! - `core` defines the internal typed AD values plus dynamic wrappers.
//! - [`runtime`] owns default-runtime selection and runtime dispatch.
//! - `tape` owns reverse-mode pullback storage.
//! - `ops` is operation-first: `einsum`, `scalar`, `reduction`, and
//!   `linalg/*` each keep primal and AD wiring together.
//! - [`StructuredTensor`] and `structured` handle structured layouts.

mod autograd_api;
mod core;
pub mod error;
pub mod forward_ad;
mod ops;
pub mod policy;
pub mod runtime;
mod structured;
mod tape;
mod traits;

pub use autograd_api::{backward, grad, BackwardOptions, GradOptions};
pub(crate) use core::DynTensorTyped;
pub(crate) use core::{AdTensor, AdValue, DynTensor};
pub use core::{
    EigResult, EigenResult, LstsqResult, LuResult, QrResult, ScalarType, SlogdetResult, SvdResult,
    Tensor,
};
pub use error::{Error, Result};
pub use ops::chainrules_api;
pub use policy::DiffPolicy;
pub use runtime::{set_default_runtime, with_default_runtime, DefaultRuntimeGuard, RuntimeContext};
pub use structured::meta::{
    plan_axis_classes_for_subscripts, AxisClassMergePlan, AxisClassPlanError, OperandAxisClassPlan,
    OperandAxisClasses,
};
pub use structured::StructuredTensor;
pub use traits::{AllowedPairs, FactorizeOptions, FactorizeResult, IndexLike, TensorKernel};
