#![allow(clippy::multiple_bound_locations)]

//! `tenferro-dyadtensor`: AD-aware tensor interface layer on top of `tenferro-rs`.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_dyadtensor::{DynAdTensor, StructuredTensor};
//! use tenferro_tensor::{MemoryOrder, Tensor};
//!
//! let payload = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
//! let diag = StructuredTensor::from_diagonal_vector(payload, 2).unwrap();
//! let x = DynAdTensor::new_primal(diag);
//! assert_eq!(x.dims(), &[2, 2]);
//! assert!(x.is_diag());
//! ```
//!
//! Builder `.run()` execution is configured through [`set_default_runtime`].
//! Reverse-mode graphs are created through [`DynTape`] and are exposed back to
//! users through dynamic helpers on [`DynAdTensor`].
//!
//! Module map:
//!
//! - `core` defines the internal typed AD values plus dynamic wrappers.
//! - [`runtime`] owns default-runtime selection and runtime dispatch.
//! - `tape` owns reverse-mode pullback storage.
//! - `ops` is operation-first: `einsum`, `scalar`, `reduction`, and
//!   `linalg/*` each keep primal and AD wiring together.
//! - [`StructuredTensor`] and `structured` handle structured layouts.

mod core;
pub mod error;
mod ops;
pub mod policy;
pub mod runtime;
mod structured;
mod tape;
mod traits;

pub(crate) use core::DynTensorTyped;
pub use core::{
    AdMode, DynAdEigResult, DynAdEigenResult, DynAdLstsqResult, DynAdLuResult, DynAdQrResult,
    DynAdSlogdetResult, DynAdSvdResult, DynAdTensor, DynScalar, DynTape, NodeId, ScalarType,
};
pub(crate) use core::{AdScalar, AdTensor, AdValue, DynTensor};
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
