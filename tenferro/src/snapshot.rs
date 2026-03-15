//! Primal-only snapshot boundary for export, storage, and FFI.
//!
//! [`crate::Tensor`] remains the compute object in the public frontend. This
//! module exposes the AD-free snapshot type returned by
//! [`crate::Tensor::primal_snapshot`].
//!
//! # Examples
//!
//! ```rust
//! use tenferro::{snapshot, Tensor};
//!
//! let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
//! let snap = x.primal_snapshot();
//! assert!(matches!(snap, snapshot::DynTensor::F64(_)));
//! ```

pub use crate::core::DynTensor;
