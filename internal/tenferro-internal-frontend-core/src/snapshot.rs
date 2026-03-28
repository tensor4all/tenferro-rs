//! Primal-only snapshot boundary shared by dynamic tenferro frontends.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_internal_frontend_core::{snapshot, DynTensor, StructuredTensor};
//! use tenferro_tensor::{MemoryOrder, Tensor};
//!
//! let payload = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
//! let snap: snapshot::DynTensor = DynTensor::from(StructuredTensor::from(payload));
//! assert!(matches!(snap, DynTensor::F64(_)));
//! ```

pub use crate::DynTensor;
