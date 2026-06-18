//! Core tensor types, views, backend traits, and backend-independent contracts.
//!
//! # Owned Tensors And Views
//!
//! [`TypedTensor<T>`](TypedTensor) and the dtype-erased [`Tensor`] enum are
//! owned tensor values. They are the right representation when a result is
//! materialized as compact column-major storage.
//!
//! [`TypedTensorView`] is a borrowed typed view over an existing tensor buffer.
//! It carries logical shape, arbitrary strides, and an offset, so metadata-only
//! layout changes such as transposes, slices, and broadcasts can be represented
//! without copying. A view can be materialized explicitly with
//! [`TypedTensorView::to_contiguous`] when a compact owned tensor is required.
//!
//! [`TensorRead`] is the dtype-erased borrowed input type used by eager kernels
//! and backend dispatch. It can borrow either an owned [`Tensor`] or a
//! [`TensorView`] with arbitrary strides. Prefer `TensorRead` for read-only
//! operation inputs so callers are not forced to materialize layout-only views.
//!
//! [`TensorOwnedView`] and [`TensorValue`] are the owned lazy-value forms. Use
//! them when an API must store a view result beyond the lifetime of a borrowed
//! input, then expose a short-lived `TensorRead` at kernel-dispatch time.
//!
//! Use [`Tensor::as_slice`] or [`TypedTensorView::as_slice`] only when compact
//! contiguous storage is part of the API contract. Use shape/stride-aware kernel
//! paths or `TensorRead` otherwise.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_tensor::{Tensor, TypedTensor};
//!
//! let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap());
//! assert_eq!(a.shape(), &[2]);
//! ```

/// Lightweight backend-independent host tensor data model.
///
/// Execution-capable tensors and backends in this crate remain separate from
/// the host-only core model during the crate-boundary split.
pub mod core {
    pub use tenferro_tensor_core::*;
}

pub use tenferro_tensor_core::{ShapeVec, SliceSpec, StrideVec, TensorRef};

pub mod backend;
pub mod cache;
pub mod config;
pub mod error;
pub mod types;
pub mod validate;

pub use backend::{
    default_backend_session, BackendCachedDot, BackendRuntimeCache, BackendSession,
    BackendSessionHost, SessionCachedDot, TensorAnalytic, TensorBackend, TensorBackendOps,
    TensorBuffer, TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing,
    TensorReduction, TensorStructural, TensorViewCanonicalization,
};
pub use cache::{CacheStats, RuntimeCacheControl};
pub use config::*;
pub use error::*;
pub use types::*;

#[cfg(test)]
mod tests;
