//! Core tensor types, views, backend traits, and backend-independent contracts.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_tensor::{Tensor, TypedTensor};
//!
//! let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]));
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
    BackendSessionHost, ElementwiseFusionInst, ElementwiseFusionOp, ElementwiseFusionPlan,
    SessionCachedDot, TensorAnalytic, TensorBackend, TensorBackendOps, TensorBuffer,
    TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing,
    TensorReduction, TensorStructural, TensorViewCanonicalization,
};
pub use cache::{CacheStats, RuntimeCacheControl};
pub use config::*;
pub use error::*;
pub use types::*;

#[cfg(test)]
mod tests;
