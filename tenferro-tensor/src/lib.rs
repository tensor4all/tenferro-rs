//! Core tensor types plus execution backends and kernels.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_tensor::{cpu::CpuBackend, Tensor, TypedTensor};
//!
//! let mut backend = CpuBackend::new();
//! let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]));
//! let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![3.0, 4.0]));
//! let c = tenferro_tensor::cpu::add(&a, &b).unwrap();
//! assert_eq!(c.shape(), &[2]);
//! ```

#[cfg(not(any(feature = "cpu-faer", feature = "cpu-blas")))]
compile_error!("enable at least one CPU backend: cpu-faer or cpu-blas");

#[cfg(all(feature = "provider-inject", not(feature = "cpu-blas")))]
compile_error!("provider-inject requires cpu-blas");

/// Lightweight backend-independent host tensor data model.
///
/// Execution-capable tensors and backends in this crate remain separate from
/// the host-only core model during the crate-boundary split.
pub mod core {
    pub use tenferro_tensor_core::*;
}

pub use tenferro_tensor_core::{ShapeVec, SliceSpec, StrideVec, TensorRef};

#[cfg(any(feature = "cpu-faer", feature = "cpu-blas"))]
pub mod backend;
#[cfg(any(feature = "cpu-faer", feature = "cpu-blas"))]
pub mod buffer_pool;
#[cfg(any(feature = "cpu-faer", feature = "cpu-blas"))]
pub mod cache;
#[cfg(any(feature = "cpu-faer", feature = "cpu-blas"))]
pub mod config;
#[cfg(any(feature = "cpu-faer", feature = "cpu-blas"))]
pub mod cpu;
#[cfg(any(feature = "cpu-faer", feature = "cpu-blas"))]
pub mod error;
#[cfg(feature = "provider-inject")]
pub mod inject;
#[cfg(any(feature = "cpu-faer", feature = "cpu-blas"))]
pub mod types;
#[cfg(any(feature = "cpu-faer", feature = "cpu-blas"))]
pub mod validate;

#[cfg(any(feature = "cpu-faer", feature = "cpu-blas"))]
pub use backend::{
    default_backend_session, BackendCachedDot, BackendRuntimeCache, BackendSession,
    BackendSessionHost, ElementwiseFusionInst, ElementwiseFusionOp, ElementwiseFusionPlan,
    SessionCachedDot, TensorAnalytic, TensorBackend, TensorBackendOps, TensorBuffer,
    TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing,
    TensorReduction, TensorStructural, TensorViewCanonicalization,
};
#[cfg(any(feature = "cpu-faer", feature = "cpu-blas"))]
pub use cache::{CacheStats, RuntimeCacheControl};
#[cfg(any(feature = "cpu-faer", feature = "cpu-blas"))]
pub use config::*;
#[cfg(any(feature = "cpu-faer", feature = "cpu-blas"))]
pub use error::*;
#[cfg(any(feature = "cpu-faer", feature = "cpu-blas"))]
pub use types::*;

#[cfg(feature = "provider-src")]
extern crate blas_src as _;
#[cfg(feature = "provider-inject")]
extern crate cblas_inject as _;
#[cfg(feature = "provider-src")]
extern crate cblas_src as _;
#[cfg(feature = "provider-inject")]
extern crate lapack_inject as _;
#[cfg(feature = "provider-src")]
extern crate lapack_src as _;

#[cfg(all(test, any(feature = "cpu-faer", feature = "cpu-blas")))]
mod tests;
