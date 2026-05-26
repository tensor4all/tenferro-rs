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

#[cfg(all(feature = "cpu-faer", feature = "cpu-blas"))]
compile_error!("enable exactly one CPU backend: cpu-faer or cpu-blas");

#[cfg(all(feature = "provider-inject", not(feature = "cpu-blas")))]
compile_error!("provider-inject requires cpu-blas");

#[cfg(all(
    any(feature = "cpu-faer", feature = "cpu-blas"),
    not(all(feature = "cpu-faer", feature = "cpu-blas"))
))]
pub mod backend;
#[cfg(all(
    any(feature = "cpu-faer", feature = "cpu-blas"),
    not(all(feature = "cpu-faer", feature = "cpu-blas"))
))]
pub mod buffer_pool;
#[cfg(all(
    any(feature = "cpu-faer", feature = "cpu-blas"),
    not(all(feature = "cpu-faer", feature = "cpu-blas"))
))]
pub mod cache;
#[cfg(all(
    any(feature = "cpu-faer", feature = "cpu-blas"),
    not(all(feature = "cpu-faer", feature = "cpu-blas"))
))]
pub mod config;
#[cfg(all(
    any(feature = "cpu-faer", feature = "cpu-blas"),
    not(all(feature = "cpu-faer", feature = "cpu-blas"))
))]
pub mod cpu;
#[cfg(all(
    any(feature = "cpu-faer", feature = "cpu-blas"),
    not(all(feature = "cpu-faer", feature = "cpu-blas"))
))]
pub mod error;
#[cfg(all(
    feature = "provider-inject",
    any(feature = "cpu-faer", feature = "cpu-blas"),
    not(all(feature = "cpu-faer", feature = "cpu-blas"))
))]
pub mod inject;
#[cfg(all(
    any(feature = "cpu-faer", feature = "cpu-blas"),
    not(all(feature = "cpu-faer", feature = "cpu-blas"))
))]
mod typed_linalg;
#[cfg(all(
    any(feature = "cpu-faer", feature = "cpu-blas"),
    not(all(feature = "cpu-faer", feature = "cpu-blas"))
))]
pub mod types;
#[cfg(all(
    any(feature = "cpu-faer", feature = "cpu-blas"),
    not(all(feature = "cpu-faer", feature = "cpu-blas"))
))]
pub mod validate;

#[cfg(all(
    any(feature = "cpu-faer", feature = "cpu-blas"),
    not(all(feature = "cpu-faer", feature = "cpu-blas"))
))]
pub use backend::{
    default_exec_session, ElementwiseFusionInst, ElementwiseFusionOp, ElementwiseFusionPlan,
    TensorBackend, TensorExec,
};
#[cfg(all(
    any(feature = "cpu-faer", feature = "cpu-blas"),
    not(all(feature = "cpu-faer", feature = "cpu-blas"))
))]
pub use cache::{CacheStats, RuntimeCacheControl};
#[cfg(all(
    any(feature = "cpu-faer", feature = "cpu-blas"),
    not(all(feature = "cpu-faer", feature = "cpu-blas"))
))]
pub use config::*;
#[cfg(all(
    any(feature = "cpu-faer", feature = "cpu-blas"),
    not(all(feature = "cpu-faer", feature = "cpu-blas"))
))]
pub use error::*;
#[cfg(all(
    any(feature = "cpu-faer", feature = "cpu-blas"),
    not(all(feature = "cpu-faer", feature = "cpu-blas"))
))]
pub use types::*;

#[cfg(all(
    feature = "provider-src",
    any(feature = "cpu-faer", feature = "cpu-blas"),
    not(all(feature = "cpu-faer", feature = "cpu-blas"))
))]
extern crate blas_src as _;
#[cfg(all(
    feature = "provider-inject",
    any(feature = "cpu-faer", feature = "cpu-blas"),
    not(all(feature = "cpu-faer", feature = "cpu-blas"))
))]
extern crate cblas_inject as _;
#[cfg(all(
    feature = "provider-src",
    any(feature = "cpu-faer", feature = "cpu-blas"),
    not(all(feature = "cpu-faer", feature = "cpu-blas"))
))]
extern crate cblas_src as _;
#[cfg(all(
    feature = "provider-inject",
    any(feature = "cpu-faer", feature = "cpu-blas"),
    not(all(feature = "cpu-faer", feature = "cpu-blas"))
))]
extern crate lapack_inject as _;
#[cfg(all(
    feature = "provider-src",
    any(feature = "cpu-faer", feature = "cpu-blas"),
    not(all(feature = "cpu-faer", feature = "cpu-blas"))
))]
extern crate lapack_src as _;

#[cfg(all(
    test,
    any(feature = "cpu-faer", feature = "cpu-blas"),
    not(all(feature = "cpu-faer", feature = "cpu-blas"))
))]
mod tests;
