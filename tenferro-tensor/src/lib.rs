//! Core tensor types plus execution backends and kernels.
//!
//! # Examples
//!
//! ```ignore
//! use tenferro_tensor::{cpu::CpuBackend, Tensor, TypedTensor};
//!
//! let mut backend = CpuBackend::new();
//! let a = Tensor::F64(TypedTensor::from_vec(vec![2], vec![1.0, 2.0]));
//! let b = Tensor::F64(TypedTensor::from_vec(vec![2], vec![3.0, 4.0]));
//! let c = tenferro_tensor::cpu::add(&a, &b);
//! assert_eq!(c.shape(), &[2]);
//! ```

pub mod backend;
pub mod buffer_pool;
pub mod config;
pub mod cpu;
pub mod error;
#[cfg(feature = "provider-inject")]
pub mod inject;
pub mod types;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "rocm")]
pub mod rocm;

pub use backend::*;
pub use config::*;
pub use error::*;
pub use types::*;

#[cfg(not(any(feature = "cpu-faer", feature = "cpu-blas")))]
compile_error!("enable at least one CPU backend: cpu-faer or cpu-blas");

#[cfg(all(feature = "cpu-faer", feature = "cpu-blas"))]
compile_error!("enable exactly one CPU backend: cpu-faer or cpu-blas");

#[cfg(all(feature = "provider-inject", not(feature = "cpu-blas")))]
compile_error!("provider-inject requires cpu-blas");

#[cfg(feature = "provider-src")]
extern crate blas_src as _;
#[cfg(feature = "provider-inject")]
extern crate cblas_inject as _;
#[cfg(feature = "provider-src")]
extern crate cblas_src as _;
#[cfg(feature = "provider-src")]
extern crate lapack_src as _;

#[cfg(test)]
mod tests;
