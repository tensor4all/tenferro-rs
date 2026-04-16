//! Core tensor types plus execution backends and kernels.
//!
//! # Examples
//!
//! ```
//! use tenferro_tensor::{Tensor, cpu::CpuBackend};
//!
//! let mut ctx = CpuBackend::new();
//! let a = Tensor::new(vec![2], vec![1.0_f64, 2.0]);
//! let b = Tensor::new(vec![2], vec![3.0_f64, 4.0]);
//! let c = a.add(&b, &mut ctx).unwrap();
//! assert_eq!(c.shape(), &[2]);
//! ```

pub mod backend;
pub mod buffer_pool;
pub mod config;
pub mod cpu;
pub mod error;
#[cfg(feature = "provider-inject")]
pub mod inject;
mod typed_linalg;
pub mod types;
pub mod validate;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "rocm")]
pub mod rocm;

pub use backend::{default_exec_session, SemiringBackend, TensorBackend, TensorExec};
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
