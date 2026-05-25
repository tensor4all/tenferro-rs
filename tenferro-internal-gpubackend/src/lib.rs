//! CubeCL kernels and launch helpers for tenferro.
//!
//! This crate owns GPU kernel definitions but does not own tenferro tensor
//! values, device placement, or backend dispatch.
//!
//! # Examples
//!
//! ```
//! use tenferro_gpubackend::reduce::{ReduceOp, ReduceStrategy};
//!
//! let _op = ReduceOp::Sum;
//! let _strategy = ReduceStrategy::Auto;
//! ```

pub mod error;
pub mod reduce;

#[doc(hidden)]
pub mod diagonal;
#[doc(hidden)]
pub mod elementwise;
mod helpers;
#[doc(hidden)]
pub mod indexing;
#[doc(hidden)]
pub mod linalg;
#[doc(hidden)]
pub mod structural;

pub use error::{CubeclKernelError, Result};
