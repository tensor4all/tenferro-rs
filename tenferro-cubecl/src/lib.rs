//! CubeCL kernels and launch helpers for tenferro.
//!
//! This crate owns GPU kernel definitions but does not own tenferro tensor
//! values, device placement, or backend dispatch.
//!
//! # Examples
//!
//! ```
//! use tenferro_cubecl::reduce::{ReduceOp, ReduceStrategy};
//!
//! let _op = ReduceOp::Sum;
//! let _strategy = ReduceStrategy::Auto;
//! ```

pub mod error;
pub mod reduce;

pub use error::{CubeclKernelError, Result};
