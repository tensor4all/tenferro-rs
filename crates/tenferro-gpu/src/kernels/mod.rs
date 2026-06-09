//! Internal CubeCL kernels and launch helpers for tenferro.
//!
//! This crate owns GPU kernel definitions but does not own tenferro tensor
//! values, device placement, or backend dispatch.

pub(crate) mod error;
pub(crate) mod reduce;

pub(crate) mod diagonal;
pub(crate) mod elementwise;
mod helpers;
pub(crate) mod indexing;
pub(crate) mod structural;

pub(crate) use error::{CubeclKernelError, Result};
