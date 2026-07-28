//! Internal CubeCL kernels and launch helpers for tenferro.
//!
//! This crate owns GPU kernel definitions but does not own tenferro tensor
//! values, device placement, or backend dispatch.

#[cfg(feature = "cuda")]
pub(crate) mod error;
#[cfg(feature = "cuda")]
pub(crate) mod reduce;

#[cfg(feature = "cuda")]
pub(crate) mod diagonal;
#[cfg(feature = "cuda")]
pub(crate) mod elementwise;
mod helpers;
#[cfg(feature = "cuda")]
pub(crate) mod indexing;
pub(crate) mod structural;

#[cfg(feature = "cuda")]
pub(crate) use error::{CubeclKernelError, Result};
