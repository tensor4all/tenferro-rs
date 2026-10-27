//! GPU backend implementations for tenferro tensors.
//!
//! # Examples
//!
//! ```rust
//! #[cfg(feature = "cuda")]
//! use tenferro_gpu::{cuda::cuda_devices, cuda::CudaEngine};
//!
//! #[cfg(feature = "cuda")]
//! fn first_cuda_backend() -> Result<Option<CudaEngine>, Box<dyn std::error::Error>> {
//!     let devices = cuda_devices()?;
//!     let Some(device) = devices.first() else {
//!         return Ok(None);
//!     };
//!     Ok(Some(CudaEngine::open(device.id())?))
//! }
//!
//! // This ordinary doctest checks the discovery-based selection API without
//! // requiring CUDA hardware at test time.
//! #[cfg(feature = "cuda")]
//! let _example: fn() -> Result<Option<CudaEngine>, Box<dyn std::error::Error>> = first_cuda_backend;
//! ```
// A misaligned pointer handed to a CUDA library is undefined behaviour and
// fails only on some library versions: `cuDoubleComplex` is `double2`, which
// CUDA declares `__align__(16)`, while `num_complex::Complex64` is 8-aligned,
// so casting `&Complex64` to `*const cuDoubleComplex` produced a pointer
// cuBLAS >= 12.9 faults on (issue #1870). Deny the whole cast class at this
// FFI boundary rather than re-auditing it by hand.
#![deny(clippy::cast_ptr_alignment)]

#[cfg(feature = "cuda")]
mod event_domain_admission;
#[cfg(feature = "cuda")]
mod event_retirement;
#[cfg(feature = "cuda")]
pub mod gpu;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "cuda")]
use tenferro_tensor::*;
