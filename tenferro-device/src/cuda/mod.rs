//! CUDA runtime substrate for `tenferro-device`.
//!
//! This module intentionally stays low level. It owns the shared runtime
//! handle cache, CUDA context binding, allocation/free wrappers, and raw copy
//! helpers that higher-level crates can build on.
//!
//! # Examples
//!
//! ```ignore
//! use tenferro_device::cuda::runtime;
//!
//! let runtime = runtime::get_or_init(0).unwrap();
//! let buffer = runtime.alloc::<f32>(4).unwrap();
//! ```

pub mod runtime;

#[cfg(test)]
mod tests;
