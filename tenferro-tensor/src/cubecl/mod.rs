//! CubeCL-based GPU backend for tenferro tensors.

mod memory;
mod runtime;

pub use memory::{device_ptr, download_tensor, upload_tensor};
pub use runtime::CubeclRuntime;

#[cfg(test)]
mod tests;
