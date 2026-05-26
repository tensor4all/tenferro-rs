//! GPU backend implementations for tenferro tensors.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_tensor::Tensor;
//!
//! let _tensor_type = core::any::type_name::<Tensor>();
//! assert!(_tensor_type.contains("Tensor"));
//! ```

#[cfg(feature = "cuda")]
use std::any::Any;

#[cfg(feature = "cuda")]
pub mod cubecl;
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub mod kernels;

#[cfg(feature = "cuda")]
pub use cubecl::{
    device_ptr, download_tensor, gpu_available, upload_tensor, CubeclBackend, CubeclRuntime,
};

#[cfg(feature = "cuda")]
use tenferro_tensor::*;

#[cfg(feature = "cuda")]
pub(crate) mod backend {
    pub use tenferro_tensor::backend::*;
}

#[cfg(feature = "cuda")]
pub(crate) mod config {
    pub use tenferro_tensor::config::*;
}

#[cfg(all(test, feature = "cuda"))]
pub(crate) mod cpu {
    pub use tenferro_tensor::cpu::*;
}

#[cfg(feature = "cuda")]
pub(crate) mod types {
    pub use crate::CubeclBuffer;
    pub use tenferro_tensor::types::*;
}

/// CubeCL-managed GPU buffer.
///
/// This wraps a CubeCL server handle that owns the underlying GPU allocation.
///
/// # Examples
///
/// ```
/// let _name = core::any::type_name::<tenferro_gpu::CubeclBuffer<f64>>();
/// assert!(_name.contains("CubeclBuffer"));
/// ```
#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct CubeclBuffer<T> {
    /// CubeCL server handle that owns the GPU allocation.
    pub handle: cubecl_runtime::server::Handle,
    /// Number of elements stored in the allocation.
    pub len: usize,
    pub(crate) _marker: std::marker::PhantomData<T>,
}

#[cfg(feature = "cuda")]
impl<T> std::fmt::Debug for CubeclBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CubeclBuffer")
            .field("handle", &self.handle)
            .field("len", &self.len)
            .finish()
    }
}

#[cfg(feature = "cuda")]
impl<T> CubeclBuffer<T> {
    /// Create a CubeCL buffer wrapper from a handle and element count.
    ///
    /// # Examples
    ///
    /// ```
    /// let _new = tenferro_gpu::CubeclBuffer::<f64>::new;
    /// let _ = _new;
    /// ```
    pub fn new(handle: cubecl_runtime::server::Handle, len: usize) -> Self {
        Self {
            handle,
            len,
            _marker: std::marker::PhantomData,
        }
    }
}

#[cfg(feature = "cuda")]
impl<T: Send + Sync + 'static> BackendBuffer<T> for CubeclBuffer<T> {
    fn backend_family(&self) -> &'static str {
        "cubecl"
    }

    fn len(&self) -> usize {
        self.len
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
