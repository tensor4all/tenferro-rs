//! GPU backend implementations for tenferro tensors.
//!
//! # Examples
//!
//! ```rust
//! #[cfg(feature = "cuda")]
//! {
//!     use tenferro_gpu::{download_tensor, gpu_available, upload_tensor, CubeclBackend};
//!     use tenferro_tensor::{Tensor, TensorElementwise};
//!
//!     if gpu_available() {
//!         let mut backend = CubeclBackend::new(0).unwrap();
//!         let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
//!         let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]);
//!         let gpu_a = upload_tensor(backend.runtime(), &a).unwrap();
//!         let gpu_b = upload_tensor(backend.runtime(), &b).unwrap();
//!         let gpu_sum = backend.add(&gpu_a, &gpu_b).unwrap();
//!         let sum = download_tensor(backend.runtime(), &gpu_sum).unwrap();
//!         assert_eq!(sum.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
//!     }
//! }
//! ```

#[cfg(feature = "cuda")]
use std::any::Any;

#[cfg(feature = "cuda")]
pub mod cubecl;
#[cfg(feature = "cuda")]
mod kernels;

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

#[cfg(feature = "cuda")]
pub(crate) mod types {
    pub use crate::CubeclBuffer;
    pub use tenferro_tensor::types::*;
}

/// CubeCL-managed GPU buffer.
///
/// This is the backend-owned buffer type stored inside tensors uploaded to a
/// CubeCL CUDA runtime. Application code should treat it as opaque and use
/// [`upload_tensor`], [`download_tensor`], and [`device_ptr`] instead of
/// constructing or inspecting buffers directly.
///
/// # Examples
///
/// ```
/// #[cfg(feature = "cuda")]
/// {
///     use tenferro_gpu::{gpu_available, upload_tensor, CubeclBackend, CubeclBuffer};
///     use tenferro_tensor::{BackendBuffer, Buffer, Tensor};
///
///     if gpu_available() {
///         let backend = CubeclBackend::new(0).unwrap();
///         let host = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
///         let gpu = upload_tensor(backend.runtime(), &host).unwrap();
///         let Tensor::F64(tensor) = gpu else { unreachable!() };
///         let Buffer::Backend(buffer) = &tensor.buffer else { unreachable!() };
///         let cubecl = buffer.as_any().downcast_ref::<CubeclBuffer<f64>>().unwrap();
///         assert_eq!(cubecl.backend_family(), "cubecl");
///     }
/// }
/// ```
#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct CubeclBuffer<T> {
    handle: cubecl_runtime::server::Handle,
    len: usize,
    pub(crate) _marker: std::marker::PhantomData<T>,
}

#[cfg(feature = "cuda")]
impl<T> std::fmt::Debug for CubeclBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CubeclBuffer")
            .field("len", &self.len)
            .finish()
    }
}

#[cfg(feature = "cuda")]
impl<T> CubeclBuffer<T> {
    pub(crate) fn new(handle: cubecl_runtime::server::Handle, len: usize) -> Self {
        Self {
            handle,
            len,
            _marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn handle(&self) -> &cubecl_runtime::server::Handle {
        &self.handle
    }

    pub(crate) fn element_len(&self) -> usize {
        self.len
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
