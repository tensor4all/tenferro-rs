//! Public tenferro-wide CubeCL session (issue #1597).
//!
//! [`Session`] is the promoted, public view over the existing owner-scoped
//! CubeCL integration helpers in [`super::interop`]. While a session is alive,
//! the wrapped `ComputeClient` is the exact tenferro CubeCL client bound to the
//! tenferro `CudaRuntime`; work enqueued through it is flushed at session exit.
//!
//! The prelude is intentionally narrow: only the types needed to write and
//! launch CubeCL kernels are re-exported. Downstream crates must depend on the
//! framework's `t4a-cubecl` package (see the design doc) to name `#[cube]`
//! kernels; this module does **not** re-export the whole of `cubecl`.

use std::marker::PhantomData;
use std::rc::Rc;

use cubecl::client::ComputeClient;
use cubecl::prelude::{ArrayArg, CubeCount, CubeDim, CubeElement, TensorBinding};
use cubecl_cuda::CudaRuntime as CubeclCudaRuntime;

use tenferro_tensor::{TensorRank, TensorScalar, TypedTensor};

use super::runtime::CudaRuntime;

/// Public CubeCL extension session.
///
/// Not constructible by users; obtained only from
/// [`CudaExecSession::with_cubecl`](super::exec_session::CudaExecSession::with_cubecl).
/// The session borrows the exact tenferro CubeCL client for the request scope
/// and is `!Send + !Sync` by construction, so the execution authority cannot
/// migrate to another thread.
pub struct Session<'s> {
    runtime: CudaRuntime,
    _scope: PhantomData<&'s ()>,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl<'s> Session<'s> {
    /// Wrap the runtime for one scoped callback.
    ///
    /// # Safety
    ///
    /// Caller must ensure the callback runs on a thread that owns the tenferro
    /// primary context activation for `runtime`, and that the returned borrow
    /// does not outlive the current context.
    pub(crate) unsafe fn new(runtime: CudaRuntime) -> Self {
        Self {
            runtime,
            _scope: PhantomData,
            _not_send_sync: PhantomData,
        }
    }

    /// Borrow the tenferro CubeCL client.
    pub fn client(&self) -> &ComputeClient<CubeclCudaRuntime> {
        self.runtime.client()
    }

    /// Build a CubeCL tensor binding for a GPU-backed tensor.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when the tensor is not CubeCL
    /// resident, or [`crate::Error::Validation`] when its layout cannot be
    /// bound.
    pub fn tensor_binding<T>(
        &self,
        tensor: &TypedTensor<T, impl TensorRank>,
        op: &'static str,
    ) -> crate::Result<TensorBinding<CubeclCudaRuntime>>
    where
        T: CubeElement + TensorScalar + Clone,
    {
        super::interop::typed_tensor_binding(tensor, op)
    }

    /// Build a CubeCL array argument for a GPU-backed tensor.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when the tensor is not CubeCL
    /// resident, or [`crate::Error::Validation`] when its layout cannot be
    /// bound.
    pub fn array_arg<T>(
        &self,
        tensor: &TypedTensor<T, impl TensorRank>,
        op: &'static str,
    ) -> crate::Result<ArrayArg<CubeclCudaRuntime>>
    where
        T: CubeElement + TensorScalar + Clone,
    {
        super::interop::typed_tensor_array_arg(tensor, op)
    }

    /// Allocate a dense GPU tensor on the session's device.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] when the shape product overflows,
    /// or [`crate::Error::BackendSource`] when allocation fails.
    pub fn alloc_output<T>(&self, shape: &[usize]) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + TensorScalar + Clone + Send + Sync + 'static,
    {
        super::interop::alloc_output(&self.runtime, shape)
    }

    /// Return the cube count for a one-dimensional kernel domain of `len`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] when the one-dimensional launch
    /// for `len` elements would require more than `u32::MAX` workgroups.
    pub fn cube_count_1d(&self, len: usize) -> crate::Result<CubeCount> {
        super::interop::cube_count_for_len(len)
    }

    /// Return the standard one-dimensional CubeCL launch dimension.
    pub fn cube_dim_1d(&self) -> CubeDim {
        super::interop::cube_dim_1d()
    }
}
