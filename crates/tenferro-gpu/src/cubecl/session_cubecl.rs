//! Public tenferro-wide CubeCL session (issue #1597).
//!
//! [`Session`] is the promoted, public view over the existing owner-scoped
//! CubeCL integration helpers in [`super::interop`]. While a session is alive,
//! the wrapped `Client` is the exact tenferro CubeCL client bound to the
//! tenferro `CudaRuntime`; work enqueued through it is flushed at session exit.
//!
//! The prelude is intentionally narrow: only the types needed to write and
//! launch CubeCL kernels are re-exported. Downstream crates must depend on the
//! framework's `t4a-cubecl` package (see the design doc) to name `#[cube]`
//! kernels; this module does **not** re-export the whole of `cubecl`.
//!
//! # Launching a downstream kernel
//!
//! The canonical flow is `alloc_output` for the output, `cube_count_1d` plus
//! `cube_dim_1d` for the domain, `array_arg` for each buffer in kernel-parameter
//! order, and `client()` for the launch itself — `KernelArguments` is never
//! built by hand. The `#[cube]` typing rules this pin enforces and the
//! kernel-cache answer are in `docs/guides/custom-cuda-kernels.md`, and
//! `samples/cubecl-kernel` runs the whole flow on CUDA.
//!
//! Nothing has to be cached downstream: the compiled artifact lives in CubeCL's
//! per-device server, which the client returned by [`Session::client`] reaches,
//! so it is reused by every later [`CudaEngine::with_cubecl`] scope and by
//! any other backend constructed for the same device.
//!
//! [`CudaEngine::with_cubecl`]: super::CudaEngine::with_cubecl

use std::marker::PhantomData;
use std::rc::Rc;

use cubecl::client::Client;
use cubecl::prelude::{BufferArg, CubeCount, CubeDim, CubeElement, CubePrimitive, TensorBinding};

use tenferro_tensor::{TensorRank, TensorScalar, TensorWrite, TypedTensor};

use super::runtime::CudaRuntime;
use cubecl::frontend::Scalar;

/// Public CubeCL extension session.
///
/// Not constructible by users; obtained only from
/// [`CudaEngine::with_cubecl`](super::exec_session::CudaExecSession::with_cubecl).
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
    ///
    /// Pass it to the `launch_unchecked` entry point a `#[cube]` kernel
    /// generates. The client is the one bound to this runtime's device, so the
    /// kernel it compiles stays cached in that device's CubeCL server across
    /// sessions; see the module docs for the cache question.
    pub fn client(&self) -> &Client {
        self.runtime.client()
    }

    /// Build a CubeCL tensor binding for a GPU-backed tensor.
    ///
    /// The binding is an owned, lifetime-free descriptor: it stays valid after the borrow of the
    /// tensor ends, which is what lets an extension kernel keep the launch argument alive. That is also
    /// why this is `unsafe`: the descriptor carries no ownership proof, so a caller that passes it to a
    /// kernel or to `ComputeServer::write` writes into storage it may not own, and no owner can claim
    /// immutable storage while such a binding circulates.
    ///
    /// # Safety
    ///
    /// The caller must hold write authority over the tensor's storage: it must own the tensor (or
    /// exclusively borrow it) and must keep the allocation alive and unreachable by other writers for as
    /// long as any launched work may touch it. A caller that only reads through the binding may create it
    /// for its own tensor under the same lifetime rule.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when the tensor is not resident
    /// on this session's runtime/device, or [`crate::Error::Validation`] when
    /// its layout cannot be bound.
    pub unsafe fn tensor_binding<T>(
        &self,
        tensor: &TypedTensor<T, impl TensorRank>,
        op: &'static str,
    ) -> crate::Result<TensorBinding>
    where
        T: CubeElement + TensorScalar + Scalar + Clone,
    {
        super::dispatch::ensure_resident_on_runtime(&self.runtime, tensor, op)?;
        super::interop::typed_tensor_binding(tensor, op)
    }

    /// Build a CubeCL array argument for a GPU-backed tensor.
    ///
    /// # Safety
    ///
    /// Same requirement as [`Self::tensor_binding`]: the caller must hold write authority over the
    /// tensor's storage, because a launch argument can be a kernel output.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when the tensor is not resident
    /// on this session's runtime/device, or [`crate::Error::Validation`] when
    /// its layout cannot be bound.
    pub unsafe fn array_arg<T>(
        &self,
        tensor: &TypedTensor<T, impl TensorRank>,
        op: &'static str,
    ) -> crate::Result<BufferArg>
    where
        T: CubeElement + TensorScalar + Scalar + Clone,
    {
        super::dispatch::ensure_resident_on_runtime(&self.runtime, tensor, op)?;
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
        T: CubeElement + TensorScalar + Scalar + Clone + Send + Sync + 'static,
    {
        super::interop::alloc_output(&self.runtime, shape)
    }

    /// Allocate a dense GPU tensor zero-filled with the session's device.
    ///
    /// Reuses the backend's fill-zero structural kernel; it never uploads a
    /// host tensor or exposes a device pointer to the caller.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] when the shape product, output
    /// byte length, or launch count overflows, [`crate::Error::RuntimeState`]
    /// when the output is not resident, or [`crate::Error::BackendSource`]
    /// when allocation or backend resource inspection fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::cubecl::Session;
    ///
    /// fn check(session: &Session<'_>) -> tenferro_tensor::Result<()> {
    ///     let _ = session.alloc_zero_output::<f32>(&[4])?;
    ///     Ok(())
    /// }
    /// ```
    pub fn alloc_zero_output<T>(&self, shape: &[usize]) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + CubePrimitive + TensorScalar + Scalar + Clone + Send + Sync + 'static,
    {
        super::interop::alloc_zero_output(&self.runtime, shape)
    }

    /// Overwrite a caller-owned CUDA destination with an exact `+0.0`.
    ///
    /// This is the `beta = 0` reset for re-executing an accumulate-form
    /// operation into storage the caller keeps: the previous contents are never
    /// read, so a stale `NaN` or `Inf` cannot survive and `-0.0` is normalized
    /// to `+0.0`, which `0 * y` cannot promise. A compact destination
    /// (including a nonzero-offset region view) is filled by one stream-ordered
    /// memset, so nothing is allocated and no kernel is compiled; a strided
    /// region view runs one native fill kernel over its own coordinates and
    /// leaves every element outside the region untouched.
    ///
    /// Supports `F32`, `F64`, `I32`, `I64`, `C32`, `C64`, and compact `Bool`
    /// destinations.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when the destination is not
    /// resident on this session's runtime, [`crate::Error::Validation`] when
    /// the destination layout or its byte span cannot be represented,
    /// [`crate::Error::Unsupported`] for a strided `Bool` destination, and
    /// [`crate::Error::BackendSource`] when the fill cannot be enqueued.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::cubecl::Session;
    /// use tenferro_tensor::{Tensor, TensorWrite};
    ///
    /// fn reset(session: &Session<'_>) -> tenferro_tensor::Result<()> {
    ///     let destination = session.alloc_output::<f32>(&[4])?;
    ///     let mut destination = Tensor::from_typed::<f32>(destination);
    ///     session.fill_zero_write(TensorWrite::from_tensor(&mut destination))
    /// }
    /// ```
    pub fn fill_zero_write(&self, output: TensorWrite<'_>) -> crate::Result<()> {
        super::interop::fill_zero_write(&self.runtime, output)
    }

    /// Scale a mutable CUDA tensor in place by a real factor.
    ///
    /// Supports F32, F64, C32, and C64 payloads; the factor is interpreted as
    /// a real scalar for complex payloads. This is the op-family scale
    /// primitive used for normalization after a vendor transform; it never
    /// exposes a device pointer to the caller.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when the tensor is not resident
    /// on this session's runtime, a typed layout error for a non-zero-offset
    /// discontinuous view, or the typed unsupported-dtype error for other
    /// payloads.
    pub fn scale_tensor_write(&self, output: TensorWrite<'_>, factor: f64) -> crate::Result<()> {
        super::interop::scale_tensor_write(&self.runtime, output, factor)
    }

    /// Return the cube count for a one-dimensional kernel domain of `len`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] carrying
    /// `ValidationError::InvalidArgument` when the one-dimensional launch for
    /// `len` elements would require more than `u32::MAX` workgroups.
    pub fn cube_count_1d(&self, len: usize) -> crate::Result<CubeCount> {
        super::interop::cube_count_for_len(len)
    }

    /// Return the standard one-dimensional CubeCL launch dimension.
    pub fn cube_dim_1d(&self) -> CubeDim {
        super::interop::cube_dim_1d()
    }
}
