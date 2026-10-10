use cubecl::prelude::{CubeElement, CubePrimitive};
use cubecl::stream_id::StreamId;
use num_complex::{Complex32, Complex64};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Mutex;
use tenferro_tensor::backend::{
    BackendSession, BackendSessionHost, ElementwiseFusionPlan, ElementwiseReadOp, SessionCachedDot,
    TensorAnalytic, TensorBuffer, TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion,
    TensorIndexing, TensorReduction, TensorStructural,
};
use tenferro_tensor::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use tenferro_tensor::DType;
use tenferro_tensor::{
    with_session_entry_guard, TensorRank, TensorScalar, TensorViewCanonicalization,
    TypedTensorView, TypedTensorViewMut,
};
use tenferro_tensor::{DotGeneralAccumulation, Tensor, TensorRead, TensorWrite, TypedTensor};

use super::identity::GpuExtensionCapability;
use super::{gemm, ops, runtime::RawContextRestore};
use super::{
    raw, session_cubecl, CudaBackend, CudaBackendState, CudaDeviceInfo, CudaExtensionCache,
    CudaRuntime, CudaRuntimeIdentity,
};

/// Best-effort exit flush for a `with_cubecl` session.
///
/// Flushes once eagerly (returned to the caller as an error if it fails) and
/// once more on `Drop` so a panic/unwind path still drains pending CubeCL
/// work.
struct CubeclExitFlush<'a> {
    op: &'static str,
    client: &'a cubecl::client::ComputeClient<cubecl_cuda::CudaRuntime>,
    flushed: bool,
}

impl<'a> CubeclExitFlush<'a> {
    fn new(
        op: &'static str,
        client: &'a cubecl::client::ComputeClient<cubecl_cuda::CudaRuntime>,
    ) -> Self {
        Self {
            op,
            client,
            flushed: false,
        }
    }

    /// Flush now and return the typed result.
    fn flush_now(&mut self) -> crate::Result<()> {
        self.client
            .flush()
            .map_err(|err| crate::Error::backend_source(self.op, err))?;
        self.flushed = true;
        Ok(())
    }
}

impl Drop for CubeclExitFlush<'_> {
    fn drop(&mut self) {
        if !self.flushed {
            let _ = self.client.flush();
        }
    }
}

/// Native-session marker for [`CudaExecSession`]; private to this crate so no other
/// crate can create a token that claims to be this session.
pub(super) struct CudaExecSessionMarker;

/// Borrowed CUDA execution capability.
///
/// This is the single public execution-authority boundary for CUDA kernel
/// extensions (issue #1597). External operation crates obtain it through
/// [`with_cuda_exec_session`] and then borrow backend/device-scoped extension
/// sessions via [`CudaExecSession::with_cubecl`] and
/// [`CudaExecSession::with_raw`].
///
/// The session is not constructible by users and is `!Send + !Sync`: it
/// carries thread-local execution capability. Success of an enrolled operation
/// means the work was enqueued; only [`CudaExecSession::synchronize`] is a
/// host barrier.
///
/// The backend owner is not an operation route, so an operation bound does not
/// hold for it:
///
/// ```compile_fail
/// fn requires_elementwise<B: tenferro_tensor::TensorElementwise>() {}
/// requires_elementwise::<tenferro_gpu::cuda::CudaBackend>();
/// ```
#[derive(Debug)]
pub struct CudaExecSession<'a> {
    backend: &'a mut CudaBackend,
    _not_send_sync: PhantomData<Rc<()>>,
}

/// The typed tensor behind `input`, or the refusal this method produces for one.
///
/// Callers reach this from a match on `input.dtype()`, so `None` means the tag table
/// and the runtime dtype disagree rather than a caller mistake.
fn gpu_resident_typed<'a, T: TensorScalar>(
    op: &'static str,
    input: &'a Tensor,
) -> crate::Result<&'a TypedTensor<T>> {
    input.as_typed::<T>().ok_or_else(|| {
        crate::Error::unsupported(
            op,
            "an externally defined payload is not supported by this GPU operation",
        )
    })
}

impl CudaExecSession<'_> {
    /// Borrow the provider runtime without exposing the backend.
    pub fn runtime(&self) -> &CudaRuntime {
        self.backend.runtime()
    }

    /// Return the identity of the borrowed provider runtime.
    pub fn runtime_identity(&self) -> CudaRuntimeIdentity {
        self.backend.runtime_identity()
    }

    /// Report whether this session supports a GPU extension capability.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::{CudaExecSession, GpuExtensionCapability};
    ///
    /// // Method-call check only: `CudaExecSession` is not user-constructible, so
    /// // the example asserts the method is callable from an external crate.
    /// fn check(session: &CudaExecSession<'_>, capability: GpuExtensionCapability) -> bool {
    ///     session.supports(capability)
    /// }
    /// let _ = check;
    /// ```
    pub fn supports(&self, capability: GpuExtensionCapability) -> bool {
        self.backend.runtime().supports_extension(capability)
    }

    /// Borrow immutable metadata for the session's device.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaExecSession;
    ///
    /// // Method-call check only: `CudaExecSession` is not user-constructible, so
    /// // the example asserts the method is callable from an external crate.
    /// fn check(session: &CudaExecSession<'_>) {
    ///     let _ = session.device_info();
    /// }
    /// let _ = check;
    /// ```
    pub fn device_info(&self) -> &CudaDeviceInfo {
        self.backend.runtime().device_info()
    }

    /// Return the allocation ownership domain of this session.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaExecSession;
    ///
    /// // Method-call check only: `CudaExecSession` is not user-constructible, so
    /// // the example asserts the method is callable from an external crate.
    /// fn check(session: &CudaExecSession<'_>) -> tenferro_tensor::AllocationDomainId {
    ///     session.allocation_domain()
    /// }
    /// let _ = check;
    /// ```
    pub fn allocation_domain(&self) -> tenferro_tensor::AllocationDomainId {
        self.backend.runtime().allocation_domain()
    }

    /// Validate that a dense GPU tensor is resident on this exact session:
    /// CubeCL-backed, same allocation domain, and placed on this runtime's
    /// CUDA device. Rejects host tensors, foreign-backend buffers, and
    /// foreign-runtime/device tensors without an implicit transfer.
    ///
    /// This is the credentialed public-seam residency guard for extension
    /// crates that receive a session but must validate inputs before entering
    /// a `with_raw`/`with_cubecl` sub-session.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when the tensor is not resident
    /// on this exact session runtime/device.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaExecSession;
    ///
    /// // Method-call check only: `CudaExecSession` is not user-constructible.
    /// fn check(session: &CudaExecSession<'_>, tensor: &tenferro_tensor::Tensor) -> tenferro_tensor::Result<()> {
    ///     session.ensure_gpu_resident(tensor, "test.ensure_gpu_resident")
    /// }
    /// let _ = check;
    /// ```
    pub fn ensure_gpu_resident(&self, input: &Tensor, op: &'static str) -> crate::Result<()> {
        match input.dtype() {
            DType::F32 => super::dispatch::ensure_resident_on_runtime(
                self.runtime(),
                gpu_resident_typed::<f32>("ensure_gpu_resident", input)?,
                op,
            ),
            DType::F64 => super::dispatch::ensure_resident_on_runtime(
                self.runtime(),
                gpu_resident_typed::<f64>("ensure_gpu_resident", input)?,
                op,
            ),
            DType::I32 => super::dispatch::ensure_resident_on_runtime(
                self.runtime(),
                gpu_resident_typed::<i32>("ensure_gpu_resident", input)?,
                op,
            ),
            DType::I64 => super::dispatch::ensure_resident_on_runtime(
                self.runtime(),
                gpu_resident_typed::<i64>("ensure_gpu_resident", input)?,
                op,
            ),
            DType::Bool => super::dispatch::ensure_resident_on_runtime(
                self.runtime(),
                gpu_resident_typed::<bool>("ensure_gpu_resident", input)?,
                op,
            ),
            DType::C32 => super::dispatch::ensure_resident_on_runtime(
                self.runtime(),
                gpu_resident_typed::<Complex32>("ensure_gpu_resident", input)?,
                op,
            ),
            DType::C64 => super::dispatch::ensure_resident_on_runtime(
                self.runtime(),
                gpu_resident_typed::<Complex64>("ensure_gpu_resident", input)?,
                op,
            ),
            // A caller-owned payload has no GPU implementation for this operation.
            DType::External(_) => Err(crate::Error::unsupported(
                "ensure_gpu_resident",
                "an externally defined payload is not supported by this GPU operation",
            )),
        }
    }

    /// Block the host until work enqueued on the session's stream completes.
    ///
    /// This is the only host barrier on the success path; ordinary successful
    /// session operations only enqueue.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] when CUDA stream
    /// synchronization fails.
    pub fn synchronize(&mut self) -> crate::Result<()> {
        self.backend.runtime().synchronize()
    }

    /// Borrow the type-safe raw CUDA extension session for one operation.
    ///
    /// The enter/exit protocol is fully contained in this call: a definite
    /// CubeCL stream is captured on the current thread, pending CubeCL work is
    /// flushed, the calling thread's previous device/context is saved, the
    /// tenferro primary context is activated, the callback runs, and the
    /// previous device/context is best-effort restored on return, `Err`, or
    /// unwind (restoration failures are logged to stderr). The success path
    /// does not synchronize.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] when CubeCL cannot expose or
    /// flush the stream, or when the CUDA context cannot be entered. Context
    /// restoration on exit is best-effort: a failure to restore the caller's
    /// previous device/context is logged to stderr rather than propagated, so
    /// a callback result is never replaced by a restore error.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaExecSession;
    ///
    /// // Method-call check only: `CudaExecSession` is not user-constructible.
    /// fn check(session: &mut CudaExecSession<'_>) -> tenferro_tensor::Result<()> {
    ///     session.with_raw("test.raw", |raw| {
    ///         let _ = raw.stream();
    ///         Ok(SessionOutcome::Done)
    ///     })?;
    ///     Ok(())
    /// }
    /// enum SessionOutcome { Done }
    /// let _ = check;
    /// ```
    pub fn with_raw<R>(
        &mut self,
        op: &'static str,
        f: impl for<'s> FnOnce(&mut raw::Session<'s>) -> crate::Result<R>,
    ) -> crate::Result<R> {
        let runtime = self.backend.runtime().clone();
        let cache = self.backend.cuda_extension_cache();
        // 1. Capture the definite CubeCL stream on this thread.
        let stream = runtime.raw_cuda_stream()?;
        // 2. Flush pending CubeCL work so raw library calls observe it.
        runtime.flush_cubecl(op)?;
        // 3-4. Save previous context, activate the tenferro primary context.
        let device_ordinal = i32::try_from(runtime.device_ordinal())
            .map_err(|source| crate::Error::backend_source(op, source))?;
        let _guard = RawContextRestore::enter(op, device_ordinal, runtime.primary_context())?;
        // 5. Build the unique raw session and run the callback.
        // SAFETY: `_guard` keeps the primary context current for the whole
        // `Session<'s>` borrow; `stream` is the captured CubeCL stream bound to
        // the current thread.
        let mut session = unsafe { raw::Session::new(runtime, cache, stream) };
        f(&mut session)
    }

    /// Borrow the public tenferro-wide CubeCL session for one operation.
    ///
    /// The session exposes the exact tenferro CubeCL client bound to this
    /// runtime. Pending CubeCL work is flushed before entering and again on
    /// exit (including `Err` and unwind) so a later raw-session or host read
    /// observes the enqueued work. The success path does not synchronize.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaExecSession;
    ///
    /// fn check(session: &mut CudaExecSession<'_>) {
    ///     let _ = session.with_cubecl("test.cubecl", |_cubecl| Ok(()));
    /// }
    /// let _ = check;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns the callback's error, or [`crate::Error::BackendSource`] when
    /// pending CubeCL work cannot be flushed on entry or exit.
    pub fn with_cubecl<R>(
        &mut self,
        op: &'static str,
        f: impl for<'s> FnOnce(&session_cubecl::Session<'s>) -> crate::Result<R>,
    ) -> crate::Result<R> {
        let runtime = self.backend.runtime().clone();
        runtime.flush_cubecl(op)?;
        let session = unsafe { session_cubecl::Session::new(runtime) };
        // Best-effort exit flush on every path via Drop.
        let mut _flush_guard = CubeclExitFlush::new(op, session.client());
        let result = f(&session);
        let flush_result = _flush_guard.flush_now();
        match result {
            Ok(value) => {
                flush_result?;
                Ok(value)
            }
            Err(err) => {
                let _ = flush_result;
                Err(err)
            }
        }
    }

    #[doc(hidden)]
    pub fn tril_typed<T>(&self, input: &TypedTensor<T>, k: i64) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + TensorScalar + CubePrimitive + Clone,
    {
        self.backend.tril_typed(input, k)
    }

    #[doc(hidden)]
    pub fn slice_typed<T>(
        &self,
        input: &TypedTensor<T>,
        config: &SliceConfig,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + TensorScalar + CubePrimitive + Clone,
    {
        self.backend.slice_typed(input, config)
    }

    /// Borrow the CUDA extension cache owned by the provider runtime.
    #[doc(hidden)]
    pub fn cuda_extension_cache(&self) -> &CudaExtensionCache {
        self.backend.cuda_extension_cache()
    }

    #[doc(hidden)]
    pub fn triu_typed<T>(&self, input: &TypedTensor<T>, k: i64) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + TensorScalar + CubePrimitive + Clone,
    {
        self.backend.triu_typed(input, k)
    }
}

// Typed view canonicalization runs on the session, never on the backend
// owner: the owner is not an execution surface (#1946 F6).
macro_rules! impl_session_view_canonicalization {
    ($to_contiguous:ident; $($ty:ty),* $(,)?) => {
        $(
            impl<R> TensorViewCanonicalization<$ty, R> for CudaExecSession<'_>
            where
                R: TensorRank,
            {
                fn to_contiguous(
                    &mut self,
                    view: &TypedTensorView<'_, $ty, R>,
                ) -> crate::Result<TypedTensor<$ty, R>> {
                    self.backend
                        .$to_contiguous(view, "CudaExecSession::to_contiguous")
                }

                fn copy_into(
                    &mut self,
                    src: &TypedTensorView<'_, $ty, R>,
                    dst: &mut TypedTensorViewMut<'_, $ty, R>,
                ) -> crate::Result<()> {
                    self.backend
                        .copy_view_to_view_typed(src, dst, "CudaExecSession::copy_into")
                }
            }
        )*
    };
}

impl_session_view_canonicalization!(
    to_contiguous_view_cutensor_or_cubecl; f32, f64, Complex32, Complex64
);
impl_session_view_canonicalization!(to_contiguous_view_typed; i32, i64);

impl<R> TensorViewCanonicalization<bool, R> for CudaExecSession<'_>
where
    R: TensorRank,
{
    fn to_contiguous(
        &mut self,
        _view: &TypedTensorView<'_, bool, R>,
    ) -> crate::Result<TypedTensor<bool, R>> {
        Err(super::error::unsupported_dtype(
            "CudaExecSession::to_contiguous",
            crate::DType::Bool,
        ))
    }

    fn copy_into(
        &mut self,
        _src: &TypedTensorView<'_, bool, R>,
        _dst: &mut TypedTensorViewMut<'_, bool, R>,
    ) -> crate::Result<()> {
        Err(super::error::unsupported_dtype(
            "CudaExecSession::copy_into",
            crate::DType::Bool,
        ))
    }
}

/// Visit a CUDA execution session through the erased backend-session surface.
///
/// This is the public entry point that borrows CUDA execution authority for
/// the duration of the callback (issue #1597). The callback cannot return a
/// borrow of the reconstructed session, so the authority cannot escape the
/// scope.
///
/// Returns `None` when `session` is not a CUDA execution session.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::cuda::{with_cuda_exec_session, CudaExecSession};
///
/// // Call-check only: the visitor borrows CUDA execution authority for the
/// // duration of the callback.
/// fn check(session: &mut dyn tenferro_tensor::backend::BackendSession) {
///     let _ = with_cuda_exec_session(session, |_session| 0usize);
/// }
/// let _ = check;
/// ```
pub fn with_cuda_exec_session<B, R>(
    session: &mut B,
    f: impl for<'a> FnOnce(&'a mut CudaExecSession<'a>) -> R,
) -> Option<R>
where
    B: BackendSession + ?Sized,
{
    let data = session
        .native_session()?
        .into_marked_ptr::<CudaExecSessionMarker>()?;
    // SAFETY: only `CudaExecSession::native_session` creates a token with the
    // crate-private `CudaExecSessionMarker`, and it points that token at a live
    // `CudaExecSession`. The token borrowed `*session` exclusively, and this function
    // keeps holding `session: &mut B` for the whole scoped visit.
    Some(unsafe { f(data.cast::<CudaExecSession<'static>>().as_mut()) })
}

macro_rules! delegate {
    ($trait:path {
        $(fn $method:ident($($arg:ident: $arg_ty:ty),* $(,)?) -> $ret:ty;)*
    }) => {
        impl $trait for CudaExecSession<'_> {
            $(
                fn $method(&mut self, $($arg: $arg_ty),*) -> $ret {
                    self.backend.$method($($arg),*)
                }
            )*
        }
    };
}

macro_rules! delegate_ops {
    ($trait:path {
        $(fn $method:ident($($arg:ident: $arg_ty:ty),* $(,)?) -> $ret:ty;)*
    } $(override { $($custom:item)* })?) => {
        impl $trait for CudaExecSession<'_> {
            $(
                fn $method(&mut self, $($arg: $arg_ty),*) -> $ret {
                    ops::$method(self.backend, $($arg),*)
                }
            )*
            $($($custom)*)?
        }
    };
}

delegate_ops!(TensorElementwise {
    fn add_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor>;
    fn sub_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor>;
    fn mul_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor>;
    fn neg_read(input: TensorRead<'_>) -> crate::Result<Tensor>;
    fn conj_read(input: TensorRead<'_>) -> crate::Result<Tensor>;
    fn div_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor>;
    fn rem_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor>;
    fn abs_read(input: TensorRead<'_>) -> crate::Result<Tensor>;
    fn sign_read(input: TensorRead<'_>) -> crate::Result<Tensor>;
    fn maximum_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor>;
    fn minimum_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor>;
    fn compare_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>, dir: &CompareDir) -> crate::Result<Tensor>;
    fn select_read(pred: TensorRead<'_>, on_true: TensorRead<'_>, on_false: TensorRead<'_>) -> crate::Result<Tensor>;
    fn clamp_read(input: TensorRead<'_>, lower: TensorRead<'_>, upper: TensorRead<'_>) -> crate::Result<Tensor>;
    fn rem(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
} override {
    // Read-into elementwise dispatch must stay session-shaped: the allocating
    // fallback in `tenferro-tensor` is generic over `TensorElementwise`, and the
    // session is the only type in this crate that implements it now. The native
    // read-into kernels still run first, so no work moves onto the allocating path.
    fn elementwise_read_into(
        &mut self,
        op: ElementwiseReadOp,
        inputs: &[TensorRead<'_>],
        mut out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        if inputs.len() != op.arity() {
            return Err(crate::Error::invalid_argument(
                op.label(),
                "inputs",
                format!("expected {} inputs, got {}", op.arity(), inputs.len()),
            ));
        }
        tenferro_tensor::backend::validate_read_into_destination(op.label(), inputs, &out)?;
        if let Some(result) = self.backend.elementwise_read_into_native(op, inputs, &mut out) {
            return result;
        }
        tenferro_tensor::backend::elementwise_read_into_via_allocating_ops(self, op, inputs, out)
    }
});

delegate_ops!(TensorAnalytic {
    fn exp_read(input: TensorRead<'_>) -> crate::Result<Tensor>;
    fn log_read(input: TensorRead<'_>) -> crate::Result<Tensor>;
    fn sin_read(input: TensorRead<'_>) -> crate::Result<Tensor>;
    fn cos_read(input: TensorRead<'_>) -> crate::Result<Tensor>;
    fn tanh_read(input: TensorRead<'_>) -> crate::Result<Tensor>;
    fn sqrt_read(input: TensorRead<'_>) -> crate::Result<Tensor>;
    fn rsqrt_read(input: TensorRead<'_>) -> crate::Result<Tensor>;
    fn pow_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor>;
    fn expm1_read(input: TensorRead<'_>) -> crate::Result<Tensor>;
    fn log1p_read(input: TensorRead<'_>) -> crate::Result<Tensor>;
    fn erf_read(input: TensorRead<'_>) -> crate::Result<Tensor>;
});

delegate_ops!(TensorStructural {
    fn transpose_read(input: TensorRead<'_>, perm: &[usize]) -> crate::Result<Tensor>;
    fn reshape_read(input: TensorRead<'_>, shape: &[usize]) -> crate::Result<Tensor>;
    fn broadcast_in_dim_read(input: TensorRead<'_>, shape: &[usize], dims: &[usize]) -> crate::Result<Tensor>;
    fn to_contiguous_read(input: TensorRead<'_>) -> crate::Result<Tensor>;
    fn copy_read_into(src: TensorRead<'_>, dst: TensorWrite<'_>) -> crate::Result<()>;
    fn cast(input: &Tensor, to: tenferro_tensor::DType) -> crate::Result<Tensor>;
    fn extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> crate::Result<Tensor>;
    fn embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> crate::Result<Tensor>;
    fn tril(input: &Tensor, k: i64) -> crate::Result<Tensor>;
    fn triu(input: &Tensor, k: i64) -> crate::Result<Tensor>;
});

delegate_ops!(TensorReduction {
    fn reduce_sum_read(input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor>;
    fn reduce_prod_read(input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor>;
    fn reduce_max_read(input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor>;
    fn reduce_min_read(input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor>;
    fn reduce_sum_squares_read(input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor>;
});

delegate_ops!(TensorDot {
    fn dot_general_with_conj(
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor>;
    fn dot_general_read(
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor>;
    fn dot_general_read_into_accum(
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        accumulation: DotGeneralAccumulation,
        out: TensorWrite<'_>,
    ) -> crate::Result<()>;
});

delegate_ops!(TensorIndexing {
    fn gather(
        operand: &Tensor,
        start_indices: &Tensor,
        config: &GatherConfig,
    ) -> crate::Result<Tensor>;
    fn scatter(
        operand: &Tensor,
        scatter_indices: &Tensor,
        updates: &Tensor,
        config: &ScatterConfig,
    ) -> crate::Result<Tensor>;
    fn slice(input: &Tensor, config: &SliceConfig) -> crate::Result<Tensor>;
    fn dynamic_slice(
        input: &Tensor,
        starts: &Tensor,
        slice_sizes: &[usize],
    ) -> crate::Result<Tensor>;
    fn dynamic_update_slice(
        operand: &Tensor,
        update: &Tensor,
        starts: &Tensor,
    ) -> crate::Result<Tensor>;
    fn pad(input: &Tensor, config: &PadConfig) -> crate::Result<Tensor>;
    fn concatenate(inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor>;
    fn reverse(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
});

delegate_ops!(TensorFusion {
    fn execute_elementwise_fusion(
        inputs: &[&Tensor],
        plan: &ElementwiseFusionPlan,
    ) -> crate::Result<Option<Vec<Tensor>>>;
    fn execute_broadcast_multiply(
        lhs: TensorRead<'_>,
        lhs_shape: &[usize],
        lhs_dims: &[usize],
        rhs: TensorRead<'_>,
        rhs_shape: &[usize],
        rhs_dims: &[usize],
    ) -> crate::Result<Option<Tensor>>;
});

// CUDA device buffers return to the runtime allocator on drop, so the
// session keeps the trait's no-op reclaim; the owner is not a buffer surface.
impl TensorBuffer for CudaExecSession<'_> {}

delegate!(TensorDeviceTransfer {
    fn download_to_host(tensor: TensorRead<'_>) -> crate::Result<Tensor>;
    fn upload_host_tensor(tensor: TensorRead<'_>) -> crate::Result<Tensor>;
});

impl SessionCachedDot for CudaExecSession<'_> {
    // Read-based cached dot paths keep strided operands on device; the plan
    // cache is per backend, so the runtime cache slot stays unused.
    fn dot_general_read_cached(
        &mut self,
        _cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        gemm::dot_general_read_allocating(self.backend, lhs, rhs, config, false, false)
    }

    fn dot_general_with_conj_read_cached(
        &mut self,
        _cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        gemm::dot_general_read_allocating(self.backend, lhs, rhs, config, lhs_conj, rhs_conj)
    }
}

impl BackendSession for CudaExecSession<'_> {
    fn vdot_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        ops::vdot_read(self.backend, lhs, rhs)
    }

    fn norm_squared_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        ops::norm_squared_read(self.backend, input)
    }

    fn axpby_read_into_accum(
        &mut self,
        alpha: tenferro_tensor::ContractionScalar,
        x: TensorRead<'_>,
        beta: tenferro_tensor::ContractionScalar,
        y: TensorWrite<'_>,
    ) -> crate::Result<()> {
        ops::axpby_read_into_accum(self.backend, alpha, x, beta, y)
    }

    fn native_session(&mut self) -> Option<tenferro_tensor::NativeSessionRef<'_>> {
        // SAFETY: `CudaExecSessionMarker` is private to this crate, and this is the only
        // place a token carrying it is created; it always points to a
        // `CudaExecSession`, exclusively borrowed for the token lifetime.
        Some(unsafe { tenferro_tensor::NativeSessionRef::new::<CudaExecSessionMarker, _>(self) })
    }
}

impl BackendSessionHost for CudaBackend {
    fn with_backend_session<R>(
        &mut self,
        f: impl FnOnce(&mut dyn BackendSession) -> R,
    ) -> Result<R, tenferro_tensor::SessionEntryError> {
        // A held session owns this state's execution binding: a new root would silently share
        // the domain with it. Report the conflict before any device call.
        self.inner.held_session.reject_root_entry()?;
        let mut session = CudaExecSession {
            backend: self,
            _not_send_sync: PhantomData,
        };
        // The portable in-session guard rejects nested entry before `f` runs;
        // the CUDA runtime must never re-enter a session closure.
        with_session_entry_guard("CudaBackend", || f(&mut session))
    }
}

/// Backend name used by this module's session-entry vocabulary.
const CUDA_BACKEND_NAME: &str = "CudaBackend";

/// Reservation slot for the one held session of a backend state.
///
/// Every [`CudaBackend`] clone shares one `Arc<CudaBackendState>`, so the slot lives there and a
/// reservation taken through one clone is visible to all of them. The check-and-set never waits:
/// a conflicting root is reported typed instead.
#[derive(Debug, Default)]
pub(super) struct CudaHeldSessionSlot {
    owner: Mutex<Option<std::thread::ThreadId>>,
}

impl CudaHeldSessionSlot {
    pub(super) fn new() -> Self {
        Self::default()
    }

    fn holder(&self) -> Result<Option<std::thread::ThreadId>, tenferro_tensor::SessionEntryError> {
        self.owner.lock().map(|owner| *owner).map_err(|_| {
            tenferro_tensor::SessionEntryError::ResourcePoisoned {
                backend: CUDA_BACKEND_NAME,
                resource: "CUDA held-session reservation",
            }
        })
    }

    /// The conflicting-root outcome for a holder seen by `thread`.
    fn conflict(
        holder: Option<std::thread::ThreadId>,
        thread: std::thread::ThreadId,
    ) -> Option<tenferro_tensor::SessionEntryError> {
        match holder {
            None => None,
            Some(holder) if holder == thread => {
                Some(tenferro_tensor::SessionEntryError::Reentered {
                    backend: CUDA_BACKEND_NAME,
                })
            }
            Some(_) => Some(tenferro_tensor::SessionEntryError::Contended {
                backend: CUDA_BACKEND_NAME,
                message: "a held CUDA session is open on another thread of this backend state; \
                          pass the held session instead of opening another root"
                    .to_string(),
            }),
        }
    }

    /// Reject a second root session while a held session is live.
    pub(super) fn reject_root_entry(&self) -> Result<(), tenferro_tensor::SessionEntryError> {
        let holder = self.holder()?;
        match Self::conflict(holder, std::thread::current().id()) {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// Take the reservation for `state`, or report the conflicting root.
    fn acquire(
        &self,
        state: &std::sync::Arc<CudaBackendState>,
    ) -> Result<CudaHeldSessionOwner, tenferro_tensor::SessionEntryError> {
        let thread = std::thread::current().id();
        let mut owner = self.owner.lock().map_err(|_| {
            tenferro_tensor::SessionEntryError::ResourcePoisoned {
                backend: CUDA_BACKEND_NAME,
                resource: "CUDA held-session reservation",
            }
        })?;
        if let Some(error) = Self::conflict(*owner, thread) {
            return Err(error);
        }
        *owner = Some(thread);
        drop(owner);
        Ok(CudaHeldSessionOwner {
            thread,
            state: std::sync::Arc::clone(state),
        })
    }

    /// Release the reservation held by `thread`, if it is still the holder.
    fn release(&self, thread: std::thread::ThreadId) {
        if let Ok(mut owner) = self.owner.lock() {
            if *owner == Some(thread) {
                *owner = None;
            }
        }
    }
}

/// Owns one backend state's held-session reservation until it drops.
struct CudaHeldSessionOwner {
    thread: std::thread::ThreadId,
    state: std::sync::Arc<CudaBackendState>,
}

impl Drop for CudaHeldSessionOwner {
    fn drop(&mut self) {
        self.state.held_session.release(self.thread);
    }
}

impl CudaBackendState {
    /// Take this state's held-session reservation.
    fn acquire_held_session(
        self: &std::sync::Arc<Self>,
    ) -> Result<CudaHeldSessionOwner, tenferro_tensor::SessionEntryError> {
        self.held_session.acquire(self)
    }
}

/// What the session observed. `Copy` and cheap: the counters report session-observed calls, not
/// what the device did.
///
/// Substrate-internal submissions and waits (interop flushes, CubeCL's automatic staging flush,
/// host-dispatch batches, workspace retirement barriers) are not attributed here: they happen
/// inside shared machinery with no session identity.
/// # Examples
///
/// ```no_run
/// use tenferro_gpu::cuda::CudaBackend;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut backend = CudaBackend::new(tenferro_gpu::cuda::CudaDeviceId::from_ordinal(0))?;
/// let session = backend.open_session()?;
/// let stats = session.stats();
/// assert_eq!(stats.held_operations, 0);
/// session.close()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaSessionStats {
    /// `with_session` callbacks observed. This is a callback count, not a tensor-operation
    /// count: one callback may run an arbitrary chain.
    pub held_operations: u64,
    /// Explicit `submit` attempts.
    pub explicit_submits: u64,
    /// Submission attempts performed by `close`.
    pub close_submits: u64,
    /// Submission attempts (`submit` or `close`) that returned an error.
    pub failed_submits: u64,
    /// Explicit `synchronize` attempts.
    pub explicit_synchronizes: u64,
    /// `synchronize` attempts that returned an error.
    pub failed_synchronizes: u64,
}

/// Held concrete CUDA session (#1945 U3).
///
/// One session keeps this backend state's execution binding — the logical CubeCL stream captured
/// when it opened — and its reservation, so a stage can run many concrete operations without
/// re-entering the backend per operation. The operations themselves are the existing
/// [`CudaExecSession`] authority, handed out per `with_session` call, so extension visitation
/// (`with_cuda_exec_session`) and every numerical route are unchanged.
///
/// The session is `!Send + !Sync`: it binds a stream and its admission to the opening thread.
/// While it is live, this backend state admits no second root: another `open_session` on the same
/// state (including through a clone) and a scoped `with_backend_session` are rejected typed, and
/// the thread carries a lifetime-visible marker that owners which serialize callers check before
/// blocking.
///
/// It does **not** make submission asynchronous. `submit`/`close` use the existing flush, which
/// dispatches on the host and waits for the previous staged batch's fence; this type records the
/// boundaries it crosses rather than claiming they are free.
///
/// # Examples
///
/// ```no_run
/// use tenferro_gpu::cuda::CudaBackend;
/// use tenferro_tensor::{BackendSession, Tensor, TensorElementwise, TensorRead};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut backend = CudaBackend::new(tenferro_gpu::cuda::CudaDeviceId::from_ordinal(0))?;
/// let mut session = backend.open_session()?;
/// let a = Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
/// let value = session.with_session(|view| {
///     view.add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&a))
/// })??;
/// assert_eq!(value.as_slice::<f64>()?, &[2.0]);
/// session.submit()?;
/// let stats = session.close()?;
/// assert_eq!(stats.held_operations, 1);
/// # Ok(())
/// # }
/// ```
pub struct CudaHeldSession<'session> {
    backend: &'session mut CudaBackend,
    _owner: CudaHeldSessionOwner,
    _marker: tenferro_tensor::HeldSessionMarker,
    stream: StreamId,
    device_ordinal: usize,
    stats: CudaSessionStats,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl std::fmt::Debug for CudaHeldSession<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaHeldSession")
            .field("device_ordinal", &self.device_ordinal)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

impl CudaHeldSession<'_> {
    /// Device ordinal whose binding this session captured.
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    /// Run one operation on the held binding, through the existing CUDA session authority.
    ///
    /// The callback receives the same `&mut dyn BackendSession` view the scoped entry hands out,
    /// so every concrete route and `with_cuda_exec_session` visitation work unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::SessionEntryError`] when another portable session entry is
    /// already active on this thread, exactly as a scoped CUDA callback behaves.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use tenferro_gpu::cuda::CudaBackend;
    /// use tenferro_tensor::{BackendSession, Tensor, TensorElementwise, TensorRead};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut backend = CudaBackend::new(tenferro_gpu::cuda::CudaDeviceId::from_ordinal(0))?;
    /// let mut session = backend.open_session()?;
    /// let a = Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
    /// let sum = session.with_session(|view| {
    ///     view.add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&a))
    /// })??;
    /// assert_eq!(sum.as_slice::<f64>()?, &[2.0]);
    /// session.close()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_session<R>(
        &mut self,
        f: impl FnOnce(&mut dyn BackendSession) -> R,
    ) -> Result<R, tenferro_tensor::SessionEntryError> {
        let stream = self.stream;
        let backend: &mut CudaBackend = self.backend;
        self.stats.held_operations = self.stats.held_operations.saturating_add(1);
        stream.executes(|| {
            let mut session = CudaExecSession {
                backend,
                _not_send_sync: PhantomData,
            };
            // The portable guard rejects nested entry before `f` runs, as it does for a scoped
            // CUDA callback; the held marker stays set for the session's whole lifetime.
            with_session_entry_guard(CUDA_BACKEND_NAME, || f(&mut session))
        })
    }

    /// Submit pending device work at a caller-chosen boundary.
    ///
    /// Uses the existing flush: the host dispatch happens here, and from the second submission
    /// onward this waits for the previous staged batch's fence, which is what makes reuse of the
    /// retired staging safe. The wait is counted as a submission boundary, not hidden.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] when the CubeCL flush reports a
    /// dispatch failure, and [`crate::Error::RuntimeState`] when the runtime's
    /// own state forbids the submission.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use tenferro_gpu::cuda::CudaBackend;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut backend = CudaBackend::new(tenferro_gpu::cuda::CudaDeviceId::from_ordinal(0))?;
    /// let mut session = backend.open_session()?;
    /// session.submit()?;
    /// session.close()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn submit(&mut self) -> crate::Result<()> {
        let result = self.flush();
        self.stats.explicit_submits = self.stats.explicit_submits.saturating_add(1);
        result
    }

    /// Explicit host barrier: submit, wait for the bound stream, resolve deferred retirement.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] when the flush or the CUDA stream
    /// synchronization fails, and [`crate::Error::RuntimeState`] when the runtime's
    /// own state forbids the barrier.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use tenferro_gpu::cuda::CudaBackend;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut backend = CudaBackend::new(tenferro_gpu::cuda::CudaDeviceId::from_ordinal(0))?;
    /// let mut session = backend.open_session()?;
    /// session.synchronize()?;
    /// session.close()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn synchronize(&mut self) -> crate::Result<()> {
        let stream = self.stream;
        let runtime = self.backend.runtime();
        let result = stream.executes(|| runtime.synchronize());
        self.stats.explicit_synchronizes = self.stats.explicit_synchronizes.saturating_add(1);
        if result.is_err() {
            self.stats.failed_synchronizes = self.stats.failed_synchronizes.saturating_add(1);
        }
        result
    }

    /// Counters observed by this session so far.
    pub fn stats(&self) -> CudaSessionStats {
        self.stats
    }

    /// Submit, release the reservation and report what the session observed.
    ///
    /// Submitting here is an explicit boundary, not a lifetime convenience: it is where a caller
    /// that queued work without calling [`Self::submit`] hands it to the device. Cleanup after
    /// this returns belongs to the substrate (CubeCL retires device allocations and staging,
    /// tenferro keeps its workspace-retirement contract), so nothing here adds a wait for that.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] when the closing submission fails, and
    /// [`crate::Error::RuntimeState`] when the runtime's own state forbids it. The
    /// reservation is released either way; `Drop` is the path that releases it without
    /// reporting.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use tenferro_gpu::cuda::CudaBackend;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut backend = CudaBackend::new(tenferro_gpu::cuda::CudaDeviceId::from_ordinal(0))?;
    /// let held = backend.open_session()?;
    /// let stats = held.close()?;
    /// assert_eq!(stats.close_submits, 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn close(mut self) -> crate::Result<CudaSessionStats> {
        let result = self.flush();
        self.stats.close_submits = self.stats.close_submits.saturating_add(1);
        result.map(|()| self.stats)
    }

    /// Dispatch pending host-side work on the captured stream.
    fn flush(&mut self) -> crate::Result<()> {
        let stream = self.stream;
        let runtime = self.backend.runtime();
        let result = stream.executes(|| runtime.flush_cubecl("CudaHeldSession"));
        if result.is_err() {
            self.stats.failed_submits = self.stats.failed_submits.saturating_add(1);
        }
        result
    }
}

impl CudaBackend {
    /// Open this backend state's held session (#1945 U3).
    ///
    /// The session borrows the backend, captures the logical CubeCL stream this thread currently
    /// executes on, and holds the state's single reservation until [`CudaHeldSession::close`] or
    /// drop. `CudaBackend` is `Clone` over one shared state, so a caller that needs its handle
    /// back after the stage can open the session on a clone.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::SessionEntryError::Reentered`] when this thread already holds
    /// this state's session, [`tenferro_tensor::SessionEntryError::Contended`] when another
    /// thread holds it, and [`tenferro_tensor::SessionEntryError::ResourcePoisoned`] when the
    /// reservation is poisoned. A thread that already holds a held session for any backend is
    /// reported the same way.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use tenferro_gpu::cuda::CudaBackend;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut backend = CudaBackend::new(tenferro_gpu::cuda::CudaDeviceId::from_ordinal(0))?;
    /// let session = backend.open_session()?;
    /// assert_eq!(session.device_ordinal(), 0);
    /// session.close()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn open_session(
        &mut self,
    ) -> Result<CudaHeldSession<'_>, tenferro_tensor::SessionEntryError> {
        let owner = self.inner.acquire_held_session()?;
        let marker = tenferro_tensor::HeldSessionMarker::enter(CUDA_BACKEND_NAME)?;
        let device_ordinal = self.inner.rt.device_ordinal();
        Ok(CudaHeldSession {
            backend: self,
            _owner: owner,
            _marker: marker,
            stream: StreamId::current(),
            device_ordinal,
            stats: CudaSessionStats::default(),
            _not_send_sync: PhantomData,
        })
    }
}
