use cubecl::prelude::{CubeElement, CubePrimitive};
use num_complex::{Complex32, Complex64};
use std::any::TypeId;
use std::marker::PhantomData;
use std::rc::Rc;
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
    with_session_entry_guard, TensorRank, TensorScalar, TensorViewCanonicalization, TypedTensorView,
};
use tenferro_tensor::{DotGeneralAccumulation, Tensor, TensorRead, TensorWrite, TypedTensor};

use super::identity::GpuExtensionCapability;
use super::{gemm, ops, runtime::RawContextRestore};
use super::{
    raw, session_cubecl, CudaBackend, CudaDeviceInfo, CudaExtensionCache, CudaRuntime,
    CudaRuntimeIdentity,
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

/// Marker for the concrete erased CUDA execution-session target.
#[doc(hidden)]
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

    #[doc(hidden)]
    pub fn to_contiguous<T, R>(
        &mut self,
        view: &TypedTensorView<'_, T, R>,
    ) -> crate::Result<TypedTensor<T, R>>
    where
        T: TensorScalar,
        R: TensorRank,
        CudaBackend: TensorViewCanonicalization<T, R>,
    {
        self.backend.to_contiguous(view)
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
    if session.session_type_id() != std::any::TypeId::of::<CudaExecSessionMarker>() {
        return None;
    }
    let data = unsafe { session.session_data_mut() };
    // SAFETY: the exact marker check and the BackendSession erased-pointer
    // contract identify the value as CudaExecSession for this scoped visit.
    Some(unsafe { f(&mut *(data.cast::<CudaExecSession<'static>>())) })
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

delegate!(TensorBuffer {
    fn reclaim_buffer(tensor: Tensor) -> ();
});

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

    fn session_type_id(&self) -> TypeId {
        TypeId::of::<CudaExecSessionMarker>()
    }

    unsafe fn session_data_mut(&mut self) -> *mut () {
        self as *mut Self as *mut ()
    }
}

impl BackendSessionHost for CudaBackend {
    fn with_backend_session<R: Send>(
        &mut self,
        f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
    ) -> R {
        let mut session = CudaExecSession {
            backend: self,
            _not_send_sync: PhantomData,
        };
        // Nested entry is caught by the portable in-session guard in debug
        // builds; the CUDA runtime must never re-enter a session closure.
        with_session_entry_guard(|| f(&mut session))
    }
}
