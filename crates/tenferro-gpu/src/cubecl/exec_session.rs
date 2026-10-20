use num_complex::{Complex32, Complex64};
use tenferro_tensor::backend::{
    ElementwiseFusionPlan, ElementwiseReadOp, Engine, SessionCachedDot, TensorAnalytic,
    TensorBuffer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing, TensorReduction,
    TensorStructural,
};
use tenferro_tensor::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use tenferro_tensor::DType;
use tenferro_tensor::{DotGeneralAccumulation, Tensor, TensorRead, TensorWrite, TypedTensor};
use tenferro_tensor::{
    TensorRank, TensorScalar, TensorViewCanonicalization, TypedTensorView, TypedTensorViewMut,
};

use cubecl::prelude::{CubeElement, CubePrimitive, Scalar};

use super::identity::GpuExtensionCapability;
#[cfg(feature = "cubecl-interop")]
use super::session_cubecl;
use super::{gemm, ops, runtime::RawContextRestore};
use super::{raw, CudaDeviceInfo, CudaEngine};

#[cfg(feature = "cubecl-interop")]
/// Best-effort exit flush for a `with_cubecl` session.
///
/// Flushes once eagerly (returned to the caller as an error if it fails) and
/// once more on `Drop` so a panic/unwind path still drains pending CubeCL
/// work.
struct CubeclExitFlush<'a> {
    op: &'static str,
    client: &'a cubecl::client::Client,
    flushed: bool,
}

#[cfg(feature = "cubecl-interop")]
impl<'a> CubeclExitFlush<'a> {
    fn new(op: &'static str, client: &'a cubecl::client::Client) -> Self {
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

#[cfg(feature = "cubecl-interop")]
impl Drop for CubeclExitFlush<'_> {
    fn drop(&mut self) {
        if !self.flushed {
            let _ = self.client.flush();
        }
    }
}

/// Native-engine marker for [`CudaEngine`]; private to this crate so no other
/// crate can create a token that claims to be a CUDA engine.
pub(super) struct CudaEngineMarker;

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

impl CudaEngine {
    /// Report whether this engine supports a GPU extension capability.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::{CudaEngine, GpuExtensionCapability};
    ///
    /// // Method-call check only: opening a `CudaEngine` needs a device, so
    /// // the example asserts the method is callable from an external crate.
    /// fn check(engine: &CudaEngine, capability: GpuExtensionCapability) -> bool {
    ///     engine.supports(capability)
    /// }
    /// let _ = check;
    /// ```
    pub fn supports(&self, capability: GpuExtensionCapability) -> bool {
        self.runtime().supports_extension(capability)
    }

    /// Borrow immutable metadata for the engine's device.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaEngine;
    ///
    /// // Method-call check only: opening a `CudaEngine` needs a device, so
    /// // the example asserts the method is callable from an external crate.
    /// fn check(engine: &CudaEngine) {
    ///     let _ = engine.device_info();
    /// }
    /// let _ = check;
    /// ```
    pub fn device_info(&self) -> &CudaDeviceInfo {
        self.runtime().device_info()
    }

    /// Return the allocation ownership domain of this engine.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaEngine;
    ///
    /// // Method-call check only: opening a `CudaEngine` needs a device, so
    /// // the example asserts the method is callable from an external crate.
    /// fn check(engine: &CudaEngine) -> tenferro_tensor::AllocationDomainId {
    ///     engine.allocation_domain()
    /// }
    /// let _ = check;
    /// ```
    pub fn allocation_domain(&self) -> tenferro_tensor::AllocationDomainId {
        self.runtime().allocation_domain()
    }

    /// Validate that a dense GPU tensor is resident on this exact engine:
    /// CubeCL-backed, same allocation domain, and placed on this runtime's
    /// CUDA device. Rejects host tensors, foreign-backend buffers, and
    /// foreign-runtime/device tensors without an implicit transfer.
    ///
    /// This is the credentialed public-seam residency guard for extension
    /// crates that receive an engine but must validate inputs before entering
    /// a `with_raw`/`with_cubecl` sub-session.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when the tensor is not resident
    /// on this exact engine runtime/device.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaEngine;
    ///
    /// // Method-call check only: opening a `CudaEngine` needs a device.
    /// fn check(engine: &CudaEngine, tensor: &tenferro_tensor::Tensor) -> tenferro_tensor::Result<()> {
    ///     engine.ensure_gpu_resident(tensor, "test.ensure_gpu_resident")
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

    /// Block the host until work enqueued on the engine's stream completes.
    ///
    /// This is the only host barrier on the success path; ordinary successful
    /// engine operations only enqueue.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] when CUDA stream
    /// synchronization fails.
    pub fn synchronize(&mut self) -> crate::Result<()> {
        let stream = self.stream;
        let result = stream.executes(|| self.runtime().synchronize());
        let retired = self.retire_pending_runs();
        result.and(retired)
    }

    /// Hand queued work to the GPU without waiting for it.
    ///
    /// Dispatching may still block briefly while CubeCL's server thread takes
    /// the queued work and retires staging buffers whose transfers finished.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] when CubeCL cannot dispatch the
    /// queued work.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaEngine;
    ///
    /// let _flush: fn(&mut CudaEngine) -> tenferro_tensor::Result<()> = CudaEngine::flush;
    /// ```
    pub fn flush(&mut self) -> crate::Result<()> {
        let stream = self.stream;
        stream.executes(|| self.runtime().flush_cubecl("CudaEngine::flush"))
    }

    /// Borrow the type-safe raw CUDA extension session for one operation.
    ///
    /// The enter/exit protocol is fully contained in this call: a definite
    /// CubeCL stream is captured on the current thread, pending CubeCL work is
    /// dispatched to that stream (host task queue drained, no device wait), the
    /// calling thread's previous device/context is saved, the tenferro primary
    /// context is activated, the callback runs, and the previous device/context
    /// is best-effort restored on return, `Err`, or unwind (restoration
    /// failures are logged to stderr). The success path does not synchronize,
    /// and **entering a raw session is not a host barrier**: an empty scope
    /// returns in microseconds even behind queued device work, so a caller can
    /// interleave raw kernels and vendor calls with eager operations instead of
    /// stalling the pipeline. [`Session::synchronize`](crate::cuda::CudaEngine::synchronize)
    /// and downloads stay the explicit host barriers.
    ///
    /// A `#[cube]` launch issued **inside** the callback is not ordered before
    /// the raw work that follows it: CubeCL submits launches from its device
    /// thread while raw kernels are enqueued from this thread, so a raw kernel
    /// can reach the shared stream first and read the value the launch was about
    /// to overwrite. Call
    /// [`raw::Session::flush_cubecl`](crate::cuda::raw::Session::flush_cubecl)
    /// after the launch and before the dependent raw work; it dispatches the
    /// queued launches to the stream and reads no result back (tenferro-rs
    /// #2092).
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] when CubeCL cannot expose the
    /// stream or report the errors of work already submitted, or when the CUDA
    /// context cannot be entered. Context restoration on exit is best-effort: a
    /// failure to restore the caller's previous device/context is logged to
    /// stderr rather than propagated, so a callback result is never replaced by
    /// a restore error.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaEngine;
    ///
    /// // Method-call check only: opening a `CudaEngine` needs a device.
    /// fn check(engine: &mut CudaEngine) -> tenferro_tensor::Result<()> {
    ///     engine.with_raw("test.raw", |raw| {
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
        self.guarded(op, |engine| {
            let stream = engine.stream;
            stream.executes(|| engine.with_raw_on_stream(op, f))
        })
    }

    fn with_raw_on_stream<R>(
        &mut self,
        op: &'static str,
        f: impl for<'s> FnOnce(&mut raw::Session<'s>) -> crate::Result<R>,
    ) -> crate::Result<R> {
        let runtime = self.runtime().clone();
        let cache = self.cuda_extension_cache();
        // 1. Capture the engine's CubeCL stream (installed by the caller).
        let stream = runtime.raw_cuda_stream()?;
        // 2. Dispatch pending CubeCL work so raw library calls observe it,
        //    without arming a fence the next submission would synchronize:
        //    entering a raw scope must not become a host barrier (tenferro-rs
        //    #2078). The host task queue is drained either way.
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

    #[cfg(feature = "cubecl-interop")]
    /// Borrow the public tenferro-wide CubeCL session for one operation.
    ///
    /// The session exposes the exact tenferro CubeCL client bound to this
    /// runtime. Pending CubeCL work is flushed on exit (including `Err` and
    /// unwind) so a later raw-session or host read observes the enqueued work.
    /// Entry does not flush: same-stream CubeCL work is already ordered by the
    /// backend queue, and an operation that needs the queued launches drained
    /// before a raw enqueue (the compact zero fill) drains them itself, so the
    /// entry flush was a redundant blocking device-thread round trip (#1887).
    /// The success path does not synchronize.
    ///
    /// # Externally owned CUDA memory
    ///
    /// This CubeCL session **cannot bind memory it does not own**. Its binding
    /// constructors (`tensor_binding`, `array_arg`, and the allocation helpers)
    /// take a tenferro `TypedTensor` resident on this runtime and derive a CubeCL
    /// allocation handle from it; CubeCL launch arguments are handles into its
    /// own pool, not raw addresses, so a `cudarc` allocation has no handle to
    /// bind here and there is no upload-free interop path through `with_cubecl`.
    ///
    /// The supported interop is **raw-only**: enter
    /// [`with_raw`](Self::with_raw) on the same runtime/device, wrap the external
    /// address with
    /// [`raw::Session::tensor_ref_from_raw`](crate::cuda::raw::Session::tensor_ref_from_raw)
    /// or
    /// [`raw::Session::tensor_mut_from_raw`](crate::cuda::raw::Session::tensor_mut_from_raw),
    /// and launch a raw module with [`raw::Session::launch`](crate::cuda::raw::Session::launch).
    /// That path requires the external allocation to belong to this runtime's
    /// primary CUDA context and to be ordered on the engine's captured stream;
    /// the full contract is on the constructors. A downstream that needs a CubeCL
    /// binding must re-host the state in a tenferro tensor instead (uploading
    /// once, not per kernel).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaEngine;
    ///
    /// fn check(engine: &mut CudaEngine) {
    ///     let _ = engine.with_cubecl("test.cubecl", |_cubecl| Ok(()));
    /// }
    /// let _ = check;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns the callback's error, or [`crate::Error::BackendSource`] when
    /// pending CubeCL work cannot be flushed on exit.
    pub fn with_cubecl<R>(
        &mut self,
        op: &'static str,
        f: impl for<'s> FnOnce(&session_cubecl::Session<'s>) -> crate::Result<R>,
    ) -> crate::Result<R> {
        self.guarded(op, |engine| {
            let stream = engine.stream;
            let runtime = engine.runtime().clone();
            stream.executes(|| Self::with_cubecl_on_stream(runtime, op, f))
        })
    }

    /// Lower-triangular part of `input` (diagonal offset `k`) on this engine's
    /// stream; used by first-party linalg kernels.
    #[doc(hidden)]
    pub fn linalg_tril_typed<T>(
        &mut self,
        input: &TypedTensor<T>,
        k: i64,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + TensorScalar + CubePrimitive + Scalar + Clone,
    {
        let stream = self.stream;
        stream.executes(|| self.tril_typed(input, k))
    }

    /// Upper-triangular part of `input` (diagonal offset `k`) on this engine's
    /// stream; used by first-party linalg kernels.
    #[doc(hidden)]
    pub fn linalg_triu_typed<T>(
        &mut self,
        input: &TypedTensor<T>,
        k: i64,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + TensorScalar + CubePrimitive + Scalar + Clone,
    {
        let stream = self.stream;
        stream.executes(|| self.triu_typed(input, k))
    }

    /// Slice of `input` on this engine's stream; used by first-party linalg
    /// kernels.
    #[doc(hidden)]
    pub fn linalg_slice_typed<T>(
        &mut self,
        input: &TypedTensor<T>,
        config: &SliceConfig,
    ) -> crate::Result<TypedTensor<T>>
    where
        T: CubeElement + TensorScalar + CubePrimitive + Scalar + Clone,
    {
        let stream = self.stream;
        stream.executes(|| self.slice_typed(input, config))
    }

    #[cfg(feature = "cubecl-interop")]
    fn with_cubecl_on_stream<R>(
        runtime: super::CudaRuntime,
        op: &'static str,
        f: impl for<'s> FnOnce(&session_cubecl::Session<'s>) -> crate::Result<R>,
    ) -> crate::Result<R> {
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
}

// Typed view canonicalization runs on the engine stream.
macro_rules! impl_session_view_canonicalization {
    ($to_contiguous:ident; $($ty:ty),* $(,)?) => {
        $(
            impl<R> TensorViewCanonicalization<$ty, R> for CudaEngine
            where
                R: TensorRank,
            {
                fn to_contiguous(
                    &mut self,
                    view: &TypedTensorView<'_, $ty, R>,
                ) -> crate::Result<TypedTensor<$ty, R>> {
                    self.guarded("CudaEngine::to_contiguous", |engine| {
                        let stream = engine.stream;
                        stream.executes(|| CudaEngine::$to_contiguous(engine, view, "CudaEngine::to_contiguous"))
                    })
                }

                fn copy_into(
                    &mut self,
                    src: &TypedTensorView<'_, $ty, R>,
                    dst: &mut TypedTensorViewMut<'_, $ty, R>,
                ) -> crate::Result<()> {
                    self.guarded("CudaEngine::copy_into", |engine| {
                        let stream = engine.stream;
                        stream.executes(|| {
                            CudaEngine::copy_view_to_view_typed(engine, src, dst, "CudaEngine::copy_into")
                        })
                    })
                }
            }
        )*
    };
}

impl_session_view_canonicalization!(
    to_contiguous_view_cutensor_or_cubecl; f32, f64, Complex32, Complex64
);
impl_session_view_canonicalization!(to_contiguous_view_typed; i32, i64);

impl<R> TensorViewCanonicalization<bool, R> for CudaEngine
where
    R: TensorRank,
{
    fn to_contiguous(
        &mut self,
        _view: &TypedTensorView<'_, bool, R>,
    ) -> crate::Result<TypedTensor<bool, R>> {
        Err(super::error::unsupported_dtype(
            "CudaEngine::to_contiguous",
            crate::DType::Bool,
        ))
    }

    fn copy_into(
        &mut self,
        _src: &TypedTensorView<'_, bool, R>,
        _dst: &mut TypedTensorViewMut<'_, bool, R>,
    ) -> crate::Result<()> {
        Err(super::error::unsupported_dtype(
            "CudaEngine::copy_into",
            crate::DType::Bool,
        ))
    }
}

/// Visit a [`CudaEngine`] through the backend-neutral [`Engine`] surface.
///
/// Returns `None` when `engine` is not a CUDA engine.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::cuda::with_cuda_engine;
///
/// fn check(engine: &mut dyn tenferro_tensor::Engine) {
///     let _ = with_cuda_engine(engine, |_cuda| 0usize);
/// }
/// let _ = check;
/// ```
pub fn with_cuda_engine<B, R>(engine: &mut B, f: impl FnOnce(&mut CudaEngine) -> R) -> Option<R>
where
    B: Engine + ?Sized,
{
    let data = engine
        .native_engine()?
        .into_marked_ptr::<CudaEngineMarker>()?;
    // SAFETY: only `CudaEngine::native_engine` creates a token with the
    // crate-private `CudaEngineMarker`, and it points that token at a live
    // `CudaEngine`. The token borrowed `*engine` exclusively, and this function
    // keeps holding `engine: &mut B` for the whole scoped visit.
    Some(f(unsafe { data.cast::<CudaEngine>().as_mut() }))
}

macro_rules! delegate_ops {
    ($trait:path {
        $(fn $method:ident($($arg:ident: $arg_ty:ty),* $(,)?) -> $ret:ty;)*
    } $(override { $($custom:item)* })?) => {
        impl $trait for CudaEngine {
            $(
                fn $method(&mut self, $($arg: $arg_ty),*) -> $ret {
                    self.guarded(stringify!($method), |engine| {
                        let stream = engine.stream;
                        stream.executes(|| ops::$method(engine, $($arg),*))
                    })
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
        let native = self.guarded(op.label(), |engine| {
            let stream = engine.stream;
            Ok(stream.executes(|| engine.elementwise_read_into_native(op, inputs, &mut out)))
        })?;
        if let Some(result) = native {
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
impl TensorBuffer for CudaEngine {}

impl SessionCachedDot for CudaEngine {
    // Read-based cached dot paths keep strided operands on device; the plan
    // cache is per backend, so the runtime cache slot stays unused.
    fn dot_general_read_cached(
        &mut self,
        _cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        self.guarded("dot_general", |engine| {
            let stream = engine.stream;
            stream.executes(|| {
                gemm::dot_general_read_allocating(engine, lhs, rhs, config, false, false)
            })
        })
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
        self.guarded("dot_general", |engine| {
            let stream = engine.stream;
            stream.executes(|| {
                gemm::dot_general_read_allocating(engine, lhs, rhs, config, lhs_conj, rhs_conj)
            })
        })
    }
}

impl Engine for CudaEngine {
    fn vdot_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.guarded("vdot", |engine| {
            let stream = engine.stream;
            stream.executes(|| ops::vdot_read(engine, lhs, rhs))
        })
    }

    fn norm_squared_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.guarded("norm_squared", |engine| {
            let stream = engine.stream;
            stream.executes(|| ops::norm_squared_read(engine, input))
        })
    }

    fn axpby_read_into_accum(
        &mut self,
        alpha: tenferro_tensor::ContractionScalar,
        x: TensorRead<'_>,
        beta: tenferro_tensor::ContractionScalar,
        y: TensorWrite<'_>,
    ) -> crate::Result<()> {
        self.guarded("axpby", |engine| {
            let stream = engine.stream;
            stream.executes(|| ops::axpby_read_into_accum(engine, alpha, x, beta, y))
        })
    }

    fn gather_read(
        &mut self,
        operand: TensorRead<'_>,
        start_indices: TensorRead<'_>,
        config: &GatherConfig,
    ) -> crate::Result<Tensor> {
        // The operand is the large one: bind it where it is when the device can
        // read it through its own strides, instead of materializing the table
        // on every call.
        let native = self.guarded("gather", |engine| {
            let stream = engine.stream;
            stream.executes(|| ops::gather_read(engine, &operand, &start_indices, config))
        })?;
        if let Some(output) = native {
            return Ok(output);
        }
        let operand = self.to_contiguous_read(operand)?;
        let start_indices = self.to_contiguous_read(start_indices)?;
        self.gather(&operand, &start_indices, config)
    }

    fn native_engine(&mut self) -> Option<tenferro_tensor::NativeSessionRef<'_>> {
        // SAFETY: `CudaEngineMarker` is private to this crate, and this is the only
        // place a token carrying it is created; it always points to a
        // `CudaEngine`, exclusively borrowed for the token lifetime.
        Some(unsafe { tenferro_tensor::NativeSessionRef::new::<CudaEngineMarker, _>(self) })
    }
}
