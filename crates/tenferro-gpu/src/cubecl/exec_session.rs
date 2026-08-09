use cubecl::prelude::{CubeElement, CubePrimitive};
use std::any::TypeId;
use tenferro_tensor::backend::{
    BackendSession, BackendSessionHost, ElementwiseFusionPlan, GroupedGemmConfig, SessionCachedDot,
    TensorAnalytic, TensorBuffer, TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion,
    TensorIndexing, TensorReduction, TensorStructural,
};
use tenferro_tensor::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use tenferro_tensor::{
    DotGeneralAccumulation, Tensor, TensorRead, TensorValue, TensorWrite, TypedTensor,
};
use tenferro_tensor::{TensorRank, TensorScalar, TensorViewCanonicalization, TypedTensorView};

use super::identity::GpuExtensionCapability;
use super::{CudaBackend, CudaDeviceInfo, CudaExtensionCache, CudaRuntime, CudaRuntimeIdentity};

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
#[derive(Debug)]
pub struct CudaExecSession<'a> {
    backend: &'a mut CudaBackend,
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

    /// Block the host until work enqueued on the session's stream completes.
    ///
    /// This is the only host barrier on the success path; ordinary successful
    /// session operations only enqueue. `with_cubecl` and `with_raw` (the
    /// extension sub-sessions) are added by the follow-up extension-session
    /// work; this task establishes the public boundary and the
    /// capability/identity surface only.
    pub fn synchronize(&mut self) -> crate::Result<()> {
        self.backend.runtime().synchronize()
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

delegate!(TensorElementwise {
    fn add(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn sub(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn mul(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn neg(input: &Tensor) -> crate::Result<Tensor>;
    fn conj(input: &Tensor) -> crate::Result<Tensor>;
    fn div(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn rem(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn abs(input: &Tensor) -> crate::Result<Tensor>;
    fn sign(input: &Tensor) -> crate::Result<Tensor>;
    fn maximum(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn minimum(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> crate::Result<Tensor>;
    fn select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> crate::Result<Tensor>;
    fn clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) -> crate::Result<Tensor>;
});

delegate!(TensorAnalytic {
    fn exp(input: &Tensor) -> crate::Result<Tensor>;
    fn log(input: &Tensor) -> crate::Result<Tensor>;
    fn sin(input: &Tensor) -> crate::Result<Tensor>;
    fn cos(input: &Tensor) -> crate::Result<Tensor>;
    fn tanh(input: &Tensor) -> crate::Result<Tensor>;
    fn sqrt(input: &Tensor) -> crate::Result<Tensor>;
    fn rsqrt(input: &Tensor) -> crate::Result<Tensor>;
    fn pow(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn expm1(input: &Tensor) -> crate::Result<Tensor>;
    fn log1p(input: &Tensor) -> crate::Result<Tensor>;
});

delegate!(TensorStructural {
    fn to_contiguous_read(input: TensorRead<'_>) -> crate::Result<Tensor>;
    fn copy_read_into(src: TensorRead<'_>, dst: TensorWrite<'_>) -> crate::Result<()>;
    fn transpose(input: &Tensor, perm: &[usize]) -> crate::Result<Tensor>;
    fn reshape(input: &Tensor, shape: &[usize]) -> crate::Result<Tensor>;
    fn broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) -> crate::Result<Tensor>;
    fn cast(input: &Tensor, to: tenferro_tensor::DType) -> crate::Result<Tensor>;
    fn extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> crate::Result<Tensor>;
    fn embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> crate::Result<Tensor>;
    fn tril(input: &Tensor, k: i64) -> crate::Result<Tensor>;
    fn triu(input: &Tensor, k: i64) -> crate::Result<Tensor>;
});

delegate!(TensorReduction {
    fn reduce_sum(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
    fn reduce_sum_squares_read(input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor>;
    fn reduce_prod(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
    fn reduce_max(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
    fn reduce_min(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
});

delegate!(TensorDot {
    fn dot_general(lhs: &Tensor, rhs: &Tensor, config: &DotGeneralConfig) -> crate::Result<Tensor>;
    fn dot_general_with_conj(
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor>;
    fn dot_general_read_into_accum(
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        accumulation: DotGeneralAccumulation,
        out: TensorWrite<'_>,
    ) -> crate::Result<()>;
});

delegate!(TensorIndexing {
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

delegate!(TensorFusion {
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
    fn execute_broadcast_multiply_value(
        lhs: TensorRead<'_>,
        lhs_shape: &[usize],
        lhs_dims: &[usize],
        rhs: TensorRead<'_>,
        rhs_shape: &[usize],
        rhs_dims: &[usize],
    ) -> crate::Result<Option<TensorValue>>;
});

delegate!(TensorBuffer {
    fn reclaim_buffer(tensor: Tensor) -> ();
});

delegate!(TensorDeviceTransfer {
    fn download_to_host(tensor: TensorRead<'_>) -> crate::Result<Tensor>;
    fn upload_host_tensor(tensor: TensorRead<'_>) -> crate::Result<Tensor>;
});

macro_rules! delegate_cached {
    ($(fn $method:ident($($arg:ident: $arg_ty:ty),* $(,)?) -> $ret:ty;)*) => {
        impl SessionCachedDot for CudaExecSession<'_> {
            $(
                fn $method(&mut self, $($arg: $arg_ty),*) -> $ret {
                    <CudaBackend as SessionCachedDot>::$method(self.backend, $($arg),*)
                }
            )*
        }
    };
}

delegate_cached! {
    fn dot_general_cached(
        cache_slot: Option<usize>,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor>;
    fn dot_general_read_cached(
        cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor>;
    fn dot_general_with_conj_cached(
        cache_slot: Option<usize>,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor>;
    fn dot_general_with_conj_read_cached(
        cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor>;
    fn dot_general_read_into_accum_cached(
        cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        accumulation: DotGeneralAccumulation,
        out: TensorWrite<'_>,
    ) -> crate::Result<()>;
    fn grouped_gemm_cached(
        cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &GroupedGemmConfig<'_>,
        out: TensorWrite<'_>,
    ) -> crate::Result<()>;
}

impl BackendSession for CudaExecSession<'_> {
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
        let mut session = CudaExecSession { backend: self };
        f(&mut session)
    }
}
