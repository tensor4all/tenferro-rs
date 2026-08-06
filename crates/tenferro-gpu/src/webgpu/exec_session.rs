use std::any::TypeId;
use tenferro_tensor::backend::{
    BackendSession, BackendSessionHost, ElementwiseFusionPlan, GroupedGemmConfig, SessionCachedDot,
    TensorAnalytic, TensorBuffer, TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion,
    TensorIndexing, TensorReduction, TensorStructural,
};
use tenferro_tensor::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use tenferro_tensor::{DotGeneralAccumulation, Tensor, TensorRead, TensorValue, TensorWrite};

use super::{WebGpuBackend, WebGpuRuntime, WebGpuRuntimeIdentity};

/// Marker for the concrete erased WebGPU execution-session target.
#[doc(hidden)]
pub(super) struct WebGpuExecSessionMarker;

/// Borrowed WebGPU execution capability.
#[doc(hidden)]
#[derive(Debug)]
pub struct WebGpuExecSession<'a> {
    backend: &'a mut WebGpuBackend,
}

impl WebGpuExecSession<'_> {
    /// Borrow the provider runtime without exposing the owning backend.
    #[doc(hidden)]
    pub fn runtime(&self) -> &WebGpuRuntime {
        self.backend.runtime()
    }

    /// Return the identity of the borrowed provider runtime.
    #[doc(hidden)]
    pub fn runtime_identity(&self) -> WebGpuRuntimeIdentity {
        self.backend.runtime_identity()
    }
}

/// Visit a WebGPU execution session through the erased backend-session surface.
///
/// The callback receives only the lifetime-bound session capability. The
/// owning [`WebGpuBackend`] never crosses this boundary.
#[doc(hidden)]
pub fn with_webgpu_exec_session<B, R>(
    session: &mut B,
    f: impl for<'a> FnOnce(&'a mut WebGpuExecSession<'a>) -> R,
) -> Option<R>
where
    B: BackendSession + ?Sized,
{
    if session.session_type_id() != std::any::TypeId::of::<WebGpuExecSessionMarker>() {
        return None;
    }
    let data = unsafe { session.session_data_mut() };
    // SAFETY: the exact marker check and BackendSession erased-pointer contract
    // identify the value as WebGpuExecSession for this scoped visit.
    Some(unsafe { f(&mut *(data.cast::<WebGpuExecSession<'static>>())) })
}

macro_rules! delegate {
    ($trait:path {
        $(fn $method:ident($($arg:ident: $arg_ty:ty),* $(,)?) -> $ret:ty;)*
    }) => {
        impl $trait for WebGpuExecSession<'_> {
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
        impl SessionCachedDot for WebGpuExecSession<'_> {
            $(
                fn $method(&mut self, $($arg: $arg_ty),*) -> $ret {
                    <WebGpuBackend as SessionCachedDot>::$method(self.backend, $($arg),*)
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

impl BackendSession for WebGpuExecSession<'_> {
    fn session_type_id(&self) -> TypeId {
        TypeId::of::<WebGpuExecSessionMarker>()
    }

    unsafe fn session_data_mut(&mut self) -> *mut () {
        self as *mut Self as *mut ()
    }
}

impl BackendSessionHost for WebGpuBackend {
    fn with_backend_session<R: Send>(
        &mut self,
        f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
    ) -> R {
        let mut session = WebGpuExecSession { backend: self };
        f(&mut session)
    }
}
