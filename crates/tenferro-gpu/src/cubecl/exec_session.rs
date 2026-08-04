use cubecl::prelude::{CubeElement, CubePrimitive};
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

use super::{CudaBackend, CudaExtensionCache, CudaRuntime, CudaRuntimeIdentity};

/// Borrowed CUDA execution capability.
#[doc(hidden)]
#[derive(Debug)]
pub struct CudaExecSession<'a> {
    backend: &'a mut CudaBackend,
}

impl CudaExecSession<'_> {
    /// Borrow the provider runtime without exposing the backend.
    #[doc(hidden)]
    pub fn runtime(&self) -> &CudaRuntime {
        self.backend.runtime()
    }

    /// Return the identity of the borrowed provider runtime.
    #[doc(hidden)]
    pub fn runtime_identity(&self) -> CudaRuntimeIdentity {
        self.backend.runtime_identity()
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
/// This is intentionally a scoped leaf-capability bridge: the callback cannot
/// return a borrow of the reconstructed session.
#[doc(hidden)]
pub fn with_cuda_exec_session<B, R>(
    session: &mut B,
    f: impl for<'a> FnOnce(&'a mut CudaExecSession<'a>) -> R,
) -> Option<R>
where
    B: BackendSession + ?Sized,
{
    if session.session_type_name() != std::any::type_name::<CudaExecSession<'static>>() {
        return None;
    }
    let data = unsafe { session.session_data_mut() };
    // SAFETY: the type-name check and the BackendSession erased-pointer
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
    fn download_to_host(tensor: &Tensor) -> crate::Result<Tensor>;
    fn upload_host_tensor(tensor: &Tensor) -> crate::Result<Tensor>;
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

impl BackendSessionHost for CudaBackend {
    fn with_backend_session<R: Send>(
        &mut self,
        f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
    ) -> R {
        let mut session = CudaExecSession { backend: self };
        f(&mut session)
    }
}
