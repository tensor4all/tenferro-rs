use tenferro_cpu::CpuBackend;
#[cfg(feature = "cuda")]
use tenferro_gpu::cubecl::CubeclBackend;
use tenferro_tensor::{
    BackendCachedDot, BackendRuntimeCache, BackendSession, BackendSessionHost, CompareDir, DType,
    DotGeneralConfig, ElementwiseFusionPlan, GatherConfig, PadConfig, Result as TensorResult,
    ScatterConfig, SliceConfig, Tensor, TensorAnalytic, TensorBackend, TensorBuffer,
    TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing, TensorRead,
    TensorReduction, TensorStructural,
};

pub enum EagerBackend {
    Cpu(CpuBackend),
    #[cfg(feature = "cuda")]
    Cuda(CubeclBackend),
}

impl EagerBackend {
    pub(crate) fn cpu(backend: CpuBackend) -> Self {
        Self::Cpu(backend)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda(backend: CubeclBackend) -> Self {
        Self::Cuda(backend)
    }

    pub(crate) fn synchronize(&mut self) -> TensorResult<()> {
        match self {
            Self::Cpu(_) => Ok(()),
            #[cfg(feature = "cuda")]
            Self::Cuda(backend) => backend.runtime().synchronize(),
        }
    }
}

macro_rules! dispatch {
    ($backend:expr, $method:ident($($arg:expr),* $(,)?)) => {
        match $backend {
            EagerBackend::Cpu(backend) => backend.$method($($arg),*),
            #[cfg(feature = "cuda")]
            EagerBackend::Cuda(backend) => backend.$method($($arg),*),
        }
    };
}

macro_rules! delegate_tensor_backend_methods {
    ($(fn $method:ident($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty;)*) => {
        $(
            fn $method(&mut self, $($arg: $ty),*) -> $ret {
                dispatch!(self, $method($($arg),*))
            }
        )*
    };
}

impl BackendRuntimeCache for EagerBackend {
    type RuntimeCache = ();
}

impl TensorElementwise for EagerBackend {
    delegate_tensor_backend_methods! {
        fn add(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn add_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor>;
        fn mul(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn mul_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor>;
        fn neg(input: &Tensor) -> TensorResult<Tensor>;
        fn neg_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn conj(input: &Tensor) -> TensorResult<Tensor>;
        fn conj_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn div(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn div_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor>;
        fn abs(input: &Tensor) -> TensorResult<Tensor>;
        fn abs_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn sign(input: &Tensor) -> TensorResult<Tensor>;
        fn sign_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn maximum(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn maximum_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor>;
        fn minimum(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn minimum_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor>;
        fn compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> TensorResult<Tensor>;
        fn compare_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>, dir: &CompareDir) -> TensorResult<Tensor>;
        fn select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> TensorResult<Tensor>;
        fn select_read(pred: TensorRead<'_>, on_true: TensorRead<'_>, on_false: TensorRead<'_>) -> TensorResult<Tensor>;
        fn clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) -> TensorResult<Tensor>;
        fn clamp_read(input: TensorRead<'_>, lower: TensorRead<'_>, upper: TensorRead<'_>) -> TensorResult<Tensor>;
    }
}

impl TensorAnalytic for EagerBackend {
    delegate_tensor_backend_methods! {
        fn exp(input: &Tensor) -> TensorResult<Tensor>;
        fn exp_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn log(input: &Tensor) -> TensorResult<Tensor>;
        fn log_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn sin(input: &Tensor) -> TensorResult<Tensor>;
        fn sin_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn cos(input: &Tensor) -> TensorResult<Tensor>;
        fn cos_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn tanh(input: &Tensor) -> TensorResult<Tensor>;
        fn tanh_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn sqrt(input: &Tensor) -> TensorResult<Tensor>;
        fn sqrt_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn rsqrt(input: &Tensor) -> TensorResult<Tensor>;
        fn rsqrt_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn pow(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn pow_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor>;
        fn expm1(input: &Tensor) -> TensorResult<Tensor>;
        fn expm1_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn log1p(input: &Tensor) -> TensorResult<Tensor>;
        fn log1p_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
    }
}

impl TensorStructural for EagerBackend {
    delegate_tensor_backend_methods! {
        fn transpose(input: &Tensor, perm: &[usize]) -> TensorResult<Tensor>;
        fn reshape(input: &Tensor, shape: &[usize]) -> TensorResult<Tensor>;
        fn reshape_read(input: TensorRead<'_>, shape: &[usize]) -> TensorResult<Tensor>;
        fn broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) -> TensorResult<Tensor>;
        fn broadcast_in_dim_read(input: TensorRead<'_>, shape: &[usize], dims: &[usize]) -> TensorResult<Tensor>;
        fn convert(input: &Tensor, to: DType) -> TensorResult<Tensor>;
        fn extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> TensorResult<Tensor>;
        fn embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> TensorResult<Tensor>;
        fn tril(input: &Tensor, k: i64) -> TensorResult<Tensor>;
        fn triu(input: &Tensor, k: i64) -> TensorResult<Tensor>;
    }
}

impl TensorReduction for EagerBackend {
    delegate_tensor_backend_methods! {
        fn reduce_sum(input: &Tensor, axes: &[usize]) -> TensorResult<Tensor>;
        fn reduce_prod(input: &Tensor, axes: &[usize]) -> TensorResult<Tensor>;
        fn reduce_max(input: &Tensor, axes: &[usize]) -> TensorResult<Tensor>;
        fn reduce_min(input: &Tensor, axes: &[usize]) -> TensorResult<Tensor>;
    }
}

impl TensorDot for EagerBackend {
    delegate_tensor_backend_methods! {
        fn dot_general(lhs: &Tensor, rhs: &Tensor, config: &DotGeneralConfig) -> TensorResult<Tensor>;
        fn dot_general_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>, config: &DotGeneralConfig) -> TensorResult<Tensor>;
        fn dot_general_with_conj(lhs: &Tensor, rhs: &Tensor, config: &DotGeneralConfig, lhs_conj: bool, rhs_conj: bool) -> TensorResult<Tensor>;
    }
}

impl TensorIndexing for EagerBackend {
    delegate_tensor_backend_methods! {
        fn gather(operand: &Tensor, start_indices: &Tensor, config: &GatherConfig) -> TensorResult<Tensor>;
        fn scatter(operand: &Tensor, scatter_indices: &Tensor, updates: &Tensor, config: &ScatterConfig) -> TensorResult<Tensor>;
        fn slice(input: &Tensor, config: &SliceConfig) -> TensorResult<Tensor>;
        fn dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) -> TensorResult<Tensor>;
        fn dynamic_update_slice(operand: &Tensor, update: &Tensor, starts: &Tensor) -> TensorResult<Tensor>;
        fn pad(input: &Tensor, config: &PadConfig) -> TensorResult<Tensor>;
        fn concatenate(inputs: &[&Tensor], axis: usize) -> TensorResult<Tensor>;
        fn reverse(input: &Tensor, axes: &[usize]) -> TensorResult<Tensor>;
    }
}

impl BackendSessionHost for EagerBackend {
    fn with_backend_session<R>(&mut self, f: impl FnOnce(&mut dyn BackendSession) -> R) -> R {
        dispatch!(self, with_backend_session(f))
    }
}

impl TensorDeviceTransfer for EagerBackend {
    delegate_tensor_backend_methods! {
        fn download_to_host(tensor: &Tensor) -> TensorResult<Tensor>;
        fn upload_host_tensor(tensor: &Tensor) -> TensorResult<Tensor>;
    }
}

impl TensorBuffer for EagerBackend {
    delegate_tensor_backend_methods! {
        fn reclaim_buffer(tensor: Tensor) -> ();
    }
}

impl TensorFusion for EagerBackend {
    delegate_tensor_backend_methods! {
        fn execute_elementwise_fusion(inputs: &[&Tensor], plan: &ElementwiseFusionPlan) -> TensorResult<Option<Vec<Tensor>>>;
    }
}

impl BackendCachedDot for EagerBackend {}

impl TensorBackend for EagerBackend {}
