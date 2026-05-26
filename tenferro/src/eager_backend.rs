#[cfg(feature = "cuda")]
use tenferro_gpu::cubecl::CubeclBackend;
use tenferro_tensor::cpu::CpuBackend;
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, ElementwiseFusionPlan, GatherConfig, PadConfig,
    Result as TensorResult, ScatterConfig, SliceConfig, Tensor, TensorBackend, TensorExec,
    TensorRead,
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

impl TensorBackend for EagerBackend {
    type RuntimeCache = ();

    delegate_tensor_backend_methods! {
        fn add(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn mul(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn neg(input: &Tensor) -> TensorResult<Tensor>;
        fn conj(input: &Tensor) -> TensorResult<Tensor>;
        fn div(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn abs(input: &Tensor) -> TensorResult<Tensor>;
        fn sign(input: &Tensor) -> TensorResult<Tensor>;
        fn maximum(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn minimum(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> TensorResult<Tensor>;
        fn select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> TensorResult<Tensor>;
        fn clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) -> TensorResult<Tensor>;
        fn exp(input: &Tensor) -> TensorResult<Tensor>;
        fn log(input: &Tensor) -> TensorResult<Tensor>;
        fn sin(input: &Tensor) -> TensorResult<Tensor>;
        fn cos(input: &Tensor) -> TensorResult<Tensor>;
        fn tanh(input: &Tensor) -> TensorResult<Tensor>;
        fn sqrt(input: &Tensor) -> TensorResult<Tensor>;
        fn rsqrt(input: &Tensor) -> TensorResult<Tensor>;
        fn pow(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn expm1(input: &Tensor) -> TensorResult<Tensor>;
        fn log1p(input: &Tensor) -> TensorResult<Tensor>;
        fn transpose(input: &Tensor, perm: &[usize]) -> TensorResult<Tensor>;
        fn reshape(input: &Tensor, shape: &[usize]) -> TensorResult<Tensor>;
        fn broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) -> TensorResult<Tensor>;
        fn convert(input: &Tensor, to: DType) -> TensorResult<Tensor>;
        fn extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> TensorResult<Tensor>;
        fn embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> TensorResult<Tensor>;
        fn tril(input: &Tensor, k: i64) -> TensorResult<Tensor>;
        fn triu(input: &Tensor, k: i64) -> TensorResult<Tensor>;
        fn reduce_sum(input: &Tensor, axes: &[usize]) -> TensorResult<Tensor>;
        fn reduce_prod(input: &Tensor, axes: &[usize]) -> TensorResult<Tensor>;
        fn reduce_max(input: &Tensor, axes: &[usize]) -> TensorResult<Tensor>;
        fn reduce_min(input: &Tensor, axes: &[usize]) -> TensorResult<Tensor>;
        fn dot_general(lhs: &Tensor, rhs: &Tensor, config: &DotGeneralConfig) -> TensorResult<Tensor>;
        fn dot_general_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>, config: &DotGeneralConfig) -> TensorResult<Tensor>;
        fn dot_general_with_conj(lhs: &Tensor, rhs: &Tensor, config: &DotGeneralConfig, lhs_conj: bool, rhs_conj: bool) -> TensorResult<Tensor>;
        fn gather(operand: &Tensor, start_indices: &Tensor, config: &GatherConfig) -> TensorResult<Tensor>;
        fn scatter(operand: &Tensor, scatter_indices: &Tensor, updates: &Tensor, config: &ScatterConfig) -> TensorResult<Tensor>;
        fn slice(input: &Tensor, config: &SliceConfig) -> TensorResult<Tensor>;
        fn dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) -> TensorResult<Tensor>;
        fn dynamic_update_slice(operand: &Tensor, update: &Tensor, starts: &Tensor) -> TensorResult<Tensor>;
        fn pad(input: &Tensor, config: &PadConfig) -> TensorResult<Tensor>;
        fn concatenate(inputs: &[&Tensor], axis: usize) -> TensorResult<Tensor>;
        fn reverse(input: &Tensor, axes: &[usize]) -> TensorResult<Tensor>;
        fn cholesky(input: &Tensor) -> TensorResult<Tensor>;
        fn triangular_solve(a: &Tensor, b: &Tensor, left_side: bool, lower: bool, transpose_a: bool, unit_diagonal: bool) -> TensorResult<Tensor>;
        fn lu(input: &Tensor) -> TensorResult<Vec<Tensor>>;
        fn full_piv_lu(input: &Tensor) -> TensorResult<Vec<Tensor>>;
        fn full_piv_lu_solve(a: &Tensor, b: &Tensor, transpose_a: bool) -> TensorResult<Tensor>;
        fn svd(input: &Tensor) -> TensorResult<Vec<Tensor>>;
        fn qr(input: &Tensor) -> TensorResult<Vec<Tensor>>;
        fn eigh(input: &Tensor) -> TensorResult<Vec<Tensor>>;
        fn eig(input: &Tensor) -> TensorResult<Vec<Tensor>>;
        fn solve(a: &Tensor, b: &Tensor) -> TensorResult<Tensor>;
    }

    fn with_exec_session<R: Send>(&mut self, f: impl FnOnce(&mut dyn TensorExec) -> R + Send) -> R {
        dispatch!(self, with_exec_session(f))
    }

    delegate_tensor_backend_methods! {
        fn download_to_host(tensor: &Tensor) -> TensorResult<Tensor>;
        fn upload_host_tensor(tensor: &Tensor) -> TensorResult<Tensor>;
        fn reclaim_buffer(tensor: Tensor) -> ();
        fn execute_elementwise_fusion(inputs: &[&Tensor], plan: &ElementwiseFusionPlan) -> TensorResult<Option<Vec<Tensor>>>;
    }
}
