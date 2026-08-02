#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(test)]
use std::sync::Arc;
use tenferro_cpu::CpuBackend;
#[cfg(feature = "cuda")]
use tenferro_gpu::CudaBackend;
#[cfg(feature = "webgpu")]
use tenferro_gpu::WebGpuBackend;
use tenferro_runtime::{
    EngineId, EngineRegistration, HardwareClassId, Runtime, RuntimeConfigError,
};
use tenferro_tensor::backend::ElementwiseFusionPlan;
use tenferro_tensor::{
    BackendCachedDot, BackendRuntimeCache, BackendSession, BackendSessionHost, CompareDir, DType,
    DotGeneralConfig, ElementwiseReadOp, GatherConfig, PadConfig, Result as TensorResult,
    ScatterConfig, SliceConfig, Tensor, TensorAnalytic, TensorBackend, TensorBuffer,
    TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing, TensorRead,
    TensorReduction, TensorStructural, TensorValue, TensorWrite,
};

pub(crate) enum EagerBackend {
    Cpu(CpuBackend),
    #[cfg(test)]
    Recording(RecordingBackend),
    #[cfg(feature = "cuda")]
    Cuda(CudaBackend),
    #[cfg(feature = "webgpu")]
    WebGpu(WebGpuBackend),
}

/// The closed set of default eager engine families known by this crate.
///
/// The CUDA and WebGPU variants remain available to provider-neutral planning
/// tests even when the corresponding backend feature is disabled. Concrete
/// [`EagerBackendRegistrationState`](crate::eager::EagerBackendRegistrationState)
/// variants are feature-gated with their backend types.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EagerEngineFamily {
    NoEngine,
    Cpu,
    Cuda,
    WebGpu,
}

/// Provider-neutral transaction shape for replacing eager engine registrations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum EagerRegistrationPlan {
    RemoveOnly {
        remove: Vec<EngineId>,
    },
    RemoveAndInstall {
        remove: Vec<EngineId>,
        family: EagerEngineFamily,
        target: EngineId,
    },
}

impl EagerRegistrationPlan {
    pub(crate) fn removals(&self) -> &[EngineId] {
        match self {
            Self::RemoveOnly { remove } | Self::RemoveAndInstall { remove, .. } => remove,
        }
    }
}

/// Provider-neutral target descriptor paired with the exact backend witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum EagerRegistrationTarget {
    NoEngine,
    Install {
        family: EagerEngineFamily,
        engine_id: EngineId,
    },
}

/// The fallible provider-specific registration produced from the exact eager
/// backend. `NoEngine` is used by the test-only recording backend, whose
/// tensor operations are intentionally not installed as a runtime engine.
#[allow(dead_code)]
pub(crate) enum EagerBackendRegistration {
    NoEngine,
    Install {
        family: EagerEngineFamily,
        registration: EngineRegistration,
    },
}

/// Build the one transaction plan used by all eager backend families.
///
/// `known` is the complete set of canonical eager engine IDs enabled in this
/// build, while `installed` is the subset currently present in the runtime
/// snapshot. Every known eager engine is removed before the new target is
/// installed. The helper deliberately carries no provider/device identity;
/// identity equality is handled by the registration state machine.
pub(crate) fn plan_eager_registration(
    known: &[(EagerEngineFamily, EngineId)],
    installed: &[EngineId],
    target: &EagerRegistrationTarget,
) -> EagerRegistrationPlan {
    let remove = known
        .iter()
        .filter(|(_, id)| installed.contains(id))
        .map(|(_, id)| id.clone())
        .collect();
    match target {
        EagerRegistrationTarget::NoEngine => EagerRegistrationPlan::RemoveOnly { remove },
        EagerRegistrationTarget::Install { family, engine_id } => {
            EagerRegistrationPlan::RemoveAndInstall {
                remove,
                family: *family,
                target: engine_id.clone(),
            }
        }
    }
}

impl std::fmt::Debug for EagerBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu(backend) => f.debug_tuple("Cpu").field(backend).finish(),
            #[cfg(test)]
            Self::Recording(backend) => f.debug_tuple("Recording").field(backend).finish(),
            #[cfg(feature = "cuda")]
            Self::Cuda(backend) => f.debug_tuple("Cuda").field(backend).finish(),
            #[cfg(feature = "webgpu")]
            Self::WebGpu(backend) => f.debug_tuple("WebGpu").field(backend).finish(),
        }
    }
}

impl EagerBackend {
    pub(crate) fn cpu(backend: CpuBackend) -> Self {
        Self::Cpu(backend)
    }

    pub(crate) fn cpu_snapshot(&self) -> Option<CpuBackend> {
        match self {
            Self::Cpu(backend) => Some(backend.clone()),
            #[cfg(test)]
            Self::Recording(_) => None,
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => None,
            #[cfg(feature = "webgpu")]
            Self::WebGpu(_) => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn recording_cpu(materializations: Arc<AtomicUsize>) -> Self {
        Self::Recording(RecordingBackend {
            materializations,
            inner: CpuBackend::new(),
        })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda(backend: CudaBackend) -> Self {
        Self::Cuda(backend)
    }

    #[cfg(feature = "webgpu")]
    pub(crate) fn webgpu(backend: WebGpuBackend) -> Self {
        Self::WebGpu(backend)
    }

    pub(crate) fn synchronize(&mut self) -> TensorResult<()> {
        match self {
            Self::Cpu(_) => Ok(()),
            #[cfg(test)]
            Self::Recording(_) => Ok(()),
            #[cfg(feature = "cuda")]
            Self::Cuda(backend) => backend.runtime().synchronize(),
            #[cfg(feature = "webgpu")]
            Self::WebGpu(backend) => backend.synchronize(),
        }
    }
}

pub(crate) fn eager_runtime_for_backend(
    backend: &EagerBackend,
) -> Result<Runtime, RuntimeConfigError> {
    let mut builder = Runtime::builder();
    match eager_engine_registration_for_backend(backend)? {
        EagerBackendRegistration::NoEngine => {}
        EagerBackendRegistration::Install { registration, .. } => {
            builder.register_engine(registration)?;
        }
    }
    builder.build()
}

pub(crate) fn eager_default_engine_families(
) -> Result<Vec<(EagerEngineFamily, EngineId)>, RuntimeConfigError> {
    let mut known = Vec::with_capacity(3);
    known.push((EagerEngineFamily::Cpu, cpu_runtime_engine_id()?));
    #[cfg(feature = "cuda")]
    known.push((
        EagerEngineFamily::Cuda,
        tenferro_gpu::cuda_runtime_engine_id()?,
    ));
    #[cfg(feature = "webgpu")]
    known.push((
        EagerEngineFamily::WebGpu,
        tenferro_gpu::webgpu_runtime_engine_id()?,
    ));
    Ok(known)
}

pub(crate) fn eager_engine_registration_for_backend(
    backend: &EagerBackend,
) -> Result<EagerBackendRegistration, RuntimeConfigError> {
    match backend {
        EagerBackend::Cpu(backend) => Ok(EagerBackendRegistration::Install {
            family: EagerEngineFamily::Cpu,
            registration: cpu_runtime_engine_registration(backend)?,
        }),
        #[cfg(test)]
        EagerBackend::Recording(_) => Ok(EagerBackendRegistration::NoEngine),
        #[cfg(feature = "cuda")]
        EagerBackend::Cuda(backend) => Ok(EagerBackendRegistration::Install {
            family: EagerEngineFamily::Cuda,
            registration: tenferro_gpu::cuda_runtime_engine_registration(backend)?,
        }),
        #[cfg(feature = "webgpu")]
        EagerBackend::WebGpu(backend) => Ok(EagerBackendRegistration::Install {
            family: EagerEngineFamily::WebGpu,
            registration: tenferro_gpu::webgpu_runtime_engine_registration(backend)?,
        }),
    }
}

pub(crate) fn cpu_runtime_engine_id() -> Result<EngineId, RuntimeConfigError> {
    tenferro_cpu::runtime_engine_id()
}

pub(crate) fn cpu_runtime_hardware_class() -> Result<HardwareClassId, RuntimeConfigError> {
    tenferro_cpu::runtime_hardware_class()
}

pub(crate) fn cpu_runtime_engine_registration(
    backend: &CpuBackend,
) -> Result<EngineRegistration, RuntimeConfigError> {
    tenferro_cpu::runtime_engine_registration(backend)
}

macro_rules! dispatch {
    ($backend:expr, $method:ident($($arg:expr),* $(,)?)) => {
        match $backend {
            EagerBackend::Cpu(backend) => backend.$method($($arg),*),
            #[cfg(test)]
            EagerBackend::Recording(backend) => backend.$method($($arg),*),
            #[cfg(feature = "cuda")]
            EagerBackend::Cuda(backend) => backend.$method($($arg),*),
            #[cfg(feature = "webgpu")]
            EagerBackend::WebGpu(backend) => backend.$method($($arg),*),
        }
    };
}

#[cfg(test)]
#[derive(Debug)]
pub struct RecordingBackend {
    materializations: Arc<AtomicUsize>,
    inner: CpuBackend,
}

#[cfg(test)]
macro_rules! delegate_recording_backend_methods {
    ($(fn $method:ident($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty;)*) => {
        $(
            fn $method(&mut self, $($arg: $ty),*) -> $ret {
                self.inner.$method($($arg),*)
            }
        )*
    };
}

#[cfg(test)]
impl BackendRuntimeCache for RecordingBackend {
    type RuntimeCache = ();
}

#[cfg(test)]
impl TensorElementwise for RecordingBackend {
    fn elementwise_read_into(
        &mut self,
        op: ElementwiseReadOp,
        inputs: &[TensorRead<'_>],
        out: TensorWrite<'_>,
    ) -> TensorResult<()> {
        self.inner.elementwise_read_into(op, inputs, out)
    }

    delegate_recording_backend_methods! {
        fn add(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn sub(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
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
    }
}

#[cfg(test)]
impl TensorAnalytic for RecordingBackend {
    delegate_recording_backend_methods! {
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
    }
}

#[cfg(test)]
impl TensorStructural for RecordingBackend {
    fn to_contiguous_read(&mut self, input: TensorRead<'_>) -> TensorResult<Tensor> {
        self.materializations.fetch_add(1, Ordering::Relaxed);
        self.inner.to_contiguous_read(input)
    }

    fn copy_read_into(&mut self, src: TensorRead<'_>, dst: TensorWrite<'_>) -> TensorResult<()> {
        self.inner.copy_read_into(src, dst)
    }

    delegate_recording_backend_methods! {
        fn transpose(input: &Tensor, perm: &[usize]) -> TensorResult<Tensor>;
        fn reshape(input: &Tensor, shape: &[usize]) -> TensorResult<Tensor>;
        fn broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) -> TensorResult<Tensor>;
        fn cast(input: &Tensor, to: DType) -> TensorResult<Tensor>;
        fn extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> TensorResult<Tensor>;
        fn embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> TensorResult<Tensor>;
        fn tril(input: &Tensor, k: i64) -> TensorResult<Tensor>;
        fn triu(input: &Tensor, k: i64) -> TensorResult<Tensor>;
    }
}

#[cfg(test)]
impl TensorReduction for RecordingBackend {
    delegate_recording_backend_methods! {
        fn reduce_sum(input: &Tensor, axes: &[usize]) -> TensorResult<Tensor>;
        fn reduce_sum_squares_read(input: TensorRead<'_>, axes: &[usize]) -> TensorResult<Tensor>;
        fn reduce_prod(input: &Tensor, axes: &[usize]) -> TensorResult<Tensor>;
        fn reduce_max(input: &Tensor, axes: &[usize]) -> TensorResult<Tensor>;
        fn reduce_min(input: &Tensor, axes: &[usize]) -> TensorResult<Tensor>;
    }
}

#[cfg(test)]
impl TensorIndexing for RecordingBackend {
    delegate_recording_backend_methods! {
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

#[cfg(test)]
impl TensorDot for RecordingBackend {
    fn dot_general(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> TensorResult<Tensor> {
        self.inner.dot_general(lhs, rhs, config)
    }
}

#[cfg(test)]
impl TensorFusion for RecordingBackend {}
#[cfg(test)]
impl TensorBuffer for RecordingBackend {}
#[cfg(test)]
impl TensorDeviceTransfer for RecordingBackend {}
#[cfg(test)]
impl BackendCachedDot for RecordingBackend {}
#[cfg(test)]
impl BackendSessionHost for RecordingBackend {
    fn with_backend_session<R: Send>(
        &mut self,
        f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
    ) -> R {
        f(self)
    }
}
#[cfg(test)]
impl TensorBackend for RecordingBackend {}

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
    fn elementwise_read_into(
        &mut self,
        op: ElementwiseReadOp,
        inputs: &[TensorRead<'_>],
        out: TensorWrite<'_>,
    ) -> TensorResult<()> {
        dispatch!(self, elementwise_read_into(op, inputs, out))
    }

    delegate_tensor_backend_methods! {
        fn add(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn add_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor>;
        fn sub(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn sub_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor>;
        fn mul(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn mul_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor>;
        fn neg(input: &Tensor) -> TensorResult<Tensor>;
        fn neg_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn conj(input: &Tensor) -> TensorResult<Tensor>;
        fn conj_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn div(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn div_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor>;
        fn rem(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn rem_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor>;
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
        fn to_contiguous_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn copy_read_into(src: TensorRead<'_>, dst: TensorWrite<'_>) -> TensorResult<()>;
        fn transpose(input: &Tensor, perm: &[usize]) -> TensorResult<Tensor>;
        fn reshape(input: &Tensor, shape: &[usize]) -> TensorResult<Tensor>;
        fn reshape_read(input: TensorRead<'_>, shape: &[usize]) -> TensorResult<Tensor>;
        fn broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) -> TensorResult<Tensor>;
        fn broadcast_in_dim_read(input: TensorRead<'_>, shape: &[usize], dims: &[usize]) -> TensorResult<Tensor>;
        fn cast(input: &Tensor, to: DType) -> TensorResult<Tensor>;
        fn extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> TensorResult<Tensor>;
        fn embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> TensorResult<Tensor>;
        fn tril(input: &Tensor, k: i64) -> TensorResult<Tensor>;
        fn triu(input: &Tensor, k: i64) -> TensorResult<Tensor>;
    }
}

impl TensorReduction for EagerBackend {
    delegate_tensor_backend_methods! {
        fn reduce_sum(input: &Tensor, axes: &[usize]) -> TensorResult<Tensor>;
        fn reduce_sum_squares_read(input: TensorRead<'_>, axes: &[usize]) -> TensorResult<Tensor>;
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
    fn with_backend_session<R: Send>(
        &mut self,
        f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
    ) -> R {
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
        fn execute_broadcast_multiply(lhs: TensorRead<'_>, lhs_shape: &[usize], lhs_dims: &[usize], rhs: TensorRead<'_>, rhs_shape: &[usize], rhs_dims: &[usize]) -> TensorResult<Option<Tensor>>;
        fn execute_broadcast_multiply_value(lhs: TensorRead<'_>, lhs_shape: &[usize], lhs_dims: &[usize], rhs: TensorRead<'_>, rhs_shape: &[usize], rhs_dims: &[usize]) -> TensorResult<Option<TensorValue>>;
    }
}

impl BackendCachedDot for EagerBackend {}

impl TensorBackend for EagerBackend {}
