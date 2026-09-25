use std::any::TypeId;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(test)]
use std::sync::Arc;
use tenferro_cpu::CpuBackend;
#[cfg(feature = "cuda")]
use tenferro_gpu::cuda::CudaBackend;
#[cfg(feature = "webgpu")]
use tenferro_gpu::webgpu::WebGpuBackend;
use tenferro_runtime::{
    EngineId, EngineRegistration, HardwareClassId, Runtime, RuntimeConfigError,
};
use tenferro_tensor::backend::ElementwiseFusionPlan;
use tenferro_tensor::{
    BackendCachedDot, BackendRuntimeCache, BackendSession, BackendSessionHost, CompareDir, DType,
    DotGeneralConfig, ElementwiseReadOp, GatherConfig, MemoryKind, PadConfig,
    Result as TensorResult, ScatterConfig, SliceConfig, Tensor, TensorAnalytic, TensorBackend,
    TensorBuffer, TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing,
    TensorRead, TensorReduction, TensorStructural, TensorValue, TensorWrite,
};

#[doc(hidden)]
struct EagerBackendSessionMarker;

/// Copy an owned host read into a fresh compact host tensor, or `None` when the
/// CPU backend's host clone would refuse it.
///
/// This mirrors `tenferro-cpu`'s `clone_host_tensor_read` acceptance: host
/// placement and a preset scalar. Everything else — a view read, a
/// backend-family buffer, a device or managed placement, a caller-owned external
/// scalar — keeps the session path, where that backend reports its own typed
/// error. `host_leaf_materialization_matches_the_cpu_backend` pins the two
/// against each other, including the decline cases.
fn cpu_host_owned_read(input: &TensorRead<'_>) -> Option<TensorResult<Tensor>> {
    if !matches!(input, TensorRead::Tensor(_))
        || input.backend_family().is_some()
        || !matches!(
            input.placement().memory_kind,
            MemoryKind::PinnedHost | MemoryKind::UnpinnedHost
        )
        || matches!(input.dtype(), DType::External(_))
    {
        return None;
    }
    Some(input.clone().tensor_view().duplicate())
}

pub(crate) enum EagerBackend {
    Cpu(CpuBackend),
    #[cfg(test)]
    Recording(RecordingBackend),
    #[cfg(feature = "cuda")]
    Cuda(CudaBackend),
    #[cfg(feature = "webgpu")]
    WebGpu(WebGpuBackend),
}

/// The fallible provider-specific registration produced from the exact eager
/// backend. `NoEngine` is used by the test-only recording backend, whose
/// tensor operations are intentionally not installed as a runtime engine.
enum EagerBackendRegistration {
    #[cfg(test)]
    NoEngine,
    Install(Box<EngineRegistration>),
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
            sessions: Arc::new(AtomicUsize::new(0)),
            inner: CpuBackend::new(),
        })
    }

    #[cfg(test)]
    pub(crate) fn recording_cpu_counting_sessions(
        materializations: Arc<AtomicUsize>,
        sessions: Arc<AtomicUsize>,
    ) -> Self {
        Self::Recording(RecordingBackend {
            materializations,
            sessions,
            inner: CpuBackend::new(),
        })
    }

    /// Materialize a host-placement read without entering a backend session.
    ///
    /// The CPU backend performs no provider work for an owned host tensor: its
    /// `to_contiguous_read` only copies the host buffer, so a session would add
    /// admission, provider exclusion, and session construction for nothing
    /// (#1704). This helper states the CPU acceptance rule in the eager layer —
    /// a host placement and a preset scalar — and copies through the tensor
    /// layer's own view copy, so the bytes come from the same code path the CPU
    /// backend uses. The recording backend shares it because it wraps a CPU
    /// backend.
    ///
    /// The device backends return `None` on purpose: their placement errors and
    /// no-implicit-transfer policy stay unchanged, so a host tensor handed to a
    /// device runtime is still rejected inside that backend's session.
    pub(crate) fn to_contiguous_host_read(
        &self,
        input: &TensorRead<'_>,
    ) -> Option<TensorResult<Tensor>> {
        match self {
            Self::Cpu(_) => cpu_host_owned_read(input),
            #[cfg(test)]
            Self::Recording(_) => cpu_host_owned_read(input),
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => None,
            #[cfg(feature = "webgpu")]
            Self::WebGpu(_) => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn recording_session_owner(&mut self) -> Option<*mut ()> {
        match self {
            Self::Recording(backend) => Some((backend as *mut RecordingBackend).cast()),
            _ => None,
        }
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

    /// Return the concrete backend as a runtime-owned erased execution
    /// context for the prepared-operation `execute` bridge (the native-context
    /// path). The executor downcasts this to the concrete backend type that
    /// matches its binding's `context_identity`.
    pub(crate) fn erased_context(&mut self) -> tenferro_runtime::ErasedExecutionContext<'_> {
        match self {
            Self::Cpu(backend) => tenferro_runtime::ErasedExecutionContext::new(backend),
            #[cfg(test)]
            Self::Recording(backend) => tenferro_runtime::ErasedExecutionContext::new(backend),
            #[cfg(feature = "cuda")]
            Self::Cuda(backend) => tenferro_runtime::ErasedExecutionContext::new(backend),
            #[cfg(feature = "webgpu")]
            Self::WebGpu(backend) => tenferro_runtime::ErasedExecutionContext::new(backend),
        }
    }
}

pub(crate) fn eager_runtime_for_backend(
    backend: &EagerBackend,
) -> Result<Runtime, RuntimeConfigError> {
    let mut builder = Runtime::builder();
    match eager_engine_registration_for_backend(backend)? {
        #[cfg(test)]
        EagerBackendRegistration::NoEngine => {}
        EagerBackendRegistration::Install(registration) => {
            builder.register_engine(*registration)?;
        }
    }
    builder.build()
}

fn eager_engine_registration_for_backend(
    backend: &EagerBackend,
) -> Result<EagerBackendRegistration, RuntimeConfigError> {
    match backend {
        EagerBackend::Cpu(backend) => Ok(EagerBackendRegistration::Install(Box::new(
            cpu_runtime_engine_registration(backend)?,
        ))),
        #[cfg(test)]
        EagerBackend::Recording(_) => Ok(EagerBackendRegistration::NoEngine),
        #[cfg(feature = "cuda")]
        EagerBackend::Cuda(backend) => Ok(EagerBackendRegistration::Install(Box::new(
            tenferro_gpu::cuda::cuda_runtime_engine_registration(
                backend,
                cuda_runtime_engine_id()?,
            )?,
        ))),
        #[cfg(feature = "webgpu")]
        EagerBackend::WebGpu(backend) => Ok(EagerBackendRegistration::Install(Box::new(
            tenferro_gpu::webgpu::webgpu_runtime_engine_registration(backend)?,
        ))),
    }
}

pub(crate) fn cpu_runtime_engine_id() -> Result<EngineId, RuntimeConfigError> {
    tenferro_cpu::runtime_engine_id()
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_runtime_engine_id() -> Result<EngineId, RuntimeConfigError> {
    EngineId::new("tenferro-ad.cuda.default.v1").map_err(RuntimeConfigError::from)
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
#[doc(hidden)]
struct RecordingBackendSessionMarker;

#[cfg(test)]
#[derive(Debug)]
pub struct RecordingBackend {
    materializations: Arc<AtomicUsize>,
    /// Backend-session entries, so a test can prove an operation stayed
    /// session-free rather than inferring it from timing.
    sessions: Arc<AtomicUsize>,
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
impl BackendSession for RecordingBackend {
    fn session_type_id(&self) -> TypeId {
        TypeId::of::<RecordingBackendSessionMarker>()
    }

    unsafe fn session_data_mut(&mut self) -> *mut () {
        self as *mut Self as *mut ()
    }
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
        fn mul_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor>;
        fn neg(input: &Tensor) -> TensorResult<Tensor>;
        fn conj(input: &Tensor) -> TensorResult<Tensor>;
        fn div(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn abs(input: &Tensor) -> TensorResult<Tensor>;
        fn sign(input: &Tensor) -> TensorResult<Tensor>;
        fn maximum(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn minimum(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> TensorResult<Tensor>;
        fn clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) -> TensorResult<Tensor>;
    }
    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn add_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor> {
        self.add(
            tenferro_tensor::backend::read_owned_tensor("add", lhs)?,
            tenferro_tensor::backend::read_owned_tensor("add", rhs)?,
        )
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn sub_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor> {
        self.sub(
            tenferro_tensor::backend::read_owned_tensor("sub", lhs)?,
            tenferro_tensor::backend::read_owned_tensor("sub", rhs)?,
        )
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn neg_read(&mut self, input: TensorRead<'_>) -> TensorResult<Tensor> {
        self.neg(tenferro_tensor::backend::read_owned_tensor("neg", input)?)
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn conj_read(&mut self, input: TensorRead<'_>) -> TensorResult<Tensor> {
        self.conj(tenferro_tensor::backend::read_owned_tensor("conj", input)?)
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn div_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor> {
        self.div(
            tenferro_tensor::backend::read_owned_tensor("div", lhs)?,
            tenferro_tensor::backend::read_owned_tensor("div", rhs)?,
        )
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn abs_read(&mut self, input: TensorRead<'_>) -> TensorResult<Tensor> {
        self.abs(tenferro_tensor::backend::read_owned_tensor("abs", input)?)
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn sign_read(&mut self, input: TensorRead<'_>) -> TensorResult<Tensor> {
        self.sign(tenferro_tensor::backend::read_owned_tensor("sign", input)?)
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn maximum_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor> {
        self.maximum(
            tenferro_tensor::backend::read_owned_tensor("maximum", lhs)?,
            tenferro_tensor::backend::read_owned_tensor("maximum", rhs)?,
        )
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn minimum_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor> {
        self.minimum(
            tenferro_tensor::backend::read_owned_tensor("minimum", lhs)?,
            tenferro_tensor::backend::read_owned_tensor("minimum", rhs)?,
        )
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn compare_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        dir: &CompareDir,
    ) -> TensorResult<Tensor> {
        self.compare(
            tenferro_tensor::backend::read_owned_tensor("compare", lhs)?,
            tenferro_tensor::backend::read_owned_tensor("compare", rhs)?,
            dir,
        )
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn select_read(
        &mut self,
        pred: TensorRead<'_>,
        on_true: TensorRead<'_>,
        on_false: TensorRead<'_>,
    ) -> TensorResult<Tensor> {
        let pred = tenferro_tensor::backend::read_owned_tensor("select", pred)?;
        let on_true = tenferro_tensor::backend::read_owned_tensor("select", on_true)?;
        let on_false = tenferro_tensor::backend::read_owned_tensor("select", on_false)?;
        self.inner.select_read(
            TensorRead::from_tensor(pred),
            TensorRead::from_tensor(on_true),
            TensorRead::from_tensor(on_false),
        )
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn clamp_read(
        &mut self,
        input: TensorRead<'_>,
        lower: TensorRead<'_>,
        upper: TensorRead<'_>,
    ) -> TensorResult<Tensor> {
        self.clamp(
            tenferro_tensor::backend::read_owned_tensor("clamp", input)?,
            tenferro_tensor::backend::read_owned_tensor("clamp", lower)?,
            tenferro_tensor::backend::read_owned_tensor("clamp", upper)?,
        )
    }
}

#[cfg(test)]
impl TensorAnalytic for RecordingBackend {
    delegate_recording_backend_methods! {
        fn exp(input: &Tensor) -> TensorResult<Tensor>;
        fn sqrt(input: &Tensor) -> TensorResult<Tensor>;
        fn pow(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
    }
    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn exp_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult<Tensor> {
        self.exp(tenferro_tensor::backend::read_owned_tensor("exp", input)?)
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn log_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult<Tensor> {
        let input = tenferro_tensor::backend::read_owned_tensor("log", input)?;
        self.inner.log_read(TensorRead::from_tensor(input))
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn sin_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult<Tensor> {
        let input = tenferro_tensor::backend::read_owned_tensor("sin", input)?;
        self.inner.sin_read(TensorRead::from_tensor(input))
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn cos_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult<Tensor> {
        let input = tenferro_tensor::backend::read_owned_tensor("cos", input)?;
        self.inner.cos_read(TensorRead::from_tensor(input))
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn tanh_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult<Tensor> {
        let input = tenferro_tensor::backend::read_owned_tensor("tanh", input)?;
        self.inner.tanh_read(TensorRead::from_tensor(input))
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn sqrt_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult<Tensor> {
        self.sqrt(tenferro_tensor::backend::read_owned_tensor("sqrt", input)?)
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn rsqrt_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult<Tensor> {
        let input = tenferro_tensor::backend::read_owned_tensor("rsqrt", input)?;
        self.inner.rsqrt_read(TensorRead::from_tensor(input))
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn pow_read(
        &mut self,
        lhs: tenferro_tensor::TensorRead<'_>,
        rhs: tenferro_tensor::TensorRead<'_>,
    ) -> TensorResult<Tensor> {
        self.pow(
            tenferro_tensor::backend::read_owned_tensor("pow", lhs)?,
            tenferro_tensor::backend::read_owned_tensor("pow", rhs)?,
        )
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn expm1_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult<Tensor> {
        let input = tenferro_tensor::backend::read_owned_tensor("expm1", input)?;
        self.inner.expm1_read(TensorRead::from_tensor(input))
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn log1p_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult<Tensor> {
        let input = tenferro_tensor::backend::read_owned_tensor("log1p", input)?;
        self.inner.log1p_read(TensorRead::from_tensor(input))
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

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected borrowed views. Reproduce it explicitly rather than
    // forwarding a view, which would widen the accepted input surface.
    fn transpose_read(&mut self, input: TensorRead<'_>, perm: &[usize]) -> TensorResult<Tensor> {
        self.transpose(
            tenferro_tensor::backend::read_owned_tensor("transpose", input)?,
            perm,
        )
    }

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected borrowed views. Reproduce it explicitly rather than
    // forwarding a view, which would widen the accepted input surface.
    fn reshape_read(&mut self, input: TensorRead<'_>, shape: &[usize]) -> TensorResult<Tensor> {
        self.reshape(
            tenferro_tensor::backend::read_owned_tensor("reshape", input)?,
            shape,
        )
    }

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected borrowed views. Reproduce it explicitly rather than
    // forwarding a view, which would widen the accepted input surface.
    fn broadcast_in_dim_read(
        &mut self,
        input: TensorRead<'_>,
        shape: &[usize],
        dims: &[usize],
    ) -> TensorResult<Tensor> {
        self.broadcast_in_dim(
            tenferro_tensor::backend::read_owned_tensor("broadcast_in_dim", input)?,
            shape,
            dims,
        )
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

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected views, which is not the same as forwarding a view to
    // the inner backend. Reproduce the old default explicitly.
    fn reduce_sum_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> TensorResult<Tensor> {
        self.reduce_sum(
            tenferro_tensor::backend::read_owned_tensor("reduce_sum", input)?,
            axes,
        )
    }

    fn reduce_prod_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> TensorResult<Tensor> {
        self.reduce_prod(
            tenferro_tensor::backend::read_owned_tensor("reduce_prod", input)?,
            axes,
        )
    }

    fn reduce_max_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> TensorResult<Tensor> {
        self.reduce_max(
            tenferro_tensor::backend::read_owned_tensor("reduce_max", input)?,
            axes,
        )
    }

    fn reduce_min_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> TensorResult<Tensor> {
        self.reduce_min(
            tenferro_tensor::backend::read_owned_tensor("reduce_min", input)?,
            axes,
        )
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
    // The previous read-half default delegated an owned pair to the one-shot
    // method and materialized borrowed views through to_contiguous_read before
    // contracting. Reproduce that exactly rather than forwarding a view.
    fn dot_general_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> TensorResult<Tensor> {
        match (lhs.as_tensor(), rhs.as_tensor()) {
            (Some(lhs), Some(rhs)) => self.dot_general(lhs, rhs, config),
            _ => {
                let lhs = self.to_contiguous_read(lhs)?;
                let rhs = self.to_contiguous_read(rhs)?;
                self.dot_general(&lhs, &rhs, config)
            }
        }
    }
}

#[cfg(test)]
impl TensorFusion for RecordingBackend {}
#[cfg(test)]
impl TensorBuffer for RecordingBackend {}
#[cfg(test)]
impl TensorDeviceTransfer for RecordingBackend {
    fn download_to_host(&mut self, tensor: TensorRead<'_>) -> TensorResult<Tensor> {
        self.inner.download_to_host(tensor)
    }

    fn upload_host_tensor(&mut self, tensor: TensorRead<'_>) -> TensorResult<Tensor> {
        self.inner.upload_host_tensor(tensor)
    }
}
#[cfg(test)]
impl BackendCachedDot for RecordingBackend {}
#[cfg(test)]
impl BackendSessionHost for RecordingBackend {
    fn with_backend_session<R: Send>(
        &mut self,
        f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
    ) -> R {
        self.sessions.fetch_add(1, Ordering::Relaxed);
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

impl BackendSession for EagerBackend {
    fn session_type_id(&self) -> TypeId {
        TypeId::of::<EagerBackendSessionMarker>()
    }

    unsafe fn session_data_mut(&mut self) -> *mut () {
        self as *mut Self as *mut ()
    }
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
        fn select_read(pred: TensorRead<'_>, on_true: TensorRead<'_>, on_false: TensorRead<'_>) -> TensorResult<Tensor>;
        fn clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) -> TensorResult<Tensor>;
        fn clamp_read(input: TensorRead<'_>, lower: TensorRead<'_>, upper: TensorRead<'_>) -> TensorResult<Tensor>;
    }
}

impl TensorAnalytic for EagerBackend {
    delegate_tensor_backend_methods! {
        fn exp(input: &Tensor) -> TensorResult<Tensor>;
        fn exp_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn log_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn sin_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn cos_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn tanh_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn sqrt(input: &Tensor) -> TensorResult<Tensor>;
        fn sqrt_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn rsqrt_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn pow(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn pow_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor>;
        fn expm1_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
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

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected borrowed views. Reproduce it explicitly rather than
    // forwarding a view, which would widen the accepted input surface.
    fn transpose_read(&mut self, input: TensorRead<'_>, perm: &[usize]) -> TensorResult<Tensor> {
        self.transpose(
            tenferro_tensor::backend::read_owned_tensor("transpose", input)?,
            perm,
        )
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

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected views. Dispatching a view to the concrete backend
    // would widen the accepted input surface, so reproduce the old default.
    fn reduce_sum_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> TensorResult<Tensor> {
        self.reduce_sum(
            tenferro_tensor::backend::read_owned_tensor("reduce_sum", input)?,
            axes,
        )
    }

    fn reduce_prod_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> TensorResult<Tensor> {
        self.reduce_prod(
            tenferro_tensor::backend::read_owned_tensor("reduce_prod", input)?,
            axes,
        )
    }

    fn reduce_max_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> TensorResult<Tensor> {
        self.reduce_max(
            tenferro_tensor::backend::read_owned_tensor("reduce_max", input)?,
            axes,
        )
    }

    fn reduce_min_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> TensorResult<Tensor> {
        self.reduce_min(
            tenferro_tensor::backend::read_owned_tensor("reduce_min", input)?,
            axes,
        )
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
        fn download_to_host(tensor: TensorRead<'_>) -> TensorResult<Tensor>;
        fn upload_host_tensor(tensor: TensorRead<'_>) -> TensorResult<Tensor>;
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
