#[cfg(test)]
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
#[cfg(test)]
use tenferro_tensor::{
    BackendCachedDot, CompareDir, DotGeneralConfig, ElementwiseReadOp, GatherConfig, PadConfig,
    ScatterConfig, SliceConfig, TensorAnalytic, TensorBackend, TensorBuffer, TensorDeviceTransfer,
    TensorDot, TensorElementwise, TensorFusion, TensorIndexing, TensorReduction, TensorStructural,
    TensorWrite,
};
use tenferro_tensor::{
    BackendRuntimeCache, BackendSession, BackendSessionHost, DType, MemoryKind,
    Result as TensorResult, Tensor, TensorRead,
};

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
            install_engine: false,
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
            install_engine: false,
        })
    }

    /// Like [`Self::recording_cpu_counting_sessions`], but the runtime also gets
    /// the inner CPU engine so compiled derivative programs (semantic VJP) can
    /// run; eager-backend session entries are still counted.
    #[cfg(test)]
    pub(crate) fn recording_cpu_counting_sessions_with_engine(
        materializations: Arc<AtomicUsize>,
        sessions: Arc<AtomicUsize>,
    ) -> Self {
        Self::Recording(RecordingBackend {
            materializations,
            sessions,
            inner: CpuBackend::new(),
            install_engine: true,
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
        EagerBackend::Recording(backend) if backend.install_engine => {
            Ok(EagerBackendRegistration::Install(Box::new(
                cpu_runtime_engine_registration(&backend.inner)?,
            )))
        }
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

#[cfg(test)]
#[derive(Debug)]
pub struct RecordingBackend {
    materializations: Arc<AtomicUsize>,
    /// Backend-session entries, so a test can prove an operation stayed
    /// session-free rather than inferring it from timing.
    sessions: Arc<AtomicUsize>,
    inner: CpuBackend,
    /// Install `inner` as the runtime engine instead of leaving the runtime
    /// without one.
    install_engine: bool,
}

#[cfg(test)]
macro_rules! delegate_recording_backend_methods {
    ($(fn $method:ident($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty;)*) => {
        $(
            fn $method(&mut self, $($arg: $ty),*) -> $ret {
                self.inner
                    .with_backend_session(|__s| __s.$method($($arg),*))?
            }
        )*
    };
}

#[cfg(test)]
macro_rules! delegate_owned_read_methods {
    ($(fn $method:ident($label:literal; $($read:ident),+ $(; $($extra:ident: $extra_ty:ty),*)?);)*) => {
        $(
            fn $method(
                &mut self,
                $($read: TensorRead<'_>,)+
                $($($extra: $extra_ty),*)?
            ) -> TensorResult<Tensor> {
                $(let $read = tenferro_tensor::backend::read_owned_tensor($label, $read)?;)+
                self.inner.with_backend_session(|__s| {
                    __s.$method($(TensorRead::from_tensor($read)),+ $(, $($extra),*)?)
                })?
            }
        )*
    };
}

#[cfg(test)]
impl BackendSession for RecordingBackend {}

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
        self.inner
            .with_backend_session(|__s| __s.elementwise_read_into(op, inputs, out))?
    }

    delegate_recording_backend_methods! {
        fn mul_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor>;
    }
    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn add_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor> {
        let lhs = tenferro_tensor::backend::read_owned_tensor("add", lhs)?;
        let rhs = tenferro_tensor::backend::read_owned_tensor("add", rhs)?;
        self.inner.with_backend_session(|__s| {
            __s.add_read(TensorRead::from_tensor(lhs), TensorRead::from_tensor(rhs))
        })?
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    delegate_owned_read_methods! {
        fn sub_read("sub"; lhs, rhs);
        fn neg_read("neg"; input);
        fn conj_read("conj"; input);
        fn div_read("div"; lhs, rhs);
        fn abs_read("abs"; input);
        fn sign_read("sign"; input);
        fn maximum_read("maximum"; lhs, rhs);
        fn minimum_read("minimum"; lhs, rhs);
        fn compare_read("compare"; lhs, rhs; dir: &CompareDir);
        fn select_read("select"; pred, on_true, on_false);
        fn clamp_read("clamp"; input, lower, upper);
    }
}

#[cfg(test)]
impl TensorAnalytic for RecordingBackend {
    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    delegate_owned_read_methods! {
        fn exp_read("exp"; input);
        fn log_read("log"; input);
        fn sin_read("sin"; input);
        fn cos_read("cos"; input);
        fn tanh_read("tanh"; input);
        fn sqrt_read("sqrt"; input);
        fn rsqrt_read("rsqrt"; input);
        fn pow_read("pow"; lhs, rhs);
        fn expm1_read("expm1"; input);
        fn log1p_read("log1p"; input);
        fn erf_read("erf"; input);
    }
}

#[cfg(test)]
impl TensorStructural for RecordingBackend {
    fn to_contiguous_read(&mut self, input: TensorRead<'_>) -> TensorResult<Tensor> {
        self.materializations.fetch_add(1, Ordering::Relaxed);
        self.inner
            .with_backend_session(|__s| __s.to_contiguous_read(input))?
    }

    fn copy_read_into(&mut self, src: TensorRead<'_>, dst: TensorWrite<'_>) -> TensorResult<()> {
        self.inner
            .with_backend_session(|__s| __s.copy_read_into(src, dst))?
    }

    delegate_recording_backend_methods! {
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
        let input = tenferro_tensor::backend::read_owned_tensor("transpose", input)?;
        self.inner
            .with_backend_session(|__s| __s.transpose_read(TensorRead::from_tensor(input), perm))?
    }

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected borrowed views. Reproduce it explicitly rather than
    // forwarding a view, which would widen the accepted input surface.
    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    delegate_owned_read_methods! {
        fn reshape_read("reshape"; input; shape: &[usize]);
    }

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected borrowed views. Reproduce it explicitly rather than
    // forwarding a view, which would widen the accepted input surface.
    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    delegate_owned_read_methods! {
        fn broadcast_in_dim_read("broadcast_in_dim"; input; shape: &[usize], dims: &[usize]);
    }
}

#[cfg(test)]
impl TensorReduction for RecordingBackend {
    delegate_recording_backend_methods! {
        fn reduce_sum_squares_read(input: TensorRead<'_>, axes: &[usize]) -> TensorResult<Tensor>;
    }

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected views, which is not the same as forwarding a view to
    // the inner backend. Reproduce the old default explicitly.
    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    delegate_owned_read_methods! {
        fn reduce_sum_read("reduce_sum"; input; axes: &[usize]);
        fn reduce_prod_read("reduce_prod"; input; axes: &[usize]);
        fn reduce_max_read("reduce_max"; input; axes: &[usize]);
        fn reduce_min_read("reduce_min"; input; axes: &[usize]);
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
            (Some(lhs), Some(rhs)) => self.inner.with_backend_session(|__s| {
                __s.dot_general_read(
                    TensorRead::from_tensor(lhs),
                    TensorRead::from_tensor(rhs),
                    config,
                )
            })?,
            _ => {
                let lhs = self.to_contiguous_read(lhs)?;
                let rhs = self.to_contiguous_read(rhs)?;
                self.inner.with_backend_session(|__s| {
                    __s.dot_general_read(
                        TensorRead::from_tensor(&lhs),
                        TensorRead::from_tensor(&rhs),
                        config,
                    )
                })?
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
    fn with_backend_session<R>(
        &mut self,
        f: impl FnOnce(&mut dyn BackendSession) -> R,
    ) -> Result<R, tenferro_tensor::SessionEntryError> {
        self.sessions.fetch_add(1, Ordering::Relaxed);
        Ok(f(self))
    }
}
#[cfg(test)]
impl TensorBackend for RecordingBackend {}

impl BackendRuntimeCache for EagerBackend {
    type RuntimeCache = ();
}

impl BackendSessionHost for EagerBackend {
    fn with_backend_session<R>(
        &mut self,
        f: impl FnOnce(&mut dyn BackendSession) -> R,
    ) -> Result<R, tenferro_tensor::SessionEntryError> {
        // Hand the caller the *concrete* backend's session. The composite enum
        // is not a session: its read halves deliberately reproduce the
        // read-boundary policy (owned inputs only), while the concrete sessions
        // are what the operation entry points expect.
        match self {
            EagerBackend::Cpu(backend) => backend.with_backend_session(f),
            #[cfg(test)]
            EagerBackend::Recording(backend) => backend.with_backend_session(f),
            #[cfg(feature = "cuda")]
            EagerBackend::Cuda(backend) => backend.with_backend_session(f),
            #[cfg(feature = "webgpu")]
            EagerBackend::WebGpu(backend) => backend.with_backend_session(f),
        }
    }
}
