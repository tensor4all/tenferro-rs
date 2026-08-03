use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use crate::arbiter::{
    inherited_or_new_execution_owner, with_execution_owner, ResourceArbiter, ResourceOwner,
    ResourcePermit,
};
use crate::buffer_pool::{BufferPool, BufferPoolStats, PoolScalar};
use crate::dot_runtime::{CpuProviderBundle, CpuProviderBundleInstallError};
use crate::engine::{CpuEngine, EngineResources};
use crate::indexed_plan_cache::{
    IndexedPlanCache, IndexedPlanCacheLimits, DEFAULT_INDEXED_PLAN_CACHE_LIMITS,
};
use crate::placement::{
    resolve_placement, resolve_placement_with_affinity, CpuEngineConstructionError,
    ResolvedCpuExecution,
};
use crate::provider::{CpuExecutionContext, CpuOperationEntry, ParallelMode};
use crate::{
    discover_cpu_topology, CpuDomainId, CpuDomainOwnership, CpuExecutorAffinity,
    CpuExecutorShutdown, CpuId, CpuPlacement, CpuPlacementError, CpuPlacementGuarantee, CpuSet,
    CpuTopology, CpuTopologyError, ExternalCpuDomain, NumaNodeId, ResolvedCpuPlacement,
};
use crate::{
    Buffer, CacheStats, Tensor, TensorRank, TensorRead, TensorScalar, TensorValue, TensorWrite,
    TypedTensor, TypedTensorView, TypedTensorViewMut,
};
use tenferro_tensor::backend::{ElementwiseFusionPlan, GroupedGemmConfig};
use tenferro_tensor::SharedTensorAllocationDomain;
use tenferro_tensor::{
    AllocationDomainId, BackendCachedDot, BackendRuntimeCache, BackendSession, BackendSessionHost,
    DotGeneralAccumulation, ElementwiseReadOp, TensorAnalytic, TensorBackend, TensorBuffer,
    TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing,
    TensorReduction, TensorStructural, TensorViewCanonicalization,
};
use tenferro_tensor::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};

use super::exec_session::CpuExecSession;
use super::{
    analytic, copy_tensor_read_into, elementwise, gemm, indexing, materialize_tensor_read,
    reduction, structural, CpuContext,
};

pub(crate) fn tag_fresh_output(output: &mut Tensor, domain: CpuDomainId) {
    macro_rules! tag {
        ($tensor:expr) => {{
            $tensor.set_cpu_affinity(Some(domain));
        }};
    }
    match output {
        Tensor::F32(tensor) => tag!(tensor),
        Tensor::F64(tensor) => tag!(tensor),
        Tensor::I32(tensor) => tag!(tensor),
        Tensor::I64(tensor) => tag!(tensor),
        Tensor::Bool(tensor) => tag!(tensor),
        Tensor::C32(tensor) => tag!(tensor),
        Tensor::C64(tensor) => tag!(tensor),
    }
}

pub(crate) fn elementwise_read_into_fallback_with_pool(
    buffers: &mut BufferPool,
    op: ElementwiseReadOp,
    inputs: &[TensorRead<'_>],
    out: TensorWrite<'_>,
) -> crate::Result<()> {
    let result = match op {
        ElementwiseReadOp::Add => {
            elementwise::add_read_with_pool(buffers, inputs[0].clone(), inputs[1].clone())?
        }
        ElementwiseReadOp::Subtract => {
            elementwise::sub_read_with_pool(buffers, inputs[0].clone(), inputs[1].clone())?
        }
        ElementwiseReadOp::Multiply => {
            elementwise::mul_read_with_pool(buffers, inputs[0].clone(), inputs[1].clone())?
        }
        ElementwiseReadOp::Negate => elementwise::neg_read_with_pool(buffers, inputs[0].clone())?,
        ElementwiseReadOp::Conj => elementwise::conj_read_with_pool(buffers, inputs[0].clone())?,
        ElementwiseReadOp::Divide => {
            elementwise::div_read_with_pool(buffers, inputs[0].clone(), inputs[1].clone())?
        }
        _ => {
            return Err(crate::Error::unsupported(
                "CpuBackend::elementwise_read_into",
                format!("CPU backend does not implement {op:?}"),
            ))
        }
    };
    copy_tensor_read_into(
        "CpuBackend::elementwise_read_into",
        TensorRead::from_tensor(&result),
        out,
    )
}

pub(crate) trait FreshCpuOutput {
    fn tag_fresh(&mut self, domain: CpuDomainId);
}

impl FreshCpuOutput for Tensor {
    fn tag_fresh(&mut self, domain: CpuDomainId) {
        tag_fresh_output(self, domain);
    }
}

impl<T, R: TensorRank> FreshCpuOutput for TypedTensor<T, R> {
    fn tag_fresh(&mut self, domain: CpuDomainId) {
        self.set_cpu_affinity(Some(domain));
    }
}

impl<T: FreshCpuOutput> FreshCpuOutput for Option<T> {
    fn tag_fresh(&mut self, domain: CpuDomainId) {
        if let Some(output) = self {
            output.tag_fresh(domain);
        }
    }
}

impl<T: FreshCpuOutput> FreshCpuOutput for Vec<T> {
    fn tag_fresh(&mut self, domain: CpuDomainId) {
        for output in self {
            output.tag_fresh(domain);
        }
    }
}

#[derive(Debug, Default, Clone)]
struct CpuSessionProfileEntry {
    calls: usize,
    total_time: Duration,
}

fn cpu_session_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env::var("TENFERRO_PROFILE_CPU_SESSION").is_ok())
}

fn cpu_session_profile_print_every() -> Option<usize> {
    static PRINT_EVERY: OnceLock<Option<usize>> = OnceLock::new();
    *PRINT_EVERY.get_or_init(|| {
        env::var("TENFERRO_PROFILE_CPU_SESSION_PRINT_EVERY")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
    })
}

fn cpu_session_profile_state() -> &'static Mutex<HashMap<&'static str, CpuSessionProfileEntry>> {
    static STATE: OnceLock<Mutex<HashMap<&'static str, CpuSessionProfileEntry>>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn record_cpu_session_profile(section: &'static str, elapsed: Duration) {
    if !cpu_session_profile_enabled() {
        return;
    }
    let Ok(mut state) = cpu_session_profile_state().lock() else {
        return;
    };
    let entry = state.entry(section).or_default();
    entry.calls += 1;
    entry.total_time += elapsed;
}

fn profile_cpu_session_section<T>(section: &'static str, f: impl FnOnce() -> T) -> T {
    if !cpu_session_profile_enabled() {
        return f();
    }
    let started = Instant::now();
    let result = f();
    record_cpu_session_profile(section, started.elapsed());
    result
}

fn maybe_print_cpu_session_profile() {
    let Some(print_every) = cpu_session_profile_print_every() else {
        return;
    };
    let should_print = {
        let Ok(state) = cpu_session_profile_state().lock() else {
            return;
        };
        state
            .get("with_backend_session_cached.total")
            .is_some_and(|entry| entry.calls % print_every == 0)
    };
    if !should_print {
        return;
    }
    let mut entries = {
        let Ok(mut state) = cpu_session_profile_state().lock() else {
            return;
        };
        let entries = state
            .iter()
            .map(|(section, entry)| (*section, entry.clone()))
            .collect::<Vec<_>>();
        state.clear();
        entries
    };
    entries.sort_by_key(|(_, entry)| Reverse(entry.total_time));
    eprintln!("=== tenferro CPU session profile ===");
    for (section, entry) in entries {
        eprintln!(
            "{section}: calls={} total={:.6}ms per_call={:.3}us",
            entry.calls,
            entry.total_time.as_secs_f64() * 1.0e3,
            entry.total_time.as_secs_f64() * 1.0e6 / entry.calls as f64,
        );
    }
}

struct BufferPoolLoan<'a> {
    buffers: &'a mut BufferPool,
}

impl<'a> BufferPoolLoan<'a> {
    fn new(buffers: &'a mut BufferPool) -> Self {
        Self { buffers }
    }

    fn get_mut(&mut self) -> &mut BufferPool {
        self.buffers
    }
}

impl Drop for BufferPoolLoan<'_> {
    fn drop(&mut self) {
        if thread::panicking() {
            self.buffers.replenish_in_flight_retained();
        } else {
            self.buffers.clear_in_flight_retained();
        }
    }
}

/// CPU provider selected by a [`CpuBackend`] instance.
///
/// CPU provider features are additive at compile time; this runtime selector
/// chooses which compiled provider an individual backend uses for provider-owned
/// kernels such as GEMM.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackendKind;
///
/// let kind = CpuBackendKind::default_compiled();
/// assert!(matches!(kind, CpuBackendKind::Faer | CpuBackendKind::Blas));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CpuBackendKind {
    /// faer-backed CPU kernels.
    Faer,
    /// BLAS/LAPACK-backed CPU kernels.
    Blas,
}

impl CpuBackendKind {
    /// Return the default compiled CPU provider.
    ///
    /// BLAS is preferred when both BLAS and faer are compiled in because an
    /// application that links a BLAS/LAPACK provider normally expects
    /// provider-backed kernels to use it by default.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackendKind;
    ///
    /// let _kind = CpuBackendKind::default_compiled();
    /// ```
    pub fn default_compiled() -> Self {
        #[cfg(feature = "cpu-blas")]
        {
            Self::Blas
        }
        #[cfg(all(not(feature = "cpu-blas"), feature = "cpu-faer"))]
        {
            Self::Faer
        }
    }

    // Used by feature-specific diagnostics; some feature combinations leave
    // the formatter path inactive.
    #[allow(dead_code)]
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Faer => "faer",
            Self::Blas => "blas",
        }
    }
}

/// Stable execution-ownership mode selected for a CPU backend handle.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::{CpuBackend, CpuExecutionMode};
///
/// let mode = CpuBackend::new().execution_info().execution_mode();
/// assert!(matches!(
///     mode,
///     CpuExecutionMode::Managed
///         | CpuExecutionMode::ExternalManaged
///         | CpuExecutionMode::ProviderDefaultExclusive
///         | CpuExecutionMode::Compatibility
/// ));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CpuExecutionMode {
    /// tenferro owns a pinned Rayon engine for the resolved CPU placement.
    Managed,
    /// The application supplied and owns the selected CPU domain executor.
    ExternalManaged,
    /// An external provider owns worker placement under a process-wide permit.
    ProviderDefaultExclusive,
    /// A legacy unpinned Rayon context is used because managed affinity is unavailable.
    Compatibility,
}

/// Failure to construct an externally managed CPU-domain registry.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::ExternalCpuDomainRegistryError;
///
/// let error = ExternalCpuDomainRegistryError::EmptyRegistry;
/// assert!(error.to_string().contains("at least one"));
/// ```
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ExternalCpuDomainRegistryError {
    /// No external domain descriptor was supplied.
    #[error("externally managed CPU registry must contain at least one domain")]
    EmptyRegistry,
    /// More than one descriptor used the same caller-stable domain ID.
    #[error("CPU domain ID {id:?} is registered more than once")]
    DuplicateDomainId {
        /// Duplicate caller-supplied identity.
        id: CpuDomainId,
    },
    /// More than one descriptor claimed the same placement identity.
    #[error("CPU placement {placement:?} is registered more than once")]
    DuplicatePlacementIdentity {
        /// Duplicate NUMA-node or all-allowed identity.
        placement: CpuPlacement,
    },
    /// A declared CPU is outside the process-allowed CPU set.
    #[error("CPU domain {domain:?} declares process-disallowed CPU {cpu}")]
    CpuOutsideAllowedSet {
        /// Domain containing the invalid CPU declaration.
        domain: CpuDomainId,
        /// CPU absent from the process affinity set.
        cpu: CpuId,
    },
    /// The selected default domain ID was not supplied.
    #[error("default CPU domain {default_domain:?} is not registered")]
    MissingDefaultDomain {
        /// Missing caller-selected default identity.
        default_domain: CpuDomainId,
    },
    /// An exact all-allowed declaration did not equal the process-allowed set.
    #[error(
        "exact all-allowed CPU domain {domain:?} declares {declared:?}, but the process allows {allowed:?}"
    )]
    ExactAllAllowedMismatch {
        /// Domain with the inconsistent all-allowed declaration.
        domain: CpuDomainId,
        /// CPUs declared by the external descriptor.
        declared: CpuSet,
        /// CPUs allowed by the current process affinity mask.
        allowed: CpuSet,
    },
}

/// Errors returned while constructing a [`CpuBackend`].
///
/// Placement failures remain typed so callers can distinguish topology
/// discovery failures from unsupported placement requests. Configuration and
/// provider-selection failures retain the existing tensor error contract.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::{CpuBackend, CpuBackendError};
///
/// let error = CpuBackend::with_threads(0).unwrap_err();
/// assert!(matches!(error, CpuBackendError::Tensor(_)));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum CpuBackendError {
    /// CPU context configuration or provider selection failed.
    #[error(transparent)]
    Tensor(#[from] crate::Error),
    /// CPU placement resolution or engine construction failed.
    #[error("{op}: {source}")]
    Placement {
        /// Constructor that observed the placement failure.
        op: &'static str,
        /// Typed placement failure.
        #[source]
        source: CpuPlacementError,
    },
    /// Externally managed domain registry validation failed.
    #[error(transparent)]
    ExternalRegistry(#[from] ExternalCpuDomainRegistryError),
}

impl CpuBackendError {
    fn placement(op: &'static str, source: CpuPlacementError) -> Self {
        Self::Placement { op, source }
    }

    /// Return the typed placement failure, when construction reached placement resolution.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuBackend, CpuBackendError};
    ///
    /// let result: Result<CpuBackend, CpuBackendError> = CpuBackend::with_threads(1);
    /// if let Err(error) = result {
    ///     let _placement_failure = error.placement_error();
    /// }
    /// ```
    pub fn placement_error(&self) -> Option<&CpuPlacementError> {
        match self {
            Self::Tensor(_) => None,
            Self::Placement { source, .. } => Some(source),
            Self::ExternalRegistry(_) => None,
        }
    }
}

impl From<CpuBackendError> for crate::Error {
    fn from(error: CpuBackendError) -> Self {
        match error {
            CpuBackendError::Tensor(error) => error,
            CpuBackendError::ExternalRegistry(source) => Self::extension(
                "CpuBackend::from_external_managed_domains",
                "cpu",
                crate::ErrorKind::Validation(crate::ValidationKind::InvalidArgument),
                source,
            ),
            CpuBackendError::Placement { op, source } => match source {
                CpuPlacementError::TopologyDiscovery { .. }
                | CpuPlacementError::ManagedAffinityUnavailable { .. }
                | CpuPlacementError::NumaDiscoveryUnavailable { .. }
                | CpuPlacementError::UnknownNumaNode { .. }
                | CpuPlacementError::UnregisteredExternalPlacement { .. } => {
                    Self::runtime_state_source(op, source)
                }
                CpuPlacementError::ExternalProviderAffinityUnmanaged { .. } => {
                    Self::extension(op, "cpu", crate::ErrorKind::Unsupported, source)
                }
                CpuPlacementError::EngineConstruction { .. } => Self::backend_source(op, source),
                CpuPlacementError::InternalState { .. } => {
                    Self::extension(op, "cpu", crate::ErrorKind::Internal, source)
                }
            },
        }
    }
}

/// Snapshot of the stable CPU execution contract and non-contractual provider diagnostics.
///
/// [`CpuBackendKind`] is the stable provider identity. The diagnostic string is
/// intended for logs and may change between builds or releases.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::{CpuBackend, CpuPlacement};
///
/// let info = CpuBackend::new().execution_info();
/// assert_eq!(info.requested_placement(), CpuPlacement::Auto);
/// assert!(!info.provider_diagnostic().is_empty());
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuExecutionInfo {
    backend_kind: CpuBackendKind,
    execution_mode: CpuExecutionMode,
    requested_placement: CpuPlacement,
    resolved_placement: Option<ResolvedCpuPlacement>,
    topology: CpuTopology,
    domain_id: CpuDomainId,
    domain_cpus: CpuSet,
    worker_count: usize,
    thread_budget: usize,
    placement_guarantee: CpuPlacementGuarantee,
    domain_ownership: CpuDomainOwnership,
    executor_affinity: CpuExecutorAffinity,
    executor_shutdown: CpuExecutorShutdown,
    provider_diagnostic: &'static str,
}

impl CpuExecutionInfo {
    /// Return the stable public provider identity.
    ///
    /// # Examples
    ///
    /// ```
    /// let info = tenferro_cpu::CpuBackend::new().execution_info();
    /// assert_eq!(info.backend_kind(), tenferro_cpu::CpuBackend::new().kind());
    /// ```
    pub fn backend_kind(&self) -> CpuBackendKind {
        self.backend_kind
    }

    /// Return the stable execution-ownership mode.
    ///
    /// # Examples
    ///
    /// ```
    /// let mode = tenferro_cpu::CpuBackend::new()
    ///     .execution_info()
    ///     .execution_mode();
    /// let _ = format!("{mode:?}");
    /// ```
    pub fn execution_mode(&self) -> CpuExecutionMode {
        self.execution_mode
    }

    /// Return the placement requested by this backend handle.
    ///
    /// # Examples
    ///
    /// ```
    /// let info = tenferro_cpu::CpuBackend::new().execution_info();
    /// assert_eq!(info.requested_placement(), tenferro_cpu::CpuPlacement::Auto);
    /// ```
    pub fn requested_placement(&self) -> CpuPlacement {
        self.requested_placement
    }

    /// Return the concrete managed placement or external placement declaration.
    ///
    /// # Examples
    ///
    /// ```
    /// let backend = tenferro_cpu::CpuBackend::new();
    /// let _managed = backend.execution_info().resolved_placement();
    /// ```
    pub fn resolved_placement(&self) -> Option<&ResolvedCpuPlacement> {
        self.resolved_placement.as_ref()
    }

    /// Return the process-visible topology used for placement resolution.
    ///
    /// # Examples
    ///
    /// ```
    /// let info = tenferro_cpu::CpuBackend::new().execution_info();
    /// assert!(!info.topology().allowed_cpus().is_empty());
    /// ```
    pub fn topology(&self) -> &CpuTopology {
        &self.topology
    }

    /// Return the coordinator-stable identity of the selected CPU domain.
    ///
    /// # Examples
    ///
    /// ```
    /// let id = tenferro_cpu::CpuBackend::new().execution_info().domain_id();
    /// let _ = id.as_u64();
    /// ```
    pub fn domain_id(&self) -> CpuDomainId {
        self.domain_id
    }

    /// Return the resolved or caller-declared logical CPUs of the selected domain.
    ///
    /// # Examples
    ///
    /// ```
    /// let info = tenferro_cpu::CpuBackend::new().execution_info();
    /// assert!(!info.domain_cpus().is_empty());
    /// ```
    pub fn domain_cpus(&self) -> &CpuSet {
        &self.domain_cpus
    }

    /// Return the worker count of the selected domain executor.
    ///
    /// # Examples
    ///
    /// ```
    /// let info = tenferro_cpu::CpuBackend::new().execution_info();
    /// assert!(info.worker_count() >= 1);
    /// ```
    pub fn worker_count(&self) -> usize {
        self.worker_count
    }

    /// Return the maximum number of participating threads requested for this domain.
    ///
    /// This can be smaller than [`Self::worker_count`] for an externally
    /// supplied executor.
    ///
    /// # Examples
    ///
    /// ```
    /// let info = tenferro_cpu::CpuBackend::new().execution_info();
    /// assert!(info.thread_budget() >= 1);
    /// assert!(info.thread_budget() <= info.worker_count());
    /// ```
    pub fn thread_budget(&self) -> usize {
        self.thread_budget
    }

    /// Return whether the selected placement is exact or advisory.
    ///
    /// # Examples
    ///
    /// ```
    /// let guarantee = tenferro_cpu::CpuBackend::new()
    ///     .execution_info()
    ///     .placement_guarantee();
    /// let _ = format!("{guarantee:?}");
    /// ```
    pub fn placement_guarantee(&self) -> CpuPlacementGuarantee {
        self.placement_guarantee
    }

    /// Return whether tenferro or the application owns the selected domain.
    ///
    /// # Examples
    ///
    /// ```
    /// let ownership = tenferro_cpu::CpuBackend::new()
    ///     .execution_info()
    ///     .domain_ownership();
    /// let _ = format!("{ownership:?}");
    /// ```
    pub fn domain_ownership(&self) -> CpuDomainOwnership {
        self.domain_ownership
    }

    /// Return the selected executor's worker-affinity claim.
    ///
    /// # Examples
    ///
    /// ```
    /// let affinity = tenferro_cpu::CpuBackend::new()
    ///     .execution_info()
    ///     .executor_affinity();
    /// let _ = format!("{affinity:?}");
    /// ```
    pub fn executor_affinity(&self) -> CpuExecutorAffinity {
        self.executor_affinity
    }

    /// Return who owns shutdown of the selected executor.
    ///
    /// # Examples
    ///
    /// ```
    /// let shutdown = tenferro_cpu::CpuBackend::new()
    ///     .execution_info()
    ///     .executor_shutdown();
    /// let _ = format!("{shutdown:?}");
    /// ```
    pub fn executor_shutdown(&self) -> CpuExecutorShutdown {
        self.executor_shutdown
    }

    /// Return a human-readable provider description for logs.
    ///
    /// This string is diagnostic only and is not a provider identity contract.
    ///
    /// # Examples
    ///
    /// ```
    /// let diagnostic = tenferro_cpu::CpuBackend::new()
    ///     .execution_info()
    ///     .provider_diagnostic();
    /// assert!(!diagnostic.is_empty());
    /// ```
    pub fn provider_diagnostic(&self) -> &'static str {
        self.provider_diagnostic
    }
}

fn provider_diagnostic(kind: CpuBackendKind, ownership: CpuDomainOwnership) -> &'static str {
    if ownership == CpuDomainOwnership::ExternalManaged {
        return match kind {
            CpuBackendKind::Faer => "faer (externally managed CPU executor)",
            CpuBackendKind::Blas => "BLAS/LAPACK (externally managed CPU executor)",
        };
    }
    match kind {
        CpuBackendKind::Faer => "faer (tenferro-managed Rayon affinity)",
        CpuBackendKind::Blas => {
            #[cfg(feature = "blas-openblas")]
            return "OpenBLAS (external worker affinity)";
            #[cfg(feature = "blas-mkl")]
            return "Intel MKL (external worker affinity)";
            #[cfg(feature = "blas-accelerate")]
            return "Apple Accelerate (external worker affinity)";
            #[cfg(feature = "provider-inject")]
            return "runtime-injected BLAS/LAPACK (external worker affinity)";
            #[cfg(not(any(
                feature = "blas-openblas",
                feature = "blas-mkl",
                feature = "blas-accelerate",
                feature = "provider-inject"
            )))]
            return "linked BLAS/LAPACK provider (identity unknown; external worker affinity)";
        }
    }
}

fn ensure_cpu_backend_kind_available(kind: CpuBackendKind, op: &'static str) -> crate::Result<()> {
    let _ = op;
    match kind {
        CpuBackendKind::Faer => {
            #[cfg(feature = "cpu-faer")]
            {
                Ok(())
            }
            #[cfg(not(feature = "cpu-faer"))]
            {
                Err(crate::Error::invalid_argument(
                    op,
                    "configuration",
                    "CpuBackendKind::Faer requires the cpu-faer feature".to_string(),
                ))
            }
        }
        CpuBackendKind::Blas => {
            #[cfg(feature = "cpu-blas")]
            {
                Ok(())
            }
            #[cfg(not(feature = "cpu-blas"))]
            {
                Err(crate::Error::invalid_argument(
                    op,
                    "configuration",
                    "CpuBackendKind::Blas requires the cpu-blas feature".to_string(),
                ))
            }
        }
    }
}

fn constructor_tensor_error(op: &'static str, error: crate::Error) -> CpuBackendError {
    CpuBackendError::Tensor(match error {
        crate::Error::Validation { source, .. } => crate::Error::validation(op, source),
        error => error,
    })
}

// Used by feature-disabled backend paths; a given feature build may compile no
// direct call site for one provider.
#[allow(dead_code)]
pub(super) fn unavailable_cpu_backend_kind(kind: CpuBackendKind, op: &'static str) -> crate::Error {
    crate::Error::invalid_argument(
        op,
        "configuration",
        format!("CPU backend kind {} is not compiled in", kind.name()),
    )
}

struct ManagedEngineRegistry {
    node_engines: Mutex<BTreeMap<NumaNodeId, Arc<CpuEngine>>>,
    node_domain_ids: BTreeMap<NumaNodeId, CpuDomainId>,
    all_allowed: OnceLock<Arc<CpuEngine>>,
    all_allowed_build: Mutex<()>,
    base_engine: Arc<CpuEngine>,
    thread_budget: usize,
}

struct ExternalEngineRegistry {
    by_id: BTreeMap<CpuDomainId, Arc<CpuEngine>>,
    by_node: BTreeMap<NumaNodeId, Arc<CpuEngine>>,
    all_allowed: Option<Arc<CpuEngine>>,
    default_domain: CpuDomainId,
}

enum CpuEngineRegistry {
    ManagedLazy(ManagedEngineRegistry),
    ExternalPrebuilt(ExternalEngineRegistry),
}

struct CpuBackendState {
    topology: CpuTopology,
    engines: CpuEngineRegistry,
    arbiter: ResourceArbiter,
    kind: CpuBackendKind,
    buffer_limit: AtomicUsize,
    indexed_plan_cache_limits: Mutex<IndexedPlanCacheLimits>,
}

impl CpuBackendState {
    fn managed_engine_for(
        &self,
        placement: &ResolvedCpuPlacement,
        requested: CpuPlacement,
    ) -> Result<Arc<CpuEngine>, CpuPlacementError> {
        // INVARIANT: cache configuration is the outermost lock for lazy engine
        // creation and limit updates. The shared order is configuration,
        // registry, then engine resources.
        let cache_configuration = self.indexed_plan_cache_limits.lock().map_err(|_| {
            CpuPlacementError::InternalState {
                requested,
                backend: self.kind,
                message: "CPU indexed-plan cache configuration lock is poisoned",
            }
        })?;
        let cache_limits = *cache_configuration;
        let CpuEngineRegistry::ManagedLazy(registry) = &self.engines else {
            return Err(CpuPlacementError::InternalState {
                requested,
                backend: self.kind,
                message: "managed placement requested from an external engine registry",
            });
        };
        match placement {
            ResolvedCpuPlacement::NumaNode { id, .. } => {
                let mut engines = registry
                    .node_engines
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Some(engine) = engines.get(id) {
                    return Ok(Arc::clone(engine));
                }
                let Some(domain_id) = registry.node_domain_ids.get(id).copied() else {
                    return Err(CpuPlacementError::InternalState {
                        requested,
                        backend: self.kind,
                        message: "managed NUMA node has no coordinator-stable domain ID",
                    });
                };
                let engine = Arc::new(
                    CpuEngine::new_managed(
                        domain_id,
                        placement.clone(),
                        registry.thread_budget,
                        self.buffer_limit.load(Ordering::Relaxed),
                    )
                    .map_err(|error| {
                        CpuPlacementError::EngineConstruction {
                            requested,
                            backend: self.kind,
                            source: CpuEngineConstructionError::Context(error),
                        }
                    })?,
                );
                self.configure_new_indexed_plan_cache(&engine, requested, cache_limits)?;
                engines.insert(*id, Arc::clone(&engine));
                Ok(engine)
            }
            ResolvedCpuPlacement::AllAllowed { .. } => {
                if let Some(engine) = registry.all_allowed.get() {
                    return Ok(Arc::clone(engine));
                }
                let _build = registry
                    .all_allowed_build
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Some(engine) = registry.all_allowed.get() {
                    return Ok(Arc::clone(engine));
                }
                let engine = Arc::new(
                    CpuEngine::new_managed(
                        CpuDomainId::new(0),
                        placement.clone(),
                        registry.thread_budget,
                        self.buffer_limit.load(Ordering::Relaxed),
                    )
                    .map_err(|error| {
                        CpuPlacementError::EngineConstruction {
                            requested,
                            backend: self.kind,
                            source: CpuEngineConstructionError::Context(error),
                        }
                    })?,
                );
                self.configure_new_indexed_plan_cache(&engine, requested, cache_limits)?;
                let _ = registry.all_allowed.set(Arc::clone(&engine));
                Ok(engine)
            }
        }
    }

    fn configure_new_indexed_plan_cache(
        &self,
        engine: &CpuEngine,
        requested: CpuPlacement,
        limits: IndexedPlanCacheLimits,
    ) -> Result<(), CpuPlacementError> {
        let mut resources =
            engine
                .resources
                .lock()
                .map_err(|_| CpuPlacementError::InternalState {
                    requested,
                    backend: self.kind,
                    message: "new CPU engine indexed-plan cache lock is poisoned",
                })?;
        resources.indexed_plan_cache.set_limits(limits);
        Ok(())
    }

    fn managed_base_engine(
        &self,
        requested: CpuPlacement,
    ) -> Result<Arc<CpuEngine>, CpuPlacementError> {
        match &self.engines {
            CpuEngineRegistry::ManagedLazy(registry) => Ok(Arc::clone(&registry.base_engine)),
            CpuEngineRegistry::ExternalPrebuilt(_) => Err(CpuPlacementError::InternalState {
                requested,
                backend: self.kind,
                message: "managed compatibility placement requested from an external registry",
            }),
        }
    }

    fn external_engine_for(
        &self,
        requested: CpuPlacement,
    ) -> Result<Arc<CpuEngine>, CpuPlacementError> {
        let CpuEngineRegistry::ExternalPrebuilt(registry) = &self.engines else {
            return Err(CpuPlacementError::InternalState {
                requested,
                backend: self.kind,
                message: "external placement requested from a managed engine registry",
            });
        };
        let engine = match requested {
            CpuPlacement::Auto => registry.by_id.get(&registry.default_domain),
            CpuPlacement::NumaNode(id) => registry.by_node.get(&id),
            CpuPlacement::AllAllowed => registry.all_allowed.as_ref(),
        };
        engine
            .cloned()
            .ok_or(CpuPlacementError::UnregisteredExternalPlacement { requested })
    }

    fn is_external(&self) -> bool {
        matches!(&self.engines, CpuEngineRegistry::ExternalPrebuilt(_))
    }

    fn initialized_engines(&self, op: &'static str) -> crate::Result<Vec<Arc<CpuEngine>>> {
        let mut engines = match &self.engines {
            CpuEngineRegistry::ManagedLazy(registry) => {
                let mut engines = vec![Arc::clone(&registry.base_engine)];
                if let Some(engine) = registry.all_allowed.get() {
                    engines.push(Arc::clone(engine));
                }
                engines.extend(
                    registry
                        .node_engines
                        .lock()
                        .map_err(|_| poisoned_cpu_lock(op, "CPU engine registry"))?
                        .values()
                        .cloned(),
                );
                engines
            }
            CpuEngineRegistry::ExternalPrebuilt(registry) => {
                registry.by_id.values().cloned().collect()
            }
        };
        if engines.len() > 1 {
            engines.sort_unstable_by_key(|engine| Arc::as_ptr(engine) as usize);
            engines.dedup_by(|left, right| Arc::ptr_eq(left, right));
        }
        Ok(engines)
    }
}

fn poisoned_cpu_lock(op: &'static str, lock: &'static str) -> crate::Error {
    crate::Error::runtime_state(op, format!("{lock} lock poisoned"))
}

fn lock_engine_resources<'a>(
    engine: &'a CpuEngine,
    op: &'static str,
) -> crate::Result<std::sync::MutexGuard<'a, EngineResources>> {
    engine
        .resources
        .lock()
        .map_err(|_| poisoned_cpu_lock(op, "CPU engine resources"))
}

fn saturating_add_tensor_cache_stats(total: &mut CacheStats, value: CacheStats) {
    total.entries = total.entries.saturating_add(value.entries);
    total.retained_bytes = total.retained_bytes.saturating_add(value.retained_bytes);
    total.hits = total.hits.saturating_add(value.hits);
    total.misses = total.misses.saturating_add(value.misses);
    total.evictions = total.evictions.saturating_add(value.evictions);
    total.clears = total.clears.saturating_add(value.clears);
}

/// A cheap cloneable handle to shared CPU execution coordination.
///
/// Clones share topology, execution engines, arbitration, and engine-owned
/// buffer resources.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
///
/// let backend = CpuBackend::new();
/// let clone = backend.clone();
/// assert_eq!(backend.kind(), clone.kind());
/// ```
#[derive(Clone)]
pub struct CpuBackend {
    runtime_identity: CpuRuntimeIdentity,
    shared: Arc<CpuBackendState>,
    requested: CpuPlacement,
    resolved: ResolvedCpuExecution,
    engine: Arc<CpuEngine>,
    provider_bundle: CpuProviderBundle,
    allocation_domain: Option<Arc<dyn SharedTensorAllocationDomain>>,
}

/// Opaque identity for one CPU backend executable witness.
///
/// The token carries no backend, execution, storage, or mutation authority.
/// Cloning a token is cheap and preserves identity; separately constructed
/// backends and backends returned after immutable witness resources change use
/// distinct tokens.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
///
/// let identity = CpuBackend::new().runtime_identity();
/// assert_eq!(identity, identity.clone());
/// ```
#[derive(Clone, Debug)]
pub struct CpuRuntimeIdentity {
    marker: Arc<()>,
}

impl CpuRuntimeIdentity {
    fn fresh() -> Self {
        Self {
            marker: Arc::new(()),
        }
    }
}

impl PartialEq for CpuRuntimeIdentity {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.marker, &other.marker)
    }
}

impl Eq for CpuRuntimeIdentity {}

fn resolve_discovered_topology(
    kind: CpuBackendKind,
    topology: Result<CpuTopology, CpuTopologyError>,
) -> Result<CpuTopology, CpuPlacementError> {
    topology.map_err(|source| CpuPlacementError::TopologyDiscovery {
        requested: CpuPlacement::Auto,
        backend: kind,
        source,
    })
}

fn coordinator_node_domain_ids(topology: &CpuTopology) -> BTreeMap<NumaNodeId, CpuDomainId> {
    topology
        .nodes()
        .iter()
        .enumerate()
        .filter_map(|(index, node)| {
            u64::try_from(index)
                .ok()
                .and_then(|index| index.checked_add(1))
                .map(|id| (node.id(), CpuDomainId::new(id)))
        })
        .collect()
}

impl fmt::Debug for CpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuBackend")
            .field("kind", &self.kind())
            .field("provider_bundle", &self.provider_bundle)
            .field("requested_placement", &self.requested)
            .field("resolved_execution", &self.resolved)
            .field("engine_placement", &self.engine.placement())
            .field("num_threads", &self.num_threads())
            .field("allocation_domain", &self.allocation_domain())
            .field("buffer_pool_cache_stats", &self.buffer_pool_cache_stats())
            .field("buffer_pool_limit_bytes", &self.buffer_pool_limit_bytes())
            .finish_non_exhaustive()
    }
}

impl CpuBackend {
    fn from_thread_budget_and_kind(
        thread_budget: usize,
        kind: CpuBackendKind,
        max_retained_capacity_bytes: usize,
    ) -> Result<Self, CpuPlacementError> {
        let topology = resolve_discovered_topology(kind, discover_cpu_topology())?;
        let resolved = resolve_placement(kind, CpuPlacement::Auto, &topology)?;
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        {
            let context = CpuContext::with_threads(thread_budget).map_err(|error| {
                CpuPlacementError::EngineConstruction {
                    requested: CpuPlacement::Auto,
                    backend: kind,
                    source: CpuEngineConstructionError::Tensor(error),
                }
            })?;
            Ok(Self::compatibility_with_topology(
                Arc::new(context),
                max_retained_capacity_bytes,
                kind,
                topology,
                resolved,
            ))
        }
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            let engine_placement = ResolvedCpuPlacement::AllAllowed {
                cpus: topology.allowed_cpus().clone(),
            };
            let engine = Arc::new(
                CpuEngine::new_managed(
                    CpuDomainId::new(0),
                    engine_placement,
                    thread_budget,
                    max_retained_capacity_bytes,
                )
                .map_err(|error| CpuPlacementError::EngineConstruction {
                    requested: CpuPlacement::Auto,
                    backend: kind,
                    source: CpuEngineConstructionError::Context(error),
                })?,
            );
            let all_allowed = OnceLock::new();
            let _ = all_allowed.set(Arc::clone(&engine));
            Ok(Self {
                shared: Arc::new(CpuBackendState {
                    engines: CpuEngineRegistry::ManagedLazy(ManagedEngineRegistry {
                        node_engines: Mutex::new(BTreeMap::new()),
                        node_domain_ids: coordinator_node_domain_ids(&topology),
                        all_allowed,
                        all_allowed_build: Mutex::new(()),
                        base_engine: Arc::clone(&engine),
                        thread_budget,
                    }),
                    topology,
                    arbiter: ResourceArbiter::global(),
                    kind,
                    buffer_limit: AtomicUsize::new(max_retained_capacity_bytes),
                    indexed_plan_cache_limits: Mutex::new(DEFAULT_INDEXED_PLAN_CACHE_LIMITS),
                }),
                runtime_identity: CpuRuntimeIdentity::fresh(),
                requested: CpuPlacement::Auto,
                resolved,
                engine,
                provider_bundle: CpuProviderBundle::standard(kind, kind == CpuBackendKind::Blas),
                allocation_domain: None,
            })
        }
    }

    fn compatibility(
        ctx: Arc<CpuContext>,
        max_retained_capacity_bytes: usize,
        kind: CpuBackendKind,
    ) -> Self {
        let topology = discover_cpu_topology().unwrap_or_else(|_| {
            let allowed = crate::process_cpu_affinity().unwrap_or_else(|| {
                CpuSet::new((0..crate::available_parallelism()).map(CpuId::new))
                    .unwrap_or_else(|_| CpuSet::singleton(CpuId::new(0)))
            });
            CpuTopology::all_allowed(allowed)
        });
        let resolved = if kind == CpuBackendKind::Blas {
            ResolvedCpuExecution::ProviderDefaultExclusive
        } else {
            ResolvedCpuExecution::Compatibility
        };
        Self::compatibility_with_topology(
            ctx,
            max_retained_capacity_bytes,
            kind,
            topology,
            resolved,
        )
    }

    fn compatibility_with_topology(
        ctx: Arc<CpuContext>,
        max_retained_capacity_bytes: usize,
        kind: CpuBackendKind,
        topology: CpuTopology,
        resolved: ResolvedCpuExecution,
    ) -> Self {
        let placement = ResolvedCpuPlacement::AllAllowed {
            cpus: topology.allowed_cpus().clone(),
        };
        let base_engine = Arc::new(CpuEngine::from_context(
            CpuDomainId::new(0),
            placement,
            ctx,
            max_retained_capacity_bytes,
        ));
        Self {
            shared: Arc::new(CpuBackendState {
                engines: CpuEngineRegistry::ManagedLazy(ManagedEngineRegistry {
                    node_engines: Mutex::new(BTreeMap::new()),
                    node_domain_ids: coordinator_node_domain_ids(&topology),
                    all_allowed: OnceLock::new(),
                    all_allowed_build: Mutex::new(()),
                    base_engine: Arc::clone(&base_engine),
                    thread_budget: base_engine.domain().thread_budget().get(),
                }),
                topology,
                arbiter: ResourceArbiter::global(),
                kind,
                buffer_limit: AtomicUsize::new(max_retained_capacity_bytes),
                indexed_plan_cache_limits: Mutex::new(DEFAULT_INDEXED_PLAN_CACHE_LIMITS),
            }),
            runtime_identity: CpuRuntimeIdentity::fresh(),
            requested: CpuPlacement::Auto,
            resolved,
            engine: base_engine,
            provider_bundle: CpuProviderBundle::standard(kind, kind == CpuBackendKind::Blas),
            allocation_domain: None,
        }
    }

    /// Create a CPU backend using the environment-driven CPU context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// ```
    pub fn new() -> Self {
        let context = Arc::new(CpuContext::from_env());
        Self::from_thread_budget_and_kind(
            context.num_threads(),
            CpuBackendKind::default_compiled(),
            crate::buffer_pool::DEFAULT_MAX_RETAINED_CAPACITY_BYTES,
        )
        .unwrap_or_else(|error| {
            eprintln!(
                "tenferro_cpu: using the unpinned compatibility context after placement error: {error}"
            );
            Self::from_context(context)
        })
    }

    /// Create one coordinator from caller-owned CPU domain executors.
    ///
    /// The descriptors are moved into prebuilt engines. `Auto` selects
    /// `default_domain`; explicit placement requests are registry-only and
    /// never construct a managed context or thread pool.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use std::sync::Arc;
    /// use tenferro_cpu::{
    ///     discover_cpu_topology, CpuBackend, CpuBackendError, CpuContext,
    ///     CpuExecutionMode, CpuPlacementGuarantee, CpuProviderBundleInstallError,
    ///     ExternalCpuDomain, ResolvedCpuPlacement,
    /// };
    /// use tenferro_tensor::CpuDomainId;
    ///
    /// let topology = discover_cpu_topology()?;
    /// let id = CpuDomainId::new(7);
    /// let domain = ExternalCpuDomain::new(
    ///     id,
    ///     ResolvedCpuPlacement::AllAllowed {
    ///         cpus: topology.allowed_cpus().clone(),
    ///     },
    ///     Arc::new(CpuContext::with_threads(1)?),
    ///     NonZeroUsize::new(1).unwrap(),
    ///     CpuPlacementGuarantee::AdvisoryDeclared,
    /// )?;
    /// match CpuBackend::from_external_managed_domains(id, [domain]) {
    ///     Ok(backend) => assert_eq!(
    ///         backend.execution_info().execution_mode(),
    ///         CpuExecutionMode::ExternalManaged,
    ///     ),
    ///     Err(CpuBackendError::Tensor(error)) => assert!(
    ///         std::error::Error::source(&error)
    ///             .and_then(|source| source.downcast_ref::<CpuProviderBundleInstallError>())
    ///             .is_some(),
    ///         "an uncontrolled compiled provider must retain its typed source",
    ///     ),
    ///     Err(error) => return Err(error.into()),
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CpuBackendError::Placement`] when process topology discovery
    /// fails. Returns [`CpuBackendError::ExternalRegistry`] for an empty
    /// registry, duplicate domain or placement identity, a CPU outside the
    /// process-allowed set, a missing default domain, or an exact
    /// [`ResolvedCpuPlacement::AllAllowed`] declaration that differs from the
    /// process-allowed CPU set. Returns [`CpuBackendError::Tensor`] with a
    /// [`CpuProviderBundleInstallError`] source when the compiled standard
    /// provider cannot satisfy an external domain contract. Applications that
    /// supply controlled providers can use
    /// [`CpuBackend::from_external_managed_domains_with_provider_bundle`].
    pub fn from_external_managed_domains(
        default_domain: CpuDomainId,
        domains: impl IntoIterator<Item = ExternalCpuDomain>,
    ) -> Result<Self, CpuBackendError> {
        let op = "CpuBackend::from_external_managed_domains";
        let kind = CpuBackendKind::default_compiled();
        let topology = resolve_discovered_topology(kind, discover_cpu_topology())
            .map_err(|source| CpuBackendError::placement(op, source))?;
        Self::from_external_managed_domains_with_topology_arbiter_and_provider_bundle(
            default_domain,
            domains,
            topology,
            ResourceArbiter::global(),
            CpuProviderBundle::standard(kind, false),
        )
    }

    /// Create one coordinator from caller-owned CPU domain executors and an
    /// immutable provider bundle.
    ///
    /// Domain registry construction and provider compatibility validation are
    /// atomic: no backend is returned unless `provider_bundle` satisfies every
    /// supplied domain. The bundle currently selects `dot_general` operation-
    /// family providers; linalg operation-family selection still follows the
    /// compiled [`CpuBackendKind`] and is not replaced by this API.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use std::sync::Arc;
    /// use tenferro_cpu::{
    ///     discover_cpu_topology, CpuBackend, CpuBackendKind, CpuContext,
    ///     CpuExecutionMode, CpuPlacementGuarantee, CpuProviderBundle,
    ///     ExternalCpuDomain, ResolvedCpuPlacement,
    /// };
    /// use tenferro_tensor::CpuDomainId;
    ///
    /// let topology = discover_cpu_topology()?;
    /// let id = CpuDomainId::new(7);
    /// let domain = ExternalCpuDomain::new(
    ///     id,
    ///     ResolvedCpuPlacement::AllAllowed {
    ///         cpus: topology.allowed_cpus().clone(),
    ///     },
    ///     Arc::new(CpuContext::with_threads(1)?),
    ///     NonZeroUsize::new(1).unwrap(),
    ///     CpuPlacementGuarantee::AdvisoryDeclared,
    /// )?;
    /// let bundle = CpuProviderBundle::builder(CpuBackendKind::Faer).build()?;
    /// let backend = CpuBackend::from_external_managed_domains_with_provider_bundle(
    ///     id,
    ///     [domain],
    ///     bundle.clone(),
    /// )?;
    /// assert_eq!(
    ///     backend.execution_info().execution_mode(),
    ///     CpuExecutionMode::ExternalManaged,
    /// );
    /// assert!(backend.provider_bundle().shares_identity_with(&bundle));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns the same topology and registry errors as
    /// [`CpuBackend::from_external_managed_domains`]. Provider incompatibility
    /// is returned as [`CpuBackendError::Tensor`]. Calling
    /// [`std::error::Error::source`] on that value yields the typed
    /// [`CpuProviderBundleInstallError`], whose own source is the rejected
    /// [`crate::CpuProviderDomainError`].
    pub fn from_external_managed_domains_with_provider_bundle(
        default_domain: CpuDomainId,
        domains: impl IntoIterator<Item = ExternalCpuDomain>,
        provider_bundle: CpuProviderBundle,
    ) -> Result<Self, CpuBackendError> {
        let op = "CpuBackend::from_external_managed_domains_with_provider_bundle";
        let kind = CpuBackendKind::default_compiled();
        let topology = resolve_discovered_topology(kind, discover_cpu_topology())
            .map_err(|source| CpuBackendError::placement(op, source))?;
        Self::from_external_managed_domains_with_topology_arbiter_and_provider_bundle(
            default_domain,
            domains,
            topology,
            ResourceArbiter::global(),
            provider_bundle,
        )
    }

    fn from_external_managed_domains_with_topology_arbiter_and_provider_bundle(
        default_domain: CpuDomainId,
        domains: impl IntoIterator<Item = ExternalCpuDomain>,
        topology: CpuTopology,
        arbiter: ResourceArbiter,
        provider_bundle: CpuProviderBundle,
    ) -> Result<Self, CpuBackendError> {
        let domains: Vec<_> = domains.into_iter().collect();
        if domains.is_empty() {
            return Err(ExternalCpuDomainRegistryError::EmptyRegistry.into());
        }

        let mut domain_ids = BTreeSet::new();
        let mut node_ids = BTreeSet::new();
        let mut has_all_allowed = false;
        for domain in &domains {
            if !domain_ids.insert(domain.id()) {
                return Err(
                    ExternalCpuDomainRegistryError::DuplicateDomainId { id: domain.id() }.into(),
                );
            }
            match domain.placement() {
                ResolvedCpuPlacement::NumaNode { id, .. } => {
                    if !node_ids.insert(*id) {
                        return Err(ExternalCpuDomainRegistryError::DuplicatePlacementIdentity {
                            placement: CpuPlacement::NumaNode(*id),
                        }
                        .into());
                    }
                }
                ResolvedCpuPlacement::AllAllowed { cpus } => {
                    if has_all_allowed {
                        return Err(ExternalCpuDomainRegistryError::DuplicatePlacementIdentity {
                            placement: CpuPlacement::AllAllowed,
                        }
                        .into());
                    }
                    has_all_allowed = true;
                    if domain.placement_guarantee() == CpuPlacementGuarantee::ExactDeclared
                        && cpus != topology.allowed_cpus()
                    {
                        return Err(ExternalCpuDomainRegistryError::ExactAllAllowedMismatch {
                            domain: domain.id(),
                            declared: cpus.clone(),
                            allowed: topology.allowed_cpus().clone(),
                        }
                        .into());
                    }
                }
            }
            if let Some(cpu) = domain
                .cpus()
                .as_slice()
                .iter()
                .copied()
                .find(|cpu| !topology.allowed_cpus().contains(*cpu))
            {
                return Err(ExternalCpuDomainRegistryError::CpuOutsideAllowedSet {
                    domain: domain.id(),
                    cpu,
                }
                .into());
            }
        }
        if !domain_ids.contains(&default_domain) {
            return Err(
                ExternalCpuDomainRegistryError::MissingDefaultDomain { default_domain }.into(),
            );
        }

        let buffer_limit = crate::buffer_pool::DEFAULT_MAX_RETAINED_CAPACITY_BYTES;
        let mut by_id = BTreeMap::new();
        let mut by_node = BTreeMap::new();
        let mut all_allowed = None;
        for domain in domains {
            let id = domain.id();
            let placement = domain.placement().clone();
            let engine = Arc::new(CpuEngine::from_external(domain, buffer_limit));
            match placement {
                ResolvedCpuPlacement::NumaNode { id, .. } => {
                    by_node.insert(id, Arc::clone(&engine));
                }
                ResolvedCpuPlacement::AllAllowed { .. } => {
                    all_allowed = Some(Arc::clone(&engine));
                }
            }
            by_id.insert(id, engine);
        }
        let Some(engine) = by_id.get(&default_domain).cloned() else {
            return Err(
                ExternalCpuDomainRegistryError::MissingDefaultDomain { default_domain }.into(),
            );
        };
        let resolved = ResolvedCpuExecution::ExternalManaged(engine.placement().clone());
        let kind = CpuBackendKind::default_compiled();
        let backend = Self {
            runtime_identity: CpuRuntimeIdentity::fresh(),
            shared: Arc::new(CpuBackendState {
                topology,
                engines: CpuEngineRegistry::ExternalPrebuilt(ExternalEngineRegistry {
                    by_id,
                    by_node,
                    all_allowed,
                    default_domain,
                }),
                arbiter,
                kind,
                buffer_limit: AtomicUsize::new(buffer_limit),
                indexed_plan_cache_limits: Mutex::new(DEFAULT_INDEXED_PLAN_CACHE_LIMITS),
            }),
            requested: CpuPlacement::Auto,
            resolved,
            engine,
            provider_bundle,
            allocation_domain: None,
        };
        backend
            .validate_provider_bundle_for_domains(&backend.provider_bundle)
            .map_err(|source| {
                CpuBackendError::Tensor(crate::Error::backend_source(
                    "CpuBackend ExternalManaged provider validation",
                    source,
                ))
            })?;
        Ok(backend)
    }

    /// Create a CPU backend using the selected compiled provider.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuBackend, CpuBackendKind};
    ///
    /// let backend = CpuBackend::with_kind(CpuBackendKind::default_compiled()).unwrap();
    /// assert_eq!(backend.kind(), CpuBackendKind::default_compiled());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CpuBackendError::Tensor`] when the provider is unavailable or
    /// its configuration is invalid, and [`CpuBackendError::Placement`] when
    /// CPU topology discovery or placement initialization fails.
    pub fn with_kind(kind: CpuBackendKind) -> Result<Self, CpuBackendError> {
        let op = "CpuBackend::with_kind";
        ensure_cpu_backend_kind_available(kind, op)
            .map_err(|error| constructor_tensor_error(op, error))?;
        let context = CpuContext::from_env();
        Self::from_thread_budget_and_kind(
            context.num_threads(),
            kind,
            crate::buffer_pool::DEFAULT_MAX_RETAINED_CAPACITY_BYTES,
        )
        .map_err(|error| CpuBackendError::placement(op, error))
    }

    /// Try to create a CPU backend using `RAYON_NUM_THREADS`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::try_new()
    ///     .unwrap_or_else(|_| CpuBackend::with_threads(1).unwrap());
    /// let _ = backend.num_threads();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CpuBackendError::Tensor`] when `RAYON_NUM_THREADS` is zero,
    /// malformed, or the compiled provider cannot be selected, and
    /// [`CpuBackendError::Placement`] when CPU topology or managed placement
    /// initialization is unavailable.
    pub fn try_new() -> Result<Self, CpuBackendError> {
        let op = "CpuBackend::try_new";
        let context =
            CpuContext::try_from_env().map_err(|error| constructor_tensor_error(op, error))?;
        Self::from_thread_budget_and_kind(
            context.num_threads(),
            CpuBackendKind::default_compiled(),
            crate::buffer_pool::DEFAULT_MAX_RETAINED_CAPACITY_BYTES,
        )
        .map_err(|error| CpuBackendError::placement(op, error))
    }

    /// Create a CPU backend from an existing context.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenferro_cpu::{CpuBackend, CpuContext};
    ///
    /// let ctx = Arc::new(CpuContext::with_threads(2).unwrap());
    /// let backend = CpuBackend::from_context(ctx);
    /// assert_eq!(backend.num_threads(), 2);
    /// ```
    pub fn from_context(ctx: Arc<CpuContext>) -> Self {
        Self::compatibility(
            ctx,
            crate::buffer_pool::DEFAULT_MAX_RETAINED_CAPACITY_BYTES,
            CpuBackendKind::default_compiled(),
        )
    }

    /// Create a CPU backend from an existing context and buffer-pool retention cap.
    ///
    /// The cap is measured in retained vector capacity bytes. A cap of zero
    /// disables buffer retention.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenferro_cpu::{CpuBackend, CpuContext};
    ///
    /// let ctx = Arc::new(CpuContext::with_threads(1).unwrap());
    /// let backend = CpuBackend::from_context_with_buffer_pool_limit(ctx, 0);
    /// assert_eq!(backend.buffer_pool_limit_bytes(), 0);
    /// ```
    pub fn from_context_with_buffer_pool_limit(
        ctx: Arc<CpuContext>,
        max_retained_capacity_bytes: usize,
    ) -> Self {
        Self::from_context_with_buffer_pool_limit_and_kind(
            ctx,
            max_retained_capacity_bytes,
            CpuBackendKind::default_compiled(),
        )
    }

    fn from_context_with_buffer_pool_limit_and_kind(
        ctx: Arc<CpuContext>,
        max_retained_capacity_bytes: usize,
        kind: CpuBackendKind,
    ) -> Self {
        Self::compatibility(ctx, max_retained_capacity_bytes, kind)
    }

    /// Create a CPU backend with a custom thread count.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::with_threads(2).unwrap();
    /// assert_eq!(backend.num_threads(), 2);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CpuBackendError::Tensor`] with `ValidationError::InvalidArgument`
    /// when `num_threads` is zero or the context cannot be configured, and
    /// [`CpuBackendError::Placement`] when CPU topology or placement fails.
    pub fn with_threads(num_threads: usize) -> Result<Self, CpuBackendError> {
        let op = "CpuBackend::with_threads";
        let context = CpuContext::with_threads(num_threads)
            .map_err(|error| constructor_tensor_error(op, error))?;
        Self::from_thread_budget_and_kind(
            context.num_threads(),
            CpuBackendKind::default_compiled(),
            crate::buffer_pool::DEFAULT_MAX_RETAINED_CAPACITY_BYTES,
        )
        .map_err(|error| CpuBackendError::placement(op, error))
    }

    /// Create a CPU backend with a custom thread count and provider.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuBackend, CpuBackendKind};
    ///
    /// let backend = CpuBackend::with_threads_and_kind(
    ///     1,
    ///     CpuBackendKind::default_compiled(),
    /// )?;
    /// assert_eq!(backend.num_threads(), 1);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CpuBackendError::Tensor`] with `ValidationError::InvalidArgument`
    /// when `num_threads` is zero or the provider is unavailable, and
    /// [`CpuBackendError::Placement`] when CPU topology or placement fails.
    pub fn with_threads_and_kind(
        num_threads: usize,
        kind: CpuBackendKind,
    ) -> Result<Self, CpuBackendError> {
        let op = "CpuBackend::with_threads_and_kind";
        ensure_cpu_backend_kind_available(kind, op)
            .map_err(|error| constructor_tensor_error(op, error))?;
        let context = CpuContext::with_threads(num_threads)
            .map_err(|error| constructor_tensor_error(op, error))?;
        Self::from_thread_budget_and_kind(
            context.num_threads(),
            kind,
            crate::buffer_pool::DEFAULT_MAX_RETAINED_CAPACITY_BYTES,
        )
        .map_err(|error| CpuBackendError::placement(op, error))
    }

    /// Clone this backend coordinator with a specific CPU placement request.
    ///
    /// Managed explicit placement is supported for faer/native execution.
    /// Externally managed coordinators resolve explicit requests only to
    /// matching registered domains and never construct a fallback engine.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuBackend, CpuPlacement};
    ///
    /// let backend = CpuBackend::new();
    /// if backend.supports_placement(CpuPlacement::AllAllowed) {
    ///     let placed = backend.for_placement(CpuPlacement::AllAllowed)?;
    ///     assert_eq!(placed.placement(), CpuPlacement::AllAllowed);
    /// }
    /// # Ok::<(), tenferro_cpu::CpuPlacementError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CpuPlacementError`] when the requested placement is not
    /// available for this backend or its affinity cannot be configured.
    pub fn for_placement(&self, requested: CpuPlacement) -> Result<Self, CpuPlacementError> {
        self.for_placement_with_affinity(
            requested,
            cfg!(any(target_os = "linux", target_os = "android")),
        )
    }

    fn for_placement_with_affinity(
        &self,
        requested: CpuPlacement,
        managed_affinity_available: bool,
    ) -> Result<Self, CpuPlacementError> {
        if self.shared.is_external() {
            let engine = self.shared.external_engine_for(requested)?;
            return Ok(Self {
                runtime_identity: CpuRuntimeIdentity::fresh(),
                shared: Arc::clone(&self.shared),
                requested,
                resolved: ResolvedCpuExecution::ExternalManaged(engine.placement().clone()),
                engine,
                provider_bundle: self.provider_bundle.clone(),
                allocation_domain: self.allocation_domain.clone(),
            });
        }
        let resolved = resolve_placement_with_affinity(
            self.kind(),
            requested,
            &self.shared.topology,
            managed_affinity_available,
        )?;
        if requested == CpuPlacement::Auto && !managed_affinity_available {
            return Ok(Self {
                runtime_identity: CpuRuntimeIdentity::fresh(),
                shared: Arc::clone(&self.shared),
                requested,
                resolved,
                engine: self.shared.managed_base_engine(requested)?,
                provider_bundle: self.provider_bundle.clone(),
                allocation_domain: self.allocation_domain.clone(),
            });
        }
        let engine_placement = match &resolved {
            ResolvedCpuExecution::Managed(placement) => placement.clone(),
            ResolvedCpuExecution::ExternalManaged(_) => {
                return Err(CpuPlacementError::InternalState {
                    requested,
                    backend: self.kind(),
                    message: "managed resolver returned an external execution mode",
                });
            }
            ResolvedCpuExecution::ProviderDefaultExclusive => ResolvedCpuPlacement::AllAllowed {
                cpus: self.shared.topology.allowed_cpus().clone(),
            },
            ResolvedCpuExecution::Compatibility => {
                return Err(CpuPlacementError::InternalState {
                    requested,
                    backend: self.kind(),
                    message: "placement resolution returned an internal compatibility mode",
                });
            }
        };
        let engine = self
            .shared
            .managed_engine_for(&engine_placement, requested)?;
        Ok(Self {
            runtime_identity: CpuRuntimeIdentity::fresh(),
            shared: Arc::clone(&self.shared),
            requested,
            resolved,
            engine,
            provider_bundle: self.provider_bundle.clone(),
            allocation_domain: self.allocation_domain.clone(),
        })
    }

    /// Return the placement requested by this handle.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuBackend, CpuPlacement};
    ///
    /// assert_eq!(CpuBackend::new().placement(), CpuPlacement::Auto);
    /// ```
    pub fn placement(&self) -> CpuPlacement {
        self.requested
    }

    /// Return the concrete managed placement or external placement declaration.
    ///
    /// Provider-default-exclusive and compatibility contexts return `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuBackend, CpuPlacement};
    ///
    /// let backend = CpuBackend::new();
    /// if backend.supports_placement(CpuPlacement::AllAllowed) {
    ///     assert!(backend
    ///         .for_placement(CpuPlacement::AllAllowed)?
    ///         .resolved_placement()
    ///         .is_some());
    /// }
    /// # Ok::<(), tenferro_cpu::CpuPlacementError>(())
    /// ```
    pub fn resolved_placement(&self) -> Option<&ResolvedCpuPlacement> {
        match &self.resolved {
            ResolvedCpuExecution::Managed(placement)
            | ResolvedCpuExecution::ExternalManaged(placement) => Some(placement),
            ResolvedCpuExecution::Compatibility
            | ResolvedCpuExecution::ProviderDefaultExclusive => None,
        }
    }

    /// Return the process-visible topology shared by all coordinator clones.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// assert!(!CpuBackend::new().topology().allowed_cpus().is_empty());
    /// ```
    pub fn topology(&self) -> &CpuTopology {
        &self.shared.topology
    }

    /// Report whether this coordinator can resolve a placement request.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuBackend, CpuPlacement};
    ///
    /// assert!(CpuBackend::new().supports_placement(CpuPlacement::Auto));
    /// ```
    pub fn supports_placement(&self, placement: CpuPlacement) -> bool {
        if self.shared.is_external() {
            self.shared.external_engine_for(placement).is_ok()
        } else {
            resolve_placement(self.kind(), placement, &self.shared.topology).is_ok()
        }
    }

    /// Return a snapshot suitable for diagnostics and placement reporting.
    ///
    /// # Examples
    ///
    /// ```
    /// let backend = tenferro_cpu::CpuBackend::new();
    /// assert_eq!(backend.execution_info().backend_kind(), backend.kind());
    /// ```
    pub fn execution_info(&self) -> CpuExecutionInfo {
        let domain = self.engine.domain();
        let capabilities = domain.executor_capabilities();
        let (executor_affinity, executor_shutdown) =
            if domain.ownership() == CpuDomainOwnership::ExternalManaged {
                (
                    CpuExecutorAffinity::CallerDeclaredUnverified,
                    CpuExecutorShutdown::CallerOwned,
                )
            } else {
                (capabilities.affinity, capabilities.shutdown)
            };
        CpuExecutionInfo {
            backend_kind: self.kind(),
            execution_mode: match &self.resolved {
                ResolvedCpuExecution::Managed(_) => CpuExecutionMode::Managed,
                ResolvedCpuExecution::ExternalManaged(_) => CpuExecutionMode::ExternalManaged,
                ResolvedCpuExecution::ProviderDefaultExclusive => {
                    CpuExecutionMode::ProviderDefaultExclusive
                }
                ResolvedCpuExecution::Compatibility => CpuExecutionMode::Compatibility,
            },
            requested_placement: self.requested,
            resolved_placement: self.resolved_placement().cloned(),
            topology: self.shared.topology.clone(),
            domain_id: domain.id(),
            domain_cpus: domain.cpus().clone(),
            worker_count: capabilities.worker_count.get(),
            thread_budget: domain.thread_budget().get(),
            placement_guarantee: domain.placement_guarantee(),
            domain_ownership: domain.ownership(),
            executor_affinity,
            executor_shutdown,
            provider_diagnostic: provider_diagnostic(self.kind(), domain.ownership()),
        }
    }

    #[cfg(all(
        test,
        feature = "cpu-faer",
        any(target_os = "linux", target_os = "android")
    ))]
    fn coordinator_id_for_test(&self) -> usize {
        Arc::as_ptr(&self.shared) as usize
    }

    #[cfg(test)]
    pub(crate) fn context_id_for_test(&self) -> usize {
        Arc::as_ptr(self.engine.domain().executor()) as *const () as usize
    }

    /// Return the runtime CPU provider selected by this backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuBackend, CpuBackendKind};
    ///
    /// let backend = CpuBackend::new();
    /// assert_eq!(backend.kind(), CpuBackendKind::default_compiled());
    /// ```
    pub fn kind(&self) -> CpuBackendKind {
        self.shared.kind
    }

    /// Return the immutable CPU provider slots selected for this handle.
    pub fn provider_bundle(&self) -> &CpuProviderBundle {
        &self.provider_bundle
    }

    /// Return the opaque identity of this backend's executable witness.
    ///
    /// The identity has no access to backend execution or storage resources.
    /// Clones of this backend retain the identity, while separately constructed
    /// backends and backends returned after changing immutable witness resources
    /// receive a distinct identity.
    pub fn runtime_identity(&self) -> CpuRuntimeIdentity {
        self.runtime_identity.clone()
    }

    /// Return this backend with an immutable construction-time provider bundle.
    ///
    /// Existing clones retain their original bundle identity.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuBackend, CpuBackendKind, CpuProviderBundle};
    /// let bundle = CpuProviderBundle::builder(CpuBackendKind::Faer).build()?;
    /// let backend = CpuBackend::new().with_provider_bundle(bundle.clone())?;
    /// assert!(backend.provider_bundle().shares_identity_with(&bundle));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CpuProviderBundleInstallError::IncompatibleDomain`] if a
    /// provider cannot satisfy one of this backend's resource-domain
    /// contracts.
    pub fn with_provider_bundle(
        mut self,
        bundle: CpuProviderBundle,
    ) -> Result<Self, CpuProviderBundleInstallError> {
        self.validate_provider_bundle_for_domains(&bundle)?;
        self.provider_bundle = bundle;
        self.runtime_identity = CpuRuntimeIdentity::fresh();
        Ok(self)
    }

    fn validate_provider_bundle_for_domains(
        &self,
        bundle: &CpuProviderBundle,
    ) -> Result<(), CpuProviderBundleInstallError> {
        let allowed = self.shared.topology.allowed_cpus();
        let validate_engine = |engine: &CpuEngine| {
            let domain = engine.domain();
            bundle.validate_for_domain(
                domain.id(),
                domain.thread_budget(),
                domain.placement_guarantee(),
                domain.cpus(),
                allowed,
            )
        };

        match &self.shared.engines {
            CpuEngineRegistry::ExternalPrebuilt(registry) => {
                for engine in registry.by_id.values() {
                    validate_engine(engine)?;
                }
            }
            CpuEngineRegistry::ManagedLazy(registry) => {
                validate_engine(&registry.base_engine)?;

                // A placed clone retains the installed bundle. Validate every
                // lazily constructible managed NUMA domain now rather than
                // allowing a later `for_placement` call to bypass the bundle
                // contract.
                #[cfg(any(target_os = "linux", target_os = "android"))]
                for node in self.shared.topology.nodes() {
                    let Some(domain_id) = registry.node_domain_ids.get(&node.id()).copied() else {
                        continue;
                    };
                    let budget =
                        std::num::NonZeroUsize::new(registry.thread_budget.min(node.cpus().len()))
                            .expect("usable topology nodes have non-empty CPU sets");
                    bundle.validate_for_domain(
                        domain_id,
                        budget,
                        CpuPlacementGuarantee::ExactDeclared,
                        node.cpus(),
                        allowed,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Return the selected CPU domain's thread budget.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::with_threads(2).unwrap();
    /// assert_eq!(backend.num_threads(), 2);
    /// ```
    pub fn num_threads(&self) -> usize {
        self.engine.domain().thread_budget().get()
    }

    /// Number of retained typed host buffers currently held by this backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// assert_eq!(backend.buffer_pool_len()?, 0);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when the engine registry or an
    /// initialized engine's resources lock is poisoned.
    pub fn buffer_pool_len(&self) -> crate::Result<usize> {
        self.shared
            .initialized_engines("CpuBackend::buffer_pool_len")?
            .iter()
            .try_fold(0, |total, engine| {
                Ok(total
                    + lock_engine_resources(engine, "CpuBackend::buffer_pool_len")?
                        .buffers
                        .len())
            })
    }

    /// Snapshot reusable typed host buffers currently retained by this backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// let stats = backend.buffer_pool_stats()?;
    /// assert_eq!(stats.buffers, 0);
    /// assert_eq!(stats.capacity_bytes, 0);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when the engine registry or an
    /// initialized engine's resources lock is poisoned.
    pub fn buffer_pool_stats(&self) -> crate::Result<BufferPoolStats> {
        self.shared
            .initialized_engines("CpuBackend::buffer_pool_stats")?
            .iter()
            .try_fold(BufferPoolStats::default(), |mut total, engine| {
                let stats = lock_engine_resources(engine, "CpuBackend::buffer_pool_stats")?
                    .buffers
                    .stats();
                total.buffers += stats.buffers;
                total.capacity_bytes += stats.capacity_bytes;
                Ok(total)
            })
    }

    /// Return cache-style stats for the CPU buffer pool.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// let stats = backend.buffer_pool_cache_stats()?;
    /// assert_eq!(stats.entries, 0);
    /// assert_eq!(stats.retained_bytes, 0);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when the engine registry or an
    /// initialized engine's resources lock is poisoned.
    pub fn buffer_pool_cache_stats(&self) -> crate::Result<CacheStats> {
        let stats = self.buffer_pool_stats()?;
        Ok(CacheStats {
            entries: stats.buffers,
            retained_bytes: stats.capacity_bytes,
            hits: 0,
            misses: 0,
            evictions: 0,
            clears: 0,
        })
    }

    /// Return the limits applied to each CPU engine's indexed-plan cache.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// assert!(backend.indexed_plan_cache_limits()?.max_entries() > 0);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when the shared cache
    /// configuration lock is poisoned.
    pub fn indexed_plan_cache_limits(&self) -> crate::Result<IndexedPlanCacheLimits> {
        self.shared
            .indexed_plan_cache_limits
            .lock()
            .map(|limits| *limits)
            .map_err(|_| {
                poisoned_cpu_lock(
                    "CpuBackend::indexed_plan_cache_limits",
                    "CPU indexed-plan cache configuration",
                )
            })
    }

    /// Update indexed-plan cache limits for current and future CPU engines.
    ///
    /// Shrinking either bound evicts least-recently-used plans immediately. A
    /// zero entry or byte bound disables retention.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuBackend, IndexedPlanCacheLimits};
    ///
    /// let mut backend = CpuBackend::new();
    /// backend.set_indexed_plan_cache_limits(IndexedPlanCacheLimits::new(8, 4096))?;
    /// assert_eq!(backend.indexed_plan_cache_limits()?.max_entries(), 8);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] without changing the configured
    /// limits when an engine registry or resource lock is poisoned.
    pub fn set_indexed_plan_cache_limits(
        &mut self,
        limits: IndexedPlanCacheLimits,
    ) -> crate::Result<()> {
        // INVARIANT: keep the configuration guard while snapshotting the
        // registry and updating every initialized engine. Lazy creation takes
        // the same guard before any registry or resource lock.
        let mut configured_limits = self.shared.indexed_plan_cache_limits.lock().map_err(|_| {
            poisoned_cpu_lock(
                "CpuBackend::set_indexed_plan_cache_limits",
                "CPU indexed-plan cache configuration",
            )
        })?;
        let engines = self
            .shared
            .initialized_engines("CpuBackend::set_indexed_plan_cache_limits")?;
        let mut resources = engines
            .iter()
            .map(|engine| {
                lock_engine_resources(engine, "CpuBackend::set_indexed_plan_cache_limits")
            })
            .collect::<crate::Result<Vec<_>>>()?;
        *configured_limits = limits;
        for resource in &mut resources {
            resource.indexed_plan_cache.set_limits(limits);
        }
        Ok(())
    }

    /// Snapshot aggregate indexed-plan cache statistics across initialized CPU engines.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// assert_eq!(backend.indexed_plan_cache_stats()?.entries, 0);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] when an engine registry or
    /// resource lock is poisoned.
    pub fn indexed_plan_cache_stats(&self) -> crate::Result<CacheStats> {
        self.shared
            .initialized_engines("CpuBackend::indexed_plan_cache_stats")?
            .iter()
            .try_fold(CacheStats::default(), |mut total, engine| {
                let stats = lock_engine_resources(engine, "CpuBackend::indexed_plan_cache_stats")?
                    .indexed_plan_cache
                    .stats();
                saturating_add_tensor_cache_stats(&mut total, stats);
                Ok(total)
            })
    }

    /// Clear indexed traversal plans retained by all initialized CPU engines.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let mut backend = CpuBackend::new();
    /// backend.clear_indexed_plan_cache()?;
    /// assert_eq!(backend.indexed_plan_cache_stats()?.entries, 0);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] without clearing any engine when
    /// an engine registry or resource lock is poisoned.
    pub fn clear_indexed_plan_cache(&mut self) -> crate::Result<()> {
        let engines = self
            .shared
            .initialized_engines("CpuBackend::clear_indexed_plan_cache")?;
        let mut resources = engines
            .iter()
            .map(|engine| lock_engine_resources(engine, "CpuBackend::clear_indexed_plan_cache"))
            .collect::<crate::Result<Vec<_>>>()?;
        for resource in &mut resources {
            resource.indexed_plan_cache.clear();
        }
        Ok(())
    }

    /// Current CPU buffer-pool retention limit in bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenferro_cpu::{CpuBackend, CpuContext};
    ///
    /// let backend = CpuBackend::from_context_with_buffer_pool_limit(
    ///     Arc::new(CpuContext::with_threads(1).unwrap()),
    ///     4096,
    /// );
    /// assert_eq!(backend.buffer_pool_limit_bytes(), 4096);
    /// ```
    pub fn buffer_pool_limit_bytes(&self) -> usize {
        self.shared.buffer_limit.load(Ordering::Relaxed)
    }

    /// Update the CPU buffer-pool retention limit in bytes.
    ///
    /// Shrinking the limit evicts retained buffers immediately. A limit of zero
    /// disables buffer retention.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let mut backend = CpuBackend::new();
    /// backend.set_buffer_pool_limit_bytes(0)?;
    /// assert_eq!(backend.buffer_pool_limit_bytes(), 0);
    /// assert_eq!(backend.buffer_pool_len()?, 0);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] without changing the configured
    /// limit when the engine registry or any initialized engine's resources
    /// lock is poisoned.
    pub fn set_buffer_pool_limit_bytes(
        &mut self,
        max_retained_capacity_bytes: usize,
    ) -> crate::Result<()> {
        let engines = self
            .shared
            .initialized_engines("CpuBackend::set_buffer_pool_limit_bytes")?;
        let mut resources = engines
            .iter()
            .map(|engine| lock_engine_resources(engine, "CpuBackend::set_buffer_pool_limit_bytes"))
            .collect::<crate::Result<Vec<_>>>()?;
        self.shared
            .buffer_limit
            .store(max_retained_capacity_bytes, Ordering::Relaxed);
        for resource in &mut resources {
            resource
                .buffers
                .set_max_retained_capacity_bytes(max_retained_capacity_bytes);
        }
        Ok(())
    }

    /// Reset reusable typed host buffers currently retained by this backend.
    ///
    /// This releases pool-owned vectors to the process allocator. Operating
    /// system RSS may not fall immediately because allocators can retain freed
    /// pages for future allocations.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let mut backend = CpuBackend::new();
    /// backend.reset_buffer_pool()?;
    /// assert_eq!(backend.buffer_pool_len()?, 0);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] without clearing any initialized
    /// engine when the engine registry or any engine's resources lock is
    /// poisoned.
    pub fn reset_buffer_pool(&mut self) -> crate::Result<()> {
        let engines = self
            .shared
            .initialized_engines("CpuBackend::reset_buffer_pool")?;
        let mut resources = engines
            .iter()
            .map(|engine| lock_engine_resources(engine, "CpuBackend::reset_buffer_pool"))
            .collect::<crate::Result<Vec<_>>>()?;
        for resource in &mut resources {
            resource.buffers.clear();
        }
        Ok(())
    }

    pub(crate) fn runtime_cache_stats(
        &self,
    ) -> crate::Result<tenferro_runtime::runtime::CacheStats> {
        let resources = lock_engine_resources(&self.engine, "CpuBackend::runtime_cache_stats")?;
        let buffers = resources.buffers.cache_stats();
        let gemm = tenferro_tensor::RuntimeCacheControl::stats(&resources.gemm_analysis_cache);
        let indexed = resources.indexed_plan_cache.stats();
        Ok(tenferro_runtime::runtime::CacheStats {
            entries: buffers
                .entries
                .saturating_add(gemm.entries)
                .saturating_add(indexed.entries),
            retained_bytes: buffers
                .retained_bytes
                .saturating_add(gemm.retained_bytes)
                .saturating_add(indexed.retained_bytes),
            hits: indexed.hits,
            misses: indexed.misses,
            evictions: indexed.evictions,
            clears: indexed.clears,
        })
    }

    pub(crate) fn clear_runtime_caches(&self) -> crate::Result<()> {
        let mut resources =
            lock_engine_resources(&self.engine, "CpuBackend::clear_runtime_caches")?;
        resources.buffers.clear();
        tenferro_tensor::RuntimeCacheControl::clear(&mut resources.gemm_analysis_cache);
        resources.indexed_plan_cache.clear();
        Ok(())
    }

    /// Run a closure in this backend's CPU execution scope.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::with_threads(1).unwrap();
    /// let value = backend.install(|| 1 + 1);
    /// assert_eq!(value, 2);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when re-entered while another CPU backend execution is active on
    /// the current thread or managed Rayon scope. This includes direct nesting
    /// and backend calls from parallel child tasks; either could violate CPU or
    /// provider exclusivity. For an externally managed domain, it also panics
    /// with the executor's typed diagnostic when synchronous executor entry
    /// fails because this convenience method cannot return a `Result`.
    pub fn install<R: Send>(&self, op: impl FnOnce() -> R + Send) -> R {
        let owner = inherited_or_new_execution_owner();
        let permit = self.acquire_execution_permit(owner);
        let entry = CpuOperationEntry::new(self.engine.domain(), &permit);
        match entry.enter(ParallelMode::Sequential, |_| op()) {
            Ok(result) => result,
            Err(error) => panic!("CpuBackend::install executor failed: {error}"),
        }
    }

    fn try_install<R: Send>(
        &self,
        op: impl FnOnce() -> crate::Result<R> + Send,
    ) -> crate::Result<R> {
        let owner = inherited_or_new_execution_owner();
        let permit = self.acquire_execution_permit(owner);
        let entry = CpuOperationEntry::new(self.engine.domain(), &permit);
        let mode = entry.preferred_engine_mode();
        entry
            .enter(mode, |context| context.with_native_parallelism(op))
            .map_err(|error| crate::Error::backend_source("CPU tensor execution", error))?
    }

    fn try_install_with_context<R: Send>(
        &self,
        op: impl FnOnce(&CpuExecutionContext<'_>) -> crate::Result<R> + Send,
    ) -> crate::Result<R> {
        let owner = inherited_or_new_execution_owner();
        let permit = self.acquire_execution_permit(owner);
        let entry = CpuOperationEntry::new(self.engine.domain(), &permit);
        let mode = entry.preferred_engine_mode();
        entry
            .enter(mode, |context| {
                context.with_native_parallelism(|| op(context))
            })
            .map_err(|error| crate::Error::backend_source("CPU tensor execution", error))?
    }

    fn try_install_fresh<R: FreshCpuOutput + Send>(
        &self,
        op: impl FnOnce() -> crate::Result<R> + Send,
    ) -> crate::Result<R> {
        let domain = self.engine.domain().id();
        let mut output = self.try_install(op)?;
        output.tag_fresh(domain);
        Ok(output)
    }

    fn try_install_fresh_with_context<R: FreshCpuOutput + Send>(
        &self,
        op: impl FnOnce(&CpuExecutionContext<'_>) -> crate::Result<R> + Send,
    ) -> crate::Result<R> {
        let domain = self.engine.domain().id();
        let mut output = self.try_install_with_context(op)?;
        output.tag_fresh(domain);
        Ok(output)
    }

    fn install_with_pool_unmarked<R: Send>(
        &mut self,
        op: impl FnOnce(&mut BufferPool) -> crate::Result<R> + Send,
    ) -> crate::Result<R> {
        let owner = inherited_or_new_execution_owner();
        let permit = self.acquire_execution_permit(owner);
        let entry = CpuOperationEntry::new(self.engine.domain(), &permit);
        let mode = entry.preferred_engine_mode();
        entry
            .enter(mode, |context| {
                context.with_native_parallelism(|| {
                    self.with_execution_resources(&permit, |resources| {
                        let mut buffers = BufferPoolLoan::new(&mut resources.buffers);
                        op(buffers.get_mut())
                    })
                })
            })
            .map_err(|error| crate::Error::backend_source("CPU tensor execution", error))?
    }

    fn install_with_pool_context_unmarked<R: Send>(
        &mut self,
        op: impl FnOnce(&CpuExecutionContext<'_>, &mut BufferPool) -> crate::Result<R> + Send,
    ) -> crate::Result<R> {
        let owner = inherited_or_new_execution_owner();
        let permit = self.acquire_execution_permit(owner);
        let entry = CpuOperationEntry::new(self.engine.domain(), &permit);
        let mode = entry.preferred_engine_mode();
        entry
            .enter(mode, |context| {
                context.with_native_parallelism(|| {
                    self.with_execution_resources(&permit, |resources| {
                        let mut buffers = BufferPoolLoan::new(&mut resources.buffers);
                        op(context, buffers.get_mut())
                    })
                })
            })
            .map_err(|error| crate::Error::backend_source("CPU tensor execution", error))?
    }

    fn install_with_indexed_pool_context_unmarked<R: Send>(
        &mut self,
        op: impl FnOnce(
                &CpuExecutionContext<'_>,
                &mut BufferPool,
                &mut IndexedPlanCache,
            ) -> crate::Result<R>
            + Send,
    ) -> crate::Result<R> {
        let owner = inherited_or_new_execution_owner();
        let permit = self.acquire_execution_permit(owner);
        let entry = CpuOperationEntry::new(self.engine.domain(), &permit);
        let mode = entry.preferred_engine_mode();
        entry
            .enter(mode, |context| {
                context.with_native_parallelism(|| {
                    self.with_execution_resources(&permit, |resources| {
                        let EngineResources {
                            buffers,
                            indexed_plan_cache,
                            ..
                        } = resources;
                        let mut buffers = BufferPoolLoan::new(buffers);
                        op(context, buffers.get_mut(), indexed_plan_cache)
                    })
                })
            })
            .map_err(|error| crate::Error::backend_source("CPU tensor execution", error))?
    }

    fn install_with_pool<R: FreshCpuOutput + Send>(
        &mut self,
        op: impl FnOnce(&mut BufferPool) -> crate::Result<R> + Send,
    ) -> crate::Result<R> {
        let domain = self.engine.domain().id();
        let mut output = self.install_with_pool_unmarked(op)?;
        output.tag_fresh(domain);
        Ok(output)
    }

    fn install_with_pool_context<R: FreshCpuOutput + Send>(
        &mut self,
        op: impl FnOnce(&CpuExecutionContext<'_>, &mut BufferPool) -> crate::Result<R> + Send,
    ) -> crate::Result<R> {
        let domain = self.engine.domain().id();
        let mut output = self.install_with_pool_context_unmarked(op)?;
        output.tag_fresh(domain);
        Ok(output)
    }

    fn install_with_indexed_pool_context<R: FreshCpuOutput + Send>(
        &mut self,
        op: impl FnOnce(
                &CpuExecutionContext<'_>,
                &mut BufferPool,
                &mut IndexedPlanCache,
            ) -> crate::Result<R>
            + Send,
    ) -> crate::Result<R> {
        let domain = self.engine.domain().id();
        let mut output = self.install_with_indexed_pool_context_unmarked(op)?;
        output.tag_fresh(domain);
        Ok(output)
    }

    /// Run an external linalg implementation with one borrowed execution
    /// context and this backend's buffer pool.
    ///
    /// This is exposed for operation-family crates that own their backend
    /// implementation while still sharing the CPU backend's allocation pool.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// let mut backend = CpuBackend::new();
    /// backend.with_linalg_pool(|context, _pool| {
    ///     assert!(context.thread_budget().get() >= 1);
    ///     Ok(())
    /// })?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::BackendSource`] with a
    /// [`crate::CpuDomainExecutorError`] source when authoritative executor
    /// admission fails. Errors returned by the operation-family closure are
    /// propagated unchanged.
    #[doc(hidden)]
    pub fn with_linalg_pool<R: Send>(
        &mut self,
        op: impl FnOnce(&CpuExecutionContext<'_>, &mut BufferPool) -> crate::Result<R> + Send,
    ) -> crate::Result<R> {
        let owner = inherited_or_new_execution_owner();
        let permit = self.acquire_execution_permit(owner);
        let entry = CpuOperationEntry::new(self.engine.domain(), &permit);
        let mode = entry.preferred_linalg_mode(self.kind());
        entry
            .enter(mode, |context| {
                context.with_native_parallelism(|| {
                    self.with_execution_resources(&permit, |resources| {
                        let mut buffers = BufferPoolLoan::new(&mut resources.buffers);
                        op(context, buffers.get_mut())
                    })
                })
            })
            .map_err(|error| crate::Error::backend_source("CPU linalg execution", error))?
    }

    fn with_execution_resources<R>(
        &self,
        permit: &ResourcePermit,
        op: impl FnOnce(&mut EngineResources) -> R,
    ) -> R {
        if permit.is_reentrant() {
            let mut resources =
                EngineResources::new(self.shared.buffer_limit.load(Ordering::Relaxed));
            return op(&mut resources);
        }
        let mut resources = self
            .engine
            .resources
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        op(&mut resources)
    }

    fn acquire_execution_permit(&self, owner: ResourceOwner) -> ResourcePermit {
        match &self.resolved {
            ResolvedCpuExecution::Managed(placement)
            | ResolvedCpuExecution::ExternalManaged(placement) => self
                .shared
                .arbiter
                .acquire_recovering(placement.cpus().clone(), owner),
            ResolvedCpuExecution::Compatibility => self
                .shared
                .arbiter
                .acquire_recovering(self.shared.topology.allowed_cpus().clone(), owner),
            ResolvedCpuExecution::ProviderDefaultExclusive => self
                .shared
                .arbiter
                .acquire_provider_exclusive_recovering(owner),
        }
    }

    #[cfg(test)]
    fn try_acquire_execution_permit_for_test(
        &self,
    ) -> Result<Option<ResourcePermit>, crate::arbiter::ResourceArbiterError> {
        match &self.resolved {
            ResolvedCpuExecution::Managed(placement)
            | ResolvedCpuExecution::ExternalManaged(placement) => {
                self.shared.arbiter.try_acquire(placement.cpus().clone())
            }
            ResolvedCpuExecution::Compatibility => self
                .shared
                .arbiter
                .try_acquire(self.shared.topology.allowed_cpus().clone()),
            ResolvedCpuExecution::ProviderDefaultExclusive => {
                self.shared.arbiter.try_acquire_provider_exclusive()
            }
        }
    }
}

impl BackendRuntimeCache for CpuBackend {
    type RuntimeCache = gemm::GemmAnalysisCache;
}

impl TensorElementwise for CpuBackend {
    fn elementwise_read_into(
        &mut self,
        op: ElementwiseReadOp,
        inputs: &[TensorRead<'_>],
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        self.install_with_pool_context_unmarked(|context, buffers| {
            let exec_context = context.strided_exec_context();
            tenferro_tensor::backend::elementwise_read_into_with_context(
                op,
                inputs,
                out,
                &exec_context,
                |inputs, out| elementwise_read_into_fallback_with_pool(buffers, op, inputs, out),
            )
        })
    }

    fn add(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::add_with_pool(buffers, lhs, rhs))
    }

    fn add_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::add_read_with_pool(buffers, lhs, rhs))
    }

    fn sub(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::sub_with_pool(buffers, lhs, rhs))
    }

    fn sub_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::sub_read_with_pool(buffers, lhs, rhs))
    }

    fn mul(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::mul_with_pool(buffers, lhs, rhs))
    }

    fn mul_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::mul_read_with_pool(buffers, lhs, rhs))
    }

    fn neg(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::neg_with_pool(buffers, input))
    }

    fn neg_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::neg_read_with_pool(buffers, input))
    }

    fn conj(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::conj_with_pool(buffers, input))
    }

    fn conj_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::conj_read_with_pool(buffers, input))
    }

    fn div(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::div_with_pool(buffers, lhs, rhs))
    }

    fn div_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::div_read_with_pool(buffers, lhs, rhs))
    }

    fn rem(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::rem_with_pool(buffers, lhs, rhs))
    }

    fn rem_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::rem_read_with_pool(buffers, lhs, rhs))
    }

    fn abs(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::abs_with_pool(buffers, input))
    }

    fn abs_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::abs_read_with_pool(buffers, input))
    }

    fn sign(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::sign_with_pool(buffers, input))
    }

    fn sign_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::sign_read_with_pool(buffers, input))
    }

    fn maximum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::maximum_with_pool(buffers, lhs, rhs))
    }

    fn maximum_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::maximum_read_with_pool(buffers, lhs, rhs))
    }

    fn minimum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::minimum_with_pool(buffers, lhs, rhs))
    }

    fn minimum_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::minimum_read_with_pool(buffers, lhs, rhs))
    }

    fn compare(&mut self, lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::compare_with_pool(buffers, lhs, rhs, dir))
    }

    fn compare_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        dir: &CompareDir,
    ) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| {
            elementwise::compare_read_with_pool(buffers, lhs, rhs, dir)
        })
    }

    fn select(
        &mut self,
        pred: &Tensor,
        on_true: &Tensor,
        on_false: &Tensor,
    ) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| {
            elementwise::select_with_pool(buffers, pred, on_true, on_false)
        })
    }

    fn select_read(
        &mut self,
        pred: TensorRead<'_>,
        on_true: TensorRead<'_>,
        on_false: TensorRead<'_>,
    ) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| {
            elementwise::select_read_with_pool(buffers, pred, on_true, on_false)
        })
    }

    fn clamp(&mut self, input: &Tensor, lower: &Tensor, upper: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::clamp_with_pool(buffers, input, lower, upper))
    }

    fn clamp_read(
        &mut self,
        input: TensorRead<'_>,
        lower: TensorRead<'_>,
        upper: TensorRead<'_>,
    ) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| {
            elementwise::clamp_read_with_pool(buffers, input, lower, upper)
        })
    }
}

impl TensorAnalytic for CpuBackend {
    fn exp(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::exp_with_pool(buffers, input))
    }

    fn exp_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::exp_read_with_pool(buffers, input))
    }

    fn log(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::log_with_pool(buffers, input))
    }

    fn log_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::log_read_with_pool(buffers, input))
    }

    fn sin(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::sin_with_pool(buffers, input))
    }

    fn sin_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::sin_read_with_pool(buffers, input))
    }

    fn cos(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::cos_with_pool(buffers, input))
    }

    fn cos_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::cos_read_with_pool(buffers, input))
    }

    fn tanh(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::tanh_with_pool(buffers, input))
    }

    fn tanh_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::tanh_read_with_pool(buffers, input))
    }

    fn sqrt(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::sqrt_with_pool(buffers, input))
    }

    fn sqrt_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::sqrt_read_with_pool(buffers, input))
    }

    fn rsqrt(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::rsqrt_with_pool(buffers, input))
    }

    fn rsqrt_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::rsqrt_read_with_pool(buffers, input))
    }

    fn pow(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::pow_with_pool(buffers, lhs, rhs))
    }

    fn pow_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::pow_read_with_pool(buffers, lhs, rhs))
    }

    fn expm1(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::expm1_with_pool(buffers, input))
    }

    fn expm1_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::expm1_read_with_pool(buffers, input))
    }

    fn log1p(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::log1p_with_pool(buffers, input))
    }

    fn log1p_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::log1p_read_with_pool(buffers, input))
    }
}

impl TensorStructural for CpuBackend {
    fn to_contiguous_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| {
            materialize_tensor_read(buffers, "CpuBackend::to_contiguous_read", input)
        })
    }

    fn copy_read_into(&mut self, src: TensorRead<'_>, dst: TensorWrite<'_>) -> crate::Result<()> {
        self.try_install(|| copy_tensor_read_into("CpuBackend::copy_read_into", src, dst))
    }

    fn transpose(&mut self, input: &Tensor, perm: &[usize]) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| structural::transpose_with_pool(buffers, input, perm))
    }

    fn transpose_read(&mut self, input: TensorRead<'_>, perm: &[usize]) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| structural::transpose_read_with_pool(buffers, input, perm))
    }

    fn reshape(&mut self, input: &Tensor, shape: &[usize]) -> crate::Result<Tensor> {
        self.try_install(|| structural::reshape(input, shape))
    }

    fn reshape_read(&mut self, input: TensorRead<'_>, shape: &[usize]) -> crate::Result<Tensor> {
        let materializes = matches!(&input, TensorRead::View(_));
        if materializes {
            self.install_with_pool(|buffers| {
                structural::reshape_read_with_pool(buffers, input, shape)
            })
        } else {
            self.install_with_pool_unmarked(|buffers| {
                structural::reshape_read_with_pool(buffers, input, shape)
            })
        }
    }

    fn broadcast_in_dim(
        &mut self,
        input: &Tensor,
        shape: &[usize],
        dims: &[usize],
    ) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| {
            structural::broadcast_in_dim_with_pool(buffers, input, shape, dims)
        })
    }

    fn broadcast_in_dim_read(
        &mut self,
        input: TensorRead<'_>,
        shape: &[usize],
        dims: &[usize],
    ) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| {
            structural::broadcast_in_dim_read_with_pool(buffers, input, shape, dims)
        })
    }

    fn cast(&mut self, input: &Tensor, to: crate::DType) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| structural::cast_with_pool(buffers, input, to))
    }

    fn extract_diagonal(
        &mut self,
        input: &Tensor,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| {
            structural::extract_diagonal_with_pool(buffers, input, axis_a, axis_b)
        })
    }

    fn embed_diagonal(
        &mut self,
        input: &Tensor,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| {
            structural::embed_diagonal_with_pool(buffers, input, axis_a, axis_b)
        })
    }

    fn tril(&mut self, input: &Tensor, k: i64) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| structural::tril_with_pool(buffers, input, k))
    }

    fn triu(&mut self, input: &Tensor, k: i64) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| structural::triu_with_pool(buffers, input, k))
    }
}

impl TensorReduction for CpuBackend {
    fn reduce_sum(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.try_install_fresh_with_context(|context| {
            let exec_context = context.strided_exec_context();
            reduction::reduce_sum(input, axes, &exec_context)
        })
    }

    fn reduce_sum_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.install_with_pool_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            reduction::reduce_sum_read(buffers, input, axes, &exec_context)
        })
    }

    fn reduce_sum_squares_read(
        &mut self,
        input: TensorRead<'_>,
        axes: &[usize],
    ) -> crate::Result<Tensor> {
        self.install_with_pool_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            reduction::reduce_sum_squares_read(buffers, input, axes, &exec_context)
        })
    }

    fn reduce_prod(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.try_install_fresh_with_context(|context| {
            let exec_context = context.strided_exec_context();
            reduction::reduce_prod(input, axes, &exec_context)
        })
    }

    fn reduce_prod_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.install_with_pool_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            reduction::reduce_prod_read(buffers, input, axes, &exec_context)
        })
    }

    fn reduce_max(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.try_install_fresh(|| reduction::reduce_max(input, axes))
    }

    fn reduce_max_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| reduction::reduce_max_read(buffers, input, axes))
    }

    fn reduce_min(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.try_install_fresh(|| reduction::reduce_min(input, axes))
    }

    fn reduce_min_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| reduction::reduce_min_read(buffers, input, axes))
    }
}

impl TensorDot for CpuBackend {
    fn dot_general(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        self.run_backend_session_cached(None, move |session| session.dot_general(lhs, rhs, config))
    }

    fn dot_general_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        self.run_backend_session_cached(None, move |session| {
            session.dot_general_read(lhs, rhs, config)
        })
    }

    fn dot_general_read_into(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        self.run_backend_session_cached(None, move |session| {
            session.dot_general_read_into(lhs, rhs, config, out)
        })
    }

    fn dot_general_read_into_accum(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        accumulation: DotGeneralAccumulation,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        self.run_backend_session_cached(None, move |session| {
            session.dot_general_read_into_accum(lhs, rhs, config, accumulation, out)
        })
    }

    fn dot_general_with_conj(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        self.run_backend_session_cached(None, move |session| {
            session.dot_general_with_conj(lhs, rhs, config, lhs_conj, rhs_conj)
        })
    }
}

impl BackendCachedDot for CpuBackend {
    fn dot_general_cached(
        &mut self,
        cache: &mut Self::RuntimeCache,
        cache_slot: Option<usize>,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        self.run_backend_session_cached(Some(cache), move |session| {
            session.dot_general_cached(cache_slot, lhs, rhs, config)
        })
    }

    fn dot_general_with_conj_cached(
        &mut self,
        cache: &mut Self::RuntimeCache,
        cache_slot: Option<usize>,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        self.run_backend_session_cached(Some(cache), move |session| {
            session.dot_general_with_conj_cached(cache_slot, lhs, rhs, config, lhs_conj, rhs_conj)
        })
    }

    fn dot_general_read_into_accum_cached(
        &mut self,
        cache: &mut Self::RuntimeCache,
        cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        accumulation: DotGeneralAccumulation,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        self.run_backend_session_cached(Some(cache), move |session| {
            session.dot_general_read_into_accum_cached(
                cache_slot,
                lhs,
                rhs,
                config,
                accumulation,
                out,
            )
        })
    }

    fn grouped_gemm_cached(
        &mut self,
        cache: &mut Self::RuntimeCache,
        cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &GroupedGemmConfig<'_>,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        self.run_backend_session_cached(Some(cache), move |session| {
            session.grouped_gemm_cached(cache_slot, lhs, rhs, config, out)
        })
    }
}

impl TensorIndexing for CpuBackend {
    fn gather(
        &mut self,
        operand: &Tensor,
        start_indices: &Tensor,
        config: &GatherConfig,
    ) -> crate::Result<Tensor> {
        self.install_with_indexed_pool_context(|context, buffers, cache| {
            let exec_context = context.strided_exec_context();
            indexing::gather_with_pool(
                buffers,
                cache,
                &exec_context,
                operand,
                start_indices,
                config,
            )
        })
    }

    fn scatter(
        &mut self,
        operand: &Tensor,
        scatter_indices: &Tensor,
        updates: &Tensor,
        config: &ScatterConfig,
    ) -> crate::Result<Tensor> {
        self.install_with_indexed_pool_context(|context, buffers, cache| {
            let exec_context = context.strided_exec_context();
            indexing::scatter_with_pool(
                buffers,
                cache,
                &exec_context,
                operand,
                scatter_indices,
                updates,
                config,
            )
        })
    }

    fn slice(&mut self, input: &Tensor, config: &SliceConfig) -> crate::Result<Tensor> {
        self.install_with_pool_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            indexing::try_slice_with_pool(buffers, &exec_context, input, config)
        })
    }

    fn dynamic_slice(
        &mut self,
        input: &Tensor,
        starts: &Tensor,
        slice_sizes: &[usize],
    ) -> crate::Result<Tensor> {
        self.install_with_indexed_pool_context(|context, buffers, cache| {
            let exec_context = context.strided_exec_context();
            indexing::dynamic_slice_with_pool(
                buffers,
                cache,
                &exec_context,
                input,
                starts,
                slice_sizes,
            )
        })
    }

    fn dynamic_update_slice(
        &mut self,
        operand: &Tensor,
        update: &Tensor,
        starts: &Tensor,
    ) -> crate::Result<Tensor> {
        self.install_with_indexed_pool_context(|context, buffers, cache| {
            let exec_context = context.strided_exec_context();
            indexing::dynamic_update_slice_with_pool(
                buffers,
                cache,
                &exec_context,
                operand,
                update,
                starts,
            )
        })
    }

    fn pad(&mut self, input: &Tensor, config: &PadConfig) -> crate::Result<Tensor> {
        self.install_with_pool_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            indexing::try_pad_with_pool(buffers, &exec_context, input, config)
        })
    }

    fn concatenate(&mut self, inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor> {
        self.install_with_pool_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            indexing::try_concatenate_with_pool(buffers, &exec_context, inputs, axis)
        })
    }

    fn reverse(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.install_with_pool_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            indexing::reverse_with_pool(buffers, &exec_context, input, axes)
        })
    }
}

impl CpuBackend {
    /// Bind this backend handle to a shared-allocation domain.
    ///
    /// Host-only CPU behavior is unchanged. Operation crates can use the domain
    /// to require guarded access to matching managed allocations.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use std::sync::Arc;
    /// use tenferro_tensor::{AllocationDomainId, DType, SharedTensorAllocationDomain, Tensor};
    ///
    /// #[derive(Debug)]
    /// struct Domain(AllocationDomainId);
    /// impl SharedTensorAllocationDomain for Domain {
    ///     fn id(&self) -> AllocationDomainId { self.0 }
    ///     fn allocate(&self, _: DType, _: &[usize]) -> tenferro_tensor::Result<Tensor> {
    ///         Err(tenferro_tensor::Error::unsupported("example", "not implemented"))
    ///     }
    /// }
    /// let id = AllocationDomainId::fresh();
    /// let backend = CpuBackend::new().with_allocation_domain(Arc::new(Domain(id)));
    /// assert_eq!(backend.allocation_domain(), Some(id));
    /// ```
    pub fn with_allocation_domain(mut self, domain: Arc<dyn SharedTensorAllocationDomain>) -> Self {
        self.allocation_domain = Some(domain);
        self.runtime_identity = CpuRuntimeIdentity::fresh();
        self
    }

    /// Return the configured shared-allocation domain.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    ///
    /// assert_eq!(CpuBackend::new().allocation_domain(), None);
    /// ```
    pub fn allocation_domain(&self) -> Option<AllocationDomainId> {
        self.allocation_domain.as_ref().map(|domain| domain.id())
    }

    /// Return the allocator for this backend's shared domain.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    ///
    /// assert!(CpuBackend::new().shared_allocation_domain().is_none());
    /// ```
    pub fn shared_allocation_domain(&self) -> Option<&Arc<dyn SharedTensorAllocationDomain>> {
        self.allocation_domain.as_ref()
    }

    fn run_backend_session_cached<R: Send>(
        &mut self,
        cache: Option<&mut gemm::GemmAnalysisCache>,
        f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
    ) -> R {
        let providers = self.provider_bundle.clone();
        let owner = inherited_or_new_execution_owner();
        let permit = self.acquire_execution_permit(owner);
        let entry = CpuOperationEntry::new(self.engine.domain(), &permit);
        let enter_managed_session = entry.supports_infallible_session_entry()
            && !matches!(
                &self.resolved,
                ResolvedCpuExecution::ProviderDefaultExclusive
            );
        let run = |entered| {
            self.with_execution_resources(&permit, |resources| {
                let mut buffers = BufferPoolLoan::new(&mut resources.buffers);
                let cache = cache.unwrap_or(&mut resources.gemm_analysis_cache);
                let session_started = Instant::now();
                let mut session = CpuExecSession {
                    entry,
                    entered,
                    buffers: buffers.get_mut(),
                    gemm_analysis_cache: cache,
                    indexed_plan_cache: &mut resources.indexed_plan_cache,
                    providers: &providers,
                    backend_kind: self.kind(),
                    allocation_domain: self.allocation_domain.as_ref(),
                };
                record_cpu_session_profile(
                    "with_backend_session_cached.session_construct",
                    session_started.elapsed(),
                );
                let exec_started = Instant::now();
                let result = f(&mut session);
                record_cpu_session_profile(
                    "with_backend_session_cached.exec_body",
                    exec_started.elapsed(),
                );
                result
            })
        };
        if enter_managed_session {
            entry.enter_managed_session(|context| run(Some(context)))
        } else {
            with_execution_owner(owner, || run(None))
        }
    }
}

impl BackendSessionHost for CpuBackend {
    fn with_backend_session<R: Send>(
        &mut self,
        f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
    ) -> R {
        self.run_backend_session_cached(None, f)
    }

    fn with_backend_session_cached<R: Send>(
        &mut self,
        cache: &mut Self::RuntimeCache,
        f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
    ) -> R {
        if !cpu_session_profile_enabled() {
            return self.run_backend_session_cached(Some(cache), f);
        }
        let total_started = Instant::now();
        let result =
            profile_cpu_session_section("with_backend_session_cached.exec_session", || {
                self.run_backend_session_cached(Some(cache), f)
            });
        record_cpu_session_profile("with_backend_session_cached.total", total_started.elapsed());
        maybe_print_cpu_session_profile();
        result
    }
}

impl TensorBuffer for CpuBackend {
    fn reclaim_buffer(&mut self, tensor: Tensor) {
        let owner = inherited_or_new_execution_owner();
        with_execution_owner(owner, || {
            let permit = self.acquire_execution_permit(owner);
            self.with_execution_resources(&permit, |resources| {
                let buffers = &mut resources.buffers;
                match tensor {
                    Tensor::F32(t) => reclaim_typed(buffers, t),
                    Tensor::F64(t) => reclaim_typed(buffers, t),
                    Tensor::I32(t) => reclaim_typed(buffers, t),
                    Tensor::I64(t) => reclaim_typed(buffers, t),
                    Tensor::Bool(t) => reclaim_typed(buffers, t),
                    Tensor::C32(t) => reclaim_typed(buffers, t),
                    Tensor::C64(t) => reclaim_typed(buffers, t),
                }
            })
        })
    }
}

impl<T, R> TensorViewCanonicalization<T, R> for CpuBackend
where
    T: TensorScalar + PoolScalar,
    R: TensorRank,
    R::Shape: Send + Sync,
    R::Strides: Send + Sync,
{
    fn to_contiguous(
        &mut self,
        view: &TypedTensorView<'_, T, R>,
    ) -> crate::Result<TypedTensor<T, R>> {
        self.install_with_pool(|buffers| {
            structural::typed_materialize_view_with_pool(buffers, view, "CpuBackend::to_contiguous")
        })
    }

    fn copy_into(
        &mut self,
        src: &TypedTensorView<'_, T, R>,
        dst: &mut TypedTensorViewMut<'_, T, R>,
    ) -> crate::Result<()> {
        self.try_install(|| structural::typed_copy_view_into(src, dst, "CpuBackend::copy_into"))
    }
}

impl TensorFusion for CpuBackend {
    fn execute_elementwise_fusion(
        &mut self,
        inputs: &[&Tensor],
        plan: &ElementwiseFusionPlan,
    ) -> crate::Result<Option<Vec<Tensor>>> {
        self.install_with_pool_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            elementwise::elementwise_fusion_with_pool(buffers, &exec_context, inputs, plan)
        })
    }

    fn execute_broadcast_multiply(
        &mut self,
        lhs: TensorRead<'_>,
        lhs_shape: &[usize],
        lhs_dims: &[usize],
        rhs: TensorRead<'_>,
        rhs_shape: &[usize],
        rhs_dims: &[usize],
    ) -> crate::Result<Option<Tensor>> {
        self.install_with_pool(|buffers| {
            elementwise::broadcast_multiply_read_with_pool(
                buffers, lhs, lhs_shape, lhs_dims, rhs, rhs_shape, rhs_dims,
            )
        })
    }

    fn execute_broadcast_multiply_value(
        &mut self,
        lhs: TensorRead<'_>,
        lhs_shape: &[usize],
        lhs_dims: &[usize],
        rhs: TensorRead<'_>,
        rhs_shape: &[usize],
        rhs_dims: &[usize],
    ) -> crate::Result<Option<TensorValue>> {
        let domain = self.engine.domain().id();
        self.install_with_pool_unmarked(|buffers| {
            elementwise::broadcast_multiply_value_with_pool_and_tag(
                buffers,
                lhs,
                lhs_shape,
                lhs_dims,
                rhs,
                rhs_shape,
                rhs_dims,
                |tensor| tag_fresh_output(tensor, domain),
            )
        })
    }
}

impl TensorDeviceTransfer for CpuBackend {
    fn download_to_host(&mut self, tensor: &Tensor) -> crate::Result<Tensor> {
        if tensor.is_backend_buffer() {
            return Err(crate::Error::runtime_state(
                "CpuBackend::download_to_host",
                "CPU backend received a backend buffer; download the tensor to host with its owning backend before CPU execution",
            ));
        }
        Ok(tensor.clone())
    }

    fn upload_host_tensor(&mut self, tensor: &Tensor) -> crate::Result<Tensor> {
        if tensor.is_backend_buffer() {
            return Err(crate::Error::runtime_state(
                "CpuBackend::upload_host_tensor",
                "CPU backend upload_host_tensor expects a host tensor; download backend buffers to host before CPU execution",
            ));
        }
        Ok(tensor.clone())
    }
}

impl TensorBackend for CpuBackend {}

pub(crate) fn reclaim_typed<T: PoolScalar>(pool: &mut BufferPool, typed: TypedTensor<T>) {
    let (buffer, _, _) = typed.into_parts();
    match buffer {
        Buffer::Host(data) => T::pool_release(pool, data),
        Buffer::Backend(_) => {}
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
