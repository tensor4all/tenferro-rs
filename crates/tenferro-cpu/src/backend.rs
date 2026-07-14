use std::cmp::Reverse;
use std::collections::{BTreeMap, HashMap};
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
use crate::engine::{CpuEngine, EngineResources};
use crate::placement::{resolve_placement, resolve_placement_with_affinity, ResolvedCpuExecution};
use crate::{
    discover_cpu_topology, CpuId, CpuPlacement, CpuPlacementError, CpuSet, CpuTopology,
    CpuTopologyError, NumaNodeId, ResolvedCpuPlacement,
};
use crate::{
    Buffer, CacheStats, Tensor, TensorRank, TensorRead, TensorValue, TensorWrite, TypedTensor,
    TypedTensorView, TypedTensorViewMut,
};
use tenferro_tensor::backend::{
    dot_general_accum_via_temp, grouped_gemm_via_sequential, validate_dot_general_accumulation,
    validate_grouped_gemm, ElementwiseFusionPlan, GroupedGemmConfig,
};
use tenferro_tensor::{
    BackendCachedDot, BackendRuntimeCache, BackendSession, BackendSessionHost,
    DotGeneralAccumulation, TensorAnalytic, TensorBackend, TensorBuffer, TensorDeviceTransfer,
    TensorDot, TensorElementwise, TensorFusion, TensorIndexing, TensorReduction, TensorStructural,
    TensorViewCanonicalization,
};
use tenferro_tensor::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};

use super::exec_session::CpuExecSession;
use super::{
    analytic, elementwise, gemm, indexing, materialize_tensor_read, reduction, structural,
    CpuContext,
};

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
    target: &'a mut BufferPool,
    buffers: Option<BufferPool>,
}

impl<'a> BufferPoolLoan<'a> {
    fn new(target: &'a mut BufferPool) -> Self {
        Self {
            buffers: Some(std::mem::take(target)),
            target,
        }
    }

    fn get_mut(&mut self) -> &mut BufferPool {
        self.buffers
            .as_mut()
            .expect("buffer pool loan already restored")
    }
}

impl Drop for BufferPoolLoan<'_> {
    fn drop(&mut self) {
        if let Some(buffers) = self.buffers.take() {
            let mut buffers = buffers;
            if thread::panicking() {
                buffers.replenish_in_flight_retained();
            } else {
                buffers.clear_in_flight_retained();
            }
            *self.target = buffers;
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
///         | CpuExecutionMode::ProviderDefaultExclusive
///         | CpuExecutionMode::Compatibility
/// ));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CpuExecutionMode {
    /// tenferro owns a pinned Rayon engine for the resolved CPU placement.
    Managed,
    /// An external provider owns worker placement under a process-wide permit.
    ProviderDefaultExclusive,
    /// A legacy unpinned Rayon context is used because managed affinity is unavailable.
    Compatibility,
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
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
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
        }
    }
}

impl From<CpuBackendError> for crate::Error {
    fn from(error: CpuBackendError) -> Self {
        match error {
            CpuBackendError::Tensor(error) => error,
            CpuBackendError::Placement { op, source } => Self::backend_failure(op, source),
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
    worker_count: usize,
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

    /// Return the concrete CPU placement when tenferro owns worker affinity.
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

    /// Return the worker count of the selected native execution context.
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

fn provider_diagnostic(kind: CpuBackendKind) -> &'static str {
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
                Err(crate::Error::InvalidConfig {
                    op,
                    message: "CpuBackendKind::Faer requires the cpu-faer feature".to_string(),
                })
            }
        }
        CpuBackendKind::Blas => {
            #[cfg(feature = "cpu-blas")]
            {
                Ok(())
            }
            #[cfg(not(feature = "cpu-blas"))]
            {
                Err(crate::Error::InvalidConfig {
                    op,
                    message: "CpuBackendKind::Blas requires the cpu-blas feature".to_string(),
                })
            }
        }
    }
}

fn constructor_tensor_error(op: &'static str, error: crate::Error) -> CpuBackendError {
    CpuBackendError::Tensor(match error {
        crate::Error::InvalidConfig { message, .. } => crate::Error::InvalidConfig { op, message },
        crate::Error::BackendFailure { message, .. } => crate::Error::backend_failure(op, message),
        error => error,
    })
}

// Used by feature-disabled backend paths; a given feature build may compile no
// direct call site for one provider.
#[allow(dead_code)]
pub(super) fn unavailable_cpu_backend_kind(kind: CpuBackendKind, op: &'static str) -> crate::Error {
    crate::Error::InvalidConfig {
        op,
        message: format!("CPU backend kind {} is not compiled in", kind.name()),
    }
}

struct CpuBackendState {
    topology: CpuTopology,
    node_engines: Mutex<BTreeMap<NumaNodeId, Arc<CpuEngine>>>,
    all_allowed: OnceLock<Arc<CpuEngine>>,
    all_allowed_build: Mutex<()>,
    base_engine: Arc<CpuEngine>,
    arbiter: ResourceArbiter,
    kind: CpuBackendKind,
    thread_budget: usize,
    buffer_limit: AtomicUsize,
}

impl CpuBackendState {
    fn engine_for(
        &self,
        placement: &ResolvedCpuPlacement,
    ) -> Result<Arc<CpuEngine>, crate::CpuContextError> {
        match placement {
            ResolvedCpuPlacement::NumaNode { id, .. } => {
                let mut engines = self
                    .node_engines
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Some(engine) = engines.get(id) {
                    return Ok(Arc::clone(engine));
                }
                let engine = Arc::new(CpuEngine::new(
                    placement.clone(),
                    self.thread_budget,
                    self.buffer_limit.load(Ordering::Relaxed),
                )?);
                engines.insert(*id, Arc::clone(&engine));
                Ok(engine)
            }
            ResolvedCpuPlacement::AllAllowed { .. } => {
                if let Some(engine) = self.all_allowed.get() {
                    return Ok(Arc::clone(engine));
                }
                let _build = self
                    .all_allowed_build
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Some(engine) = self.all_allowed.get() {
                    return Ok(Arc::clone(engine));
                }
                let engine = Arc::new(CpuEngine::new(
                    placement.clone(),
                    self.thread_budget,
                    self.buffer_limit.load(Ordering::Relaxed),
                )?);
                let _ = self.all_allowed.set(Arc::clone(&engine));
                Ok(engine)
            }
        }
    }

    fn initialized_engines(&self) -> Vec<Arc<CpuEngine>> {
        let mut engines = vec![Arc::clone(&self.base_engine)];
        if let Some(engine) = self.all_allowed.get() {
            engines.push(Arc::clone(engine));
        }
        engines.extend(
            self.node_engines
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .values()
                .cloned(),
        );
        engines.sort_unstable_by_key(|engine| Arc::as_ptr(engine) as usize);
        engines.dedup_by(|left, right| Arc::ptr_eq(left, right));
        engines
    }
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
    shared: Arc<CpuBackendState>,
    requested: CpuPlacement,
    resolved: ResolvedCpuExecution,
    engine: Arc<CpuEngine>,
}

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

impl fmt::Debug for CpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuBackend")
            .field("kind", &self.kind())
            .field("requested_placement", &self.requested)
            .field("resolved_execution", &self.resolved)
            .field("engine_placement", &self.engine.placement())
            .field("num_threads", &self.num_threads())
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
        if !cfg!(any(target_os = "linux", target_os = "android")) {
            let context = CpuContext::with_threads(thread_budget).map_err(|error| {
                CpuPlacementError::EngineConstruction {
                    requested: CpuPlacement::Auto,
                    backend: kind,
                    message: error.to_string(),
                }
            })?;
            return Ok(Self::compatibility_with_topology(
                Arc::new(context),
                max_retained_capacity_bytes,
                kind,
                topology,
                resolved,
            ));
        }
        let engine_placement = ResolvedCpuPlacement::AllAllowed {
            cpus: topology.allowed_cpus().clone(),
        };
        let engine = Arc::new(
            CpuEngine::new(engine_placement, thread_budget, max_retained_capacity_bytes).map_err(
                |error| CpuPlacementError::EngineConstruction {
                    requested: CpuPlacement::Auto,
                    backend: kind,
                    message: error.to_string(),
                },
            )?,
        );
        let all_allowed = OnceLock::new();
        let _ = all_allowed.set(Arc::clone(&engine));
        Ok(Self {
            shared: Arc::new(CpuBackendState {
                topology,
                node_engines: Mutex::new(BTreeMap::new()),
                all_allowed,
                all_allowed_build: Mutex::new(()),
                base_engine: Arc::clone(&engine),
                arbiter: ResourceArbiter::global(),
                kind,
                thread_budget,
                buffer_limit: AtomicUsize::new(max_retained_capacity_bytes),
            }),
            requested: CpuPlacement::Auto,
            resolved,
            engine,
        })
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
            placement,
            ctx,
            max_retained_capacity_bytes,
        ));
        Self {
            shared: Arc::new(CpuBackendState {
                topology,
                node_engines: Mutex::new(BTreeMap::new()),
                all_allowed: OnceLock::new(),
                all_allowed_build: Mutex::new(()),
                base_engine: Arc::clone(&base_engine),
                arbiter: ResourceArbiter::global(),
                kind,
                thread_budget: base_engine.context().num_threads(),
                buffer_limit: AtomicUsize::new(max_retained_capacity_bytes),
            }),
            requested: CpuPlacement::Auto,
            resolved,
            engine: base_engine,
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
    /// Returns an error when the provider is unavailable or CPU topology and
    /// placement initialization fails.
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
    /// Returns an error when the environment-driven thread configuration is
    /// invalid or CPU topology and placement initialization fails.
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
    /// Returns an error when `num_threads` is zero, Rayon rejects the pool, or
    /// CPU topology and placement initialization fails.
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
    /// Returns an error when `num_threads` is zero, Rayon rejects the pool, or
    /// the selected provider is unavailable, or CPU topology and placement
    /// initialization fails.
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
    /// Explicit placement is supported for faer/native execution. External
    /// BLAS providers accept only [`CpuPlacement::Auto`] because tenferro does
    /// not control their worker affinity.
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
        let resolved = resolve_placement_with_affinity(
            self.kind(),
            requested,
            &self.shared.topology,
            managed_affinity_available,
        )?;
        if requested == CpuPlacement::Auto && !managed_affinity_available {
            return Ok(Self {
                shared: Arc::clone(&self.shared),
                requested,
                resolved,
                engine: Arc::clone(&self.shared.base_engine),
            });
        }
        let engine_placement = match &resolved {
            ResolvedCpuExecution::Managed(placement) => placement.clone(),
            ResolvedCpuExecution::ProviderDefaultExclusive => ResolvedCpuPlacement::AllAllowed {
                cpus: self.shared.topology.allowed_cpus().clone(),
            },
            ResolvedCpuExecution::Compatibility => {
                return Err(CpuPlacementError::EngineConstruction {
                    requested,
                    backend: self.kind(),
                    message: "placement resolution returned an internal compatibility mode"
                        .to_owned(),
                });
            }
        };
        let engine = self.shared.engine_for(&engine_placement).map_err(|error| {
            CpuPlacementError::EngineConstruction {
                requested,
                backend: self.kind(),
                message: error.to_string(),
            }
        })?;
        Ok(Self {
            shared: Arc::clone(&self.shared),
            requested,
            resolved,
            engine,
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

    /// Return the concrete managed placement, if tenferro owns worker affinity.
    ///
    /// External-provider and compatibility contexts return `None`.
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
            ResolvedCpuExecution::Managed(placement) => Some(placement),
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

    /// Report whether this public provider kind accepts a placement request.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuBackend, CpuPlacement};
    ///
    /// assert!(CpuBackend::new().supports_placement(CpuPlacement::Auto));
    /// ```
    pub fn supports_placement(&self, placement: CpuPlacement) -> bool {
        resolve_placement(self.kind(), placement, &self.shared.topology).is_ok()
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
        CpuExecutionInfo {
            backend_kind: self.kind(),
            execution_mode: match &self.resolved {
                ResolvedCpuExecution::Managed(_) => CpuExecutionMode::Managed,
                ResolvedCpuExecution::ProviderDefaultExclusive => {
                    CpuExecutionMode::ProviderDefaultExclusive
                }
                ResolvedCpuExecution::Compatibility => CpuExecutionMode::Compatibility,
            },
            requested_placement: self.requested,
            resolved_placement: self.resolved_placement().cloned(),
            topology: self.shared.topology.clone(),
            worker_count: self.num_threads(),
            provider_diagnostic: provider_diagnostic(self.kind()),
        }
    }

    #[cfg(all(test, feature = "cpu-faer"))]
    fn coordinator_id_for_test(&self) -> usize {
        Arc::as_ptr(&self.shared) as usize
    }

    #[cfg(test)]
    pub(crate) fn context_id_for_test(&self) -> usize {
        Arc::as_ptr(&self.engine.context_arc()) as usize
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

    /// Return the number of threads in this backend's CPU context.
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
        self.engine.context().num_threads()
    }

    /// Number of retained typed host buffers currently held by this backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// assert_eq!(backend.buffer_pool_len(), 0);
    /// ```
    pub fn buffer_pool_len(&self) -> usize {
        self.shared
            .initialized_engines()
            .into_iter()
            .map(|engine| {
                engine
                    .resources
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .buffers
                    .len()
            })
            .sum()
    }

    /// Snapshot reusable typed host buffers currently retained by this backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// let stats = backend.buffer_pool_stats();
    /// assert_eq!(stats.buffers, 0);
    /// assert_eq!(stats.capacity_bytes, 0);
    /// ```
    pub fn buffer_pool_stats(&self) -> BufferPoolStats {
        self.shared.initialized_engines().into_iter().fold(
            BufferPoolStats::default(),
            |mut total, engine| {
                let stats = engine
                    .resources
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .buffers
                    .stats();
                total.buffers += stats.buffers;
                total.capacity_bytes += stats.capacity_bytes;
                total
            },
        )
    }

    /// Return cache-style stats for the CPU buffer pool.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// let stats = backend.buffer_pool_cache_stats();
    /// assert_eq!(stats.entries, 0);
    /// assert_eq!(stats.retained_bytes, 0);
    /// ```
    pub fn buffer_pool_cache_stats(&self) -> CacheStats {
        let stats = self.buffer_pool_stats();
        CacheStats {
            entries: stats.buffers,
            retained_bytes: stats.capacity_bytes,
        }
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
    /// backend.set_buffer_pool_limit_bytes(0);
    /// assert_eq!(backend.buffer_pool_limit_bytes(), 0);
    /// assert_eq!(backend.buffer_pool_len(), 0);
    /// ```
    pub fn set_buffer_pool_limit_bytes(&mut self, max_retained_capacity_bytes: usize) {
        self.shared
            .buffer_limit
            .store(max_retained_capacity_bytes, Ordering::Relaxed);
        for engine in self.shared.initialized_engines() {
            engine
                .resources
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .buffers
                .set_max_retained_capacity_bytes(max_retained_capacity_bytes);
        }
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
    /// backend.reset_buffer_pool();
    /// assert_eq!(backend.buffer_pool_len(), 0);
    /// ```
    pub fn reset_buffer_pool(&mut self) {
        for engine in self.shared.initialized_engines() {
            engine
                .resources
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .buffers
                .clear();
        }
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
    /// provider exclusivity.
    pub fn install<R: Send>(&self, op: impl FnOnce() -> R + Send) -> R {
        let owner = inherited_or_new_execution_owner();
        let _permit = self.acquire_execution_permit(owner);
        self.engine
            .context()
            .install_with_execution_owner(owner, op)
    }

    fn install_with_pool<R: Send>(&mut self, op: impl FnOnce(&mut BufferPool) -> R + Send) -> R {
        let ctx = self.engine.context_arc();
        let owner = inherited_or_new_execution_owner();
        let permit = self.acquire_execution_permit(owner);
        ctx.install_with_execution_owner(owner, || {
            self.with_execution_resources(&permit, |resources| {
                let mut buffers = BufferPoolLoan::new(&mut resources.buffers);
                op(buffers.get_mut())
            })
        })
    }

    // Selected when the BLAS provider is active; default Faer-only builds keep
    // it dormant.
    #[allow(dead_code)]
    fn run_with_pool<R>(&mut self, op: impl FnOnce(&mut BufferPool) -> R) -> R {
        let owner = inherited_or_new_execution_owner();
        with_execution_owner(owner, || {
            let permit = self.acquire_execution_permit(owner);
            self.with_execution_resources(&permit, |resources| {
                let mut buffers = BufferPoolLoan::new(&mut resources.buffers);
                op(buffers.get_mut())
            })
        })
    }

    fn linalg_with_pool<R: Send>(&mut self, op: impl FnOnce(&mut BufferPool) -> R + Send) -> R {
        match self.kind() {
            CpuBackendKind::Faer => self.install_with_pool(op),
            CpuBackendKind::Blas => self.run_with_pool(op),
        }
    }

    /// Run an external linalg implementation with this backend's buffer pool.
    ///
    /// This is exposed for operation-family crates that own their backend
    /// implementation while still sharing the CPU backend's allocation pool.
    #[doc(hidden)]
    pub fn with_linalg_pool<R: Send>(&mut self, op: impl FnOnce(&mut BufferPool) -> R + Send) -> R {
        self.linalg_with_pool(op)
    }

    /// Clone the CPU context used by external linalg implementations.
    #[cfg(feature = "cpu-faer")]
    #[doc(hidden)]
    pub fn linalg_context(&self) -> Arc<CpuContext> {
        self.engine.context_arc()
    }

    // Selected when the Faer provider handles cached GEMM execution; some
    // feature combinations compile only the uncached or BLAS path.
    #[allow(dead_code)]
    fn install_with_pool_and_gemm_cache<R: Send>(
        &mut self,
        gemm_analysis_cache: &mut gemm::GemmAnalysisCache,
        op: impl FnOnce(&mut BufferPool, &mut gemm::GemmAnalysisCache) -> R + Send,
    ) -> R {
        let ctx = self.engine.context_arc();
        let owner = inherited_or_new_execution_owner();
        let permit = self.acquire_execution_permit(owner);
        ctx.install_with_execution_owner(owner, || {
            self.with_execution_resources(&permit, |resources| {
                let mut buffers = BufferPoolLoan::new(&mut resources.buffers);
                op(buffers.get_mut(), gemm_analysis_cache)
            })
        })
    }

    // Selected when the BLAS provider handles cached GEMM execution; default
    // Faer-only builds keep it dormant.
    #[allow(dead_code)]
    fn run_with_pool_and_gemm_cache<R>(
        &mut self,
        gemm_analysis_cache: &mut gemm::GemmAnalysisCache,
        op: impl FnOnce(&mut BufferPool, &mut gemm::GemmAnalysisCache) -> R,
    ) -> R {
        let owner = inherited_or_new_execution_owner();
        with_execution_owner(owner, || {
            let permit = self.acquire_execution_permit(owner);
            self.with_execution_resources(&permit, |resources| {
                let mut buffers = BufferPoolLoan::new(&mut resources.buffers);
                op(buffers.get_mut(), gemm_analysis_cache)
            })
        })
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
            ResolvedCpuExecution::Managed(placement) => self
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
}

impl BackendRuntimeCache for CpuBackend {
    type RuntimeCache = gemm::GemmAnalysisCache;
}

impl TensorElementwise for CpuBackend {
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
    fn transpose(&mut self, input: &Tensor, perm: &[usize]) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| structural::transpose_with_pool(buffers, input, perm))
    }

    fn transpose_read(&mut self, input: TensorRead<'_>, perm: &[usize]) -> crate::Result<Tensor> {
        if let Some(input) = input.as_tensor() {
            return self.transpose(input, perm);
        }

        let input = materialize_tensor_read("transpose", input)?;
        self.transpose(&input, perm)
    }

    fn reshape(&mut self, input: &Tensor, shape: &[usize]) -> crate::Result<Tensor> {
        self.install(|| structural::reshape(input, shape))
    }

    fn reshape_read(&mut self, input: TensorRead<'_>, shape: &[usize]) -> crate::Result<Tensor> {
        if let Some(input) = input.as_tensor() {
            return self.reshape(input, shape);
        }

        let input = materialize_tensor_read("reshape", input)?;
        self.reshape(&input, shape)
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
        if let Some(input) = input.as_tensor() {
            return self.broadcast_in_dim(input, shape, dims);
        }

        let input = materialize_tensor_read("broadcast_in_dim", input)?;
        self.broadcast_in_dim(&input, shape, dims)
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
        self.install(|| reduction::reduce_sum(input, axes))
    }

    fn reduce_sum_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.install(|| reduction::reduce_sum_read(input, axes))
    }

    fn reduce_prod(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.install(|| reduction::reduce_prod(input, axes))
    }

    fn reduce_prod_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.install(|| reduction::reduce_prod_read(input, axes))
    }

    fn reduce_max(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.install(|| reduction::reduce_max(input, axes))
    }

    fn reduce_max_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.install(|| reduction::reduce_max_read(input, axes))
    }

    fn reduce_min(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.install(|| reduction::reduce_min(input, axes))
    }

    fn reduce_min_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.install(|| reduction::reduce_min_read(input, axes))
    }
}

impl TensorDot for CpuBackend {
    fn dot_general(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        let mut cache = gemm::GemmAnalysisCache::default();
        BackendCachedDot::dot_general_cached(self, &mut cache, None, lhs, rhs, config)
    }

    fn dot_general_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        let mut cache = gemm::GemmAnalysisCache::default();
        let direct = match self.kind() {
            CpuBackendKind::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    let ctx = self.engine.context_arc();
                    self.install_with_pool_and_gemm_cache(&mut cache, |buffers, cache| {
                        gemm::dot_general_faer_read_cached(
                            buffers,
                            cache,
                            None,
                            ctx.as_ref(),
                            lhs.clone(),
                            rhs.clone(),
                            config,
                        )
                    })?
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    return Err(unavailable_cpu_backend_kind(self.kind(), "dot_general"));
                }
            }
            CpuBackendKind::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    self.run_with_pool_and_gemm_cache(&mut cache, |buffers, cache| {
                        gemm::dot_general_blas_read_cached(
                            buffers,
                            cache,
                            None,
                            lhs.clone(),
                            rhs.clone(),
                            config,
                        )
                    })?
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    return Err(unavailable_cpu_backend_kind(self.kind(), "dot_general"));
                }
            }
        };
        if let Some(result) = direct {
            return Ok(result);
        }

        let lhs = materialize_tensor_read("dot_general", lhs)?;
        let rhs = materialize_tensor_read("dot_general", rhs)?;
        BackendCachedDot::dot_general_cached(self, &mut cache, None, &lhs, &rhs, config)
    }

    fn dot_general_read_into(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        let accumulation = DotGeneralAccumulation::overwrite(lhs.dtype())?;
        self.dot_general_read_into_accum(lhs, rhs, config, accumulation, out)
    }

    fn dot_general_read_into_accum(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        accumulation: DotGeneralAccumulation,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        let mut cache = gemm::GemmAnalysisCache::default();
        BackendCachedDot::dot_general_read_into_accum_cached(
            self,
            &mut cache,
            None,
            lhs,
            rhs,
            config,
            accumulation,
            out,
        )
    }

    fn dot_general_with_conj(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        let mut cache = gemm::GemmAnalysisCache::default();
        BackendCachedDot::dot_general_with_conj_cached(
            self, &mut cache, None, lhs, rhs, config, lhs_conj, rhs_conj,
        )
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
        match self.kind() {
            CpuBackendKind::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    let ctx = self.engine.context_arc();
                    self.install_with_pool_and_gemm_cache(cache, |buffers, cache| {
                        match (lhs, rhs) {
                            (Tensor::F32(a), Tensor::F32(b)) => gemm::dot_general_faer_cached(
                                buffers,
                                cache,
                                cache_slot,
                                ctx.as_ref(),
                                a,
                                b,
                                config,
                            )
                            .map(Tensor::F32),
                            (Tensor::F64(a), Tensor::F64(b)) => gemm::dot_general_faer_cached(
                                buffers,
                                cache,
                                cache_slot,
                                ctx.as_ref(),
                                a,
                                b,
                                config,
                            )
                            .map(Tensor::F64),
                            (Tensor::C32(a), Tensor::C32(b)) => gemm::dot_general_faer_cached(
                                buffers,
                                cache,
                                cache_slot,
                                ctx.as_ref(),
                                a,
                                b,
                                config,
                            )
                            .map(Tensor::C32),
                            (Tensor::C64(a), Tensor::C64(b)) => gemm::dot_general_faer_cached(
                                buffers,
                                cache,
                                cache_slot,
                                ctx.as_ref(),
                                a,
                                b,
                                config,
                            )
                            .map(Tensor::C64),
                            _ => Err(crate::Error::DTypeMismatch {
                                op: "dot_general",
                                lhs: lhs.dtype(),
                                rhs: rhs.dtype(),
                            }),
                        }
                    })
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    Err(unavailable_cpu_backend_kind(self.kind(), "dot_general"))
                }
            }
            CpuBackendKind::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    self.run_with_pool_and_gemm_cache(cache, |buffers, cache| match (lhs, rhs) {
                        (Tensor::F32(a), Tensor::F32(b)) => {
                            gemm::dot_general_blas_cached(buffers, cache, cache_slot, a, b, config)
                                .map(Tensor::F32)
                        }
                        (Tensor::F64(a), Tensor::F64(b)) => {
                            gemm::dot_general_blas_cached(buffers, cache, cache_slot, a, b, config)
                                .map(Tensor::F64)
                        }
                        (Tensor::C32(a), Tensor::C32(b)) => {
                            gemm::dot_general_blas_cached(buffers, cache, cache_slot, a, b, config)
                                .map(Tensor::C32)
                        }
                        (Tensor::C64(a), Tensor::C64(b)) => {
                            gemm::dot_general_blas_cached(buffers, cache, cache_slot, a, b, config)
                                .map(Tensor::C64)
                        }
                        _ => Err(crate::Error::DTypeMismatch {
                            op: "dot_general",
                            lhs: lhs.dtype(),
                            rhs: rhs.dtype(),
                        }),
                    })
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    Err(unavailable_cpu_backend_kind(self.kind(), "dot_general"))
                }
            }
        }
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
        match self.kind() {
            CpuBackendKind::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    let ctx = self.engine.context_arc();
                    self.install_with_pool_and_gemm_cache(cache, |buffers, cache| {
                        match (lhs, rhs) {
                            (Tensor::F32(a), Tensor::F32(b)) => {
                                gemm::dot_general_faer_with_conj_cached(
                                    buffers,
                                    cache,
                                    cache_slot,
                                    ctx.as_ref(),
                                    a,
                                    b,
                                    config,
                                    lhs_conj,
                                    rhs_conj,
                                )
                                .map(Tensor::F32)
                            }
                            (Tensor::F64(a), Tensor::F64(b)) => {
                                gemm::dot_general_faer_with_conj_cached(
                                    buffers,
                                    cache,
                                    cache_slot,
                                    ctx.as_ref(),
                                    a,
                                    b,
                                    config,
                                    lhs_conj,
                                    rhs_conj,
                                )
                                .map(Tensor::F64)
                            }
                            (Tensor::C32(a), Tensor::C32(b)) => {
                                gemm::dot_general_faer_with_conj_cached(
                                    buffers,
                                    cache,
                                    cache_slot,
                                    ctx.as_ref(),
                                    a,
                                    b,
                                    config,
                                    lhs_conj,
                                    rhs_conj,
                                )
                                .map(Tensor::C32)
                            }
                            (Tensor::C64(a), Tensor::C64(b)) => {
                                gemm::dot_general_faer_with_conj_cached(
                                    buffers,
                                    cache,
                                    cache_slot,
                                    ctx.as_ref(),
                                    a,
                                    b,
                                    config,
                                    lhs_conj,
                                    rhs_conj,
                                )
                                .map(Tensor::C64)
                            }
                            _ => Err(crate::Error::DTypeMismatch {
                                op: "dot_general",
                                lhs: lhs.dtype(),
                                rhs: rhs.dtype(),
                            }),
                        }
                    })
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    Err(unavailable_cpu_backend_kind(self.kind(), "dot_general"))
                }
            }
            CpuBackendKind::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    self.run_with_pool_and_gemm_cache(cache, |buffers, cache| match (lhs, rhs) {
                        (Tensor::F32(a), Tensor::F32(b)) => {
                            gemm::dot_general_blas_with_conj_cached(
                                buffers, cache, cache_slot, a, b, config, lhs_conj, rhs_conj,
                            )
                            .map(Tensor::F32)
                        }
                        (Tensor::F64(a), Tensor::F64(b)) => {
                            gemm::dot_general_blas_with_conj_cached(
                                buffers, cache, cache_slot, a, b, config, lhs_conj, rhs_conj,
                            )
                            .map(Tensor::F64)
                        }
                        (Tensor::C32(a), Tensor::C32(b)) => {
                            gemm::dot_general_blas_with_conj_cached(
                                buffers, cache, cache_slot, a, b, config, lhs_conj, rhs_conj,
                            )
                            .map(Tensor::C32)
                        }
                        (Tensor::C64(a), Tensor::C64(b)) => {
                            gemm::dot_general_blas_with_conj_cached(
                                buffers, cache, cache_slot, a, b, config, lhs_conj, rhs_conj,
                            )
                            .map(Tensor::C64)
                        }
                        _ => Err(crate::Error::DTypeMismatch {
                            op: "dot_general",
                            lhs: lhs.dtype(),
                            rhs: rhs.dtype(),
                        }),
                    })
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    Err(unavailable_cpu_backend_kind(self.kind(), "dot_general"))
                }
            }
        }
    }

    fn dot_general_read_into_accum_cached(
        &mut self,
        cache: &mut Self::RuntimeCache,
        cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        accumulation: DotGeneralAccumulation,
        mut out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        validate_dot_general_accumulation(&lhs, &rhs, config, accumulation, &out, "dot_general")?;
        let direct = match self.kind() {
            CpuBackendKind::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    let ctx = self.engine.context_arc();
                    self.install_with_pool_and_gemm_cache(cache, |_buffers, cache| {
                        gemm::dot_general_faer_read_into_accum_cached(
                            cache,
                            cache_slot,
                            ctx.as_ref(),
                            lhs.clone(),
                            rhs.clone(),
                            config,
                            accumulation,
                            &mut out,
                        )
                    })?
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    return Err(unavailable_cpu_backend_kind(self.kind(), "dot_general"));
                }
            }
            CpuBackendKind::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    self.run_with_pool_and_gemm_cache(cache, |buffers, cache| {
                        gemm::dot_general_blas_read_into_accum_cached(
                            buffers,
                            cache,
                            cache_slot,
                            lhs.clone(),
                            rhs.clone(),
                            config,
                            accumulation,
                            &mut out,
                        )
                    })?
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    return Err(unavailable_cpu_backend_kind(self.kind(), "dot_general"));
                }
            }
        };
        if direct {
            return Ok(());
        }

        dot_general_accum_via_temp(self, lhs, rhs, config, accumulation, out)
    }

    fn grouped_gemm_cached(
        &mut self,
        _cache: &mut Self::RuntimeCache,
        _cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &GroupedGemmConfig<'_>,
        mut out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        validate_grouped_gemm(&lhs, &rhs, &out, config, "grouped_gemm")?;
        let direct = match self.kind() {
            CpuBackendKind::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    let ctx = self.engine.context_arc();
                    self.install_with_pool(|_buffers| {
                        gemm::grouped_gemm_faer_cached(
                            ctx.as_ref(),
                            lhs.clone(),
                            rhs.clone(),
                            config,
                            &mut out,
                        )
                    })?
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    return Err(unavailable_cpu_backend_kind(self.kind(), "grouped_gemm"));
                }
            }
            CpuBackendKind::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    self.run_with_pool(|_buffers| {
                        gemm::grouped_gemm_blas_cached(lhs.clone(), rhs.clone(), config, &mut out)
                    })?
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    return Err(unavailable_cpu_backend_kind(self.kind(), "grouped_gemm"));
                }
            }
        };
        if direct {
            return Ok(());
        }

        grouped_gemm_via_sequential(self, lhs, rhs, config, out)
    }
}

impl TensorIndexing for CpuBackend {
    fn gather(
        &mut self,
        operand: &Tensor,
        start_indices: &Tensor,
        config: &GatherConfig,
    ) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| {
            indexing::gather_with_pool(buffers, operand, start_indices, config)
        })
    }

    fn scatter(
        &mut self,
        operand: &Tensor,
        scatter_indices: &Tensor,
        updates: &Tensor,
        config: &ScatterConfig,
    ) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| {
            indexing::scatter_with_pool(buffers, operand, scatter_indices, updates, config)
        })
    }

    fn slice(&mut self, input: &Tensor, config: &SliceConfig) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| indexing::try_slice_with_pool(buffers, input, config))
    }

    fn dynamic_slice(
        &mut self,
        input: &Tensor,
        starts: &Tensor,
        slice_sizes: &[usize],
    ) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| {
            indexing::dynamic_slice_with_pool(buffers, input, starts, slice_sizes)
        })
    }

    fn dynamic_update_slice(
        &mut self,
        operand: &Tensor,
        update: &Tensor,
        starts: &Tensor,
    ) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| {
            indexing::dynamic_update_slice_with_pool(buffers, operand, update, starts)
        })
    }

    fn pad(&mut self, input: &Tensor, config: &PadConfig) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| indexing::try_pad_with_pool(buffers, input, config))
    }

    fn concatenate(&mut self, inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| indexing::try_concatenate_with_pool(buffers, inputs, axis))
    }

    fn reverse(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| indexing::reverse_with_pool(buffers, input, axes))
    }
}

impl CpuBackend {
    fn run_backend_session_cached<R: Send>(
        &mut self,
        cache: Option<&mut gemm::GemmAnalysisCache>,
        f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
    ) -> R {
        let ctx = self.engine.context_arc();
        let kind = self.kind();
        let owner = inherited_or_new_execution_owner();
        let permit = self.acquire_execution_permit(owner);
        let run = || {
            self.with_execution_resources(&permit, |resources| {
                let mut buffers = BufferPoolLoan::new(&mut resources.buffers);
                let cache = cache.unwrap_or(&mut resources.gemm_analysis_cache);
                let session_started = Instant::now();
                let mut session = CpuExecSession {
                    ctx: ctx.as_ref(),
                    buffers: buffers.get_mut(),
                    gemm_analysis_cache: cache,
                    kind,
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
        match kind {
            CpuBackendKind::Faer => ctx.install_with_execution_owner(owner, run),
            CpuBackendKind::Blas => with_execution_owner(owner, run),
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
    T: Clone + 'static,
    R: TensorRank,
{
    fn to_contiguous(
        &mut self,
        view: &TypedTensorView<'_, T, R>,
    ) -> crate::Result<TypedTensor<T, R>> {
        if view.backend_buffer().is_some() {
            return Err(crate::Error::backend_failure(
                "CpuBackend::to_contiguous",
                "CPU backend received a backend tensor view; download the tensor to host before CPU view canonicalization",
            ));
        }
        view.to_contiguous()
    }

    fn copy_from_contiguous(
        &mut self,
        src: &TypedTensor<T, R>,
        dst: &mut TypedTensorViewMut<'_, T, R>,
    ) -> crate::Result<()> {
        if matches!(src.buffer(), Buffer::Backend(_)) {
            return Err(crate::Error::backend_failure(
                "CpuBackend::copy_from_contiguous",
                "CPU backend received a backend source tensor; download the tensor to host before CPU view copy-back",
            ));
        }
        if dst.backend_buffer().is_some() {
            return Err(crate::Error::backend_failure(
                "CpuBackend::copy_from_contiguous",
                "CPU backend received a backend destination view; download the tensor to host before CPU view copy-back",
            ));
        }
        dst.copy_from_contiguous(src)
    }
}

impl TensorFusion for CpuBackend {
    fn execute_elementwise_fusion(
        &mut self,
        inputs: &[&Tensor],
        plan: &ElementwiseFusionPlan,
    ) -> crate::Result<Option<Vec<Tensor>>> {
        self.install_with_pool(|buffers| {
            elementwise::elementwise_fusion_with_pool(buffers, inputs, plan)
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
        self.install_with_pool(|buffers| {
            elementwise::broadcast_multiply_value_with_pool(
                buffers, lhs, lhs_shape, lhs_dims, rhs, rhs_shape, rhs_dims,
            )
        })
    }
}

impl TensorDeviceTransfer for CpuBackend {
    fn download_to_host(&mut self, tensor: &Tensor) -> crate::Result<Tensor> {
        if tensor.is_backend_buffer() {
            return Err(crate::Error::backend_failure(
                "CpuBackend::download_to_host",
                "CPU backend received a backend buffer; download the tensor to host with its owning backend before CPU execution",
            ));
        }
        Ok(tensor.clone())
    }

    fn upload_host_tensor(&mut self, tensor: &Tensor) -> crate::Result<Tensor> {
        if tensor.is_backend_buffer() {
            return Err(crate::Error::backend_failure(
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
