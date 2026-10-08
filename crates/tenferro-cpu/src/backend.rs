use num_complex::{Complex32, Complex64};
use std::cmp::Reverse;
use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};
use strided_kernel::ExecContext;
use tenferro_tensor::DType;

use crate::arbiter::{ResourceArbiter, ResourceOwner, ResourcePermit};
use crate::buffer_pool::{BufferPool, BufferPoolStats, PoolScalar};
use crate::engine::{CpuEngine, EngineResources};
use crate::indexed_plan_cache::{IndexedPlanCacheLimits, DEFAULT_INDEXED_PLAN_CACHE_LIMITS};
use crate::placement::{resolve_placement, CpuEngineConstructionError, ResolvedCpuExecution};
use crate::provider::{CpuOperationEntry, ParallelMode};
use crate::{
    discover_cpu_topology, CpuDomainId, CpuId, CpuPlacement, CpuPlacementError, CpuSet,
    CpuTopology, CpuTopologyError, NumaNodeId, ResolvedCpuPlacement,
};
use crate::{CacheStats, Tensor, TensorRank, TensorRead, TensorScalar, TensorWrite, TypedTensor};
use tenferro_tensor::{
    AllocationDomainId, BackendRuntimeCache, BackendSession, BackendSessionHost, ElementwiseReadOp,
    TensorBackend, TensorDeviceTransfer,
};
use tenferro_tensor::{SessionEntryError, SharedTensorAllocationDomain};

use super::exec_session::CpuExecSession;
use super::{copy_tensor_read_into, elementwise, gemm, CpuContext};

fn lock_contraction_workspaces(
    engines: &[Arc<CpuEngine>],
) -> crate::Result<Vec<crate::contraction::WorkspaceLease<'_>>> {
    let mut contexts: Vec<&CpuContext> = Vec::new();
    for context in engines.iter().map(|engine| engine.context.as_ref()) {
        if !contexts
            .iter()
            .any(|&existing| std::ptr::eq(existing, context))
        {
            contexts.push(context);
        }
    }
    contexts
        .into_iter()
        .map(|context| context.contraction_workspaces().lock())
        .collect()
}

pub(crate) fn tag_fresh_output(output: &mut Tensor, domain: CpuDomainId) {
    match output.dtype() {
        DType::F32 => tag_fresh_typed::<f32>(output, domain),
        DType::F64 => tag_fresh_typed::<f64>(output, domain),
        DType::I32 => tag_fresh_typed::<i32>(output, domain),
        DType::I64 => tag_fresh_typed::<i64>(output, domain),
        DType::Bool => tag_fresh_typed::<bool>(output, domain),
        DType::C32 => tag_fresh_typed::<Complex32>(output, domain),
        DType::C64 => tag_fresh_typed::<Complex64>(output, domain),
        // A caller-owned payload has no pooled storage to tag, and a tag the accessor
        // cannot recover leaves the placement untouched, which is the same outcome.
        DType::External(_) => {}
    }
}

/// Mark a freshly allocated output with the CPU domain its pool belongs to.
fn tag_fresh_typed<T: TensorScalar>(output: &mut Tensor, domain: CpuDomainId) {
    if let Some(tensor) = output.as_typed_mut::<T>() {
        tensor.set_cpu_affinity(Some(domain));
    }
}

pub(crate) fn elementwise_read_into_fallback_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    op: ElementwiseReadOp,
    inputs: &[TensorRead<'_>],
    out: TensorWrite<'_>,
) -> crate::Result<()> {
    let result = match op {
        ElementwiseReadOp::Add => {
            elementwise::add_read_with_pool(buffers, ctx, inputs[0].clone(), inputs[1].clone())?
        }
        ElementwiseReadOp::Subtract => {
            elementwise::sub_read_with_pool(buffers, ctx, inputs[0].clone(), inputs[1].clone())?
        }
        ElementwiseReadOp::Multiply => {
            elementwise::mul_read_with_pool(buffers, ctx, inputs[0].clone(), inputs[1].clone())?
        }
        ElementwiseReadOp::Negate => {
            elementwise::neg_read_with_pool(buffers, ctx, inputs[0].clone())?
        }
        ElementwiseReadOp::Conj => {
            elementwise::conj_read_with_pool(buffers, ctx, inputs[0].clone())?
        }
        ElementwiseReadOp::Divide => {
            elementwise::div_read_with_pool(buffers, ctx, inputs[0].clone(), inputs[1].clone())?
        }
        _ => {
            return Err(crate::Error::unsupported(
                "CpuBackend::elementwise_read_into",
                format!("CPU backend does not implement {op:?}"),
            ))
        }
    };
    let copied = copy_tensor_read_into(
        "CpuBackend::elementwise_read_into",
        TensorRead::from_tensor(&result),
        out,
    );
    // The staged result is scratch: return it to the pool for the next op.
    reclaim_tensor(buffers, result);
    copied
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

/// Backend name reported by CPU session-entry failures.
pub(crate) const CPU_BACKEND: &str = "CpuBackend";

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
            CpuBackendError::Placement { op, source } => match source {
                CpuPlacementError::TopologyDiscovery { .. }
                | CpuPlacementError::ManagedAffinityUnavailable { .. }
                | CpuPlacementError::NumaDiscoveryUnavailable { .. }
                | CpuPlacementError::UnknownNumaNode { .. } => {
                    Self::runtime_state_source(op, source)
                }
                CpuPlacementError::EngineConstruction { .. } => Self::backend_source(op, source),
                CpuPlacementError::InternalState { .. } => {
                    Self::extension(op, "cpu", crate::ErrorKind::Internal, source)
                }
            },
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

struct ManagedEngineRegistry {
    node_engines: Mutex<BTreeMap<NumaNodeId, Arc<CpuEngine>>>,
    node_domain_ids: BTreeMap<NumaNodeId, CpuDomainId>,
    all_allowed: OnceLock<Arc<CpuEngine>>,
    all_allowed_build: Mutex<()>,
    base_engine: Arc<CpuEngine>,
    thread_budget: usize,
}

struct CpuBackendState {
    topology: CpuTopology,
    engines: ManagedEngineRegistry,
    arbiter: ResourceArbiter,
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
                message: "CPU indexed-plan cache configuration lock is poisoned",
            }
        })?;
        let cache_limits = *cache_configuration;
        let registry = &self.engines;
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
                        message: "managed NUMA node has no coordinator-stable domain ID",
                    });
                };
                let engine = Arc::new(
                    CpuEngine::new(
                        domain_id,
                        placement.clone(),
                        registry.thread_budget,
                        self.buffer_limit.load(Ordering::Relaxed),
                    )
                    .map_err(|error| {
                        CpuPlacementError::EngineConstruction {
                            requested,
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
                    CpuEngine::new(
                        CpuDomainId::new(0),
                        placement.clone(),
                        registry.thread_budget,
                        self.buffer_limit.load(Ordering::Relaxed),
                    )
                    .map_err(|error| {
                        CpuPlacementError::EngineConstruction {
                            requested,
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
                    message: "new CPU engine indexed-plan cache lock is poisoned",
                })?;
        resources.indexed_plan_cache.set_limits(limits);
        Ok(())
    }

    fn initialized_engines(&self, op: &'static str) -> crate::Result<Vec<Arc<CpuEngine>>> {
        let registry = &self.engines;
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
/// assert_eq!(backend.num_threads(), clone.num_threads());
/// ```
#[derive(Clone)]
pub struct CpuBackend {
    runtime_identity: CpuRuntimeIdentity,
    shared: Arc<CpuBackendState>,
    requested: CpuPlacement,
    resolved: ResolvedCpuExecution,
    pub(crate) engine: Arc<CpuEngine>,
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
    topology: Result<CpuTopology, CpuTopologyError>,
) -> Result<CpuTopology, CpuPlacementError> {
    topology.map_err(|source| CpuPlacementError::TopologyDiscovery {
        requested: CpuPlacement::Auto,
        source,
    })
}

// The error type carries a `DType`, which grew when the tag gained an
// externally defined variant; boxing it per call would cost more than it saves.

fn engine_cpus(placement: &ResolvedCpuPlacement) -> CpuSet {
    placement.cpus().clone()
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

/// Builder for a tenferro-owned CPU backend with an explicit placement.
///
/// Every worker, pool and buffer the backend owns is built by tenferro; the
/// builder never borrows an application-owned executor.
#[derive(Clone, Debug, Default)]
pub struct CpuBackendBuilder {
    threads: Option<usize>,
    cpus: Option<CpuSet>,
    numa_node: Option<NumaNodeId>,
    worker_stack: Option<usize>,
    buffer_limit: Option<usize>,
}

impl CpuBackendBuilder {
    /// Set the worker count of the tenferro-owned pool.
    ///
    /// # Errors
    ///
    /// Returns [`CpuBackendError::Tensor`] with `ValidationError::InvalidArgument`
    /// for a zero worker count.
    #[allow(clippy::result_large_err)]
    pub fn threads(mut self, threads: usize) -> Result<Self, CpuBackendError> {
        if threads == 0 {
            return Err(CpuBackendError::Tensor(crate::Error::invalid_argument(
                "CpuBackendBuilder::threads",
                "configuration",
                "thread count must be at least 1",
            )));
        }
        self.threads = Some(threads);
        Ok(self)
    }

    /// Confine tenferro-owned workers to this CPU set.
    pub fn cpus(mut self, cpus: CpuSet) -> Self {
        self.cpus = Some(cpus);
        self
    }

    /// Use every CPU of one OS NUMA node.
    pub fn numa_node(mut self, id: NumaNodeId) -> Self {
        self.numa_node = Some(id);
        self
    }

    /// Set the stack size reserved for tenferro-owned workers.
    ///
    /// # Errors
    ///
    /// Returns [`CpuBackendError::Tensor`] with `ValidationError::InvalidArgument`
    /// when `bytes` is below the minimum a provider call needs.
    #[allow(clippy::result_large_err)]
    pub fn worker_stack(mut self, bytes: usize) -> Result<Self, CpuBackendError> {
        if bytes < crate::context::MIN_WORKER_STACK_BYTES {
            return Err(CpuBackendError::Tensor(crate::Error::invalid_argument(
                "CpuBackendBuilder::worker_stack",
                "configuration",
                "worker stack size must be at least 65536 bytes",
            )));
        }
        self.worker_stack = Some(bytes);
        Ok(self)
    }

    /// Set the retained-buffer ceiling of this backend's pool.
    pub fn buffer_limit(mut self, bytes: usize) -> Self {
        self.buffer_limit = Some(bytes);
        self
    }

    /// Build the backend.
    ///
    /// # Errors
    ///
    /// Returns [`CpuBackendError::Placement`] when the placement cannot be
    /// resolved on this process topology, or [`CpuBackendError::Tensor`] when
    /// the worker pool cannot be constructed.
    #[allow(clippy::result_large_err)]
    pub fn build(self) -> Result<CpuBackend, CpuBackendError> {
        let op = "CpuBackendBuilder::build";
        let placement_error = |source| CpuBackendError::Placement { op, source };
        let topology = discover_cpu_topology().map_err(|source| {
            placement_error(CpuPlacementError::TopologyDiscovery {
                requested: CpuPlacement::Auto,
                source,
            })
        })?;
        let constrained = self.cpus.is_some() || self.numa_node.is_some();
        let cpus = match (self.cpus, self.numa_node) {
            (Some(cpus), None) => cpus,
            (None, Some(id)) => topology
                .nodes()
                .iter()
                .find(|node| node.id() == id)
                .map(|node| node.cpus().clone())
                .ok_or_else(|| {
                    placement_error(CpuPlacementError::UnknownNumaNode {
                        requested: CpuPlacement::NumaNode(id),
                        node: id,
                    })
                })?,
            (None, None) => topology.allowed_cpus().clone(),
            (Some(_), Some(id)) => {
                return Err(placement_error(CpuPlacementError::UnknownNumaNode {
                    requested: CpuPlacement::NumaNode(id),
                    node: id,
                }))
            }
        };
        let threads = self
            .threads
            .unwrap_or_else(crate::available_parallelism)
            .min(cpus.len());
        let thread_budget = NonZeroUsize::new(threads).ok_or_else(|| {
            CpuBackendError::Tensor(crate::Error::invalid_argument(
                op,
                "configuration",
                "the placement contains no usable CPU",
            ))
        })?;
        let context = match self.worker_stack {
            Some(bytes) => CpuContext::with_pinned_cpus_and_worker_stack(
                cpus.clone(),
                thread_budget.get(),
                bytes,
                crate::affinity::SystemThreadAffinity,
            ),
            None => CpuContext::with_pinned_cpus(cpus.clone(), thread_budget.get()),
        }
        .map_err(|source| {
            placement_error(CpuPlacementError::EngineConstruction {
                requested: CpuPlacement::Auto,
                source: CpuEngineConstructionError::Context(source),
            })
        })?;
        let buffer_limit = self
            .buffer_limit
            .unwrap_or(crate::buffer_pool::DEFAULT_MAX_RETAINED_CAPACITY_BYTES);
        // An explicitly requested CPU set or NUMA node stays a constrained
        // caller-affinity target even though the resolved placement is the
        // all-allowed set.
        let placement = ResolvedCpuPlacement::AllAllowed { cpus };
        let caller_cpus = constrained.then(|| engine_cpus(&placement));
        let engine = Arc::new(CpuEngine::from_context(
            CpuDomainId::new(0),
            placement,
            Arc::new(context),
            thread_budget,
            buffer_limit,
            caller_cpus,
        ));
        Ok(CpuBackend::from_engine(
            engine,
            topology,
            ResourceArbiter::global(),
            buffer_limit,
        ))
    }
}

impl fmt::Debug for CpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuBackend")
            .field("engine_placement", &self.engine.placement())
            .field("num_threads", &self.num_threads())
            .field("allocation_domain", &self.allocation_domain())
            .field("buffer_pool_cache_stats", &self.buffer_pool_cache_stats())
            .field("buffer_pool_limit_bytes", &self.buffer_pool_limit_bytes())
            .finish_non_exhaustive()
    }
}

impl CpuBackend {
    fn from_thread_budget(
        thread_budget: usize,
        max_retained_capacity_bytes: usize,
    ) -> Result<Self, CpuPlacementError> {
        Self::from_thread_budget_kind_and_arbiter(
            thread_budget,
            max_retained_capacity_bytes,
            ResourceArbiter::global(),
        )
    }

    /// A backend whose CPU admission arbiter is private to it, so a parallel
    /// test run's other backends cannot contend with it.
    ///
    /// This exists so tests can observe admission without the process-global
    /// arbiter's unrelated holders.
    #[doc(hidden)]
    pub fn with_threads_isolated_arbiter_for_test(num_threads: usize) -> Self {
        Self::from_thread_budget_kind_and_arbiter(
            num_threads,
            crate::buffer_pool::DEFAULT_MAX_RETAINED_CAPACITY_BYTES,
            ResourceArbiter::new(),
        )
        .expect("test-only CPU backend construction must succeed")
    }

    fn from_thread_budget_kind_and_arbiter(
        thread_budget: usize,
        max_retained_capacity_bytes: usize,
        arbiter: ResourceArbiter,
    ) -> Result<Self, CpuPlacementError> {
        let topology = resolve_discovered_topology(discover_cpu_topology())?;
        let resolved = resolve_placement(CpuPlacement::Auto, &topology)?;
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        {
            let context = CpuContext::with_threads(thread_budget).map_err(|error| {
                CpuPlacementError::EngineConstruction {
                    requested: CpuPlacement::Auto,
                    source: CpuEngineConstructionError::Tensor(error),
                }
            })?;
            Ok(Self::compatibility_with_topology(
                Arc::new(context),
                max_retained_capacity_bytes,
                topology,
                resolved,
                arbiter,
            ))
        }
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            let engine_placement = ResolvedCpuPlacement::AllAllowed {
                cpus: topology.allowed_cpus().clone(),
            };
            let engine = Arc::new(
                CpuEngine::new(
                    CpuDomainId::new(0),
                    engine_placement,
                    thread_budget,
                    max_retained_capacity_bytes,
                )
                .map_err(|error| CpuPlacementError::EngineConstruction {
                    requested: CpuPlacement::Auto,
                    source: CpuEngineConstructionError::Context(error),
                })?,
            );
            let all_allowed = OnceLock::new();
            let _ = all_allowed.set(Arc::clone(&engine));
            Ok(Self {
                shared: Arc::new(CpuBackendState {
                    engines: ManagedEngineRegistry {
                        node_engines: Mutex::new(BTreeMap::new()),
                        node_domain_ids: coordinator_node_domain_ids(&topology),
                        all_allowed,
                        all_allowed_build: Mutex::new(()),
                        base_engine: Arc::clone(&engine),
                        thread_budget,
                    },
                    topology,
                    arbiter,
                    buffer_limit: AtomicUsize::new(max_retained_capacity_bytes),
                    indexed_plan_cache_limits: Mutex::new(DEFAULT_INDEXED_PLAN_CACHE_LIMITS),
                }),
                runtime_identity: CpuRuntimeIdentity::fresh(),
                requested: CpuPlacement::Auto,
                resolved,
                engine,
                allocation_domain: None,
            })
        }
    }

    fn from_engine(
        engine: Arc<CpuEngine>,
        topology: CpuTopology,
        arbiter: ResourceArbiter,
        buffer_limit: usize,
    ) -> Self {
        let all_allowed = OnceLock::new();
        let _ = all_allowed.set(Arc::clone(&engine));
        Self {
            shared: Arc::new(CpuBackendState {
                engines: ManagedEngineRegistry {
                    node_engines: Mutex::new(BTreeMap::new()),
                    node_domain_ids: coordinator_node_domain_ids(&topology),
                    all_allowed,
                    all_allowed_build: Mutex::new(()),
                    base_engine: Arc::clone(&engine),
                    thread_budget: engine.domain().thread_budget().get(),
                },
                topology,
                arbiter,
                buffer_limit: AtomicUsize::new(buffer_limit),
                indexed_plan_cache_limits: Mutex::new(DEFAULT_INDEXED_PLAN_CACHE_LIMITS),
            }),
            runtime_identity: CpuRuntimeIdentity::fresh(),
            requested: CpuPlacement::Auto,
            resolved: ResolvedCpuExecution::Managed(engine.placement().clone()),
            engine,
            allocation_domain: None,
        }
    }

    fn compatibility(ctx: Arc<CpuContext>, max_retained_capacity_bytes: usize) -> Self {
        let topology = discover_cpu_topology().unwrap_or_else(|_| {
            let allowed = crate::process_cpu_affinity().unwrap_or_else(|| {
                CpuSet::new((0..crate::available_parallelism()).map(CpuId::new))
                    .unwrap_or_else(|_| CpuSet::singleton(CpuId::new(0)))
            });
            CpuTopology::all_allowed(allowed)
        });
        let resolved = ResolvedCpuExecution::Compatibility;
        Self::compatibility_with_topology(
            ctx,
            max_retained_capacity_bytes,
            topology,
            resolved,
            ResourceArbiter::global(),
        )
    }

    fn compatibility_with_topology(
        ctx: Arc<CpuContext>,
        max_retained_capacity_bytes: usize,
        topology: CpuTopology,
        resolved: ResolvedCpuExecution,
        arbiter: ResourceArbiter,
    ) -> Self {
        let placement = ResolvedCpuPlacement::AllAllowed {
            cpus: topology.allowed_cpus().clone(),
        };
        let thread_budget = NonZeroUsize::new(ctx.num_threads())
            .expect("CpuContext always has at least one thread");
        let base_engine = Arc::new(CpuEngine::from_context(
            CpuDomainId::new(0),
            placement,
            ctx,
            thread_budget,
            max_retained_capacity_bytes,
            None,
        ));
        Self {
            shared: Arc::new(CpuBackendState {
                engines: ManagedEngineRegistry {
                    node_engines: Mutex::new(BTreeMap::new()),
                    node_domain_ids: coordinator_node_domain_ids(&topology),
                    all_allowed: OnceLock::new(),
                    all_allowed_build: Mutex::new(()),
                    base_engine: Arc::clone(&base_engine),
                    thread_budget: base_engine.domain().thread_budget().get(),
                },
                topology,
                arbiter,
                buffer_limit: AtomicUsize::new(max_retained_capacity_bytes),
                indexed_plan_cache_limits: Mutex::new(DEFAULT_INDEXED_PLAN_CACHE_LIMITS),
            }),
            runtime_identity: CpuRuntimeIdentity::fresh(),
            requested: CpuPlacement::Auto,
            resolved,
            engine: base_engine,
            allocation_domain: None,
        }
    }

    /// Start building a CPU backend with an explicit placement.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{discover_cpu_topology, CpuBackend};
    ///
    /// let cpus = discover_cpu_topology()?.allowed_cpus().clone();
    /// let backend = CpuBackend::builder()
    ///     .cpus(cpus)
    ///     .threads(1)?
    ///     .build()?;
    /// assert_eq!(backend.num_threads(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    pub fn builder() -> CpuBackendBuilder {
        CpuBackendBuilder::default()
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
    #[allow(clippy::result_large_err)]
    pub fn new() -> Self {
        let context = Arc::new(CpuContext::from_env());
        Self::from_thread_budget(
            context.num_threads(),
            crate::buffer_pool::DEFAULT_MAX_RETAINED_CAPACITY_BYTES,
        )
        .unwrap_or_else(|error| {
            eprintln!(
                "tenferro_cpu: using the unpinned compatibility context after placement error: {error}"
            );
            Self::from_context(context)
        })
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
    #[doc(hidden)]
    pub fn from_context(ctx: Arc<CpuContext>) -> Self {
        Self::compatibility(ctx, crate::buffer_pool::DEFAULT_MAX_RETAINED_CAPACITY_BYTES)
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
    #[doc(hidden)]
    pub fn from_context_with_buffer_pool_limit(
        ctx: Arc<CpuContext>,
        max_retained_capacity_bytes: usize,
    ) -> Self {
        Self::compatibility(ctx, max_retained_capacity_bytes)
    }

    // The error type carries a `DType`, which grew when the tag gained an
    // externally defined variant; boxing it per call would cost more than it saves.
    #[allow(clippy::result_large_err)]
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
    /// Returns [`CpuBackendError::Tensor`] when `RAYON_NUM_THREADS` is zero or
    /// malformed, and [`CpuBackendError::Placement`] when CPU topology or
    /// managed placement initialization is unavailable.
    #[allow(clippy::result_large_err)]
    pub fn try_new() -> Result<Self, CpuBackendError> {
        let op = "CpuBackend::try_new";
        let context =
            CpuContext::try_from_env().map_err(|error| constructor_tensor_error(op, error))?;
        Self::from_thread_budget(
            context.num_threads(),
            crate::buffer_pool::DEFAULT_MAX_RETAINED_CAPACITY_BYTES,
        )
        .map_err(|error| CpuBackendError::placement(op, error))
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
    #[allow(clippy::result_large_err)]
    pub fn with_threads(num_threads: usize) -> Result<Self, CpuBackendError> {
        let op = "CpuBackend::with_threads";
        let context = CpuContext::with_threads(num_threads)
            .map_err(|error| constructor_tensor_error(op, error))?;
        Self::from_thread_budget(
            context.num_threads(),
            crate::buffer_pool::DEFAULT_MAX_RETAINED_CAPACITY_BYTES,
        )
        .map_err(|error| CpuBackendError::placement(op, error))
    }

    // The error type carries a `DType`, which grew when the tag gained an
    // externally defined variant; boxing it per call would cost more than it saves.
    #[allow(clippy::result_large_err)]
    /// Create a CPU backend with a custom thread count.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::with_threads(1)?;
    /// assert_eq!(backend.num_threads(), 1);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Return this coordinator's resolved CPU topology.
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
        resolve_placement(placement, &self.shared.topology).is_ok()
    }

    /// Return the stable identity of this backend's selected CPU domain.
    #[doc(hidden)]
    pub fn domain_id(&self) -> CpuDomainId {
        self.engine.domain().id()
    }

    #[cfg(all(test, feature = "blas"))]
    fn try_acquire_execution_permit_for_test(
        &self,
    ) -> Result<Option<ResourcePermit>, crate::arbiter::ResourceArbiterError> {
        match &self.resolved {
            ResolvedCpuExecution::Managed(placement) => {
                self.shared.arbiter.try_acquire(placement.cpus().clone())
            }
            ResolvedCpuExecution::Compatibility => self
                .shared
                .arbiter
                .try_acquire(self.shared.topology.allowed_cpus().clone()),
        }
    }

    #[cfg(test)]
    pub(crate) fn domain_id_for_test(&self) -> CpuDomainId {
        self.engine.domain().id()
    }

    #[cfg(test)]
    pub(crate) fn context_id_for_test(&self) -> usize {
        Arc::as_ptr(self.engine.domain().context()) as *const () as usize
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

    /// Create a backend for one explicit CPU placement on this backend's topology.
    ///
    /// The returned backend shares this backend's topology, arbiter and buffer
    /// policy and owns an engine for the resolved placement.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuBackend, CpuPlacement};
    ///
    /// let backend = CpuBackend::new();
    /// assert_eq!(backend.placement(), CpuPlacement::Auto);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CpuPlacementError`] when the requested placement cannot be
    /// resolved on this process topology or its engine cannot be constructed.
    pub fn for_placement(&self, requested: CpuPlacement) -> Result<Self, CpuPlacementError> {
        let resolved = resolve_placement(requested, &self.shared.topology)?;
        let placement = match &resolved {
            ResolvedCpuExecution::Managed(placement) => placement.clone(),
            _ => ResolvedCpuPlacement::AllAllowed {
                cpus: self.shared.topology.allowed_cpus().clone(),
            },
        };
        let engine = self.shared.managed_engine_for(&placement, requested)?;
        Ok(Self {
            runtime_identity: CpuRuntimeIdentity::fresh(),
            shared: Arc::clone(&self.shared),
            requested,
            resolved,
            engine,
            allocation_domain: self.allocation_domain.clone(),
        })
    }

    /// Return the CPU placement requested for this backend.
    pub fn placement(&self) -> CpuPlacement {
        self.requested
    }

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
    /// lock is poisoned. Also returns a typed backend-source error without
    /// changing the limit when an owned N-ary workspace is borrowed or poisoned.
    /// The configured retention ceiling also applies to contraction workspaces.
    pub fn set_buffer_pool_limit_bytes(
        &mut self,
        max_retained_capacity_bytes: usize,
    ) -> crate::Result<()> {
        // INVARIANT: the configuration guard serializes this mutation with lazy
        // engine creation, so every engine either exists in the snapshot below
        // or reads the new limit when it is constructed.
        let _configuration = self.shared.indexed_plan_cache_limits.lock().map_err(|_| {
            poisoned_cpu_lock(
                "CpuBackend::set_buffer_pool_limit_bytes",
                "CPU engine configuration",
            )
        })?;
        let engines = self
            .shared
            .initialized_engines("CpuBackend::set_buffer_pool_limit_bytes")?;
        let mut resources = engines
            .iter()
            .map(|engine| lock_engine_resources(engine, "CpuBackend::set_buffer_pool_limit_bytes"))
            .collect::<crate::Result<Vec<_>>>()?;
        let mut workspaces = lock_contraction_workspaces(&engines)?;
        self.shared
            .buffer_limit
            .store(max_retained_capacity_bytes, Ordering::Relaxed);
        for resource in &mut resources {
            resource
                .buffers
                .set_max_retained_capacity_bytes(max_retained_capacity_bytes);
        }
        for workspace in &mut workspaces {
            workspace.trim(max_retained_capacity_bytes);
        }
        for context in engines.iter().map(|engine| engine.context.as_ref()) {
            if context.contraction_retained_bytes() > max_retained_capacity_bytes {
                context.trim_contraction_workspace();
            }
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
    /// poisoned. An owned N-ary workspace that is borrowed or poisoned is also
    /// reported before any store is cleared. Idle contraction workspace storage
    /// is released alongside typed host buffers.
    pub fn reset_buffer_pool(&mut self) -> crate::Result<()> {
        let _configuration = self.shared.indexed_plan_cache_limits.lock().map_err(|_| {
            poisoned_cpu_lock("CpuBackend::reset_buffer_pool", "CPU engine configuration")
        })?;
        let engines = self
            .shared
            .initialized_engines("CpuBackend::reset_buffer_pool")?;
        let mut resources = engines
            .iter()
            .map(|engine| lock_engine_resources(engine, "CpuBackend::reset_buffer_pool"))
            .collect::<crate::Result<Vec<_>>>()?;
        let mut workspaces = lock_contraction_workspaces(&engines)?;
        for resource in &mut resources {
            resource.buffers.clear();
        }
        for workspace in &mut workspaces {
            workspace.clear();
        }
        for context in engines.iter().map(|engine| engine.context.as_ref()) {
            context.trim_contraction_workspace();
        }
        Ok(())
    }

    pub(crate) fn runtime_cache_stats(
        &self,
    ) -> crate::Result<tenferro_runtime::runtime::CacheStats> {
        let resources = lock_engine_resources(&self.engine, "CpuBackend::runtime_cache_stats")?;
        let workspaces = lock_contraction_workspaces(std::slice::from_ref(&self.engine))?;
        let (workspace_entries, nary_bytes) =
            workspaces
                .iter()
                .fold((0usize, 0usize), |(count, bytes), workspace| {
                    let stats = workspace.stats();
                    (count.saturating_add(stats.0), bytes.saturating_add(stats.1))
                });
        let workspace_bytes =
            nary_bytes.saturating_add(self.engine.context.contraction_retained_bytes());
        let buffers = resources.buffers.cache_stats();
        let gemm = tenferro_tensor::RuntimeCacheControl::stats(&resources.gemm_analysis_cache);
        let indexed = resources.indexed_plan_cache.stats();
        Ok(tenferro_runtime::runtime::CacheStats {
            entries: buffers
                .entries
                .saturating_add(gemm.entries)
                .saturating_add(indexed.entries)
                .saturating_add(workspace_entries),
            retained_bytes: buffers
                .retained_bytes
                .saturating_add(gemm.retained_bytes)
                .saturating_add(indexed.retained_bytes)
                .saturating_add(workspace_bytes),
            hits: indexed.hits.saturating_add(gemm.hits),
            misses: indexed.misses.saturating_add(gemm.misses),
            evictions: indexed.evictions.saturating_add(gemm.evictions),
            clears: resources.runtime_clears,
        })
    }

    pub(crate) fn clear_runtime_caches(&self) -> crate::Result<()> {
        let mut resources =
            lock_engine_resources(&self.engine, "CpuBackend::clear_runtime_caches")?;
        let mut workspaces = lock_contraction_workspaces(std::slice::from_ref(&self.engine))?;
        resources.buffers.clear();
        for workspace in &mut workspaces {
            workspace.clear();
        }
        self.engine.context.trim_contraction_workspace();
        tenferro_tensor::RuntimeCacheControl::clear(&mut resources.gemm_analysis_cache);
        resources.indexed_plan_cache.clear();
        resources.runtime_clears = resources.runtime_clears.saturating_add(1);
        Ok(())
    }

    /// Run a closure in this backend's CPU execution scope.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::with_threads(1)?;
    /// let value = backend.install(|| 1 + 1)?;
    /// assert_eq!(value, 2);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::SessionEntry`] without running `op` when another
    /// CPU backend execution is already active on the current thread or managed
    /// Rayon scope ([`SessionEntryError::Reentered`]; direct nesting and backend
    /// calls from parallel child tasks could violate CPU or provider
    /// exclusivity), when a caller-managed domain is already executing, or when
    /// admission state is poisoned. Returns [`crate::Error::BackendSource`] with
    /// the executor's typed diagnostic when an externally managed executor
    /// cannot be entered.
    pub fn install<R: Send>(&self, op: impl FnOnce() -> R + Send) -> crate::Result<R> {
        let admission = self.execution_admission()?;
        let permit = admission.permit();
        let entry = CpuOperationEntry::new(self.engine.domain(), permit);
        Ok(entry.enter(ParallelMode::Sequential, |_| op()))
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
        // INVARIANT: this lock is poisoned only by a session callback that
        // unwound while holding it, and `BufferPoolLoan` restores the pool's
        // in-flight accounting on unwind, so the resources are consistent and
        // the next session may reuse them. Pool introspection
        // (`buffer_pool_len`, `buffer_pool_stats`) still reports the poison.
        let mut resources = self
            .engine
            .resources
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        op(&mut resources)
    }

    fn acquire_execution_permit(
        &self,
        owner: ResourceOwner,
    ) -> Result<ResourcePermit, SessionEntryError> {
        let arbiter_poisoned = |error| match error {
            crate::arbiter::ResourceArbiterError::Contended => SessionEntryError::Contended {
                backend: CPU_BACKEND,
                message: "a Rayon worker cannot wait for conflicting CPU resources".to_owned(),
            },
            _ => SessionEntryError::ResourcePoisoned {
                backend: CPU_BACKEND,
                resource: "the CPU resource arbiter",
            },
        };
        match &self.resolved {
            ResolvedCpuExecution::Managed(placement) => self
                .shared
                .arbiter
                .acquire_waiting(placement.cpus().clone(), owner)
                .map_err(arbiter_poisoned),
            _ => self
                .shared
                .arbiter
                .acquire_waiting(self.shared.topology.allowed_cpus().clone(), owner)
                .map_err(arbiter_poisoned),
        }
    }
}

impl BackendRuntimeCache for CpuBackend {
    type RuntimeCache = gemm::GemmAnalysisCache;
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

    fn run_backend_session_cached<R>(
        &mut self,
        cache: Option<&mut gemm::GemmAnalysisCache>,
        f: impl FnOnce(&mut dyn BackendSession) -> R,
    ) -> Result<R, SessionEntryError> {
        let admission = self.execution_admission()?;
        let affinity_error = |source| SessionEntryError::Executor {
            backend: CPU_BACKEND,
            source: Box::new(source),
        };
        let affinity =
            crate::affinity::CallerAffinityGuard::enter(self.engine.domain().caller_cpus())
                .map_err(affinity_error)?;
        let permit = admission.permit();
        let owner = permit.owner();
        let entry = CpuOperationEntry::new(self.engine.domain(), permit);
        // Provider-owned BLAS threading does not change session entry: the
        // permit, including provider exclusion, spans this entire callback.
        let run = |entered| {
            self.with_execution_resources(permit, |resources| {
                let mut buffers = BufferPoolLoan::new(&mut resources.buffers);
                let cache = cache.unwrap_or(&mut resources.gemm_analysis_cache);
                let session_started = Instant::now();
                let mut session = CpuExecSession {
                    entry,
                    context: Some(self.engine.context.as_ref()),
                    entered,
                    buffers: buffers.get_mut(),
                    gemm_analysis_cache: cache,
                    indexed_plan_cache: &mut resources.indexed_plan_cache,
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
        let _ = owner;
        let result = entry.enter_managed_session(|context| run(Some(context)));
        affinity.finish().map_err(affinity_error)?;
        result
    }
}

impl BackendSessionHost for CpuBackend {
    fn with_backend_session<R>(
        &mut self,
        f: impl FnOnce(&mut dyn BackendSession) -> R,
    ) -> Result<R, SessionEntryError> {
        self.run_backend_session_cached(None, f)
    }

    fn with_backend_session_cached<R>(
        &mut self,
        cache: &mut Self::RuntimeCache,
        f: impl FnOnce(&mut dyn BackendSession) -> R,
    ) -> Result<R, SessionEntryError> {
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

/// Hand a tensor back to the pool's typed free list, keyed by its runtime tag.
///
/// A caller-owned payload owns no pooled storage, and a tag the conversion cannot recover
/// behaves the same way rather than guessing. Every reclaim entry point shares this one table.
pub(crate) fn reclaim_tensor(buffers: &mut BufferPool, tensor: Tensor) {
    match tensor.dtype() {
        DType::F32 => reclaim_tensor_typed::<f32>(buffers, tensor),
        DType::F64 => reclaim_tensor_typed::<f64>(buffers, tensor),
        DType::I32 => reclaim_tensor_typed::<i32>(buffers, tensor),
        DType::I64 => reclaim_tensor_typed::<i64>(buffers, tensor),
        DType::Bool => reclaim_tensor_typed::<bool>(buffers, tensor),
        DType::C32 => reclaim_tensor_typed::<Complex32>(buffers, tensor),
        DType::C64 => reclaim_tensor_typed::<Complex64>(buffers, tensor),
        DType::External(_) => {}
    }
}

/// Hand the typed tensor back to the pool when the tag table reached the matching tag.
fn reclaim_tensor_typed<T: tenferro_cpu_basic::PoolScalar>(
    buffers: &mut BufferPool,
    tensor: Tensor,
) {
    if let Ok(typed) = tensor.into_typed::<T>() {
        reclaim_typed(buffers, typed);
    }
}

impl TensorDeviceTransfer for CpuBackend {
    fn download_to_host(&mut self, tensor: TensorRead<'_>) -> crate::Result<Tensor> {
        if tensor.backend_family().is_some() {
            return Err(crate::Error::runtime_state(
                "CpuBackend::download_to_host",
                "CPU backend received a backend buffer; download the tensor to host with its owning backend before CPU execution",
            ));
        }
        tensor.tensor_view().duplicate()
    }

    fn upload_host_tensor(&mut self, tensor: TensorRead<'_>) -> crate::Result<Tensor> {
        if tensor.backend_family().is_some() {
            return Err(crate::Error::runtime_state(
                "CpuBackend::upload_host_tensor",
                "CPU backend upload_host_tensor expects a host tensor; download backend buffers to host before CPU execution",
            ));
        }
        tensor.tensor_view().duplicate()
    }
}

impl TensorBackend for CpuBackend {}

pub(crate) fn reclaim_typed<T: PoolScalar>(pool: &mut BufferPool, typed: TypedTensor<T>) {
    if typed.backend_buffer().is_some() {
        return;
    }
    if let Ok(data) = typed.into_host_vec() {
        T::pool_release(pool, data);
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) mod execution_scope;

#[cfg(test)]
mod tests;
