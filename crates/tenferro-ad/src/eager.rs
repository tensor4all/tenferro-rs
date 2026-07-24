use std::cell::{Cell, RefCell};
use std::cmp::Reverse;
use std::collections::HashMap;
use std::env;
use std::fmt;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};
use std::time::{Duration, Instant};

use crate::extension_cache::{ExtensionCacheLimits, ExtensionCacheStore};
use crate::extension_runtime::{ExtensionExecutor, ExtensionRuntimeRegistryError};
#[cfg(test)]
use computegraph::graph::Graph;
use computegraph::ValueKey;
#[cfg(test)]
use computegraph::ValueRef;
use tenferro_cpu::{CpuBackend, CpuBackendError, CpuPlacement};
#[cfg(feature = "cuda")]
use tenferro_gpu::CudaBackend;
#[cfg(feature = "webgpu")]
use tenferro_gpu::WebGpuBackend;
use tenferro_ops::ad::ExtensionAdDispatcher;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::ShapeGuardContext;
use tenferro_runtime::ad_support::ones_tensor;
use tenferro_runtime::{
    CoreCapabilityBundle, EngineId, ErrorPhase, ExecutionContextIdentity, HardwareClassId,
    RegistrationIdentity, Runtime, RuntimeConfigError, RuntimeConfigSnapshot, RuntimeEpoch,
};
use tenferro_tensor::{BackendSession, BackendSessionHost};
use tenferro_tensor::{
    CacheStats, DType, IntoShapeVec, Tensor, TensorBackend, TensorElementwise, TensorRead,
    TensorScalar, TensorValue, TypedTensor,
};
use tidu::eager::{self, EagerInput, EagerOutput, KeySource, RecordedGraph, Recorder, Trace};
use tidu::{ADRuleError, ADRuleKind, LinearizedGraph};

use self::backward::TenferroBackwardCallbacks;
use self::functional::{functional_jvp, functional_vjp_optional};
use crate::eager_backend::{
    cpu_runtime_engine_id, cpu_runtime_engine_registration, cpu_runtime_hardware_class,
    eager_runtime_for_backend, EagerBackend,
};
#[cfg(test)]
use crate::eager_exec::exec_standard_op_on_tensor_reads_in_session;
use crate::eager_exec::{
    exec_op_on_tensor_reads_with_extension_executor, exec_op_on_tensors_with_extension_executor,
};
use crate::error::{ContextId, Error, Result};
#[cfg(test)]
use crate::metadata::push_metadata_scope;
use crate::metadata::{
    metadata_scopes_for_scope, register_scoped_metadata_batch, register_scoped_value_metadata,
    tensor_meta_from_tensor, GlobalMetadataScope,
};
use crate::traced::next_input_key;
use crate::transform_cache::{AdTransformCache, AdTransformCacheLimits, EagerAdTransformCacheKey};

use crate::AdContext;

mod backward;
mod functional;

pub(crate) type GradSlot = Arc<Mutex<Option<Arc<Tensor>>>>;
pub(crate) type WeakGradSlot = Weak<Mutex<Option<Arc<Tensor>>>>;

#[cfg(test)]
pub(crate) static CPU_RUNTIME_SELECTION_REFRESHES: AtomicUsize = AtomicUsize::new(0);

struct CpuRuntimeSelection {
    snapshot: Arc<RuntimeConfigSnapshot>,
    epoch: RuntimeEpoch,
    engine_id: EngineId,
    registration_identity: RegistrationIdentity,
    capabilities: CoreCapabilityBundle,
}

#[derive(Debug, Default, Clone)]
struct EagerOpProfileEntry {
    calls: usize,
    total_time: Duration,
}

thread_local! {
    static EAGER_OP_PROFILE_STATE: RefCell<HashMap<&'static str, EagerOpProfileEntry>> =
        RefCell::new(HashMap::new());
    static EAGER_NO_GRAD_DEPTH: Cell<usize> = const { Cell::new(0) };
    #[cfg(test)]
    static EAGER_OP_PROFILE_ENABLED_OVERRIDE: RefCell<Option<bool>> = const { RefCell::new(None) };
    #[cfg(test)]
    static EAGER_OP_PROFILE_PRINT_EVERY_OVERRIDE: RefCell<Option<Option<usize>>> = const { RefCell::new(None) };
}

pub(crate) fn eager_grad_recording_enabled() -> bool {
    EAGER_NO_GRAD_DEPTH.with(|depth| depth.get() == 0)
}

/// Scope guard that temporarily disables eager operation recording.
///
/// Values computed while this guard is alive are concrete eager tensors, but
/// they do not participate in reverse-mode gradient tracking.
///
/// # Examples
///
/// ```
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
/// use tenferro_cpu::CpuBackend;
///
/// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
/// let x = EagerTensor::requires_grad_in(
///     Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap(),
///     ctx.clone(),
/// )?;
/// let y = {
///     let _guard = ctx.no_grad();
///     x.mul(&x)?
/// };
/// assert!(!y.tracks_grad());
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
#[derive(Debug)]
pub struct EagerNoGradGuard {
    active: bool,
}

impl Drop for EagerNoGradGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        EAGER_NO_GRAD_DEPTH.with(|depth| {
            depth.set(depth.get().saturating_sub(1));
        });
        self.active = false;
    }
}

pub(crate) fn eager_op_profile_enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = EAGER_OP_PROFILE_ENABLED_OVERRIDE.with(|state| *state.borrow()) {
        return value;
    }

    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env::var("TENFERRO_PROFILE_EAGER_OP_AGG").is_ok())
}

pub(crate) fn eager_op_profile_start() -> Option<Instant> {
    eager_op_profile_enabled().then(Instant::now)
}

pub(crate) fn record_eager_op_profile(section: &'static str, elapsed: Duration) {
    if !eager_op_profile_enabled() {
        return;
    }
    EAGER_OP_PROFILE_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let entry = state.entry(section).or_default();
        entry.calls += 1;
        entry.total_time += elapsed;
    });
}

pub(crate) fn profile_eager_op_section<T>(section: &'static str, f: impl FnOnce() -> T) -> T {
    if !eager_op_profile_enabled() {
        return f();
    }
    let started = Instant::now();
    let result = f();
    record_eager_op_profile(section, started.elapsed());
    result
}

pub(crate) fn maybe_print_eager_op_profile() {
    if !eager_op_profile_enabled() {
        return;
    }
    let Some(print_every) = eager_op_profile_print_every() else {
        return;
    };
    if print_every == 0 {
        return;
    }

    let should_print = EAGER_OP_PROFILE_STATE.with(|state| {
        state
            .borrow()
            .get("nary_op.total")
            .is_some_and(|entry| entry.calls % print_every == 0)
    });
    if should_print {
        print_and_reset_eager_op_profile();
    }
}

fn eager_op_profile_print_every() -> Option<usize> {
    #[cfg(test)]
    if let Some(value) = EAGER_OP_PROFILE_PRINT_EVERY_OVERRIDE.with(|state| *state.borrow()) {
        return value;
    }

    env::var("TENFERRO_PROFILE_EAGER_OP_PRINT_EVERY")
        .ok()?
        .parse()
        .ok()
}

pub(crate) fn print_and_reset_eager_op_profile() {
    EAGER_OP_PROFILE_STATE.with(|state| {
        let mut entries: Vec<_> = state
            .borrow()
            .iter()
            .map(|(section, entry)| (*section, entry.clone()))
            .collect();
        state.borrow_mut().clear();
        entries.sort_by_key(|(_, entry)| Reverse(entry.total_time));

        eprintln!("=== tenferro eager op profile ===");
        for (section, entry) in entries {
            let Some(per_call_us) = eager_op_profile_per_call_us(&entry) else {
                continue;
            };
            eprintln!(
                "{section}: calls={} total={:.6}ms per_call={:.3}us",
                entry.calls,
                entry.total_time.as_secs_f64() * 1.0e3,
                per_call_us,
            );
        }
    });
}

fn eager_op_profile_per_call_us(entry: &EagerOpProfileEntry) -> Option<f64> {
    (entry.calls != 0).then(|| entry.total_time.as_secs_f64() * 1.0e6 / entry.calls as f64)
}

fn build_eager_runtime_for_backend(backend: &EagerBackend) -> Runtime {
    match eager_runtime_for_backend(backend) {
        Ok(runtime) => runtime,
        Err(source) => {
            // INVARIANT: eager runtime construction uses fixed, private CPU
            // identifiers and one storage class; failing here means the
            // hard-coded registration contract has been broken.
            panic!("fixed eager runtime configuration failed: {source}");
        }
    }
}

fn runtime_config_error(op: &'static str, source: RuntimeConfigError) -> Error {
    Error::runtime_state_source(op, ErrorPhase::Execution, source)
}

fn runtime_state_source<E>(op: &'static str, source: E) -> Error
where
    E: std::error::Error + Send + Sync + 'static,
{
    Error::runtime_state_source(op, ErrorPhase::Execution, source)
}

fn cpu_runtime_bridge_unsupported(message: impl Into<String>) -> Error {
    Error::unsupported(
        "CpuPlacementBoundEager::refresh_runtime_selection",
        ErrorPhase::Execution,
        message,
    )
}

fn select_cpu_runtime(runtime: &Runtime) -> Result<CpuRuntimeSelection> {
    let snapshot = runtime
        .snapshot()
        .map_err(|source| runtime_state_source("EagerRuntime::runtime_snapshot", source))?;
    let engine_id = cpu_runtime_engine_id()
        .map_err(|source| runtime_config_error("EagerRuntime::cpu_runtime_engine_id", source))?;
    let expected_hardware = cpu_runtime_hardware_class().map_err(|source| {
        runtime_config_error("EagerRuntime::cpu_runtime_hardware_class", source)
    })?;
    let engine = snapshot
        .engine(&engine_id)
        .ok_or_else(|| cpu_runtime_bridge_unsupported("missing CPU runtime engine"))?;
    validate_cpu_runtime_engine(
        engine.context_identity(),
        engine.hardware_class(),
        engine.capabilities(),
        &expected_hardware,
    )?;
    let epoch = snapshot.epoch();
    let registration_identity = engine.registration_identity();
    let capabilities = engine.capabilities().clone();
    Ok(CpuRuntimeSelection {
        snapshot,
        epoch,
        engine_id,
        registration_identity,
        capabilities,
    })
}

fn validate_cpu_runtime_engine(
    context_identity: ExecutionContextIdentity,
    hardware_class: &HardwareClassId,
    capabilities: &CoreCapabilityBundle,
    expected_hardware: &HardwareClassId,
) -> Result<()> {
    if context_identity != ExecutionContextIdentity::of::<CpuBackend>() {
        return Err(cpu_runtime_bridge_unsupported(
            "CPU runtime context mismatch",
        ));
    }
    if hardware_class != expected_hardware {
        return Err(cpu_runtime_bridge_unsupported(
            "CPU runtime hardware mismatch",
        ));
    }
    if capabilities.elementwise().is_none() {
        return Err(cpu_runtime_bridge_unsupported(
            "missing CPU runtime capability: elementwise",
        ));
    }
    if capabilities.reduction().is_none() {
        return Err(cpu_runtime_bridge_unsupported(
            "missing CPU runtime capability: reduction",
        ));
    }
    if capabilities.indexing().is_none() {
        return Err(cpu_runtime_bridge_unsupported(
            "missing CPU runtime capability: indexing",
        ));
    }
    if capabilities.dot_general().is_none() {
        return Err(cpu_runtime_bridge_unsupported(
            "missing CPU runtime capability: dot_general",
        ));
    }
    if capabilities.layout().is_none() {
        return Err(cpu_runtime_bridge_unsupported(
            "missing CPU runtime capability: layout",
        ));
    }
    Ok(())
}

/// Stats for caches owned by an [`EagerRuntime`].
///
/// `retained_bytes` fields are logical payload estimates, not process RSS.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EagerRuntimeCacheStats {
    /// Generic extension runtime caches.
    pub extensions: CacheStats,
    /// Eager AD transform memoization cache.
    pub ad_transforms: CacheStats,
}

#[cfg(test)]
pub(crate) struct EagerGraphExecution {
    pub(crate) outputs: Vec<Arc<Tensor>>,
    pub(crate) retained_values: HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
}

/// Placement-selected CPU view of one [`EagerRuntime`].
///
/// The view snapshots the runtime's CPU coordinator/provider bundle and the
/// immutable runtime registration metadata when [`EagerRuntime::on_cpu`] is
/// called. It holds no resource permit while idle and enters one backend
/// session only while [`Self::with_eager_session`] runs. The session exposes
/// core [`BackendSession`] operations on concrete [`Tensor`] values. This
/// bridge deliberately does not expose the eager runtime's linalg, FFT, einsum,
/// or extension-runtime registries.
///
/// The value is intentionally not `Clone`: mutable use makes concurrent
/// session ownership explicit without adding another backend mutex.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::EagerRuntime;
/// use tenferro_cpu::CpuPlacement;
///
/// let runtime = EagerRuntime::new();
/// let cpu = runtime.on_cpu(CpuPlacement::Auto)?;
/// assert_eq!(cpu.runtime_id(), runtime.id());
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
pub struct CpuPlacementBoundEager {
    runtime: Arc<EagerRuntime>,
    backend: CpuBackend,
    snapshot: Arc<RuntimeConfigSnapshot>,
    epoch: RuntimeEpoch,
    engine_id: EngineId,
    registration_identity: RegistrationIdentity,
    capabilities: CoreCapabilityBundle,
}

impl fmt::Debug for CpuPlacementBoundEager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuPlacementBoundEager")
            .field("runtime_id", &self.runtime.id())
            .field("placement", &self.backend.placement())
            .field("runtime_epoch", &self.epoch)
            .field("engine_id", &self.engine_id)
            .field("registration_identity", &self.registration_identity)
            .finish_non_exhaustive()
    }
}

impl CpuPlacementBoundEager {
    fn refresh_runtime_selection(&mut self) -> Result<()> {
        let current_epoch = self.runtime.runtime.epoch().map_err(|source| {
            runtime_state_source("CpuPlacementBoundEager::refresh_runtime_selection", source)
        })?;
        if current_epoch == self.epoch {
            return Ok(());
        }

        #[cfg(test)]
        CPU_RUNTIME_SELECTION_REFRESHES.fetch_add(1, Ordering::SeqCst);

        let selection = select_cpu_runtime(&self.runtime.runtime)?;
        self.snapshot = selection.snapshot;
        self.epoch = selection.epoch;
        self.engine_id = selection.engine_id;
        self.registration_identity = selection.registration_identity;
        self.capabilities = selection.capabilities;
        Ok(())
    }

    /// Return the identity of the original eager runtime.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::EagerRuntime;
    /// use tenferro_cpu::CpuPlacement;
    ///
    /// let runtime = EagerRuntime::new();
    /// let cpu = runtime.on_cpu(CpuPlacement::Auto)?;
    /// assert_eq!(cpu.runtime_id(), runtime.id());
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    pub fn runtime_id(&self) -> ContextId {
        self.runtime.id()
    }

    /// Return the placement requested when this view was created.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::EagerRuntime;
    /// use tenferro_cpu::CpuPlacement;
    ///
    /// let runtime = EagerRuntime::new();
    /// let cpu = runtime.on_cpu(CpuPlacement::Auto)?;
    /// assert_eq!(cpu.placement(), CpuPlacement::Auto);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    pub fn placement(&self) -> CpuPlacement {
        self.backend.placement()
    }

    /// Enter one CPU backend session and run core operations through it.
    ///
    /// One call creates one backend session. Tenferro-managed CPU executors
    /// enter once around the closure and core operations reuse that compatible
    /// execution scope. The closure may borrow stack data and need not be
    /// `'static`.
    ///
    /// This phase-2 bridge accepts only core [`BackendSession`] operations. It
    /// does not lock or dispatch the eager runtime's linalg, FFT, einsum, or
    /// extension registries.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Error};
    /// use tenferro_cpu::CpuPlacement;
    /// use tenferro_tensor::{Tensor, TensorElementwise};
    ///
    /// let runtime = EagerRuntime::new();
    /// let mut cpu = runtime.on_cpu(CpuPlacement::Auto)?;
    /// let lhs = Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
    /// let rhs = Tensor::from_vec_col_major(vec![1], vec![2.0_f64])?;
    /// let output = cpu.with_eager_session(|session| {
    ///     TensorElementwise::add(session, &lhs, &rhs).map_err(Error::from)
    /// })?;
    /// assert_eq!(output.as_slice::<f64>().unwrap(), &[3.0]);
    /// # Ok::<(), Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns the callback's [`Error`] unchanged. Core backend operations may
    /// report validation, unsupported capability, backend, or runtime-state
    /// failures through that error.
    ///
    /// # Panics
    ///
    /// The existing CPU backend re-entry guard panics if the callback enters a
    /// public `CpuBackend` or calls an ordinary `EagerTensor` operation on this
    /// same runtime. Use only the borrowed `session` for work inside the scope.
    pub fn with_eager_session<R: Send>(
        &mut self,
        f: impl FnOnce(&mut dyn BackendSession) -> Result<R> + Send,
    ) -> Result<R> {
        self.refresh_runtime_selection()?;
        self.backend.with_backend_session(f)
    }
}

/// Shared eager execution context for tensors on a backend.
///
/// Reusing one context lets eager tensors share backend state, extension
/// runtime caches, and gradient storage across a computation.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
///
/// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
/// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(), ctx.clone()).unwrap();
/// let y = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap(), ctx).unwrap();
/// let z = x.add(&y).unwrap();
///
/// assert_eq!(z.materialized().unwrap().as_slice::<f64>().unwrap(), &[3.0]);
/// ```
pub struct EagerRuntime {
    id: ContextId,
    runtime: Runtime,
    pub(crate) backend: Mutex<EagerBackend>,
    pub(crate) extension_executor: Mutex<ExtensionExecutor<EagerBackend>>,
    extension_ad_dispatcher: Option<Arc<dyn ExtensionAdDispatcher>>,
    grad_slots: Mutex<HashMap<ValueKey<StdTensorOp>, WeakGradSlot>>,
    value_records: Mutex<HashMap<ValueKey<StdTensorOp>, Weak<EagerTensorRecord>>>,
    value_ptr_records: Mutex<HashMap<usize, Weak<EagerTensorRecord>>>,
    ad_transform_cache: Arc<AdTransformCache>,
}

impl fmt::Debug for EagerRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("EagerRuntime");
        debug.field("id", &self.id);
        debug.field("runtime_id", &self.runtime.id());
        debug.field("runtime_epoch", &self.runtime.epoch().ok());
        match self.backend.try_lock() {
            Ok(backend) => {
                debug.field("backend", &*backend);
            }
            Err(_) => {
                debug.field("backend", &"<locked>");
            }
        }
        match self.extension_executor.try_lock() {
            Ok(executor) => {
                debug.field("extension_executor", &*executor);
            }
            Err(_) => {
                debug.field("extension_executor", &"<locked>");
            }
        }
        debug.field(
            "has_extension_ad_dispatcher",
            &self.extension_ad_dispatcher.is_some(),
        );
        match self.grad_slots.try_lock() {
            Ok(slots) => {
                debug.field("grad_slots_len", &slots.len());
            }
            Err(_) => {
                debug.field("grad_slots_len", &"<locked>");
            }
        }
        match self.value_records.try_lock() {
            Ok(records) => {
                debug.field("value_records_len", &records.len());
            }
            Err(_) => {
                debug.field("value_records_len", &"<locked>");
            }
        }
        match self.value_ptr_records.try_lock() {
            Ok(records) => {
                debug.field("value_ptr_records_len", &records.len());
            }
            Err(_) => {
                debug.field("value_ptr_records_len", &"<locked>");
            }
        }
        match self.ad_transform_cache.stats() {
            Ok(stats) => {
                debug.field("ad_transform_cache_stats", &stats);
            }
            Err(err) => {
                debug.field("ad_transform_cache_stats", &format_args!("{err}"));
            }
        }
        debug.finish_non_exhaustive()
    }
}

impl EagerRuntime {
    fn lock_backend(&self) -> Result<MutexGuard<'_, EagerBackend>> {
        self.backend.lock().map_err(|_| {
            Error::runtime_state("eager_backend", ErrorPhase::Execution, "lock poisoned")
        })
    }

    fn lock_extension_executor(&self) -> Result<MutexGuard<'_, ExtensionExecutor<EagerBackend>>> {
        self.extension_executor.lock().map_err(|_| {
            Error::runtime_state(
                "eager_extension_executor",
                ErrorPhase::Execution,
                "lock poisoned",
            )
        })
    }

    fn lock_grad_slots(
        &self,
    ) -> Result<MutexGuard<'_, HashMap<ValueKey<StdTensorOp>, WeakGradSlot>>> {
        self.grad_slots.lock().map_err(|_| {
            Error::runtime_state(
                "eager_gradient_slots",
                ErrorPhase::Execution,
                "lock poisoned",
            )
        })
    }

    fn lock_value_records(
        &self,
    ) -> Result<MutexGuard<'_, HashMap<ValueKey<StdTensorOp>, Weak<EagerTensorRecord>>>> {
        self.value_records.lock().map_err(|_| {
            Error::runtime_state(
                "eager_value_registry",
                ErrorPhase::Execution,
                "lock poisoned",
            )
        })
    }

    fn lock_value_ptr_records(
        &self,
    ) -> Result<MutexGuard<'_, HashMap<usize, Weak<EagerTensorRecord>>>> {
        self.value_ptr_records.lock().map_err(|_| {
            Error::runtime_state(
                "eager_value_pointer_registry",
                ErrorPhase::Execution,
                "lock poisoned",
            )
        })
    }

    fn synchronize_runtime_registration_for_backend(&self, backend: &EagerBackend) -> Result<()> {
        let engine_id = cpu_runtime_engine_id().map_err(|source| {
            runtime_config_error("EagerRuntime::cpu_runtime_engine_id", source)
        })?;
        let snapshot = self
            .runtime
            .snapshot()
            .map_err(|source| runtime_state_source("EagerRuntime::runtime_snapshot", source))?;
        let has_cpu_engine = snapshot.engine(&engine_id).is_some();
        match backend.cpu_snapshot() {
            Some(backend) => {
                let registration = cpu_runtime_engine_registration(&backend).map_err(|source| {
                    runtime_config_error("EagerRuntime::cpu_runtime_engine_registration", source)
                })?;
                self.runtime
                    .reconfigure(|edit| {
                        if has_cpu_engine {
                            edit.replace_engine(registration)?;
                        } else {
                            edit.register_engine(registration)?;
                        }
                        Ok(())
                    })
                    .map(|_| ())
                    .map_err(|source| {
                        runtime_state_source("EagerRuntime::runtime_reconfiguration", source)
                    })
            }
            None if has_cpu_engine => self
                .runtime
                .reconfigure(|edit| {
                    edit.remove_engine(&engine_id)?;
                    Ok(())
                })
                .map(|_| ())
                .map_err(|source| {
                    runtime_state_source("EagerRuntime::runtime_reconfiguration", source)
                }),
            None => Ok(()),
        }
    }

    fn from_backend(backend: EagerBackend) -> Self {
        Self::from_backend_with_extension_ad_dispatcher(backend, None)
    }

    fn from_backend_with_extension_ad_dispatcher(
        backend: EagerBackend,
        extension_ad_dispatcher: Option<Arc<dyn ExtensionAdDispatcher>>,
    ) -> Self {
        Self::from_backend_with_extension_ad_dispatcher_and_cache(
            backend,
            extension_ad_dispatcher,
            Arc::new(AdTransformCache::new()),
        )
    }

    fn from_backend_with_extension_ad_dispatcher_and_cache(
        backend: EagerBackend,
        extension_ad_dispatcher: Option<Arc<dyn ExtensionAdDispatcher>>,
        ad_transform_cache: Arc<AdTransformCache>,
    ) -> Self {
        let runtime = build_eager_runtime_for_backend(&backend);
        Self {
            id: ContextId::fresh(),
            runtime,
            backend: Mutex::new(backend),
            extension_executor: Mutex::new(ExtensionExecutor::new()),
            extension_ad_dispatcher,
            grad_slots: Mutex::new(HashMap::new()),
            value_records: Mutex::new(HashMap::new()),
            value_ptr_records: Mutex::new(HashMap::new()),
            ad_transform_cache,
        }
    }

    /// Create a shared CPU eager execution context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ad::EagerRuntime;
    ///
    /// let ctx = EagerRuntime::new();
    /// assert_eq!(std::sync::Arc::strong_count(&ctx), 1);
    /// ```
    pub fn new() -> Arc<Self> {
        Self::with_cpu_backend(CpuBackend::new())
    }

    /// Create a shared eager execution context from a configured CPU backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1).unwrap());
    /// assert_eq!(std::sync::Arc::strong_count(&ctx), 1);
    /// ```
    pub fn with_cpu_backend(backend: CpuBackend) -> Arc<Self> {
        Arc::new(Self::from_backend(EagerBackend::cpu(backend)))
    }

    /// Snapshot a placement-selected CPU handle from this eager runtime.
    ///
    /// The eager backend lock is held only long enough to verify the backend
    /// kind and clone its CPU coordinator/provider snapshot. Placement
    /// resolution happens after that guard is dropped. The returned value does
    /// not hold a resource permit or a second runtime/backend mutex while idle.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::EagerRuntime;
    /// use tenferro_cpu::CpuPlacement;
    ///
    /// let runtime = EagerRuntime::new();
    /// let cpu = runtime.on_cpu(CpuPlacement::Auto)?;
    /// assert_eq!(cpu.runtime_id(), runtime.id());
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeState`] if the eager backend lock is poisoned,
    /// [`Error::Unsupported`] if the runtime is not CPU-backed, or a typed
    /// tensor runtime error retaining [`tenferro_cpu::CpuPlacementError`] when
    /// the requested placement cannot be resolved.
    pub fn on_cpu(self: &Arc<Self>, placement: CpuPlacement) -> Result<CpuPlacementBoundEager> {
        let backend = {
            let backend = self.lock_backend()?;
            backend.cpu_snapshot().ok_or_else(|| {
                Error::unsupported(
                    "EagerRuntime::on_cpu",
                    ErrorPhase::Execution,
                    "the eager runtime is not CPU-backed",
                )
            })?
        };
        let selection = select_cpu_runtime(&self.runtime)?;
        let backend = backend.for_placement(placement).map_err(|source| {
            let error: tenferro_tensor::Error = CpuBackendError::Placement {
                op: "EagerRuntime::on_cpu",
                source,
            }
            .into();
            Error::from(error)
        })?;
        Ok(CpuPlacementBoundEager {
            runtime: Arc::clone(self),
            backend,
            snapshot: selection.snapshot,
            epoch: selection.epoch,
            engine_id: selection.engine_id,
            registration_identity: selection.registration_identity,
            capabilities: selection.capabilities,
        })
    }

    /// Create a shared CPU eager context with explicit AD extension rules.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{AdContext, EagerRuntime};
    ///
    /// let ad = AdContext::builder().build().unwrap();
    /// let ctx = EagerRuntime::with_cpu_backend_and_ad_context(CpuBackend::new(), &ad);
    /// assert_eq!(std::sync::Arc::strong_count(&ctx), 1);
    /// ```
    pub fn with_cpu_backend_and_ad_context(backend: CpuBackend, ad: &AdContext) -> Arc<Self> {
        Arc::new(Self::from_backend_with_extension_ad_dispatcher_and_cache(
            EagerBackend::cpu(backend),
            ad.extension_ad_dispatcher(),
            ad.ad_transform_cache(),
        ))
    }

    /// Create a shared eager execution context from a configured CUDA backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::CudaBackend;
    /// use tenferro_ad::EagerRuntime;
    ///
    /// let _ctor: fn(CudaBackend) -> std::sync::Arc<EagerRuntime> =
    ///     EagerRuntime::with_cuda_backend;
    /// ```
    #[cfg(feature = "cuda")]
    pub fn with_cuda_backend(backend: CudaBackend) -> Arc<Self> {
        Arc::new(Self::from_backend(EagerBackend::cuda(backend)))
    }

    /// Create a shared CUDA eager context with explicit AD extension rules.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::{AdContext, EagerRuntime};
    /// use tenferro_gpu::CudaBackend;
    ///
    /// let _ctor: fn(CudaBackend, &AdContext) -> std::sync::Arc<EagerRuntime> =
    ///     EagerRuntime::with_cuda_backend_and_ad_context;
    /// ```
    #[cfg(feature = "cuda")]
    pub fn with_cuda_backend_and_ad_context(backend: CudaBackend, ad: &AdContext) -> Arc<Self> {
        Arc::new(Self::from_backend_with_extension_ad_dispatcher_and_cache(
            EagerBackend::cuda(backend),
            ad.extension_ad_dispatcher(),
            ad.ad_transform_cache(),
        ))
    }

    /// Create a shared eager execution context from a configured WebGPU backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ad::EagerRuntime;
    /// use tenferro_gpu::WebGpuBackend;
    ///
    /// let _ctor: fn(WebGpuBackend) -> std::sync::Arc<EagerRuntime> =
    ///     EagerRuntime::with_webgpu_backend;
    /// ```
    #[cfg(feature = "webgpu")]
    pub fn with_webgpu_backend(backend: WebGpuBackend) -> Arc<Self> {
        Arc::new(Self::from_backend(EagerBackend::webgpu(backend)))
    }

    /// Create a shared WebGPU eager context with explicit AD extension rules.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::{AdContext, EagerRuntime};
    /// use tenferro_gpu::WebGpuBackend;
    ///
    /// let _ctor: fn(WebGpuBackend, &AdContext) -> std::sync::Arc<EagerRuntime> =
    ///     EagerRuntime::with_webgpu_backend_and_ad_context;
    /// ```
    #[cfg(feature = "webgpu")]
    pub fn with_webgpu_backend_and_ad_context(backend: WebGpuBackend, ad: &AdContext) -> Arc<Self> {
        Arc::new(Self::from_backend_with_extension_ad_dispatcher_and_cache(
            EagerBackend::webgpu(backend),
            ad.extension_ad_dispatcher(),
            ad.ad_transform_cache(),
        ))
    }

    /// Return an opaque identifier for this context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// assert_ne!(ctx.id(), EagerRuntime::with_cpu_backend(CpuBackend::new()).id());
    /// ```
    pub fn id(&self) -> ContextId {
        self.id
    }

    /// Disable eager operation recording on the current thread until the guard is dropped.
    ///
    /// This is useful for optimizer updates, metric calculations, and other
    /// eager computations that should not become part of the AD tape.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::requires_grad_in(
    ///     Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap(),
    ///     ctx.clone(),
    /// )?;
    /// let y = {
    ///     let _guard = ctx.no_grad();
    ///     x.mul(&x)?
    /// };
    /// assert!(!y.tracks_grad());
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    pub fn no_grad(&self) -> EagerNoGradGuard {
        EAGER_NO_GRAD_DEPTH.with(|depth| {
            depth.set(depth.get().saturating_add(1));
        });
        EagerNoGradGuard { active: true }
    }

    /// Register one extension runtime on this eager context.
    ///
    /// # Errors
    ///
    /// Returns [`ExtensionRuntimeRegistryError::PoisonedLock`] when the
    /// extension executor is unavailable, or propagates the callback's
    /// [`ExtensionRuntimeRegistryError::MalformedFamilyId`] and
    /// [`ExtensionRuntimeRegistryError::MissingHostReference`] failures.
    pub fn register_extension(
        &self,
        register: impl FnOnce(
            &mut ExtensionExecutor<EagerBackend>,
        ) -> std::result::Result<(), ExtensionRuntimeRegistryError>,
    ) -> std::result::Result<(), ExtensionRuntimeRegistryError> {
        let mut executor = self.extension_executor.lock().map_err(|_| {
            ExtensionRuntimeRegistryError::PoisonedLock {
                name: "extension executor lock",
            }
        })?;
        register(&mut executor)
    }

    /// Clear generic extension runtime cache entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// ctx.clear_extension_caches()?;
    /// assert_eq!(ctx.cache_stats()?.extensions.entries, 0);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] when the extension
    /// executor lock is poisoned.
    pub fn clear_extension_caches(&self) -> Result<()> {
        self.lock_extension_executor()?.clear_caches();
        Ok(())
    }

    /// Clear every cache owned by this eager context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// ctx.clear_caches()?;
    /// assert_eq!(ctx.cache_stats()?.extensions.entries, 0);
    /// assert_eq!(ctx.cache_stats()?.ad_transforms.entries, 0);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] when either the
    /// extension executor or AD-transform cache is poisoned.
    pub fn clear_caches(&self) -> Result<()> {
        self.clear_extension_caches()?;
        self.clear_ad_transform_caches()?;
        Ok(())
    }

    /// Return eager runtime cache-entry and retained-byte stats.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let stats = ctx.cache_stats()?;
    /// assert_eq!(stats.extensions.entries, 0);
    /// assert_eq!(stats.ad_transforms.entries, 0);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] when an executor or
    /// AD-transform cache lock is poisoned.
    pub fn cache_stats(&self) -> Result<EagerRuntimeCacheStats> {
        Ok(EagerRuntimeCacheStats {
            extensions: self.lock_extension_executor()?.cache_stats(),
            ad_transforms: self.ad_transform_cache.stats()?,
        })
    }

    /// Return the AD transform cache retention limits.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ad::EagerRuntime;
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// assert!(ctx.ad_transform_cache_limits()?.max_entries().get() > 0);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] if the AD-transform
    /// cache lock is poisoned.
    pub fn ad_transform_cache_limits(&self) -> Result<AdTransformCacheLimits> {
        self.ad_transform_cache.limits()
    }

    /// Replace AD transform cache retention limits.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use tenferro_ad::{AdTransformCacheLimits, EagerRuntime};
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let limits = AdTransformCacheLimits::new(NonZeroUsize::new(1).unwrap());
    /// ctx.set_ad_transform_cache_limits(limits)?;
    /// assert_eq!(ctx.ad_transform_cache_limits()?, limits);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] if the AD-transform
    /// cache lock is poisoned while updating limits.
    pub fn set_ad_transform_cache_limits(&self, limits: AdTransformCacheLimits) -> Result<()> {
        self.ad_transform_cache.set_limits(limits)
    }

    /// Clear AD transform cache entries visible through this eager runtime.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ad::EagerRuntime;
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// ctx.clear_ad_transform_caches()?;
    /// assert_eq!(ctx.cache_stats()?.ad_transforms.entries, 0);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] if the AD-transform
    /// cache lock is poisoned while clearing entries.
    pub fn clear_ad_transform_caches(&self) -> Result<()> {
        self.ad_transform_cache.clear()
    }

    /// Return the extension cache retention limits.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] if the extension
    /// executor lock is poisoned.
    pub fn extension_cache_limits(&self) -> Result<ExtensionCacheLimits> {
        Ok(self.lock_extension_executor()?.cache_limits())
    }

    /// Replace extension cache retention limits.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] if the extension
    /// executor lock is poisoned.
    pub fn set_extension_cache_limits(&self, limits: ExtensionCacheLimits) -> Result<()> {
        self.lock_extension_executor()?.set_cache_limits(limits);
        Ok(())
    }

    /// Mutably borrow generic extension runtime cache storage.
    ///
    /// This hook is for standard extension crates that need cache entries
    /// owned by an eager runtime while preserving eager value semantics outside
    /// a registered extension execution boundary.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ad::EagerRuntime;
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::ExtensionCacheKey;
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let key = ExtensionCacheKey::new("example.cache.v1", "plans", 1);
    ///
    /// ctx.with_extension_caches_mut(|caches| {
    ///     caches.put(key, 7_usize, std::mem::size_of::<usize>());
    /// });
    ///
    /// assert_eq!(ctx.cache_stats().unwrap().extensions.entries, 1);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] if the extension
    /// executor lock is poisoned before the closure can run.
    pub fn with_extension_caches_mut<R>(
        &self,
        f: impl FnOnce(&mut ExtensionCacheStore) -> R,
    ) -> Result<R> {
        let mut executor = self.lock_extension_executor()?;
        Ok(f(executor.caches_mut()))
    }

    /// Mutably borrow this runtime's backend.
    ///
    /// This hook lets standard extension crates run a whole contraction program
    /// in a single backend session (instead of one eager op per step) while
    /// preserving eager value semantics for untracked tensors.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ad::EagerRuntime;
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// // The closure receives `&mut EagerBackend`; standard extension crates
    /// // use it to open one backend session for a whole contraction program.
    /// let answer = ctx.with_backend_mut(|_backend| 42).unwrap();
    /// assert_eq!(answer, 42);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] if the eager backend
    /// lock is poisoned before the closure can run.
    pub fn with_backend_mut<R>(&self, f: impl FnOnce(&mut EagerBackend) -> R) -> Result<R> {
        let mut backend = self.lock_backend()?;
        let result = f(&mut backend);
        self.synchronize_runtime_registration_for_backend(&backend)?;
        Ok(result)
    }

    pub(crate) fn materialize_value(&self, value: &TensorValue) -> Result<Tensor> {
        if let Some(tensor) = value.as_tensor_arc() {
            return Ok(tensor.as_ref().clone());
        }

        let mut backend = self.lock_backend()?;
        backend
            .with_backend_session(|exec| exec.to_contiguous_read(value.tensor_read()))
            .map_err(Error::from)
    }

    /// Block the current thread until backend work submitted by this eager runtime completes.
    ///
    /// CPU runtimes return immediately. CUDA and WebGPU runtimes synchronize
    /// their current backend work queue.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::EagerRuntime;
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// ctx.synchronize().unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] if the backend lock is
    /// poisoned, or a typed tensor backend error if synchronization fails.
    pub fn synchronize(&self) -> Result<()> {
        self.lock_backend()?.synchronize().map_err(Error::from)
    }

    fn exec_outputs_with_optional_extension_lock<R>(
        &self,
        lock_backend_section: &'static str,
        lock_extensions_section: &'static str,
        exec_section: &'static str,
        op: &StdTensorOp,
        execute: impl FnOnce(
            &mut EagerBackend,
            Option<&mut ExtensionExecutor<EagerBackend>>,
        ) -> Result<R>,
    ) -> Result<R> {
        let mut backend = profile_eager_op_section(lock_backend_section, || self.lock_backend())?;
        if matches!(op, StdTensorOp::Extension(_)) {
            // Lock ordering: eager execution always acquires backend before
            // extension_executor. Extension runtimes must not re-enter this
            // same EagerRuntime while their callback is running.
            let mut extension_executor = profile_eager_op_section(lock_extensions_section, || {
                self.lock_extension_executor()
            })?;
            return profile_eager_op_section(exec_section, || {
                execute(&mut backend, Some(&mut *extension_executor))
            });
        }

        profile_eager_op_section(exec_section, || execute(&mut backend, None))
    }

    pub(crate) fn exec_outputs(&self, op: &StdTensorOp, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        self.exec_outputs_with_optional_extension_lock(
            "exec_outputs.lock_backend",
            "exec_outputs.lock_extensions",
            "exec_outputs.exec_op",
            op,
            |backend, extension_executor| {
                exec_op_on_tensors_with_extension_executor(op, inputs, backend, extension_executor)
            },
        )
    }

    pub(crate) fn exec_outputs_read(
        &self,
        op: &StdTensorOp,
        inputs: &[TensorRead<'_>],
    ) -> Result<Vec<Tensor>> {
        self.exec_outputs_with_optional_extension_lock(
            "exec_outputs_read.lock_backend",
            "exec_outputs_read.lock_extensions",
            "exec_outputs_read.exec_op",
            op,
            |backend, extension_executor| {
                exec_op_on_tensor_reads_with_extension_executor(
                    op,
                    inputs,
                    backend,
                    extension_executor,
                )
            },
        )
    }

    #[cfg(test)]
    pub(crate) fn exec_standard_graph_outputs(
        &self,
        graph: &Graph<StdTensorOp>,
        initial_data: &HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
    ) -> Result<EagerGraphExecution> {
        let mut backend =
            profile_eager_op_section("exec_graph.lock_backend", || self.lock_backend())?;
        let mut all_values = initial_data.clone();

        profile_eager_op_section("exec_graph.with_backend_session", || {
            backend.with_backend_session(|exec| -> Result<()> {
                for op_node in graph.operations() {
                    let outputs = {
                        let input_values = op_node
                            .inputs
                            .iter()
                            .map(|input| {
                                let key = match input {
                                    ValueRef::Local(local_id) => &graph.values()[*local_id].key,
                                    ValueRef::External(key) => key,
                                };
                                all_values.get(key).cloned().ok_or_else(|| {
                                    Error::Internal(format!(
                                        "standard graph eager execution missing value for {key:?}"
                                    ))
                                })
                            })
                            .collect::<Result<Vec<_>>>()?;
                        let input_reads = input_values
                            .iter()
                            .map(|value| TensorRead::from_tensor(value.as_ref()))
                            .collect::<Vec<_>>();
                        exec_standard_op_on_tensor_reads_in_session(
                            &op_node.operation,
                            &input_reads,
                            exec,
                        )?
                    };

                    if outputs.len() != op_node.outputs.len() {
                        return Err(Error::Internal(format!(
                            "standard graph eager execution expected {} outputs for {:?}, got {}",
                            op_node.outputs.len(),
                            op_node.operation,
                            outputs.len()
                        )));
                    }

                    for (output_id, output) in op_node.outputs.iter().zip(outputs) {
                        let key = graph.values()[*output_id].key.clone();
                        all_values.insert(key, Arc::new(output));
                    }
                }
                Ok(())
            })
        })?;

        let outputs = graph
            .outputs()
            .iter()
            .map(|&output_id| {
                let key = &graph.values()[output_id].key;
                all_values.get(key).cloned().ok_or_else(|| {
                    Error::Internal(format!(
                        "standard graph eager execution missing graph output {key:?}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(EagerGraphExecution {
            outputs,
            retained_values: all_values,
        })
    }

    pub(crate) fn try_register_grad_slot(
        &self,
        key: &ValueKey<StdTensorOp>,
        slot: &GradSlot,
    ) -> Result<()> {
        self.lock_grad_slots()?
            .insert(key.clone(), Arc::downgrade(slot));
        Ok(())
    }

    pub(crate) fn try_register_value_record(
        &self,
        key: &ValueKey<StdTensorOp>,
        record: &Arc<EagerTensorRecord>,
    ) -> Result<()> {
        self.lock_value_records()?
            .insert(key.clone(), Arc::downgrade(record));
        self.try_register_value_record_ptr(record)?;
        Ok(())
    }

    pub(crate) fn try_register_value_record_ptr(
        &self,
        record: &Arc<EagerTensorRecord>,
    ) -> Result<()> {
        let tensor = match record.value.as_tensor_arc() {
            Some(tensor) => Some(Arc::clone(tensor)),
            None => record.materialized_cache.get().cloned(),
        };
        let Some(tensor) = tensor else {
            return Ok(());
        };
        self.lock_value_ptr_records()?
            .insert(tensor_ptr(&tensor), Arc::downgrade(record));
        Ok(())
    }

    pub(crate) fn value_record(
        &self,
        key: &ValueKey<StdTensorOp>,
    ) -> Result<Option<Arc<EagerTensorRecord>>> {
        let mut records = self.lock_value_records()?;
        let Some(record) = records.get(key).cloned() else {
            return Ok(None);
        };
        match record.upgrade() {
            Some(record) => Ok(Some(record)),
            None => {
                records.remove(key);
                Ok(None)
            }
        }
    }

    pub(crate) fn value_record_by_tensor(
        &self,
        tensor: &Arc<Tensor>,
    ) -> Result<Option<Arc<EagerTensorRecord>>> {
        let ptr = tensor_ptr(tensor);
        let mut records = self.lock_value_ptr_records()?;
        let Some(record) = records.get(&ptr).cloned() else {
            return Ok(None);
        };
        match record.upgrade() {
            Some(record) => Ok(Some(record)),
            None => {
                records.remove(&ptr);
                Ok(None)
            }
        }
    }

    pub(crate) fn cached_linearize_recorded_graph(
        &self,
        graph: &RecordedGraph<StdTensorOp>,
        output_slots: &[usize],
        ctx: &mut ShapeGuardContext,
    ) -> tidu::ADRuleResult<Arc<LinearizedGraph<StdTensorOp>>> {
        let key = EagerAdTransformCacheKey::new(graph, output_slots);
        if let Some(linear) = self
            .ad_transform_cache
            .get_eager_linearized(&key)
            .map_err(eager_ad_transform_cache_error)?
        {
            return Ok(linear);
        }

        let linear = Arc::new(graph.linearize(output_slots, ctx)?);
        self.ad_transform_cache
            .put_eager_linearized(key, Arc::clone(&linear))
            .map_err(eager_ad_transform_cache_error)?;
        Ok(linear)
    }

    /// Clear all live gradient slots tracked by this context.
    ///
    /// This resets the stored gradients to `None` without unregistering the
    /// tensors, so future `backward()` calls can accumulate again.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]).unwrap(), ctx.clone()).unwrap();
    /// let loss = x.mul(&y).unwrap().reduce_sum(Some(&[0])).unwrap();
    /// let _ = loss.backward().unwrap();
    ///
    /// ctx.clear_grads()?;
    ///
    /// assert!(x.grad()?.is_none());
    /// assert!(y.grad()?.is_none());
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] if a gradient-slot
    /// lock is poisoned while clearing live gradients.
    pub fn clear_grads(&self) -> Result<()> {
        let live_slots = {
            let mut live_slots = Vec::new();
            self.lock_grad_slots()?.retain(|_, slot| {
                if let Some(slot) = slot.upgrade() {
                    live_slots.push(slot);
                    true
                } else {
                    false
                }
            });
            live_slots
        };

        let mut poisoned_slot = false;
        for slot in live_slots {
            match slot.lock() {
                Ok(mut current) => {
                    *current = None;
                }
                Err(_) => {
                    poisoned_slot = true;
                }
            }
        }
        if poisoned_slot {
            return Err(Error::runtime_state(
                "eager_gradient_slot",
                ErrorPhase::Execution,
                "lock poisoned",
            ));
        }
        Ok(())
    }

    /// Import a concrete tensor into this context as an untracked constant.
    ///
    /// The returned tensor does not participate in gradient tracking.
    /// Use this for fixed masks, quadrature weights, physical constants,
    /// and other data that should not receive gradients.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let c = ctx.constant_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap())?;
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(), ctx)?;
    /// let z = x.add(&c).unwrap();
    ///
    /// assert_eq!(z.materialized()?.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] when metadata cannot
    /// be registered or the backend lock is poisoned.
    pub fn constant_from(self: &Arc<Self>, tensor: Tensor) -> Result<EagerTensor> {
        EagerTensor::new_leaf(Arc::clone(self), tensor, false)
    }

    /// Import a concrete tensor into this context as a trainable variable.
    ///
    /// The returned tensor participates in gradient tracking; its gradient
    /// slot is registered in this context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let p = ctx.variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap())?;
    /// let loss = p.exp().unwrap().reduce_sum(Some(&[0])).unwrap();
    /// let _ = loss.backward().unwrap();
    ///
    /// let grad = p.grad().unwrap().unwrap();
    /// assert_eq!(grad.shape(), &[2]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] when gradient metadata
    /// or the eager backend state cannot be registered.
    pub fn variable_from(self: &Arc<Self>, tensor: Tensor) -> Result<EagerTensor> {
        EagerTensor::new_leaf(Arc::clone(self), tensor, true)
    }

    /// Gradient of a scalar eager output with respect to an eager tensor.
    ///
    /// Functional eager gradients return ordinary eager tensors and do not
    /// write into `grad()` slots. The returned tensor keeps a trace when the
    /// derivative computation depends on tracked eager values.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::requires_grad_in(
    ///     Tensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap(),
    ///     ctx.clone(),
    /// )?;
    /// let loss = x.mul(&x)?;
    /// let dx = ctx.grad(&loss, &x)?;
    /// assert_eq!(dx.materialized()?.as_slice::<f64>().unwrap(), &[6.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::NonScalarGrad`] for a non-scalar
    /// output, [`Error::ContextMismatch`] for tensors from another runtime,
    /// [`Error::UnsupportedAdRule`] when an AD rule is unavailable, or a typed
    /// validation/backend error from eager execution.
    pub fn grad(self: &Arc<Self>, output: &EagerTensor, wrt: &EagerTensor) -> Result<EagerTensor> {
        self.grad_optional(output, wrt)?
            .ok_or_else(|| Error::Internal(format!("grad output is inactive for {:?}", wrt.key)))
    }

    /// Gradient that returns `None` when `wrt` is inactive.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::requires_grad_in(
    ///     Tensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap(),
    ///     ctx.clone(),
    /// )?;
    /// let y = EagerTensor::requires_grad_in(
    ///     Tensor::from_vec_col_major(vec![], vec![4.0_f64]).unwrap(),
    ///     ctx.clone(),
    /// )?;
    /// let loss = y.mul(&y)?;
    /// assert!(ctx.grad_optional(&loss, &x)?.is_none());
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::NonScalarGrad`] for a non-scalar
    /// output, [`Error::ContextMismatch`] for a foreign runtime, or a typed
    /// validation/backend/runtime-state error from eager execution.
    pub fn grad_optional(
        self: &Arc<Self>,
        output: &EagerTensor,
        wrt: &EagerTensor,
    ) -> Result<Option<EagerTensor>> {
        if !output.shape().is_empty() {
            return Err(Error::NonScalarGrad {
                shape: output.shape().to_vec(),
            });
        }

        let value = output.materialized_arc()?;
        let seed = {
            let mut backend = self.lock_backend()?;
            one_like_tensor(value.as_ref(), &mut *backend)?
        };
        let seed = EagerTensor::new_result_arc(
            Arc::clone(self),
            eager_val_key(),
            Arc::new(seed),
            false,
            None,
            Vec::new(),
        )?;
        self.vjp_optional(output, wrt, &seed)
    }

    /// Reverse-mode vector-Jacobian product for eager tensors.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::requires_grad_in(
    ///     Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
    ///     ctx.clone(),
    /// )?;
    /// let y = x.mul(&x)?;
    /// let seed = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 1.0]).unwrap(),
    ///     ctx.clone(),
    /// )?;
    /// let dx = ctx.vjp(&y, &x, &seed)?;
    /// assert_eq!(dx.materialized()?.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::ContextMismatch`] for tensors from different eager
    /// runtimes, [`Error::Validation`] when the cotangent shape or dtype does
    /// not match the output, [`Error::UnsupportedAdRule`] when a rule is not
    /// registered, or a typed backend/runtime-state error.
    pub fn vjp(
        self: &Arc<Self>,
        output: &EagerTensor,
        wrt: &EagerTensor,
        cotangent: &EagerTensor,
    ) -> Result<EagerTensor> {
        self.vjp_optional(output, wrt, cotangent)?
            .ok_or_else(|| Error::Internal(format!("vjp output is inactive for {:?}", wrt.key)))
    }

    /// Reverse-mode vector-Jacobian product that returns `None` for inactive inputs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::requires_grad_in(
    ///     Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap(),
    ///     ctx.clone(),
    /// )?;
    /// let y = EagerTensor::requires_grad_in(
    ///     Tensor::from_vec_col_major(vec![1], vec![4.0_f64]).unwrap(),
    ///     ctx.clone(),
    /// )?;
    /// let seed = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
    ///     ctx.clone(),
    /// )?;
    /// let loss = y.mul(&y)?;
    /// assert!(ctx.vjp_optional(&loss, &x, &seed)?.is_none());
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::ContextMismatch`] for tensors from different eager
    /// runtimes, [`Error::Validation`] when the cotangent shape or dtype does
    /// not match the output, [`Error::UnsupportedAdRule`] when a rule is not
    /// registered, or a typed backend/runtime-state error.
    pub fn vjp_optional(
        self: &Arc<Self>,
        output: &EagerTensor,
        wrt: &EagerTensor,
        cotangent: &EagerTensor,
    ) -> Result<Option<EagerTensor>> {
        validate_same_runtime(self, output, "vjp output")?;
        validate_same_runtime(self, wrt, "vjp wrt")?;
        validate_same_runtime(self, cotangent, "vjp cotangent")?;
        validate_seed_tensor("vjp", output, cotangent)?;
        functional_vjp_optional(self, output, wrt, cotangent)
    }

    /// Forward-mode Jacobian-vector product for eager tensors.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::requires_grad_in(
    ///     Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap(),
    ///     ctx.clone(),
    /// )?;
    /// let tangent = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
    ///     ctx.clone(),
    /// )?;
    /// let y = x.mul(&x)?;
    /// let dy = ctx.jvp(&y, &x, &tangent)?;
    /// assert_eq!(dy.materialized()?.as_slice::<f64>().unwrap(), &[6.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::ContextMismatch`] for tensors from different eager
    /// runtimes, [`Error::Validation`] when the tangent shape or dtype does not
    /// match `wrt`, [`Error::UnsupportedAdRule`] when a rule is unavailable, or
    /// a typed backend/runtime-state error.
    pub fn jvp(
        self: &Arc<Self>,
        output: &EagerTensor,
        wrt: &EagerTensor,
        tangent: &EagerTensor,
    ) -> Result<EagerTensor> {
        self.jvp_optional(output, wrt, tangent)?
            .ok_or_else(|| Error::Internal(format!("jvp output is inactive for {:?}", wrt.key)))
    }

    /// Forward-mode Jacobian-vector product that returns `None` for inactive outputs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::requires_grad_in(
    ///     Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap(),
    ///     ctx.clone(),
    /// )?;
    /// let y = EagerTensor::requires_grad_in(
    ///     Tensor::from_vec_col_major(vec![1], vec![4.0_f64]).unwrap(),
    ///     ctx.clone(),
    /// )?;
    /// let tangent = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
    ///     ctx.clone(),
    /// )?;
    /// let loss = y.mul(&y)?;
    /// assert!(ctx.jvp_optional(&loss, &x, &tangent)?.is_none());
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::ContextMismatch`] for tensors from different eager
    /// runtimes, [`Error::Validation`] when the tangent shape or dtype does not
    /// match `wrt`, [`Error::UnsupportedAdRule`] when a rule is unavailable, or
    /// a typed backend/runtime-state error.
    pub fn jvp_optional(
        self: &Arc<Self>,
        output: &EagerTensor,
        wrt: &EagerTensor,
        tangent: &EagerTensor,
    ) -> Result<Option<EagerTensor>> {
        validate_same_runtime(self, output, "jvp output")?;
        validate_same_runtime(self, wrt, "jvp wrt")?;
        validate_same_runtime(self, tangent, "jvp tangent")?;
        validate_seed_tensor("jvp", wrt, tangent)?;
        functional_jvp(self, output, wrt, tangent)
    }

    fn store_grads(
        &self,
        cotangents: &HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
        backend: &mut EagerBackend,
    ) -> Result<()> {
        let mut updates = Vec::new();

        {
            let mut slots = self.lock_grad_slots()?;
            slots.retain(|key, slot| {
                let Some(slot) = slot.upgrade() else {
                    return false;
                };

                if let Some(incoming) = cotangents.get(key) {
                    updates.push((slot, Arc::clone(incoming)));
                }

                true
            });
        }

        for (slot, incoming) in updates {
            let mut current = slot.lock().map_err(|_| {
                Error::runtime_state(
                    "eager_gradient_slot",
                    ErrorPhase::Execution,
                    "lock poisoned",
                )
            })?;
            let next = match current.as_ref() {
                Some(existing) => Arc::new(backend.add(existing.as_ref(), incoming.as_ref())?),
                None => incoming,
            };
            *current = Some(next);
        }

        Ok(())
    }
}

fn validate_same_runtime(
    runtime: &Arc<EagerRuntime>,
    tensor: &EagerTensor,
    role: &'static str,
) -> Result<()> {
    if tensor.ctx_id() != runtime.id() {
        return Err(Error::ContextMismatch {
            lhs: runtime.id(),
            rhs: tensor.ctx_id(),
        });
    }
    let _ = role;
    Ok(())
}

pub(crate) fn tensor_ptr(tensor: &Arc<Tensor>) -> usize {
    Arc::as_ptr(tensor) as usize
}

fn validate_seed_tensor(op: &'static str, primal: &EagerTensor, seed: &EagerTensor) -> Result<()> {
    if primal.dtype() != seed.dtype() {
        return Err(
            tenferro_tensor::Error::dtype_mismatch(op, primal.dtype(), seed.dtype()).into(),
        );
    }
    if primal.shape() != seed.shape() {
        return Err(
            tenferro_tensor::Error::shape_mismatch(op, primal.shape(), seed.shape()).into(),
        );
    }
    Ok(())
}

/// Eager tensor with reverse-mode autodiff over concrete tensor values.
///
/// This executes each primitive immediately and records a lightweight reverse
/// DAG for `backward()`. Gradients accumulate across repeated `backward()`
/// calls until they are cleared explicitly.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
///
/// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
/// let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap(), ctx)?;
/// let loss = x.mul(&x).unwrap().reduce_sum(Some(&[0])).unwrap();
/// let _cotangents = loss.backward().unwrap();
/// let loss = x.mul(&x).unwrap().reduce_sum(Some(&[0])).unwrap();
/// let _cotangents = loss.backward().unwrap();
///
/// assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[4.0, 8.0, 12.0]);
/// x.clear_grad();
///
/// assert!(x.grad().unwrap().is_none());
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
#[derive(Clone)]
pub struct EagerTensor {
    pub(crate) value: Arc<TensorValue>,
    materialized_cache: Arc<OnceLock<Arc<Tensor>>>,
    pub(crate) key: ValueKey<StdTensorOp>,
    pub(crate) trace: Option<Trace<StdTensorOp>>,
    pub(crate) requires_grad: bool,
    grad_slot: GradSlot,
    pub(crate) metadata_scopes: Vec<Arc<GlobalMetadataScope>>,
    pub(crate) ctx: Arc<EagerRuntime>,
    _record: Arc<EagerTensorRecord>,
}

pub(crate) struct EagerTensorRecord {
    value: Arc<TensorValue>,
    materialized_cache: Arc<OnceLock<Arc<Tensor>>>,
    key: ValueKey<StdTensorOp>,
    trace: Option<Trace<StdTensorOp>>,
    requires_grad: bool,
    grad_slot: GradSlot,
    metadata_scopes: Vec<Arc<GlobalMetadataScope>>,
    ctx: Arc<EagerRuntime>,
}

impl fmt::Debug for EagerTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EagerTensor")
            .field("dtype", &self.dtype())
            .field("shape", &self.shape())
            .field("key", &self.key)
            .field("requires_grad", &self.requires_grad)
            .field("has_trace", &self.trace.is_some())
            .field("ctx_id", &self.ctx_id())
            .finish_non_exhaustive()
    }
}

impl EagerTensor {
    /// Create an untracked eager tensor inside an existing eager context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(), ctx)?;
    ///
    /// assert_eq!(x.materialized()?.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] when metadata cannot
    /// be registered in the target context, or a typed tensor/backend error
    /// while materializing the source value.
    pub fn from_tensor_in(tensor: Tensor, ctx: Arc<EagerRuntime>) -> Result<Self> {
        Self::new_leaf(ctx, tensor, false)
    }

    /// Create an untracked eager tensor from compact column-major data inside
    /// an existing eager runtime.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TensorRuntime`] with
    /// [`tenferro_tensor::ValidationError::ShapeMismatch`] when the shape and
    /// data length disagree, or with
    /// [`tenferro_tensor::ValidationError::IntegerOverflow`] when shape
    /// arithmetic overflows. Returns [`Error::RuntimeState`] when eager
    /// metadata cannot be registered.
    pub fn from_vec_col_major_in<T: TensorScalar>(
        shape: impl IntoShapeVec,
        data: Vec<T>,
        ctx: Arc<EagerRuntime>,
    ) -> Result<Self> {
        Self::from_tensor_in(Tensor::from_vec_col_major(shape, data)?, ctx)
    }

    /// Create a tracked eager leaf inside an existing eager context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(), ctx)?;
    ///
    /// assert!(x.grad().unwrap().is_none());
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] when gradient metadata
    /// cannot be registered in the target context, or a typed tensor/backend
    /// error while creating the leaf.
    pub fn requires_grad_in(tensor: Tensor, ctx: Arc<EagerRuntime>) -> Result<Self> {
        Self::new_leaf(ctx, tensor, true)
    }

    pub(crate) fn new_leaf(
        ctx: Arc<EagerRuntime>,
        tensor: Tensor,
        requires_grad: bool,
    ) -> Result<Self> {
        let key = eager_val_key();
        let metadata_scope =
            register_scoped_value_metadata(key.clone(), tensor_meta_from_tensor(&tensor)).map_err(
                |err| {
                    Error::runtime_state_source("eager leaf metadata", ErrorPhase::GraphBuild, err)
                },
            )?;
        Self::from_parts(
            ctx,
            key,
            requires_grad,
            None,
            Arc::new(TensorValue::from_tensor_arc(Arc::new(tensor))),
            metadata_scopes_for_scope(metadata_scope),
            true,
        )
    }

    pub(crate) fn new_result(
        ctx: Arc<EagerRuntime>,
        key: ValueKey<StdTensorOp>,
        tensor: Tensor,
        requires_grad: bool,
        trace: Option<Trace<StdTensorOp>>,
        metadata_scopes: Vec<Arc<GlobalMetadataScope>>,
    ) -> Result<Self> {
        Self::new_result_arc(
            ctx,
            key,
            Arc::new(tensor),
            requires_grad,
            trace,
            metadata_scopes,
        )
    }

    pub(crate) fn new_result_arc(
        ctx: Arc<EagerRuntime>,
        key: ValueKey<StdTensorOp>,
        tensor: Arc<Tensor>,
        requires_grad: bool,
        trace: Option<Trace<StdTensorOp>>,
        metadata_scopes: Vec<Arc<GlobalMetadataScope>>,
    ) -> Result<Self> {
        Self::from_parts(
            ctx,
            key,
            requires_grad,
            trace,
            Arc::new(TensorValue::from_tensor_arc(tensor)),
            metadata_scopes,
            true,
        )
    }

    pub(crate) fn new_result_value(
        ctx: Arc<EagerRuntime>,
        key: ValueKey<StdTensorOp>,
        value: TensorValue,
        requires_grad: bool,
        trace: Option<Trace<StdTensorOp>>,
        metadata_scopes: Vec<Arc<GlobalMetadataScope>>,
    ) -> Result<Self> {
        Self::from_parts(
            ctx,
            key,
            requires_grad,
            trace,
            Arc::new(value),
            metadata_scopes,
            true,
        )
    }

    fn from_parts(
        ctx: Arc<EagerRuntime>,
        key: ValueKey<StdTensorOp>,
        requires_grad: bool,
        trace: Option<Trace<StdTensorOp>>,
        value: Arc<TensorValue>,
        metadata_scopes: Vec<Arc<GlobalMetadataScope>>,
        register_value: bool,
    ) -> Result<Self> {
        let grad_slot = Arc::new(Mutex::new(None));
        if requires_grad {
            ctx.try_register_grad_slot(&key, &grad_slot)?;
        }
        let materialized_cache = Arc::new(OnceLock::new());
        let record = Arc::new(EagerTensorRecord {
            value: Arc::clone(&value),
            materialized_cache: Arc::clone(&materialized_cache),
            key: key.clone(),
            trace: trace.clone(),
            requires_grad,
            grad_slot: Arc::clone(&grad_slot),
            metadata_scopes: metadata_scopes.clone(),
            ctx: Arc::clone(&ctx),
        });
        if register_value {
            ctx.try_register_value_record(&key, &record)?;
        }

        Ok(Self {
            value,
            materialized_cache,
            key,
            trace,
            requires_grad,
            grad_slot,
            metadata_scopes,
            ctx,
            _record: record,
        })
    }

    pub(crate) fn new_untracked_result(ctx: Arc<EagerRuntime>, tensor: Tensor) -> Result<Self> {
        Self::new_result(ctx, eager_val_key(), tensor, false, None, Vec::new())
    }

    pub(crate) fn new_untracked_value_result(ctx: Arc<EagerRuntime>, value: TensorValue) -> Self {
        let value = Arc::new(value);
        let materialized_cache = Arc::new(OnceLock::new());
        let key = eager_val_key();
        let grad_slot = Arc::new(Mutex::new(None));
        let record = Arc::new(EagerTensorRecord {
            value: Arc::clone(&value),
            materialized_cache: Arc::clone(&materialized_cache),
            key: key.clone(),
            trace: None,
            requires_grad: false,
            grad_slot: Arc::clone(&grad_slot),
            metadata_scopes: Vec::new(),
            ctx: Arc::clone(&ctx),
        });
        Self {
            value,
            materialized_cache,
            key,
            trace: None,
            requires_grad: false,
            grad_slot,
            metadata_scopes: Vec::new(),
            ctx,
            _record: record,
        }
    }

    pub(crate) fn from_record(record: Arc<EagerTensorRecord>) -> Self {
        Self {
            value: Arc::clone(&record.value),
            materialized_cache: Arc::clone(&record.materialized_cache),
            key: record.key.clone(),
            trace: record.trace.clone(),
            requires_grad: record.requires_grad,
            grad_slot: Arc::clone(&record.grad_slot),
            metadata_scopes: record.metadata_scopes.clone(),
            ctx: Arc::clone(&record.ctx),
            _record: record,
        }
    }

    /// Detach this tensor from the reverse graph.
    ///
    /// The returned tensor keeps the concrete value but no longer contributes
    /// gradients to the original graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(), ctx)?;
    /// let y = x.detach();
    ///
    /// assert_eq!(y.materialized()?.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    /// assert!(y.grad().unwrap().is_none());
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    pub fn detach(&self) -> Self {
        Self::new_untracked_value_result(self.ctx.clone(), self.value.as_ref().clone())
    }

    /// Detach this tensor from its graph and re-register it in a different
    /// context as an untracked leaf.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx_a = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let ctx_b = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(), ctx_a)?;
    /// let d = x.detach_into(&ctx_b)?;
    ///
    /// assert!(!d.tracks_grad());
    /// assert_eq!(d.ctx_id(), ctx_b.id());
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeState`] if the source cannot be materialized or
    /// the target context cannot register its metadata.
    pub fn detach_into(&self, ctx: &Arc<EagerRuntime>) -> Result<Self> {
        Self::from_tensor_in(self.to_tensor()?, Arc::clone(ctx))
    }

    /// Materialize and share the concrete tensor value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap(), ctx)?;
    /// assert_eq!(x.materialized()?.as_slice::<f64>().unwrap(), &[3.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeState`] when lazy/backend-resident storage cannot
    /// be materialized, or when eager value-record state is poisoned.
    pub fn materialized(&self) -> Result<Arc<Tensor>> {
        self.materialized_arc()
    }

    /// Return this tensor's scalar dtype without materializing through
    /// [`materialized`](Self::materialized).
    pub fn dtype(&self) -> DType {
        self.value.dtype()
    }

    /// Return this tensor's logical shape without materializing through
    /// [`materialized`](Self::materialized).
    pub fn shape(&self) -> &[usize] {
        self.value.shape()
    }

    /// Borrow this tensor value as a [`TensorRead`].
    ///
    /// This is the preferred borrowed input boundary for executor calls. It
    /// preserves the option to replace eager storage with non-contiguous views
    /// without forcing callers through [`materialized`](Self::materialized).
    pub fn tensor_read(&self) -> TensorRead<'_> {
        self.value.tensor_read()
    }

    /// Materialize this eager tensor as an owned [`Tensor`].
    ///
    /// This is the owned materialization boundary for callers that need a
    /// standalone compact tensor. The operation is fallible because eager
    /// values may be backed by lazy or backend-resident storage.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeState`] if backend state is unavailable, or a
    /// typed tensor backend error when contiguous materialization fails.
    pub fn to_tensor(&self) -> Result<Tensor> {
        self.ctx.materialize_value(self.value.as_ref())
    }

    pub(crate) fn materialized_arc(&self) -> Result<Arc<Tensor>> {
        if let Some(tensor) = self.value.as_tensor_arc() {
            self.ctx.try_register_value_record_ptr(&self._record)?;
            return Ok(Arc::clone(tensor));
        }
        if let Some(tensor) = self.materialized_cache.get() {
            self.ctx.try_register_value_record_ptr(&self._record)?;
            return Ok(Arc::clone(tensor));
        }

        let materialized = Arc::new(self.ctx.materialize_value(self.value.as_ref())?);
        let _ = self.materialized_cache.set(Arc::clone(&materialized));
        self.ctx.try_register_value_record_ptr(&self._record)?;
        Ok(self
            .materialized_cache
            .get()
            .map(Arc::clone)
            .unwrap_or(materialized))
    }

    #[cfg(test)]
    pub(crate) fn materialized_cache_is_initialized(&self) -> bool {
        self.materialized_cache.get().is_some()
    }

    /// Return the accumulated gradient currently stored for this tensor.
    ///
    /// The stored gradient accumulates across repeated `backward()` calls
    /// until it is cleared explicitly.
    ///
    /// For complex scalar losses, stored gradients use tenferro's
    /// Hermitian-adjoint cotangent convention. See
    /// <https://tensor4all.org/tenferro-rs/guides/complex-ad.html>.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(), ctx).unwrap();
    /// let loss = x.exp().unwrap().reduce_sum(Some(&[0])).unwrap();
    /// let _cotangents = loss.backward().unwrap();
    ///
    /// let grad = x.grad()?.unwrap();
    /// assert_eq!(grad.shape(), &[2]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeState`] if the gradient slot is poisoned or no
    /// longer available.
    pub fn grad(&self) -> Result<Option<Arc<Tensor>>> {
        self.grad_slot
            .lock()
            .map_err(|_| {
                Error::runtime_state(
                    "eager_gradient_slot",
                    ErrorPhase::Execution,
                    "lock poisoned",
                )
            })
            .map(|slot| slot.clone())
    }

    /// Clear the accumulated gradient stored for this tensor.
    ///
    /// This only affects this tensor's gradient slot. Other tensors in the
    /// same context retain their gradients until they are cleared explicitly or
    /// overwritten by later accumulation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]).unwrap(), ctx).unwrap();
    /// let loss = x.mul(&y).unwrap().reduce_sum(Some(&[0])).unwrap();
    /// let _ = loss.backward().unwrap();
    ///
    /// x.clear_grad()?;
    ///
    /// assert!(x.grad()?.is_none());
    /// assert!(y.grad()?.is_some());
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeState`] if the gradient slot lock is poisoned.
    pub fn clear_grad(&self) -> Result<()> {
        *self.grad_slot.lock().map_err(|_| {
            Error::runtime_state(
                "eager_gradient_slot",
                ErrorPhase::Execution,
                "lock poisoned",
            )
        })? = None;
        Ok(())
    }

    /// Report whether this tensor participates in gradient tracking.
    ///
    /// Tracked tensors keep a gradient slot in their eager context; untracked
    /// tensors and detached tensors do not.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let plain = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(), ctx.clone()).unwrap();
    /// let tracked = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(), ctx.clone()).unwrap();
    /// let detached = tracked.detach();
    ///
    /// assert!(!plain.tracks_grad());
    /// assert!(tracked.tracks_grad());
    /// assert!(!detached.tracks_grad());
    /// ```
    pub fn tracks_grad(&self) -> bool {
        self.requires_grad
    }

    #[cfg(test)]
    fn debug_trace_saved_value_count(&self) -> Option<usize> {
        self.trace.as_ref().map(|trace| trace.saved_values().len())
    }

    /// Return the opaque identifier of the context this tensor belongs to.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(), ctx.clone()).unwrap();
    ///
    /// assert_eq!(x.ctx_id(), ctx.id());
    /// ```
    pub fn ctx_id(&self) -> ContextId {
        self.ctx.id()
    }

    /// Borrow the eager runtime context that owns this tensor.
    pub fn runtime(&self) -> &Arc<EagerRuntime> {
        &self.ctx
    }

    /// Check whether two tensors belong to the same eager context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(), ctx.clone()).unwrap();
    /// let y = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap(), ctx).unwrap();
    ///
    /// assert!(x.same_context(&y));
    /// ```
    pub fn same_context(&self, other: &Self) -> bool {
        self.ctx_id() == other.ctx_id()
    }

    #[cfg(test)]
    pub(crate) fn standard_graph_op(
        inputs: &[&Self],
        build_graph: impl FnOnce(&[TensorInputKey]) -> Result<Arc<Graph<StdTensorOp>>>,
    ) -> Result<Vec<Self>> {
        let Some(first) = inputs.first() else {
            return Err(Error::Internal(
                "standard eager graph op requires at least one input tensor".to_string(),
            ));
        };
        let ctx = Arc::clone(&first.ctx);
        for tensor in inputs.iter().skip(1) {
            if !first.same_context(tensor) {
                return Err(Error::ContextMismatch {
                    lhs: first.ctx_id(),
                    rhs: tensor.ctx_id(),
                });
            }
        }

        let mut recorder = Recorder::new(EagerTensorKeySource);
        let graph_input_keys = recorder.fresh_input_keys::<StdTensorOp>(inputs.len());
        let graph = build_graph(&graph_input_keys)?;
        let initial_data = graph_input_keys
            .iter()
            .zip(inputs.iter())
            .map(|(key, tensor)| Ok((ValueKey::Input(key.clone()), tensor.materialized_arc()?)))
            .collect::<Result<HashMap<_, _>>>()?;
        let execution = ctx.exec_standard_graph_outputs(graph.as_ref(), &initial_data)?;
        if execution.outputs.len() != graph.outputs().len() {
            return Err(Error::Internal(format!(
                "standard eager graph op expected {} graph outputs, got {}",
                graph.outputs().len(),
                execution.outputs.len()
            )));
        }

        if !eager_grad_recording_enabled() || !inputs.iter().any(|input| input.requires_grad) {
            return execution
                .outputs
                .into_iter()
                .map(|output| {
                    Self::new_result_arc(
                        Arc::clone(&ctx),
                        eager_val_key(),
                        output,
                        false,
                        None,
                        Vec::new(),
                    )
                })
                .collect();
        }

        let output_keys = graph
            .outputs()
            .iter()
            .map(|&output_id| graph.values()[output_id].key.clone())
            .collect();
        let recorded_graph = RecordedGraph::new(Arc::clone(&graph), graph_input_keys, output_keys)
            .map_err(eager_record_error)?;
        let recorded = record_eager_recorded_graph_outputs(
            &mut recorder,
            recorded_graph,
            &execution.outputs,
            execution.retained_values,
            inputs,
        )?;
        if recorded.traces.len() != execution.outputs.len() {
            return Err(Error::Internal(format!(
                "standard eager graph op expected {} eager traces, got {}",
                execution.outputs.len(),
                recorded.traces.len()
            )));
        }

        let mut metadata_scopes = vec![Arc::clone(&recorded.metadata_scope)];
        for input in inputs {
            for scope in &input.metadata_scopes {
                push_metadata_scope(&mut metadata_scopes, Arc::clone(scope));
            }
        }

        recorded
            .traces
            .into_iter()
            .zip(execution.outputs)
            .map(|(trace, output)| {
                Self::new_result_arc(
                    Arc::clone(&ctx),
                    trace.key,
                    output,
                    trace.requires_grad,
                    trace.trace,
                    metadata_scopes.clone(),
                )
            })
            .collect()
    }

    /// Run reverse-mode AD from this scalar output.
    ///
    /// Returns the full cotangent map produced by the reverse pass and also
    /// accumulates into `grad()` for tracked eager tensors reachable from this
    /// output.
    ///
    /// For complex scalar outputs, cotangents use tenferro's Hermitian
    /// real-inner-product convention. See
    /// <https://tensor4all.org/tenferro-rs/guides/complex-ad.html>.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap(), ctx).unwrap();
    /// let loss = x.add(&x).unwrap().reduce_sum(Some(&[0])).unwrap();
    /// let _cotangents = loss.backward().unwrap();
    /// let loss = x.add(&x).unwrap().reduce_sum(Some(&[0])).unwrap();
    /// let _cotangents = loss.backward().unwrap();
    ///
    /// assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[4.0, 4.0, 4.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::NonScalarGrad`] when this output is not scalar,
    /// [`Error::UnsupportedAdRule`] when a graph operation lacks a reverse rule,
    /// or a typed validation/backend/runtime-state error during the reverse pass.
    pub fn backward(&self) -> Result<HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>> {
        if !self.shape().is_empty() {
            return Err(Error::NonScalarGrad {
                shape: self.shape().to_vec(),
            });
        }

        let value = self.materialized_arc()?;
        let seed = {
            let mut backend = self.ctx.lock_backend()?;
            Arc::new(one_like_tensor(value.as_ref(), &mut *backend)?)
        };
        self.backward_from_seed(seed)
    }

    /// Run reverse-mode AD from this output with an explicit cotangent seed.
    ///
    /// This is the stateful eager VJP sugar: it returns the cotangent map and
    /// accumulates reachable tracked leaves into their `grad()` slots. Use
    /// [`EagerRuntime::vjp`] when the VJP result should be returned as a
    /// composable eager tensor without touching grad slots.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::requires_grad_in(
    ///     Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
    ///     ctx.clone(),
    /// )?;
    /// let seed = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
    ///     ctx,
    /// )?;
    /// let y = x.mul(&x)?;
    /// y.backward_with(&seed)?;
    /// assert_eq!(x.grad()?.unwrap().as_slice::<f64>().unwrap(), &[4.0, 12.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::ContextMismatch`] when `cotangent` belongs to another
    /// eager runtime, [`Error::Validation`] when its shape or dtype is not a
    /// valid seed, [`Error::UnsupportedAdRule`] for an unavailable reverse
    /// rule, or a typed backend/runtime-state error during execution.
    pub fn backward_with(
        &self,
        cotangent: &EagerTensor,
    ) -> Result<HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>> {
        if !self.same_context(cotangent) {
            return Err(Error::ContextMismatch {
                lhs: self.ctx_id(),
                rhs: cotangent.ctx_id(),
            });
        }
        validate_seed_tensor("backward", self, cotangent)?;
        self.backward_from_seed(cotangent.materialized_arc()?)
    }

    fn backward_from_seed(
        &self,
        seed: Arc<Tensor>,
    ) -> Result<HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>> {
        let mut backend = self.ctx.lock_backend()?;
        let mut extension_executor = self.ctx.lock_extension_executor()?;
        let mut callbacks = TenferroBackwardCallbacks::with_runtime(
            &mut *backend,
            Some(&mut *extension_executor),
            Some(self.ctx.as_ref()),
            self.metadata_scopes.clone(),
        );
        let mut ad_ctx = ShapeGuardContext::with_global_metadata();
        if let Some(dispatcher) = &self.ctx.extension_ad_dispatcher {
            ad_ctx = ad_ctx.with_extension_ad_dispatcher(Arc::clone(dispatcher));
        }
        let cotangents_result = eager::backward(
            &self.key,
            self.trace.as_ref(),
            seed,
            &mut callbacks,
            &mut ad_ctx,
        );
        let callback_typed_error = callbacks.take_typed_error();
        let callback_error = callbacks.take_error();
        drop(callbacks);
        if let Some(err) = callback_typed_error {
            return Err(err);
        }
        let cotangents = match (cotangents_result, callback_error) {
            (_, Some(err)) => {
                return Err(crate::ad_rule_error::ad_rule_error_with_context(
                    "backward",
                    err,
                    &mut ad_ctx,
                ));
            }
            (Err(err), None) => {
                return Err(crate::ad_rule_error::ad_rule_error_with_context(
                    "backward",
                    err,
                    &mut ad_ctx,
                ));
            }
            (Ok(cotangents), None) => cotangents,
        };
        self.ctx.store_grads(&cotangents, &mut backend)?;
        Ok(cotangents)
    }
}

pub(crate) fn eager_val_key() -> ValueKey<StdTensorOp> {
    ValueKey::Input(next_input_key())
}

pub(crate) struct EagerTensorKeySource;

impl KeySource<StdTensorOp> for EagerTensorKeySource {
    fn fresh_input_key(&mut self) -> TensorInputKey {
        next_input_key()
    }
}

pub(crate) fn eager_value(tensor: &EagerTensor) -> Result<EagerInput<StdTensorOp>> {
    Ok(EagerInput {
        key: tensor.key.clone(),
        trace: tensor.trace.clone(),
        requires_grad: tensor.requires_grad,
        data: tensor.materialized_arc()?,
    })
}

pub(crate) struct RecordedEagerOutputs {
    pub(crate) traces: Vec<EagerOutput<StdTensorOp>>,
    pub(crate) metadata_scope: Arc<GlobalMetadataScope>,
}

pub(crate) fn record_eager_outputs(
    op: &StdTensorOp,
    outputs: &[Arc<Tensor>],
    inputs: &[&EagerTensor],
) -> Result<RecordedEagerOutputs> {
    let mut recorder = Recorder::new(EagerTensorKeySource);
    let graph_input_keys = recorder.fresh_input_keys::<StdTensorOp>(inputs.len());
    let graph =
        RecordedGraph::from_primitive(op.clone(), graph_input_keys).map_err(eager_record_error)?;
    let retained_values = graph
        .output_keys()
        .iter()
        .cloned()
        .zip(outputs.iter().cloned())
        .collect();
    record_eager_recorded_graph_outputs(&mut recorder, graph, outputs, retained_values, inputs)
}

pub(crate) fn record_eager_recorded_graph_outputs(
    recorder: &mut Recorder<EagerTensorKeySource>,
    graph: RecordedGraph<StdTensorOp>,
    outputs: &[Arc<Tensor>],
    retained_values: HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
    inputs: &[&EagerTensor],
) -> Result<RecordedEagerOutputs> {
    let input_values: Vec<_> = inputs
        .iter()
        .map(|tensor| eager_value(tensor))
        .collect::<Result<_>>()?;
    let traces = recorder
        .record_graph(graph, &input_values, outputs, retained_values)
        .map_err(eager_record_error)?;

    let mut registrations = Vec::new();
    for trace in &traces {
        if let Some(output) = outputs.get(trace.output_slot) {
            registrations.push((trace.key.clone(), tensor_meta_from_tensor(output.as_ref())));
        }
    }

    if let Some(trace) = traces.iter().find_map(|output| output.trace.as_ref()) {
        for (key, value) in trace.saved_values() {
            registrations.push((key.clone(), tensor_meta_from_tensor(value.as_ref())));
        }
    }

    Ok(RecordedEagerOutputs {
        traces,
        metadata_scope: Arc::new(register_scoped_metadata_batch(registrations)?),
    })
}

fn eager_record_error(err: tidu::eager::EagerRecordError) -> Error {
    Error::runtime_state_source("eager recording", ErrorPhase::GraphBuild, err)
}

fn eager_ad_transform_cache_error(message: impl ToString) -> ADRuleError {
    ADRuleError::invalid_input(
        "tenferro-ad.eager.transform-cache",
        ADRuleKind::Jvp,
        message.to_string(),
    )
}

pub(crate) fn exec_single_output(
    op: &StdTensorOp,
    inputs: &[&Tensor],
    ctx: &EagerRuntime,
) -> Result<Tensor> {
    let mut outputs = ctx.exec_outputs(op, inputs)?;
    if outputs.len() != 1 {
        return Err(Error::Internal(format!(
            "expected one eager output for {:?}, got {}",
            op,
            outputs.len()
        )));
    }
    Ok(profile_eager_op_section(
        "exec_single_output.remove_output",
        || outputs.remove(0),
    ))
}

pub(crate) fn exec_single_output_read(
    op: &StdTensorOp,
    inputs: &[TensorRead<'_>],
    ctx: &EagerRuntime,
) -> Result<Tensor> {
    let mut outputs = ctx.exec_outputs_read(op, inputs)?;
    if outputs.len() != 1 {
        return Err(Error::Internal(format!(
            "expected one eager output for {:?}, got {}",
            op,
            outputs.len()
        )));
    }
    Ok(profile_eager_op_section(
        "exec_single_output_read.remove_output",
        || outputs.remove(0),
    ))
}

pub(crate) fn zero_like_tensor<B: TensorBackend>(
    input: &Tensor,
    backend: &mut B,
) -> Result<Tensor> {
    let host = match input {
        Tensor::F32(tensor) => Tensor::F32(TypedTensor::zeros(tensor.shape().to_vec())?),
        Tensor::F64(tensor) => Tensor::F64(TypedTensor::zeros(tensor.shape().to_vec())?),
        Tensor::I32(tensor) => Tensor::I32(TypedTensor::zeros(tensor.shape().to_vec())?),
        Tensor::I64(tensor) => Tensor::I64(TypedTensor::zeros(tensor.shape().to_vec())?),
        Tensor::Bool(tensor) => Tensor::Bool(TypedTensor::from_vec_col_major(
            tensor.shape().to_vec(),
            vec![false; tensor.n_elements()],
        )?),
        Tensor::C32(tensor) => Tensor::C32(TypedTensor::zeros(tensor.shape().to_vec())?),
        Tensor::C64(tensor) => Tensor::C64(TypedTensor::zeros(tensor.shape().to_vec())?),
    };
    backend.upload_host_tensor(&host).map_err(Error::from)
}

pub(crate) fn one_like_tensor<B: TensorBackend>(input: &Tensor, backend: &mut B) -> Result<Tensor> {
    let host = ones_tensor(input.dtype(), input.shape().to_vec())?;
    backend.upload_host_tensor(&host).map_err(Error::from)
}

#[cfg(test)]
mod tests;
