use std::cell::{Cell, RefCell};
use std::cmp::Reverse;
use std::collections::HashMap;
use std::env;
use std::fmt;
use std::marker::PhantomData;
use std::mem::{size_of, size_of_val};
use std::rc::Rc;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};
use std::time::{Duration, Instant};

use lru::LruCache;

use crate::extension::{
    validate_eager_extension_target, EagerExtensionBackendKind, EagerExtensionTarget,
};
use crate::extension_cache::{ExtensionCacheLimits, ExtensionCacheSelector, ExtensionCacheStore};
#[cfg(test)]
use computegraph::graph::Graph;
use computegraph::ValueKey;
#[cfg(test)]
use computegraph::ValueRef;
use tenferro_cpu::{CpuBackend, CpuBackendError, CpuPlacement};
#[cfg(feature = "cuda")]
use tenferro_gpu::cuda::CudaBackend;
#[cfg(feature = "webgpu")]
use tenferro_gpu::webgpu::WebGpuBackend;
#[cfg(test)]
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::{std_tensor_op::StdTensorOp, SymDim, TensorMeta};
use tenferro_runtime::ad_support::{compile_ad_source, ones_tensor, RetainedValue};
use tenferro_runtime::program::{ProgramValueMetadata, SemanticFingerprint, SemanticProgram};
use tenferro_runtime::{
    CompiledGraph, CoreCapabilityBundle, EngineId, ErrorPhase, ExecutionContextIdentity,
    ExtensionModule, GraphCompiler, HardwareClassId, PreparedCompiledGraph, RegistrationIdentity,
    Runtime, RuntimeConfigError, RuntimeConfigSnapshot, RuntimeEpoch, TracedTensor,
};
#[cfg(test)]
use tenferro_tensor::TypedTensor;
use tenferro_tensor::{
    AllocationGroup, CacheStats, DType, DescriptorSlot, GroupError, IntoShapeVec, Tensor,
    TensorBackend, TensorRead, TensorScalar, TensorValue, TensorView,
};
use tenferro_tensor::{BackendSession, BackendSessionHost};

#[cfg(feature = "cuda")]
use crate::eager_backend::cuda_runtime_engine_id;
use crate::eager_backend::{
    cpu_runtime_engine_id, cpu_runtime_hardware_class, eager_runtime_for_backend, EagerBackend,
};
#[cfg(test)]
use crate::eager_exec::exec_standard_op_on_tensor_reads_in_session;
use crate::eager_exec::{exec_op_on_tensor_reads_with_runtime, exec_op_on_tensors_with_runtime};
use crate::error::{ContextId, Error, Result};
#[cfg(test)]
use crate::metadata::push_metadata_scope;
use crate::metadata::{
    metadata_scopes_for_scope, register_scoped_metadata_batch, register_scoped_value_metadata,
    tensor_meta_from_tensor, GlobalMetadataScope,
};
use crate::semantic_extension::SemanticExtensionRuleSet;
use crate::traced::{derivative_trace_from_frozen_program, next_input_key};
use crate::transform_cache::{AdTransformCache, AdTransformCacheLimits};

use crate::AdContext;

pub(crate) type GradSlot = Arc<Mutex<Option<Arc<AdValueRecord>>>>;
pub(crate) type WeakGradSlot = Weak<Mutex<Option<Arc<AdValueRecord>>>>;

#[derive(Clone, Debug)]
pub(crate) struct EagerTrace;

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
    static EAGER_CAPTURE_DEPTH: Cell<usize> = const { Cell::new(0) };
    #[cfg(test)]
    static EAGER_OP_PROFILE_ENABLED_OVERRIDE: RefCell<Option<bool>> = const { RefCell::new(None) };
    #[cfg(test)]
    static EAGER_OP_PROFILE_PRINT_EVERY_OVERRIDE: RefCell<Option<Option<usize>>> = const { RefCell::new(None) };
    #[cfg(test)]
    static EAGER_SEMANTIC_VJP_ENABLED_OVERRIDE: RefCell<Option<bool>> = const { RefCell::new(None) };
}

#[cfg(test)]
pub(crate) static EAGER_SEMANTIC_VJP_EXECUTIONS: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn eager_grad_recording_enabled() -> bool {
    EAGER_NO_GRAD_DEPTH.with(|depth| depth.get() == 0)
}

pub(crate) fn eager_capture_active() -> bool {
    EAGER_CAPTURE_DEPTH.with(|depth| depth.get() > 0)
}

fn eager_semantic_vjp_enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = EAGER_SEMANTIC_VJP_ENABLED_OVERRIDE.with(|state| *state.borrow()) {
        return value;
    }

    // Semantic eager VJP/JVP on by default (Unification 7).
    // Set TENFERRO_EAGER_SEMANTIC_VJP=0 to disable.
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env::var("TENFERRO_EAGER_SEMANTIC_VJP").map_or(true, |v| v != "0"))
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
/// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
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
    // Thread-local depth guard: must not be Send so it cannot be moved to and
    // dropped on another thread (which would corrupt the creator's depth).
    _not_send: PhantomData<Rc<()>>,
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

/// Scope guard that keeps semantic-trace recording active for untracked
/// intermediates.
///
/// Under active-edge semantics (issue #1665 Def 1), an operation whose inputs
/// are all untracked produces no autograd nodes and drops its semantic trace.
/// Inside this guard, such operations still record their semantic trace, so a
/// later functional JVP/VJP can differentiate with respect to an untracked or
/// detached leaf. This replaces the pre-Def-1 implicit recording.
///
/// # Examples
///
/// ```
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
/// use tenferro_cpu::CpuBackend;
///
/// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
/// let x = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
///     ctx.clone(),
/// )?;
/// let (y, x) = {
///     let _capture = ctx.capture_trace();
///     let y = x.mul(&x)?;
///     (y, x)
/// };
/// let seed = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 1.0]).unwrap(),
///     ctx.clone(),
/// )?;
/// let dx = ctx.vjp(&y, &x, &seed)?;
/// assert_eq!(dx.value()?.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
#[derive(Debug)]
pub struct EagerTraceCaptureGuard {
    active: bool,
    // Thread-local depth guard: must not be Send so it cannot be moved to and
    // dropped on another thread (which would corrupt the creator's depth).
    _not_send: PhantomData<Rc<()>>,
}

impl Drop for EagerTraceCaptureGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        EAGER_CAPTURE_DEPTH.with(|depth| {
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
    /// Prepared eager derivative program cache.
    pub prepared_derivatives: CacheStats,
}

#[cfg(test)]
pub(crate) struct EagerGraphExecution {
    pub(crate) outputs: Vec<Tensor>,
}

/// A read-only value view retained by an eager tensor record.
///
/// The guard borrows the record's allocation group. It never owns a tensor and
/// cannot be converted into a mutable view.
///
/// # Examples
///
/// ```
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
/// use tenferro_cpu::CpuBackend;
///
/// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
/// let value = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?,
///     ctx,
/// )?;
/// let view = value.value()?;
/// assert_eq!(view.shape(), &[2]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
#[derive(Debug)]
pub struct ValueGuard<'a> {
    view: TensorView<'a>,
}

impl<'a> ValueGuard<'a> {
    /// Return the scalar dtype of the retained value.
    pub fn dtype(&self) -> DType {
        self.view.dtype()
    }

    /// Return the logical shape of the retained value.
    pub fn shape(&self) -> &[usize] {
        self.view.shape()
    }

    /// Borrow the dtype-erased tensor view.
    pub fn as_tensor_view(&self) -> &TensorView<'_> {
        &self.view
    }

    /// Borrow compact host bytes through the tensor's explicit scalar type.
    ///
    /// Backend-resident values return the backend's typed host-access error;
    /// this method does not download storage implicitly.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::ValidationError::DTypeMismatch`] when
    /// `T` does not match the view dtype, [`tenferro_tensor::ValidationError::NonContiguousViewAsSlice`]
    /// for a non-contiguous view, or [`tenferro_tensor::Error::HostAccess`]
    /// when backend storage cannot be mapped as a host slice.
    pub fn as_slice<T: TensorScalar>(&self) -> tenferro_tensor::Result<&'a [T]> {
        self.view.as_slice()
    }

    fn duplicate_host_tensor(&self) -> tenferro_tensor::Result<Tensor> {
        match &self.view {
            TensorView::F32(view) => {
                <f32 as TensorScalar>::into_tensor(view.shape().to_vec(), view.as_slice()?.to_vec())
            }
            TensorView::F64(view) => {
                <f64 as TensorScalar>::into_tensor(view.shape().to_vec(), view.as_slice()?.to_vec())
            }
            TensorView::I32(view) => {
                <i32 as TensorScalar>::into_tensor(view.shape().to_vec(), view.as_slice()?.to_vec())
            }
            TensorView::I64(view) => {
                <i64 as TensorScalar>::into_tensor(view.shape().to_vec(), view.as_slice()?.to_vec())
            }
            TensorView::Bool(view) => <bool as TensorScalar>::into_tensor(
                view.shape().to_vec(),
                view.as_slice()?.to_vec(),
            ),
            TensorView::C32(view) => <num_complex::Complex32 as TensorScalar>::into_tensor(
                view.shape().to_vec(),
                view.as_slice()?.to_vec(),
            ),
            TensorView::C64(view) => <num_complex::Complex64 as TensorScalar>::into_tensor(
                view.shape().to_vec(),
                view.as_slice()?.to_vec(),
            ),
        }
    }
}

/// Read-only retained gradient value.
///
/// # Examples
///
/// ```
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
/// use tenferro_cpu::CpuBackend;
///
/// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
/// let x = EagerTensor::requires_grad_in(
///     Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0])?,
///     ctx,
/// )?;
/// let loss = x.mul(&x)?.reduce_sum(Some(&[0]))?;
/// let _gradients = loss.backward()?;
/// let gradient = x.grad()?.expect("tracked leaf has a gradient");
/// assert_eq!(gradient.shape(), &[2]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct GradientValue {
    record: Arc<AdValueRecord>,
    ctx: Arc<EagerRuntime>,
}

impl GradientValue {
    /// Return the scalar dtype of the gradient.
    pub fn dtype(&self) -> DType {
        self.record.dtype()
    }

    /// Return the logical shape of the gradient.
    pub fn shape(&self) -> &[usize] {
        self.record.shape()
    }

    /// Borrow the gradient's value guard.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeState`] when the retained gradient record is
    /// unavailable or its allocation-group descriptor is invalid.
    pub fn value(&self) -> Result<ValueGuard<'_>> {
        self.record.value("GradientValue::value")
    }

    /// Borrow the gradient as a dtype-erased read target.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeState`] when the retained gradient record or
    /// its allocation-group descriptor is unavailable.
    pub fn tensor_read(&self) -> Result<TensorRead<'_>> {
        self.record.tensor_read("GradientValue::tensor_read")
    }

    /// Borrow a compact host slice without downloading backend storage.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeState`] when the retained value is unavailable,
    /// [`tenferro_tensor::ValidationError::DTypeMismatch`] when `T` does
    /// not match the gradient dtype, or [`tenferro_tensor::Error::HostAccess`]
    /// when backend storage cannot be mapped as a host slice.
    pub fn as_slice<T: TensorScalar>(&self) -> tenferro_tensor::Result<&[T]> {
        self.record
            .value("GradientValue::as_slice")
            .map_err(|error| {
                tenferro_tensor::Error::runtime_state_source("GradientValue::as_slice", error)
            })?
            .as_slice()
    }

    /// Explicitly copy a host-resident gradient into a standalone tensor.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeState`] when the retained value or execution
    /// session is unavailable, or a typed backend/host-access error when the
    /// gradient cannot be materialized as a contiguous tensor.
    pub fn to_tensor(&self) -> Result<Tensor> {
        let value = self
            .record
            .value("GradientValue::to_tensor")
            .map_err(|error| {
                Error::runtime_state_source(
                    "GradientValue::to_tensor",
                    ErrorPhase::Execution,
                    error,
                )
            })?;
        match value.duplicate_host_tensor() {
            Ok(tensor) => Ok(tensor),
            Err(_) => {
                let read = self.record.tensor_read("GradientValue::to_tensor")?;
                self.ctx
                    .with_execution_session(|session| session.to_contiguous_read(read))?
                    .map_err(Error::from)
            }
        }
    }
}

/// Move-only accumulated gradient bundle backed by one allocation group.
///
/// # Examples
///
/// ```
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
/// use tenferro_cpu::CpuBackend;
///
/// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
/// let x = EagerTensor::requires_grad_in(
///     Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0])?,
///     ctx,
/// )?;
/// let loss = x.mul(&x)?.reduce_sum(Some(&[0]))?;
/// let gradients = loss.backward()?;
/// assert!(!gradients.is_empty());
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
#[derive(Debug)]
pub struct Gradients {
    group: AllocationGroup,
    slots: HashMap<ValueKey<StdTensorOp>, DescriptorSlot>,
}

impl Gradients {
    fn from_tensors(tensors: HashMap<ValueKey<StdTensorOp>, Tensor>) -> Result<Self> {
        let (keys, values): (Vec<_>, Vec<_>) = tensors.into_iter().unzip();
        let (group, bindings) = AllocationGroup::from_tensors(values).map_err(|error| {
            Error::runtime_state_source("Gradients::from_tensors", ErrorPhase::Execution, error)
        })?;
        let slots = keys.into_iter().zip(bindings).collect();
        Ok(Self { group, slots })
    }

    /// Return the number of retained gradient descriptors.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Return whether no gradient was produced.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Borrow one gradient view by its local value key.
    pub fn grad(&self, key: &ValueKey<StdTensorOp>) -> Option<TensorView<'_>> {
        let slot = self.slots.get(key).copied()?;
        let mut reads = self.group.read_views(std::slice::from_ref(&slot)).ok()?;
        match reads.pop()? {
            TensorRead::View(view) => Some(view),
            TensorRead::Tensor(_) => None,
        }
    }

    /// Consume one gradient owner while leaving the bundle unchanged on failure.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::RuntimeState`] when the descriptor is
    /// invalid or its allocation is aliased. A missing key is reported as
    /// `Ok(None)`.
    pub fn take_grad(
        &mut self,
        key: &ValueKey<StdTensorOp>,
    ) -> tenferro_tensor::Result<Option<Tensor>> {
        let Some(&slot) = self.slots.get(key) else {
            return Ok(None);
        };
        let tensor = self.group.take_tensor(slot).map_err(|error| {
            tenferro_tensor::Error::runtime_state_source("Gradients::take_grad", error)
        })?;
        self.slots.remove(key);
        Ok(Some(tensor))
    }
}

/// Error returned when a value cannot be consumed without changing its owner.
///
/// # Examples
///
/// ```
/// use tenferro_ad::{EagerRuntime, EagerTensor, IntoValueError, Tensor};
/// use tenferro_cpu::CpuBackend;
///
/// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
/// let value = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?,
///     ctx,
/// )?;
/// let _shared = value.clone();
/// assert!(matches!(
///     value.into_value(),
///     Err(IntoValueError::NotUnique(_))
/// ));
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
#[derive(Debug)]
pub enum IntoValueError<H> {
    /// Another eager handle, tape record, or checkpoint retains the value.
    NotUnique(H),
    /// Group extraction failed after the handle was uniquely acquired.
    Extract { value: H, error: GroupError },
}

impl<H> std::fmt::Display for IntoValueError<H> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotUnique(_) => formatter.write_str("eager value is retained by another handle"),
            Self::Extract { error, .. } => {
                write!(formatter, "eager value extraction failed: {error}")
            }
        }
    }
}

impl<H: std::fmt::Debug + Send + Sync + 'static> std::error::Error for IntoValueError<H> {}

/// One direct retention container owns the physical allocation group.
#[derive(Debug)]
struct RetentionContainer {
    group: AllocationGroup,
}

/// Read-only descriptor record used by eager handles and the AD registries.
#[derive(Debug)]
pub(crate) struct AdValueRecord {
    container: Arc<RetentionContainer>,
    slot: DescriptorSlot,
    dtype: DType,
    shape: Box<[usize]>,
}

impl AdValueRecord {
    fn from_group(
        group: AllocationGroup,
        slot: DescriptorSlot,
        dtype: DType,
        shape: Vec<usize>,
    ) -> Arc<Self> {
        Arc::new(Self {
            container: Arc::new(RetentionContainer { group }),
            slot,
            dtype,
            shape: shape.into_boxed_slice(),
        })
    }

    fn from_tensor(tensor: Tensor, op: &'static str) -> Result<Arc<Self>> {
        let dtype = tensor.dtype();
        let shape = tensor.shape().to_vec();
        let (group, bindings) = AllocationGroup::from_tensors(vec![tensor])
            .map_err(|error| Error::runtime_state_source(op, ErrorPhase::Execution, error))?;
        let slot = bindings.first().copied().ok_or_else(|| {
            Error::runtime_state(op, ErrorPhase::Execution, "empty allocation-group binding")
        })?;
        Ok(Self::from_group(group, slot, dtype, shape))
    }

    fn tensor_read(&self, op: &'static str) -> Result<TensorRead<'_>> {
        let mut reads = self
            .container
            .group
            .read_views(std::slice::from_ref(&self.slot))
            .map_err(|error| Error::runtime_state_source(op, ErrorPhase::Execution, error))?;
        reads.pop().ok_or_else(|| {
            Error::runtime_state(op, ErrorPhase::Execution, "empty allocation-group binding")
        })
    }

    fn value(&self, op: &'static str) -> Result<ValueGuard<'_>> {
        match self.tensor_read(op)? {
            TensorRead::View(view) => Ok(ValueGuard { view }),
            TensorRead::Tensor(_) => Err(Error::runtime_state(
                op,
                ErrorPhase::Execution,
                "allocation-group value did not produce a borrowed descriptor view",
            )),
        }
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }
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
/// let runtime = EagerRuntime::new()?;
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
    /// let runtime = EagerRuntime::new()?;
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
    /// let runtime = EagerRuntime::new()?;
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
    /// let runtime = EagerRuntime::new()?;
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
/// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
/// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(), ctx.clone()).unwrap();
/// let y = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap(), ctx).unwrap();
/// let z = x.add(&y).unwrap();
///
/// assert_eq!(z.value().unwrap().as_slice::<f64>().unwrap(), &[3.0]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
pub struct EagerRuntime {
    id: ContextId,
    runtime: Runtime,
    // The backend and its exact runtime engine registration are selected
    // together during construction and remain paired for this runtime's
    // lifetime. The mutex only serializes mutable backend operations.
    backend: Mutex<EagerBackend>,
    extension_install_lock: Mutex<()>,
    pub(crate) extension_caches: Mutex<ExtensionCacheStore>,
    semantic_extension_rules: SemanticExtensionRuleSet,
    grad_slots: Mutex<HashMap<ValueKey<StdTensorOp>, WeakGradSlot>>,
    value_records: Mutex<HashMap<ValueKey<StdTensorOp>, Weak<EagerTensorRecord>>>,
    ad_transform_cache: Arc<AdTransformCache>,
    /// S2: prepared derivative programs keyed by semantic structure, wrt input,
    /// and concrete bound input metadata. Avoids re-running freeze+AD
    /// transform+compile_frozen on warm structure hits.
    prepared_derivative_cache: Mutex<PreparedDerivativeCache>,
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
        match self.extension_caches.try_lock() {
            Ok(caches) => {
                debug.field(
                    "extension_cache_stats",
                    &caches.stats(ExtensionCacheSelector::All),
                );
            }
            Err(_) => {
                debug.field("extension_cache_stats", &"<locked>");
            }
        }
        match self.extension_install_lock.try_lock() {
            Ok(_) => {
                debug.field("extension_install_lock", &"<unlocked>");
            }
            Err(_) => {
                debug.field("extension_install_lock", &"<locked>");
            }
        }
        debug.field("semantic_extension_rules", &self.semantic_extension_rules);
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
        match self.ad_transform_cache.stats() {
            Ok(stats) => {
                debug.field("ad_transform_cache_stats", &stats);
            }
            Err(err) => {
                debug.field("ad_transform_cache_stats", &format_args!("{err}"));
            }
        }
        match self.prepared_derivative_cache.try_lock() {
            Ok(cache) => {
                debug.field("prepared_derivative_cache_stats", &cache.stats());
            }
            Err(_) => {
                debug.field("prepared_derivative_cache_stats", &"<locked>");
            }
        }
        debug.finish_non_exhaustive()
    }
}

impl EagerRuntime {
    pub(crate) fn lock_backend(&self) -> Result<MutexGuard<'_, EagerBackend>> {
        self.backend.lock().map_err(|_| {
            Error::runtime_state("eager_backend", ErrorPhase::Execution, "lock poisoned")
        })
    }

    fn lock_extension_caches(&self) -> Result<MutexGuard<'_, ExtensionCacheStore>> {
        self.extension_caches.lock().map_err(|_| {
            Error::runtime_state(
                "eager_extension_caches",
                ErrorPhase::Execution,
                "lock poisoned",
            )
        })
    }

    fn lock_extension_install(&self) -> Result<MutexGuard<'_, ()>> {
        self.extension_install_lock.lock().map_err(|_| {
            Error::runtime_state(
                "eager_extension_install",
                ErrorPhase::Execution,
                "lock poisoned",
            )
        })
    }

    fn lock_prepared_derivative_cache(&self) -> Result<MutexGuard<'_, PreparedDerivativeCache>> {
        self.prepared_derivative_cache.lock().map_err(|_| {
            Error::runtime_state(
                "prepared_derivative_cache",
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

    fn from_backend(backend: EagerBackend) -> Result<Self> {
        Self::from_backend_with_rules_and_cache(
            backend,
            SemanticExtensionRuleSet::default(),
            Arc::new(AdTransformCache::new()),
        )
    }

    fn from_backend_with_rules_and_cache(
        backend: EagerBackend,
        semantic_extension_rules: SemanticExtensionRuleSet,
        ad_transform_cache: Arc<AdTransformCache>,
    ) -> Result<Self> {
        let runtime = eager_runtime_for_backend(&backend)
            .map_err(|source| runtime_config_error("EagerRuntime::from_backend", source))?;
        Ok(Self {
            id: ContextId::fresh(),
            runtime,
            backend: Mutex::new(backend),
            extension_install_lock: Mutex::new(()),
            extension_caches: Mutex::new(ExtensionCacheStore::new()),
            semantic_extension_rules,
            grad_slots: Mutex::new(HashMap::new()),
            value_records: Mutex::new(HashMap::new()),
            ad_transform_cache,
            prepared_derivative_cache: Mutex::new(PreparedDerivativeCache::default()),
        })
    }

    /// Create a shared CPU eager execution context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ad::EagerRuntime;
    ///
    /// let ctx = EagerRuntime::new()?;
    /// assert_eq!(std::sync::Arc::strong_count(&ctx), 1);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when provider runtime
    /// registration cannot be configured, preserving the underlying
    /// [`RuntimeConfigError`] as the typed error source.
    pub fn new() -> Result<Arc<Self>> {
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1)?)?;
    /// assert_eq!(std::sync::Arc::strong_count(&ctx), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when provider runtime
    /// registration cannot be configured, preserving the underlying
    /// [`RuntimeConfigError`] as the typed error source.
    pub fn with_cpu_backend(backend: CpuBackend) -> Result<Arc<Self>> {
        Ok(Arc::new(Self::from_backend(EagerBackend::cpu(backend))?))
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
    /// let runtime = EagerRuntime::new()?;
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
    /// let ctx = EagerRuntime::with_cpu_backend_and_ad_context(CpuBackend::new(), &ad)?;
    /// assert_eq!(std::sync::Arc::strong_count(&ctx), 1);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when provider runtime
    /// registration cannot be configured, preserving the underlying
    /// [`RuntimeConfigError`] as the typed error source.
    pub fn with_cpu_backend_and_ad_context(
        backend: CpuBackend,
        ad: &AdContext,
    ) -> Result<Arc<Self>> {
        Ok(Arc::new(Self::from_backend_with_rules_and_cache(
            EagerBackend::cpu(backend),
            ad.semantic_extension_rules().clone(),
            ad.ad_transform_cache(),
        )?))
    }

    /// Create a shared eager execution context from a configured CUDA backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cuda::CudaBackend;
    /// use tenferro_ad::EagerRuntime;
    ///
    /// let _ctor: fn(CudaBackend) -> tenferro_ad::Result<std::sync::Arc<EagerRuntime>> =
    ///     EagerRuntime::with_cuda_backend;
    /// ```
    #[cfg(feature = "cuda")]
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when provider runtime
    /// registration cannot be configured, preserving the underlying
    /// [`RuntimeConfigError`] as the typed error source.
    pub fn with_cuda_backend(backend: CudaBackend) -> Result<Arc<Self>> {
        Ok(Arc::new(Self::from_backend(EagerBackend::cuda(backend))?))
    }

    /// Create a shared CUDA eager context with explicit AD extension rules.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::{AdContext, EagerRuntime};
    /// use tenferro_gpu::cuda::CudaBackend;
    ///
    /// let _ctor: fn(CudaBackend, &AdContext) -> tenferro_ad::Result<std::sync::Arc<EagerRuntime>> =
    ///     EagerRuntime::with_cuda_backend_and_ad_context;
    /// ```
    #[cfg(feature = "cuda")]
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when provider runtime
    /// registration cannot be configured, preserving the underlying
    /// [`RuntimeConfigError`] as the typed error source.
    pub fn with_cuda_backend_and_ad_context(
        backend: CudaBackend,
        ad: &AdContext,
    ) -> Result<Arc<Self>> {
        Ok(Arc::new(Self::from_backend_with_rules_and_cache(
            EagerBackend::cuda(backend),
            ad.semantic_extension_rules().clone(),
            ad.ad_transform_cache(),
        )?))
    }

    /// Create a shared eager execution context from a configured WebGPU backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ad::EagerRuntime;
    /// use tenferro_gpu::webgpu::WebGpuBackend;
    ///
    /// let _ctor: fn(WebGpuBackend) -> tenferro_ad::Result<std::sync::Arc<EagerRuntime>> =
    ///     EagerRuntime::with_webgpu_backend;
    /// ```
    #[cfg(feature = "webgpu")]
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when provider runtime
    /// registration cannot be configured, preserving the underlying
    /// [`RuntimeConfigError`] as the typed error source.
    pub fn with_webgpu_backend(backend: WebGpuBackend) -> Result<Arc<Self>> {
        Ok(Arc::new(Self::from_backend(EagerBackend::webgpu(backend))?))
    }

    /// Create a shared WebGPU eager context with explicit AD extension rules.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::{AdContext, EagerRuntime};
    /// use tenferro_gpu::webgpu::WebGpuBackend;
    ///
    /// let _ctor: fn(WebGpuBackend, &AdContext) -> tenferro_ad::Result<std::sync::Arc<EagerRuntime>> =
    ///     EagerRuntime::with_webgpu_backend_and_ad_context;
    /// ```
    #[cfg(feature = "webgpu")]
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeStateSource`] when provider runtime
    /// registration cannot be configured, preserving the underlying
    /// [`RuntimeConfigError`] as the typed error source.
    pub fn with_webgpu_backend_and_ad_context(
        backend: WebGpuBackend,
        ad: &AdContext,
    ) -> Result<Arc<Self>> {
        Ok(Arc::new(Self::from_backend_with_rules_and_cache(
            EagerBackend::webgpu(backend),
            ad.semantic_extension_rules().clone(),
            ad.ad_transform_cache(),
        )?))
    }

    /// Return an opaque identifier for this context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// assert_ne!(ctx.id(), EagerRuntime::with_cpu_backend(CpuBackend::new())?.id());
    /// # Ok::<(), tenferro_ad::Error>(())
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
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
        EagerNoGradGuard {
            active: true,
            _not_send: PhantomData,
        }
    }

    /// Keep semantic-trace recording active for untracked intermediates.
    ///
    /// See [`EagerTraceCaptureGuard`] for the full contract and an example.
    pub fn capture_trace(&self) -> EagerTraceCaptureGuard {
        EAGER_CAPTURE_DEPTH.with(|depth| {
            depth.set(depth.get().saturating_add(1));
        });
        EagerTraceCaptureGuard {
            active: true,
            _not_send: PhantomData,
        }
    }

    /// Install or replace one extension module on this eager context's runtime.
    ///
    /// Eager extension wrappers call this as an idempotent "ensure installed"
    /// step. When the exact module instance (same module ID and allocation) is
    /// already installed, this is a read-only no-op that returns the current
    /// runtime epoch without acquiring the install lock or reconfiguring. The
    /// cold or replacement paths keep the transactional install-or-replace
    /// behavior, serialized so parallel first-use of the same extension family
    /// cannot publish over another thread's base snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] when runtime
    /// reconfiguration fails or the extension module transaction is invalid.
    pub fn install_extension_module(
        &self,
        module: Arc<dyn ExtensionModule>,
    ) -> Result<RuntimeEpoch> {
        let snapshot = self.runtime.snapshot().map_err(|source| {
            runtime_state_source("EagerRuntime::install_extension_module", source)
        })?;
        if snapshot.has_extension_module_identical(&module) {
            return Ok(snapshot.epoch());
        }
        let _install_guard = self.lock_extension_install()?;
        self.runtime
            .reconfigure(|edit| {
                edit.replace_extension_module(module)?;
                Ok(())
            })
            .map_err(|source| {
                runtime_state_source("EagerRuntime::install_extension_module", source)
            })
    }

    pub(crate) fn ensure_extension_module_for_engine(
        &self,
        module: Arc<dyn ExtensionModule>,
        family_id: &'static str,
        engine_id: &EngineId,
    ) -> Result<RuntimeEpoch> {
        let snapshot = self.runtime.snapshot().map_err(|source| {
            runtime_state_source("EagerRuntime::ensure_extension_module_for_engine", source)
        })?;
        if snapshot.has_extension_module_engine(module.module_id(), family_id, engine_id) {
            return Ok(snapshot.epoch());
        }
        let _install_guard = self.lock_extension_install()?;
        self.runtime
            .reconfigure(|edit| {
                edit.ensure_extension_module_for_engine(module, family_id, engine_id)?;
                Ok(())
            })
            .map_err(|source| {
                runtime_state_source("EagerRuntime::ensure_extension_module_for_engine", source)
            })
    }

    pub(crate) fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    pub(crate) fn eager_extension_target(&self) -> Result<EagerExtensionTarget> {
        let (engine_id, backend_kind) = {
            let backend = self.lock_backend()?;
            match &*backend {
                EagerBackend::Cpu(_) => (
                    cpu_runtime_engine_id().map_err(|source| {
                        runtime_config_error("EagerRuntime::eager_extension_target", source)
                    })?,
                    EagerExtensionBackendKind::Cpu,
                ),
                #[cfg(test)]
                EagerBackend::Recording(_) => {
                    return Err(Error::unsupported(
                        "EagerRuntime::eager_extension_target",
                        ErrorPhase::Execution,
                        "the recording backend has no registered eager extension engine",
                    ));
                }
                #[cfg(feature = "cuda")]
                EagerBackend::Cuda(_) => (
                    cuda_runtime_engine_id().map_err(|source| {
                        runtime_config_error("EagerRuntime::eager_extension_target", source)
                    })?,
                    EagerExtensionBackendKind::Cuda,
                ),
                #[cfg(feature = "webgpu")]
                EagerBackend::WebGpu(_) => (
                    tenferro_gpu::webgpu::webgpu_runtime_engine_id().map_err(|source| {
                        runtime_config_error("EagerRuntime::eager_extension_target", source)
                    })?,
                    EagerExtensionBackendKind::WebGpu,
                ),
            }
        };
        let target = EagerExtensionTarget {
            engine_id,
            backend_kind,
        };
        validate_eager_extension_target(&self.runtime, &target)?;
        Ok(target)
    }

    /// Clear generic extension runtime cache entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// ctx.clear_extension_caches()?;
    /// assert_eq!(ctx.cache_stats()?.extensions.entries, 0);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] when the extension
    /// cache lock is poisoned.
    pub fn clear_extension_caches(&self) -> Result<()> {
        self.lock_extension_caches()?.clear();
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// ctx.clear_caches()?;
    /// assert_eq!(ctx.cache_stats()?.extensions.entries, 0);
    /// assert_eq!(ctx.cache_stats()?.ad_transforms.entries, 0);
    /// assert_eq!(ctx.cache_stats()?.prepared_derivatives.entries, 0);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] when either the
    /// extension cache or AD-transform cache is poisoned.
    pub fn clear_caches(&self) -> Result<()> {
        self.clear_extension_caches()?;
        self.clear_ad_transform_caches()?;
        self.clear_prepared_derivative_cache()?;
        Ok(())
    }

    /// Clear prepared derivative program cache entries.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::EagerRuntime;
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// ctx.clear_prepared_derivative_cache()?;
    /// assert_eq!(ctx.cache_stats()?.prepared_derivatives.entries, 0);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] if the prepared
    /// derivative cache lock is poisoned.
    pub fn clear_prepared_derivative_cache(&self) -> Result<()> {
        self.lock_prepared_derivative_cache()?.clear();
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let stats = ctx.cache_stats()?;
    /// assert_eq!(stats.extensions.entries, 0);
    /// assert_eq!(stats.ad_transforms.entries, 0);
    /// assert_eq!(stats.prepared_derivatives.entries, 0);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] when a cache or
    /// AD-transform cache lock is poisoned.
    pub fn cache_stats(&self) -> Result<EagerRuntimeCacheStats> {
        Ok(EagerRuntimeCacheStats {
            extensions: self
                .lock_extension_caches()?
                .stats(ExtensionCacheSelector::All),
            ad_transforms: self.ad_transform_cache.stats()?,
            prepared_derivatives: self.lock_prepared_derivative_cache()?.stats(),
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
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

    /// Return prepared derivative cache retention limits.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::EagerRuntime;
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// assert!(ctx.prepared_derivative_cache_limits()?.max_entries().get() > 0);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] if the prepared
    /// derivative cache lock is poisoned.
    pub fn prepared_derivative_cache_limits(&self) -> Result<AdTransformCacheLimits> {
        Ok(self.lock_prepared_derivative_cache()?.limits())
    }

    /// Replace prepared derivative cache retention limits.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    /// use tenferro_ad::{AdTransformCacheLimits, EagerRuntime};
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let limits = AdTransformCacheLimits::new(NonZeroUsize::new(1).unwrap());
    /// ctx.set_prepared_derivative_cache_limits(limits)?;
    /// assert_eq!(ctx.prepared_derivative_cache_limits()?, limits);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] if the prepared
    /// derivative cache lock is poisoned.
    pub fn set_prepared_derivative_cache_limits(
        &self,
        limits: AdTransformCacheLimits,
    ) -> Result<()> {
        self.lock_prepared_derivative_cache()?.set_limits(limits);
        Ok(())
    }

    /// Return the extension cache retention limits.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] if the extension
    /// cache lock is poisoned.
    pub fn extension_cache_limits(&self) -> Result<ExtensionCacheLimits> {
        Ok(self.lock_extension_caches()?.limits())
    }

    /// Replace extension cache retention limits.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] if the extension
    /// cache lock is poisoned.
    pub fn set_extension_cache_limits(&self, limits: ExtensionCacheLimits) -> Result<()> {
        self.lock_extension_caches()?.set_limits(limits);
        Ok(())
    }

    /// Enter one backend execution session and run provider-neutral operations.
    ///
    /// The callback receives only a lifetime-bound, non-owning backend session.
    /// The backend and its engine registration are fixed when the eager runtime
    /// is constructed. Extension modules are installed separately and remain
    /// available to later extension operations.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ad::EagerRuntime;
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::{Tensor, TensorElementwise};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let lhs = Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
    /// let rhs = Tensor::from_vec_col_major(vec![1], vec![2.0_f64])?;
    /// let output = ctx.with_execution_session(|session| {
    ///     TensorElementwise::add(session, &lhs, &rhs)
    /// })??;
    /// assert_eq!(output.as_slice::<f64>()?, &[3.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] if the eager backend
    /// lock is poisoned. Backend operations retain their typed tensor/backend
    /// errors inside the callback result.
    pub fn with_execution_session<R: Send>(
        &self,
        f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
    ) -> Result<R> {
        let mut backend = self.lock_backend()?;
        Ok(backend.with_backend_session(f))
    }

    // Lock ordering: the eager backend owner is locked first; the
    // extension-cache lock is acquired only after it and remains held through
    // the borrowed session callback.
    /// Run an extension-owned eager operation with a borrowed backend session
    /// and the eager runtime's extension cache store.
    ///
    /// The eager backend owner is locked before the extension-cache lock is
    /// acquired. The callback receives an
    /// [`tenferro_runtime::ExtensionExecutionContext`] so cache access and
    /// backend execution share one lifetime-bound context without exposing the
    /// owning eager backend. The backend and its engine registration remain
    /// fixed for the eager runtime's lifetime.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ad::EagerRuntime;
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::{Tensor, TensorElementwise};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let lhs = Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
    /// let rhs = Tensor::from_vec_col_major(vec![1], vec![2.0_f64])?;
    /// let output = ctx.with_extension_execution_context(|extension_ctx| {
    ///     TensorElementwise::add(extension_ctx.backend_mut(), &lhs, &rhs)
    /// })??;
    /// assert_eq!(output.as_slice::<f64>()?, &[3.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] if the eager backend
    /// or extension-cache lock is poisoned. Errors returned by the callback
    /// remain in its result value.
    pub fn with_extension_execution_context<R: Send>(
        &self,
        f: impl FnOnce(
                &mut tenferro_runtime::ExtensionExecutionContext<'_, dyn BackendSession + '_>,
            ) -> R
            + Send,
    ) -> Result<R> {
        let mut backend = self.lock_backend()?;
        let mut extension_cache_guard = self.lock_extension_caches()?;
        let extension_caches: &mut ExtensionCacheStore = &mut extension_cache_guard;
        Ok(backend.with_backend_session(move |session| {
            let mut extension_ctx =
                tenferro_runtime::ExtensionExecutionContext::new(session, extension_caches);
            f(&mut extension_ctx)
        }))
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// ctx.synchronize().unwrap();
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::RuntimeState`] if the backend lock is
    /// poisoned, or a typed tensor backend error if synchronization fails.
    pub fn synchronize(&self) -> Result<()> {
        self.lock_backend()?.synchronize().map_err(Error::from)
    }

    fn exec_outputs_with_runtime<R>(
        &self,
        lock_backend_section: &'static str,
        exec_section: &'static str,
        op: &StdTensorOp,
        execute: impl FnOnce(&mut EagerBackend, Option<&Runtime>) -> Result<R>,
    ) -> Result<R> {
        // Lock ordering: eager execution holds the backend lock while standard
        // ops run without runtime extension access; extension ops receive the
        // runtime so extension cache locks are acquired only from that path.
        let mut backend = profile_eager_op_section(lock_backend_section, || self.lock_backend())?;
        let runtime = matches!(op, StdTensorOp::Extension(_)).then_some(&self.runtime);
        profile_eager_op_section(exec_section, || execute(&mut backend, runtime))
    }

    pub(crate) fn exec_outputs(&self, op: &StdTensorOp, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        self.exec_outputs_with_runtime(
            "exec_outputs.lock_backend",
            "exec_outputs.exec_op",
            op,
            |backend, runtime| exec_op_on_tensors_with_runtime(op, inputs, backend, runtime),
        )
    }

    pub(crate) fn exec_outputs_read(
        &self,
        op: &StdTensorOp,
        inputs: &[TensorRead<'_>],
    ) -> Result<Vec<Tensor>> {
        self.exec_outputs_with_runtime(
            "exec_outputs_read.lock_backend",
            "exec_outputs_read.exec_op",
            op,
            |backend, runtime| exec_op_on_tensor_reads_with_runtime(op, inputs, backend, runtime),
        )
    }

    #[cfg(test)]
    pub(crate) fn exec_standard_graph_outputs(
        &self,
        graph: &Graph<StdTensorOp>,
        initial_data: HashMap<ValueKey<StdTensorOp>, Tensor>,
    ) -> Result<EagerGraphExecution> {
        let mut backend =
            profile_eager_op_section("exec_graph.lock_backend", || self.lock_backend())?;
        let mut all_values = initial_data;

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
                                all_values.get(key).ok_or_else(|| {
                                    Error::Internal(format!(
                                        "standard graph eager execution missing value for {key:?}"
                                    ))
                                })
                            })
                            .collect::<Result<Vec<_>>>()?;
                        let input_reads = input_values
                            .iter()
                            .map(|value| TensorRead::from_tensor(value))
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
                        all_values.insert(key, output);
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
                all_values
                    .get(key)
                    .ok_or_else(|| {
                        Error::Internal(format!(
                            "standard graph eager execution missing graph output {key:?}"
                        ))
                    })?
                    .duplicate()
                    .map_err(Error::from)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(EagerGraphExecution { outputs })
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let c = ctx.constant_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap())?;
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(), ctx)?;
    /// let z = x.add(&c).unwrap();
    ///
    /// assert_eq!(z.value()?.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let x = EagerTensor::requires_grad_in(
    ///     Tensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap(),
    ///     ctx.clone(),
    /// )?;
    /// let loss = x.mul(&x)?;
    /// let dx = ctx.grad(&loss, &x)?;
    /// assert_eq!(dx.value()?.as_slice::<f64>().unwrap(), &[6.0]);
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
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

        let value = output.to_tensor()?;
        let seed = {
            let mut backend = self.lock_backend()?;
            one_like_tensor(&value, &mut *backend)?
        };
        let seed = EagerTensor::new_result(
            Arc::clone(self),
            eager_val_key(),
            seed,
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
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
    /// assert_eq!(dx.value()?.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
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
        // Unification 7: semantic path is the only VJP path.
        match semantic_eager_vjp_optional(self, output, wrt, cotangent)? {
            Some(result) => Ok(result),
            None => Ok(None),
        }
    }

    /// Forward-mode Jacobian-vector product for eager tensors.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
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
    /// assert_eq!(dy.value()?.as_slice::<f64>().unwrap(), &[6.0]);
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
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
        // Unification 7: semantic path is the only JVP path.
        match semantic_eager_jvp_optional(self, output, wrt, tangent)? {
            Some(result) => Ok(result),
            None => Ok(None),
        }
    }

    fn store_grads(
        &self,
        cotangents: &HashMap<ValueKey<StdTensorOp>, Tensor>,
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
                    updates.push((slot, incoming));
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
                Some(existing) => {
                    let existing_read = existing.tensor_read("EagerRuntime::store_grads")?;
                    let incoming_read = TensorRead::from_tensor(incoming);
                    let tensor = backend
                        .with_backend_session(|session| {
                            session.add_read(existing_read, incoming_read)
                        })
                        .map_err(Error::from)?;
                    AdValueRecord::from_tensor(tensor, "EagerRuntime::store_grads")?
                }
                None => {
                    let duplicate = backend
                        .with_backend_session(|session| {
                            session.to_contiguous_read(TensorRead::from_tensor(incoming))
                        })
                        .map_err(Error::from)?;
                    AdValueRecord::from_tensor(duplicate, "EagerRuntime::store_grads")?
                }
            };
            *current = Some(next);
        }

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PreparedDerivativeCacheKey {
    semantic_fingerprint: SemanticFingerprint,
    runtime_epoch: RuntimeEpoch,
    wrt_input_index: usize,
    input_metadata: Box<[ProgramValueMetadata]>,
}

/// Cached prepared derivative: program + index metadata.
#[derive(Debug)]
struct PreparedDerivative {
    program: Arc<CompiledGraph>,
    prepared: Arc<PreparedCompiledGraph>,
    seed_input_index: usize,
    derivative_output_index: usize,
}

#[derive(Debug)]
struct PreparedDerivativeCache {
    limits: AdTransformCacheLimits,
    entries: LruCache<PreparedDerivativeCacheKey, PreparedDerivativeCacheEntry>,
    stats: CacheStats,
}

impl PreparedDerivativeCache {
    fn limits(&self) -> AdTransformCacheLimits {
        self.limits
    }

    fn set_limits(&mut self, limits: AdTransformCacheLimits) {
        self.limits = limits;
        self.evict_to_limits();
    }

    fn clear(&mut self) {
        let clears = self.stats.clears.saturating_add(1);
        self.entries.clear();
        self.stats = CacheStats {
            clears,
            ..CacheStats::empty()
        };
    }

    fn stats(&self) -> CacheStats {
        self.stats
    }

    fn get(&mut self, key: &PreparedDerivativeCacheKey) -> Option<Arc<PreparedDerivative>> {
        match self.entries.get(key) {
            Some(entry) => {
                self.stats.hits = self.stats.hits.saturating_add(1);
                Some(Arc::clone(&entry.value))
            }
            None => {
                self.stats.misses = self.stats.misses.saturating_add(1);
                None
            }
        }
    }

    fn insert(&mut self, key: PreparedDerivativeCacheKey, value: Arc<PreparedDerivative>) {
        let retained_bytes = prepared_derivative_cache_entry_retained_bytes(&key, value.as_ref());
        let entry = PreparedDerivativeCacheEntry {
            value,
            retained_bytes,
        };
        self.stats.retained_bytes = self.stats.retained_bytes.saturating_add(retained_bytes);
        if let Some((_old_key, old_entry)) = self.entries.push(key, entry) {
            self.stats.retained_bytes = self
                .stats
                .retained_bytes
                .saturating_sub(old_entry.retained_bytes);
        }
        self.stats.entries = self.entries.len();
        self.evict_to_limits();
    }

    fn evict_to_limits(&mut self) {
        while self.entries.len() > self.limits.max_entries().get()
            || self
                .limits
                .max_retained_bytes()
                .is_some_and(|limit| self.stats.retained_bytes > limit.get())
        {
            let Some((_key, entry)) = self.entries.pop_lru() else {
                break;
            };
            self.stats.retained_bytes = self
                .stats
                .retained_bytes
                .saturating_sub(entry.retained_bytes);
            self.stats.evictions = self.stats.evictions.saturating_add(1);
        }
        self.stats.entries = self.entries.len();
    }
}

impl Default for PreparedDerivativeCache {
    fn default() -> Self {
        Self {
            limits: AdTransformCacheLimits::default(),
            entries: LruCache::unbounded(),
            stats: CacheStats::empty(),
        }
    }
}

#[derive(Debug)]
struct PreparedDerivativeCacheEntry {
    value: Arc<PreparedDerivative>,
    retained_bytes: usize,
}

fn prepared_derivative_cache_entry_retained_bytes(
    key: &PreparedDerivativeCacheKey,
    value: &PreparedDerivative,
) -> usize {
    size_of::<PreparedDerivativeCacheKey>()
        .saturating_add(
            key.input_metadata
                .len()
                .saturating_mul(size_of::<ProgramValueMetadata>()),
        )
        .saturating_add(size_of::<PreparedDerivative>())
        .saturating_add(compiled_graph_retained_bytes(value.program.as_ref()))
        .saturating_add(prepared_compiled_graph_retained_bytes(
            value.prepared.as_ref(),
            value.program.as_ref(),
        ))
}

fn prepared_compiled_graph_retained_bytes(
    prepared: &PreparedCompiledGraph,
    derivative_program: &CompiledGraph,
) -> usize {
    size_of_val(prepared).saturating_add(compiled_graph_retained_bytes(derivative_program))
}

fn compiled_graph_retained_bytes(program: &CompiledGraph) -> usize {
    size_of::<CompiledGraph>()
        .saturating_add(size_of_val(program.input_keys()))
        .saturating_add(program.bindings().len().saturating_mul(size_of::<usize>()))
        .saturating_add(semantic_program_retained_bytes(program.program()))
}

fn semantic_program_retained_bytes(program: &SemanticProgram) -> usize {
    size_of::<SemanticProgram>()
        .saturating_add(size_of_val(program.inputs()))
        .saturating_add(size_of_val(program.outputs()))
        .saturating_add(
            program
                .operations()
                .len()
                .saturating_mul(size_of::<usize>()),
        )
        .saturating_add(
            program
                .shape_guards()
                .len()
                .saturating_mul(size_of::<usize>()),
        )
}

fn semantic_eager_vjp_optional(
    ctx: &Arc<EagerRuntime>,
    output: &EagerTensor,
    wrt: &EagerTensor,
    cotangent: &EagerTensor,
) -> Result<Option<Option<EagerTensor>>> {
    if !eager_semantic_vjp_enabled() {
        return Ok(None);
    }
    let (Some(output_trace), Some(wrt_trace)) =
        (output.semantic_trace.as_ref(), wrt.semantic_trace.as_ref())
    else {
        return Ok(None);
    };
    let Some(wrt_key) = wrt_trace.input_key() else {
        return Ok(None);
    };
    if !output_trace.has_attached_input_key(&wrt_key) {
        return Ok(None);
    }

    // First compile the trace to get bindings and wrt_input_index.
    // (The compile step is needed even for cache hits to extract tensor bindings.)
    let mut compiler = GraphCompiler::new();
    let source = compile_ad_source(&mut compiler, output_trace)?;
    if source.output_count() != 1
        || source.input_keys().len() != source.input_count()
        || source.bindings().len() != source.input_count()
    {
        return Ok(None);
    }
    let Some(wrt_input_index) = source.input_key_index(&wrt_key) else {
        return Ok(None);
    };

    // S2: check prepared-derivative cache before AD transform + compile_frozen.
    let cache_key = PreparedDerivativeCacheKey {
        semantic_fingerprint: source.program().semantic_fingerprint(),
        runtime_epoch: ctx.runtime.epoch().map_err(|source| {
            Error::runtime_state_source("semantic_eager_vjp", ErrorPhase::Execution, source)
        })?,
        wrt_input_index,
        input_metadata: source.frozen_program().input_metadata_with_bound_shapes(),
    };
    let prepared = { ctx.lock_prepared_derivative_cache()?.get(&cache_key) };
    let (seed_input_index, derivative_output_index, derivative_program, prepared_runtime) =
        if let Some(prepared) = prepared {
            (
                prepared.seed_input_index,
                prepared.derivative_output_index,
                Arc::clone(&prepared.program),
                Some(Arc::clone(&prepared.prepared)),
            )
        } else {
            let mut active_inputs = vec![false; source.input_count()];
            if let Some(active) = active_inputs.get_mut(wrt_input_index) {
                *active = true;
            } else {
                return Ok(None);
            }
            let active_outputs = vec![true; source.output_count()];
            let ad = AdContext::with_rules_and_transform_cache(
                ctx.semantic_extension_rules.clone(),
                Arc::clone(&ctx.ad_transform_cache),
            );
            let derivative = ad
                .vjp_program(source.frozen_program(), &active_inputs, &active_outputs)
                .map_err(|source| {
                    Error::runtime_state_source(
                        "semantic_eager_vjp",
                        ErrorPhase::GraphBuild,
                        source,
                    )
                })?;
            let seed_input_index = derivative
                .derivative_input_indices()
                .first()
                .copied()
                .flatten();
            let derivative_output_index = derivative
                .derivative_output_indices()
                .get(wrt_input_index)
                .copied()
                .flatten();
            let (Some(seed_input_index), Some(derivative_output_index)) =
                (seed_input_index, derivative_output_index)
            else {
                return Ok(Some(None));
            };
            let program = Arc::new(compiler.compile_frozen_program(derivative.frozen())?);
            (seed_input_index, derivative_output_index, program, None)
        };

    let cotangent_tensor = Arc::new(RetainedValue::from_tensor(cotangent.to_tensor()?));
    let input_count = derivative_program.input_count();
    let mut owned_inputs: Vec<Option<Tensor>> = (0..input_count).map(|_| None).collect();
    for (source_input_index, (_, tensor)) in source.bindings().iter().enumerate() {
        let Some(slot) = owned_inputs.get_mut(source_input_index) else {
            return Err(Error::Internal(format!(
                "semantic eager VJP derivative program has no primal input slot {source_input_index}"
            )));
        };
        *slot = Some(copy_value_for_runtime(ctx, tensor)?);
    }
    let Some(slot) = owned_inputs.get_mut(seed_input_index) else {
        return Err(Error::Internal(format!(
            "semantic eager VJP seed input index {seed_input_index} is outside {} inputs",
            owned_inputs.len()
        )));
    };
    *slot = Some(copy_value_for_runtime(ctx, cotangent_tensor.as_ref())?);
    let input_refs = owned_inputs
        .iter()
        .enumerate()
        .map(|(index, tensor)| {
            tensor.as_ref().ok_or_else(|| {
                Error::Internal(format!(
                    "semantic eager VJP derivative input {index} was not populated"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let prepared_runtime = if let Some(prepared_runtime) = prepared_runtime {
        prepared_runtime
    } else {
        let prepared_runtime = Arc::new(
            ctx.runtime
                .prepare_compiled(&derivative_program, &input_refs)?,
        );
        let entry = Arc::new(PreparedDerivative {
            program: Arc::clone(&derivative_program),
            prepared: Arc::clone(&prepared_runtime),
            seed_input_index,
            derivative_output_index,
        });
        ctx.lock_prepared_derivative_cache()?
            .insert(cache_key, entry);
        prepared_runtime
    };
    let outputs = ctx.runtime.run_prepared(&prepared_runtime, &input_refs)?;
    let output_count = outputs.len();
    let Some(result) = outputs.into_iter().nth(derivative_output_index) else {
        return Err(Error::Internal(format!(
            "semantic eager VJP derivative output index {derivative_output_index} is outside {} outputs",
            output_count
        )));
    };
    let cotangent_trace =
        TracedTensor::from_shared_tensor_value_symbolic_shape(Arc::clone(&cotangent_tensor))?;
    let semantic_trace = derivative_trace_from_frozen_program(
        &source,
        derivative_program.frozen_program(),
        derivative_output_index,
        &[(seed_input_index, Arc::clone(&cotangent_tensor))],
        &[output_trace, wrt_trace, &cotangent_trace],
        None,
        "semantic_eager_vjp",
    )?;

    #[cfg(test)]
    EAGER_SEMANTIC_VJP_EXECUTIONS.fetch_add(1, Ordering::Relaxed);

    Ok(Some(Some(EagerTensor::new_result_with_semantic_trace(
        Arc::clone(ctx),
        eager_val_key(),
        result,
        true,
        None,
        Some(semantic_trace),
        Vec::new(),
    )?)))
}

fn semantic_eager_jvp_optional(
    ctx: &Arc<EagerRuntime>,
    output: &EagerTensor,
    wrt: &EagerTensor,
    tangent: &EagerTensor,
) -> Result<Option<Option<EagerTensor>>> {
    if !eager_semantic_vjp_enabled() {
        return Ok(None);
    }
    let (Some(output_trace), Some(wrt_trace)) =
        (output.semantic_trace.as_ref(), wrt.semantic_trace.as_ref())
    else {
        return Ok(None);
    };
    let Some(wrt_key) = wrt_trace.input_key() else {
        return Ok(None);
    };
    if !output_trace.has_attached_input_key(&wrt_key) {
        return Ok(None);
    }

    let mut compiler = GraphCompiler::new();
    let source = compile_ad_source(&mut compiler, output_trace)?;
    if source.output_count() != 1
        || source.input_keys().len() != source.input_count()
        || source.bindings().len() != source.input_count()
    {
        return Ok(None);
    }
    let Some(wrt_input_index) = source.input_key_index(&wrt_key) else {
        return Ok(None);
    };

    let mut active_inputs = vec![false; source.input_count()];
    if let Some(active) = active_inputs.get_mut(wrt_input_index) {
        *active = true;
    } else {
        return Ok(None);
    }
    let ad = AdContext::with_rules_and_transform_cache(
        ctx.semantic_extension_rules.clone(),
        Arc::clone(&ctx.ad_transform_cache),
    );
    let derivative = ad
        .jvp_program(source.frozen_program(), &active_inputs)
        .map_err(|source| {
            Error::runtime_state_source("semantic_eager_jvp", ErrorPhase::GraphBuild, source)
        })?;
    // derivative_input_indices maps source input → derivative seed input.
    let Some(seed_input_index) = derivative
        .derivative_input_indices()
        .get(wrt_input_index)
        .copied()
        .flatten()
    else {
        return Ok(Some(None));
    };
    // derivative_output_indices maps source output → derivative output.
    // There is always exactly one source output (guarded above).
    let Some(derivative_output_index) = derivative
        .derivative_output_indices()
        .first()
        .copied()
        .flatten()
    else {
        return Ok(Some(None));
    };

    let derivative_program = compiler.compile_frozen_program(derivative.frozen())?;
    let tangent_tensor = Arc::new(RetainedValue::from_tensor(tangent.to_tensor()?));
    let input_count = derivative_program.input_count();
    let mut owned_inputs: Vec<Option<Tensor>> = (0..input_count).map(|_| None).collect();
    for (source_input_index, (_, tensor)) in source.bindings().iter().enumerate() {
        let Some(slot) = owned_inputs.get_mut(source_input_index) else {
            return Err(Error::Internal(format!(
                "semantic eager JVP derivative program has no primal input slot {source_input_index}"
            )));
        };
        *slot = Some(copy_value_for_runtime(ctx, tensor)?);
    }
    let Some(slot) = owned_inputs.get_mut(seed_input_index) else {
        return Err(Error::Internal(format!(
            "semantic eager JVP seed input index {seed_input_index} is outside {} inputs",
            owned_inputs.len()
        )));
    };
    *slot = Some(copy_value_for_runtime(ctx, tangent_tensor.as_ref())?);
    let input_refs = owned_inputs
        .iter()
        .enumerate()
        .map(|(index, tensor)| {
            tensor.as_ref().ok_or_else(|| {
                Error::Internal(format!(
                    "semantic eager JVP derivative input {index} was not populated"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let outputs = ctx.runtime.run_compiled(&derivative_program, &input_refs)?;
    let output_count = outputs.len();
    let Some(result) = outputs.into_iter().nth(derivative_output_index) else {
        return Err(Error::Internal(format!(
            "semantic eager JVP derivative output index {derivative_output_index} is outside {} outputs",
            output_count
        )));
    };
    let tangent_trace =
        TracedTensor::from_shared_tensor_value_symbolic_shape(Arc::clone(&tangent_tensor))?;
    let semantic_trace = derivative_trace_from_frozen_program(
        &source,
        derivative.frozen(),
        derivative_output_index,
        &[(seed_input_index, Arc::clone(&tangent_tensor))],
        &[output_trace, wrt_trace, &tangent_trace],
        None,
        "semantic_eager_jvp",
    )?;

    Ok(Some(Some(EagerTensor::new_result_with_semantic_trace(
        Arc::clone(ctx),
        eager_val_key(),
        result,
        true,
        None,
        Some(semantic_trace),
        Vec::new(),
    )?)))
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

fn copy_value_for_runtime(ctx: &EagerRuntime, value: &RetainedValue) -> Result<Tensor> {
    let read = value.tensor_read().map_err(|error| {
        Error::runtime_state_source("copy_value_for_runtime", ErrorPhase::Execution, error)
    })?;
    ctx.with_execution_session(|session| session.to_contiguous_read(read))?
        .map_err(Error::from)
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
/// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
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
    pub(crate) key: ValueKey<StdTensorOp>,
    pub(crate) trace: Option<EagerTrace>,
    pub(crate) semantic_trace: Option<TracedTensor>,
    pub(crate) requires_grad: bool,
    grad_slot: GradSlot,
    pub(crate) metadata_scopes: Vec<Arc<GlobalMetadataScope>>,
    pub(crate) ctx: Arc<EagerRuntime>,
    _record: Arc<EagerTensorRecord>,
}

pub(crate) struct EagerTensorRecord {
    value: Arc<AdValueRecord>,
    key: ValueKey<StdTensorOp>,
    trace: Option<EagerTrace>,
    semantic_trace: Option<TracedTensor>,
    requires_grad: bool,
    grad_slot: GradSlot,
    metadata_scopes: Vec<Arc<GlobalMetadataScope>>,
    ctx: Arc<EagerRuntime>,
}

struct EagerTensorParts {
    ctx: Arc<EagerRuntime>,
    key: ValueKey<StdTensorOp>,
    requires_grad: bool,
    trace: Option<EagerTrace>,
    semantic_trace: Option<TracedTensor>,
    value: Arc<AdValueRecord>,
    metadata_scopes: Vec<Arc<GlobalMetadataScope>>,
    register_value: bool,
}

impl fmt::Debug for EagerTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EagerTensor")
            .field("dtype", &self.dtype())
            .field("shape", &self.shape())
            .field("key", &self.key)
            .field("requires_grad", &self.requires_grad)
            .field("has_trace", &self.trace.is_some())
            .field("has_semantic_trace", &self.semantic_trace.is_some())
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(), ctx)?;
    ///
    /// assert_eq!(x.value()?.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
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
        let semantic_tensor = ctx
            .with_execution_session(|session| {
                session.to_contiguous_read(TensorRead::from_tensor(&tensor))
            })?
            .map_err(Error::from)?;
        let semantic_value = Arc::new(RetainedValue::from_tensor(semantic_tensor));
        let semantic_trace = TracedTensor::from_shared_tensor_value_symbolic_shape(semantic_value)?;
        let metadata_scope =
            register_scoped_value_metadata(key.clone(), tensor_meta_from_tensor(&tensor)).map_err(
                |err| {
                    Error::runtime_state_source("eager leaf metadata", ErrorPhase::GraphBuild, err)
                },
            )?;
        let value = AdValueRecord::from_tensor(tensor, "EagerTensor::new_leaf")?;
        Self::from_parts(EagerTensorParts {
            ctx,
            key,
            requires_grad,
            trace: None,
            semantic_trace: Some(semantic_trace),
            value,
            metadata_scopes: metadata_scopes_for_scope(metadata_scope),
            register_value: true,
        })
    }

    pub(crate) fn new_result(
        ctx: Arc<EagerRuntime>,
        key: ValueKey<StdTensorOp>,
        tensor: Tensor,
        requires_grad: bool,
        trace: Option<EagerTrace>,
        metadata_scopes: Vec<Arc<GlobalMetadataScope>>,
    ) -> Result<Self> {
        Self::new_result_with_semantic_trace(
            ctx,
            key,
            tensor,
            requires_grad,
            trace,
            None,
            metadata_scopes,
        )
    }

    pub(crate) fn new_result_with_semantic_trace(
        ctx: Arc<EagerRuntime>,
        key: ValueKey<StdTensorOp>,
        tensor: Tensor,
        requires_grad: bool,
        trace: Option<EagerTrace>,
        semantic_trace: Option<TracedTensor>,
        metadata_scopes: Vec<Arc<GlobalMetadataScope>>,
    ) -> Result<Self> {
        let value = AdValueRecord::from_tensor(tensor, "EagerTensor::new_result")?;
        Self::from_parts(EagerTensorParts {
            ctx,
            key,
            requires_grad,
            trace,
            semantic_trace,
            value,
            metadata_scopes,
            register_value: true,
        })
    }

    pub(crate) fn new_unregistered_result_with_semantic_trace(
        ctx: Arc<EagerRuntime>,
        key: ValueKey<StdTensorOp>,
        tensor: Tensor,
        requires_grad: bool,
        trace: Option<EagerTrace>,
        semantic_trace: Option<TracedTensor>,
        metadata_scopes: Vec<Arc<GlobalMetadataScope>>,
    ) -> Result<Self> {
        let value = AdValueRecord::from_tensor(tensor, "EagerTensor::new_unregistered_result")?;
        Self::from_parts(EagerTensorParts {
            ctx,
            key,
            requires_grad,
            trace,
            semantic_trace,
            value,
            metadata_scopes,
            register_value: false,
        })
    }

    pub(crate) fn new_result_value(
        ctx: Arc<EagerRuntime>,
        key: ValueKey<StdTensorOp>,
        value: TensorValue,
        requires_grad: bool,
        trace: Option<EagerTrace>,
        semantic_trace: Option<TracedTensor>,
        metadata_scopes: Vec<Arc<GlobalMetadataScope>>,
    ) -> Result<Self> {
        let (group, slot, dtype, shape) = value.try_into_group_parts().map_err(|_| {
            Error::runtime_state(
                "EagerTensor::new_result_value",
                ErrorPhase::Execution,
                "a TensorValue could not be transferred into its allocation group",
            )
        })?;
        let value = AdValueRecord::from_group(group, slot, dtype, shape);
        Self::from_parts(EagerTensorParts {
            ctx,
            key,
            requires_grad,
            trace,
            semantic_trace,
            value,
            metadata_scopes,
            register_value: true,
        })
    }

    fn from_parts(parts: EagerTensorParts) -> Result<Self> {
        let EagerTensorParts {
            ctx,
            key,
            requires_grad,
            trace,
            semantic_trace,
            value,
            metadata_scopes,
            register_value,
        } = parts;
        let grad_slot = Arc::new(Mutex::new(None));
        if requires_grad {
            ctx.try_register_grad_slot(&key, &grad_slot)?;
        }
        let record = Arc::new(EagerTensorRecord {
            value: Arc::clone(&value),
            key: key.clone(),
            trace: trace.clone(),
            semantic_trace: semantic_trace.clone(),
            requires_grad,
            grad_slot: Arc::clone(&grad_slot),
            metadata_scopes: metadata_scopes.clone(),
            ctx: Arc::clone(&ctx),
        });
        if register_value {
            ctx.try_register_value_record(&key, &record)?;
        }

        Ok(Self {
            key,
            trace,
            semantic_trace,
            requires_grad,
            grad_slot,
            metadata_scopes,
            ctx,
            _record: record,
        })
    }

    pub(crate) fn new_untracked_result(ctx: Arc<EagerRuntime>, tensor: Tensor) -> Result<Self> {
        let value = AdValueRecord::from_tensor(tensor, "EagerTensor::new_untracked_result")?;
        Ok(Self::new_untracked_value_record(ctx, value, None))
    }

    pub(crate) fn new_untracked_value_result(
        ctx: Arc<EagerRuntime>,
        value: TensorValue,
    ) -> Result<Self> {
        Self::new_untracked_value_result_with_semantic_trace(ctx, value, None)
    }

    pub(crate) fn new_untracked_value_result_with_semantic_trace(
        ctx: Arc<EagerRuntime>,
        value: TensorValue,
        semantic_trace: Option<TracedTensor>,
    ) -> Result<Self> {
        let (group, slot, dtype, shape) = value.try_into_group_parts().map_err(|_| {
            Error::runtime_state(
                "EagerTensor::new_untracked_value_result",
                ErrorPhase::Execution,
                "a TensorValue could not be transferred into its allocation group",
            )
        })?;
        let value = AdValueRecord::from_group(group, slot, dtype, shape);
        Ok(Self::new_untracked_value_record(ctx, value, semantic_trace))
    }

    fn new_untracked_value_record(
        ctx: Arc<EagerRuntime>,
        value: Arc<AdValueRecord>,
        semantic_trace: Option<TracedTensor>,
    ) -> Self {
        let key = eager_val_key();
        let grad_slot = Arc::new(Mutex::new(None));
        let record = Arc::new(EagerTensorRecord {
            value,
            key: key.clone(),
            trace: None,
            semantic_trace: semantic_trace.clone(),
            requires_grad: false,
            grad_slot: Arc::clone(&grad_slot),
            metadata_scopes: Vec::new(),
            ctx: Arc::clone(&ctx),
        });
        Self {
            key,
            trace: None,
            semantic_trace,
            requires_grad: false,
            grad_slot,
            metadata_scopes: Vec::new(),
            ctx,
            _record: record,
        }
    }

    pub(crate) fn from_record(record: Arc<EagerTensorRecord>) -> Self {
        Self {
            key: record.key.clone(),
            trace: record.trace.clone(),
            semantic_trace: record.semantic_trace.clone(),
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(), ctx)?;
    /// let y = x.detach();
    ///
    /// assert_eq!(y.value()?.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    /// assert!(y.grad().unwrap().is_none());
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    pub fn detach(&self) -> Self {
        let semantic_trace = self
            .duplicate_value()
            .ok()
            .and_then(|tensor| TracedTensor::from_tensor_symbolic_shape(tensor).ok());
        Self::new_untracked_value_record(
            self.ctx.clone(),
            Arc::clone(&self._record.value),
            semantic_trace,
        )
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
    /// let ctx_a = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let ctx_b = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
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

    /// Borrow the retained value without creating an owner or copy.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeState`] when the retained allocation-group
    /// descriptor is unavailable or invalid.
    pub fn value(&self) -> Result<ValueGuard<'_>> {
        self._record.value.value("EagerTensor::value")
    }

    /// Explicitly duplicate this value into a fresh standalone allocation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeState`] when the retained value or execution
    /// session is unavailable, or a typed host/backend error when the value
    /// cannot be materialized as a contiguous tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let value = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?,
    ///     ctx,
    /// )?;
    /// let duplicate = value.duplicate_value()?;
    /// assert_eq!(duplicate.as_slice::<f64>()?, &[1.0, 2.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    pub fn duplicate_value(&self) -> Result<Tensor> {
        let value = self.value()?;
        match value.duplicate_host_tensor() {
            Ok(tensor) => Ok(tensor),
            Err(_) => {
                let read = self
                    ._record
                    .value
                    .tensor_read("EagerTensor::duplicate_value")?;
                self.ctx
                    .with_execution_session(|session| session.to_contiguous_read(read))?
                    .map_err(Error::from)
            }
        }
    }

    // INVARIANT: the error variants return the unchanged eager handle so a
    // caller can retry ownership extraction without an implicit copy.
    #[allow(clippy::result_large_err)]
    /// Consume this handle and structurally extract its retained allocation.
    ///
    /// A shared handle is returned unchanged as [`IntoValueError::NotUnique`].
    /// Group extraction failures return the unchanged handle and typed group
    /// error; no copy or fallback materialization is attempted.
    ///
    /// # Errors
    ///
    /// Returns [`IntoValueError::NotUnique`] when another handle retains the
    /// value, or [`IntoValueError::Extract`] when structural group extraction
    /// fails because the allocation is aliased or its descriptor is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let value = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major(vec![1], vec![3.0_f64])?,
    ///     ctx,
    /// )?;
    /// let owner = value
    ///     .into_value()
    ///     .expect("a uniquely owned value should be extractable");
    /// assert_eq!(owner.as_slice::<f64>()?, &[3.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    pub fn into_value(self) -> std::result::Result<Tensor, IntoValueError<Self>> {
        if Arc::strong_count(&self._record) != 1 {
            return Err(IntoValueError::NotUnique(self));
        }
        let Self { _record, .. } = self;
        let record = match Arc::try_unwrap(_record) {
            Ok(record) => record,
            Err(record) => return Err(IntoValueError::NotUnique(Self::from_record(record))),
        };
        let EagerTensorRecord {
            value,
            key,
            trace,
            semantic_trace,
            requires_grad,
            grad_slot,
            metadata_scopes,
            ctx,
        } = record;
        let value = match Arc::try_unwrap(value) {
            Ok(value) => value,
            Err(value) => {
                let record = Arc::new(EagerTensorRecord {
                    value,
                    key,
                    trace,
                    semantic_trace,
                    requires_grad,
                    grad_slot,
                    metadata_scopes,
                    ctx,
                });
                return Err(IntoValueError::NotUnique(Self::from_record(record)));
            }
        };
        let AdValueRecord {
            container,
            slot,
            dtype,
            shape,
        } = value;
        let container = match Arc::try_unwrap(container) {
            Ok(container) => container,
            Err(container) => {
                let record = Arc::new(EagerTensorRecord {
                    value: Arc::new(AdValueRecord {
                        container,
                        slot,
                        dtype,
                        shape,
                    }),
                    key,
                    trace,
                    semantic_trace,
                    requires_grad,
                    grad_slot,
                    metadata_scopes,
                    ctx,
                });
                return Err(IntoValueError::NotUnique(Self::from_record(record)));
            }
        };
        match container.group.into_tensor(slot) {
            Ok(tensor) => Ok(tensor),
            Err((group, error)) => {
                let record = Arc::new(EagerTensorRecord {
                    value: Arc::new(AdValueRecord {
                        container: Arc::new(RetentionContainer { group }),
                        slot,
                        dtype,
                        shape,
                    }),
                    key,
                    trace,
                    semantic_trace,
                    requires_grad,
                    grad_slot,
                    metadata_scopes,
                    ctx,
                });
                Err(IntoValueError::Extract {
                    value: Self::from_record(record),
                    error,
                })
            }
        }
    }

    /// Return this tensor's scalar dtype without materializing through
    /// [`value`](Self::value).
    pub fn dtype(&self) -> DType {
        self._record.value.dtype()
    }

    /// Return this tensor's logical shape without materializing through
    /// [`value`](Self::value).
    pub fn shape(&self) -> &[usize] {
        self._record.value.shape()
    }

    /// Borrow this tensor value as a [`TensorRead`].
    ///
    /// This is the preferred borrowed input boundary for executor calls. It
    /// preserves the option to replace eager storage with non-contiguous views
    /// without forcing callers through [`value`](Self::value).
    ///
    /// # Panics
    ///
    /// Panics if a validated eager value record becomes unavailable, which
    /// indicates an internal invariant violation.
    pub fn tensor_read(&self) -> TensorRead<'_> {
        self._record
            .value
            .tensor_read("EagerTensor::tensor_read")
            .expect("validated eager value record")
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
        self.duplicate_value()
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
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
    pub fn grad(&self) -> Result<Option<GradientValue>> {
        self.grad_slot
            .lock()
            .map_err(|_| {
                Error::runtime_state(
                    "eager_gradient_slot",
                    ErrorPhase::Execution,
                    "lock poisoned",
                )
            })
            .map(|slot| {
                slot.as_ref().map(|record| GradientValue {
                    record: Arc::clone(record),
                    ctx: Arc::clone(&self.ctx),
                })
            })
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let plain = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(), ctx.clone()).unwrap();
    /// let tracked = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(), ctx.clone()).unwrap();
    /// let detached = tracked.detach();
    ///
    /// assert!(!plain.tracks_grad());
    /// assert!(tracked.tracks_grad());
    /// assert!(!detached.tracks_grad());
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    pub fn tracks_grad(&self) -> bool {
        self.requires_grad
    }

    #[cfg(test)]
    fn debug_trace_saved_value_count(&self) -> Option<usize> {
        None
    }

    /// Return the opaque identifier of the context this tensor belongs to.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(), ctx.clone()).unwrap();
    ///
    /// assert_eq!(x.ctx_id(), ctx.id());
    /// # Ok::<(), tenferro_ad::Error>(())
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(), ctx.clone()).unwrap();
    /// let y = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap(), ctx).unwrap();
    ///
    /// assert!(x.same_context(&y));
    /// # Ok::<(), tenferro_ad::Error>(())
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

        let graph_input_keys = (0..inputs.len())
            .map(|_| next_input_key())
            .collect::<Vec<_>>();
        let graph = build_graph(&graph_input_keys)?;
        let initial_data = graph_input_keys
            .iter()
            .zip(inputs.iter())
            .map(|(key, tensor)| Ok((ValueKey::Input(key.clone()), tensor.to_tensor()?)))
            .collect::<Result<HashMap<_, _>>>()?;
        let execution = ctx.exec_standard_graph_outputs(graph.as_ref(), initial_data)?;
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
                    Self::new_unregistered_result_with_semantic_trace(
                        Arc::clone(&ctx),
                        eager_val_key(),
                        output,
                        false,
                        None,
                        None,
                        Vec::new(),
                    )
                })
                .collect();
        }

        let recorded = record_eager_graph_outputs(
            graph.as_ref(),
            &graph_input_keys,
            &execution.outputs,
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
            .zip(recorded.semantic_traces)
            .zip(execution.outputs)
            .map(|((trace, semantic_trace), output)| {
                Self::new_result_with_semantic_trace(
                    Arc::clone(&ctx),
                    trace.key,
                    output,
                    trace.requires_grad,
                    trace.trace,
                    semantic_trace,
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap(), ctx).unwrap();
    /// let loss = x.add(&x).unwrap().reduce_sum(Some(&[0])).unwrap();
    /// let _cotangents = loss.backward().unwrap();
    /// let loss = x.add(&x).unwrap().reduce_sum(Some(&[0])).unwrap();
    /// let _cotangents = loss.backward().unwrap();
    ///
    /// assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[4.0, 4.0, 4.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::NonScalarGrad`] when this output is not scalar,
    /// [`Error::UnsupportedAdRule`] when a graph operation lacks a reverse rule,
    /// or a typed validation/backend/runtime-state error during the reverse pass.
    pub fn backward(&self) -> Result<Gradients> {
        if !self.shape().is_empty() {
            return Err(Error::NonScalarGrad {
                shape: self.shape().to_vec(),
            });
        }

        let value = self.to_tensor()?;
        let seed = {
            let mut backend = self.ctx.lock_backend()?;
            one_like_tensor(&value, &mut *backend)?
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
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
    pub fn backward_with(&self, cotangent: &EagerTensor) -> Result<Gradients> {
        if !self.same_context(cotangent) {
            return Err(Error::ContextMismatch {
                lhs: self.ctx_id(),
                rhs: cotangent.ctx_id(),
            });
        }
        validate_seed_tensor("backward", self, cotangent)?;
        self.backward_from_seed(cotangent.to_tensor()?)
    }

    fn backward_from_seed(&self, seed: Tensor) -> Result<Gradients> {
        let cotangent = EagerTensor::new_result(
            Arc::clone(&self.ctx),
            eager_val_key(),
            seed,
            false,
            None,
            Vec::new(),
        )?;
        let candidate_keys = {
            let mut slots = self.ctx.lock_grad_slots()?;
            let mut keys = Vec::new();
            slots.retain(|key, slot| {
                if slot.upgrade().is_some() {
                    keys.push(key.clone());
                    true
                } else {
                    false
                }
            });
            keys
        };

        let mut cotangents = HashMap::new();
        for key in candidate_keys {
            let Some(record) = self.ctx.value_record(&key)? else {
                continue;
            };
            if !record.requires_grad {
                continue;
            }
            let wrt = EagerTensor::from_record(record);
            let Some(grad) = self.ctx.vjp_optional(self, &wrt, &cotangent)? else {
                continue;
            };
            let tensor = match grad.into_value() {
                Ok(tensor) => tensor,
                Err(IntoValueError::NotUnique(handle)) => handle.duplicate_value()?,
                Err(IntoValueError::Extract { error, .. }) => {
                    return Err(Error::runtime_state_source(
                        "EagerTensor::backward",
                        ErrorPhase::Execution,
                        error,
                    ));
                }
            };
            cotangents.insert(key, tensor);
        }
        let mut backend = self.ctx.lock_backend()?;
        self.ctx.store_grads(&cotangents, &mut backend)?;
        Gradients::from_tensors(cotangents)
    }
}

pub(crate) fn eager_val_key() -> ValueKey<StdTensorOp> {
    ValueKey::Input(next_input_key())
}

pub(crate) struct RecordedEagerTrace {
    pub(crate) key: ValueKey<StdTensorOp>,
    pub(crate) trace: Option<EagerTrace>,
    pub(crate) requires_grad: bool,
}

pub(crate) struct RecordedEagerOutputs {
    pub(crate) traces: Vec<RecordedEagerTrace>,
    pub(crate) semantic_traces: Vec<Option<TracedTensor>>,
    pub(crate) metadata_scope: Arc<GlobalMetadataScope>,
}

pub(crate) fn record_eager_outputs(
    op: &StdTensorOp,
    outputs: &[&Tensor],
    inputs: &[&EagerTensor],
) -> Result<RecordedEagerOutputs> {
    let semantic_traces = record_semantic_eager_outputs(op, outputs.len(), inputs)?;
    let output_metadata = outputs.iter().map(|output| tensor_meta_from_tensor(output));
    record_eager_outputs_from_metadata(output_metadata, semantic_traces, inputs)
}

pub(crate) fn record_eager_value_outputs(
    op: &StdTensorOp,
    outputs: &[&TensorValue],
    inputs: &[&EagerTensor],
) -> Result<RecordedEagerOutputs> {
    let semantic_traces = record_semantic_eager_outputs(op, outputs.len(), inputs)?;
    let output_metadata = outputs.iter().map(|output| tensor_meta_from_value(output));
    record_eager_outputs_from_metadata(output_metadata, semantic_traces, inputs)
}

fn record_semantic_eager_outputs(
    op: &StdTensorOp,
    output_count: usize,
    inputs: &[&EagerTensor],
) -> Result<Vec<Option<TracedTensor>>> {
    // Materialize a constant semantic leaf for any untracked input that lost
    // its implicit semantic trace on the active-edge fast path. This keeps
    // "untracked constant feeds tracked AD" working (PyTorch-style: untracked
    // = constant leaf, no gradient flows to it) without re-recording every
    // untracked op at creation time.
    let mut owned_constants = Vec::<TracedTensor>::new();
    for input in inputs {
        if input.semantic_trace.is_none() {
            owned_constants.push(TracedTensor::from_tensor_symbolic_shape(
                input.to_tensor()?,
            )?);
        }
    }
    let mut constants = owned_constants.iter();
    let semantic_inputs: Vec<&TracedTensor> = inputs
        .iter()
        .map(|input| {
            input
                .semantic_trace
                .as_ref()
                .unwrap_or_else(|| constants.next().expect("materialized constant"))
        })
        .collect();
    let semantic_outputs = match op {
        StdTensorOp::Extension(ext) => {
            tenferro_runtime::extension::apply(Arc::clone(ext), &semantic_inputs)?
        }
        _ => tenferro_runtime::extension::apply_standard_op(op.clone(), &semantic_inputs)?,
    };
    if semantic_outputs.len() != output_count {
        return Err(Error::Internal(format!(
            "semantic eager recording expected {output_count} outputs for {op:?}, got {}",
            semantic_outputs.len()
        )));
    }
    Ok(semantic_outputs.into_iter().map(Some).collect())
}

#[cfg(test)]
fn record_eager_graph_outputs(
    graph: &Graph<StdTensorOp>,
    graph_input_keys: &[TensorInputKey],
    outputs: &[Tensor],
    inputs: &[&EagerTensor],
) -> Result<RecordedEagerOutputs> {
    let semantic_traces = record_semantic_eager_graph_outputs(graph, graph_input_keys, inputs)?;
    let output_metadata = outputs.iter().map(tensor_meta_from_tensor);
    record_eager_outputs_from_metadata(output_metadata, semantic_traces, inputs)
}

#[cfg(test)]
fn record_semantic_eager_graph_outputs(
    graph: &Graph<StdTensorOp>,
    graph_input_keys: &[TensorInputKey],
    inputs: &[&EagerTensor],
) -> Result<Vec<Option<TracedTensor>>> {
    let Some(semantic_inputs) = inputs
        .iter()
        .map(|input| input.semantic_trace.as_ref())
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(vec![None; graph.outputs().len()]);
    };
    if graph_input_keys.len() != semantic_inputs.len() {
        return Err(Error::Internal(format!(
            "semantic graph recording expected {} input keys, got {}",
            semantic_inputs.len(),
            graph_input_keys.len()
        )));
    }

    let mut values = HashMap::new();
    for (key, tensor) in graph_input_keys.iter().zip(semantic_inputs) {
        values.insert(ValueKey::Input(key.clone()), tensor.clone());
    }

    for op_node in graph.operations() {
        let input_values = op_node
            .inputs
            .iter()
            .map(|input| {
                let key = match input {
                    ValueRef::Local(local_id) => &graph.values()[*local_id].key,
                    ValueRef::External(key) => key,
                };
                values.get(key).cloned().ok_or_else(|| {
                    Error::Internal(format!(
                        "semantic graph recording missing value for {key:?}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let input_refs = input_values.iter().collect::<Vec<_>>();
        let semantic_outputs = match &op_node.operation {
            StdTensorOp::Extension(ext) => {
                tenferro_runtime::extension::apply(Arc::clone(ext), &input_refs)?
            }
            op => tenferro_runtime::extension::apply_standard_op(op.clone(), &input_refs)?,
        };
        if semantic_outputs.len() != op_node.outputs.len() {
            return Err(Error::Internal(format!(
                "semantic graph recording expected {} outputs for {:?}, got {}",
                op_node.outputs.len(),
                op_node.operation,
                semantic_outputs.len()
            )));
        }
        for (output_id, output) in op_node.outputs.iter().copied().zip(semantic_outputs) {
            values.insert(graph.values()[output_id].key.clone(), output);
        }
    }

    graph
        .outputs()
        .iter()
        .map(|&output_id| {
            let key = &graph.values()[output_id].key;
            values.get(key).cloned().map(Some).ok_or_else(|| {
                Error::Internal(format!(
                    "semantic graph recording missing output for {key:?}"
                ))
            })
        })
        .collect()
}

fn record_eager_outputs_from_metadata(
    output_metadata: impl IntoIterator<Item = TensorMeta>,
    semantic_traces: Vec<Option<TracedTensor>>,
    inputs: &[&EagerTensor],
) -> Result<RecordedEagerOutputs> {
    let output_metadata = output_metadata.into_iter().collect::<Vec<_>>();
    if semantic_traces.len() != output_metadata.len() {
        return Err(Error::Internal(format!(
            "eager recording expected {} semantic traces, got {}",
            output_metadata.len(),
            semantic_traces.len()
        )));
    }
    let requires_grad =
        eager_grad_recording_enabled() && inputs.iter().any(|input| input.requires_grad);
    let mut registrations = Vec::with_capacity(output_metadata.len());
    let traces = output_metadata
        .into_iter()
        .map(|metadata| {
            let key = eager_val_key();
            registrations.push((key.clone(), metadata));
            RecordedEagerTrace {
                key,
                trace: None,
                requires_grad,
            }
        })
        .collect();

    Ok(RecordedEagerOutputs {
        traces,
        semantic_traces,
        metadata_scope: Arc::new(register_scoped_metadata_batch(registrations)?),
    })
}

fn tensor_meta_from_value(value: &TensorValue) -> TensorMeta {
    TensorMeta::exact(
        value.dtype(),
        value.shape().iter().copied().map(SymDim::from).collect(),
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

#[cfg(test)]
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
    backend
        .upload_host_tensor(TensorRead::from_tensor(&host))
        .map_err(Error::from)
}

pub(crate) fn one_like_tensor<B: TensorBackend>(input: &Tensor, backend: &mut B) -> Result<Tensor> {
    let host = ones_tensor(input.dtype(), input.shape().to_vec())?;
    backend
        .upload_host_tensor(TensorRead::from_tensor(&host))
        .map_err(Error::from)
}

#[cfg(test)]
mod tests;
