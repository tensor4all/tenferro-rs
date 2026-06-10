use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::HashMap;
use std::env;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use crate::extension_cache::ExtensionCacheLimits;
use crate::extension_runtime::{ExtensionExecutor, ExtensionRuntimeRegistryError};
use computegraph::ValueKey;
use tenferro_cpu::CpuBackend;
#[cfg(feature = "cuda")]
use tenferro_gpu::cubecl::CubeclBackend;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::ExtensionRuleSet;
use tenferro_ops::ShapeGuardContext;
use tenferro_tensor::{
    CacheStats, DType, Tensor, TensorBackend, TensorElementwise, TensorRead, TensorValue,
    TypedTensor,
};
use tidu::eager::{self, EagerInput, EagerOutput, KeySource, Recorder, Trace};

use self::backward::TenferroBackwardCallbacks;
use crate::eager_backend::EagerBackend;
use crate::eager_exec::{
    exec_op_on_tensor_reads_with_extension_executor, exec_op_on_tensors_with_extension_executor,
};
use crate::error::{ContextId, Error, Result};
use crate::metadata::{
    metadata_scopes_for_scope, register_scoped_metadata_batch, register_scoped_value_metadata,
    tensor_meta_from_tensor, MetadataScope,
};
use crate::traced::next_input_key;

use crate::AdContext;

mod backward;

pub(crate) type GradSlot = Arc<Mutex<Option<Arc<Tensor>>>>;
pub(crate) type WeakGradSlot = Weak<Mutex<Option<Arc<Tensor>>>>;

#[derive(Debug, Default, Clone)]
struct EagerOpProfileEntry {
    calls: usize,
    total_time: Duration,
}

thread_local! {
    static EAGER_OP_PROFILE_STATE: RefCell<HashMap<&'static str, EagerOpProfileEntry>> =
        RefCell::new(HashMap::new());
}

pub(crate) fn eager_op_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env::var("TENFERRO_PROFILE_EAGER_OP_AGG").is_ok())
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
    let Ok(print_every) = env::var("TENFERRO_PROFILE_EAGER_OP_PRINT_EVERY") else {
        return;
    };
    let Ok(print_every) = print_every.parse::<usize>() else {
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
            eprintln!(
                "{section}: calls={} total={:.6}ms per_call={:.3}us",
                entry.calls,
                entry.total_time.as_secs_f64() * 1.0e3,
                entry.total_time.as_secs_f64() * 1.0e6 / entry.calls as f64,
            );
        }
    });
}

/// Stats for caches owned by an [`EagerRuntime`].
///
/// `retained_bytes` fields are logical payload estimates, not process RSS.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EagerRuntimeCacheStats {
    /// Generic extension runtime caches.
    pub extensions: CacheStats,
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
/// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![1.0_f64]), ctx.clone());
/// let y = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![2.0_f64]), ctx);
/// let z = &x + &y;
///
/// assert_eq!(z.data().as_slice::<f64>().unwrap(), &[3.0]);
/// ```
pub struct EagerRuntime {
    pub(crate) backend: Mutex<EagerBackend>,
    pub(crate) extension_executor: Mutex<ExtensionExecutor<EagerBackend>>,
    extension_rules: Option<ExtensionRuleSet>,
    grad_slots: Mutex<HashMap<ValueKey<StdTensorOp>, WeakGradSlot>>,
}

impl EagerRuntime {
    fn from_backend(backend: EagerBackend) -> Self {
        Self::from_backend_with_extension_rules(backend, None)
    }

    fn from_backend_with_extension_rules(
        backend: EagerBackend,
        extension_rules: Option<ExtensionRuleSet>,
    ) -> Self {
        Self {
            backend: Mutex::new(backend),
            extension_executor: Mutex::new(ExtensionExecutor::new()),
            extension_rules,
            grad_slots: Mutex::new(HashMap::new()),
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
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1));
    /// assert_eq!(std::sync::Arc::strong_count(&ctx), 1);
    /// ```
    pub fn with_cpu_backend(backend: CpuBackend) -> Arc<Self> {
        Arc::new(Self::from_backend(EagerBackend::cpu(backend)))
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
        Arc::new(Self::from_backend_with_extension_rules(
            EagerBackend::cpu(backend),
            Some(ad.extension_rule_set()),
        ))
    }

    /// Create a shared eager execution context from a configured CUDA backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::cubecl::CubeclBackend;
    /// use tenferro_ad::EagerRuntime;
    ///
    /// let _ctor: fn(CubeclBackend) -> std::sync::Arc<EagerRuntime> =
    ///     EagerRuntime::with_cuda_backend;
    /// ```
    #[cfg(feature = "cuda")]
    pub fn with_cuda_backend(backend: CubeclBackend) -> Arc<Self> {
        Arc::new(Self::from_backend(EagerBackend::cuda(backend)))
    }

    /// Create a shared CUDA eager context with explicit AD extension rules.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::{AdContext, EagerRuntime};
    /// use tenferro_gpu::cubecl::CubeclBackend;
    ///
    /// let _ctor: fn(CubeclBackend, &AdContext) -> std::sync::Arc<EagerRuntime> =
    ///     EagerRuntime::with_cuda_backend_and_ad_context;
    /// ```
    #[cfg(feature = "cuda")]
    pub fn with_cuda_backend_and_ad_context(backend: CubeclBackend, ad: &AdContext) -> Arc<Self> {
        Arc::new(Self::from_backend_with_extension_rules(
            EagerBackend::cuda(backend),
            Some(ad.extension_rule_set()),
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
        ContextId::from_ptr(self)
    }

    /// Register one extension runtime on this eager context.
    pub fn register_extension(
        &self,
        register: impl FnOnce(
            &mut ExtensionExecutor<EagerBackend>,
        ) -> std::result::Result<(), ExtensionRuntimeRegistryError>,
    ) -> std::result::Result<(), ExtensionRuntimeRegistryError> {
        register(&mut self.extension_executor.lock().unwrap())
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
    /// ctx.clear_extension_caches();
    /// assert_eq!(ctx.cache_stats().extensions.entries, 0);
    /// ```
    pub fn clear_extension_caches(&self) {
        self.extension_executor.lock().unwrap().clear_caches();
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
    /// ctx.clear_caches();
    /// assert_eq!(ctx.cache_stats().extensions.entries, 0);
    /// ```
    pub fn clear_caches(&self) {
        self.clear_extension_caches();
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
    /// let stats = ctx.cache_stats();
    /// assert_eq!(stats.extensions.entries, 0);
    /// ```
    pub fn cache_stats(&self) -> EagerRuntimeCacheStats {
        EagerRuntimeCacheStats {
            extensions: self.extension_executor.lock().unwrap().cache_stats(),
        }
    }

    /// Return the extension cache retention limits.
    pub fn extension_cache_limits(&self) -> ExtensionCacheLimits {
        self.extension_executor.lock().unwrap().cache_limits()
    }

    /// Replace extension cache retention limits.
    pub fn set_extension_cache_limits(&self, limits: ExtensionCacheLimits) {
        self.extension_executor
            .lock()
            .unwrap()
            .set_cache_limits(limits);
    }

    /// Block the current thread until backend work submitted by this eager runtime completes.
    ///
    /// CPU runtimes return immediately. CUDA runtimes synchronize the current backend stream.
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
    pub fn synchronize(&self) -> Result<()> {
        self.backend
            .lock()
            .unwrap()
            .synchronize()
            .map_err(Error::from)
    }

    pub(crate) fn exec_outputs(&self, op: &StdTensorOp, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        let mut backend =
            profile_eager_op_section("exec_outputs.lock_backend", || self.backend.lock().unwrap());
        let mut extension_executor =
            profile_eager_op_section("exec_outputs.lock_extensions", || {
                self.extension_executor.lock().unwrap()
            });
        profile_eager_op_section("exec_outputs.exec_op", || {
            exec_op_on_tensors_with_extension_executor(
                op,
                inputs,
                &mut *backend,
                Some(&mut *extension_executor),
            )
        })
    }

    pub(crate) fn exec_outputs_read(
        &self,
        op: &StdTensorOp,
        inputs: &[TensorRead<'_>],
    ) -> Result<Vec<Tensor>> {
        let mut backend = profile_eager_op_section("exec_outputs_read.lock_backend", || {
            self.backend.lock().unwrap()
        });
        let mut extension_executor =
            profile_eager_op_section("exec_outputs_read.lock_extensions", || {
                self.extension_executor.lock().unwrap()
            });
        profile_eager_op_section("exec_outputs_read.exec_op", || {
            exec_op_on_tensor_reads_with_extension_executor(
                op,
                inputs,
                &mut *backend,
                Some(&mut *extension_executor),
            )
        })
    }

    pub(crate) fn register_grad_slot(&self, key: &ValueKey<StdTensorOp>, slot: &GradSlot) {
        self.grad_slots
            .lock()
            .unwrap()
            .insert(key.clone(), Arc::downgrade(slot));
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
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]), ctx.clone());
    /// let y = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]), ctx.clone());
    /// let loss = (&x * &y).reduce_sum(&[0]).unwrap();
    /// let _ = loss.backward().unwrap();
    ///
    /// ctx.clear_grads();
    ///
    /// assert!(x.grad().is_none());
    /// assert!(y.grad().is_none());
    /// ```
    pub fn clear_grads(&self) {
        self.grad_slots.lock().unwrap().retain(|_, slot| {
            if let Some(slot) = slot.upgrade() {
                *slot.lock().unwrap() = None;
                true
            } else {
                false
            }
        });
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
    /// let c = ctx.constant_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]));
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]), ctx);
    /// let z = x.add(&c).unwrap();
    ///
    /// assert_eq!(z.data().as_slice::<f64>().unwrap(), &[4.0, 6.0]);
    /// ```
    pub fn constant_from(self: &Arc<Self>, tensor: Tensor) -> EagerTensor {
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
    /// let p = ctx.variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]));
    /// let loss = p.exp().unwrap().reduce_sum(&[0]).unwrap();
    /// let _ = loss.backward().unwrap();
    ///
    /// let grad = p.grad().unwrap();
    /// assert_eq!(grad.shape(), &[2]);
    /// ```
    pub fn variable_from(self: &Arc<Self>, tensor: Tensor) -> EagerTensor {
        EagerTensor::new_leaf(Arc::clone(self), tensor, true)
    }

    fn store_grads(
        &self,
        cotangents: &HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
        backend: &mut EagerBackend,
    ) -> Result<()> {
        let mut updates = Vec::new();
        let mut staged = Vec::new();

        {
            let mut slots = self.grad_slots.lock().unwrap();
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
            let next = {
                let current = slot.lock().unwrap();
                match current.as_ref() {
                    Some(existing) => Arc::new(backend.add(existing.as_ref(), incoming.as_ref())?),
                    None => incoming,
                }
            };
            staged.push((slot, next));
        }

        for (slot, next) in staged {
            *slot.lock().unwrap() = Some(next);
        }

        Ok(())
    }
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
/// let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]), ctx);
/// let loss = (&x * &x).reduce_sum(&[0]).unwrap();
/// let _cotangents = loss.backward().unwrap();
/// let loss = (&x * &x).reduce_sum(&[0]).unwrap();
/// let _cotangents = loss.backward().unwrap();
///
/// assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[4.0, 8.0, 12.0]);
/// x.clear_grad();
///
/// assert!(x.grad().is_none());
/// ```
#[derive(Clone)]
pub struct EagerTensor {
    pub(crate) value: Arc<TensorValue>,
    materialized_cache: Arc<OnceLock<Arc<Tensor>>>,
    pub(crate) key: ValueKey<StdTensorOp>,
    pub(crate) trace: Option<Trace<StdTensorOp>>,
    pub(crate) requires_grad: bool,
    grad_slot: GradSlot,
    pub(crate) metadata_scopes: Vec<Arc<MetadataScope>>,
    pub(crate) ctx: Arc<EagerRuntime>,
}

impl std::ops::Add for &EagerTensor {
    type Output = EagerTensor;

    fn add(self, rhs: &EagerTensor) -> Self::Output {
        EagerTensor::add(self, rhs).unwrap_or_else(|err| panic!("eager add failed: {}", err))
    }
}

impl std::ops::Mul for &EagerTensor {
    type Output = EagerTensor;

    fn mul(self, rhs: &EagerTensor) -> Self::Output {
        EagerTensor::mul(self, rhs).unwrap_or_else(|err| panic!("eager mul failed: {}", err))
    }
}

impl std::ops::Neg for &EagerTensor {
    type Output = EagerTensor;

    fn neg(self) -> Self::Output {
        EagerTensor::neg(self).unwrap_or_else(|err| panic!("eager neg failed: {}", err))
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
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]), ctx);
    ///
    /// assert_eq!(x.data().as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    /// ```
    pub fn from_tensor_in(tensor: Tensor, ctx: Arc<EagerRuntime>) -> Self {
        Self::new_leaf(ctx, tensor, false)
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
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]), ctx);
    ///
    /// assert!(x.grad().is_none());
    /// ```
    pub fn requires_grad_in(tensor: Tensor, ctx: Arc<EagerRuntime>) -> Self {
        Self::new_leaf(ctx, tensor, true)
    }

    pub(crate) fn new_leaf(ctx: Arc<EagerRuntime>, tensor: Tensor, requires_grad: bool) -> Self {
        let key = eager_val_key();
        let metadata_scope =
            register_scoped_value_metadata(key.clone(), tensor_meta_from_tensor(&tensor));
        let tensor = Arc::new(tensor);
        let grad_slot = Arc::new(Mutex::new(None));
        if requires_grad {
            ctx.register_grad_slot(&key, &grad_slot);
        }

        Self {
            value: Arc::new(TensorValue::from_tensor_arc(tensor)),
            materialized_cache: Arc::new(OnceLock::new()),
            key,
            trace: None,
            requires_grad,
            grad_slot,
            metadata_scopes: metadata_scopes_for_scope(metadata_scope),
            ctx,
        }
    }

    pub(crate) fn new_result(
        ctx: Arc<EagerRuntime>,
        key: ValueKey<StdTensorOp>,
        tensor: Tensor,
        requires_grad: bool,
        trace: Option<Trace<StdTensorOp>>,
        metadata_scopes: Vec<Arc<MetadataScope>>,
    ) -> Self {
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
        metadata_scopes: Vec<Arc<MetadataScope>>,
    ) -> Self {
        let grad_slot = Arc::new(Mutex::new(None));
        if requires_grad {
            ctx.register_grad_slot(&key, &grad_slot);
        }

        Self {
            value: Arc::new(TensorValue::from_tensor_arc(tensor)),
            materialized_cache: Arc::new(OnceLock::new()),
            key,
            trace,
            requires_grad,
            grad_slot,
            metadata_scopes,
            ctx,
        }
    }

    pub(crate) fn new_result_value(
        ctx: Arc<EagerRuntime>,
        key: ValueKey<StdTensorOp>,
        value: TensorValue,
        requires_grad: bool,
        trace: Option<Trace<StdTensorOp>>,
        metadata_scopes: Vec<Arc<MetadataScope>>,
    ) -> Self {
        let grad_slot = Arc::new(Mutex::new(None));
        if requires_grad {
            ctx.register_grad_slot(&key, &grad_slot);
        }

        Self {
            value: Arc::new(value),
            materialized_cache: Arc::new(OnceLock::new()),
            key,
            trace,
            requires_grad,
            grad_slot,
            metadata_scopes,
            ctx,
        }
    }

    pub(crate) fn new_untracked_result(ctx: Arc<EagerRuntime>, tensor: Tensor) -> Self {
        Self::new_result(ctx, eager_val_key(), tensor, false, None, Vec::new())
    }

    pub(crate) fn new_untracked_value_result(ctx: Arc<EagerRuntime>, value: TensorValue) -> Self {
        Self {
            value: Arc::new(value),
            materialized_cache: Arc::new(OnceLock::new()),
            key: eager_val_key(),
            trace: None,
            requires_grad: false,
            grad_slot: Arc::new(Mutex::new(None)),
            metadata_scopes: Vec::new(),
            ctx,
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
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]), ctx);
    /// let y = x.detach();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    /// assert!(y.grad().is_none());
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
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]), ctx_a);
    /// let d = x.detach_into(&ctx_b);
    ///
    /// assert!(!d.tracks_grad());
    /// assert_eq!(d.ctx_id(), ctx_b.id());
    /// ```
    pub fn detach_into(&self, ctx: &Arc<EagerRuntime>) -> Self {
        Self::new_untracked_value_result(Arc::clone(ctx), self.value.as_ref().clone())
    }

    /// Borrow the concrete tensor value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![3.0_f64]), ctx);
    /// assert_eq!(x.data().as_slice::<f64>().unwrap(), &[3.0]);
    /// ```
    pub fn data(&self) -> &Tensor {
        if let Some(tensor) = self.value.as_tensor_arc() {
            return tensor.as_ref();
        }
        self.materialized_cache
            .get_or_init(|| Arc::new(self.value.to_tensor()))
            .as_ref()
    }

    /// Return this tensor's scalar dtype without materializing through
    /// [`data`](Self::data).
    pub fn dtype(&self) -> DType {
        self.value.dtype()
    }

    /// Return this tensor's logical shape without materializing through
    /// [`data`](Self::data).
    pub fn shape(&self) -> &[usize] {
        self.value.shape()
    }

    /// Borrow this tensor value as a [`TensorRead`].
    ///
    /// This is the preferred borrowed input boundary for executor calls. It
    /// preserves the option to replace eager storage with non-contiguous views
    /// without forcing callers through [`data`](Self::data).
    pub fn tensor_read(&self) -> TensorRead<'_> {
        self.value.tensor_read()
    }

    /// Materialize this eager tensor as an owned [`Tensor`].
    ///
    /// Today eager tensors are stored as compact tensors, so this clones the
    /// current value. This method is the intended compatibility boundary for
    /// callers that need owned materialized data after lazy view storage is
    /// introduced.
    pub fn to_tensor(&self) -> Result<Tensor> {
        Ok(self.value.to_tensor())
    }

    pub(crate) fn materialized_arc(&self) -> Arc<Tensor> {
        if let Some(tensor) = self.value.as_tensor_arc() {
            return Arc::clone(tensor);
        }
        Arc::clone(
            self.materialized_cache
                .get_or_init(|| Arc::new(self.value.to_tensor())),
        )
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
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]), ctx);
    /// let loss = x.exp().unwrap().reduce_sum(&[0]).unwrap();
    /// let _cotangents = loss.backward().unwrap();
    ///
    /// let grad = x.grad().unwrap();
    /// assert_eq!(grad.shape(), &[2]);
    /// ```
    pub fn grad(&self) -> Option<Arc<Tensor>> {
        self.grad_slot.lock().unwrap().clone()
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
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]), ctx.clone());
    /// let y = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]), ctx);
    /// let loss = (&x * &y).reduce_sum(&[0]).unwrap();
    /// let _ = loss.backward().unwrap();
    ///
    /// x.clear_grad();
    ///
    /// assert!(x.grad().is_none());
    /// assert!(y.grad().is_some());
    /// ```
    pub fn clear_grad(&self) {
        *self.grad_slot.lock().unwrap() = None;
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
    /// let plain = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]), ctx.clone());
    /// let tracked = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]), ctx.clone());
    /// let detached = tracked.detach();
    ///
    /// assert!(!plain.tracks_grad());
    /// assert!(tracked.tracks_grad());
    /// assert!(!detached.tracks_grad());
    /// ```
    pub fn tracks_grad(&self) -> bool {
        self.requires_grad
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
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![1.0_f64]), ctx.clone());
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
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![1.0_f64]), ctx.clone());
    /// let y = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![2.0_f64]), ctx);
    ///
    /// assert!(x.same_context(&y));
    /// ```
    pub fn same_context(&self, other: &Self) -> bool {
        self.ctx_id() == other.ctx_id()
    }

    /// Run reverse-mode AD from this scalar output.
    ///
    /// Returns the full cotangent map produced by the reverse pass and also
    /// accumulates into `grad()` for tracked eager tensors reachable from this
    /// output.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]), ctx);
    /// let loss = (&x + &x).reduce_sum(&[0]).unwrap();
    /// let _cotangents = loss.backward().unwrap();
    /// let loss = (&x + &x).reduce_sum(&[0]).unwrap();
    /// let _cotangents = loss.backward().unwrap();
    ///
    /// assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[4.0, 4.0, 4.0]);
    /// ```
    pub fn backward(&self) -> Result<HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>> {
        if !self.shape().is_empty() {
            return Err(Error::NonScalarGrad {
                shape: self.shape().to_vec(),
            });
        }

        let value = self.materialized_arc();
        let mut backend = self.ctx.backend.lock().unwrap();
        let mut extension_executor = self.ctx.extension_executor.lock().unwrap();
        let seed = Arc::new(one_like_tensor(value.as_ref(), &mut *backend));
        let mut callbacks = TenferroBackwardCallbacks::new(
            &mut *backend,
            Some(&mut *extension_executor),
            self.metadata_scopes.clone(),
        );
        let mut ad_ctx = ShapeGuardContext::with_global_metadata();
        if let Some(extension_rules) = &self.ctx.extension_rules {
            ad_ctx = ad_ctx.with_extension_rules(extension_rules.clone());
        }
        let cotangents = eager::try_backward(
            &self.key,
            self.trace.as_ref(),
            seed,
            &mut callbacks,
            &mut ad_ctx,
        )
        .map_err(|err| Error::Internal(err.to_string()))?;
        drop(callbacks);
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

pub(crate) fn eager_value(tensor: &EagerTensor) -> EagerInput<StdTensorOp> {
    EagerInput {
        key: tensor.key.clone(),
        trace: tensor.trace.clone(),
        requires_grad: tensor.requires_grad,
        data: tensor.materialized_arc(),
    }
}

pub(crate) struct RecordedEagerOutputs {
    pub(crate) traces: Vec<EagerOutput<StdTensorOp>>,
    pub(crate) metadata_scope: Arc<MetadataScope>,
}

pub(crate) fn record_eager_outputs(
    op: &StdTensorOp,
    outputs: &[Arc<Tensor>],
    inputs: &[&EagerTensor],
) -> RecordedEagerOutputs {
    let input_values: Vec<_> = inputs.iter().map(|tensor| eager_value(tensor)).collect();
    let mut recorder = Recorder::new(EagerTensorKeySource);
    let traces = recorder.record(op.clone(), &input_values, outputs);

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

    RecordedEagerOutputs {
        traces,
        metadata_scope: Arc::new(register_scoped_metadata_batch(registrations)),
    }
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

pub(crate) fn zero_like_tensor<B: TensorBackend>(input: &Tensor, backend: &mut B) -> Tensor {
    let host = match input {
        Tensor::F32(tensor) => Tensor::F32(TypedTensor::zeros(tensor.shape().to_vec())),
        Tensor::F64(tensor) => Tensor::F64(TypedTensor::zeros(tensor.shape().to_vec())),
        Tensor::I32(tensor) => Tensor::I32(TypedTensor::zeros(tensor.shape().to_vec())),
        Tensor::I64(tensor) => Tensor::I64(TypedTensor::zeros(tensor.shape().to_vec())),
        Tensor::Bool(tensor) => Tensor::Bool(TypedTensor::from_vec_col_major(
            tensor.shape().to_vec(),
            vec![false; tensor.n_elements()],
        )),
        Tensor::C32(tensor) => Tensor::C32(TypedTensor::zeros(tensor.shape().to_vec())),
        Tensor::C64(tensor) => Tensor::C64(TypedTensor::zeros(tensor.shape().to_vec())),
    };
    backend
        .upload_host_tensor(&host)
        .unwrap_or_else(|err| panic!("zero_like upload failed: {}", err))
}

pub(crate) fn one_like_tensor<B: TensorBackend>(input: &Tensor, backend: &mut B) -> Tensor {
    let zero = zero_like_tensor(input, backend);
    backend
        .exp(&zero)
        .unwrap_or_else(|err| panic!("one_like exp failed: {}", err))
}

#[cfg(test)]
mod tests;
