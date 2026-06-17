//! Backend-parametric runtime dispatch for extension ops.
//!
//! This module is intentionally generic: extension crates can register an
//! executor for a family and keep runtime cache state outside both the
//! semantic [`ExtensionOp`] payload and the
//! tensor backend implementation.

use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::sync::Arc;

use tenferro_ops::ext_op::ExtensionOp;
use tenferro_tensor::{CacheStats, Tensor, TensorBackend, TensorRead};

use crate::extension_cache::{ExtensionCacheLimits, ExtensionCacheSelector, ExtensionCacheStore};

/// Errors returned by backend-parametric extension runtime registries.
#[derive(Debug, thiserror::Error)]
pub enum ExtensionRuntimeRegistryError {
    /// The `family_id` does not match the namespaced format
    /// `"<crate-name>.<op-name>.v<major>"`.
    #[error("family_id {family_id:?} does not match the namespaced format")]
    MalformedFamilyId { family_id: &'static str },
    /// A registry lock was poisoned by a panic in another thread.
    #[error("{name} poisoned")]
    PoisonedLock { name: &'static str },
}

/// Backend and cache state passed to one extension execution.
pub struct ExtensionExecutionContext<'a, B: TensorBackend> {
    backend: &'a mut B,
    caches: &'a mut ExtensionCacheStore,
}

impl<B: TensorBackend> fmt::Debug for ExtensionExecutionContext<'_, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtensionExecutionContext")
            .field("backend_type", &std::any::type_name::<B>())
            .field("caches", &self.caches)
            .finish_non_exhaustive()
    }
}

impl<'a, B: TensorBackend> ExtensionExecutionContext<'a, B> {
    /// Build a context from externally-owned backend and cache state.
    pub fn new(backend: &'a mut B, caches: &'a mut ExtensionCacheStore) -> Self {
        Self { backend, caches }
    }

    /// Borrow the backend for non-mutating inspection.
    pub fn backend(&self) -> &B {
        self.backend
    }

    /// Borrow the backend mutably for extension execution.
    pub fn backend_mut(&mut self) -> &mut B {
        self.backend
    }

    /// Borrow the extension runtime cache store.
    pub fn caches(&self) -> &ExtensionCacheStore {
        self.caches
    }

    /// Borrow the extension runtime cache store mutably.
    pub fn caches_mut(&mut self) -> &mut ExtensionCacheStore {
        self.caches
    }

    /// Execute a core-only execution program one instruction at a time.
    ///
    /// This is for extension runtimes that lower their own operation into a
    /// temporary `ExecProgram` containing only core tensor ops. Nested
    /// `ExecOp::Extension` instructions are rejected so extension dispatch
    /// cannot bypass the owning runtime registry.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ops::dim_expr::DimExpr;
    /// use tenferro_runtime::extension::{ExecInstruction, ExecOp, ExecProgram};
    /// use tenferro_runtime::{DType, ExtensionCacheStore, ExtensionExecutionContext, Tensor};
    ///
    /// let program = ExecProgram {
    ///     instructions: vec![ExecInstruction {
    ///         op: ExecOp::Add,
    ///         input_slots: vec![0, 1],
    ///         output_slots: vec![2],
    ///         dtype: DType::F64,
    ///         output_shapes: vec![vec![]].into(),
    ///         output_extents: vec![vec![]].into(),
    ///         last_use: vec![true, true],
    ///     }],
    ///     input_slots: vec![0, 1],
    ///     output_slots: vec![2],
    ///     n_slots: 3,
    /// };
    /// let lhs = Tensor::from_vec_col_major(vec![], vec![1.0_f64]);
    /// let rhs = Tensor::from_vec_col_major(vec![], vec![2.0_f64]);
    ///
    /// let mut backend = CpuBackend::new();
    /// let mut caches = ExtensionCacheStore::new();
    /// let mut ctx = ExtensionExecutionContext::new(&mut backend, &mut caches);
    /// let outputs = ctx
    ///     .execute_core_exec_program_unsegmented(&program, vec![lhs, rhs])
    ///     .unwrap();
    /// assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[3.0]);
    /// ```
    pub fn execute_core_exec_program_unsegmented(
        &mut self,
        program: &crate::extension::ExecProgram,
        inputs: Vec<Tensor>,
    ) -> crate::error::Result<Vec<Tensor>>
    where
        B: 'static,
    {
        crate::exec::ensure_core_exec_program(
            program,
            "ExtensionExecutionContext::execute_core_exec_program_unsegmented",
        )?;
        crate::exec::eval_exec_ir_unsegmented_with_cache(self.backend, program, inputs)
    }

    /// Borrow backend and extension cache store as disjoint mutable parts.
    pub fn parts_mut(&mut self) -> (&mut B, &mut ExtensionCacheStore) {
        (self.backend, self.caches)
    }
}

/// A backend-specific runtime executor for one extension family.
pub trait ExtensionRuntime<B: TensorBackend + 'static>: Debug + Send + Sync + 'static {
    /// Extension family handled by this executor.
    fn family_id(&self) -> &'static str;

    /// Execute the extension op with backend and cache state supplied by core.
    fn execute(
        &self,
        op: &dyn ExtensionOp,
        inputs: &[&Tensor],
        ctx: &mut ExtensionExecutionContext<'_, B>,
    ) -> tenferro_tensor::Result<Vec<Tensor>>;

    /// Execute the extension op on borrowed tensor reads.
    ///
    /// Runtime implementations that can consume strided views should override
    /// this method. The default fallback preserves compatibility with
    /// tensor-only runtimes by materializing view reads at this explicit ABI
    /// boundary before delegating to [`ExtensionRuntime::execute`].
    fn execute_reads(
        &self,
        op: &dyn ExtensionOp,
        inputs: &[TensorRead<'_>],
        ctx: &mut ExtensionExecutionContext<'_, B>,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        let concrete_inputs = concrete_tensor_reads(inputs)?;
        let input_refs: Vec<&Tensor> = concrete_inputs
            .iter()
            .map(ConcreteTensorRead::tensor)
            .collect();
        self.execute(op, &input_refs, ctx)
    }
}

enum ConcreteTensorRead<'a> {
    Borrowed(&'a Tensor),
    Owned(Box<Tensor>),
}

impl ConcreteTensorRead<'_> {
    fn tensor(&self) -> &Tensor {
        match self {
            Self::Borrowed(tensor) => tensor,
            Self::Owned(tensor) => tensor.as_ref(),
        }
    }
}

fn concrete_tensor_reads<'a>(
    inputs: &[TensorRead<'a>],
) -> tenferro_tensor::Result<Vec<ConcreteTensorRead<'a>>> {
    let mut concrete_inputs = Vec::with_capacity(inputs.len());
    for input in inputs {
        concrete_inputs.push(match input {
            TensorRead::Tensor(tensor) => ConcreteTensorRead::Borrowed(tensor),
            TensorRead::View(view) => ConcreteTensorRead::Owned(Box::new(view.try_to_tensor()?)),
        });
    }
    Ok(concrete_inputs)
}

fn validate_runtime_output_count(
    op: &dyn ExtensionOp,
    outputs: Vec<Tensor>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let expected = op.output_count();
    if outputs.len() != expected {
        return Err(tenferro_tensor::Error::InvalidConfig {
            op: "extension",
            message: format!(
                "family_id {:?}: runtime returned {} outputs but op declared {} outputs",
                op.family_id(),
                outputs.len(),
                expected
            ),
        });
    }
    Ok(outputs)
}

/// Registry of backend-specific extension runtime executors.
pub struct ExtensionRegistry<B: TensorBackend + 'static> {
    executors: HashMap<&'static str, Arc<dyn ExtensionRuntime<B>>>,
}

impl<B: TensorBackend + 'static> fmt::Debug for ExtensionRegistry<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut families = self.executors.keys().copied().collect::<Vec<_>>();
        families.sort_unstable();
        f.debug_struct("ExtensionRegistry")
            .field("backend_type", &std::any::type_name::<B>())
            .field("len", &self.executors.len())
            .field("families", &families)
            .finish_non_exhaustive()
    }
}

impl<B: TensorBackend + 'static> ExtensionRegistry<B> {
    /// Create an empty extension runtime registry.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::ExtensionRegistry;
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let registry = ExtensionRegistry::<CpuBackend>::new();
    /// assert!(!registry.contains("example.identity.v1"));
    /// ```
    pub fn new() -> Self {
        Self {
            executors: HashMap::new(),
        }
    }

    /// Register one runtime executor.
    ///
    /// Registration is idempotent by family id: registering the same extension
    /// family more than once succeeds and keeps the first runtime. This lets
    /// extension crates register their own dependency extensions defensively.
    pub fn register(
        &mut self,
        executor: Arc<dyn ExtensionRuntime<B>>,
    ) -> Result<(), ExtensionRuntimeRegistryError> {
        let family_id = executor.family_id();
        if !is_valid_family_id(family_id) {
            return Err(ExtensionRuntimeRegistryError::MalformedFamilyId { family_id });
        }
        if self.executors.contains_key(family_id) {
            return Ok(());
        }
        self.executors.insert(family_id, executor);
        Ok(())
    }

    /// Look up an executor by extension family id.
    pub fn get(&self, family_id: &str) -> Option<Arc<dyn ExtensionRuntime<B>>> {
        self.executors.get(family_id).cloned()
    }

    /// Return whether an executor is registered for `family_id`.
    pub fn contains(&self, family_id: &str) -> bool {
        self.executors.contains_key(family_id)
    }

    /// Number of registered runtime executors.
    pub fn len(&self) -> usize {
        self.executors.len()
    }

    /// Return whether no runtime executors are registered.
    pub fn is_empty(&self) -> bool {
        self.executors.is_empty()
    }
}

impl<B: TensorBackend + 'static> Default for ExtensionRegistry<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime owner for backend-specific extension dispatch and caches.
pub struct ExtensionExecutor<B: TensorBackend + 'static> {
    registry: ExtensionRegistry<B>,
    caches: ExtensionCacheStore,
    _backend: PhantomData<fn() -> B>,
}

impl<B: TensorBackend + 'static> fmt::Debug for ExtensionExecutor<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtensionExecutor")
            .field("backend_type", &std::any::type_name::<B>())
            .field("registry", &self.registry)
            .field("caches", &self.caches)
            .finish_non_exhaustive()
    }
}

impl<B: TensorBackend + 'static> ExtensionExecutor<B> {
    /// Create an executor with an empty registry and default cache limits.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::ExtensionExecutor;
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let executor = ExtensionExecutor::<CpuBackend>::new();
    /// assert_eq!(executor.cache_stats().entries, 0);
    /// ```
    pub fn new() -> Self {
        Self {
            registry: ExtensionRegistry::new(),
            caches: ExtensionCacheStore::new(),
            _backend: PhantomData,
        }
    }

    /// Create an executor from explicit registry and cache store.
    pub fn with_parts(registry: ExtensionRegistry<B>, caches: ExtensionCacheStore) -> Self {
        Self {
            registry,
            caches,
            _backend: PhantomData,
        }
    }

    /// Borrow the runtime executor registry.
    pub fn registry(&self) -> &ExtensionRegistry<B> {
        &self.registry
    }

    /// Borrow the runtime executor registry mutably.
    pub fn registry_mut(&mut self) -> &mut ExtensionRegistry<B> {
        &mut self.registry
    }

    /// Borrow the extension cache store.
    pub fn caches(&self) -> &ExtensionCacheStore {
        &self.caches
    }

    /// Borrow the extension cache store mutably.
    pub fn caches_mut(&mut self) -> &mut ExtensionCacheStore {
        &mut self.caches
    }

    /// Execute an extension using a registered runtime executor.
    pub fn execute(
        &mut self,
        backend: &mut B,
        op: &dyn ExtensionOp,
        inputs: &[&Tensor],
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        let Some(executor) = self.registry.get(op.family_id()) else {
            return Err(tenferro_tensor::Error::InvalidConfig {
                op: "extension",
                message: format!(
                    "missing runtime for family_id {:?}; register the extension on this runtime owner, for example `executor.register_extension(<extension_crate>::register_runtime)` or `eager_runtime.register_extension(<extension_crate>::register_runtime)`",
                    op.family_id()
                ),
            });
        };
        let mut ctx = ExtensionExecutionContext::new(backend, &mut self.caches);
        validate_runtime_output_count(op, executor.execute(op, inputs, &mut ctx)?)
    }

    /// Execute an extension using borrowed tensor reads.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::any::Any;
    /// use std::hash::Hasher;
    /// use std::sync::Arc;
    ///
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ops::{ext_op::ExtensionOp, SymDim};
    /// use tenferro_runtime::{
    ///     DType, ExtensionExecutionContext, ExtensionExecutor, ExtensionRuntime, Tensor,
    /// };
    /// use tenferro_tensor::TensorRead;
    ///
    /// #[derive(Clone, Debug)]
    /// struct IdentityOp;
    ///
    /// impl ExtensionOp for IdentityOp {
    ///     fn family_id(&self) -> &'static str {
    ///         "example.identity.v1"
    ///     }
    ///
    ///     fn payload_hash(&self, _hasher: &mut dyn Hasher) {}
    ///
    ///     fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
    ///         other.as_any().is::<IdentityOp>()
    ///     }
    ///
    ///     fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
    ///         Arc::new(self.clone())
    ///     }
    ///
    ///     fn as_any(&self) -> &dyn Any {
    ///         self
    ///     }
    ///
    ///     fn input_count(&self) -> usize {
    ///         1
    ///     }
    ///
    ///     fn output_count(&self) -> usize {
    ///         1
    ///     }
    ///
    ///     fn infer_output_meta(
    ///         &self,
    ///         input_dtypes: &[DType],
    ///         input_shapes: &[&[SymDim]],
    ///     ) -> Vec<(DType, Vec<SymDim>)> {
    ///         vec![(input_dtypes[0], input_shapes[0].to_vec())]
    ///     }
    ///
    ///     fn eager_execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
    ///         Ok(vec![inputs[0].clone()])
    ///     }
    /// }
    ///
    /// #[derive(Debug)]
    /// struct IdentityRuntime;
    ///
    /// impl ExtensionRuntime<CpuBackend> for IdentityRuntime {
    ///     fn family_id(&self) -> &'static str {
    ///         "example.identity.v1"
    ///     }
    ///
    ///     fn execute(
    ///         &self,
    ///         op: &dyn ExtensionOp,
    ///         inputs: &[&Tensor],
    ///         _ctx: &mut ExtensionExecutionContext<'_, CpuBackend>,
    ///     ) -> tenferro_tensor::Result<Vec<Tensor>> {
    ///         op.eager_execute(inputs)
    ///     }
    /// }
    ///
    /// let mut executor = ExtensionExecutor::<CpuBackend>::new();
    /// executor.registry_mut().register(Arc::new(IdentityRuntime))?;
    /// let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    /// let read = TensorRead::from_tensor(&input);
    /// let mut backend = CpuBackend::new();
    ///
    /// let outputs = executor.execute_reads(&mut backend, &IdentityOp, &[read])?;
    ///
    /// assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn execute_reads(
        &mut self,
        backend: &mut B,
        op: &dyn ExtensionOp,
        inputs: &[TensorRead<'_>],
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        let Some(executor) = self.registry.get(op.family_id()) else {
            return Err(tenferro_tensor::Error::InvalidConfig {
                op: "extension",
                message: format!(
                    "missing runtime for family_id {:?}; register the extension on this runtime owner, for example `executor.register_extension(<extension_crate>::register_runtime)` or `eager_runtime.register_extension(<extension_crate>::register_runtime)`",
                    op.family_id()
                ),
            });
        };
        let mut ctx = ExtensionExecutionContext::new(backend, &mut self.caches);
        validate_runtime_output_count(op, executor.execute_reads(op, inputs, &mut ctx)?)
    }

    /// Clear every runtime extension cache entry.
    pub fn clear_caches(&mut self) {
        self.caches.clear();
    }

    /// Return extension cache stats for all entries.
    pub fn cache_stats(&self) -> CacheStats {
        self.caches.stats(ExtensionCacheSelector::All)
    }

    /// Return the extension cache retention limits.
    pub fn cache_limits(&self) -> ExtensionCacheLimits {
        self.caches.limits()
    }

    /// Replace extension cache retention limits.
    pub fn set_cache_limits(&mut self, limits: ExtensionCacheLimits) {
        self.caches.set_limits(limits);
    }
}

impl<B: TensorBackend + 'static> Default for ExtensionExecutor<B> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;

fn is_valid_family_id(family_id: &str) -> bool {
    let mut parts = family_id.rsplitn(2, '.');
    let Some(version_part) = parts.next() else {
        return false;
    };
    let Some(prefix) = parts.next() else {
        return false;
    };
    if !version_part.starts_with('v') {
        return false;
    }
    let digits = &version_part[1..];
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    let Some((crate_name, op_name)) = prefix.split_once('.') else {
        return false;
    };
    if crate_name.is_empty() || op_name.is_empty() {
        return false;
    }
    let any_invalid = |s: &str| s.chars().any(|c| c.is_whitespace() || !c.is_ascii());
    !any_invalid(crate_name) && !any_invalid(op_name)
}
