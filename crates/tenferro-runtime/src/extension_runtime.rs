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
use tenferro_tensor::{
    CacheStats, Error as TensorError, ErrorKind, Tensor, TensorBackend, TensorRead,
};

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
    /// A host-reference executor was asked to run an operation without a
    /// host implementation.
    #[error("extension family {family_id:?} has no host reference implementation")]
    MissingHostReference { family_id: &'static str },
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
    ///     shape_guards: Vec::new(),
    /// };
    /// let lhs = Tensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap();
    /// let rhs = Tensor::from_vec_col_major(vec![], vec![2.0_f64]).unwrap();
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
    /// Implementations that need compact tensors must materialize inputs here
    /// explicitly. Keeping this method required prevents implicit read-path
    /// fallbacks from hiding backend or view handling bugs.
    fn execute_reads(
        &self,
        op: &dyn ExtensionOp,
        inputs: &[TensorRead<'_>],
        ctx: &mut ExtensionExecutionContext<'_, B>,
    ) -> tenferro_tensor::Result<Vec<Tensor>>;
}

/// Runtime adapter that delegates execution to an extension op's optional
/// host/reference implementation.
///
/// Register one adapter per extension family. Backend-specific runtimes should
/// implement [`ExtensionRuntime`] directly instead of using this adapter.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{ExtensionRuntime, HostReferenceRuntime};
///
/// let runtime = HostReferenceRuntime::<CpuBackend>::new("example.identity.v1");
/// assert_eq!(runtime.family_id(), "example.identity.v1");
/// ```
#[derive(Clone, Copy)]
pub struct HostReferenceRuntime<B: TensorBackend + 'static> {
    family_id: &'static str,
    _backend: PhantomData<fn() -> B>,
}

impl<B: TensorBackend + 'static> HostReferenceRuntime<B> {
    /// Create a host-reference runtime for one extension family.
    pub fn new(family_id: &'static str) -> Self {
        Self {
            family_id,
            _backend: PhantomData,
        }
    }
}

impl<B: TensorBackend + 'static> Debug for HostReferenceRuntime<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostReferenceRuntime")
            .field("backend_type", &std::any::type_name::<B>())
            .field("family_id", &self.family_id)
            .finish()
    }
}

impl<B: TensorBackend + 'static> ExtensionRuntime<B> for HostReferenceRuntime<B> {
    fn family_id(&self) -> &'static str {
        self.family_id
    }

    fn execute(
        &self,
        op: &dyn ExtensionOp,
        inputs: &[&Tensor],
        _ctx: &mut ExtensionExecutionContext<'_, B>,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        let host = op.host_reference().ok_or_else(|| {
            TensorError::extension(
                "extension",
                op.family_id(),
                ErrorKind::Unsupported,
                ExtensionRuntimeRegistryError::MissingHostReference {
                    family_id: op.family_id(),
                },
            )
        })?;
        host.execute(inputs)
    }

    fn execute_reads(
        &self,
        op: &dyn ExtensionOp,
        inputs: &[TensorRead<'_>],
        ctx: &mut ExtensionExecutionContext<'_, B>,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        let materialized_inputs = ctx.backend_mut().with_backend_session(|exec| {
            inputs
                .iter()
                .cloned()
                .map(|input| exec.to_contiguous_read(input))
                .collect::<tenferro_tensor::Result<Vec<_>>>()
        })?;
        let input_refs: Vec<&Tensor> = materialized_inputs.iter().collect();
        self.execute(op, &input_refs, ctx)
    }
}

fn validate_runtime_output_count(
    op: &dyn ExtensionOp,
    outputs: Vec<Tensor>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let expected = op.output_count();
    if outputs.len() != expected {
        return Err(TensorError::invalid_argument(
            "extension",
            "outputs",
            format!(
                "family_id {:?}: runtime returned {} outputs but op declared {} outputs",
                op.family_id(),
                outputs.len(),
                expected
            ),
        ));
    }
    Ok(outputs)
}

fn validate_runtime_input_count(
    op: &dyn ExtensionOp,
    actual: usize,
) -> tenferro_tensor::Result<()> {
    let expected = op.input_count();
    if actual != expected {
        return Err(TensorError::invalid_argument(
            "extension",
            "inputs",
            format!(
                "family_id {:?}: op expects {} inputs, got {}",
                op.family_id(),
                expected,
                actual
            ),
        ));
    }
    Ok(())
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
        validate_runtime_input_count(op, inputs.len())?;
        let Some(executor) = self.registry.get(op.family_id()) else {
            return Err(TensorError::invalid_argument(
                "extension",
                "runtime",
                format!(
                    "missing runtime for family_id {:?}; register the extension on this runtime owner, for example `executor.register_extension(<extension_crate>::register_runtime)` or `eager_runtime.register_extension(<extension_crate>::register_runtime)`",
                    op.family_id()
                ),
            ));
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
    /// use std::sync::Arc;
    ///
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ops::{ext_op::{ExtensionOp, HostReference}, SymDim};
    /// use tenferro_runtime::{DType, ExtensionExecutor, HostReferenceRuntime, Tensor};
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
    ///     fn payload_hash(&self, _hasher: &mut dyn std::hash::Hasher) {}
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
    ///         ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ///     ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
    ///         Ok(vec![(ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec())])
    ///     }
    ///
    ///     fn host_reference(&self) -> Option<&dyn HostReference> {
    ///         Some(self)
    ///     }
    /// }
    ///
    /// impl HostReference for IdentityOp {
    ///     fn execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
    ///         Ok(vec![inputs[0].clone()])
    ///     }
    /// }
    ///
    /// let mut executor = ExtensionExecutor::<CpuBackend>::new();
    /// executor
    ///     .registry_mut()
    ///     .register(Arc::new(HostReferenceRuntime::<CpuBackend>::new(
    ///         "example.identity.v1",
    ///     )))?;
    /// let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
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
        validate_runtime_input_count(op, inputs.len())?;
        let Some(executor) = self.registry.get(op.family_id()) else {
            return Err(TensorError::invalid_argument(
                "extension",
                "runtime",
                format!(
                    "missing runtime for family_id {:?}; register the extension on this runtime owner, for example `executor.register_extension(<extension_crate>::register_runtime)` or `eager_runtime.register_extension(<extension_crate>::register_runtime)`",
                    op.family_id()
                ),
            ));
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
