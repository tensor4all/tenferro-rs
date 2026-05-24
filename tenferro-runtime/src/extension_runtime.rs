//! Backend-parametric runtime dispatch for extension ops.
//!
//! This module is intentionally generic: extension crates can register an
//! executor for a family and keep runtime cache state outside both the
//! semantic [`ExtensionOp`](tenferro_ops::ext_op::ExtensionOp) payload and the
//! tensor backend implementation.

use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

use tenferro_ops::ext_op::ExtensionOp;
use tenferro_tensor::{CacheStats, Tensor, TensorBackend};

use crate::extension_cache::{ExtensionCacheLimits, ExtensionCacheSelector, ExtensionCacheStore};

/// Errors returned by backend-parametric extension runtime registries.
#[derive(Debug, thiserror::Error)]
pub enum ExtensionRuntimeRegistryError {
    /// The `family_id` does not match the namespaced format
    /// `"<crate-name>.<op-name>.v<major>"`.
    #[error("family_id {family_id:?} does not match the namespaced format")]
    MalformedFamilyId { family_id: &'static str },
}

/// Backend and cache state passed to one extension execution.
pub struct ExtensionExecutionContext<'a, B: TensorBackend> {
    backend: &'a mut B,
    caches: &'a mut ExtensionCacheStore,
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
}

/// Registry of backend-specific extension runtime executors.
pub struct ExtensionRegistry<B: TensorBackend + 'static> {
    executors: HashMap<&'static str, Arc<dyn ExtensionRuntime<B>>>,
}

impl<B: TensorBackend + 'static> ExtensionRegistry<B> {
    /// Create an empty extension runtime registry.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::ExtensionRegistry;
    /// use tenferro_tensor::cpu::CpuBackend;
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

impl<B: TensorBackend + 'static> ExtensionExecutor<B> {
    /// Create an executor with an empty registry and default cache limits.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::ExtensionExecutor;
    /// use tenferro_tensor::cpu::CpuBackend;
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
        executor.execute(op, inputs, &mut ctx)
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
