//! Runtime-owned context passed to prepared extension operations.
//!
//! Extension dispatch is owned by [`crate::Runtime`] through installed
//! [`crate::ExtensionModule`] values. This module intentionally exposes only the
//! backend/cache context that prepared operations receive at execution time.

use std::fmt;

use tenferro_tensor::TensorBackend;

use crate::extension_cache::ExtensionCacheStore;

/// Backend and cache state passed to one prepared extension execution.
///
/// Extension crates should obtain this value from their
/// [`crate::PreparedOperation::execute`] implementation and use it only for the
/// duration of that call.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{
///     ExtensionCacheSelector, ExtensionCacheStore, ExtensionExecutionContext,
/// };
///
/// let mut backend = CpuBackend::new();
/// let mut caches = ExtensionCacheStore::new();
/// let context = ExtensionExecutionContext::new(&mut backend, &mut caches);
///
/// assert_eq!(context.caches().stats(ExtensionCacheSelector::All).entries, 0);
/// ```
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

    /// Borrow backend and extension cache store as disjoint mutable parts.
    pub fn parts_mut(&mut self) -> (&mut B, &mut ExtensionCacheStore) {
        (self.backend, self.caches)
    }
}
