//! Runtime-owned context passed to prepared extension operations.
//!
//! Extension dispatch is owned by [`crate::Runtime`] through installed
//! [`crate::ExtensionModule`] values. This module intentionally exposes only the
//! backend/cache context that prepared operations receive at execution time.

use std::fmt;

use tenferro_tensor::BackendSession;

use crate::extension_cache::ExtensionCacheStore;

/// Backend and cache state passed to one prepared extension execution.
///
/// Extension crates should obtain this value from their hidden
/// [`crate::PreparedOperationExecutor`] bridge and use it only for the duration
/// of that call.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuBackend;
/// use tenferro_tensor::{BackendSessionHost, Tensor};
/// use tenferro_runtime::{
///     ExtensionCacheSelector, ExtensionCacheStore, ExtensionExecutionContext,
/// };
///
/// let mut backend = CpuBackend::new();
/// let mut caches = ExtensionCacheStore::new();
/// backend.with_backend_session(|session| {
///     let mut context = ExtensionExecutionContext::new(session, &mut caches);
///     let lhs = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
///     let rhs = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
///     let output = context.backend_mut().add(&lhs, &rhs).unwrap();
///
///     assert_eq!(output.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
///     assert_eq!(context.caches().stats(ExtensionCacheSelector::All).entries, 0);
/// });
/// ```
///
/// A session borrow cannot escape the call that supplied it.
///
/// ```compile_fail
/// use tenferro_cpu::{with_cpu_exec_session, CpuBackend, CpuExecSession};
/// use tenferro_runtime::{ExtensionCacheStore, ExtensionExecutionContext};
/// use tenferro_tensor::BackendSessionHost;
///
/// fn leak_context<'a>(
///     backend: &'a mut CpuBackend,
///     caches: &'a mut ExtensionCacheStore,
/// ) -> ExtensionExecutionContext<'a, CpuExecSession<'a>> {
///     backend.with_backend_session(move |session| {
///         with_cpu_exec_session(session, |cpu_session| {
///             ExtensionExecutionContext::new(cpu_session, caches)
///         })
///         .unwrap()
///     })
/// }
/// ```
pub struct ExtensionExecutionContext<'a, B: BackendSession + ?Sized> {
    backend: &'a mut B,
    caches: &'a mut ExtensionCacheStore,
}

impl<B: BackendSession + ?Sized> fmt::Debug for ExtensionExecutionContext<'_, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtensionExecutionContext")
            .field("backend_type", &std::any::type_name::<B>())
            .field("caches", &self.caches)
            .finish_non_exhaustive()
    }
}

impl<'a, B: BackendSession + ?Sized> ExtensionExecutionContext<'a, B> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_cpu::CpuBackend;
    use tenferro_tensor::{BackendSession, BackendSessionHost, Tensor};

    use crate::ExtensionCacheSelector;

    #[test]
    fn context_accepts_non_owning_backend_session() {
        let mut backend = CpuBackend::new();
        let mut caches = ExtensionCacheStore::new();

        backend.with_backend_session(|session| {
            let mut context = ExtensionExecutionContext::new(session, &mut caches);
            let _: &dyn BackendSession = context.backend();
            let lhs = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
            let rhs = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
            let output = context.backend_mut().add(&lhs, &rhs).unwrap();

            assert_eq!(output.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
            assert_eq!(
                context.caches().stats(ExtensionCacheSelector::All).entries,
                0
            );

            let (_, caches) = context.parts_mut();
            assert_eq!(caches.stats(ExtensionCacheSelector::All).entries, 0);
        });
    }
}
