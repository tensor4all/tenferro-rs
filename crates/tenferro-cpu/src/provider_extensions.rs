//! Typed provider extensions of operation-family crates.
//!
//! Kept out of the GEMM/`dot_general` dispatch module: extensions are looked
//! up only by the crates that own them (for example tenferro-linalg's
//! kernels, once per linalg call), never on the contraction hot path.

use std::any::{Any, TypeId};
use std::sync::Arc;

/// Provider objects installed by operation-family crates that tenferro-cpu
/// does not name (for example tenferro-linalg's kernel trait), keyed by
/// their Rust type.
#[derive(Clone, Default)]
pub(crate) struct ProviderExtensions(Vec<(TypeId, Arc<dyn Any + Send + Sync>)>);

impl ProviderExtensions {
    pub(crate) fn insert<E: Any + Send + Sync>(&mut self, extension: Arc<E>) {
        let id = TypeId::of::<E>();
        self.0.retain(|(key, _)| *key != id);
        self.0.push((id, extension));
    }

    pub(crate) fn get<E: Any + Send + Sync>(&self) -> Option<Arc<E>> {
        let id = TypeId::of::<E>();
        self.0
            .iter()
            .find(|(key, _)| *key == id)
            .and_then(|(_, extension)| Arc::clone(extension).downcast::<E>().ok())
    }
}

impl std::fmt::Debug for ProviderExtensions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderExtensions")
            .field("count", &self.0.len())
            .finish()
    }
}
