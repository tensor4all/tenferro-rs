//! Runtime-owned compiled-graph execution boundary.
//!
//! Phase 5 fills this module with the private execution bridge used by
//! `Runtime::run_compiled*`. The legacy `GraphExecutor` compatibility facade
//! must not grow new runtime ownership after this boundary exists.

use std::fmt;
use std::sync::{Arc, Mutex};

use tenferro_tensor::TensorBackend;

use crate::extension_runtime::ExtensionExecutor;

#[allow(
    dead_code,
    reason = "Phase 5 runtime execution task adds erased dispatch methods"
)]
pub(super) trait ErasedTensorBackendExecutor: fmt::Debug + Send + Sync {
    fn backend_type_name(&self) -> &'static str;
}

pub(super) fn erased_tensor_backend_executor<B>(backend: B) -> Arc<dyn ErasedTensorBackendExecutor>
where
    B: TensorBackend + Clone + Send + Sync + 'static,
{
    Arc::new(TensorBackendExecutor::<B>::new(backend))
}

#[allow(
    dead_code,
    reason = "Phase 5 runtime execution task consumes backend execution state"
)]
struct TensorBackendExecutorState<B: TensorBackend + 'static> {
    backend: B,
    backend_cache: B::RuntimeCache,
    extension_executor: ExtensionExecutor<B>,
}

struct TensorBackendExecutor<B: TensorBackend + 'static> {
    state: Mutex<TensorBackendExecutorState<B>>,
}

impl<B> TensorBackendExecutor<B>
where
    B: TensorBackend + Clone + Send + Sync + 'static,
{
    fn new(backend: B) -> Self {
        Self {
            state: Mutex::new(TensorBackendExecutorState {
                backend,
                backend_cache: B::RuntimeCache::default(),
                extension_executor: ExtensionExecutor::new(),
            }),
        }
    }
}

impl<B: TensorBackend + 'static> fmt::Debug for TensorBackendExecutor<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TensorBackendExecutor")
            .field("backend_type", &std::any::type_name::<B>())
            .field("state_poisoned", &self.state.is_poisoned())
            .finish_non_exhaustive()
    }
}

impl<B> ErasedTensorBackendExecutor for TensorBackendExecutor<B>
where
    B: TensorBackend + Clone + Send + Sync + 'static,
{
    fn backend_type_name(&self) -> &'static str {
        std::any::type_name::<B>()
    }
}
