use std::fmt;

use tenferro_runtime::ExtensionCacheStore;
use tenferro_tensor::{Tensor, TensorBackend};

use crate::{FftPlanCache, FftPlanSpec};

#[derive(Clone, Copy, Debug)]
enum FftCacheOwner {
    CallerOwned,
    RuntimeOwned,
}

/// Execution-cache state supplied to an [`FftBackend`].
///
/// Direct repeated calls use a caller-owned [`FftPlanCache`], while traced
/// execution uses the owning runtime's [`ExtensionCacheStore`]. Both ownership
/// paths expose the same bounded typed store to a backend, so CPU, Metal, CUDA,
/// and future implementations can retain private plans or workspaces in their
/// own cache namespace. Constructors keep the owner representation closed.
///
/// # Examples
///
/// ```
/// use tenferro_fft::{FftExecutionCache, FftPlanCache};
///
/// let mut plans = FftPlanCache::default();
/// let cache = FftExecutionCache::caller_owned(&mut plans);
/// assert!(format!("{cache:?}").contains("CallerOwned"));
/// ```
pub struct FftExecutionCache<'a> {
    owner: FftCacheOwner,
    store: &'a mut ExtensionCacheStore,
}

impl fmt::Debug for FftExecutionCache<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FftExecutionCache")
            .field("owner", &self.owner)
            .field(
                "stats",
                &self
                    .store
                    .stats(tenferro_runtime::ExtensionCacheSelector::All),
            )
            .finish_non_exhaustive()
    }
}

impl<'a> FftExecutionCache<'a> {
    /// Build a context backed by a caller-owned typed FFT execution cache.
    pub fn caller_owned(cache: &'a mut FftPlanCache) -> Self {
        Self {
            owner: FftCacheOwner::CallerOwned,
            store: cache.store_mut(),
        }
    }

    /// Build a context backed by an extension runtime cache store.
    pub fn runtime_owned(cache: &'a mut ExtensionCacheStore) -> Self {
        Self {
            owner: FftCacheOwner::RuntimeOwned,
            store: cache,
        }
    }

    /// Borrow the bounded typed store owned by the caller or extension runtime.
    ///
    /// Backend implementations should use a stable family/cache namespace and
    /// include every plan-identity field in the key discriminator. The store
    /// owns LRU bounds, typed retrieval, clear behavior, entry counts, and the
    /// retained-byte estimates supplied at insertion.
    pub fn store_mut(&mut self) -> &mut ExtensionCacheStore {
        self.store
    }
}

/// Explicit backend capability required by concrete and traced FFT execution.
///
/// Implementations must execute on the input's existing placement. Unsupported
/// dtypes, layouts, placements, or operations return an error; they must never
/// transfer the tensor or select a different backend.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_fft::FftBackend;
///
/// fn accepts_fft_backend<B: FftBackend>(_backend: &mut B) {}
///
/// let mut backend = CpuBackend::new();
/// accepts_fft_backend(&mut backend);
/// ```
///
/// A generic tensor backend without this explicit capability cannot build
/// the FFT extension module:
///
/// ```compile_fail
/// use tenferro_tensor::{Tensor, TensorBackend};
/// use tenferro_fft::{FftNorm, TensorFftExt};
///
/// fn use_without_fft_capability<B: TensorBackend + 'static>(backend: &mut B, input: &Tensor) {
///     let _module =
///         tenferro_fft::extension_module::<B>(tenferro_cpu::runtime_engine_id().unwrap()).unwrap();
///     let _ = input.fft(None, -1, FftNorm::Backward, backend);
/// }
/// ```
pub trait FftBackend: TensorBackend {
    /// Execute one validated FFT request on `input`'s existing placement.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// operation, dtype, layout, or placement;
    /// [`tenferro_tensor::Error::Validation`] for inconsistent input/spec
    /// metadata or checked shape arithmetic; and a typed backend or runtime
    /// source when plan creation, cache access, or execution fails.
    fn execute_fft(
        &mut self,
        input: &Tensor,
        spec: &FftPlanSpec,
        cache: FftExecutionCache<'_>,
    ) -> tenferro_tensor::Result<Tensor>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_cache_debug_identifies_both_owners_and_exposes_the_store() {
        let mut caller = FftPlanCache::default();
        let mut caller_cache = FftExecutionCache::caller_owned(&mut caller);
        assert!(format!("{caller_cache:?}").contains("CallerOwned"));
        assert_eq!(
            caller_cache
                .store_mut()
                .stats(tenferro_runtime::ExtensionCacheSelector::All)
                .entries,
            0
        );

        let mut runtime = ExtensionCacheStore::default();
        let mut runtime_cache = FftExecutionCache::runtime_owned(&mut runtime);
        assert!(format!("{runtime_cache:?}").contains("RuntimeOwned"));
        assert_eq!(
            runtime_cache
                .store_mut()
                .stats(tenferro_runtime::ExtensionCacheSelector::All)
                .entries,
            0
        );
    }
}
