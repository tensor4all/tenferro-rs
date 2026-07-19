use std::fmt;

use tenferro_runtime::ExtensionCacheStore;
use tenferro_tensor::{Tensor, TensorBackend};

use crate::{FftPlanCache, FftPlanSpec};

pub(crate) enum FftCacheSource<'a> {
    CallerOwned(&'a mut FftPlanCache),
    RuntimeOwned(&'a mut ExtensionCacheStore),
}

/// Execution-cache state supplied to an [`FftBackend`].
///
/// Direct repeated calls use a caller-owned [`FftPlanCache`], while traced
/// execution uses the owning runtime's [`ExtensionCacheStore`]. Constructors
/// keep the representation closed so future backends can add plan/workspace
/// entries without exposing RustFFT plan types.
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
    source: FftCacheSource<'a>,
}

impl fmt::Debug for FftExecutionCache<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.source {
            FftCacheSource::CallerOwned(cache) => f
                .debug_struct("FftExecutionCache")
                .field("owner", &"CallerOwned")
                .field("stats", &cache.stats())
                .finish_non_exhaustive(),
            FftCacheSource::RuntimeOwned(_) => f
                .debug_struct("FftExecutionCache")
                .field("owner", &"RuntimeOwned")
                .finish_non_exhaustive(),
        }
    }
}

impl<'a> FftExecutionCache<'a> {
    /// Build a context backed by a caller-owned RustFFT plan cache.
    pub fn caller_owned(cache: &'a mut FftPlanCache) -> Self {
        Self {
            source: FftCacheSource::CallerOwned(cache),
        }
    }

    /// Build a context backed by an extension runtime cache store.
    pub fn runtime_owned(cache: &'a mut ExtensionCacheStore) -> Self {
        Self {
            source: FftCacheSource::RuntimeOwned(cache),
        }
    }

    /// Borrow the runtime cache store when traced execution owns this context.
    ///
    /// Caller-owned direct execution returns `None`. Backend implementations
    /// can use the returned store for backend-specific plans or workspaces.
    pub fn runtime_store_mut(&mut self) -> Option<&mut ExtensionCacheStore> {
        match &mut self.source {
            FftCacheSource::CallerOwned(_) => None,
            FftCacheSource::RuntimeOwned(cache) => Some(cache),
        }
    }

    pub(crate) fn into_source(self) -> FftCacheSource<'a> {
        self.source
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
/// A generic tensor backend without this explicit capability cannot register
/// the FFT runtime:
///
/// ```compile_fail
/// use tenferro_runtime::ExtensionExecutor;
/// use tenferro_tensor::{Tensor, TensorBackend};
/// use tenferro_fft::{FftNorm, TensorFftExt};
///
/// fn register_without_fft_capability<B: TensorBackend + 'static>(
///     executor: &mut ExtensionExecutor<B>,
///     backend: &mut B,
///     input: &Tensor,
/// ) {
///     tenferro_fft::register_runtime(executor).unwrap();
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
