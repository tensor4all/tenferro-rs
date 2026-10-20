use std::fmt;

use tenferro_runtime::ExtensionCacheStore;
use tenferro_tensor::{Engine, Tensor, TensorRead};

use crate::{FftPlanCache, FftPlanSpec};

#[derive(Clone, Copy, Debug)]
enum FftCacheOwner {
    CallerOwned,
    EngineOwned,
}

/// Execution-cache state supplied to an [`FftBackend`].
///
/// Direct repeated calls use a caller-owned [`FftPlanCache`], while traced
/// and eager execution use the executing engine's own [`ExtensionCacheStore`]
/// (lent by the runtime for the run). The two differ in what they may hold:
///
/// * a caller-owned cache may outlive the engine and be used with several
///   engines, so it holds only engine-independent plans (the CPU's rustfft
///   plans);
/// * an engine-owned store dies with its engine, so a device backend keeps
///   device-bound plans and workspaces (cuFFT plans) only there. Given a
///   caller-owned cache, a device backend keeps its device-bound plans in its
///   own engine instead.
///
/// Constructors keep the owner representation closed.
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

    /// Build a context backed by the executing engine's own extension cache
    /// store, which may hold device-bound plans.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_fft::FftExecutionCache;
    /// use tenferro_runtime::ExtensionCacheStore;
    ///
    /// let mut store = ExtensionCacheStore::default();
    /// let cache = FftExecutionCache::engine_owned(&mut store);
    /// assert!(cache.is_engine_owned());
    /// ```
    pub fn engine_owned(cache: &'a mut ExtensionCacheStore) -> Self {
        Self {
            owner: FftCacheOwner::EngineOwned,
            store: cache,
        }
    }

    /// Whether this store belongs to the executing engine, so device-bound
    /// entries may be retained in it.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_fft::{FftExecutionCache, FftPlanCache};
    ///
    /// let mut plans = FftPlanCache::default();
    /// assert!(!FftExecutionCache::caller_owned(&mut plans).is_engine_owned());
    /// ```
    pub fn is_engine_owned(&self) -> bool {
        matches!(self.owner, FftCacheOwner::EngineOwned)
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
/// Backend surface required by the FFT extension runtime and the built-in
/// concrete FFT dispatch.
///
/// This is a backend SPI: third-party backend implementers implement this
/// trait on their execution sessions to provide FFT capability. The concrete
/// [`TensorFftExt`](crate::TensorFftExt)/[`TensorReadFftExt`](crate::TensorReadFftExt) methods and [`FftExecutor`](crate::FftExecutor) take an
/// erased `&mut dyn Engine` and dispatch **internally** to the
/// built-in CPU/CUDA/WebGPU execution sessions; a session without this
/// capability returns a typed `Error::Unsupported` instead of being accepted
/// as an arbitrary third-party backend (issue #1680 Phase 3).
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuEngine;
/// use tenferro_fft::FftBackend;
///
/// fn accepts_fft_backend<B: FftBackend>() {}
///
/// accepts_fft_backend::<CpuEngine>();
/// ```
///
/// A generic engine does not expose the FFT capability at the SPI level:
///
/// ```compile_fail
/// use tenferro_tensor::Engine;
/// use tenferro_fft::FftBackend;
///
/// fn requires_fft_backend<B: Engine>(engine: &mut B) {
///     let _: &mut dyn FftBackend = engine;
/// }
/// ```
pub trait FftBackend: Engine {
    /// Validate a borrowed runtime input before the extension read path
    /// materializes it into a compact tensor.
    ///
    /// Backends that can materialize their own placement may keep the default
    /// no-op. CPU overrides this to report device inputs as unsupported FFT
    /// placement errors instead of leaking a generic host-materialization error.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] when this backend cannot
    /// consume the borrowed input placement without an explicit transfer.
    fn validate_fft_read_input(
        &self,
        _op: &'static str,
        _input: &TensorRead<'_>,
    ) -> tenferro_tensor::Result<()> {
        Ok(())
    }

    /// Execute a validated FFT from borrowed storage.
    ///
    /// The default explicitly canonicalizes view inputs. Backends supporting
    /// compact borrowed storage override this method to avoid input copies.
    ///
    /// # Examples
    /// The borrowed public surface dispatches through this backend hook:
    /// ```
    /// use num_complex::Complex64;
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_fft::{FftNorm, TensorReadFftExt};
    /// use tenferro_tensor::{Tensor, TensorRead};
    /// let input = Tensor::from_vec_col_major([2], vec![Complex64::new(1., 0.); 2])?;
    /// let mut engine = CpuEngine::serial();
    /// let output =
    ///     TensorRead::Tensor(&input).fft_read(None, 0, FftNorm::Backward, &mut engine)?;
    /// assert_eq!(output.as_slice::<Complex64>()?, &[Complex64::new(2., 0.), Complex64::new(0., 0.)]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    /// Returns [`tenferro_tensor::Error::Unsupported`] for unsupported dtype,
    /// layout, or placement, and [`tenferro_tensor::Error::Validation`] when
    /// input metadata disagrees with the validated spec. Same-placement
    /// canonicalization, plan creation, and execution failures preserve the
    /// selected backend's typed error source; no implicit transfer is performed.
    fn execute_fft_read(
        &mut self,
        input: TensorRead<'_>,
        spec: &FftPlanSpec,
        cache: FftExecutionCache<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        match input {
            TensorRead::Tensor(input) => self.execute_fft(input, spec, cache),
            input => {
                let owned = self.to_contiguous_read(input)?;
                self.execute_fft(&owned, spec, cache)
            }
        }
    }

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
mod tests;
