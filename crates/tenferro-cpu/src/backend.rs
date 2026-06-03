use std::cmp::Reverse;
use std::collections::HashMap;
use std::env;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::buffer_pool::{BufferPool, BufferPoolStats, PoolScalar};
use crate::{
    Buffer, CacheStats, Tensor, TensorRank, TensorRead, TypedTensor, TypedTensorView,
    TypedTensorViewMut,
};
use tenferro_tensor::{
    BackendCachedDot, BackendRuntimeCache, BackendSession, BackendSessionHost, TensorAnalytic,
    TensorBackend, TensorBuffer, TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion,
    TensorIndexing, TensorReduction, TensorStructural, TensorViewCanonicalization,
};
use tenferro_tensor::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};

use super::exec_session::CpuExecSession;
use super::{
    analytic, elementwise, gemm, indexing, materialize_tensor_read, reduction, structural,
    CpuContext,
};

#[derive(Debug, Default, Clone)]
struct CpuSessionProfileEntry {
    calls: usize,
    total_time: Duration,
}

fn cpu_session_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env::var("TENFERRO_PROFILE_CPU_SESSION").is_ok())
}

fn cpu_session_profile_print_every() -> Option<usize> {
    static PRINT_EVERY: OnceLock<Option<usize>> = OnceLock::new();
    *PRINT_EVERY.get_or_init(|| {
        env::var("TENFERRO_PROFILE_CPU_SESSION_PRINT_EVERY")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
    })
}

fn cpu_session_profile_state() -> &'static Mutex<HashMap<&'static str, CpuSessionProfileEntry>> {
    static STATE: OnceLock<Mutex<HashMap<&'static str, CpuSessionProfileEntry>>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn record_cpu_session_profile(section: &'static str, elapsed: Duration) {
    if !cpu_session_profile_enabled() {
        return;
    }
    let mut state = cpu_session_profile_state()
        .lock()
        .expect("CPU session profile mutex poisoned");
    let entry = state.entry(section).or_default();
    entry.calls += 1;
    entry.total_time += elapsed;
}

fn profile_cpu_session_section<T>(section: &'static str, f: impl FnOnce() -> T) -> T {
    if !cpu_session_profile_enabled() {
        return f();
    }
    let started = Instant::now();
    let result = f();
    record_cpu_session_profile(section, started.elapsed());
    result
}

fn maybe_print_cpu_session_profile() {
    let Some(print_every) = cpu_session_profile_print_every() else {
        return;
    };
    let should_print = {
        let state = cpu_session_profile_state()
            .lock()
            .expect("CPU session profile mutex poisoned");
        state
            .get("with_backend_session_cached.total")
            .is_some_and(|entry| entry.calls % print_every == 0)
    };
    if !should_print {
        return;
    }
    let mut entries = {
        let mut state = cpu_session_profile_state()
            .lock()
            .expect("CPU session profile mutex poisoned");
        let entries = state
            .iter()
            .map(|(section, entry)| (*section, entry.clone()))
            .collect::<Vec<_>>();
        state.clear();
        entries
    };
    entries.sort_by_key(|(_, entry)| Reverse(entry.total_time));
    eprintln!("=== tenferro CPU session profile ===");
    for (section, entry) in entries {
        eprintln!(
            "{section}: calls={} total={:.6}ms per_call={:.3}us",
            entry.calls,
            entry.total_time.as_secs_f64() * 1.0e3,
            entry.total_time.as_secs_f64() * 1.0e6 / entry.calls as f64,
        );
    }
}

/// CPU provider selected by a [`CpuBackend`] instance.
///
/// CPU provider features are additive at compile time; this runtime selector
/// chooses which compiled provider an individual backend uses for provider-owned
/// kernels such as GEMM.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackendKind;
///
/// let kind = CpuBackendKind::default_compiled();
/// assert!(matches!(kind, CpuBackendKind::Faer | CpuBackendKind::Blas));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CpuBackendKind {
    /// faer-backed CPU kernels.
    Faer,
    /// BLAS/LAPACK-backed CPU kernels.
    Blas,
}

impl CpuBackendKind {
    /// Return the default compiled CPU provider.
    ///
    /// faer is preferred when both faer and BLAS are compiled in because it has
    /// no external provider-link requirement.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackendKind;
    ///
    /// let _kind = CpuBackendKind::default_compiled();
    /// ```
    pub fn default_compiled() -> Self {
        #[cfg(feature = "cpu-faer")]
        {
            Self::Faer
        }
        #[cfg(all(not(feature = "cpu-faer"), feature = "cpu-blas"))]
        {
            Self::Blas
        }
    }

    // Used by feature-specific diagnostics; some feature combinations leave
    // the formatter path inactive.
    #[allow(dead_code)]
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Faer => "faer",
            Self::Blas => "blas",
        }
    }
}

fn ensure_cpu_backend_kind_available(kind: CpuBackendKind, op: &'static str) -> crate::Result<()> {
    let _ = op;
    match kind {
        CpuBackendKind::Faer => {
            #[cfg(feature = "cpu-faer")]
            {
                Ok(())
            }
            #[cfg(not(feature = "cpu-faer"))]
            {
                Err(crate::Error::InvalidConfig {
                    op,
                    message: "CpuBackendKind::Faer requires the cpu-faer feature".to_string(),
                })
            }
        }
        CpuBackendKind::Blas => {
            #[cfg(feature = "cpu-blas")]
            {
                Ok(())
            }
            #[cfg(not(feature = "cpu-blas"))]
            {
                Err(crate::Error::InvalidConfig {
                    op,
                    message: "CpuBackendKind::Blas requires the cpu-blas feature".to_string(),
                })
            }
        }
    }
}

// Used by feature-disabled backend paths; a given feature build may compile no
// direct call site for one provider.
#[allow(dead_code)]
pub(super) fn unavailable_cpu_backend_kind(kind: CpuBackendKind, op: &'static str) -> crate::Error {
    crate::Error::InvalidConfig {
        op,
        message: format!("CPU backend kind {} is not compiled in", kind.name()),
    }
}

/// CPU execution backend.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
///
/// let backend = CpuBackend::new();
/// ```
pub struct CpuBackend {
    pub(crate) ctx: Arc<CpuContext>,
    pub(crate) buffers: BufferPool,
    kind: CpuBackendKind,
}

impl CpuBackend {
    /// Create a CPU backend using the environment-driven CPU context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// ```
    pub fn new() -> Self {
        Self::from_context(Arc::new(CpuContext::from_env()))
    }

    /// Create a CPU backend using the selected compiled provider.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuBackend, CpuBackendKind};
    ///
    /// let backend = CpuBackend::with_kind(CpuBackendKind::default_compiled()).unwrap();
    /// assert_eq!(backend.kind(), CpuBackendKind::default_compiled());
    /// ```
    pub fn with_kind(kind: CpuBackendKind) -> crate::Result<Self> {
        Self::try_from_context_with_kind(Arc::new(CpuContext::from_env()), kind)
    }

    /// Try to create a CPU backend using `RAYON_NUM_THREADS`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::try_new()
    ///     .unwrap_or_else(|_| CpuBackend::with_threads(1));
    /// let _ = backend.num_threads();
    /// ```
    pub fn try_new() -> crate::Result<Self> {
        CpuContext::try_from_env().map(|ctx| Self::from_context(Arc::new(ctx)))
    }

    /// Create a CPU backend from an existing context.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenferro_cpu::{CpuBackend, CpuContext};
    ///
    /// let ctx = Arc::new(CpuContext::with_threads(2));
    /// let backend = CpuBackend::from_context(ctx);
    /// assert_eq!(backend.num_threads(), 2);
    /// ```
    pub fn from_context(ctx: Arc<CpuContext>) -> Self {
        Self {
            ctx,
            buffers: BufferPool::new(),
            kind: CpuBackendKind::default_compiled(),
        }
    }

    fn try_from_context_with_kind(
        ctx: Arc<CpuContext>,
        kind: CpuBackendKind,
    ) -> crate::Result<Self> {
        ensure_cpu_backend_kind_available(kind, "CpuBackend::with_kind")?;
        Ok(Self {
            ctx,
            buffers: BufferPool::new(),
            kind,
        })
    }

    /// Create a CPU backend from an existing context and buffer-pool retention cap.
    ///
    /// The cap is measured in retained vector capacity bytes. A cap of zero
    /// disables buffer retention.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenferro_cpu::{CpuBackend, CpuContext};
    ///
    /// let ctx = Arc::new(CpuContext::with_threads(1));
    /// let backend = CpuBackend::from_context_with_buffer_pool_limit(ctx, 0);
    /// assert_eq!(backend.buffer_pool_limit_bytes(), 0);
    /// ```
    pub fn from_context_with_buffer_pool_limit(
        ctx: Arc<CpuContext>,
        max_retained_capacity_bytes: usize,
    ) -> Self {
        Self::from_context_with_buffer_pool_limit_and_kind(
            ctx,
            max_retained_capacity_bytes,
            CpuBackendKind::default_compiled(),
        )
    }

    fn from_context_with_buffer_pool_limit_and_kind(
        ctx: Arc<CpuContext>,
        max_retained_capacity_bytes: usize,
        kind: CpuBackendKind,
    ) -> Self {
        Self {
            ctx,
            buffers: BufferPool::with_max_retained_capacity_bytes(max_retained_capacity_bytes),
            kind,
        }
    }

    /// Create a CPU backend with a custom thread count.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::with_threads(2);
    /// assert_eq!(backend.num_threads(), 2);
    /// ```
    pub fn with_threads(num_threads: usize) -> Self {
        match Self::try_with_threads(num_threads) {
            Ok(backend) => backend,
            Err(err) => panic!("{err}"),
        }
    }

    /// Try to create a CPU backend with a custom thread count.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::try_with_threads(1).unwrap();
    /// assert_eq!(backend.num_threads(), 1);
    /// ```
    pub fn try_with_threads(num_threads: usize) -> crate::Result<Self> {
        CpuContext::try_with_threads(num_threads)
            .map(|ctx| Self::from_context(Arc::new(ctx)))
            .map_err(|err| match err {
                crate::Error::InvalidConfig { message, .. } => crate::Error::InvalidConfig {
                    op: "CpuBackend::try_with_threads",
                    message,
                },
                crate::Error::BackendFailure { message, .. } => {
                    crate::Error::backend_failure("CpuBackend::try_with_threads", message)
                }
                err => err,
            })
    }

    /// Try to create a CPU backend with a custom thread count and provider.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuBackend, CpuBackendKind};
    ///
    /// let backend = CpuBackend::try_with_threads_and_kind(
    ///     1,
    ///     CpuBackendKind::default_compiled(),
    /// )?;
    /// assert_eq!(backend.num_threads(), 1);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn try_with_threads_and_kind(
        num_threads: usize,
        kind: CpuBackendKind,
    ) -> crate::Result<Self> {
        ensure_cpu_backend_kind_available(kind, "CpuBackend::try_with_threads_and_kind")?;
        CpuContext::try_with_threads(num_threads)
            .map(|ctx| Self {
                ctx: Arc::new(ctx),
                buffers: BufferPool::new(),
                kind,
            })
            .map_err(|err| match err {
                crate::Error::InvalidConfig { message, .. } => crate::Error::InvalidConfig {
                    op: "CpuBackend::try_with_threads_and_kind",
                    message,
                },
                crate::Error::BackendFailure { message, .. } => {
                    crate::Error::backend_failure("CpuBackend::try_with_threads_and_kind", message)
                }
                err => err,
            })
    }

    /// Return the runtime CPU provider selected by this backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuBackend, CpuBackendKind};
    ///
    /// let backend = CpuBackend::new();
    /// assert_eq!(backend.kind(), CpuBackendKind::default_compiled());
    /// ```
    pub fn kind(&self) -> CpuBackendKind {
        self.kind
    }

    /// Return the number of threads in this backend's CPU context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::with_threads(2);
    /// assert_eq!(backend.num_threads(), 2);
    /// ```
    pub fn num_threads(&self) -> usize {
        self.ctx.num_threads()
    }

    /// Number of retained typed host buffers currently held by this backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// assert_eq!(backend.buffer_pool_len(), 0);
    /// ```
    pub fn buffer_pool_len(&self) -> usize {
        self.buffers.len()
    }

    /// Snapshot reusable typed host buffers currently retained by this backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// let stats = backend.buffer_pool_stats();
    /// assert_eq!(stats.buffers, 0);
    /// assert_eq!(stats.capacity_bytes, 0);
    /// ```
    pub fn buffer_pool_stats(&self) -> BufferPoolStats {
        self.buffers.stats()
    }

    /// Return cache-style stats for the CPU buffer pool.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// let stats = backend.buffer_pool_cache_stats();
    /// assert_eq!(stats.entries, 0);
    /// assert_eq!(stats.retained_bytes, 0);
    /// ```
    pub fn buffer_pool_cache_stats(&self) -> CacheStats {
        self.buffers.cache_stats()
    }

    /// Current CPU buffer-pool retention limit in bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenferro_cpu::{CpuBackend, CpuContext};
    ///
    /// let backend = CpuBackend::from_context_with_buffer_pool_limit(
    ///     Arc::new(CpuContext::with_threads(1)),
    ///     4096,
    /// );
    /// assert_eq!(backend.buffer_pool_limit_bytes(), 4096);
    /// ```
    pub fn buffer_pool_limit_bytes(&self) -> usize {
        self.buffers.max_retained_capacity_bytes()
    }

    /// Update the CPU buffer-pool retention limit in bytes.
    ///
    /// Shrinking the limit evicts retained buffers immediately. A limit of zero
    /// disables buffer retention.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let mut backend = CpuBackend::new();
    /// backend.set_buffer_pool_limit_bytes(0);
    /// assert_eq!(backend.buffer_pool_limit_bytes(), 0);
    /// assert_eq!(backend.buffer_pool_len(), 0);
    /// ```
    pub fn set_buffer_pool_limit_bytes(&mut self, max_retained_capacity_bytes: usize) {
        self.buffers
            .set_max_retained_capacity_bytes(max_retained_capacity_bytes);
    }

    /// Reset reusable typed host buffers currently retained by this backend.
    ///
    /// This releases pool-owned vectors to the process allocator. Operating
    /// system RSS may not fall immediately because allocators can retain freed
    /// pages for future allocations.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let mut backend = CpuBackend::new();
    /// backend.reset_buffer_pool();
    /// assert_eq!(backend.buffer_pool_len(), 0);
    /// ```
    pub fn reset_buffer_pool(&mut self) {
        self.buffers.clear();
    }

    /// Run a closure in this backend's CPU execution scope.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::with_threads(1);
    /// let value = backend.install(|| 1 + 1);
    /// assert_eq!(value, 2);
    /// ```
    pub fn install<R>(&self, op: impl FnOnce() -> R) -> R {
        self.ctx.install(op)
    }

    fn install_with_pool<R>(&mut self, op: impl FnOnce(&mut BufferPool) -> R) -> R {
        let mut buffers = std::mem::take(&mut self.buffers);
        let result = op(&mut buffers);
        self.buffers = buffers;
        result
    }

    // Selected when the BLAS provider is active; default Faer-only builds keep
    // it dormant.
    #[allow(dead_code)]
    fn run_with_pool<R>(&mut self, op: impl FnOnce(&mut BufferPool) -> R) -> R {
        let mut buffers = std::mem::take(&mut self.buffers);
        let result = op(&mut buffers);
        self.buffers = buffers;
        result
    }

    fn linalg_with_pool<R>(&mut self, op: impl FnOnce(&mut BufferPool) -> R) -> R {
        match self.kind {
            CpuBackendKind::Faer => self.install_with_pool(op),
            CpuBackendKind::Blas => self.run_with_pool(op),
        }
    }

    /// Run an external linalg implementation with this backend's buffer pool.
    ///
    /// This is exposed for operation-family crates that own their backend
    /// implementation while still sharing the CPU backend's allocation pool.
    #[doc(hidden)]
    pub fn with_linalg_pool<R>(&mut self, op: impl FnOnce(&mut BufferPool) -> R) -> R {
        self.linalg_with_pool(op)
    }

    /// Clone the CPU context used by external linalg implementations.
    #[cfg(feature = "cpu-faer")]
    #[doc(hidden)]
    pub fn linalg_context(&self) -> Arc<CpuContext> {
        Arc::clone(&self.ctx)
    }

    // Selected when the Faer provider handles cached GEMM execution; some
    // feature combinations compile only the uncached or BLAS path.
    #[allow(dead_code)]
    fn install_with_pool_and_gemm_cache<R>(
        &mut self,
        gemm_analysis_cache: &mut gemm::GemmAnalysisCache,
        op: impl FnOnce(&mut BufferPool, &mut gemm::GemmAnalysisCache) -> R,
    ) -> R {
        let mut buffers = std::mem::take(&mut self.buffers);
        let result = op(&mut buffers, gemm_analysis_cache);
        self.buffers = buffers;
        result
    }

    // Selected when the BLAS provider handles cached GEMM execution; default
    // Faer-only builds keep it dormant.
    #[allow(dead_code)]
    fn run_with_pool_and_gemm_cache<R>(
        &mut self,
        gemm_analysis_cache: &mut gemm::GemmAnalysisCache,
        op: impl FnOnce(&mut BufferPool, &mut gemm::GemmAnalysisCache) -> R,
    ) -> R {
        let mut buffers = std::mem::take(&mut self.buffers);
        let result = op(&mut buffers, gemm_analysis_cache);
        self.buffers = buffers;
        result
    }
}

impl BackendRuntimeCache for CpuBackend {
    type RuntimeCache = gemm::GemmAnalysisCache;
}

impl TensorElementwise for CpuBackend {
    fn add(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::add_with_pool(buffers, lhs, rhs))
    }

    fn add_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::add_read_with_pool(buffers, lhs, rhs))
    }

    fn mul(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::mul_with_pool(buffers, lhs, rhs))
    }

    fn mul_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::mul_read_with_pool(buffers, lhs, rhs))
    }

    fn neg(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::neg_with_pool(buffers, input))
    }

    fn neg_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::neg_read_with_pool(buffers, input))
    }

    fn conj(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::conj_with_pool(buffers, input))
    }

    fn conj_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::conj_read_with_pool(buffers, input))
    }

    fn div(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::div_with_pool(buffers, lhs, rhs))
    }

    fn div_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::div_read_with_pool(buffers, lhs, rhs))
    }

    fn abs(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::abs_with_pool(buffers, input))
    }

    fn abs_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::abs_read_with_pool(buffers, input))
    }

    fn sign(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::sign_with_pool(buffers, input))
    }

    fn sign_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::sign_read_with_pool(buffers, input))
    }

    fn maximum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::maximum_with_pool(buffers, lhs, rhs))
    }

    fn maximum_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::maximum_read_with_pool(buffers, lhs, rhs))
    }

    fn minimum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::minimum_with_pool(buffers, lhs, rhs))
    }

    fn minimum_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::minimum_read_with_pool(buffers, lhs, rhs))
    }

    fn compare(&mut self, lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::compare_with_pool(buffers, lhs, rhs, dir))
    }

    fn compare_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        dir: &CompareDir,
    ) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| {
            elementwise::compare_read_with_pool(buffers, lhs, rhs, dir)
        })
    }

    fn select(
        &mut self,
        pred: &Tensor,
        on_true: &Tensor,
        on_false: &Tensor,
    ) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| {
            elementwise::select_with_pool(buffers, pred, on_true, on_false)
        })
    }

    fn select_read(
        &mut self,
        pred: TensorRead<'_>,
        on_true: TensorRead<'_>,
        on_false: TensorRead<'_>,
    ) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| {
            elementwise::select_read_with_pool(buffers, pred, on_true, on_false)
        })
    }

    fn clamp(&mut self, input: &Tensor, lower: &Tensor, upper: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::clamp_with_pool(buffers, input, lower, upper))
    }

    fn clamp_read(
        &mut self,
        input: TensorRead<'_>,
        lower: TensorRead<'_>,
        upper: TensorRead<'_>,
    ) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| {
            elementwise::clamp_read_with_pool(buffers, input, lower, upper)
        })
    }
}

impl TensorAnalytic for CpuBackend {
    fn exp(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::exp_with_pool(buffers, input))
    }

    fn exp_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::exp_read_with_pool(buffers, input))
    }

    fn log(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::log_with_pool(buffers, input))
    }

    fn log_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::log_read_with_pool(buffers, input))
    }

    fn sin(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::sin_with_pool(buffers, input))
    }

    fn sin_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::sin_read_with_pool(buffers, input))
    }

    fn cos(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::cos_with_pool(buffers, input))
    }

    fn cos_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::cos_read_with_pool(buffers, input))
    }

    fn tanh(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::tanh_with_pool(buffers, input))
    }

    fn tanh_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::tanh_read_with_pool(buffers, input))
    }

    fn sqrt(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::sqrt_with_pool(buffers, input))
    }

    fn sqrt_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::sqrt_read_with_pool(buffers, input))
    }

    fn rsqrt(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::rsqrt_with_pool(buffers, input))
    }

    fn rsqrt_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::rsqrt_read_with_pool(buffers, input))
    }

    fn pow(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::pow_with_pool(buffers, lhs, rhs))
    }

    fn pow_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::pow_read_with_pool(buffers, lhs, rhs))
    }

    fn expm1(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::expm1_with_pool(buffers, input))
    }

    fn expm1_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::expm1_read_with_pool(buffers, input))
    }

    fn log1p(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::log1p_with_pool(buffers, input))
    }

    fn log1p_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::log1p_read_with_pool(buffers, input))
    }
}

impl TensorStructural for CpuBackend {
    fn transpose(&mut self, input: &Tensor, perm: &[usize]) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| structural::transpose_with_pool(buffers, input, perm))
    }

    fn transpose_read(&mut self, input: TensorRead<'_>, perm: &[usize]) -> crate::Result<Tensor> {
        if let Some(input) = input.as_tensor() {
            return self.transpose(input, perm);
        }

        let input = materialize_tensor_read("transpose", input)?;
        self.transpose(&input, perm)
    }

    fn reshape(&mut self, input: &Tensor, shape: &[usize]) -> crate::Result<Tensor> {
        self.install(|| structural::reshape(input, shape))
    }

    fn reshape_read(&mut self, input: TensorRead<'_>, shape: &[usize]) -> crate::Result<Tensor> {
        if let Some(input) = input.as_tensor() {
            return self.reshape(input, shape);
        }

        let input = materialize_tensor_read("reshape", input)?;
        self.reshape(&input, shape)
    }

    fn broadcast_in_dim(
        &mut self,
        input: &Tensor,
        shape: &[usize],
        dims: &[usize],
    ) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| {
            structural::broadcast_in_dim_with_pool(buffers, input, shape, dims)
        })
    }

    fn broadcast_in_dim_read(
        &mut self,
        input: TensorRead<'_>,
        shape: &[usize],
        dims: &[usize],
    ) -> crate::Result<Tensor> {
        if let Some(input) = input.as_tensor() {
            return self.broadcast_in_dim(input, shape, dims);
        }

        let input = materialize_tensor_read("broadcast_in_dim", input)?;
        self.broadcast_in_dim(&input, shape, dims)
    }

    fn convert(&mut self, input: &Tensor, to: crate::DType) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| structural::convert_with_pool(buffers, input, to))
    }

    fn extract_diagonal(
        &mut self,
        input: &Tensor,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| {
            structural::extract_diagonal_with_pool(buffers, input, axis_a, axis_b)
        })
    }

    fn embed_diagonal(
        &mut self,
        input: &Tensor,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| {
            structural::embed_diagonal_with_pool(buffers, input, axis_a, axis_b)
        })
    }

    fn tril(&mut self, input: &Tensor, k: i64) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| structural::tril_with_pool(buffers, input, k))
    }

    fn triu(&mut self, input: &Tensor, k: i64) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| structural::triu_with_pool(buffers, input, k))
    }
}

impl TensorReduction for CpuBackend {
    fn reduce_sum(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.install(|| reduction::reduce_sum(input, axes))
    }

    fn reduce_sum_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.install(|| reduction::reduce_sum_read(input, axes))
    }

    fn reduce_prod(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.install(|| reduction::reduce_prod(input, axes))
    }

    fn reduce_prod_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.install(|| reduction::reduce_prod_read(input, axes))
    }

    fn reduce_max(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.install(|| reduction::reduce_max(input, axes))
    }

    fn reduce_max_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.install(|| reduction::reduce_max_read(input, axes))
    }

    fn reduce_min(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.install(|| reduction::reduce_min(input, axes))
    }

    fn reduce_min_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.install(|| reduction::reduce_min_read(input, axes))
    }
}

impl TensorDot for CpuBackend {
    fn dot_general(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        let mut cache = gemm::GemmAnalysisCache::default();
        BackendCachedDot::dot_general_cached(self, &mut cache, None, lhs, rhs, config)
    }

    fn dot_general_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        let mut cache = gemm::GemmAnalysisCache::default();
        let direct = match self.kind {
            CpuBackendKind::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    let ctx = Arc::clone(&self.ctx);
                    self.install_with_pool_and_gemm_cache(&mut cache, |buffers, cache| {
                        gemm::dot_general_faer_read_cached(
                            buffers,
                            cache,
                            None,
                            ctx.as_ref(),
                            lhs.clone(),
                            rhs.clone(),
                            config,
                        )
                    })?
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    return Err(unavailable_cpu_backend_kind(self.kind, "dot_general"));
                }
            }
            CpuBackendKind::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    self.run_with_pool_and_gemm_cache(&mut cache, |buffers, cache| {
                        gemm::dot_general_blas_read_cached(
                            buffers,
                            cache,
                            None,
                            lhs.clone(),
                            rhs.clone(),
                            config,
                        )
                    })?
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    return Err(unavailable_cpu_backend_kind(self.kind, "dot_general"));
                }
            }
        };
        if let Some(result) = direct {
            return Ok(result);
        }

        let lhs = materialize_tensor_read("dot_general", lhs)?;
        let rhs = materialize_tensor_read("dot_general", rhs)?;
        BackendCachedDot::dot_general_cached(self, &mut cache, None, &lhs, &rhs, config)
    }

    fn dot_general_with_conj(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        let mut cache = gemm::GemmAnalysisCache::default();
        BackendCachedDot::dot_general_with_conj_cached(
            self, &mut cache, None, lhs, rhs, config, lhs_conj, rhs_conj,
        )
    }
}

impl BackendCachedDot for CpuBackend {
    fn dot_general_cached(
        &mut self,
        cache: &mut Self::RuntimeCache,
        cache_slot: Option<usize>,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        match self.kind {
            CpuBackendKind::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    let ctx = Arc::clone(&self.ctx);
                    self.install_with_pool_and_gemm_cache(cache, |buffers, cache| {
                        match (lhs, rhs) {
                            (Tensor::F32(a), Tensor::F32(b)) => gemm::dot_general_faer_cached(
                                buffers,
                                cache,
                                cache_slot,
                                ctx.as_ref(),
                                a,
                                b,
                                config,
                            )
                            .map(Tensor::F32),
                            (Tensor::F64(a), Tensor::F64(b)) => gemm::dot_general_faer_cached(
                                buffers,
                                cache,
                                cache_slot,
                                ctx.as_ref(),
                                a,
                                b,
                                config,
                            )
                            .map(Tensor::F64),
                            (Tensor::C32(a), Tensor::C32(b)) => gemm::dot_general_faer_cached(
                                buffers,
                                cache,
                                cache_slot,
                                ctx.as_ref(),
                                a,
                                b,
                                config,
                            )
                            .map(Tensor::C32),
                            (Tensor::C64(a), Tensor::C64(b)) => gemm::dot_general_faer_cached(
                                buffers,
                                cache,
                                cache_slot,
                                ctx.as_ref(),
                                a,
                                b,
                                config,
                            )
                            .map(Tensor::C64),
                            _ => Err(crate::Error::DTypeMismatch {
                                op: "dot_general",
                                lhs: lhs.dtype(),
                                rhs: rhs.dtype(),
                            }),
                        }
                    })
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    Err(unavailable_cpu_backend_kind(self.kind, "dot_general"))
                }
            }
            CpuBackendKind::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    self.run_with_pool_and_gemm_cache(cache, |buffers, cache| match (lhs, rhs) {
                        (Tensor::F32(a), Tensor::F32(b)) => {
                            gemm::dot_general_blas_cached(buffers, cache, cache_slot, a, b, config)
                                .map(Tensor::F32)
                        }
                        (Tensor::F64(a), Tensor::F64(b)) => {
                            gemm::dot_general_blas_cached(buffers, cache, cache_slot, a, b, config)
                                .map(Tensor::F64)
                        }
                        (Tensor::C32(a), Tensor::C32(b)) => {
                            gemm::dot_general_blas_cached(buffers, cache, cache_slot, a, b, config)
                                .map(Tensor::C32)
                        }
                        (Tensor::C64(a), Tensor::C64(b)) => {
                            gemm::dot_general_blas_cached(buffers, cache, cache_slot, a, b, config)
                                .map(Tensor::C64)
                        }
                        _ => Err(crate::Error::DTypeMismatch {
                            op: "dot_general",
                            lhs: lhs.dtype(),
                            rhs: rhs.dtype(),
                        }),
                    })
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    Err(unavailable_cpu_backend_kind(self.kind, "dot_general"))
                }
            }
        }
    }

    fn dot_general_with_conj_cached(
        &mut self,
        cache: &mut Self::RuntimeCache,
        cache_slot: Option<usize>,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        match self.kind {
            CpuBackendKind::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    let ctx = Arc::clone(&self.ctx);
                    self.install_with_pool_and_gemm_cache(cache, |buffers, cache| {
                        match (lhs, rhs) {
                            (Tensor::F32(a), Tensor::F32(b)) => {
                                gemm::dot_general_faer_with_conj_cached(
                                    buffers,
                                    cache,
                                    cache_slot,
                                    ctx.as_ref(),
                                    a,
                                    b,
                                    config,
                                    lhs_conj,
                                    rhs_conj,
                                )
                                .map(Tensor::F32)
                            }
                            (Tensor::F64(a), Tensor::F64(b)) => {
                                gemm::dot_general_faer_with_conj_cached(
                                    buffers,
                                    cache,
                                    cache_slot,
                                    ctx.as_ref(),
                                    a,
                                    b,
                                    config,
                                    lhs_conj,
                                    rhs_conj,
                                )
                                .map(Tensor::F64)
                            }
                            (Tensor::C32(a), Tensor::C32(b)) => {
                                gemm::dot_general_faer_with_conj_cached(
                                    buffers,
                                    cache,
                                    cache_slot,
                                    ctx.as_ref(),
                                    a,
                                    b,
                                    config,
                                    lhs_conj,
                                    rhs_conj,
                                )
                                .map(Tensor::C32)
                            }
                            (Tensor::C64(a), Tensor::C64(b)) => {
                                gemm::dot_general_faer_with_conj_cached(
                                    buffers,
                                    cache,
                                    cache_slot,
                                    ctx.as_ref(),
                                    a,
                                    b,
                                    config,
                                    lhs_conj,
                                    rhs_conj,
                                )
                                .map(Tensor::C64)
                            }
                            _ => Err(crate::Error::DTypeMismatch {
                                op: "dot_general",
                                lhs: lhs.dtype(),
                                rhs: rhs.dtype(),
                            }),
                        }
                    })
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    Err(unavailable_cpu_backend_kind(self.kind, "dot_general"))
                }
            }
            CpuBackendKind::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    self.run_with_pool_and_gemm_cache(cache, |buffers, cache| match (lhs, rhs) {
                        (Tensor::F32(a), Tensor::F32(b)) => {
                            gemm::dot_general_blas_with_conj_cached(
                                buffers, cache, cache_slot, a, b, config, lhs_conj, rhs_conj,
                            )
                            .map(Tensor::F32)
                        }
                        (Tensor::F64(a), Tensor::F64(b)) => {
                            gemm::dot_general_blas_with_conj_cached(
                                buffers, cache, cache_slot, a, b, config, lhs_conj, rhs_conj,
                            )
                            .map(Tensor::F64)
                        }
                        (Tensor::C32(a), Tensor::C32(b)) => {
                            gemm::dot_general_blas_with_conj_cached(
                                buffers, cache, cache_slot, a, b, config, lhs_conj, rhs_conj,
                            )
                            .map(Tensor::C32)
                        }
                        (Tensor::C64(a), Tensor::C64(b)) => {
                            gemm::dot_general_blas_with_conj_cached(
                                buffers, cache, cache_slot, a, b, config, lhs_conj, rhs_conj,
                            )
                            .map(Tensor::C64)
                        }
                        _ => Err(crate::Error::DTypeMismatch {
                            op: "dot_general",
                            lhs: lhs.dtype(),
                            rhs: rhs.dtype(),
                        }),
                    })
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    Err(unavailable_cpu_backend_kind(self.kind, "dot_general"))
                }
            }
        }
    }
}

impl TensorIndexing for CpuBackend {
    fn gather(
        &mut self,
        operand: &Tensor,
        start_indices: &Tensor,
        config: &GatherConfig,
    ) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| {
            indexing::gather_with_pool(buffers, operand, start_indices, config)
        })
    }

    fn scatter(
        &mut self,
        operand: &Tensor,
        scatter_indices: &Tensor,
        updates: &Tensor,
        config: &ScatterConfig,
    ) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| {
            indexing::scatter_with_pool(buffers, operand, scatter_indices, updates, config)
        })
    }

    fn slice(&mut self, input: &Tensor, config: &SliceConfig) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| indexing::try_slice_with_pool(buffers, input, config))
    }

    fn dynamic_slice(
        &mut self,
        input: &Tensor,
        starts: &Tensor,
        slice_sizes: &[usize],
    ) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| {
            indexing::dynamic_slice_with_pool(buffers, input, starts, slice_sizes)
        })
    }

    fn dynamic_update_slice(
        &mut self,
        operand: &Tensor,
        update: &Tensor,
        starts: &Tensor,
    ) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| {
            indexing::dynamic_update_slice_with_pool(buffers, operand, update, starts)
        })
    }

    fn pad(&mut self, input: &Tensor, config: &PadConfig) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| indexing::try_pad_with_pool(buffers, input, config))
    }

    fn concatenate(&mut self, inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| indexing::try_concatenate_with_pool(buffers, inputs, axis))
    }

    fn reverse(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| indexing::reverse_with_pool(buffers, input, axes))
    }
}

impl BackendSessionHost for CpuBackend {
    fn with_backend_session<R>(&mut self, f: impl FnOnce(&mut dyn BackendSession) -> R) -> R {
        let mut cache = profile_cpu_session_section("with_backend_session.cache_default", || {
            gemm::GemmAnalysisCache::default()
        });
        self.with_backend_session_cached(&mut cache, f)
    }

    fn with_backend_session_cached<R>(
        &mut self,
        cache: &mut Self::RuntimeCache,
        f: impl FnOnce(&mut dyn BackendSession) -> R,
    ) -> R {
        if !cpu_session_profile_enabled() {
            let mut buffers = std::mem::take(&mut self.buffers);
            let ctx = Arc::clone(&self.ctx);
            let mut session = CpuExecSession {
                ctx: ctx.as_ref(),
                buffers: &mut buffers,
                gemm_analysis_cache: cache,
                kind: self.kind,
            };
            let result = f(&mut session);
            self.buffers = buffers;
            return result;
        }

        let total_started = Instant::now();
        let mut buffers =
            profile_cpu_session_section("with_backend_session_cached.take_buffers", || {
                std::mem::take(&mut self.buffers)
            });
        let ctx = Arc::clone(&self.ctx);
        let result =
            profile_cpu_session_section("with_backend_session_cached.exec_session", || {
                let session_started = Instant::now();
                let mut session = CpuExecSession {
                    ctx: ctx.as_ref(),
                    buffers: &mut buffers,
                    gemm_analysis_cache: cache,
                    kind: self.kind,
                };
                record_cpu_session_profile(
                    "with_backend_session_cached.session_construct",
                    session_started.elapsed(),
                );

                let exec_started = Instant::now();
                let result = f(&mut session);
                record_cpu_session_profile(
                    "with_backend_session_cached.exec_body",
                    exec_started.elapsed(),
                );
                result
            });
        profile_cpu_session_section("with_backend_session_cached.restore_buffers", || {
            self.buffers = buffers;
        });
        record_cpu_session_profile("with_backend_session_cached.total", total_started.elapsed());
        maybe_print_cpu_session_profile();
        result
    }
}

impl TensorBuffer for CpuBackend {
    fn reclaim_buffer(&mut self, tensor: Tensor) {
        match tensor {
            Tensor::F32(t) => reclaim_typed(&mut self.buffers, t),
            Tensor::F64(t) => reclaim_typed(&mut self.buffers, t),
            Tensor::I32(t) => reclaim_typed(&mut self.buffers, t),
            Tensor::I64(t) => reclaim_typed(&mut self.buffers, t),
            Tensor::Bool(t) => reclaim_typed(&mut self.buffers, t),
            Tensor::C32(t) => reclaim_typed(&mut self.buffers, t),
            Tensor::C64(t) => reclaim_typed(&mut self.buffers, t),
        }
    }
}

impl<T, R> TensorViewCanonicalization<T, R> for CpuBackend
where
    T: Clone + 'static,
    R: TensorRank,
{
    fn to_contiguous(
        &mut self,
        view: &TypedTensorView<'_, T, R>,
    ) -> crate::Result<TypedTensor<T, R>> {
        if view.backend_buffer().is_some() {
            return Err(crate::Error::backend_failure(
                "CpuBackend::to_contiguous",
                "CPU backend received a backend tensor view; download the tensor to host before CPU view canonicalization",
            ));
        }
        view.to_contiguous()
    }

    fn copy_from_contiguous(
        &mut self,
        src: &TypedTensor<T, R>,
        dst: &mut TypedTensorViewMut<'_, T, R>,
    ) -> crate::Result<()> {
        if matches!(&src.buffer, Buffer::Backend(_)) {
            return Err(crate::Error::backend_failure(
                "CpuBackend::copy_from_contiguous",
                "CPU backend received a backend source tensor; download the tensor to host before CPU view copy-back",
            ));
        }
        if dst.backend_buffer().is_some() {
            return Err(crate::Error::backend_failure(
                "CpuBackend::copy_from_contiguous",
                "CPU backend received a backend destination view; download the tensor to host before CPU view copy-back",
            ));
        }
        dst.copy_from_contiguous(src)
    }
}

impl TensorFusion for CpuBackend {}

impl TensorDeviceTransfer for CpuBackend {}

impl TensorBackend for CpuBackend {}

pub(crate) fn reclaim_typed<T: PoolScalar>(pool: &mut BufferPool, typed: TypedTensor<T>) {
    match typed.buffer {
        Buffer::Host(data) => T::pool_release(pool, data),
        Buffer::Backend(_) => {}
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
