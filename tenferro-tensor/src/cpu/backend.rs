use std::cmp::Reverse;
use std::collections::HashMap;
use std::env;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::backend::{TensorBackend, TensorExec};
use crate::buffer_pool::{BufferPool, BufferPoolStats, PoolScalar};
use crate::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use crate::{Buffer, CacheStats, Tensor, TensorRead, TypedTensor};

use super::exec_session::CpuExecSession;
use super::{analytic, elementwise, gemm, indexing, linalg, reduction, structural, CpuContext};

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
            .get("with_exec_session_cached.total")
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

/// CPU execution backend.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::cpu::CpuBackend;
///
/// let backend = CpuBackend::new();
/// ```
pub struct CpuBackend {
    pub(crate) ctx: Arc<CpuContext>,
    pub(crate) buffers: BufferPool,
}

impl CpuBackend {
    /// Create a CPU backend using the environment-driven CPU context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// ```
    pub fn new() -> Self {
        Self::from_context(Arc::new(CpuContext::from_env()))
    }

    /// Try to create a CPU backend using `RAYON_NUM_THREADS`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cpu::CpuBackend;
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
    /// use tenferro_tensor::cpu::{CpuBackend, CpuContext};
    ///
    /// let ctx = Arc::new(CpuContext::with_threads(2));
    /// let backend = CpuBackend::from_context(ctx);
    /// assert_eq!(backend.num_threads(), 2);
    /// ```
    pub fn from_context(ctx: Arc<CpuContext>) -> Self {
        Self {
            ctx,
            buffers: BufferPool::new(),
        }
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
    /// use tenferro_tensor::cpu::{CpuBackend, CpuContext};
    ///
    /// let ctx = Arc::new(CpuContext::with_threads(1));
    /// let backend = CpuBackend::from_context_with_buffer_pool_limit(ctx, 0);
    /// assert_eq!(backend.buffer_pool_limit_bytes(), 0);
    /// ```
    pub fn from_context_with_buffer_pool_limit(
        ctx: Arc<CpuContext>,
        max_retained_capacity_bytes: usize,
    ) -> Self {
        Self {
            ctx,
            buffers: BufferPool::with_max_retained_capacity_bytes(max_retained_capacity_bytes),
        }
    }

    /// Create a CPU backend with a custom thread count.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cpu::CpuBackend;
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
    /// use tenferro_tensor::cpu::CpuBackend;
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
                crate::Error::BackendFailure { message, .. } => crate::Error::BackendFailure {
                    op: "CpuBackend::try_with_threads",
                    message,
                },
                err => err,
            })
    }

    /// Return the number of threads in this backend's CPU context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cpu::CpuBackend;
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
    /// use tenferro_tensor::cpu::CpuBackend;
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
    /// use tenferro_tensor::cpu::CpuBackend;
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
    /// use tenferro_tensor::cpu::CpuBackend;
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
    /// use tenferro_tensor::cpu::{CpuBackend, CpuContext};
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
    /// use tenferro_tensor::cpu::CpuBackend;
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
    /// use tenferro_tensor::cpu::CpuBackend;
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
    /// use tenferro_tensor::cpu::CpuBackend;
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

    #[cfg(feature = "cpu-blas")]
    fn run_with_pool<R>(&mut self, op: impl FnOnce(&mut BufferPool) -> R) -> R {
        let mut buffers = std::mem::take(&mut self.buffers);
        let result = op(&mut buffers);
        self.buffers = buffers;
        result
    }

    #[cfg(feature = "cpu-faer")]
    fn linalg_with_pool<R>(&mut self, op: impl FnOnce(&mut BufferPool) -> R) -> R {
        self.install_with_pool(op)
    }

    #[cfg(feature = "cpu-blas")]
    fn linalg_with_pool<R>(&mut self, op: impl FnOnce(&mut BufferPool) -> R) -> R {
        self.run_with_pool(op)
    }

    #[cfg(feature = "cpu-faer")]
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

    #[cfg(feature = "cpu-blas")]
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

impl TensorBackend for CpuBackend {
    type RuntimeCache = gemm::GemmAnalysisCache;

    fn add(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::add_with_pool(buffers, lhs, rhs))
    }

    fn mul(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::mul_with_pool(buffers, lhs, rhs))
    }

    fn neg(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::neg_with_pool(buffers, input))
    }

    fn conj(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::conj_with_pool(buffers, input))
    }

    fn div(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::div_with_pool(buffers, lhs, rhs))
    }

    fn abs(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::abs_with_pool(buffers, input))
    }

    fn sign(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::sign_with_pool(buffers, input))
    }

    fn maximum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::maximum_with_pool(buffers, lhs, rhs))
    }

    fn minimum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::minimum_with_pool(buffers, lhs, rhs))
    }

    fn compare(&mut self, lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::compare_with_pool(buffers, lhs, rhs, dir))
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

    fn clamp(&mut self, input: &Tensor, lower: &Tensor, upper: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| elementwise::clamp_with_pool(buffers, input, lower, upper))
    }

    fn exp(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::exp_with_pool(buffers, input))
    }

    fn log(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::log_with_pool(buffers, input))
    }

    fn sin(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::sin_with_pool(buffers, input))
    }

    fn cos(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::cos_with_pool(buffers, input))
    }

    fn tanh(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::tanh_with_pool(buffers, input))
    }

    fn sqrt(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::sqrt_with_pool(buffers, input))
    }

    fn rsqrt(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::rsqrt_with_pool(buffers, input))
    }

    fn pow(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::pow_with_pool(buffers, lhs, rhs))
    }

    fn expm1(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::expm1_with_pool(buffers, input))
    }

    fn log1p(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| analytic::log1p_with_pool(buffers, input))
    }

    fn transpose(&mut self, input: &Tensor, perm: &[usize]) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| structural::transpose_with_pool(buffers, input, perm))
    }

    fn reshape(&mut self, input: &Tensor, shape: &[usize]) -> crate::Result<Tensor> {
        self.install(|| structural::reshape(input, shape))
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

    fn reduce_sum(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.install(|| reduction::reduce_sum(input, axes))
    }

    fn reduce_prod(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.install(|| reduction::reduce_prod(input, axes))
    }

    fn reduce_max(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.install(|| reduction::reduce_max(input, axes))
    }

    fn reduce_min(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.install(|| reduction::reduce_min(input, axes))
    }

    fn dot_general(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        let mut cache = gemm::GemmAnalysisCache::default();
        self.dot_general_cached(&mut cache, None, lhs, rhs, config)
    }

    fn dot_general_cached(
        &mut self,
        cache: &mut Self::RuntimeCache,
        cache_slot: Option<usize>,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        #[cfg(feature = "cpu-blas")]
        {
            return self.run_with_pool_and_gemm_cache(cache, |buffers, cache| match (lhs, rhs) {
                (Tensor::F32(a), Tensor::F32(b)) => {
                    gemm::dot_general_cached(buffers, cache, cache_slot, a, b, config)
                        .map(Tensor::F32)
                }
                (Tensor::F64(a), Tensor::F64(b)) => {
                    gemm::dot_general_cached(buffers, cache, cache_slot, a, b, config)
                        .map(Tensor::F64)
                }
                (Tensor::C32(a), Tensor::C32(b)) => {
                    gemm::dot_general_cached(buffers, cache, cache_slot, a, b, config)
                        .map(Tensor::C32)
                }
                (Tensor::C64(a), Tensor::C64(b)) => {
                    gemm::dot_general_cached(buffers, cache, cache_slot, a, b, config)
                        .map(Tensor::C64)
                }
                _ => Err(crate::Error::DTypeMismatch {
                    op: "dot_general",
                    lhs: lhs.dtype(),
                    rhs: rhs.dtype(),
                }),
            });
        }

        #[cfg(feature = "cpu-faer")]
        {
            let ctx = Arc::clone(&self.ctx);
            self.install_with_pool_and_gemm_cache(cache, |buffers, cache| match (lhs, rhs) {
                (Tensor::F32(a), Tensor::F32(b)) => {
                    gemm::dot_general_cached(buffers, cache, cache_slot, ctx.as_ref(), a, b, config)
                        .map(Tensor::F32)
                }
                (Tensor::F64(a), Tensor::F64(b)) => {
                    gemm::dot_general_cached(buffers, cache, cache_slot, ctx.as_ref(), a, b, config)
                        .map(Tensor::F64)
                }
                (Tensor::C32(a), Tensor::C32(b)) => {
                    gemm::dot_general_cached(buffers, cache, cache_slot, ctx.as_ref(), a, b, config)
                        .map(Tensor::C32)
                }
                (Tensor::C64(a), Tensor::C64(b)) => {
                    gemm::dot_general_cached(buffers, cache, cache_slot, ctx.as_ref(), a, b, config)
                        .map(Tensor::C64)
                }
                _ => Err(crate::Error::DTypeMismatch {
                    op: "dot_general",
                    lhs: lhs.dtype(),
                    rhs: rhs.dtype(),
                }),
            })
        }
    }

    fn dot_general_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        let mut cache = gemm::GemmAnalysisCache::default();
        #[cfg(feature = "cpu-faer")]
        let direct = {
            let ctx = Arc::clone(&self.ctx);
            self.install_with_pool_and_gemm_cache(&mut cache, |buffers, cache| {
                gemm::dot_general_read_cached(buffers, cache, None, ctx.as_ref(), lhs, rhs, config)
            })?
        };
        #[cfg(feature = "cpu-blas")]
        let direct = self.run_with_pool_and_gemm_cache(&mut cache, |buffers, cache| {
            gemm::dot_general_read_cached(buffers, cache, None, lhs, rhs, config)
        })?;
        if let Some(result) = direct {
            return Ok(result);
        }

        let lhs = lhs.to_tensor();
        let rhs = rhs.to_tensor();
        self.dot_general_cached(&mut cache, None, &lhs, &rhs, config)
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
        self.dot_general_with_conj_cached(&mut cache, None, lhs, rhs, config, lhs_conj, rhs_conj)
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
        #[cfg(feature = "cpu-blas")]
        {
            return self.run_with_pool_and_gemm_cache(cache, |buffers, cache| match (lhs, rhs) {
                (Tensor::F32(a), Tensor::F32(b)) => gemm::dot_general_with_conj_cached(
                    buffers, cache, cache_slot, a, b, config, lhs_conj, rhs_conj,
                )
                .map(Tensor::F32),
                (Tensor::F64(a), Tensor::F64(b)) => gemm::dot_general_with_conj_cached(
                    buffers, cache, cache_slot, a, b, config, lhs_conj, rhs_conj,
                )
                .map(Tensor::F64),
                (Tensor::C32(a), Tensor::C32(b)) => gemm::dot_general_with_conj_cached(
                    buffers, cache, cache_slot, a, b, config, lhs_conj, rhs_conj,
                )
                .map(Tensor::C32),
                (Tensor::C64(a), Tensor::C64(b)) => gemm::dot_general_with_conj_cached(
                    buffers, cache, cache_slot, a, b, config, lhs_conj, rhs_conj,
                )
                .map(Tensor::C64),
                _ => Err(crate::Error::DTypeMismatch {
                    op: "dot_general",
                    lhs: lhs.dtype(),
                    rhs: rhs.dtype(),
                }),
            });
        }

        #[cfg(feature = "cpu-faer")]
        {
            let ctx = Arc::clone(&self.ctx);
            self.install_with_pool_and_gemm_cache(cache, |buffers, cache| match (lhs, rhs) {
                (Tensor::F32(a), Tensor::F32(b)) => gemm::dot_general_with_conj_cached(
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
                .map(Tensor::F32),
                (Tensor::F64(a), Tensor::F64(b)) => gemm::dot_general_with_conj_cached(
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
                .map(Tensor::F64),
                (Tensor::C32(a), Tensor::C32(b)) => gemm::dot_general_with_conj_cached(
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
                .map(Tensor::C32),
                (Tensor::C64(a), Tensor::C64(b)) => gemm::dot_general_with_conj_cached(
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
                .map(Tensor::C64),
                _ => Err(crate::Error::DTypeMismatch {
                    op: "dot_general",
                    lhs: lhs.dtype(),
                    rhs: rhs.dtype(),
                }),
            })
        }
    }

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

    fn cholesky(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        #[cfg(feature = "cpu-faer")]
        let ctx = Arc::clone(&self.ctx);
        self.linalg_with_pool(|buffers| match input {
            #[cfg(feature = "cpu-faer")]
            Tensor::F32(t) => linalg::cholesky(ctx.as_ref(), buffers, t).map(Tensor::F32),
            #[cfg(feature = "cpu-blas")]
            Tensor::F32(t) => linalg::cholesky(buffers, t).map(Tensor::F32),
            #[cfg(feature = "cpu-blas")]
            Tensor::C32(t) => linalg::cholesky(buffers, t).map(Tensor::C32),
            #[cfg(feature = "cpu-faer")]
            Tensor::C32(t) => linalg::cholesky(ctx.as_ref(), buffers, t).map(Tensor::C32),
            #[cfg(feature = "cpu-faer")]
            Tensor::F64(t) => linalg::cholesky(ctx.as_ref(), buffers, t).map(Tensor::F64),
            #[cfg(feature = "cpu-blas")]
            Tensor::F64(t) => linalg::cholesky(buffers, t).map(Tensor::F64),
            #[cfg(feature = "cpu-blas")]
            Tensor::C64(t) => linalg::cholesky(buffers, t).map(Tensor::C64),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => linalg::cholesky(ctx.as_ref(), buffers, t).map(Tensor::C64),
            _ => Err(unsupported_dtype("cholesky", input.dtype())),
        })
    }

    fn triangular_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> crate::Result<Tensor> {
        #[cfg(feature = "cpu-faer")]
        let ctx = Arc::clone(&self.ctx);
        self.linalg_with_pool(|buffers| match (a, b) {
            #[cfg(feature = "cpu-faer")]
            (Tensor::F32(a), Tensor::F32(b)) => linalg::triangular_solve(
                ctx.as_ref(),
                buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::F32),
            #[cfg(feature = "cpu-faer")]
            (Tensor::F64(a), Tensor::F64(b)) => linalg::triangular_solve(
                ctx.as_ref(),
                buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::F64),
            #[cfg(feature = "cpu-blas")]
            (Tensor::F32(a), Tensor::F32(b)) => linalg::triangular_solve(
                buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::F32),
            #[cfg(feature = "cpu-blas")]
            (Tensor::F64(a), Tensor::F64(b)) => linalg::triangular_solve(
                buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::F64),
            #[cfg(feature = "cpu-blas")]
            (Tensor::C32(a), Tensor::C32(b)) => linalg::triangular_solve(
                buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::C32),
            #[cfg(feature = "cpu-blas")]
            (Tensor::C64(a), Tensor::C64(b)) => linalg::triangular_solve(
                buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::C64),
            #[cfg(feature = "cpu-faer")]
            (Tensor::C32(a), Tensor::C32(b)) => linalg::triangular_solve(
                ctx.as_ref(),
                buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::C32),
            #[cfg(feature = "cpu-faer")]
            (Tensor::C64(a), Tensor::C64(b)) => linalg::triangular_solve(
                ctx.as_ref(),
                buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::C64),
            _ => {
                if a.dtype() != b.dtype() {
                    Err(crate::Error::DTypeMismatch {
                        op: "triangular_solve",
                        lhs: a.dtype(),
                        rhs: b.dtype(),
                    })
                } else {
                    Err(unsupported_dtype("triangular_solve", a.dtype()))
                }
            }
        })
    }

    fn lu(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
        #[cfg(feature = "cpu-faer")]
        let ctx = Arc::clone(&self.ctx);
        self.linalg_with_pool(|buffers| match input {
            #[cfg(feature = "cpu-faer")]
            Tensor::F32(t) => linalg::lu(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::F64(t) => linalg::lu(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::F32(t) => {
                linalg::lu(buffers, t).map(|outputs| outputs.into_iter().map(Tensor::F32).collect())
            }
            #[cfg(feature = "cpu-blas")]
            Tensor::F64(t) => {
                linalg::lu(buffers, t).map(|outputs| outputs.into_iter().map(Tensor::F64).collect())
            }
            #[cfg(feature = "cpu-blas")]
            Tensor::C32(t) => {
                linalg::lu(buffers, t).map(|outputs| outputs.into_iter().map(Tensor::C32).collect())
            }
            #[cfg(feature = "cpu-blas")]
            Tensor::C64(t) => {
                linalg::lu(buffers, t).map(|outputs| outputs.into_iter().map(Tensor::C64).collect())
            }
            #[cfg(feature = "cpu-faer")]
            Tensor::C32(t) => linalg::lu(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => linalg::lu(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
            _ => Err(unsupported_dtype("lu", input.dtype())),
        })
    }

    fn full_piv_lu(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
        #[cfg(feature = "cpu-faer")]
        let ctx = Arc::clone(&self.ctx);
        self.linalg_with_pool(|buffers| match input {
            #[cfg(feature = "cpu-faer")]
            Tensor::F32(t) => linalg::full_piv_lu(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::F64(t) => linalg::full_piv_lu(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::F32(t) => linalg::full_piv_lu(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::F64(t) => linalg::full_piv_lu(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::C32(t) => linalg::full_piv_lu(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::C64(t) => linalg::full_piv_lu(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::C32(t) => linalg::full_piv_lu(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => linalg::full_piv_lu(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
            _ => Err(unsupported_dtype("full_piv_lu", input.dtype())),
        })
    }

    fn full_piv_lu_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        transpose_a: bool,
    ) -> crate::Result<Tensor> {
        if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
            return Ok(zeros_like_tensor(b));
        }

        let (rhs, restore_shape) = if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
            (
                self.reshape(b, &matrix_rhs_shape)?,
                Some(b.shape().to_vec()),
            )
        } else {
            (b.clone(), None)
        };

        #[cfg(feature = "cpu-faer")]
        let ctx = Arc::clone(&self.ctx);
        let result = self.linalg_with_pool(|buffers| match (a, &rhs) {
            #[cfg(feature = "cpu-faer")]
            (Tensor::F32(a), Tensor::F32(b)) => {
                linalg::full_piv_lu_solve(ctx.as_ref(), buffers, a, b, transpose_a).map(Tensor::F32)
            }
            #[cfg(feature = "cpu-faer")]
            (Tensor::F64(a), Tensor::F64(b)) => {
                linalg::full_piv_lu_solve(ctx.as_ref(), buffers, a, b, transpose_a).map(Tensor::F64)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::F32(a), Tensor::F32(b)) => {
                linalg::full_piv_lu_solve(buffers, a, b, transpose_a).map(Tensor::F32)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::F64(a), Tensor::F64(b)) => {
                linalg::full_piv_lu_solve(buffers, a, b, transpose_a).map(Tensor::F64)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::C32(a), Tensor::C32(b)) => {
                linalg::full_piv_lu_solve(buffers, a, b, transpose_a).map(Tensor::C32)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::C64(a), Tensor::C64(b)) => {
                linalg::full_piv_lu_solve(buffers, a, b, transpose_a).map(Tensor::C64)
            }
            #[cfg(feature = "cpu-faer")]
            (Tensor::C32(a), Tensor::C32(b)) => {
                linalg::full_piv_lu_solve(ctx.as_ref(), buffers, a, b, transpose_a).map(Tensor::C32)
            }
            #[cfg(feature = "cpu-faer")]
            (Tensor::C64(a), Tensor::C64(b)) => {
                linalg::full_piv_lu_solve(ctx.as_ref(), buffers, a, b, transpose_a).map(Tensor::C64)
            }
            _ => {
                if a.dtype() != rhs.dtype() {
                    Err(crate::Error::DTypeMismatch {
                        op: "full_piv_lu_solve",
                        lhs: a.dtype(),
                        rhs: rhs.dtype(),
                    })
                } else {
                    Err(unsupported_dtype("full_piv_lu_solve", a.dtype()))
                }
            }
        })?;

        if let Some(shape) = restore_shape {
            self.reshape(&result, &shape)
        } else {
            Ok(result)
        }
    }

    fn svd(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
        #[cfg(feature = "cpu-faer")]
        let ctx = Arc::clone(&self.ctx);
        self.linalg_with_pool(|buffers| match input {
            #[cfg(feature = "cpu-faer")]
            Tensor::F32(t) => linalg::svd(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::F64(t) => linalg::svd(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::F32(t) => linalg::svd(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::F64(t) => linalg::svd(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::C32(t) => linalg::svd(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::C64(t) => linalg::svd(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::C32(t) => linalg::svd(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => linalg::svd(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
            _ => Err(unsupported_dtype("svd", input.dtype())),
        })
    }

    fn qr(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
        #[cfg(feature = "cpu-faer")]
        let ctx = Arc::clone(&self.ctx);
        self.linalg_with_pool(|buffers| match input {
            #[cfg(feature = "cpu-faer")]
            Tensor::F32(t) => linalg::qr(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::F64(t) => linalg::qr(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::F32(t) => {
                linalg::qr(buffers, t).map(|outputs| outputs.into_iter().map(Tensor::F32).collect())
            }
            #[cfg(feature = "cpu-blas")]
            Tensor::F64(t) => {
                linalg::qr(buffers, t).map(|outputs| outputs.into_iter().map(Tensor::F64).collect())
            }
            #[cfg(feature = "cpu-blas")]
            Tensor::C32(t) => {
                linalg::qr(buffers, t).map(|outputs| outputs.into_iter().map(Tensor::C32).collect())
            }
            #[cfg(feature = "cpu-blas")]
            Tensor::C64(t) => {
                linalg::qr(buffers, t).map(|outputs| outputs.into_iter().map(Tensor::C64).collect())
            }
            #[cfg(feature = "cpu-faer")]
            Tensor::C32(t) => linalg::qr(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => linalg::qr(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
            _ => Err(unsupported_dtype("qr", input.dtype())),
        })
    }

    fn eigh(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
        #[cfg(feature = "cpu-faer")]
        let ctx = Arc::clone(&self.ctx);
        self.linalg_with_pool(|buffers| match input {
            #[cfg(feature = "cpu-faer")]
            Tensor::F32(t) => linalg::eigh(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::F64(t) => linalg::eigh(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::F32(t) => linalg::eigh(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::F64(t) => linalg::eigh(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::C32(t) => linalg::eigh(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::C64(t) => linalg::eigh(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::C32(t) => linalg::eigh(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => linalg::eigh(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
            _ => Err(unsupported_dtype("eigh", input.dtype())),
        })
    }

    fn eig(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
        if !matches!(
            input,
            Tensor::F32(_) | Tensor::F64(_) | Tensor::C32(_) | Tensor::C64(_)
        ) {
            return Err(unsupported_dtype("eig", input.dtype()));
        }
        #[cfg(feature = "cpu-faer")]
        let ctx = Arc::clone(&self.ctx);
        self.linalg_with_pool(|buffers| {
            #[cfg(feature = "cpu-faer")]
            {
                linalg::eig(ctx.as_ref(), buffers, input)
            }
            #[cfg(feature = "cpu-blas")]
            {
                linalg::eig(buffers, input)
            }
        })
    }

    fn solve(&mut self, a: &Tensor, b: &Tensor) -> crate::Result<Tensor> {
        if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
            return Ok(zeros_like_tensor(b));
        }

        let (rhs, restore_shape) = if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
            (
                self.reshape(b, &matrix_rhs_shape)?,
                Some(b.shape().to_vec()),
            )
        } else {
            (b.clone(), None)
        };

        #[cfg(feature = "cpu-faer")]
        let ctx = Arc::clone(&self.ctx);
        let result = self.linalg_with_pool(|buffers| match (a, &rhs) {
            #[cfg(feature = "cpu-faer")]
            (Tensor::F32(a), Tensor::F32(b)) => {
                linalg::solve(ctx.as_ref(), buffers, a, b, false).map(Tensor::F32)
            }
            #[cfg(feature = "cpu-faer")]
            (Tensor::F64(a), Tensor::F64(b)) => {
                linalg::solve(ctx.as_ref(), buffers, a, b, false).map(Tensor::F64)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::F32(a), Tensor::F32(b)) => {
                linalg::solve(buffers, a, b, false).map(Tensor::F32)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::F64(a), Tensor::F64(b)) => {
                linalg::solve(buffers, a, b, false).map(Tensor::F64)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::C32(a), Tensor::C32(b)) => {
                linalg::solve(buffers, a, b, false).map(Tensor::C32)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::C64(a), Tensor::C64(b)) => {
                linalg::solve(buffers, a, b, false).map(Tensor::C64)
            }
            #[cfg(feature = "cpu-faer")]
            (Tensor::C32(a), Tensor::C32(b)) => {
                linalg::solve(ctx.as_ref(), buffers, a, b, false).map(Tensor::C32)
            }
            #[cfg(feature = "cpu-faer")]
            (Tensor::C64(a), Tensor::C64(b)) => {
                linalg::solve(ctx.as_ref(), buffers, a, b, false).map(Tensor::C64)
            }
            _ => {
                if a.dtype() != rhs.dtype() {
                    Err(crate::Error::DTypeMismatch {
                        op: "solve",
                        lhs: a.dtype(),
                        rhs: rhs.dtype(),
                    })
                } else {
                    Err(unsupported_dtype("solve", a.dtype()))
                }
            }
        })?;

        if let Some(shape) = restore_shape {
            self.reshape(&result, &shape)
        } else {
            Ok(result)
        }
    }

    fn with_exec_session<R: Send>(&mut self, f: impl FnOnce(&mut dyn TensorExec) -> R + Send) -> R {
        let mut cache = profile_cpu_session_section("with_exec_session.cache_default", || {
            gemm::GemmAnalysisCache::default()
        });
        self.with_exec_session_cached(&mut cache, f)
    }

    fn with_exec_session_cached<R: Send>(
        &mut self,
        cache: &mut Self::RuntimeCache,
        f: impl FnOnce(&mut dyn TensorExec) -> R + Send,
    ) -> R {
        if !cpu_session_profile_enabled() {
            let mut buffers = std::mem::take(&mut self.buffers);
            let ctx = Arc::clone(&self.ctx);
            let mut session = CpuExecSession {
                ctx: ctx.as_ref(),
                buffers: &mut buffers,
                gemm_analysis_cache: cache,
            };
            let result = f(&mut session);
            self.buffers = buffers;
            return result;
        }

        let total_started = Instant::now();
        let mut buffers =
            profile_cpu_session_section("with_exec_session_cached.take_buffers", || {
                std::mem::take(&mut self.buffers)
            });
        let ctx = Arc::clone(&self.ctx);
        let result = profile_cpu_session_section("with_exec_session_cached.exec_session", || {
            let session_started = Instant::now();
            let mut session = CpuExecSession {
                ctx: ctx.as_ref(),
                buffers: &mut buffers,
                gemm_analysis_cache: cache,
            };
            record_cpu_session_profile(
                "with_exec_session_cached.session_construct",
                session_started.elapsed(),
            );

            let exec_started = Instant::now();
            let result = f(&mut session);
            record_cpu_session_profile(
                "with_exec_session_cached.exec_body",
                exec_started.elapsed(),
            );
            result
        });
        profile_cpu_session_section("with_exec_session_cached.restore_buffers", || {
            self.buffers = buffers;
        });
        record_cpu_session_profile("with_exec_session_cached.total", total_started.elapsed());
        maybe_print_cpu_session_profile();
        result
    }

    fn reclaim_buffer(&mut self, tensor: Tensor) {
        match tensor {
            Tensor::F32(t) => reclaim_typed(&mut self.buffers, t),
            Tensor::F64(t) => reclaim_typed(&mut self.buffers, t),
            Tensor::I64(t) => reclaim_typed(&mut self.buffers, t),
            Tensor::C32(t) => reclaim_typed(&mut self.buffers, t),
            Tensor::C64(t) => reclaim_typed(&mut self.buffers, t),
        }
    }
}

fn has_zero_dim(shape: &[usize]) -> bool {
    shape.contains(&0)
}

fn batched_vector_rhs_shape(a: &Tensor, b: &Tensor) -> Option<Vec<usize>> {
    if b.shape().len() == 1 {
        return Some(vec![b.shape()[0], 1]);
    }

    let is_batched_vector_rhs = a.shape().len() == b.shape().len() + 1
        && !b.shape().is_empty()
        && b.shape()[0] == a.shape()[0]
        && b.shape()[1..] == a.shape()[2..];
    if !is_batched_vector_rhs {
        return None;
    }

    let mut rhs_shape = vec![b.shape()[0], 1];
    rhs_shape.extend_from_slice(&b.shape()[1..]);
    Some(rhs_shape)
}

pub(crate) fn reclaim_typed<T: PoolScalar>(pool: &mut BufferPool, typed: TypedTensor<T>) {
    match typed.buffer {
        Buffer::Host(data) => T::pool_release(pool, data),
        Buffer::Backend(_) => {}
        #[cfg(feature = "cuda")]
        Buffer::Cubecl(_) => panic!("GPU tensor (Buffer::Cubecl) passed to CPU backend. Use cubecl::download_tensor() to transfer to CPU first."),
    }
}

fn zeros_like_tensor(input: &Tensor) -> Tensor {
    match input {
        Tensor::F32(t) => Tensor::F32(TypedTensor::zeros(t.shape.clone())),
        Tensor::F64(t) => Tensor::F64(TypedTensor::zeros(t.shape.clone())),
        Tensor::I64(t) => Tensor::I64(TypedTensor::zeros(t.shape.clone())),
        Tensor::C32(t) => Tensor::C32(TypedTensor::zeros(t.shape.clone())),
        Tensor::C64(t) => Tensor::C64(TypedTensor::zeros(t.shape.clone())),
    }
}

pub(crate) fn unsupported_dtype(op: &'static str, dtype: crate::DType) -> crate::Error {
    crate::Error::BackendFailure {
        op,
        message: format!("unsupported dtype {dtype:?}"),
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_session_profile_helpers_cover_current_profile_mode() {
        let state = cpu_session_profile_state();
        state
            .lock()
            .expect("CPU session profile mutex poisoned")
            .clear();

        let profiling_enabled = cpu_session_profile_enabled();
        let _ = cpu_session_profile_print_every();

        let value = profile_cpu_session_section("test.profile_section", || 7);
        assert_eq!(value, 7);
        record_cpu_session_profile("test.manual_record", Duration::from_nanos(1));

        let entries = state.lock().expect("CPU session profile mutex poisoned");
        if profiling_enabled {
            assert!(entries.contains_key("test.profile_section"));
            assert!(entries.contains_key("test.manual_record"));
        } else {
            assert!(entries.is_empty());
        }
        drop(entries);

        maybe_print_cpu_session_profile();
    }
}
