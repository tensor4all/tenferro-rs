use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use tenferro_algebra::{Conjugate, Scalar};
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::{infra::plan_cache::PlanCache, TensorTempPoolContext};

#[cfg(feature = "gemm-blas")]
use super::scratch::{ScratchBuf, ScratchPool};
use super::temp_pool::TempPool;
use crate::cpu::common;

/// CPU execution context.
///
/// Encapsulates CPU-side execution resources, analogous to cuTENSOR's
/// `cutensorHandle_t`. Holds a rayon thread pool, a [`PlanCache`] for plan
/// reuse, and reusable temporary buffers for host-side execution helpers.
///
/// # Examples
///
/// ```
/// use tenferro_prims::CpuContext;
///
/// # fn demo() -> tenferro_device::Result<()> {
/// let mut ctx = CpuContext::try_new(4)?; // 4-thread pool
/// assert_eq!(ctx.num_threads(), 4);
/// # Ok(())
/// # }
/// ```
pub struct CpuContext {
    pub(super) pool: Arc<rayon::ThreadPool>,
    pub(super) plan_cache: PlanCache,
    #[allow(dead_code)]
    temp_pool: TempPool,
    #[cfg(feature = "gemm-blas")]
    scratch: ScratchPool,
}

fn shared_thread_pools() -> &'static Mutex<HashMap<usize, Arc<rayon::ThreadPool>>> {
    static SHARED_THREAD_POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> =
        OnceLock::new();
    SHARED_THREAD_POOLS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn shared_thread_pool(num_threads: usize) -> Result<Arc<rayon::ThreadPool>> {
    let mut pools = shared_thread_pools()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(pool) = pools.get(&num_threads) {
        return Ok(Arc::clone(pool));
    }

    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| Error::DeviceError(format!("failed to build rayon thread pool: {e}")))?,
    );
    pools.insert(num_threads, Arc::clone(&pool));
    Ok(pool)
}

#[cfg(target_os = "linux")]
fn affinity_thread_count() -> Option<usize> {
    let mut set = std::mem::MaybeUninit::<libc::cpu_set_t>::zeroed();
    let rc = unsafe {
        libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), set.as_mut_ptr())
    };
    if rc != 0 {
        return None;
    }
    let set = unsafe { set.assume_init() };
    let count = unsafe { libc::CPU_COUNT(&set) as usize };
    (count > 0).then_some(count)
}

#[cfg(not(target_os = "linux"))]
fn affinity_thread_count() -> Option<usize> {
    None
}

impl CpuContext {
    /// Return the backend-defined default CPU thread count.
    ///
    /// On Linux this prefers the current process CPU affinity mask. Other
    /// platforms fall back to [`std::thread::available_parallelism`].
    pub fn default_num_threads() -> usize {
        affinity_thread_count()
            .or_else(|| std::thread::available_parallelism().ok().map(usize::from))
            .unwrap_or(1)
            .max(1)
    }

    /// Create a new CPU context with the given number of threads.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_device::Error::InvalidArgument`] when
    /// `num_threads == 0`, or [`tenferro_device::Error::DeviceError`] when the
    /// underlying Rayon thread-pool construction fails.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn demo() -> tenferro_device::Result<()> {
    /// use tenferro_prims::CpuContext;
    ///
    /// let ctx = CpuContext::try_new(2)?;
    /// assert_eq!(ctx.num_threads(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_new(num_threads: usize) -> Result<Self> {
        if num_threads == 0 {
            return Err(Error::InvalidArgument(
                "CpuContext::try_new requires num_threads >= 1".into(),
            ));
        }
        Ok(Self {
            pool: shared_thread_pool(num_threads)?,
            plan_cache: PlanCache::new(),
            temp_pool: TempPool::default(),
            #[cfg(feature = "gemm-blas")]
            scratch: ScratchPool::default(),
        })
    }

    /// Create a new CPU context using the backend-defined default thread count.
    pub fn try_new_default() -> Result<Self> {
        Self::try_new(Self::default_num_threads())
    }

    /// Create a new CPU context with the given number of threads.
    ///
    /// This is a convenience wrapper around [`CpuContext::try_new`]. Production
    /// code should generally prefer the fallible constructor so context setup
    /// errors stay in the normal `Result` flow.
    ///
    /// # Panics
    ///
    /// Panics if [`CpuContext::try_new`] returns an error.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_prims::CpuContext;
    ///
    /// let ctx = CpuContext::new(1);
    /// assert_eq!(ctx.num_threads(), 1);
    /// ```
    pub fn new(num_threads: usize) -> Self {
        Self::try_new(num_threads)
            .unwrap_or_else(|e| panic!("failed to initialize CpuContext: {e}"))
    }

    /// Create a new CPU context using the backend-defined default thread count.
    pub fn new_default() -> Self {
        Self::try_new_default()
            .unwrap_or_else(|e| panic!("failed to initialize CpuContext with defaults: {e}"))
    }

    /// Returns the number of threads in the pool.
    pub fn num_threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Returns a reference to the underlying rayon thread pool.
    pub fn thread_pool(&self) -> &rayon::ThreadPool {
        self.pool.as_ref()
    }

    /// Run a closure inside the owned rayon thread pool.
    pub fn install<R>(&self, op: impl FnOnce() -> R + Send) -> R
    where
        R: Send,
    {
        self.pool.install(op)
    }

    /// Returns a mutable reference to the plan cache.
    pub fn plan_cache_mut(&mut self) -> &mut PlanCache {
        &mut self.plan_cache
    }

    #[allow(dead_code)]
    pub(crate) fn temp_pool_mut(&mut self) -> &mut TempPool {
        &mut self.temp_pool
    }

    #[cfg(feature = "gemm-blas")]
    pub(super) fn take_scratch<T>(&mut self, len: usize) -> Result<ScratchBuf<T>> {
        self.scratch.take(len)
    }

    #[cfg(feature = "gemm-blas")]
    pub(super) fn put_scratch<T>(&mut self, buf: ScratchBuf<T>) {
        self.scratch.put(buf);
    }
}

impl Default for CpuContext {
    fn default() -> Self {
        Self::new_default()
    }
}

impl TensorTempPoolContext for CpuContext {
    fn take_temp_vec<T: Send + 'static>(&mut self, len: usize) -> Vec<T> {
        self.temp_pool_mut().take_vec::<T>(len)
    }

    fn put_temp_vec<T: Send + 'static>(&mut self, vec: Vec<T>) {
        self.temp_pool_mut().put_vec(vec);
    }
}

/// CPU backend using strided-kernel and GEMM.
///
/// Dispatched automatically when tensors reside on
/// [`LogicalMemorySpace::MainMemory`](tenferro_device::LogicalMemorySpace::MainMemory).
/// Implements the semiring core and semiring fast-path families for
/// [`Standard<T>`](tenferro_algebra::Standard).
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_prims::{CpuBackend, CpuContext, SemiringCoreDescriptor, TensorSemiringCore};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::try_new(4).unwrap();
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a_base = Tensor::<f64>::zeros(&[3, 4], mem, col).unwrap();
/// let a = a_base.permute(&[1, 0]).unwrap();
/// let mut b = Tensor::<f64>::zeros(&[4, 3], mem, col).unwrap();
/// let plan = <CpuBackend as TensorSemiringCore<Standard<f64>>>::plan(
///     &mut ctx,
///     &SemiringCoreDescriptor::MakeContiguous,
///     &[&[4, 3], &[4, 3]],
/// )
/// .unwrap();
/// <CpuBackend as TensorSemiringCore<Standard<f64>>>::execute(
///     &mut ctx,
///     &plan,
///     1.0,
///     &[&a],
///     0.0,
///     &mut b,
/// )
/// .unwrap();
/// ```
pub struct CpuBackend;

impl CpuBackend {
    pub(super) fn supports_batched_gemm_type<T: Scalar>() -> bool {
        common::is_supported_scalar_type::<T>()
    }

    /// Materialize a lazily-conjugated tensor.
    ///
    /// If `src.is_conjugated()` is `false`, returns a shallow clone.
    /// If `true`, routes through the tensor-layer logical combine substrate so
    /// the result is resolved (`conjugated = false`) without reimplementing the
    /// copy logic here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_prims::{CpuBackend, CpuContext};
    ///
    /// let mut ctx = CpuContext::try_new(1).unwrap();
    /// let a_conj = a.into_conj(); // lazy
    /// let a_resolved = CpuBackend::resolve_conj(&mut ctx, &a_conj);
    /// assert!(!a_resolved.is_conjugated());
    /// ```
    pub fn resolve_conj<T: Scalar + Conjugate>(
        _ctx: &mut CpuContext,
        src: &Tensor<T>,
    ) -> Tensor<T> {
        if !src.is_conjugated() {
            return src.clone();
        }

        Tensor::stack(&[src], 0)
            .and_then(|tensor| tensor.squeeze_dim(0))
            .unwrap_or_else(|_| src.clone())
    }
}
