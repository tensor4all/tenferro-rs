use tenferro_algebra::{Conjugate, Scalar};
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::infra::plan_cache::PlanCache;

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
    pub(super) pool: rayon::ThreadPool,
    pub(super) plan_cache: PlanCache,
    pub(super) temp_pool: TempPool,
    #[cfg(feature = "gemm-blas")]
    scratch: ScratchPool,
}

impl CpuContext {
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
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| Error::DeviceError(format!("failed to build rayon thread pool: {e}")))?;
        Ok(Self {
            pool,
            plan_cache: PlanCache::new(),
            temp_pool: TempPool::default(),
            #[cfg(feature = "gemm-blas")]
            scratch: ScratchPool::default(),
        })
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

    /// Returns the number of threads in the pool.
    pub fn num_threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Returns a reference to the underlying rayon thread pool.
    pub fn thread_pool(&self) -> &rayon::ThreadPool {
        &self.pool
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

    /// Returns a mutable reference to the reusable temporary pool.
    pub fn temp_pool_mut(&mut self) -> &mut TempPool {
        &mut self.temp_pool
    }

    #[cfg(feature = "gemm-faer")]
    /// Returns the faer parallelism policy derived from this context.
    pub fn faer_parallelism(&self) -> faer::Par {
        faer::Par::rayon(self.num_threads())
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
