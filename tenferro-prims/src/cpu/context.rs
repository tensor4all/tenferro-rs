use std::any::TypeId;

use num_complex::{Complex32, Complex64};
use tenferro_algebra::{Conjugate, Scalar};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::PlanCache;

#[cfg(feature = "gemm-blas")]
use super::scratch::{ScratchBuf, ScratchPool};

/// CPU execution context.
///
/// Encapsulates CPU-side execution resources, analogous to cuTENSOR's
/// `cutensorHandle_t`. Holds a rayon thread pool and a [`PlanCache`]
/// for plan reuse. Intermediate buffer allocation relies on the global
/// allocator (e.g., mimalloc/jemalloc) rather than a custom buffer pool.
///
/// # Examples
///
/// ```
/// use tenferro_prims::CpuContext;
///
/// let mut ctx = CpuContext::new(4); // 4-thread pool
/// assert_eq!(ctx.num_threads(), 4);
/// ```
pub struct CpuContext {
    pub(super) pool: rayon::ThreadPool,
    pub(super) plan_cache: PlanCache,
    #[cfg(feature = "gemm-blas")]
    scratch: ScratchPool,
}

impl CpuContext {
    /// Create a new CPU context with the given number of threads.
    pub fn new(num_threads: usize) -> Self {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap_or_else(|e| panic!("failed to build rayon thread pool: {e}"));
        Self {
            pool,
            plan_cache: PlanCache::new(),
            #[cfg(feature = "gemm-blas")]
            scratch: ScratchPool::default(),
        }
    }

    /// Returns the number of threads in the pool.
    pub fn num_threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Returns a reference to the underlying rayon thread pool.
    pub fn thread_pool(&self) -> &rayon::ThreadPool {
        &self.pool
    }

    /// Returns a mutable reference to the plan cache.
    pub fn plan_cache_mut(&mut self) -> &mut PlanCache {
        &mut self.plan_cache
    }

    #[cfg(feature = "gemm-blas")]
    pub(super) fn take_scratch<T>(&mut self, len: usize) -> ScratchBuf<T> {
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
/// let mut ctx = CpuContext::new(4);
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a_base = Tensor::<f64>::zeros(&[3, 4], mem, col);
/// let a = a_base.permute(&[1, 0]).unwrap();
/// let mut b = Tensor::<f64>::zeros(&[4, 3], mem, col);
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
        let tid = TypeId::of::<T>();
        tid == TypeId::of::<f32>()
            || tid == TypeId::of::<f64>()
            || tid == TypeId::of::<Complex32>()
            || tid == TypeId::of::<Complex64>()
    }

    /// Materialize a lazily-conjugated tensor.
    ///
    /// If `src.is_conjugated()` is `false`, returns a shallow clone.
    /// If `true`, applies element-wise conjugation directly and returns a new
    /// tensor with `conjugated = false`.
    ///
    /// This is the equivalent of PyTorch's `torch.resolve_conj()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_prims::{CpuBackend, CpuContext};
    ///
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
        let contiguous = src.contiguous(MemoryOrder::ColumnMajor);
        let Some(data) = contiguous.buffer().as_slice() else {
            return src.clone();
        };
        let conjugated_data: Vec<T> = data.iter().map(|&v| v.conj()).collect();
        Tensor::from_slice(&conjugated_data, src.dims(), MemoryOrder::ColumnMajor)
            .unwrap_or_else(|_| src.clone())
    }
}
