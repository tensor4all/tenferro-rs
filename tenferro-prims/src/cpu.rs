use std::any::{Any, TypeId};
use std::cmp::Ordering;
use std::marker::PhantomData;

#[cfg(feature = "gemm-blas")]
use std::alloc::{self, Layout};
#[cfg(feature = "gemm-blas")]
use std::collections::BTreeMap;
#[cfg(feature = "gemm-blas")]
use std::ops::{Deref, DerefMut};
#[cfg(feature = "gemm-blas")]
use std::ptr::NonNull;

use num_complex::{Complex32, Complex64};
use strided_perm::try_fuse_group;
use strided_view::{StridedView, StridedViewMut};
use tenferro_algebra::{Conjugate, Scalar, Standard};
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::{
    for_each_index, mode_position, validate_execute_inputs, validate_rank, validate_shape_count,
    validate_shape_eq, Extension, PlanCache, PrimDescriptor, ReduceOp, TensorPrims, UnaryOp,
};

/// Trait for types that support strided GEMM via faer (zero-copy, zero-allocation).
///
/// Computes `C = beta * C + alpha * A * B` using faer's `matmul` with arbitrary strides.
#[cfg(feature = "gemm-faer")]
trait FaerGemm: Scalar {
    /// # Safety
    /// All pointers must be valid for the given dimensions and strides.
    #[allow(clippy::too_many_arguments)]
    unsafe fn strided_gemm(
        alpha: Self,
        a_ptr: *const Self,
        m: usize,
        k: usize,
        a_rs: isize,
        a_cs: isize,
        b_ptr: *const Self,
        n: usize,
        b_rs: isize,
        b_cs: isize,
        beta: Self,
        c_ptr: *mut Self,
        c_rs: isize,
        c_cs: isize,
    );
}

#[cfg(feature = "gemm-faer")]
macro_rules! impl_faer_gemm {
    ($ty:ty) => {
        impl FaerGemm for $ty {
            unsafe fn strided_gemm(
                alpha: $ty,
                a_ptr: *const $ty,
                m: usize,
                k: usize,
                a_rs: isize,
                a_cs: isize,
                b_ptr: *const $ty,
                n: usize,
                b_rs: isize,
                b_cs: isize,
                beta: $ty,
                c_ptr: *mut $ty,
                c_rs: isize,
                c_cs: isize,
            ) {
                use faer::{Accum, MatMut, MatRef, Par};
                let a_mat = MatRef::<$ty>::from_raw_parts(a_ptr, m, k, a_rs, a_cs);
                let b_mat = MatRef::<$ty>::from_raw_parts(b_ptr, k, n, b_rs, b_cs);
                let zero = <$ty as num_traits::Zero>::zero();
                let one = <$ty as num_traits::One>::one();
                let accum = if beta == zero {
                    Accum::Replace
                } else {
                    if beta != one {
                        let mut col_off = 0isize;
                        for _ in 0..n {
                            let mut off = col_off;
                            for _ in 0..m {
                                *c_ptr.offset(off) *= beta;
                                off += c_rs;
                            }
                            col_off += c_cs;
                        }
                    }
                    Accum::Add
                };
                let mut c_mat = MatMut::<$ty>::from_raw_parts_mut(c_ptr, m, n, c_rs, c_cs);
                faer::linalg::matmul::matmul(
                    &mut c_mat,
                    accum,
                    &a_mat,
                    &b_mat,
                    alpha,
                    Par::rayon(0),
                );
            }
        }
    };
}

#[cfg(feature = "gemm-faer")]
impl_faer_gemm!(f64);
#[cfg(feature = "gemm-faer")]
impl_faer_gemm!(f32);
#[cfg(feature = "gemm-faer")]
impl_faer_gemm!(Complex64);
#[cfg(feature = "gemm-faer")]
impl_faer_gemm!(Complex32);

/// Convert a CPU tensor to an immutable strided view.
fn tensor_to_view<T: Scalar>(t: &Tensor<T>) -> Result<StridedView<'_, T>> {
    let data = t
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?;
    StridedView::new(data, t.dims(), t.strides(), t.offset())
        .map_err(|e| Error::StrideError(format!("{e}")))
}

/// Convert a CPU tensor to a mutable strided view.
fn tensor_to_view_mut<T: Scalar>(t: &mut Tensor<T>) -> Result<StridedViewMut<'_, T>> {
    let dims = t.dims().to_vec();
    let strides = t.strides().to_vec();
    let offset = t.offset();
    let data = t
        .buffer_mut()
        .as_mut_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?;
    StridedViewMut::new(data, &dims, &strides, offset)
        .map_err(|e| Error::StrideError(format!("{e}")))
}

/// Compute connected components from a list of paired axis positions using union-find.
///
/// Returns `(components, comp_dims)` where:
/// - `components[i]` = sorted list of all axis positions in the i-th component
/// - `comp_dims[i]` = the shared dimension of the i-th component (looked up from `shape`)
fn compute_paired_components(
    paired_axes: &[(usize, usize)],
    shape: &[usize],
) -> (Vec<Vec<usize>>, Vec<usize>) {
    use std::collections::HashMap;

    if paired_axes.is_empty() {
        return (vec![], vec![]);
    }

    // Collect all axes that appear in paired_axes
    let mut all_axes: Vec<usize> = Vec::new();
    for &(ax1, ax2) in paired_axes {
        all_axes.push(ax1);
        all_axes.push(ax2);
    }
    all_axes.sort();
    all_axes.dedup();

    // Union-find: parent map
    let mut parent: HashMap<usize, usize> = all_axes.iter().map(|&ax| (ax, ax)).collect();

    fn find(parent: &mut HashMap<usize, usize>, x: usize) -> usize {
        let p = parent[&x];
        if p != x {
            let root = find(parent, p);
            parent.insert(x, root);
            root
        } else {
            x
        }
    }

    // Union each pair
    for &(ax1, ax2) in paired_axes {
        let r1 = find(&mut parent, ax1);
        let r2 = find(&mut parent, ax2);
        if r1 != r2 {
            // Union: make smaller root the parent (deterministic)
            let (lo, hi) = if r1 < r2 { (r1, r2) } else { (r2, r1) };
            parent.insert(hi, lo);
        }
    }

    // Group by root
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for &ax in &all_axes {
        let root = find(&mut parent, ax);
        groups.entry(root).or_default().push(ax);
    }

    // Sort components by their minimum axis for determinism
    let mut components: Vec<Vec<usize>> = groups.into_values().collect();
    components.sort_by_key(|c| c[0]);

    let comp_dims: Vec<usize> = components.iter().map(|c| shape[c[0]]).collect();

    (components, comp_dims)
}

/// Pre-computed mode analysis for Contract GEMM fast path.
#[derive(Debug, Clone)]
pub(crate) struct ContractGemmSpec {
    /// Target mode order for A: [batch, m, k]
    a_target: Vec<u32>,
    /// Target mode order for B: [batch, k, n]
    b_target: Vec<u32>,
    /// Target mode order for C: [batch, m, n]
    c_target: Vec<u32>,
    batch_modes: Vec<u32>,
    m_modes: Vec<u32>,
    n_modes: Vec<u32>,
    k_modes: Vec<u32>,
}

/// Build a [`ContractGemmSpec`] from mode labels, or `None` if the
/// contraction is not a valid batched-GEMM pattern.
fn build_contract_gemm_spec(
    modes_a: &[u32],
    modes_b: &[u32],
    modes_c: &[u32],
) -> Option<ContractGemmSpec> {
    let batch_modes: Vec<u32> = modes_c
        .iter()
        .copied()
        .filter(|m| modes_a.contains(m) && modes_b.contains(m))
        .collect();
    let m_modes: Vec<u32> = modes_c
        .iter()
        .copied()
        .filter(|m| modes_a.contains(m) && !modes_b.contains(m))
        .collect();
    let n_modes: Vec<u32> = modes_c
        .iter()
        .copied()
        .filter(|m| modes_b.contains(m) && !modes_a.contains(m))
        .collect();
    let k_modes: Vec<u32> = modes_a
        .iter()
        .copied()
        .filter(|m| modes_b.contains(m) && !modes_c.contains(m))
        .collect();

    let expected_a = batch_modes.len() + m_modes.len() + k_modes.len();
    let expected_b = batch_modes.len() + k_modes.len() + n_modes.len();
    if expected_a != modes_a.len() || expected_b != modes_b.len() {
        return None;
    }
    if batch_modes.len() + m_modes.len() + n_modes.len() != modes_c.len() {
        return None;
    }

    let a_target: Vec<u32> = batch_modes
        .iter()
        .chain(m_modes.iter())
        .chain(k_modes.iter())
        .copied()
        .collect();
    let b_target: Vec<u32> = batch_modes
        .iter()
        .chain(k_modes.iter())
        .chain(n_modes.iter())
        .copied()
        .collect();
    let c_target: Vec<u32> = batch_modes
        .iter()
        .chain(m_modes.iter())
        .chain(n_modes.iter())
        .copied()
        .collect();

    Some(ContractGemmSpec {
        a_target,
        b_target,
        c_target,
        batch_modes,
        m_modes,
        n_modes,
        k_modes,
    })
}

/// CPU plan — concrete enum, no type erasure.
///
/// Created by [`CpuBackend::plan`](TensorPrims::plan) and consumed by
/// [`CpuBackend::execute`](TensorPrims::execute).
#[derive(Debug, Clone)]
pub enum CpuPlan<T: Scalar> {
    /// Plan for batched GEMM.
    BatchedGemm {
        /// Batch dimension sizes.
        batch_dims: Vec<usize>,
        /// Number of rows.
        m: usize,
        /// Number of columns.
        n: usize,
        /// Contraction dimension.
        k: usize,
        _marker: PhantomData<T>,
    },
    /// Plan for reduction.
    Reduce {
        /// Axes to reduce over (positions in input tensor).
        reduced_axes: Vec<usize>,
        /// Reduction operation.
        op: ReduceOp,
        _marker: PhantomData<T>,
    },
    /// Plan for trace.
    Trace {
        /// Output axis positions mapping.
        free_axes: Vec<usize>,
        /// Connected components of paired axes (union-find groups).
        /// Each inner Vec contains all axis positions in one component.
        components: Vec<Vec<usize>>,
        /// Dimension of each component (all axes in a component share the same dim).
        comp_dims: Vec<usize>,
        _marker: PhantomData<T>,
    },
    /// Plan for permutation.
    Permute {
        /// Permutation mapping (`perm[out_axis] = in_axis`).
        perm: Vec<usize>,
        _marker: PhantomData<T>,
    },
    /// Plan for anti-trace (AD backward).
    AntiTrace {
        /// Paired axis positions in output tensor.
        paired_axes: Vec<(usize, usize)>,
        /// Input axis positions mapping.
        free_axes: Vec<usize>,
        /// Connected components of paired axes (union-find groups).
        components: Vec<Vec<usize>>,
        /// Dimension of each component.
        comp_dims: Vec<usize>,
        _marker: PhantomData<T>,
    },
    /// Plan for anti-diag (AD backward).
    AntiDiag {
        /// Paired axis positions in output tensor.
        paired_axes: Vec<(usize, usize)>,
        /// Input axis positions mapping.
        free_axes: Vec<usize>,
        /// Connected components of paired axes (union-find groups).
        components: Vec<Vec<usize>>,
        /// Dimension of each component.
        comp_dims: Vec<usize>,
        /// Indices of generative components (no overlap with free axes).
        generative_comps: Vec<usize>,
        _marker: PhantomData<T>,
    },
    /// Plan for element-wise unary operation.
    ElementwiseUnary {
        /// Unary operation.
        op: UnaryOp,
        _marker: PhantomData<T>,
    },
    /// Plan for fused contraction.
    Contract {
        /// Mode labels for input A.
        modes_a: Vec<u32>,
        /// Mode labels for input B.
        modes_b: Vec<u32>,
        /// Mode labels for output C.
        modes_c: Vec<u32>,
        /// Cached GEMM mode analysis (None if not a valid GEMM pattern).
        #[allow(private_interfaces)]
        gemm_spec: Option<ContractGemmSpec>,
        _marker: PhantomData<T>,
    },
    /// Plan for element-wise multiplication (extended op).
    ElementwiseMul { _marker: PhantomData<T> },
    /// Plan for making a tensor contiguous.
    MakeContiguous { _marker: PhantomData<T> },
}

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
    pool: rayon::ThreadPool,
    plan_cache: PlanCache,
    #[cfg(feature = "gemm-blas")]
    scratch: ScratchPool,
}

/// Alignment for all scratch allocations (cache-line / AVX-512).
#[cfg(feature = "gemm-blas")]
const SCRATCH_ALIGN: usize = 64;

// Compile-time guarantee: scratch alignment is sufficient for the widest scalar type.
#[cfg(feature = "gemm-blas")]
const _: () = assert!(SCRATCH_ALIGN >= std::mem::align_of::<f64>());

/// Raw byte buffer stored in the pool. Does NOT impl Drop — the pool
/// handles deallocation in its own Drop impl.
#[cfg(feature = "gemm-blas")]
struct RawBuf {
    ptr: NonNull<u8>,
    cap_bytes: usize,
}

// SAFETY: The underlying allocation is exclusively owned; sending across
// threads is safe as long as no aliased references exist.
#[cfg(feature = "gemm-blas")]
unsafe impl Send for RawBuf {}

/// Typed scratch buffer obtained from [`ScratchPool`].
///
/// Dereferences to `&[T]` / `&mut [T]`. On the normal path the caller
/// returns the buffer to the pool via [`ScratchPool::put`]; if dropped
/// without returning (e.g. during a panic), Drop deallocates the raw
/// memory so there is no leak.
#[cfg(feature = "gemm-blas")]
pub(crate) struct ScratchBuf<T> {
    ptr: NonNull<u8>,
    cap_bytes: usize,
    len: usize,
    _marker: PhantomData<T>,
}

#[cfg(feature = "gemm-blas")]
impl<T> ScratchBuf<T> {
    /// Extract the raw buffer, consuming self **without** running Drop.
    fn into_raw(self) -> RawBuf {
        let raw = RawBuf {
            ptr: self.ptr,
            cap_bytes: self.cap_bytes,
        };
        std::mem::forget(self);
        raw
    }
}

#[cfg(feature = "gemm-blas")]
impl<T> Deref for ScratchBuf<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: ptr is SCRATCH_ALIGN-aligned (>= align_of::<T>()),
        // and len elements fit within cap_bytes.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const T, self.len) }
    }
}

#[cfg(feature = "gemm-blas")]
impl<T> DerefMut for ScratchBuf<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        if self.len == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut T, self.len) }
    }
}

#[cfg(feature = "gemm-blas")]
impl<T> Drop for ScratchBuf<T> {
    fn drop(&mut self) {
        if self.cap_bytes > 0 {
            // SAFETY: ptr was allocated with Layout(cap_bytes, SCRATCH_ALIGN).
            let layout = Layout::from_size_align(self.cap_bytes, SCRATCH_ALIGN)
                .expect("invalid scratch layout in drop");
            unsafe { alloc::dealloc(self.ptr.as_ptr(), layout) };
        }
    }
}

/// Type-independent byte-level scratch pool. Buffers are keyed by byte
/// capacity so an f64 allocation can be reused for f32 or vice-versa.
#[derive(Default)]
#[cfg(feature = "gemm-blas")]
struct ScratchPool {
    pool: BTreeMap<usize, Vec<RawBuf>>,
}

#[cfg(feature = "gemm-blas")]
impl ScratchPool {
    /// Obtain a scratch buffer holding at least `len` elements of `T`.
    /// Contents are **uninitialized**; callers must overwrite before reading.
    fn take<T>(&mut self, len: usize) -> ScratchBuf<T> {
        debug_assert!(
            SCRATCH_ALIGN >= std::mem::align_of::<T>(),
            "SCRATCH_ALIGN ({SCRATCH_ALIGN}) < align_of::<T> ({})",
            std::mem::align_of::<T>(),
        );
        let needed = len
            .checked_mul(std::mem::size_of::<T>())
            .expect("scratch size overflow");
        let raw = self
            .pool
            .range(needed..)
            .next()
            .map(|(&k, _)| k)
            .and_then(|k| {
                let bucket = self.pool.get_mut(&k)?;
                let buf = bucket.pop()?;
                if bucket.is_empty() {
                    self.pool.remove(&k);
                }
                Some(buf)
            });
        let (ptr, cap_bytes) = match raw {
            Some(buf) => (buf.ptr, buf.cap_bytes),
            None => {
                if needed == 0 {
                    return ScratchBuf {
                        ptr: NonNull::dangling(),
                        cap_bytes: 0,
                        len: 0,
                        _marker: PhantomData,
                    };
                }
                let layout =
                    Layout::from_size_align(needed, SCRATCH_ALIGN).expect("invalid scratch layout");
                // SAFETY: layout has non-zero size.
                let ptr = unsafe { alloc::alloc(layout) };
                let ptr = NonNull::new(ptr).expect("scratch allocation failed");
                (ptr, needed)
            }
        };
        ScratchBuf {
            ptr,
            cap_bytes,
            len,
            _marker: PhantomData,
        }
    }

    /// Return a scratch buffer to the pool for later reuse.
    fn put<T>(&mut self, buf: ScratchBuf<T>) {
        let raw = buf.into_raw();
        if raw.cap_bytes == 0 {
            return;
        }
        self.pool.entry(raw.cap_bytes).or_default().push(raw);
    }
}

#[cfg(feature = "gemm-blas")]
impl Drop for ScratchPool {
    fn drop(&mut self) {
        for (_, bufs) in std::mem::take(&mut self.pool) {
            for buf in bufs {
                let layout = Layout::from_size_align(buf.cap_bytes, SCRATCH_ALIGN)
                    .expect("invalid scratch layout in pool drop");
                // SAFETY: each RawBuf was allocated with this layout.
                unsafe { alloc::dealloc(buf.ptr.as_ptr(), layout) };
            }
        }
    }
}

impl CpuContext {
    /// Create a new CPU context with the given number of threads.
    pub fn new(num_threads: usize) -> Self {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("failed to build rayon thread pool");
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
    fn take_scratch<T>(&mut self, len: usize) -> ScratchBuf<T> {
        self.scratch.take(len)
    }

    #[cfg(feature = "gemm-blas")]
    fn put_scratch<T>(&mut self, buf: ScratchBuf<T>) {
        self.scratch.put(buf);
    }
}

/// CPU backend using strided-kernel and GEMM.
///
/// Dispatched automatically when tensors reside on
/// [`LogicalMemorySpace::MainMemory`](tenferro_device::LogicalMemorySpace::MainMemory).
/// Implements [`TensorPrims<Standard<T>>`](TensorPrims) for standard arithmetic.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(4);
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
/// let mut b = Tensor::<f64>::zeros(&[4, 3], mem, col);
/// let desc = PrimDescriptor::Permute {
///     modes_a: vec![0, 1],
///     modes_b: vec![1, 0],
/// };
/// let plan =
///     <CpuBackend as TensorPrims<Standard<f64>>>::plan(&mut ctx, &desc, &[&[3, 4], &[4, 3]])
///         .unwrap();
/// <CpuBackend as TensorPrims<Standard<f64>>>::execute(
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
    fn supports_batched_gemm_type<T: Scalar>() -> bool {
        let tid = TypeId::of::<T>();
        tid == TypeId::of::<f32>()
            || tid == TypeId::of::<f64>()
            || tid == TypeId::of::<Complex32>()
            || tid == TypeId::of::<Complex64>()
    }

    fn supports_ordered_reduce_type<T: Scalar>() -> bool {
        let tid = TypeId::of::<T>();
        tid == TypeId::of::<f32>() || tid == TypeId::of::<f64>()
    }

    /// Materialize a lazily-conjugated tensor.
    ///
    /// If `src.is_conjugated()` is `false`, returns a shallow clone.
    /// If `true`, applies element-wise conjugation via
    /// `ElementwiseUnary(Conj)` and returns a new tensor with
    /// `conjugated = false`.
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
        src: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_tensor::Tensor<T> {
        if !src.is_conjugated() {
            return src.clone();
        }
        // Create a fresh non-conjugated copy with element-wise conjugation applied.
        // For real types (f64, f32), Conjugate::conj() is identity so this is a plain copy.
        // For complex types (Complex64, Complex32), conj() negates the imaginary part.
        let contiguous = src.contiguous(tenferro_tensor::MemoryOrder::ColumnMajor);
        let data = contiguous
            .buffer()
            .as_slice()
            .expect("CPU tensor must have CPU-accessible data");
        let conjugated_data: Vec<T> = data.iter().map(|&v| v.conj()).collect();
        tenferro_tensor::Tensor::from_slice(
            &conjugated_data,
            src.dims(),
            tenferro_tensor::MemoryOrder::ColumnMajor,
        )
        .expect("from_slice should succeed with valid data and dims")
    }

    /// Build a CPU plan from a descriptor and shapes (without cache lookup).
    ///
    /// This is the internal plan construction logic, factored out of
    /// [`TensorPrims::plan`] so that the trait method can wrap it with
    /// cache lookup/insert.
    fn build_plan<T: Scalar>(desc: &PrimDescriptor, shapes: &[&[usize]]) -> Result<CpuPlan<T>> {
        match desc {
            PrimDescriptor::BatchedGemm {
                batch_dims,
                m,
                n,
                k,
            } => {
                // BatchedGemm expects 3 shapes: A, B, C
                // Layout: [m/k, k/n, batch...] — batch dims are trailing
                validate_shape_count(shapes, 3, "BatchedGemm")?;
                if !Self::supports_batched_gemm_type::<T>() {
                    return Err(Error::InvalidArgument(format!(
                        "BatchedGemm supports only f32, f64, Complex32, and Complex64 (got {})",
                        std::any::type_name::<T>()
                    )));
                }
                let expected_a: Vec<usize> = [*m, *k]
                    .iter()
                    .copied()
                    .chain(batch_dims.iter().copied())
                    .collect();
                let expected_b: Vec<usize> = [*k, *n]
                    .iter()
                    .copied()
                    .chain(batch_dims.iter().copied())
                    .collect();
                let expected_c: Vec<usize> = [*m, *n]
                    .iter()
                    .copied()
                    .chain(batch_dims.iter().copied())
                    .collect();
                validate_shape_eq(shapes[0], &expected_a, "BatchedGemm input A")?;
                validate_shape_eq(shapes[1], &expected_b, "BatchedGemm input B")?;
                validate_shape_eq(shapes[2], &expected_c, "BatchedGemm output C")?;
                Ok(CpuPlan::BatchedGemm {
                    batch_dims: batch_dims.clone(),
                    m: *m,
                    n: *n,
                    k: *k,
                    _marker: PhantomData,
                })
            }

            PrimDescriptor::Reduce {
                modes_a,
                modes_c,
                op,
            } => {
                // Reduce expects 2 shapes: A (input), C (output)
                validate_shape_count(shapes, 2, "Reduce")?;
                validate_rank(shapes[0], modes_a.len(), "Reduce input A")?;
                validate_rank(shapes[1], modes_c.len(), "Reduce output C")?;
                if matches!(op, ReduceOp::Max | ReduceOp::Min)
                    && !Self::supports_ordered_reduce_type::<T>()
                {
                    return Err(Error::InvalidArgument(format!(
                        "Reduce Max/Min supports only f32 and f64 on CpuBackend (got {})",
                        std::any::type_name::<T>()
                    )));
                }
                // reduced_axes = positions in modes_a not present in modes_c
                let reduced_axes: Vec<usize> = modes_a
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| !modes_c.contains(m))
                    .map(|(i, _)| i)
                    .collect();
                // Validate: reduced_axes must be sorted, unique, and within rank
                for w in reduced_axes.windows(2) {
                    if w[0] >= w[1] {
                        return Err(Error::InvalidArgument(format!(
                            "Reduce: reduced_axes must be sorted and unique, got {reduced_axes:?}"
                        )));
                    }
                }
                if let Some(&last) = reduced_axes.last() {
                    if last >= modes_a.len() {
                        return Err(Error::InvalidArgument(format!(
                            "Reduce: reduced axis {last} out of range for rank {}",
                            modes_a.len()
                        )));
                    }
                }
                Ok(CpuPlan::Reduce {
                    reduced_axes,
                    op: *op,
                    _marker: PhantomData,
                })
            }

            PrimDescriptor::Trace {
                modes_a,
                modes_c,
                paired,
            } => {
                // Trace expects 2 shapes: A (input), C (output)
                validate_shape_count(shapes, 2, "Trace")?;
                validate_rank(shapes[0], modes_a.len(), "Trace input A")?;
                validate_rank(shapes[1], modes_c.len(), "Trace output C")?;
                let paired_axes: Vec<(usize, usize)> = paired
                    .iter()
                    .map(|(m1, m2)| {
                        Ok((mode_position(modes_a, *m1)?, mode_position(modes_a, *m2)?))
                    })
                    .collect::<Result<_>>()?;
                // Validate that paired axes have equal dimensions
                for &(ax1, ax2) in &paired_axes {
                    if shapes[0][ax1] != shapes[0][ax2] {
                        return Err(Error::InvalidArgument(format!(
                            "Trace paired axes ({ax1}, {ax2}) have mismatched dimensions: {} vs {}",
                            shapes[0][ax1], shapes[0][ax2]
                        )));
                    }
                }
                let free_axes: Vec<usize> = modes_c
                    .iter()
                    .map(|m| mode_position(modes_a, *m))
                    .collect::<Result<_>>()?;
                let (components, comp_dims) = compute_paired_components(&paired_axes, shapes[0]);
                Ok(CpuPlan::Trace {
                    free_axes,
                    components,
                    comp_dims,
                    _marker: PhantomData,
                })
            }

            PrimDescriptor::Permute { modes_a, modes_b } => {
                // Permute expects 2 shapes: A (input), B (output)
                validate_shape_count(shapes, 2, "Permute")?;
                validate_rank(shapes[0], modes_a.len(), "Permute input A")?;
                validate_rank(shapes[1], modes_b.len(), "Permute output B")?;
                // perm[out_axis] = in_axis
                let perm: Vec<usize> = modes_b
                    .iter()
                    .map(|m| mode_position(modes_a, *m))
                    .collect::<Result<_>>()?;
                Ok(CpuPlan::Permute {
                    perm,
                    _marker: PhantomData,
                })
            }

            PrimDescriptor::AntiTrace {
                modes_a,
                modes_c,
                paired,
            } => {
                // AntiTrace expects 2 shapes: A (input), C (output)
                validate_shape_count(shapes, 2, "AntiTrace")?;
                validate_rank(shapes[0], modes_a.len(), "AntiTrace input A")?;
                validate_rank(shapes[1], modes_c.len(), "AntiTrace output C")?;
                let paired_axes: Vec<(usize, usize)> = paired
                    .iter()
                    .map(|(m1, m2)| {
                        Ok((mode_position(modes_c, *m1)?, mode_position(modes_c, *m2)?))
                    })
                    .collect::<Result<_>>()?;
                // Validate that paired axes in output have equal dimensions
                for &(ax1, ax2) in &paired_axes {
                    if shapes[1][ax1] != shapes[1][ax2] {
                        return Err(Error::InvalidArgument(format!(
                            "AntiTrace paired axes ({ax1}, {ax2}) have mismatched dimensions: {} vs {}",
                            shapes[1][ax1], shapes[1][ax2]
                        )));
                    }
                }
                let free_axes: Vec<usize> = modes_a
                    .iter()
                    .map(|m| mode_position(modes_c, *m))
                    .collect::<Result<_>>()?;
                let (components, comp_dims) = compute_paired_components(&paired_axes, shapes[1]);
                Ok(CpuPlan::AntiTrace {
                    paired_axes,
                    free_axes,
                    components,
                    comp_dims,
                    _marker: PhantomData,
                })
            }

            PrimDescriptor::AntiDiag {
                modes_a,
                modes_c,
                paired,
            } => {
                // AntiDiag expects 2 shapes: A (input), C (output)
                validate_shape_count(shapes, 2, "AntiDiag")?;
                validate_rank(shapes[0], modes_a.len(), "AntiDiag input A")?;
                validate_rank(shapes[1], modes_c.len(), "AntiDiag output C")?;
                let paired_axes: Vec<(usize, usize)> = paired
                    .iter()
                    .map(|(m1, m2)| {
                        Ok((mode_position(modes_c, *m1)?, mode_position(modes_c, *m2)?))
                    })
                    .collect::<Result<_>>()?;
                let free_axes: Vec<usize> = modes_a
                    .iter()
                    .map(|m| mode_position(modes_c, *m))
                    .collect::<Result<_>>()?;
                let (components, comp_dims) = compute_paired_components(&paired_axes, shapes[1]);
                // Generative components: those whose axes have no overlap with free_axes
                let free_ax_set: std::collections::HashSet<usize> =
                    free_axes.iter().copied().collect();
                let generative_comps: Vec<usize> = components
                    .iter()
                    .enumerate()
                    .filter(|(_, comp)| comp.iter().all(|ax| !free_ax_set.contains(ax)))
                    .map(|(i, _)| i)
                    .collect();
                Ok(CpuPlan::AntiDiag {
                    paired_axes,
                    free_axes,
                    components,
                    comp_dims,
                    generative_comps,
                    _marker: PhantomData,
                })
            }

            PrimDescriptor::ElementwiseUnary { op } => {
                // ElementwiseUnary expects 2 shapes: A (input), C (output)
                validate_shape_count(shapes, 2, "ElementwiseUnary")?;
                validate_shape_eq(shapes[1], shapes[0], "ElementwiseUnary output")?;
                Ok(CpuPlan::ElementwiseUnary {
                    op: *op,
                    _marker: PhantomData,
                })
            }

            PrimDescriptor::Contract {
                modes_a,
                modes_b,
                modes_c,
            } => {
                // Contract expects 3 shapes: A, B, C.
                // When GEMM acceleration is available (gemm-blas feature), the
                // plan builder tries to map the contraction to a batched GEMM.
                // If the mode layout is incompatible with GEMM (e.g. non-trivial
                // batch alignment), the plan falls back to the generic O(m·n·k)
                // loop in execute_contract_generic. This is correct but slower.
                validate_shape_count(shapes, 3, "Contract")?;
                validate_rank(shapes[0], modes_a.len(), "Contract input A")?;
                validate_rank(shapes[1], modes_b.len(), "Contract input B")?;
                validate_rank(shapes[2], modes_c.len(), "Contract output C")?;
                // Validate contracted dimensions match between A and B
                for &mode in modes_a.iter() {
                    if let Some(b_pos) = modes_b.iter().position(|&m| m == mode) {
                        let a_pos = modes_a.iter().position(|&m| m == mode).unwrap();
                        if shapes[0][a_pos] != shapes[1][b_pos] {
                            return Err(Error::InvalidArgument(format!(
                                "Contract mode {mode} has mismatched dimensions: A={} vs B={}",
                                shapes[0][a_pos], shapes[1][b_pos]
                            )));
                        }
                    }
                }
                let gemm_spec = build_contract_gemm_spec(modes_a, modes_b, modes_c);
                Ok(CpuPlan::Contract {
                    modes_a: modes_a.clone(),
                    modes_b: modes_b.clone(),
                    modes_c: modes_c.clone(),
                    gemm_spec,
                    _marker: PhantomData,
                })
            }

            PrimDescriptor::ElementwiseMul => {
                // ElementwiseMul expects 3 shapes: A, B, C
                validate_shape_count(shapes, 3, "ElementwiseMul")?;
                validate_shape_eq(shapes[1], shapes[0], "ElementwiseMul input B")?;
                validate_shape_eq(shapes[2], shapes[0], "ElementwiseMul output C")?;
                Ok(CpuPlan::ElementwiseMul {
                    _marker: PhantomData,
                })
            }

            PrimDescriptor::MakeContiguous => {
                // MakeContiguous expects 2 shapes: A (input), C (output)
                validate_shape_count(shapes, 2, "MakeContiguous")?;
                validate_shape_eq(shapes[1], shapes[0], "MakeContiguous output")?;
                Ok(CpuPlan::MakeContiguous {
                    _marker: PhantomData,
                })
            }
        }
    }
}

// ===========================================================================
// CPU execute helpers for each operation
// ===========================================================================

fn execute_permute<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    perm: &[usize],
) -> Result<()> {
    let permuted = input
        .permute(perm)
        .map_err(|e| Error::StrideError(e.to_string()))?;

    if alpha == T::one() && beta == T::zero() {
        // Fast path: use strided-perm HPTT-based copy
        strided_perm::copy_into_par(output, &permuted)
            .map_err(|e| Error::StrideError(e.to_string()))?;
    } else {
        let dims = output.dims().to_vec();
        for_each_index(&dims, |idx| {
            let val = alpha * permuted.get(idx);
            if beta == T::zero() {
                output.set(idx, val);
            } else {
                output.set(idx, val + beta * output.get(idx));
            }
        });
    }
    Ok(())
}

fn execute_make_contiguous<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
) -> Result<()> {
    if alpha == T::one() && beta == T::zero() {
        strided_perm::copy_into_par(output, input)
            .map_err(|e| Error::StrideError(e.to_string()))?;
    } else {
        let dims = output.dims().to_vec();
        for_each_index(&dims, |idx| {
            let val = alpha * input.get(idx);
            if beta == T::zero() {
                output.set(idx, val);
            } else {
                output.set(idx, val + beta * output.get(idx));
            }
        });
    }
    Ok(())
}

/// Fallback for the OpenBLAS backend, which requires contiguous column-major data.
/// Packs strided A, B, C into scratch buffers, calls contiguous gemm, unpacks C.
#[cfg(feature = "gemm-blas")]
fn execute_batched_gemm_contiguous<T: Scalar + 'static>(
    ctx: &mut CpuContext,
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
    batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
    gemm_fn: fn(T, &[T], &[T], T, &mut [T], usize, usize, usize) -> Result<()>,
) -> Result<()> {
    let a = inputs[0];
    let b = inputs[1];
    let nb = batch_dims.len();
    let a_strides = a.strides();
    let b_strides = b.strides();
    let c_strides = output.strides();
    // Layout: [row, col, batch...] — GEMM dims are leading, batch dims trailing
    let a_row = a_strides[0];
    let a_col = a_strides[1];
    let a_batch = &a_strides[2..];
    let b_row = b_strides[0];
    let b_col = b_strides[1];
    let b_batch = &b_strides[2..];
    let c_row = c_strides[0];
    let c_col = c_strides[1];
    let c_batch = &c_strides[2..];

    let a_ptr = a.ptr();
    let b_ptr = b.ptr();
    let c_ptr = output.as_mut_ptr();

    // Validate that batch strides are non-zero when batch dims are > 1,
    // ensuring that each batch slice accesses distinct memory.
    debug_assert!(
        batch_dims
            .iter()
            .enumerate()
            .all(|(i, &d)| d <= 1 || (a_batch[i] != 0 && b_batch[i] != 0 && c_batch[i] != 0)),
        "batch GEMM stride must be non-zero for batch dims > 1"
    );

    // Fast path: contiguous column-major → no packing needed
    if nb == 0
        && a_row == 1
        && a_col == m as isize
        && b_row == 1
        && b_col == k as isize
        && c_row == 1
        && c_col == m as isize
    {
        let a_mat = unsafe { std::slice::from_raw_parts(a_ptr, m * k) };
        let b_mat = unsafe { std::slice::from_raw_parts(b_ptr, k * n) };
        let c_mat = unsafe { std::slice::from_raw_parts_mut(c_ptr, m * n) };
        return gemm_fn(alpha, a_mat, b_mat, beta, c_mat, m, n, k);
    }

    let mut a_mat = ctx.take_scratch::<T>(m * k);
    let mut b_mat = ctx.take_scratch::<T>(k * n);
    let mut c_mat = ctx.take_scratch::<T>(m * n);

    // Per-batch GEMM kernel (packs strided operands into scratch buffers)
    let do_batch = |a_off: isize,
                    b_off: isize,
                    c_off: isize,
                    a_mat: &mut [T],
                    b_mat: &mut [T],
                    c_mat: &mut [T]|
     -> Result<()> {
        // SAFETY: All pointer offsets below are within the allocation bounds of
        // their respective tensors. The plan validation (validate_shape_eq) ensures
        // that m, k, n match the tensor dimensions, and batch offsets are computed
        // from strides that were validated at tensor construction time. The product
        // of dims is bounded by the allocation size of each tensor's data buffer.
        for kk in 0..k {
            for i in 0..m {
                let src_off = a_off + i as isize * a_row + kk as isize * a_col;
                a_mat[i + kk * m] = unsafe { *a_ptr.offset(src_off) };
            }
        }

        for j in 0..n {
            for kk in 0..k {
                let src_off = b_off + kk as isize * b_row + j as isize * b_col;
                b_mat[kk + j * k] = unsafe { *b_ptr.offset(src_off) };
            }
        }

        if beta == T::zero() {
            c_mat.fill(T::zero());
        } else {
            for j in 0..n {
                for i in 0..m {
                    let src_off = c_off + i as isize * c_row + j as isize * c_col;
                    c_mat[i + j * m] = unsafe { *c_ptr.offset(src_off) };
                }
            }
        }

        gemm_fn(alpha, a_mat, b_mat, beta, c_mat, m, n, k)?;

        for j in 0..n {
            for i in 0..m {
                let dst_off = c_off + i as isize * c_row + j as isize * c_col;
                unsafe {
                    *c_ptr.offset(dst_off) = c_mat[i + j * m];
                }
            }
        }
        Ok(())
    };

    let mut result = Ok(());

    if nb == 0 {
        result = do_batch(0, 0, 0, &mut a_mat, &mut b_mat, &mut c_mat);
    } else {
        // Independent fusability check per operand
        let a_fused = strided_perm::try_fuse_group(batch_dims, a_batch);
        let b_fused = strided_perm::try_fuse_group(batch_dims, b_batch);
        let c_fused = strided_perm::try_fuse_group(batch_dims, c_batch);
        let total: usize = batch_dims.iter().product();

        if let (Some((_, a_step)), Some((_, b_step)), Some((_, c_step))) =
            (a_fused, b_fused, c_fused)
        {
            // All-fused fast path: simple pointer increment
            let mut a_off = 0isize;
            let mut b_off = 0isize;
            let mut c_off = 0isize;
            for _ in 0..total {
                if let Err(e) = do_batch(a_off, b_off, c_off, &mut a_mat, &mut b_mat, &mut c_mat) {
                    result = Err(e);
                    break;
                }
                a_off += a_step;
                b_off += b_step;
                c_off += c_step;
            }
        } else {
            // Mixed path: each operand independently fused or strided
            let mut idx = vec![0usize; nb];
            for flat in 0..total {
                let a_off = batch_offset(flat, &idx, a_fused, a_batch);
                let b_off = batch_offset(flat, &idx, b_fused, b_batch);
                let c_off = batch_offset(flat, &idx, c_fused, c_batch);
                if let Err(e) = do_batch(a_off, b_off, c_off, &mut a_mat, &mut b_mat, &mut c_mat) {
                    result = Err(e);
                    break;
                }

                // Advance shared multi-index (carry-based)
                if flat + 1 < total {
                    for ax in 0..nb {
                        let next = idx[ax] + 1;
                        if next < batch_dims[ax] {
                            idx[ax] = next;
                            break;
                        } else {
                            idx[ax] = 0;
                        }
                    }
                }
            }
        }
    }

    ctx.put_scratch(a_mat);
    ctx.put_scratch(b_mat);
    ctx.put_scratch(c_mat);
    result
}

/// Compute batch offset for one operand: fused path uses flat index × step,
/// strided path uses multi-index dot product with strides.
#[inline]
fn batch_offset(
    flat: usize,
    idx: &[usize],
    fused: Option<(usize, isize)>,
    batch_strides: &[isize],
) -> isize {
    if let Some((_, step)) = fused {
        flat as isize * step
    } else {
        idx.iter()
            .zip(batch_strides)
            .map(|(&i, &s)| i as isize * s)
            .sum()
    }
}

/// Strided batched GEMM via faer — zero allocation, zero copy.
#[cfg(feature = "gemm-faer")]
#[allow(clippy::too_many_arguments)]
fn execute_batched_gemm_strided<T: FaerGemm>(
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
    batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let a = inputs[0];
    let b = inputs[1];
    let nb = batch_dims.len();
    let a_strides = a.strides();
    let b_strides = b.strides();
    let c_strides = output.strides();
    // Layout: [row, col, batch...] — GEMM dims are leading, batch dims trailing
    let a_row = a_strides[0];
    let a_col = a_strides[1];
    let a_batch = &a_strides[2..];
    let b_row = b_strides[0];
    let b_col = b_strides[1];
    let b_batch = &b_strides[2..];
    let c_row = c_strides[0];
    let c_col = c_strides[1];
    let c_batch = &c_strides[2..];

    let a_ptr = a.ptr();
    let b_ptr = b.ptr();
    let c_ptr = output.as_mut_ptr();

    let do_batch = |a_off: isize, b_off: isize, c_off: isize| unsafe {
        T::strided_gemm(
            alpha,
            a_ptr.offset(a_off),
            m,
            k,
            a_row,
            a_col,
            b_ptr.offset(b_off),
            n,
            b_row,
            b_col,
            beta,
            c_ptr.offset(c_off),
            c_row,
            c_col,
        );
    };

    if nb == 0 {
        do_batch(0, 0, 0);
    } else {
        // Independent fusability check per operand
        let a_fused = strided_perm::try_fuse_group(batch_dims, a_batch);
        let b_fused = strided_perm::try_fuse_group(batch_dims, b_batch);
        let c_fused = strided_perm::try_fuse_group(batch_dims, c_batch);
        let total: usize = batch_dims.iter().product();

        if let (Some((_, a_step)), Some((_, b_step)), Some((_, c_step))) =
            (a_fused, b_fused, c_fused)
        {
            // All-fused fast path: simple pointer increment
            let mut a_off = 0isize;
            let mut b_off = 0isize;
            let mut c_off = 0isize;
            for _ in 0..total {
                do_batch(a_off, b_off, c_off);
                a_off += a_step;
                b_off += b_step;
                c_off += c_step;
            }
        } else {
            // Mixed path: each operand independently fused or strided
            let mut idx = vec![0usize; nb];
            for flat in 0..total {
                let a_off = batch_offset(flat, &idx, a_fused, a_batch);
                let b_off = batch_offset(flat, &idx, b_fused, b_batch);
                let c_off = batch_offset(flat, &idx, c_fused, c_batch);
                do_batch(a_off, b_off, c_off);

                // Advance shared multi-index (carry-based)
                if flat + 1 < total {
                    for ax in 0..nb {
                        let next = idx[ax] + 1;
                        if next < batch_dims[ax] {
                            idx[ax] = next;
                            break;
                        } else {
                            idx[ax] = 0;
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(feature = "gemm-blas")]
fn gemm_f64(
    alpha: f64,
    a: &[f64],
    b: &[f64],
    beta: f64,
    c: &mut [f64],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let m_i32 = i32::try_from(m).map_err(|_| Error::InvalidArgument("m too large".into()))?;
    let n_i32 = i32::try_from(n).map_err(|_| Error::InvalidArgument("n too large".into()))?;
    let k_i32 = i32::try_from(k).map_err(|_| Error::InvalidArgument("k too large".into()))?;
    unsafe {
        cblas_sys::cblas_dgemm(
            cblas_sys::CBLAS_LAYOUT::CblasColMajor,
            cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
            cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
            m_i32,
            n_i32,
            k_i32,
            alpha,
            a.as_ptr(),
            m_i32,
            b.as_ptr(),
            k_i32,
            beta,
            c.as_mut_ptr(),
            m_i32,
        );
    }
    Ok(())
}

#[cfg(feature = "gemm-blas")]
fn gemm_f32(
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let m_i32 = i32::try_from(m).map_err(|_| Error::InvalidArgument("m too large".into()))?;
    let n_i32 = i32::try_from(n).map_err(|_| Error::InvalidArgument("n too large".into()))?;
    let k_i32 = i32::try_from(k).map_err(|_| Error::InvalidArgument("k too large".into()))?;
    unsafe {
        cblas_sys::cblas_sgemm(
            cblas_sys::CBLAS_LAYOUT::CblasColMajor,
            cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
            cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
            m_i32,
            n_i32,
            k_i32,
            alpha,
            a.as_ptr(),
            m_i32,
            b.as_ptr(),
            k_i32,
            beta,
            c.as_mut_ptr(),
            m_i32,
        );
    }
    Ok(())
}

#[cfg(feature = "gemm-blas")]
fn gemm_c64(
    alpha: Complex64,
    a: &[Complex64],
    b: &[Complex64],
    beta: Complex64,
    c: &mut [Complex64],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let m_i32 = i32::try_from(m).map_err(|_| Error::InvalidArgument("m too large".into()))?;
    let n_i32 = i32::try_from(n).map_err(|_| Error::InvalidArgument("n too large".into()))?;
    let k_i32 = i32::try_from(k).map_err(|_| Error::InvalidArgument("k too large".into()))?;
    let alpha_ri = [alpha.re, alpha.im];
    let beta_ri = [beta.re, beta.im];
    unsafe {
        cblas_sys::cblas_zgemm(
            cblas_sys::CBLAS_LAYOUT::CblasColMajor,
            cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
            cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
            m_i32,
            n_i32,
            k_i32,
            &alpha_ri,
            a.as_ptr() as *const _,
            m_i32,
            b.as_ptr() as *const _,
            k_i32,
            &beta_ri,
            c.as_mut_ptr() as *mut _,
            m_i32,
        );
    }
    Ok(())
}

#[cfg(feature = "gemm-blas")]
fn gemm_c32(
    alpha: Complex32,
    a: &[Complex32],
    b: &[Complex32],
    beta: Complex32,
    c: &mut [Complex32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let m_i32 = i32::try_from(m).map_err(|_| Error::InvalidArgument("m too large".into()))?;
    let n_i32 = i32::try_from(n).map_err(|_| Error::InvalidArgument("n too large".into()))?;
    let k_i32 = i32::try_from(k).map_err(|_| Error::InvalidArgument("k too large".into()))?;
    let alpha_ri = [alpha.re, alpha.im];
    let beta_ri = [beta.re, beta.im];
    unsafe {
        cblas_sys::cblas_cgemm(
            cblas_sys::CBLAS_LAYOUT::CblasColMajor,
            cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
            cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
            m_i32,
            n_i32,
            k_i32,
            &alpha_ri,
            a.as_ptr() as *const _,
            m_i32,
            b.as_ptr() as *const _,
            k_i32,
            &beta_ri,
            c.as_mut_ptr() as *mut _,
            m_i32,
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_batched_gemm<T: Scalar + 'static>(
    _ctx: &mut CpuContext,
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
    batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let tid = TypeId::of::<T>();

    macro_rules! dispatch_gemm {
        ($ty:ty, strided) => {{
            let a = unsafe { &*(inputs[0] as *const StridedView<T> as *const StridedView<$ty>) };
            let b = unsafe { &*(inputs[1] as *const StridedView<T> as *const StridedView<$ty>) };
            let out =
                unsafe { &mut *(output as *mut StridedViewMut<T> as *mut StridedViewMut<$ty>) };
            let alpha = unsafe { *(&alpha as *const T as *const $ty) };
            let beta = unsafe { *(&beta as *const T as *const $ty) };
            return execute_batched_gemm_strided(alpha, &[a, b], beta, out, batch_dims, m, n, k);
        }};
        ($ty:ty, contiguous, $gemm_fn:expr) => {{
            let a = unsafe { &*(inputs[0] as *const StridedView<T> as *const StridedView<$ty>) };
            let b = unsafe { &*(inputs[1] as *const StridedView<T> as *const StridedView<$ty>) };
            let out =
                unsafe { &mut *(output as *mut StridedViewMut<T> as *mut StridedViewMut<$ty>) };
            let alpha = unsafe { *(&alpha as *const T as *const $ty) };
            let beta = unsafe { *(&beta as *const T as *const $ty) };
            return execute_batched_gemm_contiguous(
                _ctx,
                alpha,
                &[a, b],
                beta,
                out,
                batch_dims,
                m,
                n,
                k,
                $gemm_fn,
            );
        }};
    }

    if tid == TypeId::of::<f64>() {
        #[cfg(feature = "gemm-faer")]
        dispatch_gemm!(f64, strided);
        #[cfg(feature = "gemm-blas")]
        dispatch_gemm!(f64, contiguous, gemm_f64);
    }

    if tid == TypeId::of::<f32>() {
        #[cfg(feature = "gemm-faer")]
        dispatch_gemm!(f32, strided);
        #[cfg(feature = "gemm-blas")]
        dispatch_gemm!(f32, contiguous, gemm_f32);
    }

    if tid == TypeId::of::<Complex64>() {
        #[cfg(feature = "gemm-faer")]
        dispatch_gemm!(Complex64, strided);
        #[cfg(feature = "gemm-blas")]
        dispatch_gemm!(Complex64, contiguous, gemm_c64);
    }

    if tid == TypeId::of::<Complex32>() {
        #[cfg(feature = "gemm-faer")]
        dispatch_gemm!(Complex32, strided);
        #[cfg(feature = "gemm-blas")]
        dispatch_gemm!(Complex32, contiguous, gemm_c32);
    }

    Err(Error::InvalidArgument(format!(
        "BatchedGemm supports only f32, f64, Complex32, and Complex64 (got {})",
        std::any::type_name::<T>()
    )))
}

fn execute_reduce_sum<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    reduced_axes: &[usize],
) -> Result<()> {
    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let reduced_dims: Vec<usize> = reduced_axes.iter().map(|&ax| in_dims[ax]).collect();
    let reduced_total: usize = reduced_dims.iter().product();

    // Pre-allocate reusable buffers outside the hot loop
    let mut red_idx = vec![0usize; reduced_dims.len()];
    let mut in_idx = vec![0usize; in_dims.len()];

    for_each_index(&out_dims, |out_idx| {
        let mut sum = T::zero();
        for red_flat in 0..reduced_total {
            unflatten_index_into(red_flat, &reduced_dims, &mut red_idx);
            // Build full input index by interleaving free and reduced
            let mut out_pos = 0;
            let mut red_pos = 0;
            for (ax, in_slot) in in_idx.iter_mut().enumerate().take(in_dims.len()) {
                if red_pos < reduced_axes.len() && reduced_axes[red_pos] == ax {
                    *in_slot = red_idx[red_pos];
                    red_pos += 1;
                } else {
                    *in_slot = out_idx[out_pos];
                    out_pos += 1;
                }
            }
            sum = sum + input.get(&in_idx);
        }
        let old = if beta == T::zero() {
            T::zero()
        } else {
            beta * output.get(out_idx)
        };
        output.set(out_idx, alpha * sum + old);
    });
    Ok(())
}

fn extrema_prefers_candidate<T: Scalar>(candidate: T, current: T, op: ReduceOp) -> Result<bool> {
    fn compare_ordered<T: PartialOrd>(candidate: &T, current: &T, op: ReduceOp) -> Result<bool> {
        match candidate.partial_cmp(current) {
            Some(Ordering::Greater) => Ok(matches!(op, ReduceOp::Max)),
            Some(Ordering::Less) => Ok(matches!(op, ReduceOp::Min)),
            Some(Ordering::Equal) => Ok(false),
            None => Err(Error::InvalidArgument(format!(
                "Reduce {:?} encountered unordered values (for example NaN)",
                op
            ))),
        }
    }

    let candidate_any = &candidate as &dyn Any;
    let current_any = &current as &dyn Any;

    if let (Some(candidate), Some(current)) = (
        candidate_any.downcast_ref::<f32>(),
        current_any.downcast_ref::<f32>(),
    ) {
        return compare_ordered(candidate, current, op);
    }
    if let (Some(candidate), Some(current)) = (
        candidate_any.downcast_ref::<f64>(),
        current_any.downcast_ref::<f64>(),
    ) {
        return compare_ordered(candidate, current, op);
    }

    Err(Error::InvalidArgument(format!(
        "Reduce Max/Min supports only f32 and f64 on CpuBackend (got {})",
        std::any::type_name::<T>()
    )))
}

fn execute_reduce_extrema<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    reduced_axes: &[usize],
    op: ReduceOp,
) -> Result<()> {
    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let reduced_dims: Vec<usize> = reduced_axes.iter().map(|&ax| in_dims[ax]).collect();
    let reduced_total: usize = reduced_dims.iter().product();
    if reduced_total == 0 {
        return Err(Error::InvalidArgument(format!(
            "Reduce {:?} requires a non-empty reduction domain",
            op
        )));
    }

    let mut red_idx = vec![0usize; reduced_dims.len()];
    let mut in_idx = vec![0usize; in_dims.len()];
    let mut error = None;

    for_each_index(&out_dims, |out_idx| {
        if error.is_some() {
            return;
        }

        let mut best = None;
        for red_flat in 0..reduced_total {
            unflatten_index_into(red_flat, &reduced_dims, &mut red_idx);
            let mut out_pos = 0;
            let mut red_pos = 0;
            for (ax, in_slot) in in_idx.iter_mut().enumerate().take(in_dims.len()) {
                if red_pos < reduced_axes.len() && reduced_axes[red_pos] == ax {
                    *in_slot = red_idx[red_pos];
                    red_pos += 1;
                } else {
                    *in_slot = out_idx[out_pos];
                    out_pos += 1;
                }
            }

            let candidate = input.get(&in_idx);
            match best {
                None => best = Some(candidate),
                Some(current) => match extrema_prefers_candidate(candidate, current, op) {
                    Ok(true) => best = Some(candidate),
                    Ok(false) => best = Some(current),
                    Err(err) => {
                        error = Some(err);
                        return;
                    }
                },
            }
        }

        let best = match best {
            Some(best) => best,
            None => {
                error = Some(Error::InvalidArgument(format!(
                    "Reduce {:?} requires a non-empty reduction domain",
                    op
                )));
                return;
            }
        };
        let old = if beta == T::zero() {
            T::zero()
        } else {
            beta * output.get(out_idx)
        };
        output.set(out_idx, alpha * best + old);
    });

    if let Some(err) = error {
        return Err(err);
    }
    Ok(())
}

/// Unflatten a linear index into a pre-allocated buffer (column-major).
fn unflatten_index_into(mut flat: usize, dims: &[usize], out: &mut [usize]) {
    debug_assert!(
        flat < dims.iter().product::<usize>(),
        "flat index {flat} out of range for dims {dims:?}"
    );
    for d in 0..dims.len() {
        out[d] = flat % dims[d];
        flat /= dims[d];
    }
}

fn execute_trace<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    components: &[Vec<usize>],
    comp_dims: &[usize],
    free_axes: &[usize],
) -> Result<()> {
    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let n_comps = comp_dims.len();

    // Pre-allocate reusable buffers outside the hot loop
    let mut comp_idx = vec![0usize; n_comps];
    let mut in_idx = vec![0usize; in_dims.len()];

    for_each_index(&out_dims, |out_idx| {
        let mut sum = T::zero();
        // Odometer over component dimensions (Cartesian product)
        comp_idx.fill(0);
        loop {
            for (out_pos, &in_ax) in free_axes.iter().enumerate() {
                in_idx[in_ax] = out_idx[out_pos];
            }
            for (t, comp) in components.iter().enumerate() {
                for &ax in comp {
                    in_idx[ax] = comp_idx[t];
                }
            }
            sum = sum + input.get(&in_idx);

            // Increment odometer
            let mut carry = true;
            for t in 0..n_comps {
                if carry {
                    comp_idx[t] += 1;
                    if comp_idx[t] < comp_dims[t] {
                        carry = false;
                    } else {
                        comp_idx[t] = 0;
                    }
                }
            }
            if carry {
                break;
            }
        }
        let old = if beta == T::zero() {
            T::zero()
        } else {
            beta * output.get(out_idx)
        };
        output.set(out_idx, alpha * sum + old);
    });
    Ok(())
}

fn execute_anti_trace<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    components: &[Vec<usize>],
    comp_dims: &[usize],
    free_axes: &[usize],
) -> Result<()> {
    // AntiTrace: C = alpha * antitrace(A) + beta * C
    // First scale output by beta (since diagonal positions may be written multiple times)
    scale_output(output, beta);

    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let n_comps = comp_dims.len();

    // Pre-allocate reusable buffers outside the hot loop
    let mut comp_idx = vec![0usize; n_comps];
    let mut out_idx = vec![0usize; out_dims.len()];

    // For each input element, scatter to all diagonal positions in output
    // using Cartesian product over component dimensions.
    for_each_index(&in_dims, |in_idx| {
        let val = alpha * input.get(in_idx);
        // Odometer over component dimensions (Cartesian product)
        comp_idx.fill(0);
        loop {
            for (in_pos, &out_ax) in free_axes.iter().enumerate() {
                out_idx[out_ax] = in_idx[in_pos];
            }
            for (t, comp) in components.iter().enumerate() {
                for &ax in comp {
                    out_idx[ax] = comp_idx[t];
                }
            }
            let old = output.get(&out_idx);
            output.set(&out_idx, old + val);

            // Increment odometer
            let mut carry = true;
            for t in 0..n_comps {
                if carry {
                    comp_idx[t] += 1;
                    if comp_idx[t] < comp_dims[t] {
                        carry = false;
                    } else {
                        comp_idx[t] = 0;
                    }
                }
            }
            if carry {
                break;
            }
        }
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_anti_diag<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    components: &[Vec<usize>],
    comp_dims: &[usize],
    free_axes: &[usize],
    generative_comps: &[usize],
) -> Result<()> {
    // AntiDiag: write input values to diagonal positions in output.
    // Anchored components: at least one axis overlaps with free_axes, constraint propagated.
    // Generative components: no axis overlaps with free_axes, need own loop.
    scale_output(output, beta);

    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();

    let gen_dims: Vec<usize> = generative_comps.iter().map(|&c| comp_dims[c]).collect();

    // Pre-allocate reusable buffers outside the hot loop
    let mut gen_idx = vec![0usize; generative_comps.len()];
    let mut out_idx = vec![0usize; out_dims.len()];

    for_each_index(&in_dims, |in_idx| {
        let val = alpha * input.get(in_idx);
        // Odometer over generative component dimensions
        gen_idx.fill(0);
        loop {
            // Set free axes from input
            for (in_pos, &out_ax) in free_axes.iter().enumerate() {
                out_idx[out_ax] = in_idx[in_pos];
            }
            // Set component axes
            for (t, comp) in components.iter().enumerate() {
                if let Some(gi) = generative_comps.iter().position(|&c| c == t) {
                    // Generative: use gen_idx
                    for &ax in comp {
                        out_idx[ax] = gen_idx[gi];
                    }
                } else {
                    // Anchored: propagate from the first axis (already set by free_axes)
                    let anchor_val = out_idx[comp[0]];
                    for &ax in &comp[1..] {
                        out_idx[ax] = anchor_val;
                    }
                }
            }
            let old = output.get(&out_idx);
            output.set(&out_idx, old + val);

            if gen_dims.is_empty() {
                break;
            }
            // Increment odometer for generative components
            let mut carry = true;
            for g in 0..gen_dims.len() {
                if carry {
                    gen_idx[g] += 1;
                    if gen_idx[g] < gen_dims[g] {
                        carry = false;
                    } else {
                        gen_idx[g] = 0;
                    }
                }
            }
            if carry {
                break;
            }
        }
    });
    Ok(())
}

fn execute_elementwise_mul<T: Scalar>(
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
) -> Result<()> {
    let a = inputs[0];
    let b = inputs[1];

    if beta == T::zero() {
        // Hot path (>99% of einsum calls): C = alpha * A * B
        // Delegate to strided-kernel which handles dimension fusing,
        // cache-optimized block traversal, and SIMD dispatch.
        let alpha_val = alpha;
        strided_kernel::zip_map2_into(output, a, b, move |a_val, b_val| {
            alpha_val * (a_val * b_val)
        })
        .map_err(|e| Error::DeviceError(e.to_string()))?;
    } else {
        // Rare path: C = alpha * A * B + beta * C
        // Cannot use zip_map2_into (overwrites dest, losing beta*C).
        // Cannot alias dest with a zip_map3 source.
        // Use index-based loop for correctness.
        let dims = output.dims().to_vec();
        for_each_index(&dims, |idx| {
            let val = alpha * (a.get(idx) * b.get(idx));
            output.set(idx, val + beta * output.get(idx));
        });
    }
    Ok(())
}

/// Apply a unary function element-wise: C = alpha * f(A) + beta * C.
fn execute_unary_map<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    f: impl Fn(T) -> T,
) -> Result<()> {
    let dims = output.dims().to_vec();
    for_each_index(&dims, |idx| {
        let val = alpha * f(input.get(idx));
        if beta == T::zero() {
            output.set(idx, val);
        } else {
            output.set(idx, val + beta * output.get(idx));
        }
    });
    Ok(())
}

/// Execute element-wise unary operation with type-based dispatch.
///
/// Since `Scalar` does not provide `Neg`, `Div`, or floating-point ops,
/// we dispatch to concrete type implementations (f32, f64, Complex32, Complex64)
/// at runtime using `TypeId`. This keeps the `TensorPrims` trait generic while
/// supporting all standard unary operations on the CPU backend.
fn execute_elementwise_unary<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    op: &UnaryOp,
) -> Result<()> {
    match op {
        UnaryOp::Conj => {
            let tid = TypeId::of::<T>();
            if tid == TypeId::of::<f64>() || tid == TypeId::of::<f32>() {
                // Real types: conjugation is identity
                execute_make_contiguous(alpha, input, beta, output)
            } else if tid == TypeId::of::<Complex64>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const Complex64) };
                    let r = x.conj();
                    unsafe { *(&r as *const Complex64 as *const T) }
                })
            } else if tid == TypeId::of::<Complex32>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const Complex32) };
                    let r = x.conj();
                    unsafe { *(&r as *const Complex32 as *const T) }
                })
            } else {
                Err(Error::InvalidArgument(
                    "Conj not supported for this scalar type".into(),
                ))
            }
        }
        UnaryOp::Negate => {
            let tid = TypeId::of::<T>();
            if tid == TypeId::of::<f64>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    // SAFETY: T is f64; transmute is safe because we checked TypeId
                    let x = unsafe { *(&v as *const T as *const f64) };
                    let r = -x;
                    unsafe { *(&r as *const f64 as *const T) }
                })
            } else if tid == TypeId::of::<f32>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const f32) };
                    let r = -x;
                    unsafe { *(&r as *const f32 as *const T) }
                })
            } else if tid == TypeId::of::<Complex64>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const Complex64) };
                    let r = -x;
                    unsafe { *(&r as *const Complex64 as *const T) }
                })
            } else if tid == TypeId::of::<Complex32>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const Complex32) };
                    let r = -x;
                    unsafe { *(&r as *const Complex32 as *const T) }
                })
            } else {
                Err(Error::InvalidArgument(
                    "Negate not supported for this scalar type".into(),
                ))
            }
        }
        UnaryOp::Reciprocal => {
            let tid = TypeId::of::<T>();
            if tid == TypeId::of::<f64>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const f64) };
                    let r = 1.0_f64 / x;
                    unsafe { *(&r as *const f64 as *const T) }
                })
            } else if tid == TypeId::of::<f32>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const f32) };
                    let r = 1.0_f32 / x;
                    unsafe { *(&r as *const f32 as *const T) }
                })
            } else if tid == TypeId::of::<Complex64>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const Complex64) };
                    let r = Complex64::new(1.0, 0.0) / x;
                    unsafe { *(&r as *const Complex64 as *const T) }
                })
            } else if tid == TypeId::of::<Complex32>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const Complex32) };
                    let r = Complex32::new(1.0, 0.0) / x;
                    unsafe { *(&r as *const Complex32 as *const T) }
                })
            } else {
                Err(Error::InvalidArgument(
                    "Reciprocal not supported for this scalar type".into(),
                ))
            }
        }
        UnaryOp::Abs => {
            let tid = TypeId::of::<T>();
            if tid == TypeId::of::<f64>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const f64) };
                    let r = x.abs();
                    unsafe { *(&r as *const f64 as *const T) }
                })
            } else if tid == TypeId::of::<f32>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const f32) };
                    let r = x.abs();
                    unsafe { *(&r as *const f32 as *const T) }
                })
            } else if tid == TypeId::of::<Complex64>() {
                // For complex, abs returns the modulus as a real number.
                // But since T is Complex64, we return it as Complex64 with zero imaginary part.
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const Complex64) };
                    let r = Complex64::new(x.norm(), 0.0);
                    unsafe { *(&r as *const Complex64 as *const T) }
                })
            } else if tid == TypeId::of::<Complex32>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const Complex32) };
                    let r = Complex32::new(x.norm(), 0.0);
                    unsafe { *(&r as *const Complex32 as *const T) }
                })
            } else {
                Err(Error::InvalidArgument(
                    "Abs not supported for this scalar type".into(),
                ))
            }
        }
        UnaryOp::Sqrt => {
            let tid = TypeId::of::<T>();
            if tid == TypeId::of::<f64>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const f64) };
                    let r = x.sqrt();
                    unsafe { *(&r as *const f64 as *const T) }
                })
            } else if tid == TypeId::of::<f32>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const f32) };
                    let r = x.sqrt();
                    unsafe { *(&r as *const f32 as *const T) }
                })
            } else if tid == TypeId::of::<Complex64>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const Complex64) };
                    let r = x.sqrt();
                    unsafe { *(&r as *const Complex64 as *const T) }
                })
            } else if tid == TypeId::of::<Complex32>() {
                execute_unary_map(alpha, input, beta, output, |v| {
                    let x = unsafe { *(&v as *const T as *const Complex32) };
                    let r = x.sqrt();
                    unsafe { *(&r as *const Complex32 as *const T) }
                })
            } else {
                Err(Error::InvalidArgument(
                    "Sqrt not supported for this scalar type".into(),
                ))
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_contract<T: Scalar>(
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
    modes_a: &[u32],
    modes_b: &[u32],
    modes_c: &[u32],
    cached_gemm_spec: Option<&ContractGemmSpec>,
) -> Result<()> {
    let a = inputs[0];
    let b = inputs[1];

    if let Some(done) = try_execute_contract_gemm(
        alpha,
        inputs,
        beta,
        output,
        modes_a,
        modes_b,
        modes_c,
        cached_gemm_spec,
    )? {
        return Ok(done);
    }

    // Determine reduction modes: any mode absent from C is summed over, whether it
    // appears in both inputs or only in one operand.
    let mut contracted_modes = Vec::new();
    for &mode in modes_a.iter().chain(modes_b.iter()) {
        if !modes_c.contains(&mode) && !contracted_modes.contains(&mode) {
            contracted_modes.push(mode);
        }
    }
    let contracted_dims: Vec<usize> = contracted_modes
        .iter()
        .map(|&m| {
            if let Some(a_pos) = modes_a.iter().position(|&mm| mm == m) {
                a.dims()[a_pos]
            } else {
                let b_pos = modes_b
                    .iter()
                    .position(|&mm| mm == m)
                    .expect("contracted mode must appear in at least one operand");
                b.dims()[b_pos]
            }
        })
        .collect();
    let contracted_total: usize = if contracted_dims.is_empty() {
        1
    } else {
        contracted_dims.iter().product()
    };

    // Precompute where each A/B axis value comes from in the inner loop.
    // 0 => from output index (c_idx), 1 => from contracted index (k_idx)
    let a_axis_map: Vec<(u8, usize)> = modes_a
        .iter()
        .map(|&mode| {
            if let Some(c_pos) = modes_c.iter().position(|&m| m == mode) {
                (0, c_pos)
            } else if let Some(k_pos) = contracted_modes.iter().position(|&m| m == mode) {
                (1, k_pos)
            } else {
                unreachable!("every A-only mode absent from C must be reduced")
            }
        })
        .collect();
    let b_axis_map: Vec<(u8, usize)> = modes_b
        .iter()
        .map(|&mode| {
            if let Some(c_pos) = modes_c.iter().position(|&m| m == mode) {
                (0, c_pos)
            } else if let Some(k_pos) = contracted_modes.iter().position(|&m| m == mode) {
                (1, k_pos)
            } else {
                unreachable!("every B-only mode absent from C must be reduced")
            }
        })
        .collect();

    // Unflatten helper that writes into a reusable buffer (no allocation).
    fn unflatten_into(mut flat: usize, dims: &[usize], out: &mut [usize]) {
        debug_assert_eq!(dims.len(), out.len());
        debug_assert!(
            dims.iter().all(|&d| d > 0),
            "unflatten_into: zero dimension in dims {dims:?}"
        );
        debug_assert!(
            flat < dims.iter().product::<usize>(),
            "flat index {flat} out of range for dims {dims:?}"
        );
        for (i, &d) in dims.iter().enumerate() {
            out[i] = flat % d;
            flat /= d;
        }
    }

    let out_dims = output.dims().to_vec();
    let mut a_idx = vec![0usize; modes_a.len()];
    let mut b_idx = vec![0usize; modes_b.len()];
    let mut k_idx = vec![0usize; contracted_dims.len()];

    for_each_index(&out_dims, |c_idx| {
        let mut sum = T::zero();
        for k_flat in 0..contracted_total {
            if !contracted_dims.is_empty() {
                unflatten_into(k_flat, &contracted_dims, &mut k_idx);
            }
            for (ax, &(src, pos)) in a_axis_map.iter().enumerate() {
                a_idx[ax] = match src {
                    0 => c_idx[pos],
                    1 => k_idx[pos],
                    _ => unreachable!("contract axis source must be output or reduction index"),
                };
            }
            for (ax, &(src, pos)) in b_axis_map.iter().enumerate() {
                b_idx[ax] = match src {
                    0 => c_idx[pos],
                    1 => k_idx[pos],
                    _ => unreachable!("contract axis source must be output or reduction index"),
                };
            }
            sum = sum + a.get(&a_idx) * b.get(&b_idx);
        }
        let old = if beta == T::zero() {
            T::zero()
        } else {
            beta * output.get(c_idx)
        };
        output.set(c_idx, alpha * sum + old);
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn try_execute_contract_gemm<T: Scalar + 'static>(
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
    modes_a: &[u32],
    modes_b: &[u32],
    modes_c: &[u32],
    cached_spec: Option<&ContractGemmSpec>,
) -> Result<Option<()>> {
    #[derive(Clone)]
    struct ModeSpec {
        batch_modes: Vec<u32>,
        m_modes: Vec<u32>,
        n_modes: Vec<u32>,
        k_modes: Vec<u32>,
    }

    fn build_mode_spec(modes_a: &[u32], modes_b: &[u32], modes_c: &[u32]) -> Option<ModeSpec> {
        let batch_modes: Vec<u32> = modes_c
            .iter()
            .copied()
            .filter(|m| modes_a.contains(m) && modes_b.contains(m))
            .collect();
        let m_modes: Vec<u32> = modes_c
            .iter()
            .copied()
            .filter(|m| modes_a.contains(m) && !modes_b.contains(m))
            .collect();
        let n_modes: Vec<u32> = modes_c
            .iter()
            .copied()
            .filter(|m| modes_b.contains(m) && !modes_a.contains(m))
            .collect();
        let k_modes: Vec<u32> = modes_a
            .iter()
            .copied()
            .filter(|m| modes_b.contains(m) && !modes_c.contains(m))
            .collect();

        let expected_a = batch_modes.len() + m_modes.len() + k_modes.len();
        let expected_b = batch_modes.len() + k_modes.len() + n_modes.len();
        if expected_a != modes_a.len() || expected_b != modes_b.len() {
            return None;
        }
        if batch_modes.len() + m_modes.len() + n_modes.len() != modes_c.len() {
            return None;
        }
        Some(ModeSpec {
            batch_modes,
            m_modes,
            n_modes,
            k_modes,
        })
    }

    fn perm_for(target: &[u32], source: &[u32]) -> Option<Vec<usize>> {
        target
            .iter()
            .map(|m| source.iter().position(|x| x == m))
            .collect()
    }

    fn reordered_dims_strides(
        modes_src: &[u32],
        dims_src: &[usize],
        strides_src: &[isize],
        target: &[u32],
    ) -> Option<(Vec<usize>, Vec<isize>)> {
        let perm = perm_for(target, modes_src)?;
        let dims = perm.iter().map(|&p| dims_src[p]).collect();
        let strides = perm.iter().map(|&p| strides_src[p]).collect();
        Some((dims, strides))
    }

    /// Pre-computed fused GEMM geometry for a contract operation.
    struct GemmLayout {
        batch_total: usize,
        m: usize,
        n: usize,
        k: usize,
        /// A row stride (m dimension)
        a_ms: isize,
        /// A col stride (k dimension)
        a_ks: isize,
        /// B row stride (k dimension)
        b_ks: isize,
        /// B col stride (n dimension)
        b_ns: isize,
        /// C row stride (m dimension)
        c_ms: isize,
        /// C col stride (n dimension)
        c_ns: isize,
        /// A batch stride
        a_bs: isize,
        /// B batch stride
        b_bs: isize,
        /// C batch stride
        c_bs: isize,
    }

    /// Compute the fused GEMM layout from mode specs and tensor shapes/strides.
    /// Returns `None` if the modes cannot be fused into a valid GEMM layout.
    #[allow(clippy::too_many_arguments)]
    fn compute_layout(
        a_dims_src: &[usize],
        a_strides_src: &[isize],
        b_dims_src: &[usize],
        b_strides_src: &[isize],
        c_dims_src: &[usize],
        c_strides_src: &[isize],
        modes_a: &[u32],
        modes_b: &[u32],
        modes_c: &[u32],
        spec: &ModeSpec,
        cached: Option<&ContractGemmSpec>,
    ) -> Option<GemmLayout> {
        let owned_a;
        let owned_b;
        let owned_c;
        let (target_a, target_b, target_c) = if let Some(cs) = cached {
            (
                cs.a_target.as_slice(),
                cs.b_target.as_slice(),
                cs.c_target.as_slice(),
            )
        } else {
            owned_a = spec
                .batch_modes
                .iter()
                .chain(spec.m_modes.iter())
                .chain(spec.k_modes.iter())
                .copied()
                .collect::<Vec<u32>>();
            owned_b = spec
                .batch_modes
                .iter()
                .chain(spec.k_modes.iter())
                .chain(spec.n_modes.iter())
                .copied()
                .collect::<Vec<u32>>();
            owned_c = spec
                .batch_modes
                .iter()
                .chain(spec.m_modes.iter())
                .chain(spec.n_modes.iter())
                .copied()
                .collect::<Vec<u32>>();
            (owned_a.as_slice(), owned_b.as_slice(), owned_c.as_slice())
        };

        let (a_dims, a_strides) =
            reordered_dims_strides(modes_a, a_dims_src, a_strides_src, target_a)?;
        let (b_dims, b_strides) =
            reordered_dims_strides(modes_b, b_dims_src, b_strides_src, target_b)?;
        let (c_dims, c_strides) =
            reordered_dims_strides(modes_c, c_dims_src, c_strides_src, target_c)?;

        let nb = spec.batch_modes.len();
        let nm = spec.m_modes.len();
        let nk = spec.k_modes.len();
        let nn = spec.n_modes.len();

        let (batch_total, a_bs, b_bs, c_bs) = if nb == 0 {
            (1usize, 0isize, 0isize, 0isize)
        } else {
            let (ta, sa) = try_fuse_group(&a_dims[..nb], &a_strides[..nb])?;
            let (tb, sb) = try_fuse_group(&b_dims[..nb], &b_strides[..nb])?;
            let (tc, sc) = try_fuse_group(&c_dims[..nb], &c_strides[..nb])?;
            if ta != tb || ta != tc {
                return None;
            }
            (ta, sa, sb, sc)
        };

        let (m_raw, a_ms) = if nm == 0 {
            (1usize, 0isize)
        } else {
            try_fuse_group(&a_dims[nb..nb + nm], &a_strides[nb..nb + nm])?
        };
        let (m_chk, c_ms) = if nm == 0 {
            (1usize, 0isize)
        } else {
            try_fuse_group(&c_dims[nb..nb + nm], &c_strides[nb..nb + nm])?
        };
        if m_raw != m_chk {
            return None;
        }

        let (k_raw, a_ks) = if nk == 0 {
            (1usize, 0isize)
        } else {
            try_fuse_group(
                &a_dims[nb + nm..nb + nm + nk],
                &a_strides[nb + nm..nb + nm + nk],
            )?
        };
        let (k_chk, b_ks) = if nk == 0 {
            (1usize, 0isize)
        } else {
            try_fuse_group(&b_dims[nb..nb + nk], &b_strides[nb..nb + nk])?
        };
        if k_raw != k_chk {
            return None;
        }

        let (n_raw, b_ns) = if nn == 0 {
            (1usize, 0isize)
        } else {
            try_fuse_group(
                &b_dims[nb + nk..nb + nk + nn],
                &b_strides[nb + nk..nb + nk + nn],
            )?
        };
        let (n_chk, c_ns) = if nn == 0 {
            (1usize, 0isize)
        } else {
            try_fuse_group(
                &c_dims[nb + nm..nb + nm + nn],
                &c_strides[nb + nm..nb + nm + nn],
            )?
        };
        if n_raw != n_chk {
            return None;
        }

        Some(GemmLayout {
            batch_total,
            m: m_raw.max(1),
            n: n_raw.max(1),
            k: k_raw.max(1),
            a_ms,
            a_ks,
            b_ks,
            b_ns,
            c_ms,
            c_ns,
            a_bs,
            b_bs,
            c_bs,
        })
    }

    /// Strided GEMM via faer — zero allocation, zero copy.
    #[cfg(feature = "gemm-faer")]
    fn run_strided<U: FaerGemm>(
        alpha: U,
        a: &StridedView<U>,
        b: &StridedView<U>,
        beta: U,
        c: &mut StridedViewMut<U>,
        layout: &GemmLayout,
    ) -> Result<()> {
        let a_ptr = a.ptr();
        let b_ptr = b.ptr();
        let c_ptr = c.as_mut_ptr();
        let mut a_off = 0isize;
        let mut b_off = 0isize;
        let mut c_off = 0isize;
        for _ in 0..layout.batch_total {
            unsafe {
                U::strided_gemm(
                    alpha,
                    a_ptr.offset(a_off),
                    layout.m,
                    layout.k,
                    layout.a_ms,
                    layout.a_ks,
                    b_ptr.offset(b_off),
                    layout.n,
                    layout.b_ks,
                    layout.b_ns,
                    beta,
                    c_ptr.offset(c_off),
                    layout.c_ms,
                    layout.c_ns,
                );
            }
            a_off += layout.a_bs;
            b_off += layout.b_bs;
            c_off += layout.c_bs;
        }
        Ok(())
    }

    /// Dense-packing GEMM for the OpenBLAS backend.
    #[cfg(feature = "gemm-blas")]
    fn run_dense<U: Scalar>(
        alpha: U,
        a: &StridedView<U>,
        b: &StridedView<U>,
        beta: U,
        c: &mut StridedViewMut<U>,
        layout: &GemmLayout,
        gemm_fn: fn(U, &[U], &[U], U, &mut [U], usize, usize, usize) -> Result<()>,
    ) -> Result<()> {
        let GemmLayout {
            batch_total,
            m,
            n,
            k,
            a_ms,
            a_ks,
            b_ks,
            b_ns,
            c_ms,
            c_ns,
            a_bs,
            b_bs,
            c_bs,
        } = *layout;

        let mut a_mat = vec![U::zero(); m * k];
        let mut b_mat = vec![U::zero(); k * n];
        let mut c_mat = vec![U::zero(); m * n];

        let a_ptr = a.ptr();
        let b_ptr = b.ptr();
        let c_ptr = c.as_mut_ptr();
        let mut a_off = 0isize;
        let mut b_off = 0isize;
        let mut c_off = 0isize;

        for _ in 0..batch_total {
            for kk in 0..k {
                for i in 0..m {
                    let off = a_off + i as isize * a_ms + kk as isize * a_ks;
                    a_mat[i + kk * m] = unsafe { *a_ptr.offset(off) };
                }
            }
            for j in 0..n {
                for kk in 0..k {
                    let off = b_off + kk as isize * b_ks + j as isize * b_ns;
                    b_mat[kk + j * k] = unsafe { *b_ptr.offset(off) };
                }
            }
            if beta == U::zero() {
                c_mat.iter_mut().for_each(|v| *v = U::zero());
            } else {
                for j in 0..n {
                    for i in 0..m {
                        let off = c_off + i as isize * c_ms + j as isize * c_ns;
                        c_mat[i + j * m] = unsafe { *c_ptr.offset(off) };
                    }
                }
            }

            gemm_fn(alpha, &a_mat, &b_mat, beta, &mut c_mat, m, n, k)?;

            for j in 0..n {
                for i in 0..m {
                    let off = c_off + i as isize * c_ms + j as isize * c_ns;
                    unsafe {
                        *c_ptr.offset(off) = c_mat[i + j * m];
                    }
                }
            }

            a_off += a_bs;
            b_off += b_bs;
            c_off += c_bs;
        }
        Ok(())
    }

    // === Build spec ===
    let spec = if let Some(cached) = cached_spec {
        ModeSpec {
            batch_modes: cached.batch_modes.clone(),
            m_modes: cached.m_modes.clone(),
            n_modes: cached.n_modes.clone(),
            k_modes: cached.k_modes.clone(),
        }
    } else {
        match build_mode_spec(modes_a, modes_b, modes_c) {
            Some(s) => s,
            None => return Ok(None),
        }
    };

    // === Compute layout (type-independent) ===
    let layout = match compute_layout(
        inputs[0].dims(),
        inputs[0].strides(),
        inputs[1].dims(),
        inputs[1].strides(),
        output.dims(),
        output.strides(),
        modes_a,
        modes_b,
        modes_c,
        &spec,
        cached_spec,
    ) {
        Some(l) => l,
        None => return Ok(None),
    };

    // === Dispatch based on concrete type ===
    let tid = TypeId::of::<T>();

    macro_rules! dispatch_contract {
        ($ty:ty, strided) => {{
            let a = unsafe { &*(inputs[0] as *const StridedView<T> as *const StridedView<$ty>) };
            let b = unsafe { &*(inputs[1] as *const StridedView<T> as *const StridedView<$ty>) };
            let c = unsafe { &mut *(output as *mut StridedViewMut<T> as *mut StridedViewMut<$ty>) };
            let alpha = unsafe { *(&alpha as *const T as *const $ty) };
            let beta = unsafe { *(&beta as *const T as *const $ty) };
            run_strided(alpha, a, b, beta, c, &layout)?;
            return Ok(Some(()));
        }};
        ($ty:ty, dense, $gemm_fn:expr) => {{
            let a = unsafe { &*(inputs[0] as *const StridedView<T> as *const StridedView<$ty>) };
            let b = unsafe { &*(inputs[1] as *const StridedView<T> as *const StridedView<$ty>) };
            let c = unsafe { &mut *(output as *mut StridedViewMut<T> as *mut StridedViewMut<$ty>) };
            let alpha = unsafe { *(&alpha as *const T as *const $ty) };
            let beta = unsafe { *(&beta as *const T as *const $ty) };
            run_dense(alpha, a, b, beta, c, &layout, $gemm_fn)?;
            return Ok(Some(()));
        }};
    }

    if tid == TypeId::of::<f64>() {
        #[cfg(feature = "gemm-faer")]
        dispatch_contract!(f64, strided);
        #[cfg(feature = "gemm-blas")]
        dispatch_contract!(f64, dense, gemm_f64);
    }
    if tid == TypeId::of::<f32>() {
        #[cfg(feature = "gemm-faer")]
        dispatch_contract!(f32, strided);
        #[cfg(feature = "gemm-blas")]
        dispatch_contract!(f32, dense, gemm_f32);
    }
    if tid == TypeId::of::<Complex64>() {
        #[cfg(feature = "gemm-faer")]
        dispatch_contract!(Complex64, strided);
        #[cfg(feature = "gemm-blas")]
        dispatch_contract!(Complex64, dense, gemm_c64);
    }
    if tid == TypeId::of::<Complex32>() {
        #[cfg(feature = "gemm-faer")]
        dispatch_contract!(Complex32, strided);
        #[cfg(feature = "gemm-blas")]
        dispatch_contract!(Complex32, dense, gemm_c32);
    }
    Ok(None)
}

// ===========================================================================
// CPU backend TensorPrims implementation
// ===========================================================================

impl<S: Scalar> TensorPrims<Standard<S>> for CpuBackend {
    type Plan = CpuPlan<S>;
    type Context = CpuContext;

    fn plan(
        ctx: &mut CpuContext,
        desc: &PrimDescriptor,
        shapes: &[&[usize]],
    ) -> Result<CpuPlan<S>> {
        // Check cache first
        if let Some(cached) = ctx.plan_cache.get::<CpuPlan<S>>(desc, shapes) {
            return Ok(cached);
        }

        let plan = Self::build_plan::<S>(desc, shapes)?;

        // Store in cache for future reuse
        ctx.plan_cache.insert(desc, shapes, plan.clone());

        Ok(plan)
    }

    fn execute(
        ctx: &mut CpuContext,
        plan: &CpuPlan<S>,
        alpha: S,
        inputs: &[&Tensor<S>],
        beta: S,
        output: &mut Tensor<S>,
    ) -> Result<()> {
        // Convert Tensor inputs to StridedView for internal dispatch
        let views: Vec<StridedView<S>> = inputs
            .iter()
            .map(|t| tensor_to_view(t))
            .collect::<Result<Vec<_>>>()?;
        let view_refs: Vec<&StridedView<S>> = views.iter().collect();
        let mut out_view = tensor_to_view_mut(output)?;

        match plan {
            CpuPlan::Permute { perm, .. } => {
                validate_execute_inputs(inputs, 1, "Permute")?;
                execute_permute(alpha, view_refs[0], beta, &mut out_view, perm)
            }

            CpuPlan::MakeContiguous { .. } => {
                validate_execute_inputs(inputs, 1, "MakeContiguous")?;
                execute_make_contiguous(alpha, view_refs[0], beta, &mut out_view)
            }

            CpuPlan::BatchedGemm {
                batch_dims,
                m,
                n,
                k,
                ..
            } => {
                validate_execute_inputs(inputs, 2, "BatchedGemm")?;
                execute_batched_gemm(
                    ctx,
                    alpha,
                    &view_refs,
                    beta,
                    &mut out_view,
                    batch_dims,
                    *m,
                    *n,
                    *k,
                )
            }

            CpuPlan::Reduce {
                reduced_axes, op, ..
            } => {
                validate_execute_inputs(inputs, 1, "Reduce")?;
                match op {
                    ReduceOp::Sum => {
                        execute_reduce_sum(alpha, view_refs[0], beta, &mut out_view, reduced_axes)
                    }
                    ReduceOp::Max | ReduceOp::Min => execute_reduce_extrema(
                        alpha,
                        view_refs[0],
                        beta,
                        &mut out_view,
                        reduced_axes,
                        *op,
                    ),
                }
            }

            CpuPlan::Trace {
                components,
                comp_dims,
                free_axes,
                ..
            } => {
                validate_execute_inputs(inputs, 1, "Trace")?;
                execute_trace(
                    alpha,
                    view_refs[0],
                    beta,
                    &mut out_view,
                    components,
                    comp_dims,
                    free_axes,
                )
            }

            CpuPlan::AntiTrace {
                free_axes,
                components,
                comp_dims,
                ..
            } => {
                validate_execute_inputs(inputs, 1, "AntiTrace")?;
                execute_anti_trace(
                    alpha,
                    view_refs[0],
                    beta,
                    &mut out_view,
                    components,
                    comp_dims,
                    free_axes,
                )
            }

            CpuPlan::AntiDiag {
                free_axes,
                components,
                comp_dims,
                generative_comps,
                ..
            } => {
                validate_execute_inputs(inputs, 1, "AntiDiag")?;
                execute_anti_diag(
                    alpha,
                    view_refs[0],
                    beta,
                    &mut out_view,
                    components,
                    comp_dims,
                    free_axes,
                    generative_comps,
                )
            }

            CpuPlan::ElementwiseUnary { op, .. } => {
                validate_execute_inputs(inputs, 1, "ElementwiseUnary")?;
                execute_elementwise_unary(alpha, view_refs[0], beta, &mut out_view, op)
            }

            CpuPlan::ElementwiseMul { .. } => {
                validate_execute_inputs(inputs, 2, "ElementwiseMul")?;
                execute_elementwise_mul(alpha, &view_refs, beta, &mut out_view)
            }

            CpuPlan::Contract {
                modes_a,
                modes_b,
                modes_c,
                gemm_spec,
                ..
            } => {
                validate_execute_inputs(inputs, 2, "Contract")?;
                execute_contract(
                    alpha,
                    &view_refs,
                    beta,
                    &mut out_view,
                    modes_a,
                    modes_b,
                    modes_c,
                    gemm_spec.as_ref(),
                )
            }
        }
    }

    fn has_extension_for(_ext: Extension) -> bool {
        matches!(_ext, Extension::Contract | Extension::ElementwiseMul)
    }
}

/// Scale all elements of the output by `beta`, or zero them if `beta == 0`.
fn scale_output<T: Scalar>(output: &mut StridedViewMut<T>, beta: T) {
    let dims = output.dims().to_vec();
    if beta == T::zero() {
        for_each_index(&dims, |idx| {
            output.set(idx, T::zero());
        });
    } else if beta != T::one() {
        for_each_index(&dims, |idx| {
            let old = output.get(idx);
            output.set(idx, beta * old);
        });
    }
    // If beta == 1, output is unchanged (identity scaling).
}

#[cfg(all(test, feature = "gemm-blas"))]
mod scratch_pool_tests {
    use super::*;

    #[test]
    fn take_put_roundtrip_f64() {
        let mut pool = ScratchPool::default();
        let mut buf = pool.take::<f64>(100);
        assert_eq!(buf.len(), 100);
        for i in 0..100 {
            buf[i] = i as f64;
        }
        assert_eq!(buf[42], 42.0);
        pool.put(buf);

        // Second take should reuse the same allocation.
        let buf2 = pool.take::<f64>(100);
        assert_eq!(buf2.len(), 100);
        pool.put(buf2);
        assert!(pool.pool.values().map(|v| v.len()).sum::<usize>() == 1);
    }

    #[test]
    fn cross_type_reuse() {
        let mut pool = ScratchPool::default();
        // Allocate 1000 f64s = 8000 bytes.
        let buf = pool.take::<f64>(1000);
        let cap = buf.cap_bytes;
        assert!(cap >= 8000);
        pool.put(buf);

        // Take 2000 f32s = 8000 bytes — should reuse the same buffer.
        let buf2 = pool.take::<f32>(2000);
        assert_eq!(buf2.cap_bytes, cap);
        assert_eq!(buf2.len(), 2000);
        pool.put(buf2);
    }

    #[test]
    fn larger_buffer_reused_for_smaller_request() {
        let mut pool = ScratchPool::default();
        let buf = pool.take::<f64>(1000);
        pool.put(buf);
        // Request fewer elements — pool returns the existing (larger) buffer.
        let buf2 = pool.take::<f64>(500);
        assert!(buf2.cap_bytes >= 8000);
        assert_eq!(buf2.len(), 500);
        pool.put(buf2);
    }

    #[test]
    fn zero_length_take() {
        let mut pool = ScratchPool::default();
        let buf = pool.take::<f64>(0);
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.cap_bytes, 0);
        pool.put(buf);
        // Pool should not store zero-capacity buffers.
        assert!(pool.pool.is_empty());
    }

    #[test]
    fn drop_without_put_does_not_leak() {
        let mut pool = ScratchPool::default();
        let buf = pool.take::<f64>(1024);
        // Drop without returning to pool — ScratchBuf::drop deallocates.
        drop(buf);
        assert!(pool.pool.is_empty());
    }
}
