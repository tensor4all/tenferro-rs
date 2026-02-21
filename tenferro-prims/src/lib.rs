//! Tensor primitive operations for the tenferro workspace.
//!
//! This crate defines the [`TensorPrims<A>`] trait, a backend-agnostic interface
//! parameterized by algebra `A`. The API follows the cuTENSOR plan-based execution
//! pattern:
//!
//! 1. Create a [`PrimDescriptor`] specifying the operation and index modes
//! 2. Build a plan via [`TensorPrims::plan`] (pre-computes kernel selection)
//! 3. Execute the plan via [`TensorPrims::execute`]
//!
//! # Operation categories
//!
//! **Core operations** (every backend must implement):
//! - [`BatchedGemm`](PrimDescriptor::BatchedGemm): Batched matrix multiplication
//! - [`Reduce`](PrimDescriptor::Reduce): Sum/max/min reduction over modes
//! - [`Trace`](PrimDescriptor::Trace): Trace (contraction of paired diagonal modes)
//! - [`Permute`](PrimDescriptor::Permute): Mode reordering
//! - [`AntiTrace`](PrimDescriptor::AntiTrace): Scatter-add to diagonal (AD backward of trace)
//! - [`AntiDiag`](PrimDescriptor::AntiDiag): Write to diagonal positions (AD backward of diag)
//! - [`ElementwiseUnary`](PrimDescriptor::ElementwiseUnary): Point-wise unary transform (negate, reciprocal, abs, sqrt)
//!
//! **Extended operations** (dynamically queried via [`TensorPrims::has_extension_for`]):
//! - [`Contract`](PrimDescriptor::Contract): Fused permute + GEMM contraction (maps to `cutensorContract`)
//! - [`ElementwiseMul`](PrimDescriptor::ElementwiseMul): Element-wise multiplication
//!
//! # Algebra parameterization
//!
//! [`TensorPrims<A>`] is parameterized by algebra `A` (e.g.,
//! [`Standard`], `MaxPlus`).
//! External crates implement `TensorPrims<MyAlgebra> for CpuBackend` (orphan rule
//! compatible). The [`HasAlgebra`](tenferro_algebra::HasAlgebra) trait on scalar types
//! enables automatic inference: `Tensor<f64>` → `Standard`.
//!
//! # Examples
//!
//! ## Plan-based GEMM
//!
//! ```ignore
//! use tenferro_prims::{CpuBackend, CpuContext, TensorPrims, PrimDescriptor};
//! use strided_view::StridedArray;
//!
//! let mut ctx = CpuContext::new(4);
//! let a = StridedArray::<f64>::col_major(&[3, 4]);
//! let b = StridedArray::<f64>::col_major(&[4, 5]);
//! let mut c = StridedArray::<f64>::col_major(&[3, 5]);
//!
//! let desc = PrimDescriptor::BatchedGemm {
//!     batch_dims: vec![], m: 3, n: 5, k: 4,
//! };
//! let plan = CpuBackend::plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[4, 5], &[3, 5]]).unwrap();
//! CpuBackend::execute(&mut ctx, &plan, 1.0, &[&a.view(), &b.view()], 0.0, &mut c.view_mut()).unwrap();
//! ```
//!
//! ## Reduction (sum over an axis)
//!
//! ```ignore
//! use tenferro_prims::{CpuBackend, CpuContext, TensorPrims, PrimDescriptor, ReduceOp};
//!
//! let mut ctx = CpuContext::new(4);
//! // Sum over columns: c_i = Σ_j A_{i,j}
//! let desc = PrimDescriptor::Reduce {
//!     modes_a: vec![0, 1], modes_c: vec![0], op: ReduceOp::Sum,
//! };
//! let plan = CpuBackend::plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[3]]).unwrap();
//! CpuBackend::execute(&mut ctx, &plan, 1.0, &[&a.view()], 0.0, &mut c.view_mut()).unwrap();
//! ```
//!
//! ## Contraction (extended operation)
//!
//! ```ignore
//! use tenferro_prims::{CpuBackend, CpuContext, TensorPrims, PrimDescriptor, Extension};
//!
//! let mut ctx = CpuContext::new(4);
//! // Contract is an extended operation — check availability first
//! if CpuBackend::has_extension_for::<f64>(Extension::Contract) {
//!     let desc = PrimDescriptor::Contract {
//!         modes_a: vec![0, 1], modes_b: vec![1, 2], modes_c: vec![0, 2],
//!     };
//!     let plan = CpuBackend::plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[4, 5], &[3, 5]]).unwrap();
//!     CpuBackend::execute(
//!         &ctx, &plan, 1.0, &[&a.view(), &b.view()], 0.0, &mut c.view_mut(),
//!     ).unwrap();
//! }
//! ```

use std::any::Any;
use std::collections::HashMap;
use std::ffi::c_void;
use std::marker::PhantomData;

use strided_traits::ScalarBase;
use strided_view::{StridedView, StridedViewMut};
use tenferro_algebra::{Scalar, Standard};
use tenferro_device::Result;

/// Reduction operation kind.
///
/// # Examples
///
/// ```
/// use tenferro_prims::ReduceOp;
///
/// let op = ReduceOp::Sum;
/// assert_eq!(op, ReduceOp::Sum);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum reduction.
    Sum,
    /// Maximum value reduction.
    Max,
    /// Minimum value reduction.
    Min,
}

/// Element-wise unary operation kind.
///
/// Used with [`PrimDescriptor::ElementwiseUnary`] for point-wise
/// transformations. Maps to `cutensorElementwiseTrinary` (unary case)
/// on GPU.
///
/// Note: square (`x²`) is omitted — expressible as
/// `ElementwiseMul(x, x)` without an extra copy.
///
/// # Examples
///
/// ```
/// use tenferro_prims::UnaryOp;
///
/// let op = UnaryOp::Reciprocal;
/// assert_eq!(op, UnaryOp::Reciprocal);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Negate: `-x`.
    Negate,
    /// Reciprocal: `1 / x`.
    Reciprocal,
    /// Absolute value: `|x|`.
    Abs,
    /// Square root: `√x`.
    Sqrt,
    /// Complex conjugate: `conj(x)`.
    ///
    /// Used by `resolve_conj()` to materialize a lazily-conjugated tensor.
    /// For real types, this is a no-op (identity).
    /// Maps to `CUTENSOR_OP_CONJ` on GPU.
    Conj,
}

/// Extended operation identifiers for dynamic capability query.
///
/// Used with [`TensorPrims::has_extension_for`] to check at runtime whether
/// a backend supports an optimized extended operation for a given scalar type.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::{CpuBackend, TensorPrims, Extension};
///
/// // Check if element-wise multiplication is available for f64
/// let available = CpuBackend::has_extension_for::<f64>(Extension::ElementwiseMul);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extension {
    /// Fused contraction (permute + GEMM). Maps to `cutensorContract` on GPU.
    Contract,
    /// Element-wise multiplication. Maps to `cutensorElementwiseBinary` on GPU.
    ElementwiseMul,
}

/// Describes a tensor primitive operation.
///
/// All operations follow the cuTENSOR pattern: describe → plan → execute.
/// Core operations must be supported by every backend. Extended operations
/// are dynamically queried via [`TensorPrims::has_extension_for`].
///
/// Modes are `u32` integer labels matching cuTENSOR conventions. Modes
/// shared between input and output tensors are batch/free dimensions;
/// modes present only in inputs are contracted.
///
/// # Examples
///
/// ```
/// use tenferro_prims::PrimDescriptor;
///
/// // Batched matrix multiplication: C = A * B
/// let desc = PrimDescriptor::BatchedGemm {
///     batch_dims: vec![], m: 3, n: 5, k: 4,
/// };
/// ```
pub enum PrimDescriptor {
    // ====================================================================
    // Core operations (every backend must implement)
    // ====================================================================
    /// Batched matrix multiplication.
    ///
    /// `C[batch, m, n] = alpha * A[batch, m, k] * B[batch, k, n] + beta * C`
    BatchedGemm {
        /// Batch dimension sizes.
        batch_dims: Vec<usize>,
        /// Number of rows in A / C.
        m: usize,
        /// Number of columns in B / C.
        n: usize,
        /// Contraction dimension (columns of A / rows of B).
        k: usize,
    },

    /// Reduction over modes not present in the output.
    ///
    /// `C[modes_c] = alpha * reduce_op(A[modes_a]) + beta * C[modes_c]`
    Reduce {
        /// Mode labels for input tensor A.
        modes_a: Vec<u32>,
        /// Mode labels for output tensor C (subset of modes_a).
        modes_c: Vec<u32>,
        /// Reduction operation (Sum, Max, Min).
        op: ReduceOp,
    },

    /// Trace: contraction of paired diagonal modes.
    ///
    /// For each pair `(i, j)`, sums over the diagonal where mode i == mode j.
    Trace {
        /// Mode labels for input tensor A.
        modes_a: Vec<u32>,
        /// Mode labels for output tensor C.
        modes_c: Vec<u32>,
        /// Pairs of modes to trace over.
        paired: Vec<(u32, u32)>,
    },

    /// Permute (reorder) tensor modes.
    ///
    /// `B[modes_b] = alpha * A[modes_a]`
    Permute {
        /// Mode labels for input tensor A.
        modes_a: Vec<u32>,
        /// Mode labels for output tensor B (same labels, different order).
        modes_b: Vec<u32>,
    },

    /// Anti-trace: scatter-add gradient to diagonal (AD backward of trace).
    AntiTrace {
        /// Mode labels for input tensor A.
        modes_a: Vec<u32>,
        /// Mode labels for output tensor C.
        modes_c: Vec<u32>,
        /// Pairs of modes for diagonal scatter.
        paired: Vec<(u32, u32)>,
    },

    /// Anti-diag: write gradient to diagonal positions (AD backward of diag).
    AntiDiag {
        /// Mode labels for input tensor A.
        modes_a: Vec<u32>,
        /// Mode labels for output tensor C.
        modes_c: Vec<u32>,
        /// Pairs of modes for diagonal write.
        paired: Vec<(u32, u32)>,
    },

    /// Element-wise unary operation.
    ///
    /// `C[modes] = alpha * op(A[modes]) + beta * C[modes]`
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_prims::{PrimDescriptor, UnaryOp};
    ///
    /// // Reciprocal: C = 1/A
    /// let desc = PrimDescriptor::ElementwiseUnary {
    ///     op: UnaryOp::Reciprocal,
    /// };
    /// ```
    ElementwiseUnary {
        /// Unary operation to apply.
        op: UnaryOp,
    },

    // ====================================================================
    // Extended operations (dynamically queried)
    // ====================================================================
    /// Fused contraction: permute + GEMM in one operation.
    ///
    /// `C[modes_c] = alpha * contract(A[modes_a], B[modes_b]) + beta * C`
    ///
    /// The backend controls internal data movement (copy elision, copy
    /// strategy). Maps to `cutensorContract` on GPU.
    ///
    /// Available when `has_extension_for::<T>(Extension::Contract)` returns true.
    Contract {
        /// Mode labels for input tensor A.
        modes_a: Vec<u32>,
        /// Mode labels for input tensor B.
        modes_b: Vec<u32>,
        /// Mode labels for output tensor C.
        modes_c: Vec<u32>,
    },

    /// Element-wise multiplication of two tensors.
    ///
    /// Available when `has_extension_for::<T>(Extension::ElementwiseMul)` returns true.
    ElementwiseMul,

    // ====================================================================
    // Data movement operations
    // ====================================================================
    /// Copy a tensor to a contiguous (column-major) layout.
    ///
    /// If the input is already contiguous, the backend may return it as-is
    /// (no-op). Used by the einsum layer before GEMM to satisfy stride
    /// requirements.
    MakeContiguous,
}

/// Backend trait for tensor primitive operations, parameterized by algebra `A`.
///
/// Provides a cuTENSOR-compatible plan-based execution model for all
/// operations. Core ops (batched_gemm, reduce, trace, permute,
/// anti_trace, anti_diag, elementwise_unary) must be implemented.
/// Extended ops (contract, elementwise_mul) are dynamically queried via
/// [`has_extension_for`](TensorPrims::has_extension_for).
///
/// # Algebra parameterization
///
/// The algebra parameter `A` enables extensibility: external crates can
/// implement `TensorPrims<MyAlgebra> for CpuBackend` (orphan rule compatible).
///
/// # Execution context
///
/// All operations receive a `&mut Self::Context` that encapsulates the backend's
/// execution resources:
///
/// - **CPU**: thread pool, buffer pool, plan cache
/// - **GPU**: CUDA stream, device handle, workspace
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::{CpuBackend, CpuContext, TensorPrims, PrimDescriptor};
///
/// let mut ctx = CpuContext::new(4); // 4 threads
///
/// let desc = PrimDescriptor::BatchedGemm {
///     batch_dims: vec![], m: 3, n: 5, k: 4,
/// };
/// let plan = CpuBackend::plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[4, 5], &[3, 5]]).unwrap();
/// CpuBackend::execute(&mut ctx, &plan, 1.0, &[&a.view(), &b.view()], 0.0, &mut c.view_mut()).unwrap();
/// ```
pub trait TensorPrims<A> {
    /// Backend-specific plan type (no type erasure).
    type Plan<T: ScalarBase>;

    /// Backend-specific execution context.
    ///
    /// Encapsulates execution resources (thread pool for CPU, CUDA stream
    /// for GPU). Analogous to cuTENSOR's `cutensorHandle_t`.
    type Context;

    /// Create an execution plan from an operation descriptor.
    ///
    /// The plan pre-computes kernel selection and workspace sizes.
    /// `shapes` contains the shape of each tensor involved in the operation
    /// (inputs first, then output).
    fn plan<T: ScalarBase>(
        ctx: &mut Self::Context,
        desc: &PrimDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan<T>>;

    /// Execute a plan with the given scaling factors and tensor views.
    ///
    /// Follows the BLAS/cuTENSOR pattern:
    /// `output = alpha * op(inputs) + beta * output`
    fn execute<T: ScalarBase>(
        ctx: &mut Self::Context,
        plan: &Self::Plan<T>,
        alpha: T,
        inputs: &[&StridedView<T>],
        beta: T,
        output: &mut StridedViewMut<T>,
    ) -> Result<()>;

    /// Query whether an extended operation is available for scalar type `T`.
    ///
    /// Returns `true` if the backend supports the given extended operation
    /// for the specified scalar type.
    fn has_extension_for<T: ScalarBase>(ext: Extension) -> bool;
}

/// CPU plan — concrete enum, no type erasure.
///
/// Created by [`CpuBackend::plan`](TensorPrims::plan) and consumed by
/// [`CpuBackend::execute`](TensorPrims::execute).
pub enum CpuPlan<T: ScalarBase> {
    /// Plan for batched GEMM.
    BatchedGemm {
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
        /// Axis to reduce over.
        axis: usize,
        /// Reduction operation.
        op: ReduceOp,
        _marker: PhantomData<T>,
    },
    /// Plan for trace.
    Trace {
        /// Paired modes.
        paired: Vec<(u32, u32)>,
        _marker: PhantomData<T>,
    },
    /// Plan for permutation.
    Permute {
        /// Permutation mapping.
        perm: Vec<usize>,
        _marker: PhantomData<T>,
    },
    /// Plan for anti-trace (AD backward).
    AntiTrace {
        /// Paired modes.
        paired: Vec<(u32, u32)>,
        _marker: PhantomData<T>,
    },
    /// Plan for anti-diag (AD backward).
    AntiDiag {
        /// Paired modes.
        paired: Vec<(u32, u32)>,
        _marker: PhantomData<T>,
    },
    /// Plan for element-wise unary operation.
    ElementwiseUnary {
        /// Unary operation.
        op: UnaryOp,
        _marker: PhantomData<T>,
    },
    /// Plan for fused contraction (core op).
    Contract { _marker: PhantomData<T> },
    /// Plan for element-wise multiplication (extended op).
    ElementwiseMul { _marker: PhantomData<T> },
    /// Plan for making a tensor contiguous.
    MakeContiguous { _marker: PhantomData<T> },
}

/// Type-erased buffer pool for reusing allocations across operations.
///
/// Stores recycled `Vec<T>` buffers keyed by `TypeId`, avoiding repeated
/// heap allocation in hot loops (e.g., N-ary contraction intermediates
/// and GEMM contiguous buffers).
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::BufferPool;
///
/// let mut pool = BufferPool::new();
/// let buf: Vec<f64> = pool.acquire(1024);
/// // ... use buf ...
/// pool.recycle(buf);
/// ```
pub struct BufferPool {
    pools: HashMap<std::any::TypeId, Box<dyn Any>>,
    enabled: bool,
    max_per_type: usize,
    max_bytes: usize,
}

impl BufferPool {
    /// Create a new buffer pool with default limits.
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
            enabled: true,
            max_per_type: 16,
            max_bytes: 64 * 1024 * 1024, // 64 MB
        }
    }

    /// Acquire a buffer of at least `len` elements.
    ///
    /// If a suitable recycled buffer exists, it is returned.
    /// Otherwise, a new `Vec<T>` is allocated.
    pub fn acquire<T: 'static>(&mut self, _len: usize) -> Vec<T> {
        todo!()
    }

    /// Return a buffer to the pool for future reuse.
    pub fn recycle<T: 'static>(&mut self, _buf: Vec<T>) {
        todo!()
    }

    /// Enable or disable the pool at runtime.
    ///
    /// When disabled, `acquire` always allocates and `recycle` drops.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Returns whether the pool is currently enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache for pre-computed execution plans, keyed by `(PrimDescriptor, shapes)`.
///
/// Avoids repeated plan generation when the same operation is executed
/// with the same shapes (e.g., single-tensor einsum steps in a loop).
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::PlanCache;
///
/// let cache = PlanCache::new();
/// ```
pub struct PlanCache {
    // Internal representation is private.
    // Key: (PrimDescriptor hash, shapes hash) → type-erased plan.
    _entries: HashMap<u64, Box<dyn Any>>,
}

impl PlanCache {
    /// Create a new empty plan cache.
    pub fn new() -> Self {
        Self {
            _entries: HashMap::new(),
        }
    }
}

impl Default for PlanCache {
    fn default() -> Self {
        Self::new()
    }
}

/// CPU execution context.
///
/// Encapsulates CPU-side execution resources, analogous to cuTENSOR's
/// `cutensorHandle_t`. Holds a rayon thread pool, a [`BufferPool`] for
/// allocation reuse, and a [`PlanCache`] for plan reuse.
///
/// # Examples
///
/// ```
/// use tenferro_prims::CpuContext;
///
/// let mut ctx = CpuContext::new(4); // 4-thread pool
/// assert_eq!(ctx.num_threads(), 4);
/// assert!(ctx.buffer_pool().is_enabled());
/// ```
pub struct CpuContext {
    pool: rayon::ThreadPool,
    buffer_pool: BufferPool,
    plan_cache: PlanCache,
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
            buffer_pool: BufferPool::new(),
            plan_cache: PlanCache::new(),
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

    /// Returns a reference to the buffer pool.
    pub fn buffer_pool(&self) -> &BufferPool {
        &self.buffer_pool
    }

    /// Returns a mutable reference to the buffer pool.
    pub fn buffer_pool_mut(&mut self) -> &mut BufferPool {
        &mut self.buffer_pool
    }

    /// Returns a mutable reference to the plan cache.
    pub fn plan_cache_mut(&mut self) -> &mut PlanCache {
        &mut self.plan_cache
    }
}

/// CPU backend using strided-kernel and GEMM.
///
/// Dispatched automatically when tensors reside on
/// [`LogicalMemorySpace::MainMemory`](tenferro_device::LogicalMemorySpace::MainMemory).
/// Implements [`TensorPrims<Standard>`] for standard arithmetic.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::{CpuBackend, CpuContext, TensorPrims, PrimDescriptor};
/// use strided_view::StridedArray;
///
/// let mut ctx = CpuContext::new(4);
/// let desc = PrimDescriptor::Permute {
///     modes_a: vec![0, 1],
///     modes_b: vec![1, 0],
/// };
/// let plan = CpuBackend::plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[4, 3]]).unwrap();
/// let a = StridedArray::<f64>::col_major(&[3, 4]);
/// let mut b = StridedArray::<f64>::col_major(&[4, 3]);
/// CpuBackend::execute(&mut ctx, &plan, 1.0, &[&a.view()], 0.0, &mut b.view_mut()).unwrap();
/// ```
pub struct CpuBackend;

impl CpuBackend {
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
    pub fn resolve_conj<T: Scalar>(
        _ctx: &mut CpuContext,
        _src: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_tensor::Tensor<T> {
        todo!()
    }
}

impl TensorPrims<Standard> for CpuBackend {
    type Plan<T: ScalarBase> = CpuPlan<T>;
    type Context = CpuContext;

    fn plan<T: ScalarBase>(
        _ctx: &mut CpuContext,
        _desc: &PrimDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<CpuPlan<T>> {
        todo!()
    }

    fn execute<T: ScalarBase>(
        _ctx: &mut CpuContext,
        _plan: &CpuPlan<T>,
        _alpha: T,
        _inputs: &[&StridedView<T>],
        _beta: T,
        _output: &mut StridedViewMut<T>,
    ) -> Result<()> {
        todo!()
    }

    fn has_extension_for<T: ScalarBase>(_ext: Extension) -> bool {
        todo!()
    }
}

// ===========================================================================
// GPU Backends (future — runtime dlopen via libloading)
// ===========================================================================

/// CUDA execution context.
///
/// Encapsulates CUDA-side execution resources: a CUDA stream, GPU workspace
/// buffer, and plan cache. Analogous to cuTENSOR's `cutensorHandle_t`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::CudaContext;
///
/// // Created internally by CudaBackend::load_cutensor()
/// ```
pub struct CudaContext {
    _stream: *mut c_void,
    _workspace: Vec<u8>,
    _plan_cache: PlanCache,
}

/// CUDA plan — wraps a cuTENSOR plan handle.
///
/// Created by [`CudaBackend::plan`](TensorPrims::plan) and consumed by
/// [`CudaBackend::execute`](TensorPrims::execute).
pub struct CudaPlan<T: ScalarBase> {
    _handle: *mut c_void,
    _workspace_size: usize,
    _marker: PhantomData<T>,
}

/// CUDA backend using cuTENSOR via runtime dlopen.
///
/// Loaded at runtime from a user-provided `.so` path. No compile-time
/// CUDA SDK dependency. Implements [`TensorPrims<Standard>`] for standard
/// arithmetic on NVIDIA GPUs.
///
/// cuTENSOR natively supports `Contract`, `Permute`, `Reduce`, and
/// `ElementwiseMul`. `AntiTrace`/`AntiDiag` are composed via
/// `Contract(eye, ∂C)`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::{CudaBackend, BackendRegistry};
///
/// let mut registry = BackendRegistry::new();
/// registry.load_cutensor("/usr/lib/libcutensor.so").unwrap();
/// ```
pub struct CudaBackend {
    _handle: *mut c_void,
    _lib: libloading::Library,
}

impl CudaBackend {
    /// Materialize a lazily-conjugated tensor on GPU.
    ///
    /// Uses `ElementwiseUnary(Conj)` via cuTENSOR to produce a new
    /// tensor with `conjugated = false`.
    pub fn resolve_conj<T: Scalar>(
        _ctx: &mut CudaContext,
        _src: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_tensor::Tensor<T> {
        todo!()
    }
}

impl TensorPrims<Standard> for CudaBackend {
    type Plan<T: ScalarBase> = CudaPlan<T>;
    type Context = CudaContext;

    fn plan<T: ScalarBase>(
        _ctx: &mut CudaContext,
        _desc: &PrimDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<CudaPlan<T>> {
        todo!()
    }

    fn execute<T: ScalarBase>(
        _ctx: &mut CudaContext,
        _plan: &CudaPlan<T>,
        _alpha: T,
        _inputs: &[&StridedView<T>],
        _beta: T,
        _output: &mut StridedViewMut<T>,
    ) -> Result<()> {
        todo!()
    }

    fn has_extension_for<T: ScalarBase>(_ext: Extension) -> bool {
        // cuTENSOR supports Contract and ElementwiseMul for f32/f64/Complex
        todo!()
    }
}

/// ROCm execution context.
///
/// Encapsulates ROCm-side execution resources: a HIP stream, GPU workspace
/// buffer, and plan cache. Analogous to hipTENSOR's handle.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::RocmContext;
///
/// // Created internally by RocmBackend::load_hiptensor()
/// ```
pub struct RocmContext {
    _stream: *mut c_void,
    _workspace: Vec<u8>,
    _plan_cache: PlanCache,
}

/// ROCm plan — wraps a hipTENSOR plan handle.
///
/// Created by [`RocmBackend::plan`](TensorPrims::plan) and consumed by
/// [`RocmBackend::execute`](TensorPrims::execute).
pub struct RocmPlan<T: ScalarBase> {
    _handle: *mut c_void,
    _workspace_size: usize,
    _marker: PhantomData<T>,
}

/// ROCm backend using hipTENSOR via runtime dlopen.
///
/// Loaded at runtime from a user-provided `.so` path. No compile-time
/// ROCm SDK dependency. Implements [`TensorPrims<Standard>`] for standard
/// arithmetic on AMD GPUs.
///
/// hipTENSOR natively supports `Contract`, `Permute`, `Reduce`, and
/// `ElementwiseMul`. `AntiTrace`/`AntiDiag` are composed via
/// `Contract(eye, ∂C)`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::{RocmBackend, BackendRegistry};
///
/// let mut registry = BackendRegistry::new();
/// registry.load_hiptensor("/usr/lib/libhiptensor.so").unwrap();
/// ```
pub struct RocmBackend {
    _handle: *mut c_void,
    _lib: libloading::Library,
}

impl RocmBackend {
    /// Materialize a lazily-conjugated tensor on GPU.
    ///
    /// Uses `ElementwiseUnary(Conj)` via hipTENSOR to produce a new
    /// tensor with `conjugated = false`.
    pub fn resolve_conj<T: Scalar>(
        _ctx: &mut RocmContext,
        _src: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_tensor::Tensor<T> {
        todo!()
    }
}

impl TensorPrims<Standard> for RocmBackend {
    type Plan<T: ScalarBase> = RocmPlan<T>;
    type Context = RocmContext;

    fn plan<T: ScalarBase>(
        _ctx: &mut RocmContext,
        _desc: &PrimDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<RocmPlan<T>> {
        todo!()
    }

    fn execute<T: ScalarBase>(
        _ctx: &mut RocmContext,
        _plan: &RocmPlan<T>,
        _alpha: T,
        _inputs: &[&StridedView<T>],
        _beta: T,
        _output: &mut StridedViewMut<T>,
    ) -> Result<()> {
        todo!()
    }

    fn has_extension_for<T: ScalarBase>(_ext: Extension) -> bool {
        // hipTENSOR supports Contract and ElementwiseMul for f32/f64/Complex
        todo!()
    }
}

// ===========================================================================
// Backend Registry
// ===========================================================================

/// Registry of available compute backends.
///
/// Holds the CPU backend (always available) and optional GPU backends
/// loaded at runtime via [`load_cutensor`](BackendRegistry::load_cutensor)
/// or [`load_hiptensor`](BackendRegistry::load_hiptensor).
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::BackendRegistry;
///
/// let mut registry = BackendRegistry::new(); // CPU only
/// registry.load_cutensor("/usr/lib/libcutensor.so").unwrap();
/// assert!(registry.cuda().is_some());
/// ```
pub struct BackendRegistry {
    cpu: CpuBackend,
    cuda: Option<CudaBackend>,
    rocm: Option<RocmBackend>,
}

impl BackendRegistry {
    /// Create a registry with CPU backend only.
    pub fn new() -> Self {
        Self {
            cpu: CpuBackend,
            cuda: None,
            rocm: None,
        }
    }

    /// Load the cuTENSOR library from the given path.
    ///
    /// The caller (Julia, Python, or standalone Rust) provides the path
    /// to the shared library. No auto-search.
    pub fn load_cutensor(&mut self, _path: &str) -> Result<()> {
        todo!()
    }

    /// Load the hipTENSOR library from the given path.
    ///
    /// The caller (Julia, Python, or standalone Rust) provides the path
    /// to the shared library. No auto-search.
    pub fn load_hiptensor(&mut self, _path: &str) -> Result<()> {
        todo!()
    }

    /// Returns a reference to the CPU backend.
    pub fn cpu(&self) -> &CpuBackend {
        &self.cpu
    }

    /// Returns a reference to the CUDA backend, if loaded.
    pub fn cuda(&self) -> Option<&CudaBackend> {
        self.cuda.as_ref()
    }

    /// Returns a reference to the ROCm backend, if loaded.
    pub fn rocm(&self) -> Option<&RocmBackend> {
        self.rocm.as_ref()
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}
