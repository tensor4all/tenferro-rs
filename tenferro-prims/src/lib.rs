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
//! [`Standard<f64>`](Standard), `MaxPlusAlgebra`). The algebra type carries
//! its scalar type via `A::Scalar` (see [`Semiring`](tenferro_algebra::Semiring)).
//! External crates implement `TensorPrims<MyAlgebra> for CpuBackend` (orphan rule
//! compatible). The [`HasAlgebra`](tenferro_algebra::HasAlgebra) trait on scalar types
//! provides UX sugar for automatic inference: `Tensor<f64>` → `Standard<f64>`.
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
use tenferro_device::{Error, Result};

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
        /// Paired axis positions in input tensor.
        paired_axes: Vec<(usize, usize)>,
        /// Output axis positions mapping.
        free_axes: Vec<usize>,
        _marker: PhantomData<T>,
    },
    /// Plan for permutation.
    Permute {
        /// Permutation mapping (perm[out_axis] = in_axis).
        perm: Vec<usize>,
        _marker: PhantomData<T>,
    },
    /// Plan for anti-trace (AD backward).
    AntiTrace {
        /// Paired axis positions in output tensor.
        paired_axes: Vec<(usize, usize)>,
        /// Input axis positions mapping.
        free_axes: Vec<usize>,
        _marker: PhantomData<T>,
    },
    /// Plan for anti-diag (AD backward).
    AntiDiag {
        /// Paired axis positions in output tensor.
        paired_axes: Vec<(usize, usize)>,
        /// Input axis positions mapping.
        free_axes: Vec<usize>,
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
        _marker: PhantomData<T>,
    },
    /// Plan for element-wise multiplication (extended op).
    ElementwiseMul { _marker: PhantomData<T> },
    /// Plan for making a tensor contiguous.
    MakeContiguous { _marker: PhantomData<T> },
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
        src: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_tensor::Tensor<T> {
        if !src.is_conjugated() {
            return src.clone();
        }
        // Create a fresh non-conjugated copy of the data.
        // For real types (f64, f32), conjugation is identity, so raw data copy is correct.
        // Complex types would additionally need element-wise conjugation (requires
        // Conjugate trait bound, not available via Scalar).
        let contiguous = src.contiguous(tenferro_tensor::MemoryOrder::ColumnMajor);
        let data = contiguous
            .buffer()
            .as_slice()
            .expect("CPU tensor must have CPU-accessible data");
        tenferro_tensor::Tensor::from_slice(
            data,
            src.dims(),
            tenferro_tensor::MemoryOrder::ColumnMajor,
        )
        .expect("from_slice should succeed with valid data and dims")
    }
}

// ===========================================================================
// Helpers for multi-index iteration
// ===========================================================================

/// Iterate over all index combinations for the given dimensions (column-major order).
fn for_each_index(dims: &[usize], mut f: impl FnMut(&[usize])) {
    let ndim = dims.len();
    if ndim == 0 {
        f(&[]);
        return;
    }
    let total: usize = dims.iter().product();
    if total == 0 {
        return;
    }
    let mut index = vec![0usize; ndim];
    for _ in 0..total {
        f(&index);
        // Increment column-major
        for d in 0..ndim {
            index[d] += 1;
            if index[d] < dims[d] {
                break;
            }
            index[d] = 0;
        }
    }
}

/// Unflatten a linear index to multi-dimensional indices (column-major).
fn unflatten_index(flat: usize, dims: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; dims.len()];
    let mut remainder = flat;
    for d in 0..dims.len() {
        indices[d] = remainder % dims[d];
        remainder /= dims[d];
    }
    indices
}

/// Find the position of a mode label in a mode list, returning an error if not found.
fn mode_position(modes: &[u32], label: u32) -> Result<usize> {
    modes
        .iter()
        .position(|&m| m == label)
        .ok_or_else(|| Error::InvalidArgument(format!("mode label {label} not found")))
}

/// Scale all elements of the output by `beta`, or zero them if `beta == 0`.
fn scale_output<T: ScalarBase>(output: &mut StridedViewMut<T>, beta: T) {
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

// ===========================================================================
// CPU execute helpers for each operation
// ===========================================================================

fn execute_permute<T: ScalarBase>(
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
        strided_perm::copy_into(output, &permuted)
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

fn execute_make_contiguous<T: ScalarBase>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
) -> Result<()> {
    if alpha == T::one() && beta == T::zero() {
        strided_perm::copy_into(output, input).map_err(|e| Error::StrideError(e.to_string()))?;
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

fn execute_batched_gemm<T: ScalarBase>(
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
    let batch_size: usize = if batch_dims.is_empty() {
        1
    } else {
        batch_dims.iter().product()
    };

    for batch_flat in 0..batch_size {
        let batch_idx = unflatten_index(batch_flat, batch_dims);
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for kk in 0..k {
                    let mut a_idx = batch_idx.clone();
                    a_idx.push(i);
                    a_idx.push(kk);
                    let mut b_idx = batch_idx.clone();
                    b_idx.push(kk);
                    b_idx.push(j);
                    sum = sum + a.get(&a_idx) * b.get(&b_idx);
                }
                let mut c_idx = batch_idx.clone();
                c_idx.push(i);
                c_idx.push(j);
                let old = if beta == T::zero() {
                    T::zero()
                } else {
                    beta * output.get(&c_idx)
                };
                output.set(&c_idx, alpha * sum + old);
            }
        }
    }
    Ok(())
}

fn execute_reduce_sum<T: ScalarBase>(
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

    for_each_index(&out_dims, |out_idx| {
        let mut sum = T::zero();
        for red_flat in 0..reduced_total {
            let red_idx = unflatten_index(red_flat, &reduced_dims);
            // Build full input index by interleaving free and reduced
            let mut in_idx = Vec::with_capacity(in_dims.len());
            let mut out_pos = 0;
            let mut red_pos = 0;
            for ax in 0..in_dims.len() {
                if red_pos < reduced_axes.len() && reduced_axes[red_pos] == ax {
                    in_idx.push(red_idx[red_pos]);
                    red_pos += 1;
                } else {
                    in_idx.push(out_idx[out_pos]);
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

fn execute_trace<T: ScalarBase>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    paired_axes: &[(usize, usize)],
    free_axes: &[usize],
) -> Result<()> {
    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    // All paired axes must have the same dimension
    let diag_dim = in_dims[paired_axes[0].0];

    for_each_index(&out_dims, |out_idx| {
        let mut sum = T::zero();
        for d in 0..diag_dim {
            let mut in_idx = vec![0; in_dims.len()];
            for (out_pos, &in_ax) in free_axes.iter().enumerate() {
                in_idx[in_ax] = out_idx[out_pos];
            }
            for &(ax1, ax2) in paired_axes {
                in_idx[ax1] = d;
                in_idx[ax2] = d;
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

fn execute_anti_trace<T: ScalarBase>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    paired_axes: &[(usize, usize)],
    free_axes: &[usize],
) -> Result<()> {
    // AntiTrace: C = alpha * antitrace(A) + beta * C
    // First scale output by beta (since diagonal positions may be written multiple times)
    scale_output(output, beta);

    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let diag_dim = out_dims[paired_axes[0].0];

    // For each input element, scatter to all diagonal positions in output
    for_each_index(&in_dims, |in_idx| {
        let val = alpha * input.get(in_idx);
        for d in 0..diag_dim {
            let mut out_idx = vec![0; out_dims.len()];
            for (in_pos, &out_ax) in free_axes.iter().enumerate() {
                out_idx[out_ax] = in_idx[in_pos];
            }
            for &(ax1, ax2) in paired_axes {
                out_idx[ax1] = d;
                out_idx[ax2] = d;
            }
            let old = output.get(&out_idx);
            output.set(&out_idx, old + val);
        }
    });
    Ok(())
}

fn execute_anti_diag<T: ScalarBase>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    paired_axes: &[(usize, usize)],
    free_axes: &[usize],
) -> Result<()> {
    // AntiDiag: write input values to diagonal positions in output
    scale_output(output, beta);

    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let diag_dim = out_dims[paired_axes[0].0];

    for_each_index(&in_dims, |in_idx| {
        let val = alpha * input.get(in_idx);
        for d in 0..diag_dim {
            let mut out_idx = vec![0; out_dims.len()];
            for (in_pos, &out_ax) in free_axes.iter().enumerate() {
                out_idx[out_ax] = in_idx[in_pos];
            }
            for &(ax1, ax2) in paired_axes {
                out_idx[ax1] = d;
                out_idx[ax2] = d;
            }
            let old = output.get(&out_idx);
            output.set(&out_idx, old + val);
        }
    });
    Ok(())
}

fn execute_elementwise_mul<T: ScalarBase>(
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
) -> Result<()> {
    let a = inputs[0];
    let b = inputs[1];
    let dims = output.dims().to_vec();
    for_each_index(&dims, |idx| {
        let val = alpha * (a.get(idx) * b.get(idx));
        if beta == T::zero() {
            output.set(idx, val);
        } else {
            output.set(idx, val + beta * output.get(idx));
        }
    });
    Ok(())
}

fn execute_contract<T: ScalarBase>(
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
    modes_a: &[u32],
    modes_b: &[u32],
    modes_c: &[u32],
) -> Result<()> {
    let a = inputs[0];
    let b = inputs[1];

    // Determine contracted modes: in both A and B but not in C
    let contracted_modes: Vec<u32> = modes_a
        .iter()
        .filter(|m| modes_b.contains(m) && !modes_c.contains(m))
        .copied()
        .collect();
    let contracted_dims: Vec<usize> = contracted_modes
        .iter()
        .map(|&m| {
            let a_pos = modes_a.iter().position(|&mm| mm == m).unwrap();
            a.dims()[a_pos]
        })
        .collect();
    let contracted_total: usize = if contracted_dims.is_empty() {
        1
    } else {
        contracted_dims.iter().product()
    };

    let out_dims = output.dims().to_vec();

    for_each_index(&out_dims, |c_idx| {
        let mut sum = T::zero();
        for k_flat in 0..contracted_total {
            let k_idx = unflatten_index(k_flat, &contracted_dims);

            // Build A indices
            let mut a_idx = vec![0; modes_a.len()];
            for (ax, &mode) in modes_a.iter().enumerate() {
                if let Some(c_pos) = modes_c.iter().position(|&m| m == mode) {
                    a_idx[ax] = c_idx[c_pos];
                } else if let Some(k_pos) = contracted_modes.iter().position(|&m| m == mode) {
                    a_idx[ax] = k_idx[k_pos];
                }
            }

            // Build B indices
            let mut b_idx = vec![0; modes_b.len()];
            for (ax, &mode) in modes_b.iter().enumerate() {
                if let Some(c_pos) = modes_c.iter().position(|&m| m == mode) {
                    b_idx[ax] = c_idx[c_pos];
                } else if let Some(k_pos) = contracted_modes.iter().position(|&m| m == mode) {
                    b_idx[ax] = k_idx[k_pos];
                }
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

// ===========================================================================
// CPU backend TensorPrims implementation
// ===========================================================================

impl<S: Scalar> TensorPrims<Standard<S>> for CpuBackend {
    type Plan<T: ScalarBase> = CpuPlan<T>;
    type Context = CpuContext;

    fn plan<T: ScalarBase>(
        _ctx: &mut CpuContext,
        desc: &PrimDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<CpuPlan<T>> {
        match desc {
            PrimDescriptor::BatchedGemm {
                batch_dims,
                m,
                n,
                k,
            } => Ok(CpuPlan::BatchedGemm {
                batch_dims: batch_dims.clone(),
                m: *m,
                n: *n,
                k: *k,
                _marker: PhantomData,
            }),

            PrimDescriptor::Reduce {
                modes_a,
                modes_c,
                op,
            } => {
                // reduced_axes = positions in modes_a not present in modes_c
                let reduced_axes: Vec<usize> = modes_a
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| !modes_c.contains(m))
                    .map(|(i, _)| i)
                    .collect();
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
                let paired_axes: Vec<(usize, usize)> = paired
                    .iter()
                    .map(|(m1, m2)| {
                        Ok((mode_position(modes_a, *m1)?, mode_position(modes_a, *m2)?))
                    })
                    .collect::<Result<_>>()?;
                let free_axes: Vec<usize> = modes_c
                    .iter()
                    .map(|m| mode_position(modes_a, *m))
                    .collect::<Result<_>>()?;
                Ok(CpuPlan::Trace {
                    paired_axes,
                    free_axes,
                    _marker: PhantomData,
                })
            }

            PrimDescriptor::Permute { modes_a, modes_b } => {
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
                Ok(CpuPlan::AntiTrace {
                    paired_axes,
                    free_axes,
                    _marker: PhantomData,
                })
            }

            PrimDescriptor::AntiDiag {
                modes_a,
                modes_c,
                paired,
            } => {
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
                Ok(CpuPlan::AntiDiag {
                    paired_axes,
                    free_axes,
                    _marker: PhantomData,
                })
            }

            PrimDescriptor::ElementwiseUnary { op } => Ok(CpuPlan::ElementwiseUnary {
                op: *op,
                _marker: PhantomData,
            }),

            PrimDescriptor::Contract {
                modes_a,
                modes_b,
                modes_c,
            } => Ok(CpuPlan::Contract {
                modes_a: modes_a.clone(),
                modes_b: modes_b.clone(),
                modes_c: modes_c.clone(),
                _marker: PhantomData,
            }),

            PrimDescriptor::ElementwiseMul => Ok(CpuPlan::ElementwiseMul {
                _marker: PhantomData,
            }),

            PrimDescriptor::MakeContiguous => Ok(CpuPlan::MakeContiguous {
                _marker: PhantomData,
            }),
        }
    }

    fn execute<T: ScalarBase>(
        _ctx: &mut CpuContext,
        plan: &CpuPlan<T>,
        alpha: T,
        inputs: &[&StridedView<T>],
        beta: T,
        output: &mut StridedViewMut<T>,
    ) -> Result<()> {
        match plan {
            CpuPlan::Permute { perm, .. } => execute_permute(alpha, inputs[0], beta, output, perm),

            CpuPlan::MakeContiguous { .. } => {
                execute_make_contiguous(alpha, inputs[0], beta, output)
            }

            CpuPlan::BatchedGemm {
                batch_dims,
                m,
                n,
                k,
                ..
            } => execute_batched_gemm(alpha, inputs, beta, output, batch_dims, *m, *n, *k),

            CpuPlan::Reduce {
                reduced_axes, op, ..
            } => match op {
                ReduceOp::Sum => execute_reduce_sum(alpha, inputs[0], beta, output, reduced_axes),
                ReduceOp::Max | ReduceOp::Min => Err(Error::InvalidArgument(
                    "Max/Min reduction requires PartialOrd, not available via ScalarBase".into(),
                )),
            },

            CpuPlan::Trace {
                paired_axes,
                free_axes,
                ..
            } => execute_trace(alpha, inputs[0], beta, output, paired_axes, free_axes),

            CpuPlan::AntiTrace {
                paired_axes,
                free_axes,
                ..
            } => execute_anti_trace(alpha, inputs[0], beta, output, paired_axes, free_axes),

            CpuPlan::AntiDiag {
                paired_axes,
                free_axes,
                ..
            } => execute_anti_diag(alpha, inputs[0], beta, output, paired_axes, free_axes),

            CpuPlan::ElementwiseUnary { op, .. } => {
                // Conj is identity for real types (ScalarBase)
                match op {
                    UnaryOp::Conj => {
                        execute_make_contiguous(alpha, inputs[0], beta, output)
                    }
                    _ => Err(Error::InvalidArgument(format!(
                        "unary op {op:?} requires additional trait bounds not available via ScalarBase"
                    ))),
                }
            }

            CpuPlan::ElementwiseMul { .. } => execute_elementwise_mul(alpha, inputs, beta, output),

            CpuPlan::Contract {
                modes_a,
                modes_b,
                modes_c,
                ..
            } => execute_contract(alpha, inputs, beta, output, modes_a, modes_b, modes_c),
        }
    }

    fn has_extension_for<T: ScalarBase>(_ext: Extension) -> bool {
        // CPU backend supports both Contract and ElementwiseMul
        true
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
/// CUDA SDK dependency. Implements [`TensorPrims<Standard<T>>`](TensorPrims)
/// for standard arithmetic on NVIDIA GPUs.
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
        unimplemented!("CUDA backend not available: load cuTENSOR library first")
    }
}

impl<S: Scalar> TensorPrims<Standard<S>> for CudaBackend {
    type Plan<T: ScalarBase> = CudaPlan<T>;
    type Context = CudaContext;

    fn plan<T: ScalarBase>(
        _ctx: &mut CudaContext,
        _desc: &PrimDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<CudaPlan<T>> {
        Err(Error::DeviceError(
            "CUDA backend not available: load cuTENSOR library first".into(),
        ))
    }

    fn execute<T: ScalarBase>(
        _ctx: &mut CudaContext,
        _plan: &CudaPlan<T>,
        _alpha: T,
        _inputs: &[&StridedView<T>],
        _beta: T,
        _output: &mut StridedViewMut<T>,
    ) -> Result<()> {
        Err(Error::DeviceError(
            "CUDA backend not available: load cuTENSOR library first".into(),
        ))
    }

    fn has_extension_for<T: ScalarBase>(_ext: Extension) -> bool {
        // cuTENSOR supports Contract and ElementwiseMul for f32/f64/Complex
        false
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
/// ROCm SDK dependency. Implements [`TensorPrims<Standard<T>>`](TensorPrims)
/// for standard arithmetic on AMD GPUs.
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
        unimplemented!("ROCm backend not available: load hipTENSOR library first")
    }
}

impl<S: Scalar> TensorPrims<Standard<S>> for RocmBackend {
    type Plan<T: ScalarBase> = RocmPlan<T>;
    type Context = RocmContext;

    fn plan<T: ScalarBase>(
        _ctx: &mut RocmContext,
        _desc: &PrimDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<RocmPlan<T>> {
        Err(Error::DeviceError(
            "ROCm backend not available: load hipTENSOR library first".into(),
        ))
    }

    fn execute<T: ScalarBase>(
        _ctx: &mut RocmContext,
        _plan: &RocmPlan<T>,
        _alpha: T,
        _inputs: &[&StridedView<T>],
        _beta: T,
        _output: &mut StridedViewMut<T>,
    ) -> Result<()> {
        Err(Error::DeviceError(
            "ROCm backend not available: load hipTENSOR library first".into(),
        ))
    }

    fn has_extension_for<T: ScalarBase>(_ext: Extension) -> bool {
        // hipTENSOR supports Contract and ElementwiseMul for f32/f64/Complex
        false
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
        Err(Error::DeviceError(
            "cuTENSOR runtime loading not yet implemented".into(),
        ))
    }

    /// Load the hipTENSOR library from the given path.
    ///
    /// The caller (Julia, Python, or standalone Rust) provides the path
    /// to the shared library. No auto-search.
    pub fn load_hiptensor(&mut self, _path: &str) -> Result<()> {
        Err(Error::DeviceError(
            "hipTENSOR runtime loading not yet implemented".into(),
        ))
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
