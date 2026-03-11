//! Tensor primitive operations for the tenferro workspace.
//!
//! This crate defines the [`TensorPrims<Alg>`] trait, a backend-agnostic interface
//! parameterized by algebra `Alg`. The API follows the cuTENSOR plan-based execution
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
//! # CPU GEMM backend selection
//!
//! `BatchedGemm` on [`CpuBackend`] requires exactly one CPU GEMM backend feature:
//! - `gemm-faer` (default): pure-Rust faer matmul backend
//! - `gemm-blas`: CBLAS backend (`cblas-sys`) with selectable symbol provider
//!
//! If `gemm-blas` is selected, choose exactly one provider:
//! - `provider-src`: link BLAS source crates (`blas-src` + `cblas-src`)
//! - `provider-inject`: link runtime-injected symbols (`cblas-inject`)
//!
//! With `provider-src`, choose exactly one `src-*` implementation:
//! `src-openblas`, `src-netlib`, `src-accelerate`, `src-r`,
//! `src-intel-mkl-dynamic-sequential`, `src-intel-mkl-dynamic-parallel`,
//! `src-intel-mkl-static-sequential`, `src-intel-mkl-static-parallel`.
//!
//! Example (OpenBLAS source provider):
//! `cargo test -p tenferro-prims --no-default-features --features "gemm-blas,provider-src,src-openblas"`
//!
//! Example (runtime-injected provider):
//! `cargo test -p tenferro-prims --no-default-features --features "gemm-blas,provider-inject"`
//!
//! On [`CpuBackend`], [`PrimDescriptor::BatchedGemm`] supports `f32`, `f64`,
//! `Complex32`, and `Complex64`. [`ReduceOp::Max`] and [`ReduceOp::Min`]
//! require ordered real scalars (`f32` or `f64`).

//! # Algebra parameterization
//!
//! [`TensorPrims<Alg>`] is parameterized by algebra `Alg` (e.g.,
//! [`Standard<f64>`](tenferro_algebra::Standard), `MaxPlusAlgebra`). The algebra type carries
//! its scalar type via `Alg::Scalar` (see [`Semiring`](tenferro_algebra::Semiring)).
//! External crates implement `TensorPrims<MyAlgebra> for CpuBackend` (orphan rule
//! compatible). The [`HasAlgebra`](tenferro_algebra::HasAlgebra) trait on scalar types
//! provides UX sugar for automatic inference: `Tensor<f64>` → `Standard<f64>`.
//!
//! # Examples
//!
//! ## Plan-based GEMM
//!
//! ```ignore
//! use tenferro_algebra::Standard;
//! use tenferro_device::LogicalMemorySpace;
//! use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};
//! use tenferro_tensor::{MemoryOrder, Tensor};
//!
//! let mut ctx = CpuContext::new(4);
//! let col = MemoryOrder::ColumnMajor;
//! let mem = LogicalMemorySpace::MainMemory;
//! let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
//! let b = Tensor::<f64>::zeros(&[4, 5], mem, col);
//! let mut c = Tensor::<f64>::zeros(&[3, 5], mem, col);
//!
//! let desc = PrimDescriptor::BatchedGemm {
//!     batch_dims: vec![],
//!     m: 3,
//!     n: 5,
//!     k: 4,
//! };
//! let plan = <CpuBackend as TensorPrims<Standard<f64>>>::plan(
//!     &mut ctx,
//!     &desc,
//!     &[&[3, 4], &[4, 5], &[3, 5]],
//! )
//! .unwrap();
//! <CpuBackend as TensorPrims<Standard<f64>>>::execute(
//!     &mut ctx,
//!     &plan,
//!     1.0,
//!     &[&a, &b],
//!     0.0,
//!     &mut c,
//! )
//! .unwrap();
//! ```
//!
//! ## Reduction (sum over an axis)
//!
//! ```ignore
//! use tenferro_algebra::Standard;
//! use tenferro_device::LogicalMemorySpace;
//! use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, ReduceOp, TensorPrims};
//! use tenferro_tensor::{MemoryOrder, Tensor};
//!
//! let mut ctx = CpuContext::new(4);
//! let col = MemoryOrder::ColumnMajor;
//! let mem = LogicalMemorySpace::MainMemory;
//! let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
//! let mut c = Tensor::<f64>::zeros(&[3], mem, col);
//!
//! let desc = PrimDescriptor::Reduce {
//!     modes_a: vec![0, 1],
//!     modes_c: vec![0],
//!     op: ReduceOp::Sum,
//! };
//! let plan =
//!     <CpuBackend as TensorPrims<Standard<f64>>>::plan(&mut ctx, &desc, &[&[3, 4], &[3]])
//!         .unwrap();
//! <CpuBackend as TensorPrims<Standard<f64>>>::execute(
//!     &mut ctx,
//!     &plan,
//!     1.0,
//!     &[&a],
//!     0.0,
//!     &mut c,
//! )
//! .unwrap();
//! ```
//!
//! ## Contraction (extended operation)
//!
//! ```ignore
//! use tenferro_algebra::Standard;
//! use tenferro_device::LogicalMemorySpace;
//! use tenferro_prims::{CpuBackend, CpuContext, Extension, PrimDescriptor, TensorPrims};
//! use tenferro_tensor::{MemoryOrder, Tensor};
//!
//! let mut ctx = CpuContext::new(4);
//! let col = MemoryOrder::ColumnMajor;
//! let mem = LogicalMemorySpace::MainMemory;
//! let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
//! let b = Tensor::<f64>::zeros(&[4, 5], mem, col);
//! let mut c = Tensor::<f64>::zeros(&[3, 5], mem, col);
//!
//! if <CpuBackend as TensorPrims<Standard<f64>>>::has_extension_for(Extension::Contract) {
//!     let desc = PrimDescriptor::Contract {
//!         modes_a: vec![0, 1],
//!         modes_b: vec![1, 2],
//!         modes_c: vec![0, 2],
//!     };
//!     let plan = <CpuBackend as TensorPrims<Standard<f64>>>::plan(
//!         &mut ctx,
//!         &desc,
//!         &[&[3, 4], &[4, 5], &[3, 5]],
//!     )
//!     .unwrap();
//!     <CpuBackend as TensorPrims<Standard<f64>>>::execute(
//!         &mut ctx,
//!         &plan,
//!         1.0,
//!         &[&a, &b],
//!         0.0,
//!         &mut c,
//!     )
//!     .unwrap();
//! }
//! ```

#[cfg(all(feature = "gemm-faer", feature = "gemm-blas"))]
compile_error!("enable exactly one GEMM backend: gemm-faer or gemm-blas");

#[cfg(all(not(feature = "gemm-faer"), not(feature = "gemm-blas")))]
compile_error!("enable exactly one GEMM backend: gemm-faer or gemm-blas");

#[cfg(all(feature = "provider-src", not(feature = "gemm-blas")))]
compile_error!("provider-src requires gemm-blas");
#[cfg(all(feature = "provider-inject", not(feature = "gemm-blas")))]
compile_error!("provider-inject requires gemm-blas");
#[cfg(all(
    any(
        feature = "src-openblas",
        feature = "src-netlib",
        feature = "src-accelerate",
        feature = "src-r",
        feature = "src-intel-mkl-dynamic-sequential",
        feature = "src-intel-mkl-dynamic-parallel",
        feature = "src-intel-mkl-static-sequential",
        feature = "src-intel-mkl-static-parallel"
    ),
    not(feature = "gemm-blas")
))]
compile_error!("src-* features require gemm-blas and provider-src");

#[cfg(feature = "gemm-blas")]
const _: () = {
    let provider_count =
        (cfg!(feature = "provider-src") as usize) + (cfg!(feature = "provider-inject") as usize);
    assert!(
        provider_count == 1,
        "gemm-blas requires exactly one provider: provider-src or provider-inject"
    );

    let src_count = (cfg!(feature = "src-openblas") as usize)
        + (cfg!(feature = "src-netlib") as usize)
        + (cfg!(feature = "src-accelerate") as usize)
        + (cfg!(feature = "src-r") as usize)
        + (cfg!(feature = "src-intel-mkl-dynamic-sequential") as usize)
        + (cfg!(feature = "src-intel-mkl-dynamic-parallel") as usize)
        + (cfg!(feature = "src-intel-mkl-static-sequential") as usize)
        + (cfg!(feature = "src-intel-mkl-static-parallel") as usize);

    if cfg!(feature = "provider-src") {
        assert!(
            src_count == 1,
            "provider-src requires exactly one src-* feature"
        );
    }
    if cfg!(feature = "provider-inject") {
        assert!(src_count == 0, "provider-inject forbids src-* features");
    }
};

#[cfg(feature = "provider-src")]
extern crate blas_src as _;
#[cfg(feature = "provider-inject")]
extern crate cblas_inject as _;
#[cfg(feature = "provider-src")]
extern crate cblas_src as _;

mod analytic_cpu;
mod analytic_prims;
mod cpu;
mod family_cpu_common;
#[cfg(all(feature = "gemm-blas", feature = "provider-inject"))]
pub mod inject;
mod registry;
mod scalar_cpu;
mod scalar_prims;
mod semiring_core;
mod semiring_fast_path;

// CUDA backend: real implementation when `cuda` feature is enabled,
// otherwise stub types that return errors.
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
mod cuda_ffi;

mod gpu_stubs;

#[doc(hidden)]
pub use analytic_cpu::CpuAnalyticPlan;
pub use analytic_prims::*;
pub use cpu::*;
#[doc(hidden)]
pub use scalar_cpu::CpuScalarPlan;
pub use scalar_prims::*;
pub use semiring_core::*;
pub use semiring_fast_path::*;

#[cfg(feature = "cuda")]
pub use cuda::*;
#[cfg(feature = "cuda")]
pub use cuda_ffi::*;

#[cfg(not(feature = "cuda"))]
pub use gpu_stubs::CudaBackend;
#[cfg(not(feature = "cuda"))]
pub use gpu_stubs::CudaContext;
#[cfg(not(feature = "cuda"))]
pub use gpu_stubs::CudaPlan;

// ROCm stubs are always from gpu_stubs (no real ROCm backend yet)
pub use gpu_stubs::RocmBackend;
pub use gpu_stubs::RocmContext;
pub use gpu_stubs::RocmPlan;

pub use registry::*;

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::hash::Hash;

use tenferro_algebra::{Algebra, Scalar};
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
/// on GPU backends (not yet implemented).
///
/// All variants are supported on the CPU backend for f32, f64,
/// Complex32, and Complex64 scalar types.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    /// Maps to `CUTENSOR_OP_CONJ` on GPU backends (not yet implemented).
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
/// // Check if contraction is available for f64
/// let contract = <CpuBackend as TensorPrims<Standard<f64>>>::has_extension_for(Extension::Contract);
/// assert!(contract);
///
/// // Check if element-wise multiplication is available for f64
/// let ewmul = <CpuBackend as TensorPrims<Standard<f64>>>::has_extension_for(Extension::ElementwiseMul);
/// assert!(ewmul);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Extension {
    /// Fused contraction (permute + GEMM). Maps to `cutensorContract` on GPU
    /// backends (not yet implemented).
    Contract,
    /// Element-wise multiplication. Maps to `cutensorElementwiseBinary` on GPU
    /// backends (not yet implemented).
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PrimDescriptor {
    // ====================================================================
    // Core operations (every backend must implement)
    // ====================================================================
    /// Batched matrix multiplication.
    ///
    /// `C[batch, m, n] = alpha * A[batch, m, k] * B[batch, k, n] + beta * C`
    ///
    /// On [`CpuBackend`], this descriptor currently supports `f32`, `f64`,
    /// `Complex32`, and `Complex64`.
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
    ///
    /// On [`CpuBackend`], [`ReduceOp::Sum`] is generic over all scalar types.
    /// [`ReduceOp::Max`] and [`ReduceOp::Min`] require `f32` or `f64`.
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
    /// strategy). Maps to `cutensorContract` on GPU backends
    /// (not yet implemented).
    ///
    /// Available when `has_extension_for(Extension::Contract)` returns true.
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
    /// Available when `has_extension_for(Extension::ElementwiseMul)` returns true.
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

/// Backend trait for tensor primitive operations, parameterized by algebra `Alg`.
///
/// Provides a cuTENSOR-compatible plan-based execution model for all
/// operations. Core ops (batched_gemm, reduce, trace, permute,
/// anti_trace, anti_diag, elementwise_unary) must be implemented.
/// Extended ops (contract, elementwise_mul) are dynamically queried via
/// [`has_extension_for`](TensorPrims::has_extension_for).
///
/// # Algebra parameterization
///
/// The algebra parameter `Alg` enables extensibility: external crates can
/// implement `TensorPrims<MyAlgebra> for CpuBackend` (orphan rule compatible).
///
/// # Execution context
///
/// All operations receive a `&mut Self::Context` that encapsulates the backend's
/// execution resources:
///
/// - **CPU** (current): thread pool, buffer pool, plan cache
/// - **GPU** (not yet implemented): CUDA stream, device handle, workspace
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
/// let b = Tensor::<f64>::zeros(&[4, 5], mem, col);
/// let mut c = Tensor::<f64>::zeros(&[3, 5], mem, col);
///
/// let desc = PrimDescriptor::BatchedGemm {
///     batch_dims: vec![],
///     m: 3,
///     n: 5,
///     k: 4,
/// };
/// let plan = <CpuBackend as TensorPrims<Standard<f64>>>::plan(
///     &mut ctx,
///     &desc,
///     &[&[3, 4], &[4, 5], &[3, 5]],
/// )
/// .unwrap();
/// <CpuBackend as TensorPrims<Standard<f64>>>::execute(
///     &mut ctx,
///     &plan,
///     1.0,
///     &[&a, &b],
///     0.0,
///     &mut c,
/// )
/// .unwrap();
/// ```
pub trait TensorPrims<Alg: Algebra> {
    /// Backend-specific plan type.
    type Plan;

    /// Backend-specific execution context.
    ///
    /// Encapsulates execution resources (thread pool for CPU; CUDA stream
    /// for GPU -- not yet implemented). Analogous to cuTENSOR's
    /// `cutensorHandle_t`.
    type Context;

    /// Create an execution plan from an operation descriptor.
    ///
    /// The plan pre-computes kernel selection and workspace sizes.
    /// `shapes` contains the shape of each tensor involved in the operation
    /// (inputs first, then output).
    fn plan(
        ctx: &mut Self::Context,
        desc: &PrimDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan>;

    /// Execute a plan with the given tensors and scaling factors.
    ///
    /// Follows the BLAS/cuTENSOR pattern:
    /// `output = alpha * op(inputs) + beta * output`
    ///
    /// Operations receive `Tensor<Alg::Scalar>` directly (PyTorch-aligned).
    /// CPU backends convert to strided views internally; GPU backends
    /// extract device pointers.
    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: Alg::Scalar,
        inputs: &[&Tensor<Alg::Scalar>],
        beta: Alg::Scalar,
        output: &mut Tensor<Alg::Scalar>,
    ) -> Result<()>;

    /// Query whether an extended operation is available for this algebra.
    ///
    /// Returns `true` if the backend supports the given extended operation
    /// for the algebra's scalar type.
    fn has_extension_for(ext: Extension) -> bool;
}

// ===========================================================================
// Plan cache
// ===========================================================================

/// Composite key for plan cache lookup.
///
/// Discriminates plans by plan type ([`TypeId`]), operation descriptor
/// ([`PrimDescriptor`]), and tensor shapes. Two calls that match on all
/// three components will produce identical plans, so the cached plan can
/// be reused.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PlanCacheKey {
    /// Plan type discriminator (e.g., `TypeId::of::<CpuPlan<f64>>()`).
    type_id: TypeId,
    /// Operation descriptor.
    descriptor: PrimDescriptor,
    /// Shapes of all tensor operands (inputs + output).
    shapes: Vec<Vec<usize>>,
}

impl PlanCacheKey {
    /// Build a cache key from plan type, descriptor, and shapes.
    fn new<P: 'static>(desc: &PrimDescriptor, shapes: &[&[usize]]) -> Self {
        Self {
            type_id: TypeId::of::<P>(),
            descriptor: desc.clone(),
            shapes: shapes.iter().map(|s| s.to_vec()).collect(),
        }
    }
}

/// Cache for pre-computed execution plans, keyed by
/// `(TypeId, PrimDescriptor, shapes)`.
///
/// Avoids repeated plan generation when the same operation is executed
/// with the same shapes (e.g., single-tensor einsum steps in a loop).
/// Plans are stored type-erased and downcast on retrieval.
///
/// The cache is generic over plan type `P` — any `Clone + 'static` type
/// can be cached (e.g., `CpuPlan<f64>`, `CudaPlan<f32>`).
///
/// # Cache semantics
///
/// - **Hit**: A call to [`get`](PlanCache::get) with the same plan type,
///   descriptor, and shapes returns a clone of the cached plan.
/// - **Miss**: A call with different shapes, descriptor, or plan type
///   returns `None`. The caller should build a new plan and
///   [`insert`](PlanCache::insert) it.
/// - **Thread safety**: `PlanCache` is not `Sync`; it is owned by a single
///   execution context and accessed via `&mut`.
///
/// # Examples
///
/// ```
/// use tenferro_prims::PlanCache;
///
/// let cache = PlanCache::new();
/// assert_eq!(cache.len(), 0);
/// ```
pub struct PlanCache {
    entries: HashMap<PlanCacheKey, Box<dyn Any>>,
}

impl PlanCache {
    /// Create a new empty plan cache.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Returns the number of cached plans.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the cache contains no plans.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up a cached plan for the given plan type, descriptor, and shapes.
    ///
    /// Returns `Some(plan)` on cache hit, `None` on miss. The plan is
    /// cloned out of the cache so the cache retains its copy.
    pub fn get<P: Clone + 'static>(&self, desc: &PrimDescriptor, shapes: &[&[usize]]) -> Option<P> {
        let key = PlanCacheKey::new::<P>(desc, shapes);
        self.entries
            .get(&key)
            .and_then(|boxed| boxed.downcast_ref::<P>())
            .cloned()
    }

    /// Insert a plan into the cache for the given plan type, descriptor,
    /// and shapes.
    ///
    /// If an entry with the same key already exists, it is replaced.
    pub fn insert<P: Clone + 'static>(
        &mut self,
        desc: &PrimDescriptor,
        shapes: &[&[usize]],
        plan: P,
    ) {
        let key = PlanCacheKey::new::<P>(desc, shapes);
        self.entries.insert(key, Box::new(plan));
    }

    /// Remove all cached plans.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl Default for PlanCache {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Helpers for multi-index iteration
// ===========================================================================

/// Iterate over all index combinations for the given dimensions (column-major order).
pub(crate) fn for_each_index(dims: &[usize], mut f: impl FnMut(&[usize])) {
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

/// Find the position of a mode label in a mode list, returning an error if not found.
pub(crate) fn mode_position(modes: &[u32], label: u32) -> Result<usize> {
    modes
        .iter()
        .position(|&m| m == label)
        .ok_or_else(|| Error::InvalidArgument(format!("mode label {label} not found")))
}

/// Validate that the number of shapes matches expectations for an operation.
pub(crate) fn validate_shape_count(
    shapes: &[&[usize]],
    expected: usize,
    op_name: &str,
) -> Result<()> {
    if shapes.len() != expected {
        return Err(Error::InvalidArgument(format!(
            "{op_name} expects {expected} shapes (got {})",
            shapes.len()
        )));
    }
    Ok(())
}

/// Validate that a shape has the expected rank.
pub(crate) fn validate_rank(shape: &[usize], expected: usize, _operand_name: &str) -> Result<()> {
    if shape.len() != expected {
        return Err(Error::RankMismatch {
            expected,
            got: shape.len(),
        });
    }
    Ok(())
}

/// Validate that a shape exactly matches the expected shape.
pub(crate) fn validate_shape_eq(
    got: &[usize],
    expected: &[usize],
    _operand_name: &str,
) -> Result<()> {
    if got != expected {
        return Err(Error::ShapeMismatch {
            expected: expected.to_vec(),
            got: got.to_vec(),
        });
    }
    Ok(())
}

/// Validate the number of input operands for execute.
pub(crate) fn validate_execute_inputs<T: Scalar>(
    inputs: &[&Tensor<T>],
    expected: usize,
    op_name: &str,
) -> Result<()> {
    if inputs.len() != expected {
        return Err(Error::InvalidArgument(format!(
            "{op_name} expects {expected} input(s) (got {})",
            inputs.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
