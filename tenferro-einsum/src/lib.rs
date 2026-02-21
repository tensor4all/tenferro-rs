//! High-level einsum with N-ary contraction tree optimization.
//!
//! This crate provides Einstein summation notation for [`Tensor`]
//! values. It supports:
//!
//! - **String notation**: `"ij,jk->ik"` (NumPy/PyTorch compatible)
//! - **Parenthesized contraction order**: `"ij,(jk,kl)->il"` to control
//!   pairwise contraction sequence in string notation
//! - **Integer label notation**: omeinsum-rs compatible, using `u32` labels
//! - **N-ary contraction**: Automatic or manual optimization of pairwise
//!   contraction order via [`ContractionTree`]
//! - **Accumulating variants**: [`einsum_into`], [`einsum_with_subscripts_into`],
//!   [`einsum_with_plan_into`] write into a pre-allocated output buffer with
//!   BLAS-style `alpha`/`beta` scaling, avoiding allocation in hot loops
//!
//! # Backend dispatch
//!
//! The backend is passed explicitly as a type parameter `B: TensorPrims<A>`
//! with a mutable context `&mut B::Context`.  This follows Rust idiom of
//! explicit ownership and mutability (no global/thread-local state).
//! The context provides access to the thread pool and plan cache.
//!
//! # Examples
//!
//! ## Common operations
//!
//! ```ignore
//! use tenferro_einsum::einsum;
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::LogicalMemorySpace;
//! use tenferro_prims::{CpuBackend, CpuContext};
//!
//! let col = MemoryOrder::ColumnMajor;
//! let mut ctx = CpuContext::new(4);
//!
//! let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], col).unwrap();
//! let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], col).unwrap();
//! let v = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], col).unwrap();
//!
//! // Matrix multiplication: C = A @ B
//! let c = einsum::<_, _, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None).unwrap();
//!
//! // Trace: tr(A)
//! let tr = einsum::<_, _, CpuBackend>(&mut ctx, "ii->", &[&a], None).unwrap();
//!
//! // Outer product: v_i * v_j -> M_{ij}
//! let outer = einsum::<_, _, CpuBackend>(&mut ctx, "i,j->ij", &[&v, &v], None).unwrap();
//!
//! // Dot product: v . v
//! let dot = einsum::<_, _, CpuBackend>(&mut ctx, "i,i->", &[&v, &v], None).unwrap();
//!
//! // Matrix-vector product: A @ v
//! let mv = einsum::<_, _, CpuBackend>(&mut ctx, "ij,j->i", &[&a, &v], None).unwrap();
//!
//! // Diagonal embedding: vector -> diagonal matrix
//! // v = [1, 2, 3] -> [[1,0,0],[0,2,0],[0,0,3]]
//! let diag = einsum::<_, _, CpuBackend>(&mut ctx, "i->ii", &[&v], None).unwrap();
//! assert_eq!(diag.dims(), &[3, 3]);
//!
//! // Diagonal extraction: matrix -> diagonal vector
//! let d = einsum::<_, _, CpuBackend>(&mut ctx, "ii->i", &[&a], None).unwrap();
//!
//! // Higher-order diagonal: 3D tensor with repeated index
//! // Creates T_{iii} from v_i
//! let t = einsum::<_, _, CpuBackend>(&mut ctx, "i->iii", &[&v], None).unwrap();
//! assert_eq!(t.dims(), &[3, 3, 3]);
//!
//! // Consuming variant: operands are moved, buffers may be reused
//! use tenferro_einsum::einsum_owned;
//! let x = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], col).unwrap();
//! let y = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], col).unwrap();
//! let z = einsum_owned::<_, _, CpuBackend>(&mut ctx, "ij,jk->ik", vec![x, y], None).unwrap();
//! ```
//!
//! ## Batch operations
//!
//! ```ignore
//! // Batched GEMM: 10 independent matrix multiplications in one call
//! // A: (batch=10, m=3, k=4), B: (batch=10, k=4, n=5) -> C: (batch=10, m=3, n=5)
//! let a = Tensor::<f64>::zeros(&[10, 3, 4], LogicalMemorySpace::MainMemory, col);
//! let b = Tensor::<f64>::zeros(&[10, 4, 5], LogicalMemorySpace::MainMemory, col);
//! let c = einsum::<_, _, CpuBackend>(&mut ctx, "bij,bjk->bik", &[&a, &b], None).unwrap();
//! assert_eq!(c.dims(), &[10, 3, 5]);
//!
//! // Multiple batch dimensions: (batch1=2, batch2=3, m, k) x (batch1=2, batch2=3, k, n)
//! let a = Tensor::<f64>::zeros(&[2, 3, 4, 5], LogicalMemorySpace::MainMemory, col);
//! let b = Tensor::<f64>::zeros(&[2, 3, 5, 6], LogicalMemorySpace::MainMemory, col);
//! let c = einsum::<_, _, CpuBackend>(&mut ctx, "abij,abjk->abik", &[&a, &b], None).unwrap();
//! assert_eq!(c.dims(), &[2, 3, 4, 6]);
//!
//! // Broadcast batch: A has batch dim, B is shared across batch
//! // A: (batch=10, m=3, k=4), B: (k=4, n=5) -> C: (batch=10, m=3, n=5)
//! let a = Tensor::<f64>::zeros(&[10, 3, 4], LogicalMemorySpace::MainMemory, col);
//! let b = Tensor::<f64>::zeros(&[4, 5], LogicalMemorySpace::MainMemory, col);
//! let c = einsum::<_, _, CpuBackend>(&mut ctx, "bij,jk->bik", &[&a, &b], None).unwrap();
//! assert_eq!(c.dims(), &[10, 3, 5]);
//! ```
//!
//! ## Integer label notation
//!
//! ```ignore
//! use tenferro_einsum::{einsum_with_subscripts, Subscripts};
//!
//! // Same as "ij,jk->ik" but with integer labels
//! // Useful when indices exceed 52 (a-z, A-Z) or are computed programmatically
//! let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
//! let c = einsum_with_subscripts::<_, _, CpuBackend>(&mut ctx, &subs, &[&a, &b], None).unwrap();
//! ```
//!
//! ## Contraction order control
//!
//! ```ignore
//! // Three matrices: D = A @ B @ C
//! // Parentheses specify: contract B*C first, then A*(BC)
//! let d = einsum::<_, _, CpuBackend>(&mut ctx, "ij,(jk,kl)->il", &[&a, &b, &c], None).unwrap();
//!
//! // Or use ContractionTree for programmatic control
//! use tenferro_einsum::ContractionTree;
//! let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
//! let tree = ContractionTree::from_pairs(
//!     &subs,
//!     &[&[3, 4], &[4, 100], &[100, 5]],
//!     &[(1, 2), (0, 3)],  // B*C first (avoids large intermediate)
//! ).unwrap();
//! let d = einsum_with_plan::<_, _, CpuBackend>(&mut ctx, &tree, &[&a, &b, &c], None).unwrap();
//! ```
//!
//! ## Accumulating into a pre-allocated output
//!
//! ```ignore
//! use tenferro_einsum::{einsum_with_plan_into, ContractionTree, Subscripts};
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::LogicalMemorySpace;
//! use tenferro_prims::{CpuBackend, CpuContext};
//!
//! let col = MemoryOrder::ColumnMajor;
//! let mut ctx = CpuContext::new(4);
//! let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
//! let tree = ContractionTree::optimize(&subs, &[&[3, 4], &[4, 5]]).unwrap();
//! let a = Tensor::<f64>::zeros(&[3, 4], LogicalMemorySpace::MainMemory, col);
//! let b = Tensor::<f64>::zeros(&[4, 5], LogicalMemorySpace::MainMemory, col);
//! let mut c = Tensor::<f64>::zeros(&[3, 5], LogicalMemorySpace::MainMemory, col);
//!
//! // Hot loop: reuse output buffer, zero allocation per iteration
//! for _ in 0..1000 {
//!     // C = 1.0 * (A @ B) + 0.0 * C  (overwrite)
//!     einsum_with_plan_into::<_, _, CpuBackend>(
//!         &mut ctx, &tree, &[&a, &b], 1.0, 0.0, &mut c, None,
//!     ).unwrap();
//! }
//! ```
//!
//! ## GPU async chaining (deferred evaluation)
//!
//! GPU einsum operations return immediately. The result tensor carries a
//! [`CompletionEvent`](tenferro_tensor::CompletionEvent) that tracks the
//! pending accelerator work. Passing this tensor to another einsum chains
//! via GPU stream dependencies — no CPU synchronization until data is
//! accessed from the host.
//!
//! - `wait()` — explicitly blocks until computation completes
//! - `tensor_view()`, `dims()`, `strides()` — implicitly call `wait()`
//! - For CPU tensors, `event` is always `None` (zero overhead)
//!
//! ```ignore
//! use tenferro_einsum::einsum;
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::LogicalMemorySpace;
//! use tenferro_prims::CudaBackend; // future
//!
//! // In production, obtain memory spaces via BackendRegistry (future API).
//! let gpu_mem = LogicalMemorySpace::GpuMemory { device_id: 0 };
//! let col = MemoryOrder::ColumnMajor;
//! let mut gpu_ctx = /* CudaContext from BackendRegistry */;
//!
//! let a = Tensor::<f64>::zeros(&[3, 4], gpu_mem, col);
//! let b = Tensor::<f64>::zeros(&[4, 5], gpu_mem, col);
//!
//! // Both einsum calls submit work to the GPU and return immediately.
//! // The second call detects c's pending event and chains on the stream.
//! let c = einsum::<_, _, CudaBackend>(&mut gpu_ctx, "ij,jk->ik", &[&a, &b], None).unwrap();
//! let d = einsum::<_, _, CudaBackend>(&mut gpu_ctx, "ij,jk->ik", &[&c, &b], None).unwrap();
//!
//! // wait() blocks until GPU computation completes
//! d.wait();
//! ```
//!
//! ## Specifying a compute device
//!
//! ```ignore
//! use tenferro_einsum::einsum;
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::{LogicalMemorySpace, ComputeDevice};
//!
//! let col = MemoryOrder::ColumnMajor;
//! // In production, obtain memory spaces via BackendRegistry (future API).
//! let gpu_mem = LogicalMemorySpace::GpuMemory { device_id: 0 };
//!
//! let mut a = Tensor::<f64>::zeros(&[3, 4], gpu_mem, col);
//! let mut b = Tensor::<f64>::zeros(&[4, 5], gpu_mem, col);
//!
//! // Pin tensors to CUDA device 1 (overrides automatic device selection).
//! // This works when CUDA device 1 can access GpuMemory { device_id: 0 }
//! // (e.g., same physical GPU or NVLink-connected peer).
//! // If the device cannot access the memory space, einsum returns
//! // Err(NoCompatibleComputeDevice). In that case, transfer explicitly:
//! //   let a = a.to_memory_space_async(GpuMemory { device_id: 1 }).unwrap();
//! a.set_preferred_compute_device(Some(ComputeDevice::Cuda { device_id: 1 }));
//! b.set_preferred_compute_device(Some(ComputeDevice::Cuda { device_id: 1 }));
//!
//! // einsum dispatches to the specified CUDA device
//! let c = einsum::<_, _, CudaBackend>(&mut gpu_ctx, "ij,jk->ik", &[&a, &b], None).unwrap();
//!
//! // Clear override — revert to automatic device selection
//! // a.set_preferred_compute_device(None);
//! ```

use std::collections::HashMap;

use chainrules::{AdResult, Differentiable, DualTensor, TrackedTensor};
use tenferro_algebra::{HasAlgebra, Scalar};
use tenferro_device::Result;
use tenferro_prims::TensorPrims;
use tenferro_tensor::Tensor;

/// Einsum subscripts using integer labels (omeinsum-rs compatible).
///
/// Each dimension is represented by a `u32` label. Labels shared across
/// multiple input tensors are contracted (summed over). Labels present
/// only in the output are free indices.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::Subscripts;
///
/// // Matrix multiplication: C_{ik} = Σ_j A_{ij} * B_{jk}
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
/// assert_eq!(subs.inputs.len(), 2);
/// assert_eq!(subs.output, vec![0, 2]);
/// ```
///
/// ```ignore
/// use tenferro_einsum::Subscripts;
///
/// // Parse from string notation
/// let subs = Subscripts::parse("ij,jk->ik").unwrap();
/// assert_eq!(subs.inputs.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct Subscripts {
    /// Index labels for each input tensor.
    pub inputs: Vec<Vec<u32>>,
    /// Index labels for the output tensor.
    pub output: Vec<u32>,
}

impl Subscripts {
    /// Create subscripts from integer label arrays.
    ///
    /// # Arguments
    ///
    /// * `inputs` — Index labels for each input tensor
    /// * `output` — Index labels for the output tensor
    pub fn new(inputs: &[&[u32]], output: &[u32]) -> Self {
        Self {
            inputs: inputs.iter().map(|s| s.to_vec()).collect(),
            output: output.to_vec(),
        }
    }

    /// Parse subscripts from NumPy/PyTorch-style string notation.
    ///
    /// Each character (`a`–`z`, `A`–`Z`) represents a dimension label.
    /// Input tensors are separated by commas, and `->` separates inputs
    /// from the output.
    ///
    /// Parentheses can be used to specify contraction order explicitly.
    /// Grouped operands are contracted first, enabling manual control
    /// over the pairwise contraction sequence without using
    /// [`ContractionTree::from_pairs`].
    ///
    /// # Examples
    ///
    /// - `"ij,jk->ik"` — matrix multiplication
    /// - `"ii->i"` — diagonal extraction
    /// - `"ijk->"` — full contraction (scalar result)
    /// - `"ij,(jk,kl)->il"` — contract B and C first, then with A
    ///
    /// # Errors
    ///
    /// Returns an error if the notation is malformed.
    pub fn parse(notation: &str) -> Result<Self> {
        todo!()
    }
}

/// Contraction tree determining pairwise contraction order for N-ary einsum.
///
/// When contracting more than two tensors, the order in which pairwise
/// contractions are performed significantly affects performance.
/// `ContractionTree` encodes this order as a binary tree.
///
/// # Optimization
///
/// Use [`ContractionTree::optimize`] for automatic cost-based optimization
/// (e.g., greedy algorithm based on tensor sizes), or
/// [`ContractionTree::from_pairs`] for manual specification.
pub struct ContractionTree {
    // Internal representation is private.
    _private: (),
}

impl ContractionTree {
    /// Automatically compute an optimized contraction order.
    ///
    /// Uses a cost-based heuristic (e.g., greedy algorithm) to determine
    /// the pairwise contraction sequence that minimizes total operation count.
    ///
    /// # Arguments
    ///
    /// * `subscripts` — Einsum subscripts for all tensors
    /// * `shapes` — Shape of each input tensor
    ///
    /// # Errors
    ///
    /// Returns an error if subscripts and shapes are inconsistent.
    pub fn optimize(subscripts: &Subscripts, shapes: &[&[usize]]) -> Result<Self> {
        todo!()
    }

    /// Manually build a contraction tree from a pairwise contraction sequence.
    ///
    /// Each pair `(i, j)` specifies which two tensors (or intermediate results)
    /// to contract next. Intermediate results are assigned indices starting
    /// from the number of input tensors.
    ///
    /// # Arguments
    ///
    /// * `subscripts` — Einsum subscripts for all tensors
    /// * `shapes` — Shape of each input tensor
    /// * `pairs` — Ordered list of pairwise contractions
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Three tensors: A[ij] B[jk] C[kl] -> D[il]
    /// // Contract B and C first, then A with the result:
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    /// let shapes = [&[3, 4][..], &[4, 5], &[5, 6]];
    /// let tree = ContractionTree::from_pairs(
    ///     &subs,
    ///     &shapes,
    ///     &[(1, 2), (0, 3)],  // B*C -> T(index=3), then A*T -> D
    /// ).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the pairs do not form a valid contraction sequence.
    pub fn from_pairs(
        subscripts: &Subscripts,
        shapes: &[&[usize]],
        pairs: &[(usize, usize)],
    ) -> Result<Self> {
        todo!()
    }
}

/// Execute einsum using string notation.
///
/// Parses the subscript string, optimizes the contraction order, and
/// executes the contraction. The backend `B` and its context `ctx`
/// are passed explicitly.
///
/// Parentheses in the subscript string specify contraction order
/// explicitly (e.g., `"ij,(jk,kl)->il"` contracts B and C first).
/// Without parentheses, the contraction order is optimized automatically.
///
/// # Arguments
///
/// * `ctx` — Mutable backend context (thread pool, plan cache)
/// * `subscripts` — Einstein summation notation (e.g., `"ij,jk->ik"`)
/// * `operands` — Input tensors
/// * `size_dict` — Optional dimension sizes for output labels not in inputs
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::{CpuBackend, CpuContext};
/// let mut ctx = CpuContext::new(4);
///
/// // Matrix multiplication
/// let c = einsum::<_, _, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None).unwrap();
///
/// // Trace
/// let tr = einsum::<_, _, CpuBackend>(&mut ctx, "ii->", &[&a], None).unwrap();
///
/// // Batch matrix multiplication
/// let c = einsum::<_, _, CpuBackend>(&mut ctx, "bij,bjk->bik", &[&a, &b], None).unwrap();
///
/// // Explicit contraction order: contract B*C first, then A
/// let d = einsum::<_, _, CpuBackend>(&mut ctx, "ij,(jk,kl)->il", &[&a, &b, &c], None).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if the notation is invalid or tensor shapes are
/// incompatible with the subscripts.
pub fn einsum<T, A, B>(
    ctx: &mut B::Context,
    subscripts: &str,
    operands: &[&Tensor<T>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
{
    todo!()
}

/// Execute einsum with pre-built [`Subscripts`].
///
/// Avoids re-parsing the subscript string on each call. Useful when the
/// same contraction pattern is applied to tensors of varying shapes.
///
/// # Errors
///
/// Returns an error if tensor shapes are incompatible with the subscripts.
pub fn einsum_with_subscripts<T, A, B>(
    ctx: &mut B::Context,
    subscripts: &Subscripts,
    operands: &[&Tensor<T>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
{
    todo!()
}

/// Execute einsum with a pre-optimized [`ContractionTree`].
///
/// Avoids both subscript parsing and contraction order optimization.
/// Ideal for hot loops where the same contraction is executed repeatedly
/// on tensors of the same shape.
///
/// # Errors
///
/// Returns an error if the operand shapes do not match those used to
/// build the contraction tree.
pub fn einsum_with_plan<T, A, B>(
    ctx: &mut B::Context,
    tree: &ContractionTree,
    operands: &[&Tensor<T>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
{
    todo!()
}

// ============================================================================
// Consuming variants (take ownership of input tensors for buffer reuse)
// ============================================================================

/// Execute einsum using string notation, consuming the input tensors.
///
/// Takes ownership of the operands, allowing the implementation to reuse
/// their buffers for intermediate results or the final output. This avoids
/// allocation when an operand buffer is already the right shape and layout.
///
/// # Examples
///
/// ```ignore
/// use tenferro_einsum::einsum_owned;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let col = MemoryOrder::ColumnMajor;
/// let mut ctx = CpuContext::new(4);
/// let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], col).unwrap();
/// let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], col).unwrap();
///
/// // `a` and `b` are consumed; their buffers may be reused
/// let c = einsum_owned::<_, _, CpuBackend>(&mut ctx, "ij,jk->ik", vec![a, b], None).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if the notation is invalid or tensor shapes are
/// incompatible with the subscripts.
pub fn einsum_owned<T, A, B>(
    _ctx: &mut B::Context,
    _subscripts: &str,
    _operands: Vec<Tensor<T>>,
    _size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
{
    todo!()
}

/// Execute einsum with pre-built [`Subscripts`], consuming the input tensors.
///
/// Combines the benefits of subscript caching ([`einsum_with_subscripts`])
/// with buffer reuse from owned operands.
///
/// # Errors
///
/// Returns an error if tensor shapes are incompatible with the subscripts.
pub fn einsum_with_subscripts_owned<T, A, B>(
    _ctx: &mut B::Context,
    _subscripts: &Subscripts,
    _operands: Vec<Tensor<T>>,
    _size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
{
    todo!()
}

/// Execute einsum with a pre-optimized [`ContractionTree`], consuming the
/// input tensors.
///
/// Combines the benefits of plan caching ([`einsum_with_plan`]) with
/// buffer reuse from owned operands. Ideal for hot loops where the
/// caller no longer needs the input tensors after contraction.
///
/// # Errors
///
/// Returns an error if the operand shapes do not match those used to
/// build the contraction tree.
pub fn einsum_with_plan_owned<T, A, B>(
    _ctx: &mut B::Context,
    _tree: &ContractionTree,
    _operands: Vec<Tensor<T>>,
    _size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
{
    todo!()
}

// ============================================================================
// Accumulating variants (write into pre-allocated output buffer)
// ============================================================================

/// Execute einsum using string notation, accumulating into an existing output.
///
/// Computes `output = alpha * einsum(operands) + beta * output`, writing
/// the result into the provided output tensor. This avoids allocating a new
/// output buffer on each call, which is critical for hot loops.
///
/// # Arguments
///
/// * `ctx` — Mutable backend context (thread pool, plan cache)
/// * `subscripts` — Einstein summation notation (e.g., `"ij,jk->ik"`)
/// * `operands` — Input tensors
/// * `alpha` — Scaling factor for the einsum result
/// * `beta` — Scaling factor for the existing output contents
/// * `output` — Pre-allocated output tensor (must have correct shape)
/// * `size_dict` — Optional dimension sizes for output labels not in inputs
///
/// # Examples
///
/// ```ignore
/// use tenferro_einsum::einsum_into;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let col = MemoryOrder::ColumnMajor;
/// let mut ctx = CpuContext::new(4);
/// let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], col).unwrap();
/// let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], col).unwrap();
/// let mut c = Tensor::<f64>::zeros(&[2, 2], LogicalMemorySpace::MainMemory, col);
///
/// // Overwrite: C = A @ B
/// einsum_into::<_, _, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], 1.0, 0.0, &mut c, None).unwrap();
///
/// // Accumulate: C += A @ B
/// einsum_into::<_, _, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], 1.0, 1.0, &mut c, None).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if the notation is invalid, tensor shapes are
/// incompatible, or the output shape does not match the expected result.
pub fn einsum_into<T, A, B>(
    ctx: &mut B::Context,
    subscripts: &str,
    operands: &[&Tensor<T>],
    alpha: T,
    beta: T,
    output: &mut Tensor<T>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<()>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
{
    todo!()
}

/// Execute einsum with pre-built [`Subscripts`], accumulating into an existing output.
///
/// Computes `output = alpha * einsum(operands) + beta * output`.
/// Avoids re-parsing the subscript string on each call.
///
/// # Examples
///
/// ```ignore
/// use tenferro_einsum::{einsum_with_subscripts_into, Subscripts};
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(4);
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
/// let mut c = Tensor::<f64>::zeros(&[3, 5], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
///
/// // C = 1.0 * (A @ B) + 0.0 * C
/// einsum_with_subscripts_into::<_, _, CpuBackend>(
///     &mut ctx, &subs, &[&a, &b], 1.0, 0.0, &mut c, None,
/// ).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if tensor shapes are incompatible with the subscripts
/// or the output shape does not match.
pub fn einsum_with_subscripts_into<T, A, B>(
    ctx: &mut B::Context,
    subscripts: &Subscripts,
    operands: &[&Tensor<T>],
    alpha: T,
    beta: T,
    output: &mut Tensor<T>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<()>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
{
    todo!()
}

/// Execute einsum with a pre-optimized [`ContractionTree`], accumulating
/// into an existing output.
///
/// Computes `output = alpha * einsum(operands) + beta * output`.
/// Avoids both subscript parsing and contraction order optimization.
/// This is the fastest variant for hot loops with pre-allocated buffers.
///
/// # Examples
///
/// ```ignore
/// use tenferro_einsum::{einsum_with_plan_into, ContractionTree, Subscripts};
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let col = MemoryOrder::ColumnMajor;
/// let mut ctx = CpuContext::new(4);
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
/// let tree = ContractionTree::optimize(&subs, &[&[3, 4], &[4, 5]]).unwrap();
/// let mut c = Tensor::<f64>::zeros(&[3, 5], LogicalMemorySpace::MainMemory, col);
///
/// // Hot loop: reuse output buffer, no allocation per iteration
/// for _ in 0..1000 {
///     einsum_with_plan_into::<_, _, CpuBackend>(
///         &mut ctx, &tree, &[&a, &b], 1.0, 0.0, &mut c, None,
///     ).unwrap();
/// }
/// ```
///
/// # Errors
///
/// Returns an error if the operand shapes do not match those used to
/// build the contraction tree, or the output shape is incorrect.
pub fn einsum_with_plan_into<T, A, B>(
    ctx: &mut B::Context,
    tree: &ContractionTree,
    operands: &[&Tensor<T>],
    alpha: T,
    beta: T,
    output: &mut Tensor<T>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<()>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
{
    todo!()
}

// ============================================================================
// Automatic differentiation support
// ============================================================================

/// Tracked einsum (reverse-mode AD).
///
/// This is the AD-aware counterpart of [`einsum`]. It records the operation
/// on the reverse-mode tape so that [`chainrules::Tape::pullback`] can
/// compute gradients through it.
///
/// # Examples
///
/// ```ignore
/// use chainrules::Tape;
/// use tenferro_einsum::tracked_einsum;
/// use tenferro_tensor::{MemoryOrder, Tensor};
/// use tenferro_device::LogicalMemorySpace;
///
/// let tape = Tape::<Tensor<f64>>::new();
/// let a = tape.leaf(Tensor::ones(
///     &[2, 3],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// ));
/// let b = tape.leaf(Tensor::ones(
///     &[3, 4],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// ));
/// let c = tracked_einsum::<_, _, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b]).unwrap();
/// let loss = tracked_einsum::<_, _, CpuBackend>(&mut ctx, "ij,ij->", &[&c, &c]).unwrap();
/// let grads = tape.pullback(&loss).unwrap();
/// let _ga = grads.get(a.node_id().unwrap()).unwrap();
/// ```
pub fn tracked_einsum<T, A, B>(
    _ctx: &mut B::Context,
    _subscripts: &str,
    _operands: &[&TrackedTensor<Tensor<T>>],
) -> AdResult<TrackedTensor<Tensor<T>>>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
    Tensor<T>: Differentiable,
{
    todo!()
}

/// Dual einsum (forward-mode JVP propagation).
///
/// This is the AD-aware counterpart of [`einsum`] for forward-mode.
/// It propagates tangent vectors through the einsum operation.
///
/// # Examples
///
/// ```ignore
/// use chainrules::DualTensor;
/// use tenferro_einsum::dual_einsum;
/// use tenferro_tensor::{MemoryOrder, Tensor};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::ones(&[2, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[2, 3], mem, col);
/// let b = Tensor::<f64>::ones(&[3, 4], mem, col);
///
/// let a_dual = DualTensor::with_tangent(a, da).unwrap();
/// let b_dual = DualTensor::new(b);
/// let c_dual = dual_einsum::<_, _, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a_dual, &b_dual]).unwrap();
/// let _tangent = c_dual.tangent();
/// ```
pub fn dual_einsum<T, A, B>(
    _ctx: &mut B::Context,
    _subscripts: &str,
    _operands: &[&DualTensor<Tensor<T>>],
) -> AdResult<DualTensor<Tensor<T>>>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
    Tensor<T>: Differentiable,
{
    todo!()
}

/// Reverse-mode rule (rrule) for einsum without building a global tape.
///
/// Computes the pullback (vector-Jacobian product) for an einsum operation.
/// Returns one gradient tensor per input operand.
///
/// Named after Julia's ChainRules.jl convention.
/// This API is intended for language interop and manual AD.
///
/// # Examples
///
/// ```ignore
/// use tenferro_einsum::einsum_rrule;
/// use tenferro_tensor::{MemoryOrder, Tensor};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::ones(&[2, 3], mem, col);
/// let b = Tensor::<f64>::ones(&[3, 4], mem, col);
/// let grad_c = Tensor::<f64>::ones(&[2, 4], mem, col);
///
/// let grads = einsum_rrule::<_, _, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], &grad_c).unwrap();
/// assert_eq!(grads.len(), 2);
/// ```
pub fn einsum_rrule<T, A, B>(
    _ctx: &mut B::Context,
    _subscripts: &str,
    _operands: &[&Tensor<T>],
    _cotangent: &Tensor<T>,
) -> Result<Vec<Tensor<T>>>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
{
    todo!()
}

/// Forward-mode rule (frule) for einsum without building a global tape.
///
/// Computes the pushforward (Jacobian-vector product) for an einsum operation.
/// Inputs without tangent should use `None`.
///
/// Named after Julia's ChainRules.jl convention.
/// This API is intended for language interop and manual AD.
///
/// # Examples
///
/// ```ignore
/// use tenferro_einsum::einsum_frule;
/// use tenferro_tensor::{MemoryOrder, Tensor};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::ones(&[2, 3], mem, col);
/// let b = Tensor::<f64>::ones(&[3, 4], mem, col);
/// let da = Tensor::<f64>::ones(&[2, 3], mem, col);
///
/// let dc = einsum_frule::<_, _, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], &[Some(&da), None]).unwrap();
/// ```
pub fn einsum_frule<T, A, B>(
    _ctx: &mut B::Context,
    _subscripts: &str,
    _primals: &[&Tensor<T>],
    _tangents: &[Option<&Tensor<T>>],
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
{
    todo!()
}

/// Local HVP rule for einsum without building a global tape.
///
/// Computes the forward-over-reverse Hessian-vector product for an einsum
/// operation. Given primals, their tangents (direction v), an output
/// cotangent ḡ, and its tangent dḡ, returns `(gradient, hvp)` pairs
/// for each input operand.
///
/// For C = einsum(subscripts, [A, B]):
/// - gradient: standard pullback (e.g., ḡ_A = einsum(ḡ_C, B))
/// - hvp: tangent of pullback (e.g., dḡ_A = einsum(dḡ_C, B) + einsum(ḡ_C, dB))
///
/// This API is intended for language interop and manual AD.
///
/// # Examples
///
/// ```ignore
/// use tenferro_einsum::einsum_hvp;
/// use tenferro_tensor::{MemoryOrder, Tensor};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::ones(&[2, 3], mem, col);
/// let b = Tensor::<f64>::ones(&[3, 4], mem, col);
/// let da = Tensor::<f64>::ones(&[2, 3], mem, col);
///
/// let grad_c = Tensor::<f64>::ones(&[2, 4], mem, col);
/// let dgrad_c = Tensor::<f64>::ones(&[2, 4], mem, col);
///
/// let results = einsum_hvp::<_, _, CpuBackend>(
///     &mut ctx,
///     "ij,jk->ik",
///     &[&a, &b],
///     &[Some(&da), None],
///     &grad_c,
///     &dgrad_c,
/// ).unwrap();
/// assert_eq!(results.len(), 2);
/// let (_grad_a, _hvp_a) = &results[0];
/// let (_grad_b, _hvp_b) = &results[1];
/// ```
pub fn einsum_hvp<T, A, B>(
    _ctx: &mut B::Context,
    _subscripts: &str,
    _primals: &[&Tensor<T>],
    _tangents: &[Option<&Tensor<T>>],
    _cotangent: &Tensor<T>,
    _cotangent_tangent: &Tensor<T>,
) -> Result<Vec<(Tensor<T>, Tensor<T>)>>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
{
    todo!()
}
