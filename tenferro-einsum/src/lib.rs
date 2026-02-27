//! High-level einsum with N-ary contraction tree optimization.
//!
//! This crate provides Einstein summation notation for [`Tensor`]
//! values. It supports:
//!
//! - **String notation**: `"ij,jk->ik"` (NumPy/PyTorch compatible)
//! - **Parenthesized notation**: `"ij,(jk,kl)->il"` respects user-specified
//!   contraction order via [`NestedEinsum`] (OMEinsum.jl-compatible)
//! - **Integer label notation**: omeinsum-rs compatible, using `u32` labels
//! - **N-ary contraction**: Automatic or manual optimization of pairwise
//!   contraction order via [`ContractionTree`]
//! - **Accumulating variants**: [`einsum_into`], [`einsum_with_subscripts_into`],
//!   [`einsum_with_plan_into`] write into a pre-allocated output buffer with
//!   BLAS-style `alpha`/`beta` scaling, avoiding allocation in hot loops
//!
//! # Backend dispatch
//!
//! The backend is passed explicitly as a type parameter `Backend: TensorPrims<Alg>`
//! with a mutable context `&mut Backend::Context`.  This follows Rust idiom of
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
//! // Consuming variant: operands are moved (buffer reuse not yet implemented)
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
//!     &[&[3, 4], &[4, 5], &[5, 6]],
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
//! > **Status: Not yet implemented.** GPU backends do not exist yet.
//! > The examples below are aspirational design targets, not working code.
//!
//! GPU einsum operations return immediately. The result tensor carries a
//! [`CompletionEvent`](tenferro_tensor::CompletionEvent) that tracks the
//! pending accelerator work. Passing this tensor to another einsum chains
//! via GPU stream dependencies — no CPU synchronization until data is
//! accessed from the host.
//!
//! - `wait()` — explicitly blocks until computation completes
//! - `dims()`, `strides()` — implicitly call `wait()`
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
//! > **Status: Not yet implemented.** See GPU note above.
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

use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::rc::Rc;

use chainrules::{AdResult, Differentiable, DualTensor, NodeId, ReverseRule, TrackedTensor};
use num_traits::{One, Zero};
use tenferro_algebra::{Algebra, HasAlgebra, Scalar};
use tenferro_device::{Error, LogicalMemorySpace, Result};
use tenferro_prims::{Extension, PrimDescriptor, ReduceOp, TensorPrims};
use tenferro_tensor::{MemoryOrder, Tensor};

// ============================================================================
// Thread-local buffer pool
// ============================================================================

thread_local! {
    static BUFFER_POOL: RefCell<HashMap<TypeId, Box<dyn Any>>>
        = RefCell::new(HashMap::new());
}

const MAX_POOL_PER_TYPE: usize = 16;
const MAX_POOLED_BYTES: usize = 64 * 1024 * 1024; // 64 MB

fn take_from_pool<T: Copy + 'static>(len: usize) -> Vec<T> {
    BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let vecs = pool
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(Vec::<Vec<T>>::new()))
            .downcast_mut::<Vec<Vec<T>>>()
            .unwrap();

        let best_idx = vecs
            .iter()
            .enumerate()
            .filter(|(_, v)| v.capacity() >= len)
            .min_by_key(|(_, v)| v.capacity())
            .map(|(i, _)| i);

        let mut data = best_idx
            .map(|i| vecs.swap_remove(i))
            .unwrap_or_else(|| Vec::with_capacity(len));
        if data.capacity() < len {
            data.reserve(len - data.capacity());
        }
        // Safety: len <= capacity; contents will be overwritten by the
        // backend (beta=0 means the output buffer is fully written).
        unsafe { data.set_len(len) };
        data
    })
}

fn return_to_pool<T: Copy + 'static>(mut data: Vec<T>) {
    let bytes = data.capacity().saturating_mul(std::mem::size_of::<T>());
    if bytes == 0 || bytes > MAX_POOLED_BYTES {
        return;
    }
    data.clear();
    BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let vecs = pool
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(Vec::<Vec<T>>::new()))
            .downcast_mut::<Vec<Vec<T>>>()
            .unwrap();
        if vecs.len() >= MAX_POOL_PER_TYPE {
            if let Some((min_i, min_cap)) = vecs
                .iter()
                .enumerate()
                .map(|(i, v)| (i, v.capacity()))
                .min_by_key(|(_, c)| *c)
            {
                if min_cap < data.capacity() {
                    vecs.swap_remove(min_i);
                    vecs.push(data);
                }
            }
        } else {
            vecs.push(data);
        }
    });
}

/// Allocate a column-major tensor from the thread-local pool, or fresh if too large.
fn alloc_tensor_pooled<T: Scalar>(dims: &[usize], memory_space: LogicalMemorySpace) -> Tensor<T> {
    let numel = dims.iter().product::<usize>().max(1);
    let bytes = numel.saturating_mul(std::mem::size_of::<T>());
    if bytes > MAX_POOLED_BYTES {
        return Tensor::zeros(dims, memory_space, MemoryOrder::ColumnMajor);
    }
    let data = take_from_pool::<T>(numel);
    // Column-major strides: [1, d0, d0*d1, ...]
    let mut strides = Vec::with_capacity(dims.len());
    let mut s = 1isize;
    for &d in dims {
        strides.push(s);
        s *= d as isize;
    }
    Tensor::from_vec(data, dims, &strides, 0)
        .unwrap_or_else(|_| Tensor::zeros(dims, memory_space, MemoryOrder::ColumnMajor))
}

/// Return a tensor's buffer to the thread-local pool (no-op if shared or GPU).
fn return_tensor_to_pool<T: Scalar>(tensor: Tensor<T>) {
    if let Some(data) = tensor.try_into_data_vec() {
        return_to_pool(data);
    }
}

// ============================================================================
// Private helpers
// ============================================================================

/// Infer the common memory space from a set of operand tensors.
///
/// # Allocation policy
///
/// Intermediate and output tensors are allocated on the same memory space
/// as the input operands:
///
/// - If all operands reside on [`LogicalMemorySpace::MainMemory`], the
///   result is `MainMemory`.
/// - If all operands reside on the same GPU memory space (same `device_id`),
///   the result is that GPU memory space.
/// - If operands span different memory spaces (e.g., CPU and GPU, or two
///   different GPU devices), this function returns
///   [`Error::CrossMemorySpaceOperation`] because implicit cross-device
///   data transfer is not supported. The caller must explicitly transfer
///   tensors to a common memory space before calling einsum.
///
/// # Errors
///
/// - Returns [`Error::InvalidArgument`] if `operands` is empty.
/// - Returns [`Error::CrossMemorySpaceOperation`] if operands reside on
///   different memory spaces.
fn infer_memory_space<T: Scalar>(operands: &[&Tensor<T>]) -> Result<LogicalMemorySpace> {
    let first = operands
        .first()
        .ok_or_else(|| Error::InvalidArgument("infer_memory_space: no operands".into()))?;
    let space = first.logical_memory_space();
    for op in &operands[1..] {
        let s = op.logical_memory_space();
        if s != space {
            return Err(Error::CrossMemorySpaceOperation {
                left: space,
                right: s,
            });
        }
    }
    Ok(space)
}

/// Convert a notation label character to internal `u32`.
///
/// Any Unicode scalar except control characters is accepted and mapped to
/// its scalar value (`char as u32`). This enables einsum benchmark instances
/// that use characters like `×`, `ë`, `ð` as index labels.
fn char_to_label(c: char) -> Result<u32> {
    if c.is_control() {
        return Err(Error::InvalidArgument(format!(
            "invalid einsum label character: control character U+{:04X}",
            c as u32
        )));
    }
    Ok(c as u32)
}

/// Split einsum notation on `->` and validate balanced parentheses.
///
/// Returns `(lhs, rhs)` where `lhs` is the input side and `rhs` is the output side.
fn split_and_validate_notation(notation: &str) -> Result<(&str, &str)> {
    let parts: Vec<&str> = notation.split("->").collect();
    if parts.len() != 2 {
        return Err(Error::InvalidArgument(format!(
            "einsum notation must contain exactly one '->', got: {notation}"
        )));
    }
    let lhs = parts[0];
    let rhs = parts[1];

    // Validate balanced parentheses in lhs
    let mut depth: i32 = 0;
    for c in lhs.chars() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth < 0 {
                    return Err(Error::InvalidArgument(format!(
                        "unmatched ')' in einsum notation: {notation}"
                    )));
                }
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err(Error::InvalidArgument(format!(
            "unmatched '(' in einsum notation: {notation}"
        )));
    }

    Ok((lhs, rhs))
}

/// Build a label → size mapping from subscripts and input shapes.
fn build_size_dict(
    subscripts: &Subscripts,
    shapes: &[&[usize]],
    extra: Option<&HashMap<u32, usize>>,
) -> Result<HashMap<u32, usize>> {
    if subscripts.inputs.len() != shapes.len() {
        return Err(Error::InvalidArgument(format!(
            "expected {} input shapes, got {}",
            subscripts.inputs.len(),
            shapes.len()
        )));
    }
    let mut size_dict: HashMap<u32, usize> = HashMap::new();
    for (i, input_subs) in subscripts.inputs.iter().enumerate() {
        if input_subs.len() != shapes[i].len() {
            return Err(Error::InvalidArgument(format!(
                "input {} has {} subscript labels but shape has {} dimensions",
                i,
                input_subs.len(),
                shapes[i].len()
            )));
        }
        for (j, &label) in input_subs.iter().enumerate() {
            let size = shapes[i][j];
            if let Some(&existing) = size_dict.get(&label) {
                if existing != size {
                    return Err(Error::ShapeMismatch {
                        expected: vec![existing],
                        got: vec![size],
                    });
                }
            } else {
                size_dict.insert(label, size);
            }
        }
    }
    if let Some(sd) = extra {
        for (&label, &size) in sd {
            size_dict.entry(label).or_insert(size);
        }
    }
    Ok(size_dict)
}

/// Compute output shape from output subscripts and size dictionary.
fn compute_output_shape(
    output_subs: &[u32],
    size_dict: &HashMap<u32, usize>,
) -> Result<Vec<usize>> {
    output_subs
        .iter()
        .map(|&label| {
            size_dict
                .get(&label)
                .copied()
                .ok_or_else(|| Error::InvalidArgument(format!("unknown size for label {label}")))
        })
        .collect()
}

/// Compute intermediate subscripts when contracting two operands.
/// Keeps labels from left/right that appear in the `needed` set.
fn intermediate_subs(subs_left: &[u32], subs_right: &[u32], needed: &HashSet<u32>) -> Vec<u32> {
    let mut seen = HashSet::new();
    let mut output = Vec::new();
    for &l in subs_left.iter().chain(subs_right.iter()) {
        if needed.contains(&l) && seen.insert(l) {
            output.push(l);
        }
    }
    output
}

/// Compute the cost (output size) of contracting two operands.
fn contraction_cost(
    subs_a: &[u32],
    subs_b: &[u32],
    needed: &HashSet<u32>,
    size_dict: &HashMap<u32, usize>,
) -> usize {
    let out_subs = intermediate_subs(subs_a, subs_b, needed);
    out_subs
        .iter()
        .map(|l| size_dict.get(l).copied().unwrap_or(1))
        .product::<usize>()
        .max(1)
}

/// Read a tensor element at the given multi-index.
fn tensor_get<T: Scalar>(t: &Tensor<T>, indices: &[usize]) -> T {
    let data = t.buffer().as_slice().expect("CPU only");
    let pos = t.offset()
        + indices
            .iter()
            .zip(t.strides())
            .map(|(&i, &s)| i as isize * s)
            .sum::<isize>();
    data[pos as usize]
}

/// Unflatten a flat index into multi-dimensional indices (column-major order).
fn unflatten_index(flat: usize, dims: &[usize]) -> Vec<usize> {
    let ndim = dims.len();
    let mut indices = vec![0usize; ndim];
    let mut remaining = flat;
    for d in 0..ndim {
        if dims[d] > 0 {
            indices[d] = remaining % dims[d];
            remaining /= dims[d];
        }
    }
    indices
}

/// Execute a manual einsum without TensorPrims (for AD pullback).
/// Only supports 1-tensor and 2-tensor contractions.
fn manual_einsum<T: Scalar>(
    subs: &Subscripts,
    operands: &[Tensor<T>],
    size_dict: &HashMap<u32, usize>,
) -> Result<Tensor<T>> {
    let output_shape = compute_output_shape(&subs.output, size_dict)?;
    let n_output: usize = output_shape.iter().product();

    // Collect all unique labels
    let mut all_labels: Vec<u32> = Vec::new();
    let mut all_label_set = HashSet::new();
    for input_subs in &subs.inputs {
        for &l in input_subs {
            if all_label_set.insert(l) {
                all_labels.push(l);
            }
        }
    }
    for &l in &subs.output {
        if all_label_set.insert(l) {
            all_labels.push(l);
        }
    }

    // Build label → size mapping
    let all_dims: Vec<usize> = all_labels
        .iter()
        .map(|l| size_dict.get(l).copied().unwrap_or(1))
        .collect();
    let n_total: usize = all_dims.iter().product();

    // Build label → position in all_labels
    let label_to_pos: HashMap<u32, usize> = all_labels
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, i))
        .collect();

    // Allocate output
    let strides = strided_view::col_major_strides(&output_shape);
    let mut out_data = vec![T::zero(); n_output];

    // Iterate over all index combinations
    for flat in 0..n_total.max(1) {
        let idx = unflatten_index(flat, &all_dims);

        // Compute output position
        let out_idx: Vec<usize> = subs.output.iter().map(|l| idx[label_to_pos[l]]).collect();

        // Compute product of all input elements
        let mut product = T::one();
        for (op_idx, input_subs) in subs.inputs.iter().enumerate() {
            let in_idx: Vec<usize> = input_subs.iter().map(|l| idx[label_to_pos[l]]).collect();
            product = product * tensor_get(&operands[op_idx], &in_idx);
        }

        // Accumulate into output
        let out_pos: isize = out_idx
            .iter()
            .zip(strides.iter())
            .map(|(&i, &s)| i as isize * s)
            .sum();
        if !out_idx.is_empty() {
            out_data[out_pos as usize] = out_data[out_pos as usize] + product;
        } else {
            out_data[0] = out_data[0] + product;
        }
    }

    Tensor::from_slice(&out_data, &output_shape, MemoryOrder::ColumnMajor)
}

// ============================================================================
// Single-tensor einsum execution
// ============================================================================

/// Execute a single-tensor einsum operation via TensorPrims.
fn execute_single_tensor_einsum<Alg, Backend>(
    ctx: &mut Backend::Context,
    subs_a: &[u32],
    subs_c: &[u32],
    input: &Tensor<Alg::Scalar>,
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
) -> Result<()>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    // Count label occurrences in input and output
    let mut label_positions: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, &l) in subs_a.iter().enumerate() {
        label_positions.entry(l).or_default().push(i);
    }
    let repeated_labels: Vec<u32> = label_positions
        .iter()
        .filter(|(_, pos)| pos.len() > 1)
        .map(|(&l, _)| l)
        .collect();

    let mut output_label_counts: HashMap<u32, usize> = HashMap::new();
    for &l in subs_c {
        *output_label_counts.entry(l).or_insert(0) += 1;
    }
    let output_has_repeated = output_label_counts.values().any(|&c| c > 1);

    if repeated_labels.is_empty() && !output_has_repeated {
        // No repeated labels in input or output
        let input_set: HashSet<u32> = subs_a.iter().copied().collect();
        let output_set: HashSet<u32> = subs_c.iter().copied().collect();

        if input_set == output_set {
            // Pure permutation
            let desc = PrimDescriptor::Permute {
                modes_a: subs_a.to_vec(),
                modes_b: subs_c.to_vec(),
            };
            let shapes = [input.dims(), output.dims()];
            let plan = Backend::plan(ctx, &desc, &shapes)?;
            Backend::execute(ctx, &plan, alpha, &[input], beta, output)
        } else if output_set.is_subset(&input_set) {
            // Reduction (sum over labels not in output)
            let desc = PrimDescriptor::Reduce {
                modes_a: subs_a.to_vec(),
                modes_c: subs_c.to_vec(),
                op: ReduceOp::Sum,
            };
            let shapes = [input.dims(), output.dims()];
            let plan = Backend::plan(ctx, &desc, &shapes)?;
            Backend::execute(ctx, &plan, alpha, &[input], beta, output)
        } else {
            Err(Error::InvalidArgument(
                "output labels contain labels not in input".into(),
            ))
        }
    } else if !repeated_labels.is_empty() && !output_has_repeated {
        // Repeated labels in input, unique labels in output
        let repeated_in_output: Vec<u32> = repeated_labels
            .iter()
            .filter(|l| subs_c.contains(l))
            .copied()
            .collect();

        if repeated_in_output.is_empty() {
            // Pure trace: all repeated labels are summed
            // Assign unique internal labels to each input dimension
            let mut unique_modes_a = Vec::new();
            let mut paired = Vec::new();
            let mut einsum_to_internal: HashMap<(u32, usize), u32> = HashMap::new();

            for (i, &l) in subs_a.iter().enumerate() {
                let internal = 1000 + i as u32;
                unique_modes_a.push(internal);
                einsum_to_internal.insert((l, i), internal);
            }

            // Build paired list from repeated labels
            for &l in &repeated_labels {
                let positions = &label_positions[&l];
                for pair in positions.windows(2) {
                    let m1 = einsum_to_internal[&(l, pair[0])];
                    let m2 = einsum_to_internal[&(l, pair[1])];
                    paired.push((m1, m2));
                }
            }

            // Build modes_c: map output labels to internal labels of non-repeated input dims
            let unique_input_labels: HashMap<u32, u32> = subs_a
                .iter()
                .enumerate()
                .filter(|(_, &l)| label_positions[&l].len() == 1)
                .map(|(i, &l)| (l, einsum_to_internal[&(l, i)]))
                .collect();

            let modes_c: Vec<u32> = subs_c
                .iter()
                .map(|&l| {
                    unique_input_labels.get(&l).copied().ok_or_else(|| {
                        Error::InvalidArgument(format!(
                            "output label {l} not found among non-repeated input labels"
                        ))
                    })
                })
                .collect::<Result<_>>()?;

            let desc = PrimDescriptor::Trace {
                modes_a: unique_modes_a,
                modes_c,
                paired,
            };
            let shapes = [input.dims(), output.dims()];
            let plan = Backend::plan(ctx, &desc, &shapes)?;
            Backend::execute(ctx, &plan, alpha, &[input], beta, output)
        } else {
            // Diagonal extraction: repeated labels appear in output
            // Diagonal extraction + copy
            let mut axis_pairs = Vec::new();
            for &l in &repeated_in_output {
                let positions = &label_positions[&l];
                if positions.len() != 2 {
                    return Err(Error::InvalidArgument(format!(
                        "label {} appears {} times in input; only 2-way diagonal supported",
                        l,
                        positions.len()
                    )));
                }
                axis_pairs.push((positions[0], positions[1]));
            }

            // Extract diagonal as a new Tensor (shares buffer via Arc)
            let diag_tensor = input.diagonal(&axis_pairs)?;

            // Build subscripts after diagonal extraction
            let mut used = vec![false; subs_a.len()];
            for &(a, b) in &axis_pairs {
                used[a] = true;
                used[b] = true;
            }
            let mut after_diag_subs: Vec<u32> = Vec::new();
            for (i, &l) in subs_a.iter().enumerate() {
                if !used[i] {
                    after_diag_subs.push(l);
                }
            }
            for &l in &repeated_in_output {
                after_diag_subs.push(l);
            }

            // Check if we need reduction or just permutation
            let after_set: HashSet<u32> = after_diag_subs.iter().copied().collect();
            let output_set: HashSet<u32> = subs_c.iter().copied().collect();
            let to_reduce: HashSet<u32> = after_set.difference(&output_set).copied().collect();

            if to_reduce.is_empty() {
                // Permute from diagonal layout to output layout
                let desc = PrimDescriptor::Permute {
                    modes_a: after_diag_subs,
                    modes_b: subs_c.to_vec(),
                };
                let shapes = [diag_tensor.dims(), output.dims()];
                let plan = Backend::plan(ctx, &desc, &shapes)?;
                Backend::execute(ctx, &plan, alpha, &[&diag_tensor], beta, output)
            } else {
                // Copy diagonal to contiguous temp, then reduce
                let diag_tensor = diag_tensor.contiguous(MemoryOrder::ColumnMajor);
                let desc = PrimDescriptor::Reduce {
                    modes_a: after_diag_subs,
                    modes_c: subs_c.to_vec(),
                    op: ReduceOp::Sum,
                };
                let shapes = [diag_tensor.dims(), output.dims()];
                let plan = Backend::plan(ctx, &desc, &shapes)?;
                Backend::execute(ctx, &plan, alpha, &[&diag_tensor], beta, output)
            }
        }
    } else if repeated_labels.is_empty() && output_has_repeated {
        // Diagonal embedding: "i->ii"
        // Assign unique internal labels to output dimensions
        let mut unique_modes_c = Vec::new();
        let mut paired = Vec::new();
        let mut label_first_internal: HashMap<u32, u32> = HashMap::new();
        let mut next_label: u32 = 1000;

        for &l in subs_c {
            let internal = next_label;
            next_label += 1;
            unique_modes_c.push(internal);

            if let Some(&first) = label_first_internal.get(&l) {
                paired.push((first, internal));
            } else {
                label_first_internal.insert(l, internal);
            }
        }

        // Map input labels to their internal equivalents
        let modes_a: Vec<u32> = subs_a
            .iter()
            .map(|&l| {
                label_first_internal.get(&l).copied().ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "input label {l} not found in output for diagonal embedding"
                    ))
                })
            })
            .collect::<Result<_>>()?;

        let desc = PrimDescriptor::AntiDiag {
            modes_a,
            modes_c: unique_modes_c,
            paired,
        };
        let shapes = [input.dims(), output.dims()];
        let plan = Backend::plan(ctx, &desc, &shapes)?;
        Backend::execute(ctx, &plan, alpha, &[input], beta, output)
    } else {
        // Both input and output have repeated labels — pipeline decomposition.
        //
        // Strategy:
        //   Stage 1: Diagonal extraction — for labels repeated in input AND present in output
        //   Stage 2: Trace/Reduce — for labels still repeated in input (not in output)
        //   Stage 3+4: Delegate — remaining unique-input to unique/repeated-output
        //
        // Each stage delegates to a recursive call that hits a DIFFERENT branch.

        let output_unique_set: HashSet<u32> = subs_c.iter().copied().collect();

        // Labels repeated in input that also appear in the output → diagonal extraction
        let diag_extract_labels: Vec<u32> = repeated_labels
            .iter()
            .filter(|l| output_unique_set.contains(l))
            .copied()
            .collect();

        let mut current = input.clone();
        let mut current_subs: Vec<u32> = subs_a.to_vec();

        // Stage 1: Diagonal extraction
        if !diag_extract_labels.is_empty() {
            let mut axis_pairs = Vec::new();
            for &l in &diag_extract_labels {
                let positions: Vec<usize> = current_subs
                    .iter()
                    .enumerate()
                    .filter(|(_, &s)| s == l)
                    .map(|(i, _)| i)
                    .collect();
                for pair in positions.windows(2) {
                    axis_pairs.push((pair[0], pair[1]));
                }
            }
            current = current.diagonal(&axis_pairs)?;

            // Rebuild subscripts: unused positions first, then one copy per diagonal label
            let mut used = vec![false; current_subs.len()];
            for &(a, b) in &axis_pairs {
                used[a] = true;
                used[b] = true;
            }
            let mut new_subs = Vec::new();
            for (i, &l) in current_subs.iter().enumerate() {
                if !used[i] {
                    new_subs.push(l);
                }
            }
            for &l in &diag_extract_labels {
                new_subs.push(l);
            }
            current_subs = new_subs;
        }

        // Stage 2: Trace/Reduce for labels remaining in input but not in output.
        // After diagonal extraction, some labels may still appear repeated in
        // current_subs (those that were repeated in input but absent from output).
        let output_label_set: HashSet<u32> = subs_c.iter().copied().collect();
        let labels_not_in_output: Vec<u32> = {
            let mut seen = HashSet::new();
            current_subs
                .iter()
                .filter(|l| !output_label_set.contains(l))
                .filter(|l| seen.insert(**l))
                .copied()
                .collect()
        };

        if !labels_not_in_output.is_empty() {
            // Intermediate subscripts: keep only labels that appear in output
            let inter_subs: Vec<u32> = current_subs
                .iter()
                .filter(|l| output_label_set.contains(l))
                .copied()
                .collect();
            // Compute intermediate shape from current tensor's dimensions
            let inter_shape: Vec<usize> = inter_subs
                .iter()
                .map(|l| {
                    let pos = current_subs.iter().position(|s| s == l).ok_or_else(|| {
                        Error::InvalidArgument(format!(
                            "label {l} not found in current subscripts during pipeline decomposition"
                        ))
                    })?;
                    Ok(current.dims()[pos])
                })
                .collect::<Result<_>>()?;
            let mut intermediate =
                alloc_tensor_pooled::<Alg::Scalar>(&inter_shape, output.logical_memory_space());
            // Recursive call for trace/reduce: current_subs → inter_subs
            // inter_subs has no repeated labels, so this hits a different branch.
            execute_single_tensor_einsum::<Alg, Backend>(
                ctx,
                &current_subs,
                &inter_subs,
                &current,
                Alg::Scalar::one(),
                Alg::Scalar::zero(),
                &mut intermediate,
            )?;
            current = intermediate;
            current_subs = inter_subs;
        }

        // Stage 3+4: Now current_subs has unique labels. Recursive call handles
        // permute + AntiDiag for repeated output labels (or just permute/identity).
        execute_single_tensor_einsum::<Alg, Backend>(
            ctx,
            &current_subs,
            subs_c,
            &current,
            alpha,
            beta,
            output,
        )
    }
}

// ============================================================================
// Pairwise contraction execution
// ============================================================================

/// Classify contraction modes into batch, left-only, right-only, and summed.
///
/// - batch: modes in A ∩ B ∩ C (preserved in both inputs and output)
/// - lo (left-only): modes in (A ∩ C) \ B (free modes of A)
/// - ro (right-only): modes in (B ∩ C) \ A (free modes of B)
/// - sum: modes in (A ∩ B) \ C (contracted/summed over)
///
/// Each category preserves the order in which modes first appear in subs_a
/// (for batch, lo, sum) or subs_b (for ro).
fn classify_modes(
    subs_a: &[u32],
    subs_b: &[u32],
    subs_c: &[u32],
) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let set_a: HashSet<u32> = subs_a.iter().copied().collect();
    let set_b: HashSet<u32> = subs_b.iter().copied().collect();
    let set_c: HashSet<u32> = subs_c.iter().copied().collect();

    let mut batch = Vec::new();
    let mut lo = Vec::new();
    let mut sum = Vec::new();
    let mut seen = HashSet::new();

    // Scan A modes: classify as batch, lo, or sum
    for &m in subs_a {
        if !seen.insert(m) {
            continue;
        }
        if set_b.contains(&m) && set_c.contains(&m) {
            batch.push(m);
        } else if set_c.contains(&m) && !set_b.contains(&m) {
            lo.push(m);
        } else if set_b.contains(&m) && !set_c.contains(&m) {
            sum.push(m);
        }
        // mode only in A and not in B or C: ignored (won't appear in output)
    }

    // Scan B modes for right-only
    let mut ro = Vec::new();
    let mut seen_b = HashSet::new();
    for &m in subs_b {
        if !seen_b.insert(m) {
            continue;
        }
        if set_c.contains(&m) && !set_a.contains(&m) {
            ro.push(m);
        }
    }

    (batch, lo, ro, sum)
}

/// Returns true if this binary contraction can be safely lowered by
/// `fallback_pairwise_contraction` (Permute + BatchedGemm pipeline).
///
/// The fallback path requires:
/// - no duplicated labels in each operand/output
/// - output labels to be a subset of input labels
///
/// Reduction-only labels present in only one input are allowed; they are
/// reduced before GEMM decomposition.
fn is_gemm_fallback_compatible(subs_a: &[u32], subs_b: &[u32], subs_c: &[u32]) -> bool {
    let unique = |subs: &[u32]| -> bool {
        let mut seen = HashSet::with_capacity(subs.len());
        subs.iter().all(|&m| seen.insert(m))
    };

    if !unique(subs_a) || !unique(subs_b) || !unique(subs_c) {
        return false;
    }

    let set_a: HashSet<u32> = subs_a.iter().copied().collect();
    let set_b: HashSet<u32> = subs_b.iter().copied().collect();
    let set_c: HashSet<u32> = subs_c.iter().copied().collect();

    // Output labels must come from inputs.
    if !set_c.iter().all(|m| set_a.contains(m) || set_b.contains(m)) {
        return false;
    }

    true
}

/// Try a specialized outer-product lowering via broadcast + ElementwiseMul.
///
/// This path targets disjoint binary einsum (no summed/shared modes), e.g.:
/// `i,j->ij`. It avoids GEMM(k=1) overhead and can be substantially faster.
fn try_outer_elementwise_contraction<A, B>(
    ctx: &mut B::Context,
    subs_a: &[u32],
    subs_b: &[u32],
    subs_c: &[u32],
    a: &Tensor<A::Scalar>,
    b: &Tensor<A::Scalar>,
    alpha: A::Scalar,
    beta: A::Scalar,
    output: &mut Tensor<A::Scalar>,
) -> Result<bool>
where
    A: Algebra,
    A::Scalar: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
{
    if !B::has_extension_for(Extension::ElementwiseMul) {
        return Ok(false);
    }

    let unique = |subs: &[u32]| -> bool {
        let mut seen = HashSet::with_capacity(subs.len());
        subs.iter().all(|&m| seen.insert(m))
    };
    if !unique(subs_a) || !unique(subs_b) || !unique(subs_c) {
        return Ok(false);
    }

    let set_a: HashSet<u32> = subs_a.iter().copied().collect();
    let set_b: HashSet<u32> = subs_b.iter().copied().collect();
    // Must be pure outer: disjoint labels between inputs.
    if set_a.iter().any(|m| set_b.contains(m)) {
        return Ok(false);
    }

    let canonical_modes: Vec<u32> = subs_a.iter().chain(subs_b.iter()).copied().collect();
    let set_c: HashSet<u32> = subs_c.iter().copied().collect();
    if set_c != canonical_modes.iter().copied().collect::<HashSet<_>>() {
        return Ok(false);
    }

    let mut size_dict = HashMap::new();
    for (&m, &d) in subs_a.iter().zip(a.dims()) {
        size_dict.insert(m, d);
    }
    for (&m, &d) in subs_b.iter().zip(b.dims()) {
        size_dict.insert(m, d);
    }

    let canonical_shape: Vec<usize> = canonical_modes.iter().map(|m| size_dict[m]).collect();
    let a_ext_shape: Vec<usize> = canonical_modes
        .iter()
        .map(|m| if set_a.contains(m) { size_dict[m] } else { 1 })
        .collect();
    let b_ext_shape: Vec<usize> = canonical_modes
        .iter()
        .map(|m| if set_b.contains(m) { size_dict[m] } else { 1 })
        .collect();

    let a_reshaped = a.reshape(&a_ext_shape)?;
    let b_reshaped = b.reshape(&b_ext_shape)?;
    let a_bcast = a_reshaped.broadcast(&canonical_shape)?;
    let b_bcast = b_reshaped.broadcast(&canonical_shape)?;

    if canonical_modes == subs_c {
        let desc = PrimDescriptor::ElementwiseMul;
        let shapes = [a_bcast.dims(), b_bcast.dims(), output.dims()];
        let plan = B::plan(ctx, &desc, &shapes)?;
        B::execute(ctx, &plan, alpha, &[&a_bcast, &b_bcast], beta, output)?;
        return Ok(true);
    }

    let memory_space = output.logical_memory_space();
    let mut temp =
        Tensor::<A::Scalar>::zeros(&canonical_shape, memory_space, MemoryOrder::ColumnMajor);
    let desc = PrimDescriptor::ElementwiseMul;
    let shapes = [a_bcast.dims(), b_bcast.dims(), temp.dims()];
    let plan = B::plan(ctx, &desc, &shapes)?;
    B::execute(
        ctx,
        &plan,
        A::Scalar::one(),
        &[&a_bcast, &b_bcast],
        A::Scalar::zero(),
        &mut temp,
    )?;
    let desc = PrimDescriptor::Permute {
        modes_a: canonical_modes,
        modes_b: subs_c.to_vec(),
    };
    let shapes = [temp.dims(), output.dims()];
    let plan = B::plan(ctx, &desc, &shapes)?;
    B::execute(ctx, &plan, alpha, &[&temp], beta, output)?;
    Ok(true)
}

/// Fallback pairwise contraction using core primitives only.
///
/// Decomposes a binary contraction into:
/// 1. Permute A → [batch, lo, sum], B → [batch, sum, ro]
/// 2. MakeContiguous (conditional copy)
/// 3. Reshape to [batch..., m, k] and [batch..., k, n]
/// 4. BatchedGemm → temp [batch..., m, n]
/// 5. Reshape temp → [batch..., lo..., ro...]
/// 6. Permute to output with alpha/beta
fn fallback_pairwise_contraction<A, B>(
    ctx: &mut B::Context,
    subs_a: &[u32],
    subs_b: &[u32],
    subs_c: &[u32],
    a: &Tensor<A::Scalar>,
    b: &Tensor<A::Scalar>,
    alpha: A::Scalar,
    beta: A::Scalar,
    output: &mut Tensor<A::Scalar>,
) -> Result<()>
where
    A: Algebra,
    A::Scalar: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
{
    // Pre-reduce axes that are present only in one input and not in output.
    // This expands GEMM decomposition coverage to general binary contractions.
    let (a_reduced, subs_a_reduced) =
        reduce_unique_only_axes::<A, B>(ctx, a, subs_a, subs_b, subs_c)?;
    let (b_reduced, subs_b_reduced) =
        reduce_unique_only_axes::<A, B>(ctx, b, subs_b, subs_a, subs_c)?;

    let (batch_modes, lo_modes, ro_modes, sum_modes) =
        classify_modes(&subs_a_reduced, &subs_b_reduced, subs_c);

    // Build size_dict from input subscripts and shapes
    let mut size_dict: HashMap<u32, usize> = HashMap::new();
    for (&label, &dim) in subs_a_reduced.iter().zip(a_reduced.dims()) {
        size_dict.insert(label, dim);
    }
    for (&label, &dim) in subs_b_reduced.iter().zip(b_reduced.dims()) {
        size_dict.insert(label, dim);
    }

    let batch_sizes: Vec<usize> = batch_modes.iter().map(|m| size_dict[m]).collect();
    let lo_sizes: Vec<usize> = lo_modes.iter().map(|m| size_dict[m]).collect();
    let ro_sizes: Vec<usize> = ro_modes.iter().map(|m| size_dict[m]).collect();
    let sum_sizes: Vec<usize> = sum_modes.iter().map(|m| size_dict[m]).collect();

    let m: usize = lo_sizes.iter().product::<usize>().max(1);
    let n: usize = ro_sizes.iter().product::<usize>().max(1);
    let k: usize = sum_sizes.iter().product::<usize>().max(1);

    // --- Steps 1-4: Permute, fuse, GEMM ---
    let target_a: Vec<u32> = batch_modes
        .iter()
        .chain(lo_modes.iter())
        .chain(sum_modes.iter())
        .copied()
        .collect();
    let target_b: Vec<u32> = batch_modes
        .iter()
        .chain(sum_modes.iter())
        .chain(ro_modes.iter())
        .copied()
        .collect();
    let c_gemm_shape: Vec<usize> = batch_sizes
        .iter()
        .copied()
        .chain(std::iter::once(m))
        .chain(std::iter::once(n))
        .collect();

    let memory_space = a.logical_memory_space();

    // Try fusability-aware path: permute (metadata-only) + fuse groups → zero-copy GEMM
    let (a_reshaped, b_reshaped) = prepare_gemm_operands::<A, B>(
        ctx,
        &a_reduced,
        &subs_a_reduced,
        &target_a,
        &batch_sizes,
        lo_modes.len(),
        sum_modes.len(),
        &b_reduced,
        &subs_b_reduced,
        &target_b,
        sum_modes.len(),
        ro_modes.len(),
        m,
        n,
        k,
    )?;

    let mut temp = alloc_tensor_pooled::<A::Scalar>(&c_gemm_shape, memory_space);

    let desc = PrimDescriptor::BatchedGemm {
        batch_dims: batch_sizes.clone(),
        m,
        n,
        k,
    };
    let shapes = [a_reshaped.dims(), b_reshaped.dims(), temp.dims()];
    let plan = B::plan(ctx, &desc, &shapes)?;
    B::execute(
        ctx,
        &plan,
        A::Scalar::one(),
        &[&a_reshaped, &b_reshaped],
        A::Scalar::zero(),
        &mut temp,
    )?;

    // --- Step 5: Reshape temp → [batch..., lo..., ro...] and permute to output ---
    let expanded_shape: Vec<usize> = batch_sizes
        .iter()
        .chain(lo_sizes.iter())
        .chain(ro_sizes.iter())
        .copied()
        .collect();
    let temp_expanded = temp.reshape(&expanded_shape)?;

    // Canonical mode labels for the expanded temp
    let canonical_modes: Vec<u32> = batch_modes
        .iter()
        .chain(lo_modes.iter())
        .chain(ro_modes.iter())
        .copied()
        .collect();

    // If canonical == subs_c, we can skip the final permute
    if canonical_modes == subs_c {
        // Direct copy with alpha/beta
        if alpha == A::Scalar::one() && beta == A::Scalar::zero() {
            *output = temp_expanded;
        } else {
            let desc = PrimDescriptor::Permute {
                modes_a: canonical_modes.clone(),
                modes_b: subs_c.to_vec(),
            };
            let shapes = [temp_expanded.dims(), output.dims()];
            let plan = B::plan(ctx, &desc, &shapes)?;
            B::execute(ctx, &plan, alpha, &[&temp_expanded], beta, output)?;
        }
    } else {
        // Permute from canonical order to output order with alpha/beta
        let desc = PrimDescriptor::Permute {
            modes_a: canonical_modes,
            modes_b: subs_c.to_vec(),
        };
        let shapes = [temp_expanded.dims(), output.dims()];
        let plan = B::plan(ctx, &desc, &shapes)?;
        B::execute(ctx, &plan, alpha, &[&temp_expanded], beta, output)?;
    }

    Ok(())
}

/// Reduce axes that are present only in `subs_self` and absent from both
/// `subs_other` and `subs_out`.
fn reduce_unique_only_axes<A, B>(
    ctx: &mut B::Context,
    tensor: &Tensor<A::Scalar>,
    subs_self: &[u32],
    subs_other: &[u32],
    subs_out: &[u32],
) -> Result<(Tensor<A::Scalar>, Vec<u32>)>
where
    A: Algebra,
    A::Scalar: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
{
    let other_set: HashSet<u32> = subs_other.iter().copied().collect();
    let out_set: HashSet<u32> = subs_out.iter().copied().collect();

    let mut reduced_axes = Vec::new();
    let mut kept_subs = Vec::with_capacity(subs_self.len());
    for (ax, &label) in subs_self.iter().enumerate() {
        if !other_set.contains(&label) && !out_set.contains(&label) {
            reduced_axes.push(ax);
        } else {
            kept_subs.push(label);
        }
    }

    if reduced_axes.is_empty() {
        return Ok((tensor.clone(), subs_self.to_vec()));
    }

    let keep_set: HashSet<usize> = reduced_axes.iter().copied().collect();
    let out_shape: Vec<usize> = tensor
        .dims()
        .iter()
        .enumerate()
        .filter_map(|(ax, &d)| {
            if keep_set.contains(&ax) {
                None
            } else {
                Some(d)
            }
        })
        .collect();
    let memory_space = tensor.logical_memory_space();
    let mut reduced = alloc_tensor_pooled::<A::Scalar>(&out_shape, memory_space);

    let desc = PrimDescriptor::Reduce {
        modes_a: subs_self.to_vec(),
        modes_c: kept_subs.clone(),
        op: ReduceOp::Sum,
    };
    let shapes = [tensor.dims(), reduced.dims()];
    let plan = B::plan(ctx, &desc, &shapes)?;
    B::execute(
        ctx,
        &plan,
        A::Scalar::one(),
        &[tensor],
        A::Scalar::zero(),
        &mut reduced,
    )?;

    Ok((reduced, kept_subs))
}

/// Permute a tensor to a target mode order and ensure contiguous layout.
///
/// Prepare two operands for GEMM with fusability-aware zero-copy optimization.
///
/// For each operand: permute to target order, then check if the lo/sum (or sum/ro)
/// dimension groups are fusable via `try_fuse_group`. If fusable, construct a
/// zero-copy view with fused [batch..., M, K] (or [batch..., K, N]) shape.
/// If not fusable, fall back to `permute_or_copy` + `reshape`.
fn prepare_gemm_operands<A, B>(
    ctx: &mut B::Context,
    a: &Tensor<A::Scalar>,
    subs_a: &[u32],
    target_a: &[u32],
    batch_sizes: &[usize],
    n_lo: usize,
    n_sum_a: usize,
    b: &Tensor<A::Scalar>,
    subs_b: &[u32],
    target_b: &[u32],
    n_sum_b: usize,
    n_ro: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(Tensor<A::Scalar>, Tensor<A::Scalar>)>
where
    A: Algebra,
    A::Scalar: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
{
    let nb = batch_sizes.len();
    let a_gemm_shape: Vec<usize> = batch_sizes
        .iter()
        .copied()
        .chain(std::iter::once(m))
        .chain(std::iter::once(k))
        .collect();
    let b_gemm_shape: Vec<usize> = batch_sizes
        .iter()
        .copied()
        .chain(std::iter::once(k))
        .chain(std::iter::once(n))
        .collect();

    let a_prepared =
        prepare_one_operand::<A, B>(ctx, a, subs_a, target_a, nb, n_lo, n_sum_a, &a_gemm_shape)?;
    let b_prepared =
        prepare_one_operand::<A, B>(ctx, b, subs_b, target_b, nb, n_sum_b, n_ro, &b_gemm_shape)?;
    Ok((a_prepared, b_prepared))
}

/// Prepare a single operand for GEMM: permute and try to fuse dimension groups.
fn prepare_one_operand<A, B>(
    ctx: &mut B::Context,
    tensor: &Tensor<A::Scalar>,
    current_subs: &[u32],
    target_subs: &[u32],
    nb: usize,
    n_group1: usize,
    n_group2: usize,
    fallback_shape: &[usize],
) -> Result<Tensor<A::Scalar>>
where
    A: Algebra,
    A::Scalar: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
{
    use strided_perm::try_fuse_group;

    // Step 1: Permute (metadata-only, zero-copy)
    let permuted = if current_subs == target_subs {
        tensor.clone()
    } else {
        let perm = compute_permutation(current_subs, target_subs);
        tensor.permute(&perm)?
    };

    let dims = permuted.dims();
    let strides = permuted.strides();

    // Step 2: Try to fuse each dimension group
    let g1_dims = &dims[nb..nb + n_group1];
    let g1_strides = &strides[nb..nb + n_group1];
    let g2_dims = &dims[nb + n_group1..nb + n_group1 + n_group2];
    let g2_strides = &strides[nb + n_group1..nb + n_group1 + n_group2];

    let fused_g1 = try_fuse_group(g1_dims, g1_strides);
    let fused_g2 = try_fuse_group(g2_dims, g2_strides);

    if let (Some((size1, stride1)), Some((size2, stride2))) = (fused_g1, fused_g2) {
        // Zero-copy: construct fused view [batch..., fused_g1, fused_g2]
        let mut fused_dims = Vec::with_capacity(nb + 2);
        let mut fused_strides = Vec::with_capacity(nb + 2);
        fused_dims.extend_from_slice(&dims[..nb]);
        fused_strides.extend_from_slice(&strides[..nb]);
        fused_dims.push(size1);
        fused_strides.push(stride1);
        fused_dims.push(size2);
        fused_strides.push(stride2);
        return permuted.view_as_strided(fused_dims, fused_strides);
    }

    // Fallback: copy to contiguous, then reshape
    let contiguous = permute_or_copy::<A, B>(ctx, tensor, current_subs, target_subs)?;
    contiguous.reshape(fallback_shape)
}

/// Compute a permutation that reorders `current` labels to match `target` order.
fn compute_permutation(current: &[u32], target: &[u32]) -> Vec<usize> {
    target
        .iter()
        .map(|t| current.iter().position(|c| c == t).unwrap())
        .collect()
}

/// Uses `Tensor::permute` (zero-copy view) first; copies to a pooled buffer
/// only when the result is non-contiguous. Falls back to MakeContiguous when
/// `current_subs == target_subs`.
fn permute_or_copy<A, B>(
    ctx: &mut B::Context,
    tensor: &Tensor<A::Scalar>,
    current_subs: &[u32],
    target_subs: &[u32],
) -> Result<Tensor<A::Scalar>>
where
    A: Algebra,
    A::Scalar: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
{
    if current_subs == target_subs {
        // No permutation needed; ensure contiguous
        return make_contiguous_if_needed::<A, B>(ctx, tensor);
    }

    // Build axis permutation: where does each target label come from?
    let label_to_pos: HashMap<u32, usize> = current_subs
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, i))
        .collect();
    let perm: Vec<usize> = target_subs.iter().map(|l| label_to_pos[l]).collect();

    // Zero-copy view permute
    let view = tensor.permute(&perm)?;

    // If the view happens to be contiguous, return it directly (zero allocation)
    if view.is_contiguous() {
        return Ok(view);
    }

    // Otherwise copy to a contiguous pooled buffer
    let memory_space = tensor.logical_memory_space();
    let mut contiguous = alloc_tensor_pooled::<A::Scalar>(view.dims(), memory_space);
    let desc = PrimDescriptor::MakeContiguous;
    let shapes = [view.dims(), contiguous.dims()];
    let plan = B::plan(ctx, &desc, &shapes)?;
    B::execute(
        ctx,
        &plan,
        A::Scalar::one(),
        &[&view],
        A::Scalar::zero(),
        &mut contiguous,
    )?;
    Ok(contiguous)
}

/// Ensure a tensor is contiguous, copying if necessary.
fn make_contiguous_if_needed<A, B>(
    ctx: &mut B::Context,
    tensor: &Tensor<A::Scalar>,
) -> Result<Tensor<A::Scalar>>
where
    A: Algebra,
    A::Scalar: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
{
    if tensor.is_contiguous() {
        return Ok(tensor.clone());
    }
    let memory_space = tensor.logical_memory_space();
    let mut result = alloc_tensor_pooled::<A::Scalar>(tensor.dims(), memory_space);
    let desc = PrimDescriptor::MakeContiguous;
    let shapes = [tensor.dims(), result.dims()];
    let plan = B::plan(ctx, &desc, &shapes)?;
    B::execute(
        ctx,
        &plan,
        A::Scalar::one(),
        &[tensor],
        A::Scalar::zero(),
        &mut result,
    )?;
    Ok(result)
}

/// Execute a pairwise contraction of two tensors via TensorPrims.
fn execute_pairwise_contraction<Alg, Backend>(
    ctx: &mut Backend::Context,
    subs_a: &[u32],
    subs_b: &[u32],
    subs_c: &[u32],
    a: &Tensor<Alg::Scalar>,
    b: &Tensor<Alg::Scalar>,
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
) -> Result<()>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    // Check for element-wise multiplication (same labels, same order)
    if subs_a == subs_b && subs_a == subs_c && Backend::has_extension_for(Extension::ElementwiseMul)
    {
        let desc = PrimDescriptor::ElementwiseMul;
        let shapes = [a.dims(), b.dims(), output.dims()];
        let plan = Backend::plan(ctx, &desc, &shapes)?;
        return Backend::execute(ctx, &plan, alpha, &[a, b], beta, output);
    }

    // Specialized outer-product path (disjoint binary einsum).
    if try_outer_elementwise_contraction::<Alg, Backend>(
        ctx, subs_a, subs_b, subs_c, a, b, alpha, beta, output,
    )? {
        return Ok(());
    }

    // Prefer GEMM decomposition when labels fit the matrix contraction model.
    // This avoids the generic Contract kernel for common high-throughput cases.
    if is_gemm_fallback_compatible(subs_a, subs_b, subs_c) {
        return fallback_pairwise_contraction::<Alg, Backend>(
            ctx, subs_a, subs_b, subs_c, a, b, alpha, beta, output,
        );
    }

    // General contraction via Contract extension
    if Backend::has_extension_for(Extension::Contract) {
        let desc = PrimDescriptor::Contract {
            modes_a: subs_a.to_vec(),
            modes_b: subs_b.to_vec(),
            modes_c: subs_c.to_vec(),
        };
        let shapes = [a.dims(), b.dims(), output.dims()];
        let plan = Backend::plan(ctx, &desc, &shapes)?;
        Backend::execute(ctx, &plan, alpha, &[a, b], beta, output)
    } else {
        // Contract is disabled in extension dispatch. For patterns that cannot
        // be lowered by fallback_pairwise_contraction, directly try Contract.
        // This keeps compatibility while making GEMM decomposition the default.
        let desc = PrimDescriptor::Contract {
            modes_a: subs_a.to_vec(),
            modes_b: subs_b.to_vec(),
            modes_c: subs_c.to_vec(),
        };
        let shapes = [a.dims(), b.dims(), output.dims()];
        let plan = Backend::plan(ctx, &desc, &shapes)?;
        Backend::execute(ctx, &plan, alpha, &[a, b], beta, output)
    }
}

// ============================================================================
// Step plan pre-computation
// ============================================================================

/// Pre-computed information for reducing axes unique to one operand.
struct ReducePlan {
    /// Subscripts of the operand before reduction.
    original_subs: Vec<u32>,
    /// Subscripts after reduction (labels kept).
    kept_subs: Vec<u32>,
    /// Shape of the tensor after reduction.
    out_shape: Vec<usize>,
}

/// Pre-computed GEMM decomposition plan for a pairwise contraction step.
struct GemmPlan {
    /// Pre-reduction plan for left operand (None if no reduction needed).
    reduce_a: Option<ReducePlan>,
    /// Pre-reduction plan for right operand (None if no reduction needed).
    reduce_b: Option<ReducePlan>,
    /// Subscripts of A after any pre-reduction.
    subs_a: Vec<u32>,
    /// Subscripts of B after any pre-reduction.
    subs_b: Vec<u32>,
    /// Left-only dimension modes.
    lo_modes: Vec<u32>,
    /// Right-only dimension modes.
    ro_modes: Vec<u32>,
    /// Summed (contracted) dimension modes.
    sum_modes: Vec<u32>,
    /// Pre-computed batch dimension sizes.
    batch_sizes: Vec<usize>,
    /// Fused left-only size (product of lo dimensions).
    m: usize,
    /// Fused right-only size (product of ro dimensions).
    n: usize,
    /// Fused summed size (product of sum dimensions).
    k: usize,
    /// Target subscript order for A: [batch..., lo..., sum...].
    target_a: Vec<u32>,
    /// Target subscript order for B: [batch..., sum..., ro...].
    target_b: Vec<u32>,
    /// Shape of GEMM output: [batch..., m, n].
    c_gemm_shape: Vec<usize>,
    /// Expanded shape: [batch..., lo..., ro...].
    expanded_shape: Vec<usize>,
    /// Canonical mode order of expanded output: [batch, lo, ro].
    canonical_modes: Vec<u32>,
    /// Whether a final permute is needed (canonical_modes != subs_c).
    needs_final_permute: bool,
}

/// Pre-computed outer-product plan for a pairwise contraction step.
struct OuterProductPlan {
    /// Canonical mode order: [a_modes..., b_modes...].
    canonical_modes: Vec<u32>,
    /// Shape of A after reshape for broadcast.
    a_ext_shape: Vec<usize>,
    /// Shape of B after reshape for broadcast.
    b_ext_shape: Vec<usize>,
    /// Full canonical shape.
    canonical_shape: Vec<usize>,
    /// Whether a final permute is needed (canonical_modes != subs_c).
    needs_final_permute: bool,
}

/// Pre-computed contraction strategy for one tree step.
enum StepStrategy {
    /// subs_a == subs_b == subs_c: direct ElementwiseMul.
    ElementwiseMul,
    /// Disjoint binary einsum: broadcast + ElementwiseMul.
    OuterProduct(OuterProductPlan),
    /// Matrix contraction: permute + BatchedGemm + permute.
    Gemm(GemmPlan),
    /// General contraction (Contract extension or fallback).
    Contract,
}

/// Pre-computed plan for a single contraction tree step.
struct StepPlan {
    strategy: StepStrategy,
}

/// Pre-compute the reduction plan for axes unique to one operand.
fn compute_reduce_plan(
    subs_self: &[u32],
    subs_other: &[u32],
    subs_out: &[u32],
    size_dict: &HashMap<u32, usize>,
) -> Option<ReducePlan> {
    let other_set: HashSet<u32> = subs_other.iter().copied().collect();
    let out_set: HashSet<u32> = subs_out.iter().copied().collect();

    let mut has_reduction = false;
    let mut kept_subs = Vec::with_capacity(subs_self.len());
    for &label in subs_self {
        if !other_set.contains(&label) && !out_set.contains(&label) {
            has_reduction = true;
        } else {
            kept_subs.push(label);
        }
    }

    if !has_reduction {
        return None;
    }

    let out_shape: Vec<usize> = kept_subs.iter().map(|m| size_dict[m]).collect();
    Some(ReducePlan {
        original_subs: subs_self.to_vec(),
        kept_subs,
        out_shape,
    })
}

/// Compile step plans for all steps in a contraction tree.
///
/// Pre-computes strategy, mode classification, sizes, and target subscripts
/// for each step, eliminating per-step HashMap/HashSet allocations at execution time.
fn compile_step_plans(tree: &ContractionTree) -> Vec<StepPlan> {
    let n_inputs = tree.subscripts.inputs.len();
    let size_dict = &tree.size_dict;

    tree.steps
        .iter()
        .enumerate()
        .map(|(step_idx, step)| {
            let subs_a = &tree.operand_subs[step.left];
            let subs_b = &tree.operand_subs[step.right];
            let is_last = step_idx == tree.steps.len() - 1;
            let subs_c = if is_last {
                &tree.subscripts.output
            } else {
                &tree.operand_subs[n_inputs + step_idx]
            };

            // Check ElementwiseMul: same labels, same order
            if subs_a == subs_b && subs_a.as_slice() == subs_c {
                return StepPlan {
                    strategy: StepStrategy::ElementwiseMul,
                };
            }

            // Check outer product: disjoint labels
            let set_a: HashSet<u32> = subs_a.iter().copied().collect();
            let set_b: HashSet<u32> = subs_b.iter().copied().collect();
            if !set_a.iter().any(|m| set_b.contains(m)) {
                let set_c: HashSet<u32> = subs_c.iter().copied().collect();
                let canonical_modes: Vec<u32> =
                    subs_a.iter().chain(subs_b.iter()).copied().collect();
                let canonical_set: HashSet<u32> = canonical_modes.iter().copied().collect();
                // Unique labels in each operand and output, and output = a ∪ b
                let unique_a = subs_a.len() == set_a.len();
                let unique_b = subs_b.len() == set_b.len();
                let unique_c = subs_c.len() == set_c.len();
                if unique_a && unique_b && unique_c && set_c == canonical_set {
                    let canonical_shape: Vec<usize> =
                        canonical_modes.iter().map(|m| size_dict[m]).collect();
                    let a_ext_shape: Vec<usize> = canonical_modes
                        .iter()
                        .map(|m| if set_a.contains(m) { size_dict[m] } else { 1 })
                        .collect();
                    let b_ext_shape: Vec<usize> = canonical_modes
                        .iter()
                        .map(|m| if set_b.contains(m) { size_dict[m] } else { 1 })
                        .collect();
                    let needs_final_permute = canonical_modes.as_slice() != subs_c;
                    return StepPlan {
                        strategy: StepStrategy::OuterProduct(OuterProductPlan {
                            canonical_modes,
                            a_ext_shape,
                            b_ext_shape,
                            canonical_shape,
                            needs_final_permute,
                        }),
                    };
                }
            }

            // Check GEMM compatibility
            let unique = |subs: &[u32]| -> bool {
                let mut seen = HashSet::with_capacity(subs.len());
                subs.iter().all(|&m| seen.insert(m))
            };
            if unique(subs_a) && unique(subs_b) && unique(subs_c) {
                let set_c: HashSet<u32> = subs_c.iter().copied().collect();
                let set_ab: HashSet<u32> = set_a.union(&set_b).copied().collect();
                if set_c.is_subset(&set_ab) {
                    // GEMM-compatible: pre-compute the full plan
                    let reduce_a = compute_reduce_plan(subs_a, subs_b, subs_c, size_dict);
                    let reduce_b = compute_reduce_plan(subs_b, subs_a, subs_c, size_dict);

                    let effective_a = reduce_a
                        .as_ref()
                        .map(|r| r.kept_subs.clone())
                        .unwrap_or_else(|| subs_a.to_vec());
                    let effective_b = reduce_b
                        .as_ref()
                        .map(|r| r.kept_subs.clone())
                        .unwrap_or_else(|| subs_b.to_vec());

                    let (batch_modes, lo_modes, ro_modes, sum_modes) =
                        classify_modes(&effective_a, &effective_b, subs_c);

                    let batch_sizes: Vec<usize> =
                        batch_modes.iter().map(|m| size_dict[m]).collect();
                    let lo_sizes: Vec<usize> = lo_modes.iter().map(|m| size_dict[m]).collect();
                    let ro_sizes: Vec<usize> = ro_modes.iter().map(|m| size_dict[m]).collect();
                    let sum_sizes: Vec<usize> = sum_modes.iter().map(|m| size_dict[m]).collect();

                    let m = lo_sizes.iter().product::<usize>().max(1);
                    let n = ro_sizes.iter().product::<usize>().max(1);
                    let k = sum_sizes.iter().product::<usize>().max(1);

                    let target_a: Vec<u32> = batch_modes
                        .iter()
                        .chain(lo_modes.iter())
                        .chain(sum_modes.iter())
                        .copied()
                        .collect();
                    let target_b: Vec<u32> = batch_modes
                        .iter()
                        .chain(sum_modes.iter())
                        .chain(ro_modes.iter())
                        .copied()
                        .collect();

                    let c_gemm_shape: Vec<usize> = batch_sizes
                        .iter()
                        .copied()
                        .chain(std::iter::once(m))
                        .chain(std::iter::once(n))
                        .collect();

                    let expanded_shape: Vec<usize> = batch_sizes
                        .iter()
                        .chain(lo_sizes.iter())
                        .chain(ro_sizes.iter())
                        .copied()
                        .collect();

                    let canonical_modes: Vec<u32> = batch_modes
                        .iter()
                        .chain(lo_modes.iter())
                        .chain(ro_modes.iter())
                        .copied()
                        .collect();

                    let needs_final_permute = canonical_modes.as_slice() != subs_c;

                    return StepPlan {
                        strategy: StepStrategy::Gemm(GemmPlan {
                            reduce_a,
                            reduce_b,
                            subs_a: effective_a,
                            subs_b: effective_b,
                            lo_modes,
                            ro_modes,
                            sum_modes,
                            batch_sizes,
                            m,
                            n,
                            k,
                            target_a,
                            target_b,
                            c_gemm_shape,
                            expanded_shape,
                            canonical_modes,
                            needs_final_permute,
                        }),
                    };
                }
            }

            // Fallback: general Contract
            StepPlan {
                strategy: StepStrategy::Contract,
            }
        })
        .collect()
}

/// Execute a pairwise contraction using a pre-computed step plan.
///
/// This avoids per-step HashMap/HashSet allocations by using the pre-computed
/// strategy, mode classification, sizes, and target subscripts.
fn execute_pairwise_with_plan<Alg, Backend>(
    ctx: &mut Backend::Context,
    plan: &StepPlan,
    subs_a: &[u32],
    subs_b: &[u32],
    subs_c: &[u32],
    a: &Tensor<Alg::Scalar>,
    b: &Tensor<Alg::Scalar>,
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
) -> Result<()>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    match &plan.strategy {
        StepStrategy::ElementwiseMul => {
            if Backend::has_extension_for(Extension::ElementwiseMul) {
                let desc = PrimDescriptor::ElementwiseMul;
                let shapes = [a.dims(), b.dims(), output.dims()];
                let prim_plan = Backend::plan(ctx, &desc, &shapes)?;
                Backend::execute(ctx, &prim_plan, alpha, &[a, b], beta, output)
            } else {
                // Fall back to non-plan path
                execute_pairwise_contraction::<Alg, Backend>(
                    ctx, subs_a, subs_b, subs_c, a, b, alpha, beta, output,
                )
            }
        }
        StepStrategy::OuterProduct(op_plan) => {
            if !Backend::has_extension_for(Extension::ElementwiseMul) {
                // Fall back to non-plan path
                return execute_pairwise_contraction::<Alg, Backend>(
                    ctx, subs_a, subs_b, subs_c, a, b, alpha, beta, output,
                );
            }
            execute_outer_with_plan::<Alg, Backend>(ctx, op_plan, subs_c, a, b, alpha, beta, output)
        }
        StepStrategy::Gemm(gemm_plan) => {
            if Backend::has_extension_for(Extension::Contract) {
                // Preferred optimization path: fused Contract
                let desc = PrimDescriptor::Contract {
                    modes_a: subs_a.to_vec(),
                    modes_b: subs_b.to_vec(),
                    modes_c: subs_c.to_vec(),
                };
                let shapes = [a.dims(), b.dims(), output.dims()];
                let prim_plan = Backend::plan(ctx, &desc, &shapes)?;
                Backend::execute(ctx, &prim_plan, alpha, &[a, b], beta, output)
            } else {
                // Fallback: core ops decomposition
                execute_gemm_with_plan::<Alg, Backend>(
                    ctx, gemm_plan, subs_c, a, b, alpha, beta, output,
                )
            }
        }
        StepStrategy::Contract => {
            // Contract extension: direct fused execution
            let desc = PrimDescriptor::Contract {
                modes_a: subs_a.to_vec(),
                modes_b: subs_b.to_vec(),
                modes_c: subs_c.to_vec(),
            };
            let shapes = [a.dims(), b.dims(), output.dims()];
            let prim_plan = Backend::plan(ctx, &desc, &shapes)?;
            Backend::execute(ctx, &prim_plan, alpha, &[a, b], beta, output)
        }
    }
}

/// Execute outer product contraction using a pre-computed plan.
fn execute_outer_with_plan<Alg, Backend>(
    ctx: &mut Backend::Context,
    plan: &OuterProductPlan,
    subs_c: &[u32],
    a: &Tensor<Alg::Scalar>,
    b: &Tensor<Alg::Scalar>,
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
) -> Result<()>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let a_reshaped = a.reshape(&plan.a_ext_shape)?;
    let b_reshaped = b.reshape(&plan.b_ext_shape)?;
    let a_bcast = a_reshaped.broadcast(&plan.canonical_shape)?;
    let b_bcast = b_reshaped.broadcast(&plan.canonical_shape)?;

    if !plan.needs_final_permute {
        let desc = PrimDescriptor::ElementwiseMul;
        let shapes = [a_bcast.dims(), b_bcast.dims(), output.dims()];
        let prim_plan = Backend::plan(ctx, &desc, &shapes)?;
        Backend::execute(ctx, &prim_plan, alpha, &[&a_bcast, &b_bcast], beta, output)
    } else {
        let memory_space = output.logical_memory_space();
        let mut temp = Tensor::<Alg::Scalar>::zeros(
            &plan.canonical_shape,
            memory_space,
            MemoryOrder::ColumnMajor,
        );
        let desc = PrimDescriptor::ElementwiseMul;
        let shapes = [a_bcast.dims(), b_bcast.dims(), temp.dims()];
        let prim_plan = Backend::plan(ctx, &desc, &shapes)?;
        Backend::execute(
            ctx,
            &prim_plan,
            Alg::Scalar::one(),
            &[&a_bcast, &b_bcast],
            Alg::Scalar::zero(),
            &mut temp,
        )?;
        let desc = PrimDescriptor::Permute {
            modes_a: plan.canonical_modes.clone(),
            modes_b: subs_c.to_vec(),
        };
        let shapes = [temp.dims(), output.dims()];
        let prim_plan = Backend::plan(ctx, &desc, &shapes)?;
        Backend::execute(ctx, &prim_plan, alpha, &[&temp], beta, output)
    }
}

/// Execute pre-reduction of unique-only axes using a pre-computed plan.
fn execute_reduce_with_plan<Alg, Backend>(
    ctx: &mut Backend::Context,
    reduce: &ReducePlan,
    tensor: &Tensor<Alg::Scalar>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let memory_space = tensor.logical_memory_space();
    let mut reduced = alloc_tensor_pooled::<Alg::Scalar>(&reduce.out_shape, memory_space);
    let desc = PrimDescriptor::Reduce {
        modes_a: reduce.original_subs.clone(),
        modes_c: reduce.kept_subs.clone(),
        op: ReduceOp::Sum,
    };
    let shapes = [tensor.dims(), reduced.dims()];
    let prim_plan = Backend::plan(ctx, &desc, &shapes)?;
    Backend::execute(
        ctx,
        &prim_plan,
        Alg::Scalar::one(),
        &[tensor],
        Alg::Scalar::zero(),
        &mut reduced,
    )?;
    Ok(reduced)
}

/// Execute GEMM contraction using a pre-computed plan.
fn execute_gemm_with_plan<Alg, Backend>(
    ctx: &mut Backend::Context,
    plan: &GemmPlan,
    subs_c: &[u32],
    a: &Tensor<Alg::Scalar>,
    b: &Tensor<Alg::Scalar>,
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
) -> Result<()>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    // Pre-reduce unique-only axes if needed
    let a_reduced;
    let a_ref = if let Some(ref reduce) = plan.reduce_a {
        a_reduced = execute_reduce_with_plan::<Alg, Backend>(ctx, reduce, a)?;
        &a_reduced
    } else {
        a
    };
    let b_reduced;
    let b_ref = if let Some(ref reduce) = plan.reduce_b {
        b_reduced = execute_reduce_with_plan::<Alg, Backend>(ctx, reduce, b)?;
        &b_reduced
    } else {
        b
    };

    let memory_space = a.logical_memory_space();
    let nb = plan.batch_sizes.len();

    // Prepare GEMM operands with fusability check
    let a_gemm_shape: Vec<usize> = plan
        .batch_sizes
        .iter()
        .copied()
        .chain(std::iter::once(plan.m))
        .chain(std::iter::once(plan.k))
        .collect();
    let b_gemm_shape: Vec<usize> = plan
        .batch_sizes
        .iter()
        .copied()
        .chain(std::iter::once(plan.k))
        .chain(std::iter::once(plan.n))
        .collect();

    let a_prepared = prepare_one_operand::<Alg, Backend>(
        ctx,
        a_ref,
        &plan.subs_a,
        &plan.target_a,
        nb,
        plan.lo_modes.len(),
        plan.sum_modes.len(),
        &a_gemm_shape,
    )?;
    let b_prepared = prepare_one_operand::<Alg, Backend>(
        ctx,
        b_ref,
        &plan.subs_b,
        &plan.target_b,
        nb,
        plan.sum_modes.len(),
        plan.ro_modes.len(),
        &b_gemm_shape,
    )?;

    // Execute GEMM
    let mut temp = alloc_tensor_pooled::<Alg::Scalar>(&plan.c_gemm_shape, memory_space);
    let desc = PrimDescriptor::BatchedGemm {
        batch_dims: plan.batch_sizes.clone(),
        m: plan.m,
        n: plan.n,
        k: plan.k,
    };
    let shapes = [a_prepared.dims(), b_prepared.dims(), temp.dims()];
    let prim_plan = Backend::plan(ctx, &desc, &shapes)?;
    Backend::execute(
        ctx,
        &prim_plan,
        Alg::Scalar::one(),
        &[&a_prepared, &b_prepared],
        Alg::Scalar::zero(),
        &mut temp,
    )?;

    // Reshape and permute to output
    let temp_expanded = temp.reshape(&plan.expanded_shape)?;

    if !plan.needs_final_permute {
        if alpha == Alg::Scalar::one() && beta == Alg::Scalar::zero() {
            *output = temp_expanded;
        } else {
            let desc = PrimDescriptor::Permute {
                modes_a: plan.canonical_modes.clone(),
                modes_b: subs_c.to_vec(),
            };
            let shapes = [temp_expanded.dims(), output.dims()];
            let prim_plan = Backend::plan(ctx, &desc, &shapes)?;
            Backend::execute(ctx, &prim_plan, alpha, &[&temp_expanded], beta, output)?;
        }
    } else {
        let desc = PrimDescriptor::Permute {
            modes_a: plan.canonical_modes.clone(),
            modes_b: subs_c.to_vec(),
        };
        let shapes = [temp_expanded.dims(), output.dims()];
        let prim_plan = Backend::plan(ctx, &desc, &shapes)?;
        Backend::execute(ctx, &prim_plan, alpha, &[&temp_expanded], beta, output)?;
    }

    Ok(())
}

// ============================================================================
// Contraction tree execution
// ============================================================================

/// Execute a ContractionTree against concrete input tensors.
fn execute_tree<Alg, Backend>(
    ctx: &mut Backend::Context,
    tree: &ContractionTree,
    operands: &[&Tensor<Alg::Scalar>],
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
) -> Result<()>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let n_inputs = tree.subscripts.inputs.len();

    if tree.steps.is_empty() {
        // Single-tensor case
        if n_inputs != 1 {
            return Err(Error::InvalidArgument(
                "ContractionTree with no steps requires exactly 1 input".into(),
            ));
        }
        return execute_single_tensor_einsum::<Alg, Backend>(
            ctx,
            &tree.subscripts.inputs[0],
            &tree.subscripts.output,
            operands[0],
            alpha,
            beta,
            output,
        );
    }

    // Multi-tensor case: follow the contraction tree.
    // Pre-compile step plans to avoid per-step HashMap/HashSet allocations.
    let step_plans = compile_step_plans(tree);

    // Use Vec-indexed storage instead of HashMap for O(1) access.
    let memory_space = infer_memory_space(operands)?;
    let total_slots = n_inputs + tree.steps.len();
    let mut intermediates: Vec<Option<Tensor<Alg::Scalar>>> = Vec::with_capacity(total_slots);
    intermediates.resize_with(total_slots, || None);

    // Count remaining uses for each operand/index in the contraction schedule.
    let mut use_counts = vec![0usize; total_slots];
    for step in &tree.steps {
        use_counts[step.left] += 1;
        use_counts[step.right] += 1;
    }

    for (step_idx, step) in tree.steps.iter().enumerate() {
        let left: &Tensor<Alg::Scalar> = if step.left < n_inputs {
            operands[step.left]
        } else {
            intermediates[step.left].as_ref().ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "missing intermediate tensor at index {}",
                    step.left
                ))
            })?
        };
        let right: &Tensor<Alg::Scalar> = if step.right < n_inputs {
            operands[step.right]
        } else {
            intermediates[step.right].as_ref().ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "missing intermediate tensor at index {}",
                    step.right
                ))
            })?
        };

        let subs_left = &tree.operand_subs[step.left];
        let subs_right = &tree.operand_subs[step.right];
        let is_last = step_idx == tree.steps.len() - 1;

        if is_last {
            // Last step: write directly to output with alpha/beta
            execute_pairwise_with_plan::<Alg, Backend>(
                ctx,
                &step_plans[step_idx],
                subs_left,
                subs_right,
                &tree.subscripts.output,
                left,
                right,
                alpha,
                beta,
                output,
            )?;
        } else {
            // Intermediate step: create new tensor with alpha=1, beta=0
            let result_idx = n_inputs + step_idx;
            let subs_result = &tree.operand_subs[result_idx];
            let result_shape = &tree.step_output_shapes[step_idx];
            let mut result = alloc_tensor_pooled::<Alg::Scalar>(result_shape, memory_space);
            execute_pairwise_with_plan::<Alg, Backend>(
                ctx,
                &step_plans[step_idx],
                subs_left,
                subs_right,
                subs_result,
                left,
                right,
                Alg::Scalar::one(),
                Alg::Scalar::zero(),
                &mut result,
            )?;
            intermediates[result_idx] = Some(result);
        }

        // Release consumed intermediates when their last use is complete.
        let mut consumed = [step.left, step.right];
        consumed.sort_unstable();
        for (i, idx) in consumed.iter().enumerate() {
            if i == 1 && consumed[0] == consumed[1] {
                continue;
            }
            use_counts[*idx] = use_counts[*idx].saturating_sub(1);
            if *idx >= n_inputs && use_counts[*idx] == 0 {
                if let Some(t) = intermediates[*idx].take() {
                    return_tensor_to_pool(t);
                }
            }
        }
    }

    Ok(())
}

/// Execute a [`NestedEinsum`] tree recursively (bottom-up).
///
/// Each leaf returns a clone of the corresponding input tensor. Each internal
/// node recursively evaluates its children, then calls
/// [`einsum_with_subscripts`] on the intermediate results.
fn execute_nested<Alg, Backend>(
    ctx: &mut Backend::Context,
    nested: &NestedEinsum,
    operands: &[&Tensor<Alg::Scalar>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    // Validate operand count at the top level
    let n_leaves = nested.count_leaves();
    if operands.len() != n_leaves {
        return Err(Error::InvalidArgument(format!(
            "NestedEinsum expects {n_leaves} operands, got {}",
            operands.len()
        )));
    }
    execute_nested_inner::<Alg, Backend>(ctx, nested, operands, size_dict)
}

/// Recursive inner implementation (no operand count check — done by caller).
fn execute_nested_inner<Alg, Backend>(
    ctx: &mut Backend::Context,
    nested: &NestedEinsum,
    operands: &[&Tensor<Alg::Scalar>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    match nested {
        NestedEinsum::Leaf(idx) => Ok(operands[*idx].clone()),
        NestedEinsum::Node {
            subscripts,
            children,
        } => {
            let intermediates: Vec<Tensor<Alg::Scalar>> = children
                .iter()
                .map(|child| execute_nested_inner::<Alg, Backend>(ctx, child, operands, size_dict))
                .collect::<Result<_>>()?;

            let refs: Vec<&Tensor<Alg::Scalar>> = intermediates.iter().collect();
            einsum_with_subscripts::<Alg, Backend>(ctx, subscripts, &refs, size_dict)
        }
    }
}

// ============================================================================
// Subscripts
// ============================================================================

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
    /// Each Unicode alphanumeric character represents a dimension label.
    /// Labels are mapped to integer IDs via Unicode scalar values (`char as u32`).
    /// Input tensors are separated by commas, and `->` separates inputs
    /// from the output.
    ///
    /// Parentheses in the notation are accepted but stripped during parsing.
    /// To respect parenthesized contraction order, use [`NestedEinsum::parse`]
    /// or pass the parenthesized string directly to [`einsum`].
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
        let (inputs_str, output_str) = split_and_validate_notation(notation)?;

        let output: Vec<u32> = output_str
            .chars()
            .map(char_to_label)
            .collect::<Result<_>>()?;

        // Validate balanced parentheses before stripping
        let mut depth: i32 = 0;
        for c in inputs_str.chars() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth < 0 {
                        return Err(Error::InvalidArgument(format!(
                            "unmatched ')' in einsum notation: {notation}"
                        )));
                    }
                }
                _ => {}
            }
        }
        if depth != 0 {
            return Err(Error::InvalidArgument(format!(
                "unmatched '(' in einsum notation: {notation}"
            )));
        }

        // Strip parentheses and parse input labels
        let clean_inputs = inputs_str.replace(['(', ')'], "");
        let inputs: Vec<Vec<u32>> = clean_inputs
            .split(',')
            .map(|s| s.chars().map(char_to_label).collect::<Result<_>>())
            .collect::<Result<_>>()?;

        Ok(Self { inputs, output })
    }
}

// ============================================================================
// NestedEinsum
// ============================================================================

/// Recursive einsum tree that preserves parenthesized grouping.
///
/// `NestedEinsum` mirrors OMEinsum.jl's `NestedEinsum`: each internal node
/// holds [`Subscripts`] describing how its children are contracted, and leaf
/// nodes reference an original input tensor by index.
///
/// # Construction
///
/// Use [`NestedEinsum::parse`] to build a tree from parenthesized string
/// notation such as `"(ij,jk),kl->il"`.  Without parentheses the result is
/// a flat root node whose children are all leaves.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::NestedEinsum;
///
/// // Flat (no grouping): root with two leaves
/// let flat = NestedEinsum::parse("ij,jk->ik").unwrap();
/// assert!(matches!(flat, NestedEinsum::Node { .. }));
///
/// // Grouped: contract first two operands, then with third
/// let grouped = NestedEinsum::parse("(ij,jk),kl->il").unwrap();
/// assert!(matches!(grouped, NestedEinsum::Node { .. }));
/// ```
#[derive(Debug, Clone)]
pub enum NestedEinsum {
    /// A leaf referencing one of the original input tensors by index.
    Leaf(usize),
    /// An internal node that contracts its children according to `subscripts`.
    Node {
        /// The subscripts for this contraction: one input per child, plus output.
        subscripts: Subscripts,
        /// Child sub-expressions (leaves or further nodes).
        children: Vec<NestedEinsum>,
    },
}

impl NestedEinsum {
    /// Count the total number of leaf operands in the tree.
    pub fn count_leaves(&self) -> usize {
        match self {
            Self::Leaf(_) => 1,
            Self::Node { children, .. } => children.iter().map(|c| c.count_leaves()).sum(),
        }
    }

    /// Parse parenthesized einsum notation into a recursive tree.
    ///
    /// Notation follows the standard `"inputs->output"` format with optional
    /// parentheses to specify contraction order. Each parenthesized group
    /// becomes an internal [`NestedEinsum::Node`]; bare operands become
    /// [`NestedEinsum::Leaf`] nodes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::NestedEinsum;
    ///
    /// let nested = NestedEinsum::parse("(ij,jk),kl->il").unwrap();
    /// // Root has two children: a group node and a leaf
    /// match &nested {
    ///     NestedEinsum::Node { children, .. } => assert_eq!(children.len(), 2),
    ///     _ => panic!("expected Node"),
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if parentheses are mismatched or the notation is
    /// otherwise malformed.
    pub fn parse(notation: &str) -> Result<Self> {
        let (lhs, output_str) = split_and_validate_notation(notation)?;

        let output: Vec<u32> = output_str
            .chars()
            .map(char_to_label)
            .collect::<Result<_>>()?;

        let mut leaf_counter: usize = 0;
        let outer_needed: HashSet<u32> = output.iter().copied().collect();
        Self::parse_group(lhs, &outer_needed, &output, &mut leaf_counter)
    }

    /// Recursively parse a group (possibly containing sub-groups) into a Node.
    ///
    /// `group_str` is a comma-separated list of items (at the top level),
    /// where each item is either a bare operand (e.g. `"ij"`) or a
    /// parenthesized sub-group (e.g. `"(ij,jk)"`).
    ///
    /// `outer_needed` contains labels that the parent or siblings need from
    /// this group.  `final_output` is the overall output of the entire
    /// expression.
    fn parse_group(
        group_str: &str,
        outer_needed: &HashSet<u32>,
        final_output: &[u32],
        leaf_counter: &mut usize,
    ) -> Result<Self> {
        let items = Self::split_top_level(group_str)?;

        let mut children = Vec::with_capacity(items.len());
        let mut child_subscript_inputs: Vec<Vec<u32>> = Vec::with_capacity(items.len());

        for (idx, item) in items.iter().enumerate() {
            if item.starts_with('(') && item.ends_with(')') {
                // Sub-group: strip outer parens and recurse
                let inner = &item[1..item.len() - 1];

                // Compute what this sub-group needs to output:
                // labels in this group that appear in outer_needed or in sibling items
                let group_labels = Self::collect_labels(inner)?;
                let sibling_labels = Self::collect_sibling_labels(&items, idx)?;
                let mut needed: HashSet<u32> = HashSet::new();
                for &label in &group_labels {
                    if outer_needed.contains(&label) || sibling_labels.contains(&label) {
                        needed.insert(label);
                    }
                }
                let mut sub_output: Vec<u32> = needed.iter().copied().collect();
                sub_output.sort();

                let child = Self::parse_group(inner, &needed, &sub_output, leaf_counter)?;
                child_subscript_inputs.push(sub_output);
                children.push(child);
            } else {
                // Bare operand -> Leaf
                let labels: Vec<u32> = item.chars().map(char_to_label).collect::<Result<_>>()?;
                child_subscript_inputs.push(labels);
                children.push(NestedEinsum::Leaf(*leaf_counter));
                *leaf_counter += 1;
            }
        }

        // Build subscripts for this node
        let node_output: Vec<u32> = final_output.to_vec();
        let subscripts = Subscripts {
            inputs: child_subscript_inputs,
            output: node_output,
        };

        Ok(NestedEinsum::Node {
            subscripts,
            children,
        })
    }

    /// Split a string on commas at the top level (depth 0), respecting parentheses.
    fn split_top_level(s: &str) -> Result<Vec<&str>> {
        let mut items = Vec::new();
        let mut depth: usize = 0;
        let mut start = 0;

        for (pos, c) in s.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    if depth == 0 {
                        return Err(Error::InvalidArgument(format!(
                            "unmatched ')' in einsum group: {s}"
                        )));
                    }
                    depth -= 1;
                }
                ',' if depth == 0 => {
                    items.push(&s[start..pos]);
                    start = pos + 1; // skip the comma
                }
                _ => {}
            }
        }
        // Push the last item
        items.push(&s[start..]);
        Ok(items)
    }

    /// Collect all unique labels from a (possibly nested) string, ignoring
    /// parentheses and commas.
    fn collect_labels(s: &str) -> Result<HashSet<u32>> {
        let mut labels = HashSet::new();
        for c in s.chars() {
            match c {
                '(' | ')' | ',' => continue,
                _ => {
                    labels.insert(char_to_label(c)?);
                }
            }
        }
        Ok(labels)
    }

    /// Collect all labels from sibling items (all items except the one at `current_idx`).
    fn collect_sibling_labels(items: &[&str], current_idx: usize) -> Result<HashSet<u32>> {
        let mut labels = HashSet::new();
        for (idx, item) in items.iter().enumerate() {
            if idx == current_idx {
                continue;
            }
            for label in Self::collect_labels(item)? {
                labels.insert(label);
            }
        }
        Ok(labels)
    }
}

// ============================================================================
// ContractionTree
// ============================================================================

/// A single step in the contraction sequence.
struct ContractionStep {
    left: usize,
    right: usize,
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
    /// Original subscripts.
    subscripts: Subscripts,
    /// Steps in the contraction (empty for single-tensor case).
    steps: Vec<ContractionStep>,
    /// Label → dimension size mapping.
    size_dict: HashMap<u32, usize>,
    /// Subscripts for each operand (0..n_inputs from input, then intermediates).
    operand_subs: Vec<Vec<u32>>,
    /// Pre-computed output shapes for each intermediate step (indexed by step_idx).
    step_output_shapes: Vec<Vec<usize>>,
}

impl ContractionTree {
    /// Automatically compute an optimized contraction order.
    ///
    /// Uses a cost-based heuristic (greedy algorithm) to determine
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
        let n_inputs = subscripts.inputs.len();
        if n_inputs <= 1 {
            return Self::from_pairs(subscripts, shapes, &[]);
        }

        let size_dict = build_size_dict(subscripts, shapes, None)?;
        let mut available: Vec<usize> = (0..n_inputs).collect();
        let mut operand_subs: Vec<Vec<u32>> = subscripts.inputs.clone();
        let mut pairs: Vec<(usize, usize)> = Vec::new();

        while available.len() > 1 {
            // Compute labels needed by remaining operands and final output
            let mut best_i = 0;
            let mut best_j = 1;
            let mut best_cost = usize::MAX;

            for i in 0..available.len() {
                for j in (i + 1)..available.len() {
                    let li = available[i];
                    let lj = available[j];
                    // Labels needed by remaining operands (excluding this pair) + final output
                    let mut needed = HashSet::new();
                    needed.extend(subscripts.output.iter().copied());
                    for &idx in &available {
                        if idx != li && idx != lj {
                            needed.extend(operand_subs[idx].iter().copied());
                        }
                    }
                    let cost =
                        contraction_cost(&operand_subs[li], &operand_subs[lj], &needed, &size_dict);
                    if cost < best_cost {
                        best_cost = cost;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            let left = available[best_i];
            let right = available[best_j];
            pairs.push((left, right));

            // Compute intermediate subscripts
            let mut needed = HashSet::new();
            needed.extend(subscripts.output.iter().copied());
            for &idx in &available {
                if idx != left && idx != right {
                    needed.extend(operand_subs[idx].iter().copied());
                }
            }
            let new_subs = intermediate_subs(&operand_subs[left], &operand_subs[right], &needed);
            let new_idx = operand_subs.len();
            operand_subs.push(new_subs);

            // Remove consumed (higher index first), add intermediate
            available.remove(best_j);
            available.remove(best_i);
            available.push(new_idx);
        }

        Self::from_pairs(subscripts, shapes, &pairs)
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
        let n_inputs = subscripts.inputs.len();
        let size_dict = build_size_dict(subscripts, shapes, None)?;

        let mut operand_subs: Vec<Vec<u32>> = subscripts.inputs.clone();
        let mut consumed = vec![false; n_inputs + pairs.len()];
        let mut steps = Vec::new();

        for &(left, right) in pairs {
            if left >= operand_subs.len() || right >= operand_subs.len() {
                return Err(Error::InvalidArgument(format!(
                    "pair ({left}, {right}) references non-existent operand"
                )));
            }
            consumed[left] = true;
            consumed[right] = true;

            // Labels needed by unconsumed operands + final output
            let mut needed: HashSet<u32> = subscripts.output.iter().copied().collect();
            for (idx, subs) in operand_subs.iter().enumerate() {
                if !consumed[idx] {
                    needed.extend(subs.iter().copied());
                }
            }

            let new_subs = intermediate_subs(&operand_subs[left], &operand_subs[right], &needed);
            operand_subs.push(new_subs);
            steps.push(ContractionStep { left, right });
        }

        // Pre-compute output shapes for each intermediate step.
        let step_output_shapes: Vec<Vec<usize>> = (0..steps.len())
            .map(|step_idx| {
                let result_idx = n_inputs + step_idx;
                compute_output_shape(&operand_subs[result_idx], &size_dict)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            subscripts: subscripts.clone(),
            steps,
            size_dict,
            operand_subs,
            step_output_shapes,
        })
    }
}

// ============================================================================
// Public einsum functions (borrowing variants)
// ============================================================================

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
pub fn einsum<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &str,
    operands: &[&Tensor<Alg::Scalar>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    // Subscripts::parse strips parentheses, giving the flat form needed
    // by both the flat execution path and the frule tangent propagation.
    let subs = Subscripts::parse(subscripts)?;
    let nested = if subscripts.contains('(') {
        Some(NestedEinsum::parse(subscripts)?)
    } else {
        None
    };

    let mut output = if let Some(ref nested) = nested {
        execute_nested::<Alg, Backend>(ctx, nested, operands, size_dict)?
    } else {
        einsum_with_subscripts::<Alg, Backend>(ctx, &subs, operands, size_dict)?
    };

    // Auto-propagate forward-mode tangents, respecting contraction order.
    if operands.iter().any(|t| t.has_fw_grad()) {
        let tangents: Vec<Option<&Tensor<Alg::Scalar>>> =
            operands.iter().map(|t| t.fw_grad()).collect();
        if let Ok(output_tangent) =
            einsum_frule_impl::<Alg, Backend>(ctx, &subs, nested.as_ref(), operands, &tangents)
        {
            output.set_fw_grad(output_tangent);
        }
    }

    Ok(output)
}

/// Execute einsum with pre-built [`Subscripts`].
///
/// Avoids re-parsing the subscript string on each call. Useful when the
/// same contraction pattern is applied to tensors of varying shapes.
///
/// # Errors
///
/// Returns an error if tensor shapes are incompatible with the subscripts.
pub fn einsum_with_subscripts<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &Subscripts,
    operands: &[&Tensor<Alg::Scalar>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let shapes: Vec<&[usize]> = operands.iter().map(|t| t.dims()).collect();
    let tree = ContractionTree::optimize(subscripts, &shapes)?;
    einsum_with_plan::<Alg, Backend>(ctx, &tree, operands, size_dict)
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
pub fn einsum_with_plan<Alg, Backend>(
    ctx: &mut Backend::Context,
    tree: &ContractionTree,
    operands: &[&Tensor<Alg::Scalar>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    // Merge size_dict from tree and optional extra
    let mut sd = tree.size_dict.clone();
    if let Some(extra) = size_dict {
        for (&k, &v) in extra {
            sd.insert(k, v);
        }
    }
    let output_shape = compute_output_shape(&tree.subscripts.output, &sd)?;
    // Allocate the output tensor on the same memory space as the operands.
    let memory_space = infer_memory_space(operands)?;
    let mut output =
        Tensor::<Alg::Scalar>::zeros(&output_shape, memory_space, MemoryOrder::ColumnMajor);
    execute_tree::<Alg, Backend>(
        ctx,
        tree,
        operands,
        Alg::Scalar::one(),
        Alg::Scalar::zero(),
        &mut output,
    )?;
    Ok(output)
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
/// **Note:** Buffer reuse is not yet implemented. The owned variants
/// currently delegate to the borrowed API.
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
pub fn einsum_owned<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &str,
    operands: Vec<Tensor<Alg::Scalar>>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let refs: Vec<&Tensor<Alg::Scalar>> = operands.iter().collect();
    einsum::<Alg, Backend>(ctx, subscripts, &refs, size_dict)
}

/// Execute einsum with pre-built [`Subscripts`], consuming the input tensors.
///
/// Combines the benefits of subscript caching ([`einsum_with_subscripts`])
/// with buffer reuse from owned operands.
///
/// # Errors
///
/// Returns an error if tensor shapes are incompatible with the subscripts.
pub fn einsum_with_subscripts_owned<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &Subscripts,
    operands: Vec<Tensor<Alg::Scalar>>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let refs: Vec<&Tensor<Alg::Scalar>> = operands.iter().collect();
    einsum_with_subscripts::<Alg, Backend>(ctx, subscripts, &refs, size_dict)
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
pub fn einsum_with_plan_owned<Alg, Backend>(
    ctx: &mut Backend::Context,
    tree: &ContractionTree,
    operands: Vec<Tensor<Alg::Scalar>>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let refs: Vec<&Tensor<Alg::Scalar>> = operands.iter().collect();
    einsum_with_plan::<Alg, Backend>(ctx, tree, &refs, size_dict)
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
pub fn einsum_into<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &str,
    operands: &[&Tensor<Alg::Scalar>],
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<()>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let subs = Subscripts::parse(subscripts)?;
    if subscripts.contains('(') {
        let nested = NestedEinsum::parse(subscripts)?;
        let result = execute_nested::<Alg, Backend>(ctx, &nested, operands, size_dict)?;
        // Apply output = alpha * result + beta * output via identity einsum
        let identity_subs = Subscripts {
            inputs: vec![subs.output.clone()],
            output: subs.output,
        };
        einsum_with_subscripts_into::<Alg, Backend>(
            ctx,
            &identity_subs,
            &[&result],
            alpha,
            beta,
            output,
            size_dict,
        )
    } else {
        einsum_with_subscripts_into::<Alg, Backend>(
            ctx, &subs, operands, alpha, beta, output, size_dict,
        )
    }
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
pub fn einsum_with_subscripts_into<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &Subscripts,
    operands: &[&Tensor<Alg::Scalar>],
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<()>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let shapes: Vec<&[usize]> = operands.iter().map(|t| t.dims()).collect();
    let tree = ContractionTree::optimize(subscripts, &shapes)?;
    einsum_with_plan_into::<Alg, Backend>(ctx, &tree, operands, alpha, beta, output, size_dict)
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
pub fn einsum_with_plan_into<Alg, Backend>(
    ctx: &mut Backend::Context,
    tree: &ContractionTree,
    operands: &[&Tensor<Alg::Scalar>],
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<()>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let _ = size_dict; // size_dict already captured in tree.size_dict
    execute_tree::<Alg, Backend>(ctx, tree, operands, alpha, beta, output)
}

// ============================================================================
// Automatic differentiation support
// ============================================================================

/// ReverseRule for einsum — stores subscripts, primal tensors, and shared
/// backend context for backend-optimized pullback.
struct EinsumReverseRule<Alg, Backend>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
    Tensor<Alg::Scalar>: Differentiable<Tangent = Tensor<Alg::Scalar>>,
{
    ctx: Rc<RefCell<Backend::Context>>,
    subscripts: Subscripts,
    primals: Vec<Tensor<Alg::Scalar>>,
    input_node_ids: Vec<Option<NodeId>>,
    _phantom: PhantomData<Alg>,
}

impl<Alg, Backend> ReverseRule<Tensor<Alg::Scalar>> for EinsumReverseRule<Alg, Backend>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
    Tensor<Alg::Scalar>: Differentiable<Tangent = Tensor<Alg::Scalar>>,
{
    fn pullback(
        &self,
        cotangent: &Tensor<Alg::Scalar>,
    ) -> AdResult<Vec<(NodeId, Tensor<Alg::Scalar>)>> {
        let n = self.primals.len();
        let mut results = Vec::new();
        let mut ctx = self.ctx.borrow_mut();

        for k in 0..n {
            let node_id = match self.input_node_ids[k] {
                Some(id) => id,
                None => continue,
            };

            // Build reverse einsum subscripts
            let mut rev_inputs_subs = vec![self.subscripts.output.clone()];
            let mut rev_operands: Vec<&Tensor<Alg::Scalar>> = vec![cotangent];

            for (i, primal) in self.primals.iter().enumerate() {
                if i != k {
                    rev_inputs_subs.push(self.subscripts.inputs[i].clone());
                    rev_operands.push(primal);
                }
            }

            let rev_subs = Subscripts {
                inputs: rev_inputs_subs,
                output: self.subscripts.inputs[k].clone(),
            };

            // Use backend-optimized einsum
            let mut grad =
                einsum_with_subscripts::<Alg, Backend>(&mut *ctx, &rev_subs, &rev_operands, None)
                    .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?;

            // Propagate fw_grad from operands through the reverse einsum
            if rev_operands.iter().any(|t| t.has_fw_grad()) {
                let tangents: Vec<Option<&Tensor<Alg::Scalar>>> =
                    rev_operands.iter().map(|t| t.fw_grad()).collect();
                if let Ok(grad_tangent) = einsum_frule_impl::<Alg, Backend>(
                    &mut *ctx,
                    &rev_subs,
                    None, // reverse subscripts are flat by construction
                    &rev_operands,
                    &tangents,
                ) {
                    grad.set_fw_grad(grad_tangent);
                }
            }

            results.push((node_id, grad));
        }

        Ok(results)
    }

    fn inputs(&self) -> Vec<NodeId> {
        self.input_node_ids.iter().filter_map(|id| *id).collect()
    }
}

/// Tracked einsum (reverse-mode AD).
///
/// This is the AD-aware counterpart of [`einsum`]. It records the operation
/// on the reverse-mode tape so that [`chainrules::Tape::pullback`] can
/// compute gradients through it.
///
/// The context is wrapped in `Rc<RefCell<>>` so the pullback rule can
/// reuse the same backend context for computing gradients.
///
/// # Examples
///
/// ```ignore
/// use std::cell::RefCell;
/// use std::rc::Rc;
/// use chainrules::Tape;
/// use tenferro_einsum::tracked_einsum;
/// use tenferro_tensor::{MemoryOrder, Tensor};
/// use tenferro_device::LogicalMemorySpace;
///
/// let ctx = Rc::new(RefCell::new(CpuContext::new(1)));
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
/// let c = tracked_einsum::<_, _, CpuBackend>(ctx.clone(), "ij,jk->ik", &[&a, &b]).unwrap();
/// let loss = tracked_einsum::<_, _, CpuBackend>(ctx.clone(), "ij,ij->", &[&c, &c]).unwrap();
/// let grads = tape.pullback(&loss).unwrap();
/// let _ga = grads.get(a.node_id().unwrap()).unwrap();
/// ```
///
pub fn tracked_einsum<Alg: 'static, Backend>(
    ctx: Rc<RefCell<Backend::Context>>,
    subscripts: &str,
    operands: &[&TrackedTensor<Tensor<Alg::Scalar>>],
) -> AdResult<TrackedTensor<Tensor<Alg::Scalar>>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg> + 'static,
    Tensor<Alg::Scalar>: Differentiable<Tangent = Tensor<Alg::Scalar>>,
{
    let subs = Subscripts::parse(subscripts)
        .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?;

    // Extract primals and run forward einsum
    let primals: Vec<&Tensor<Alg::Scalar>> = operands.iter().map(|op| op.value()).collect();
    let output = einsum::<Alg, Backend>(&mut *ctx.borrow_mut(), subscripts, &primals, None)
        .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?;

    // Check if any operand requires gradients
    let any_requires_grad = operands.iter().any(|op| op.requires_grad());

    if !any_requires_grad {
        return Ok(TrackedTensor::new(output));
    }

    // Find tape from the first tracked operand that has one
    let tape = operands
        .iter()
        .filter(|op| op.requires_grad())
        .find_map(|op| op.tape())
        .ok_or(chainrules::AutodiffError::MissingNode)?
        .clone();

    // Reject mixed-tape operands: all grad-tracked tensors must share the same tape
    for op in operands.iter().filter(|op| op.requires_grad()) {
        if let Some(op_tape) = op.tape() {
            if !tape.same_tape(op_tape) {
                return Err(chainrules::AutodiffError::InvalidArgument(
                    "tracked_einsum: operands belong to different AD tapes".into(),
                ));
            }
        }
    }

    let rule = EinsumReverseRule::<Alg, Backend> {
        ctx: ctx.clone(),
        subscripts: subs,
        primals: primals.iter().map(|&t| t.clone()).collect(),
        input_node_ids: operands.iter().map(|op| op.node_id()).collect(),
        _phantom: PhantomData,
    };

    // Record the operation on the tape so pullback can compute gradients
    let result = tape.record_op(output, Box::new(rule), None);

    Ok(result)
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
pub fn dual_einsum<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &str,
    operands: &[&DualTensor<Tensor<Alg::Scalar>>],
) -> AdResult<DualTensor<Tensor<Alg::Scalar>>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
    Tensor<Alg::Scalar>: Differentiable<Tangent = Tensor<Alg::Scalar>>,
{
    // Extract primals
    let primals: Vec<&Tensor<Alg::Scalar>> = operands.iter().map(|op| op.primal()).collect();

    // Compute primal output
    let output = einsum::<Alg, Backend>(ctx, subscripts, &primals, None)
        .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?;

    // Compute tangent: dC = sum_k einsum(subs, [A0, ..., dAk, ..., An])
    let tangents: Vec<Option<&Tensor<Alg::Scalar>>> =
        operands.iter().map(|op| op.tangent()).collect();

    let tangent = einsum_frule::<Alg, Backend>(ctx, subscripts, &primals, &tangents);

    match tangent {
        Ok(t) => DualTensor::with_tangent(output, t),
        Err(_) => Ok(DualTensor::new(output)),
    }
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
pub fn einsum_rrule<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &str,
    operands: &[&Tensor<Alg::Scalar>],
    cotangent: &Tensor<Alg::Scalar>,
) -> Result<Vec<Tensor<Alg::Scalar>>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let subs = Subscripts::parse(subscripts)?;
    let n = operands.len();
    let mut grads = Vec::with_capacity(n);

    for k in 0..n {
        // Build reverse subscripts for operand k:
        // grad_Ak = einsum([cotangent, A_0, ..., A_{k-1}, A_{k+1}, ..., A_n])
        let mut rev_inputs_subs = vec![subs.output.clone()];
        let mut rev_operands: Vec<&Tensor<Alg::Scalar>> = vec![cotangent];

        for (i, &op) in operands.iter().enumerate() {
            if i != k {
                rev_inputs_subs.push(subs.inputs[i].clone());
                rev_operands.push(op);
            }
        }

        let rev_subs = Subscripts {
            inputs: rev_inputs_subs,
            output: subs.inputs[k].clone(),
        };

        let grad = einsum_with_subscripts::<Alg, Backend>(ctx, &rev_subs, &rev_operands, None)?;
        grads.push(grad);
    }

    Ok(grads)
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
/// Internal frule implementation with pre-parsed subscripts.
///
/// When `nested` is `Some`, each frule term is computed via the nested tree
/// (respecting parenthesized contraction order). Otherwise the flat
/// `Subscripts` path is used.
fn einsum_frule_impl<Alg, Backend>(
    ctx: &mut Backend::Context,
    subs: &Subscripts,
    nested: Option<&NestedEinsum>,
    primals: &[&Tensor<Alg::Scalar>],
    tangents: &[Option<&Tensor<Alg::Scalar>>],
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let n = primals.len();

    // dC = sum_k einsum(subs, [A0, ..., dAk, ..., An]) for each k with tangent
    let mut result: Option<Tensor<Alg::Scalar>> = None;

    for k in 0..n {
        if let Some(tangent_k) = tangents[k] {
            let mut ops: Vec<&Tensor<Alg::Scalar>> = primals.to_vec();
            ops[k] = tangent_k;

            let term = if let Some(nested) = nested {
                execute_nested::<Alg, Backend>(ctx, nested, &ops, None)?
            } else {
                einsum_with_subscripts::<Alg, Backend>(ctx, subs, &ops, None)?
            };

            result = Some(match result {
                None => term,
                Some(existing) => Tensor::<Alg::Scalar>::accumulate_tangent(existing, &term),
            });
        }
    }

    result.ok_or_else(|| Error::InvalidArgument("no tangents provided".into()))
}

pub fn einsum_frule<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &str,
    primals: &[&Tensor<Alg::Scalar>],
    tangents: &[Option<&Tensor<Alg::Scalar>>],
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let subs = Subscripts::parse(subscripts)?;
    let nested = if subscripts.contains('(') {
        Some(NestedEinsum::parse(subscripts)?)
    } else {
        None
    };
    einsum_frule_impl::<Alg, Backend>(ctx, &subs, nested.as_ref(), primals, tangents)
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
pub fn einsum_hvp<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &str,
    primals: &[&Tensor<Alg::Scalar>],
    tangents: &[Option<&Tensor<Alg::Scalar>>],
    cotangent: &Tensor<Alg::Scalar>,
    cotangent_tangent: &Tensor<Alg::Scalar>,
) -> Result<Vec<(Tensor<Alg::Scalar>, Tensor<Alg::Scalar>)>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let subs = Subscripts::parse(subscripts)?;
    let n = primals.len();
    let mut results = Vec::with_capacity(n);

    for k in 0..n {
        // gradient_k = einsum([cotangent, A_0, ..., A_{k-1}, A_{k+1}, ..., An])
        // hvp_k = d/dv (gradient_k) = sum over sources of tangent:
        //   - from cotangent_tangent: einsum([dḡ, A_others...])
        //   - from each tangent_j (j != k): einsum([ḡ, A_0, ..., dA_j, ..., An])

        // Build reverse subscripts for operand k
        let mut rev_inputs_subs = vec![subs.output.clone()];
        for (i, _) in primals.iter().enumerate() {
            if i != k {
                rev_inputs_subs.push(subs.inputs[i].clone());
            }
        }
        let rev_subs = Subscripts {
            inputs: rev_inputs_subs,
            output: subs.inputs[k].clone(),
        };

        // Compute gradient_k
        let mut rev_operands: Vec<&Tensor<Alg::Scalar>> = vec![cotangent];
        for (i, &op) in primals.iter().enumerate() {
            if i != k {
                rev_operands.push(op);
            }
        }
        let grad_k = einsum_with_subscripts::<Alg, Backend>(ctx, &rev_subs, &rev_operands, None)?;

        // Compute hvp_k: differentiate the gradient w.r.t. v
        // hvp_k = einsum([dḡ, A_others...]) + sum_{j!=k} einsum([ḡ, ..., dA_j, ...])
        let mut hvp_k: Option<Tensor<Alg::Scalar>>;

        // Term from cotangent_tangent
        let mut ops: Vec<&Tensor<Alg::Scalar>> = vec![cotangent_tangent];
        for (i, &op) in primals.iter().enumerate() {
            if i != k {
                ops.push(op);
            }
        }
        let term = einsum_with_subscripts::<Alg, Backend>(ctx, &rev_subs, &ops, None)?;
        hvp_k = Some(term);

        // Terms from tangents of other primals
        for j in 0..n {
            if j == k {
                continue;
            }
            if let Some(tangent_j) = tangents[j] {
                let mut ops: Vec<&Tensor<Alg::Scalar>> = vec![cotangent];
                for (i, &op) in primals.iter().enumerate() {
                    if i != k {
                        if i == j {
                            ops.push(tangent_j);
                        } else {
                            ops.push(op);
                        }
                    }
                }
                let term = einsum_with_subscripts::<Alg, Backend>(ctx, &rev_subs, &ops, None)?;
                hvp_k = Some(match hvp_k {
                    None => term,
                    Some(existing) => Tensor::<Alg::Scalar>::accumulate_tangent(existing, &term),
                });
            }
        }

        // When no tangent contributions exist, allocate the zero HVP tensor
        // on the same memory space as the corresponding primal.
        let hvp_k = match hvp_k {
            Some(t) => t,
            None => {
                let space = primals[k].logical_memory_space();
                Tensor::zeros(primals[k].dims(), space, MemoryOrder::ColumnMajor)
            }
        };

        results.push((grad_k, hvp_k));
    }

    Ok(results)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_device::LogicalMemorySpace;
    use tenferro_tensor::MemoryOrder;

    #[test]
    fn infer_memory_space_single_cpu() {
        let a = Tensor::<f64>::zeros(
            &[2, 3],
            LogicalMemorySpace::MainMemory,
            MemoryOrder::ColumnMajor,
        );
        let space = infer_memory_space(&[&a]).unwrap();
        assert_eq!(space, LogicalMemorySpace::MainMemory);
    }

    #[test]
    fn infer_memory_space_multiple_cpu() {
        let a = Tensor::<f64>::zeros(
            &[2, 3],
            LogicalMemorySpace::MainMemory,
            MemoryOrder::ColumnMajor,
        );
        let b = Tensor::<f64>::zeros(
            &[3, 4],
            LogicalMemorySpace::MainMemory,
            MemoryOrder::ColumnMajor,
        );
        let c = Tensor::<f64>::zeros(
            &[4, 5],
            LogicalMemorySpace::MainMemory,
            MemoryOrder::ColumnMajor,
        );
        let space = infer_memory_space(&[&a, &b, &c]).unwrap();
        assert_eq!(space, LogicalMemorySpace::MainMemory);
    }

    #[test]
    fn infer_memory_space_empty_operands_errors() {
        let operands: &[&Tensor<f64>] = &[];
        let result = infer_memory_space(operands);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, Error::InvalidArgument(_)),
            "expected InvalidArgument, got: {err:?}"
        );
    }

    #[test]
    fn infer_memory_space_mixed_errors() {
        // We cannot construct GPU tensors in tests (assertion in Tensor::zeros
        // prevents GPU allocation), so we verify the logic by testing
        // the happy path (all CPU) and the error path (empty).
        // A true mixed-memory test requires GPU support which is not yet
        // available in the POC.
        //
        // This test documents the intended behaviour: calling einsum with
        // operands on different memory spaces returns
        // Error::CrossMemorySpaceOperation.
        let a = Tensor::<f64>::zeros(
            &[2, 3],
            LogicalMemorySpace::MainMemory,
            MemoryOrder::ColumnMajor,
        );
        // Verify that identical spaces produce Ok
        let space = infer_memory_space(&[&a, &a]).unwrap();
        assert_eq!(space, LogicalMemorySpace::MainMemory);
    }
}
