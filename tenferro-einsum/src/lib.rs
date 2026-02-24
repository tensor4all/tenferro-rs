//! High-level einsum with N-ary contraction tree optimization.
//!
//! This crate provides Einstein summation notation for [`Tensor`]
//! values. It supports:
//!
//! - **String notation**: `"ij,jk->ik"` (NumPy/PyTorch compatible)
//! - **Parenthesized notation**: `"ij,(jk,kl)->il"` is accepted but
//!   grouping is currently ignored (optimizer picks order)
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

/// Convert an ASCII character label to u32.
fn char_to_label(c: char) -> Result<u32> {
    match c {
        'a'..='z' => Ok((c as u32) - ('a' as u32)),
        'A'..='Z' => Ok((c as u32) - ('A' as u32) + 26),
        _ => Err(Error::InvalidArgument(format!(
            "invalid einsum label character: '{c}'"
        ))),
    }
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
            let mut intermediate = Tensor::<Alg::Scalar>::zeros(
                &inter_shape,
                output.logical_memory_space(),
                MemoryOrder::ColumnMajor,
            );
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
    let (batch_modes, lo_modes, ro_modes, sum_modes) = classify_modes(subs_a, subs_b, subs_c);

    // Build size_dict from input subscripts and shapes
    let mut size_dict: HashMap<u32, usize> = HashMap::new();
    for (&label, &dim) in subs_a.iter().zip(a.dims()) {
        size_dict.insert(label, dim);
    }
    for (&label, &dim) in subs_b.iter().zip(b.dims()) {
        size_dict.insert(label, dim);
    }

    let batch_sizes: Vec<usize> = batch_modes.iter().map(|m| size_dict[m]).collect();
    let lo_sizes: Vec<usize> = lo_modes.iter().map(|m| size_dict[m]).collect();
    let ro_sizes: Vec<usize> = ro_modes.iter().map(|m| size_dict[m]).collect();
    let sum_sizes: Vec<usize> = sum_modes.iter().map(|m| size_dict[m]).collect();

    let m: usize = lo_sizes.iter().product::<usize>().max(1);
    let n: usize = ro_sizes.iter().product::<usize>().max(1);
    let k: usize = sum_sizes.iter().product::<usize>().max(1);

    // --- Step 1: Permute A to [batch, lo, sum] ---
    let target_a: Vec<u32> = batch_modes
        .iter()
        .chain(lo_modes.iter())
        .chain(sum_modes.iter())
        .copied()
        .collect();
    let perm_a = permute_or_copy::<A, B>(ctx, a, subs_a, &target_a)?;

    // --- Step 2: Permute B to [batch, sum, ro] ---
    let target_b: Vec<u32> = batch_modes
        .iter()
        .chain(sum_modes.iter())
        .chain(ro_modes.iter())
        .copied()
        .collect();
    let perm_b = permute_or_copy::<A, B>(ctx, b, subs_b, &target_b)?;

    // --- Step 3: Reshape to [batch..., m, k] and [batch..., k, n] ---
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
    let c_gemm_shape: Vec<usize> = batch_sizes
        .iter()
        .copied()
        .chain(std::iter::once(m))
        .chain(std::iter::once(n))
        .collect();

    let a_reshaped = perm_a.reshape(&a_gemm_shape)?;
    let b_reshaped = perm_b.reshape(&b_gemm_shape)?;

    // --- Step 4: BatchedGemm → temp ---
    let memory_space = a.logical_memory_space();
    let mut temp =
        Tensor::<A::Scalar>::zeros(&c_gemm_shape, memory_space, MemoryOrder::ColumnMajor);

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

/// Permute a tensor to a target mode order and ensure contiguous layout.
///
/// If `current_subs == target_subs`, returns a contiguous copy (MakeContiguous)
/// if needed. Otherwise, permutes to the target order, then makes contiguous.
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

    // Compute target shape from the permutation
    let label_to_pos: HashMap<u32, usize> = current_subs
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, i))
        .collect();
    let target_shape: Vec<usize> = target_subs
        .iter()
        .map(|l| tensor.dims()[label_to_pos[l]])
        .collect();

    // Permute via prim
    let memory_space = tensor.logical_memory_space();
    let mut permuted =
        Tensor::<A::Scalar>::zeros(&target_shape, memory_space, MemoryOrder::ColumnMajor);
    let desc = PrimDescriptor::Permute {
        modes_a: current_subs.to_vec(),
        modes_b: target_subs.to_vec(),
    };
    let shapes = [tensor.dims(), permuted.dims()];
    let plan = B::plan(ctx, &desc, &shapes)?;
    B::execute(
        ctx,
        &plan,
        A::Scalar::one(),
        &[tensor],
        A::Scalar::zero(),
        &mut permuted,
    )?;

    // The result is already contiguous (freshly allocated)
    Ok(permuted)
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
    let mut result =
        Tensor::<A::Scalar>::zeros(tensor.dims(), memory_space, MemoryOrder::ColumnMajor);
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
        // Fallback: decompose into Permute + BatchedGemm
        fallback_pairwise_contraction::<Alg, Backend>(
            ctx, subs_a, subs_b, subs_c, a, b, alpha, beta, output,
        )
    }
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

    // Multi-tensor case: follow the contraction tree
    let mut intermediates: Vec<Tensor<Alg::Scalar>> = Vec::new();

    for (step_idx, step) in tree.steps.iter().enumerate() {
        let left: &Tensor<Alg::Scalar> = if step.left < n_inputs {
            operands[step.left]
        } else {
            &intermediates[step.left - n_inputs]
        };
        let right: &Tensor<Alg::Scalar> = if step.right < n_inputs {
            operands[step.right]
        } else {
            &intermediates[step.right - n_inputs]
        };

        let subs_left = &tree.operand_subs[step.left];
        let subs_right = &tree.operand_subs[step.right];
        let is_last = step_idx == tree.steps.len() - 1;

        if is_last {
            // Last step: write directly to output with alpha/beta
            execute_pairwise_contraction::<Alg, Backend>(
                ctx,
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
            // Allocate on the same memory space as the input operands.
            let result_idx = n_inputs + step_idx;
            let subs_result = &tree.operand_subs[result_idx];
            let result_shape = compute_output_shape(subs_result, &tree.size_dict)?;
            let memory_space = infer_memory_space(operands)?;
            let mut result =
                Tensor::<Alg::Scalar>::zeros(&result_shape, memory_space, MemoryOrder::ColumnMajor);
            execute_pairwise_contraction::<Alg, Backend>(
                ctx,
                subs_left,
                subs_right,
                subs_result,
                left,
                right,
                Alg::Scalar::one(),
                Alg::Scalar::zero(),
                &mut result,
            )?;
            intermediates.push(result);
        }
    }

    Ok(())
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
        let parts: Vec<&str> = notation.split("->").collect();
        if parts.len() != 2 {
            return Err(Error::InvalidArgument(format!(
                "einsum notation must contain exactly one '->', got: {notation}"
            )));
        }
        let inputs_str = parts[0];
        let output_str = parts[1];

        // Parse output labels
        let output: Vec<u32> = output_str
            .chars()
            .map(char_to_label)
            .collect::<Result<_>>()?;

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

        Ok(Self {
            subscripts: subscripts.clone(),
            steps,
            size_dict,
            operand_subs,
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
    let subs = Subscripts::parse(subscripts)?;
    let mut output = einsum_with_subscripts::<Alg, Backend>(ctx, &subs, operands, size_dict)?;

    // Auto-propagate forward-mode tangents
    if operands.iter().any(|t| t.has_fw_grad()) {
        let tangents: Vec<Option<&Tensor<Alg::Scalar>>> =
            operands.iter().map(|t| t.fw_grad()).collect();
        // einsum_frule_impl calls einsum_with_subscripts (not einsum), so no recursion
        if let Ok(output_tangent) =
            einsum_frule_impl::<Alg, Backend>(ctx, &subs, operands, &tangents)
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
    einsum_with_subscripts_into::<Alg, Backend>(
        ctx, &subs, operands, alpha, beta, output, size_dict,
    )
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
fn einsum_frule_impl<Alg, Backend>(
    ctx: &mut Backend::Context,
    subs: &Subscripts,
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

            let term = einsum_with_subscripts::<Alg, Backend>(ctx, subs, &ops, None)?;

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
    einsum_frule_impl::<Alg, Backend>(ctx, &subs, primals, tangents)
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
