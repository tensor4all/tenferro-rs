use std::collections::{HashMap, HashSet};

use tenferro_algebra::Scalar;
use tenferro_device::{Error, LogicalMemorySpace, Result};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::pool::BufferPool;
use crate::subscripts::Subscripts;

pub(crate) const MAX_POOLED_BYTES: usize = 64 * 1024 * 1024; // 64 MB

/// Allocate a column-major tensor from the buffer pool, or fresh if too large.
pub(crate) fn alloc_tensor_from_pool<T: Scalar>(
    dims: &[usize],
    memory_space: LogicalMemorySpace,
    pool: &mut BufferPool<T>,
) -> Tensor<T> {
    let numel = dims.iter().product::<usize>().max(1);
    let bytes = numel.saturating_mul(std::mem::size_of::<T>());
    if bytes > MAX_POOLED_BYTES {
        return Tensor::zeros(dims, memory_space, MemoryOrder::ColumnMajor);
    }
    let data = pool.take(numel);
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
pub(crate) fn infer_memory_space<T: Scalar>(operands: &[&Tensor<T>]) -> Result<LogicalMemorySpace> {
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

/// Build a label → size mapping from subscripts and input shapes.
pub(crate) fn build_size_dict(
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
pub(crate) fn compute_output_shape(
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
pub(crate) fn intermediate_subs(subs_left: &[u32], subs_right: &[u32], needed: &HashSet<u32>) -> Vec<u32> {
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
pub(crate) fn contraction_cost(
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
pub(crate) fn tensor_get<T: Scalar>(t: &Tensor<T>, indices: &[usize]) -> T {
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
pub(crate) fn unflatten_index(flat: usize, dims: &[usize]) -> Vec<usize> {
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
