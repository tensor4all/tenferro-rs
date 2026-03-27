use std::collections::HashMap;

use tenferro_algebra::{HasAlgebra, Scalar, Semiring};
use tenferro_device::{Error, LogicalMemorySpace, Result};
use tenferro_prims::{SemiringCoreDescriptor, TensorSemiringCore, TensorTempPoolContext};
use tenferro_tensor::Tensor;

use crate::execution::backend::BackendContext;
use crate::execution::pool::TensorBufferPool;
use crate::execution::util::alloc_tensor_from_pool;
use crate::layout::try_fuse_group_in_target_order;
use crate::planning::classify::compute_permutation;

#[cfg(feature = "profile-dispatch")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "profile-dispatch")]
static PREPARE_ZEROCOPY: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-dispatch")]
static PREPARE_FALLBACK: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-dispatch")]
static PREPARE_FALLBACK_ELEMS: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "profile-dispatch")]
pub(crate) fn print_and_reset_prepare_profile() {
    let zc = PREPARE_ZEROCOPY.swap(0, Ordering::Relaxed);
    let fb = PREPARE_FALLBACK.swap(0, Ordering::Relaxed);
    let elems = PREPARE_FALLBACK_ELEMS.swap(0, Ordering::Relaxed);
    eprintln!("[prepare profile] zero_copy={zc}  fallback_copy={fb}  fallback_elems={elems}");
}

#[cfg(feature = "profile-dispatch")]
#[inline]
fn record_prepare_fallback(dims: &[usize]) {
    PREPARE_FALLBACK.fetch_add(1, Ordering::Relaxed);
    let n_elems: usize = dims.iter().product();
    PREPARE_FALLBACK_ELEMS.fetch_add(n_elems as u64, Ordering::Relaxed);
}

/// Prepare a single operand for GEMM: permute and try to fuse dimension groups.
pub(crate) fn prepare_one_operand<A, B, P>(
    ctx: &mut BackendContext<A, B>,
    tensor: &Tensor<A::Scalar>,
    current_subs: &[u32],
    target_subs: &[u32],
    nb: usize,
    n_group1: usize,
    n_group2: usize,
    fallback_shape: &[usize],
    pool: &mut P,
) -> Result<Tensor<A::Scalar>>
where
    A: Semiring,
    A::Scalar: Scalar + HasAlgebra<Algebra = A>,
    B: TensorSemiringCore<A>,
    BackendContext<A, B>: TensorTempPoolContext,
    P: TensorBufferPool<A::Scalar> + ?Sized,
{
    // Step 1: Permute (metadata-only, zero-copy)
    let permuted = if current_subs == target_subs {
        tensor.clone()
    } else {
        let perm =
            compute_permutation(current_subs, target_subs).map_err(Error::InvalidArgument)?;
        tensor.permute(&perm)?
    };

    let dims = permuted.dims();
    let strides = permuted.strides();

    // Step 2: Try to fuse each dimension group
    // Layout after permute: [g1..., g2..., batch...]
    let g1_dims = &dims[..n_group1];
    let g1_strides = &strides[..n_group1];
    let g2_dims = &dims[n_group1..n_group1 + n_group2];
    let g2_strides = &strides[n_group1..n_group1 + n_group2];
    let batch_dims = &dims[n_group1 + n_group2..];
    let batch_strides = &strides[n_group1 + n_group2..];

    let fused_g1 = try_fuse_group_in_target_order(g1_dims, g1_strides);
    let fused_g2 = try_fuse_group_in_target_order(g2_dims, g2_strides);

    match (fused_g1, fused_g2) {
        (Some((size1, stride1)), Some((size2, stride2))) => {
            // Zero-copy: construct fused view [fused_g1, fused_g2, batch...]
            #[cfg(feature = "profile-dispatch")]
            PREPARE_ZEROCOPY.fetch_add(1, Ordering::Relaxed);
            let mut fused_dims = Vec::with_capacity(nb + 2);
            let mut fused_strides = Vec::with_capacity(nb + 2);
            fused_dims.push(size1);
            fused_strides.push(stride1);
            fused_dims.push(size2);
            fused_strides.push(stride2);
            fused_dims.extend_from_slice(batch_dims);
            fused_strides.extend_from_slice(batch_strides);
            permuted.view_as_strided(fused_dims, fused_strides)
        }
        (Some((size1, stride1)), None) => {
            // Partial fallback: g1 is already fusable, so preserve it and
            // materialize contiguity for the failing g2 side only.
            #[cfg(feature = "profile-dispatch")]
            record_prepare_fallback(dims);
            let mut partial_dims = Vec::with_capacity(1 + n_group2 + nb);
            let mut partial_strides = Vec::with_capacity(1 + n_group2 + nb);
            partial_dims.push(size1);
            partial_strides.push(stride1);
            partial_dims.extend_from_slice(g2_dims);
            partial_dims.extend_from_slice(batch_dims);
            partial_strides.extend_from_slice(g2_strides);
            partial_strides.extend_from_slice(batch_strides);
            let partial = permuted.view_as_strided(partial_dims, partial_strides)?;
            let contiguous = make_contiguous_if_needed::<A, B, P>(ctx, &partial, pool)?;
            contiguous.reshape(fallback_shape)
        }
        (None, Some((size2, stride2))) => {
            // Partial fallback: g2 is already fusable, so preserve it and
            // materialize contiguity for the failing g1 side only.
            #[cfg(feature = "profile-dispatch")]
            record_prepare_fallback(dims);
            let mut partial_dims = Vec::with_capacity(n_group1 + 1 + nb);
            let mut partial_strides = Vec::with_capacity(n_group1 + 1 + nb);
            partial_dims.extend_from_slice(g1_dims);
            partial_dims.push(size2);
            partial_dims.extend_from_slice(batch_dims);
            partial_strides.extend_from_slice(g1_strides);
            partial_strides.push(stride2);
            partial_strides.extend_from_slice(batch_strides);
            let partial = permuted.view_as_strided(partial_dims, partial_strides)?;
            let contiguous = make_contiguous_if_needed::<A, B, P>(ctx, &partial, pool)?;
            contiguous.reshape(fallback_shape)
        }
        (None, None) => {
            // Full fallback: both groups are non-fusable; materialize full target
            // layout, then reshape for GEMM.
            #[cfg(feature = "profile-dispatch")]
            record_prepare_fallback(dims);
            let contiguous = make_contiguous_if_needed::<A, B, P>(ctx, &permuted, pool)?;
            contiguous.reshape(fallback_shape)
        }
    }
}

/// Uses `Tensor::permute` (zero-copy view) first; copies to a pooled buffer
/// only when the result is non-contiguous. Falls back to MakeContiguous when
/// `current_subs == target_subs`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn permute_or_copy<A, B, P>(
    ctx: &mut BackendContext<A, B>,
    tensor: &Tensor<A::Scalar>,
    current_subs: &[u32],
    target_subs: &[u32],
    pool: &mut P,
) -> Result<Tensor<A::Scalar>>
where
    A: Semiring,
    A::Scalar: Scalar + HasAlgebra<Algebra = A>,
    B: TensorSemiringCore<A>,
    BackendContext<A, B>: TensorTempPoolContext,
    P: TensorBufferPool<A::Scalar> + ?Sized,
{
    if current_subs == target_subs {
        // No permutation needed; ensure contiguous
        return make_contiguous_if_needed::<A, B, P>(ctx, tensor, pool);
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
    if view.is_col_major_contiguous() {
        return Ok(view);
    }

    // Otherwise copy to a contiguous pooled buffer
    let memory_space = tensor.logical_memory_space();
    let mut contiguous =
        alloc_tensor_from_pool::<A::Scalar, _, _>(ctx, view.dims(), memory_space, pool)?;
    let desc = SemiringCoreDescriptor::MakeContiguous;
    let shapes = [view.dims(), contiguous.dims()];
    let plan = <B as TensorSemiringCore<A>>::plan(ctx, &desc, &shapes)?;
    <B as TensorSemiringCore<A>>::execute(
        ctx,
        &plan,
        A::one(),
        &[&view],
        A::zero(),
        &mut contiguous,
    )?;
    Ok(contiguous)
}

/// Ensure a tensor is contiguous, copying if necessary.
pub(crate) fn make_contiguous_if_needed<A, B, P>(
    ctx: &mut BackendContext<A, B>,
    tensor: &Tensor<A::Scalar>,
    pool: &mut P,
) -> Result<Tensor<A::Scalar>>
where
    A: Semiring,
    A::Scalar: Scalar + HasAlgebra<Algebra = A>,
    B: TensorSemiringCore<A>,
    BackendContext<A, B>: TensorTempPoolContext,
    P: TensorBufferPool<A::Scalar> + ?Sized,
{
    if tensor.is_col_major_contiguous() {
        return Ok(tensor.clone());
    }

    // CPU fast-path: avoid family-descriptor planning overhead for pure host
    // copies by using strided-perm directly while still reusing pooled buffers.
    if tensor.logical_memory_space() == LogicalMemorySpace::MainMemory {
        if let Some(src_data) = tensor.buffer().as_slice() {
            let dims = tensor.dims();
            let mut result = alloc_tensor_from_pool::<A::Scalar, _, _>(
                ctx,
                dims,
                LogicalMemorySpace::MainMemory,
                pool,
            )?;

            let src =
                strided_view::StridedView::new(src_data, dims, tensor.strides(), tensor.offset())
                    .map_err(|e| Error::StrideError(e.to_string()))?;

            // alloc_tensor_from_pool always returns column-major contiguous layout.
            let mut dst_strides = Vec::with_capacity(dims.len());
            let mut s = 1isize;
            for &d in dims {
                dst_strides.push(s);
                s *= d as isize;
            }

            let dst_data = result.buffer_mut().as_mut_slice().ok_or_else(|| {
                Error::DeviceError("CPU tensor expected in host copy path".into())
            })?;
            let mut dst = strided_view::StridedViewMut::new(dst_data, dims, &dst_strides, 0)
                .map_err(|e| Error::StrideError(e.to_string()))?;

            strided_perm::copy_into(&mut dst, &src)
                .map_err(|e| Error::StrideError(e.to_string()))?;
            return Ok(result);
        }
    }

    let memory_space = tensor.logical_memory_space();
    let mut result =
        alloc_tensor_from_pool::<A::Scalar, _, _>(ctx, tensor.dims(), memory_space, pool)?;
    let desc = SemiringCoreDescriptor::MakeContiguous;
    let shapes = [tensor.dims(), result.dims()];
    let plan = <B as TensorSemiringCore<A>>::plan(ctx, &desc, &shapes)?;
    <B as TensorSemiringCore<A>>::execute(ctx, &plan, A::one(), &[tensor], A::zero(), &mut result)?;
    Ok(result)
}

#[cfg(test)]
mod tests;
