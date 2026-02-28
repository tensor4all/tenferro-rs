use std::collections::HashMap;

use num_traits::{One, Zero};
use tenferro_algebra::{Algebra, HasAlgebra, Scalar};
use tenferro_device::Result;
use tenferro_prims::{PrimDescriptor, TensorPrims};
use tenferro_tensor::Tensor;

use crate::classify::compute_permutation;
use crate::pool::BufferPool;
use crate::util::alloc_tensor_from_pool;

/// Prepare a single operand for GEMM: permute and try to fuse dimension groups.
pub(crate) fn prepare_one_operand<A, B>(
    ctx: &mut B::Context,
    tensor: &Tensor<A::Scalar>,
    current_subs: &[u32],
    target_subs: &[u32],
    nb: usize,
    n_group1: usize,
    n_group2: usize,
    fallback_shape: &[usize],
    pool: &mut BufferPool<A::Scalar>,
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
    let contiguous = permute_or_copy::<A, B>(ctx, tensor, current_subs, target_subs, pool)?;
    contiguous.reshape(fallback_shape)
}

/// Uses `Tensor::permute` (zero-copy view) first; copies to a pooled buffer
/// only when the result is non-contiguous. Falls back to MakeContiguous when
/// `current_subs == target_subs`.
pub(crate) fn permute_or_copy<A, B>(
    ctx: &mut B::Context,
    tensor: &Tensor<A::Scalar>,
    current_subs: &[u32],
    target_subs: &[u32],
    pool: &mut BufferPool<A::Scalar>,
) -> Result<Tensor<A::Scalar>>
where
    A: Algebra,
    A::Scalar: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
{
    if current_subs == target_subs {
        // No permutation needed; ensure contiguous
        return make_contiguous_if_needed::<A, B>(ctx, tensor, pool);
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
    let mut contiguous = alloc_tensor_from_pool::<A::Scalar>(view.dims(), memory_space, pool);
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
pub(crate) fn make_contiguous_if_needed<A, B>(
    ctx: &mut B::Context,
    tensor: &Tensor<A::Scalar>,
    pool: &mut BufferPool<A::Scalar>,
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
    let mut result = alloc_tensor_from_pool::<A::Scalar>(tensor.dims(), memory_space, pool);
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
