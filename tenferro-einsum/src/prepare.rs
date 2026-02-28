use std::collections::{HashMap, HashSet};

use num_traits::{One, Zero};
use tenferro_algebra::{Algebra, HasAlgebra, Scalar};
use tenferro_device::Result;
use tenferro_prims::{PrimDescriptor, ReduceOp, TensorPrims};
use tenferro_tensor::Tensor;

use crate::classify::compute_permutation;
use crate::pool::BufferPool;
use crate::util::alloc_tensor_from_pool;

/// Reduce axes that are present only in `subs_self` and absent from both
/// `subs_other` and `subs_out`.
pub(crate) fn reduce_unique_only_axes<A, B>(
    ctx: &mut B::Context,
    tensor: &Tensor<A::Scalar>,
    subs_self: &[u32],
    subs_other: &[u32],
    subs_out: &[u32],
    pool: &mut BufferPool<A::Scalar>,
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
    let mut reduced = alloc_tensor_from_pool::<A::Scalar>(&out_shape, memory_space, pool);

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
pub(crate) fn prepare_gemm_operands<A, B>(
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
    pool: &mut BufferPool<A::Scalar>,
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

    let a_prepared = prepare_one_operand::<A, B>(
        ctx,
        a,
        subs_a,
        target_a,
        nb,
        n_lo,
        n_sum_a,
        &a_gemm_shape,
        pool,
    )?;
    let b_prepared = prepare_one_operand::<A, B>(
        ctx,
        b,
        subs_b,
        target_b,
        nb,
        n_sum_b,
        n_ro,
        &b_gemm_shape,
        pool,
    )?;
    Ok((a_prepared, b_prepared))
}

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
