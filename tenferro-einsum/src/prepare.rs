use std::collections::HashMap;

use num_traits::{One, Zero};
use tenferro_algebra::{Algebra, HasAlgebra, Scalar};
use tenferro_device::Result;
use tenferro_prims::{PrimDescriptor, TensorPrims};
use tenferro_tensor::Tensor;

use crate::classify::compute_permutation;
use crate::pool::BufferPool;
use crate::util::alloc_tensor_from_pool;

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
    // Layout after permute: [g1..., g2..., batch...]
    let g1_dims = &dims[..n_group1];
    let g1_strides = &strides[..n_group1];
    let g2_dims = &dims[n_group1..n_group1 + n_group2];
    let g2_strides = &strides[n_group1..n_group1 + n_group2];
    let batch_dims = &dims[n_group1 + n_group2..];
    let batch_strides = &strides[n_group1 + n_group2..];

    let fused_g1 = try_fuse_group(g1_dims, g1_strides);
    let fused_g2 = try_fuse_group(g2_dims, g2_strides);

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
            let contiguous = make_contiguous_if_needed::<A, B>(ctx, &partial, pool)?;
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
            let contiguous = make_contiguous_if_needed::<A, B>(ctx, &partial, pool)?;
            contiguous.reshape(fallback_shape)
        }
        (None, None) => {
            // Full fallback: both groups are non-fusable; materialize full target
            // layout, then reshape for GEMM.
            #[cfg(feature = "profile-dispatch")]
            record_prepare_fallback(dims);
            let contiguous = make_contiguous_if_needed::<A, B>(ctx, &permuted, pool)?;
            contiguous.reshape(fallback_shape)
        }
    }
}

/// Uses `Tensor::permute` (zero-copy view) first; copies to a pooled buffer
/// only when the result is non-contiguous. Falls back to MakeContiguous when
/// `current_subs == target_subs`.
#[cfg_attr(not(test), allow(dead_code))]
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

#[cfg(test)]
mod tests {
    use tenferro_algebra::Standard;
    use tenferro_device::LogicalMemorySpace;
    use tenferro_prims::{CpuBackend, CpuContext};
    use tenferro_tensor::MemoryOrder;

    use super::*;
    use crate::util::{tensor_get, unflatten_index};

    fn tensor(data: &[f64], dims: &[usize]) -> Tensor<f64> {
        Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
    }

    fn assert_tensor_close(lhs: &Tensor<f64>, rhs: &Tensor<f64>) {
        assert_eq!(lhs.dims(), rhs.dims());
        let numel: usize = lhs.dims().iter().product();
        for flat in 0..numel {
            let idx = unflatten_index(flat, lhs.dims());
            let l = tensor_get(lhs, &idx);
            let r = tensor_get(rhs, &idx);
            assert!(
                (l - r).abs() < 1e-10,
                "mismatch at {:?}: left={} right={}",
                idx,
                l,
                r
            );
        }
    }

    #[test]
    fn prepare_one_operand_zero_copy_fuses_groups() {
        let mut ctx = CpuContext::new(1);
        let mut pool = BufferPool::new();
        let input = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let prepared = prepare_one_operand::<Standard<f64>, CpuBackend>(
            &mut ctx,
            &input,
            &[0, 1],
            &[0, 1],
            0,
            1,
            1,
            &[2, 2],
            &mut pool,
        )
        .unwrap();

        assert_eq!(prepared.dims(), &[2, 2]);
        assert_eq!(prepared.buffer().as_ptr(), input.buffer().as_ptr());
    }

    #[test]
    fn prepare_one_operand_partial_fallback_when_group2_nonfusable() {
        let mut ctx = CpuContext::new(1);
        let mut pool = BufferPool::new();
        let mut ref_pool = BufferPool::new();
        let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        let input = tensor(&data, &[2, 3, 4]);
        let fallback_shape = [3, 8];

        let prepared = prepare_one_operand::<Standard<f64>, CpuBackend>(
            &mut ctx,
            &input,
            &[0, 1, 2],
            &[1, 2, 0],
            0,
            1,
            2,
            &fallback_shape,
            &mut pool,
        )
        .unwrap();
        let expected = permute_or_copy::<Standard<f64>, CpuBackend>(
            &mut ctx,
            &input,
            &[0, 1, 2],
            &[1, 2, 0],
            &mut ref_pool,
        )
        .unwrap()
        .reshape(&fallback_shape)
        .unwrap();

        assert_eq!(prepared.dims(), &fallback_shape);
        assert!(prepared.is_contiguous());
        assert_tensor_close(&prepared, &expected);
    }

    #[test]
    fn prepare_one_operand_partial_fallback_when_group1_nonfusable() {
        let mut ctx = CpuContext::new(1);
        let mut pool = BufferPool::new();
        let mut ref_pool = BufferPool::new();
        let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        let input = tensor(&data, &[2, 3, 4]);
        let fallback_shape = [8, 3];

        let prepared = prepare_one_operand::<Standard<f64>, CpuBackend>(
            &mut ctx,
            &input,
            &[0, 1, 2],
            &[2, 0, 1],
            0,
            2,
            1,
            &fallback_shape,
            &mut pool,
        )
        .unwrap();
        let expected = permute_or_copy::<Standard<f64>, CpuBackend>(
            &mut ctx,
            &input,
            &[0, 1, 2],
            &[2, 0, 1],
            &mut ref_pool,
        )
        .unwrap()
        .reshape(&fallback_shape)
        .unwrap();

        assert_eq!(prepared.dims(), &fallback_shape);
        assert!(prepared.is_contiguous());
        assert_tensor_close(&prepared, &expected);
    }

    #[test]
    fn permute_or_copy_transpose_materializes_contiguous_copy() {
        let mut ctx = CpuContext::new(1);
        let mut pool = BufferPool::new();
        let input = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

        let prepared = permute_or_copy::<Standard<f64>, CpuBackend>(
            &mut ctx,
            &input,
            &[0, 1],
            &[1, 0],
            &mut pool,
        )
        .unwrap();

        assert_eq!(prepared.dims(), &[3, 2]);
        assert!(prepared.is_contiguous());
        assert!((tensor_get(&prepared, &[0, 0]) - 1.0).abs() < 1e-10);
        assert!((tensor_get(&prepared, &[1, 0]) - 3.0).abs() < 1e-10);
        assert!((tensor_get(&prepared, &[2, 0]) - 5.0).abs() < 1e-10);
        assert!((tensor_get(&prepared, &[0, 1]) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn permute_or_copy_returns_contiguous_view_for_unit_extent_permute() {
        let mut ctx = CpuContext::new(1);
        let mut pool = BufferPool::new();
        let input = tensor(&[1.0, 2.0], &[2, 1]);

        let prepared = permute_or_copy::<Standard<f64>, CpuBackend>(
            &mut ctx,
            &input,
            &[0, 1],
            &[1, 0],
            &mut pool,
        )
        .unwrap();

        assert_eq!(prepared.dims(), &[1, 2]);
        assert!(prepared.is_contiguous());
        assert_eq!(prepared.buffer().as_ptr(), input.buffer().as_ptr());
    }

    #[test]
    fn make_contiguous_if_needed_copies_only_when_required() {
        let mut ctx = CpuContext::new(1);
        let mut pool = BufferPool::new();
        let contiguous = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let shared = make_contiguous_if_needed::<Standard<f64>, CpuBackend>(
            &mut ctx,
            &contiguous,
            &mut pool,
        )
        .unwrap();
        assert_eq!(shared.buffer().as_ptr(), contiguous.buffer().as_ptr());

        let transposed = contiguous.permute(&[1, 0]).unwrap();
        let copied = make_contiguous_if_needed::<Standard<f64>, CpuBackend>(
            &mut ctx,
            &transposed,
            &mut pool,
        )
        .unwrap();

        assert!(copied.is_contiguous());
        assert_eq!(copied.dims(), &[2, 2]);
        assert_eq!(
            copied.logical_memory_space(),
            LogicalMemorySpace::MainMemory
        );
        assert!((tensor_get(&copied, &[0, 0]) - 1.0).abs() < 1e-10);
        assert!((tensor_get(&copied, &[1, 0]) - 3.0).abs() < 1e-10);
        assert!((tensor_get(&copied, &[0, 1]) - 2.0).abs() < 1e-10);
    }
}
