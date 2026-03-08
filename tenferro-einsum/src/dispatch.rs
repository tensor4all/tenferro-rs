// One and Zero are needed to call Alg::Scalar::one() / ::zero() via the
// Scalar supertrait. Rust incorrectly flags these as unused when the call
// site is an associated type (Alg::Scalar) rather than a concrete type.
#[allow(unused_imports)]
use num_traits::{One, Zero};
use tenferro_algebra::{Algebra, HasAlgebra, Scalar};
use tenferro_device::{Error, Result};
use tenferro_prims::{Extension, PrimDescriptor, ReduceOp, TensorPrims};
use tenferro_tensor::Tensor;

use crate::classify::compute_permutation;
use crate::plan::{GemmPlan, ReducePlan, StepPlan};
use crate::pool::BufferPool;
use crate::prepare::prepare_one_operand;
use crate::util::alloc_tensor_from_pool;

#[cfg(feature = "profile-dispatch")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "profile-dispatch")]
static PREPARE_NS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-dispatch")]
static GEMM_NS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-dispatch")]
static PERMUTE_NS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-dispatch")]
static EWMUL_NS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-dispatch")]
static REDUCE_NS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-dispatch")]
static CALL_COUNT: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-dispatch")]
static GEMM_FLOPS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "profile-dispatch")]
static GEMM_CALLS_TINY: AtomicU64 = AtomicU64::new(0);

/// Reset and print accumulated dispatch profiling counters.
#[cfg(feature = "profile-dispatch")]
pub fn print_and_reset_profile() {
    crate::prepare::print_and_reset_prepare_profile();
    let calls = CALL_COUNT.swap(0, Ordering::Relaxed);
    let prepare = PREPARE_NS.swap(0, Ordering::Relaxed);
    let gemm = GEMM_NS.swap(0, Ordering::Relaxed);
    let perm = PERMUTE_NS.swap(0, Ordering::Relaxed);
    let ewmul = EWMUL_NS.swap(0, Ordering::Relaxed);
    let reduce = REDUCE_NS.swap(0, Ordering::Relaxed);
    let flops = GEMM_FLOPS.swap(0, Ordering::Relaxed);
    let tiny = GEMM_CALLS_TINY.swap(0, Ordering::Relaxed);
    eprintln!(
        "[dispatch profile] calls={calls}  prepare={:.1}ms  gemm={:.1}ms  permute={:.1}ms  ewmul={:.1}ms  reduce={:.1}ms  gemm_flops={:.2e}  tiny_gemms={tiny}",
        prepare as f64 / 1e6,
        gemm as f64 / 1e6,
        perm as f64 / 1e6,
        ewmul as f64 / 1e6,
        reduce as f64 / 1e6,
        flops as f64,
    );
}

/// Execute a pairwise contraction using a pre-computed step plan.
///
/// Linear dispatch: diagonal extraction → pre-reduce → EwMul / Contract / BatchedGemm.
///
/// When `lazy_output` is true and `alpha=1, beta=0`, the output may be a
/// non-contiguous view (lazy permute) instead of a physical copy. Only safe
/// for intermediate tensors consumed by subsequent einsum steps.
pub(crate) fn execute_pairwise_with_plan<Alg, Backend>(
    ctx: &mut Backend::Context,
    plan: &StepPlan,
    ewmul_plan: Option<&Backend::Plan>,
    gemm_plan: Option<&Backend::Plan>,
    _subs_a: &[u32],
    _subs_b: &[u32],
    subs_c: &[u32],
    a: &Tensor<Alg::Scalar>,
    b: &Tensor<Alg::Scalar>,
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
    pool: &mut BufferPool<Alg::Scalar>,
    lazy_output: bool,
) -> Result<()>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    // 1. Diagonal extraction (zero-copy view)
    let a_diag;
    let a_ref = if let Some(ref dp) = plan.diag_a {
        a_diag = a.diagonal(&dp.axis_pairs)?;
        &a_diag
    } else {
        a
    };
    let b_diag;
    let b_ref = if let Some(ref dp) = plan.diag_b {
        b_diag = b.diagonal(&dp.axis_pairs)?;
        &b_diag
    } else {
        b
    };

    // 2. Pre-reduce unique-only axes
    #[cfg(feature = "profile-dispatch")]
    let _reduce_t0 = std::time::Instant::now();
    let a_reduced;
    let a_ref = if let Some(ref reduce) = plan.gemm.reduce_a {
        a_reduced = execute_reduce_with_plan::<Alg, Backend>(ctx, reduce, a_ref, pool)?;
        &a_reduced
    } else {
        a_ref
    };
    let b_reduced;
    let b_ref = if let Some(ref reduce) = plan.gemm.reduce_b {
        b_reduced = execute_reduce_with_plan::<Alg, Backend>(ctx, reduce, b_ref, pool)?;
        &b_reduced
    } else {
        b_ref
    };
    #[cfg(feature = "profile-dispatch")]
    REDUCE_NS.fetch_add(_reduce_t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

    // 3. EwMul fast-path (pure batch only: no lo/ro/sum modes).
    //
    // Important: `m=n=k=1` alone is insufficient, because contractions over
    // unit-extent summed axes (or unit-extent lo/ro axes) can still require
    // non-elementwise semantics and shape changes.
    let pure_batch_ewmul = plan.gemm.lo_modes.is_empty()
        && plan.gemm.ro_modes.is_empty()
        && plan.gemm.sum_modes.is_empty();
    if pure_batch_ewmul {
        if let Some(pp) = ewmul_plan {
            #[cfg(feature = "profile-dispatch")]
            let _t0 = std::time::Instant::now();
            let r = Backend::execute(ctx, pp, alpha, &[a_ref, b_ref], beta, output);
            #[cfg(feature = "profile-dispatch")]
            EWMUL_NS.fetch_add(_t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            #[cfg(feature = "profile-dispatch")]
            CALL_COUNT.fetch_add(1, Ordering::Relaxed);
            return r;
        }
        if Backend::has_extension_for(Extension::ElementwiseMul) {
            let desc = PrimDescriptor::ElementwiseMul;
            let shapes = [a_ref.dims(), b_ref.dims(), output.dims()];
            let pp = Backend::plan(ctx, &desc, &shapes)?;
            #[cfg(feature = "profile-dispatch")]
            let _t0 = std::time::Instant::now();
            let r = Backend::execute(ctx, &pp, alpha, &[a_ref, b_ref], beta, output);
            #[cfg(feature = "profile-dispatch")]
            EWMUL_NS.fetch_add(_t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            #[cfg(feature = "profile-dispatch")]
            CALL_COUNT.fetch_add(1, Ordering::Relaxed);
            return r;
        }
    }

    // 4. Contract extension (user-provided backends, e.g. cuTENSOR)
    if Backend::has_extension_for(Extension::Contract) {
        let desc = PrimDescriptor::Contract {
            modes_a: plan.gemm.subs_a.clone(),
            modes_b: plan.gemm.subs_b.clone(),
            modes_c: subs_c.to_vec(),
        };
        let shapes = [a_ref.dims(), b_ref.dims(), output.dims()];
        let pp = Backend::plan(ctx, &desc, &shapes)?;
        return Backend::execute(ctx, &pp, alpha, &[a_ref, b_ref], beta, output);
    }

    // 5. Fallback: prepare + BatchedGemm + permute
    execute_gemm_after_reduce::<Alg, Backend>(
        ctx,
        &plan.gemm,
        gemm_plan,
        subs_c,
        a_ref,
        b_ref,
        alpha,
        beta,
        output,
        pool,
        lazy_output,
    )
}

/// Execute pre-reduction of unique-only axes using a pre-computed plan.
fn execute_reduce_with_plan<Alg, Backend>(
    ctx: &mut Backend::Context,
    reduce: &ReducePlan,
    tensor: &Tensor<Alg::Scalar>,
    pool: &mut BufferPool<Alg::Scalar>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let memory_space = tensor.logical_memory_space();
    let mut reduced = alloc_tensor_from_pool::<Alg::Scalar>(&reduce.out_shape, memory_space, pool);
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

/// Execute GEMM fallback after pre-reduction has already been applied.
///
/// Takes already-reduced tensors (a_ref, b_ref) and performs:
/// prepare_one_operand + BatchedGemm + reshape + permute to output.
///
/// When `lazy_output` is true and `alpha=1, beta=0`, skips the physical
/// permute and returns a zero-copy view with rearranged strides.
fn execute_gemm_after_reduce<Alg, Backend>(
    ctx: &mut Backend::Context,
    plan: &GemmPlan,
    gemm_plan: Option<&Backend::Plan>,
    subs_c: &[u32],
    a_ref: &Tensor<Alg::Scalar>,
    b_ref: &Tensor<Alg::Scalar>,
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
    pool: &mut BufferPool<Alg::Scalar>,
    lazy_output: bool,
) -> Result<()>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let memory_space = a_ref.logical_memory_space();
    let nb = plan.batch_sizes.len();

    // Prepare GEMM operands with fusability check
    // Layout: [g1, g2, batch...] — batch dims are trailing (col-major friendly)
    #[cfg(feature = "profile-dispatch")]
    let _prep_t0 = std::time::Instant::now();
    let a_prepared = prepare_one_operand::<Alg, Backend>(
        ctx,
        a_ref,
        &plan.subs_a,
        &plan.target_a,
        nb,
        plan.lo_modes.len(),
        plan.sum_modes.len(),
        &plan.a_gemm_shape,
        pool,
    )?;
    let b_prepared = prepare_one_operand::<Alg, Backend>(
        ctx,
        b_ref,
        &plan.subs_b,
        &plan.target_b,
        nb,
        plan.sum_modes.len(),
        plan.ro_modes.len(),
        &plan.b_gemm_shape,
        pool,
    )?;
    #[cfg(feature = "profile-dispatch")]
    PREPARE_NS.fetch_add(_prep_t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

    // Try to write GEMM directly into output (skip temp buffer).
    // Requires: no final permute needed AND output's lo/ro dims are fusable.
    let n_lo = plan.lo_modes.len();
    let n_ro = plan.ro_modes.len();
    let c_direct = !plan.needs_final_permute && {
        let c_dims = output.dims();
        let c_strides = output.strides();
        let g1 = strided_perm::try_fuse_group(&c_dims[..n_lo], &c_strides[..n_lo]);
        let g2 =
            strided_perm::try_fuse_group(&c_dims[n_lo..n_lo + n_ro], &c_strides[n_lo..n_lo + n_ro]);
        g1.is_some() && g2.is_some()
    };

    if c_direct {
        // Direct write: fuse output's lo→M, ro→N dims and write GEMM directly.
        //
        // We must take sole ownership of the underlying buffer so that
        // Arc::get_mut succeeds when the GEMM backend requests mutable
        // access.  `view_as_strided` clones the Arc, so we swap output
        // with a placeholder, create the fused view, and drop the
        // intermediate to bring the refcount back to 1.
        let c_dims = output.dims().to_vec();
        let c_strides = output.strides().to_vec();
        let (_, m_stride) =
            strided_perm::try_fuse_group(&c_dims[..n_lo], &c_strides[..n_lo]).unwrap();
        let (_, n_stride) =
            strided_perm::try_fuse_group(&c_dims[n_lo..n_lo + n_ro], &c_strides[n_lo..n_lo + n_ro])
                .unwrap();
        let mut fused_dims = Vec::with_capacity(2 + nb);
        let mut fused_strides = Vec::with_capacity(2 + nb);
        fused_dims.push(plan.m);
        fused_strides.push(m_stride);
        fused_dims.push(plan.n);
        fused_strides.push(n_stride);
        fused_dims.extend_from_slice(&c_dims[n_lo + n_ro..]);
        fused_strides.extend_from_slice(&c_strides[n_lo + n_ro..]);

        // Swap output with a tiny placeholder to get sole Arc ownership.
        let placeholder = alloc_tensor_from_pool::<Alg::Scalar>(&[], memory_space, pool);
        let out_tensor = std::mem::replace(output, placeholder);
        let mut c_fused = out_tensor.view_as_strided(fused_dims, fused_strides)?;
        drop(out_tensor); // Arc refcount → 1

        let owned_plan;
        let prim_plan: &Backend::Plan = if let Some(gp) = gemm_plan {
            gp
        } else {
            let desc = PrimDescriptor::BatchedGemm {
                batch_dims: plan.batch_sizes.clone(),
                m: plan.m,
                n: plan.n,
                k: plan.k,
            };
            let shapes = [a_prepared.dims(), b_prepared.dims(), c_fused.dims()];
            owned_plan = Backend::plan(ctx, &desc, &shapes)?;
            &owned_plan
        };
        #[cfg(feature = "profile-dispatch")]
        let _gemm_t0 = std::time::Instant::now();
        Backend::execute(
            ctx,
            prim_plan,
            alpha,
            &[&a_prepared, &b_prepared],
            beta,
            &mut c_fused,
        )?;
        #[cfg(feature = "profile-dispatch")]
        {
            GEMM_NS.fetch_add(_gemm_t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            let batch_total: usize = plan.batch_sizes.iter().product();
            let flops = 2 * batch_total * plan.m * plan.n * plan.k;
            GEMM_FLOPS.fetch_add(flops as u64, Ordering::Relaxed);
            if plan.m * plan.n * plan.k <= 64 {
                GEMM_CALLS_TINY.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Restore original shape and write back to output.
        let restored = c_fused.view_as_strided(c_dims, c_strides)?;
        drop(c_fused); // Arc refcount → 1
        *output = restored;
    } else {
        // Fallback: GEMM into temp buffer, then permute/copy to output.
        let mut temp =
            alloc_tensor_from_pool::<Alg::Scalar>(&plan.c_gemm_shape, memory_space, pool);
        let owned_plan;
        let prim_plan: &Backend::Plan = if let Some(gp) = gemm_plan {
            gp
        } else {
            let desc = PrimDescriptor::BatchedGemm {
                batch_dims: plan.batch_sizes.clone(),
                m: plan.m,
                n: plan.n,
                k: plan.k,
            };
            let shapes = [a_prepared.dims(), b_prepared.dims(), temp.dims()];
            owned_plan = Backend::plan(ctx, &desc, &shapes)?;
            &owned_plan
        };
        #[cfg(feature = "profile-dispatch")]
        let _gemm_t0 = std::time::Instant::now();
        Backend::execute(
            ctx,
            prim_plan,
            Alg::Scalar::one(),
            &[&a_prepared, &b_prepared],
            Alg::Scalar::zero(),
            &mut temp,
        )?;
        #[cfg(feature = "profile-dispatch")]
        {
            GEMM_NS.fetch_add(_gemm_t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            let batch_total: usize = plan.batch_sizes.iter().product();
            let flops = 2 * batch_total * plan.m * plan.n * plan.k;
            GEMM_FLOPS.fetch_add(flops as u64, Ordering::Relaxed);
            if plan.m * plan.n * plan.k <= 64 {
                GEMM_CALLS_TINY.fetch_add(1, Ordering::Relaxed);
            }
        }

        let temp_expanded = temp.reshape(&plan.expanded_shape)?;

        #[cfg(feature = "profile-dispatch")]
        let _perm_t0 = std::time::Instant::now();
        if lazy_output && alpha == Alg::Scalar::one() && beta == Alg::Scalar::zero() {
            // Lazy permute: zero-copy view with rearranged strides.
            // temp data moves to output — cannot return to pool.
            if !plan.needs_final_permute {
                *output = temp_expanded;
            } else {
                let perm = compute_permutation(&plan.canonical_modes, subs_c)
                    .map_err(|e| Error::InvalidArgument(e))?;
                *output = temp_expanded.permute(&perm)?;
            }
        } else {
            // Physical permute (with optional alpha/beta scaling).
            let desc = PrimDescriptor::Permute {
                modes_a: plan.canonical_modes.clone(),
                modes_b: subs_c.to_vec(),
            };
            let shapes = [temp_expanded.dims(), output.dims()];
            let prim_plan = Backend::plan(ctx, &desc, &shapes)?;
            Backend::execute(ctx, &prim_plan, alpha, &[&temp_expanded], beta, output)?;
            // Return temp buffer to pool (data was copied to output).
            drop(temp_expanded);
            if let Some(data) = temp.try_into_data_vec() {
                pool.return_buf(data);
            }
        }
        #[cfg(feature = "profile-dispatch")]
        PERMUTE_NS.fetch_add(_perm_t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    // Return prepared operand buffers to pool (no longer needed after GEMM).
    if let Some(data) = a_prepared.try_into_data_vec() {
        pool.return_buf(data);
    }
    if let Some(data) = b_prepared.try_into_data_vec() {
        pool.return_buf(data);
    }

    #[cfg(feature = "profile-dispatch")]
    CALL_COUNT.fetch_add(1, Ordering::Relaxed);

    Ok(())
}
