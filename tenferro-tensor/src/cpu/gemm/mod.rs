use num_traits::{One, Zero};
use smallvec::SmallVec;

use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::config::DotGeneralConfig;
use crate::cpu::structural::typed_transpose;
use crate::types::{col_major_strides, Buffer, TypedTensor};
use crate::Error;

#[cfg(feature = "cpu-blas")]
mod blas_gemm;
#[cfg(feature = "cpu-faer")]
mod faer_gemm;

#[cfg(feature = "cpu-blas")]
use blas_gemm::BlasGemm;
#[cfg(feature = "cpu-faer")]
use faer_gemm::FaerGemm;

struct GemmDims {
    m: usize,
    n: usize,
    k: usize,
    batch_total: usize,
    a_rs: isize,
    a_cs: isize,
    a_bs: isize,
    b_rs: isize,
    b_cs: isize,
    b_bs: isize,
    c_rs: isize,
    c_cs: isize,
    c_bs: isize,
    out_shape: SmallVec<[usize; 8]>,
}

fn validate_axis_list(
    op: &'static str,
    role: &'static str,
    axes: &[usize],
    rank: usize,
) -> crate::Result<()> {
    let mut seen: SmallVec<[bool; 8]> = smallvec::smallvec![false; rank];
    for &axis in axes {
        if axis >= rank {
            return Err(Error::AxisOutOfBounds { op, axis, rank });
        }
        if seen[axis] {
            return Err(Error::DuplicateAxis { op, axis, role });
        }
        seen[axis] = true;
    }
    Ok(())
}

fn validate_role_disjoint(
    op: &'static str,
    first_role: &'static str,
    first_axes: &[usize],
    second_role: &'static str,
    second_axes: &[usize],
) -> crate::Result<()> {
    for &axis in first_axes {
        if second_axes.contains(&axis) {
            return Err(Error::AxisRoleConflict {
                op,
                axis,
                first_role,
                second_role,
            });
        }
    }
    Ok(())
}

fn validate_dot_general<T>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
) -> crate::Result<()> {
    const OP: &str = "dot_general";

    if config.lhs_contracting_dims.len() != config.rhs_contracting_dims.len() {
        return Err(Error::InvalidConfig {
            op: OP,
            message: "lhs/rhs contracting dim counts differ".into(),
        });
    }
    if config.lhs_batch_dims.len() != config.rhs_batch_dims.len() {
        return Err(Error::InvalidConfig {
            op: OP,
            message: "lhs/rhs batch dim counts differ".into(),
        });
    }

    let lhs_rank = lhs.shape.len();
    let rhs_rank = rhs.shape.len();
    validate_axis_list(
        OP,
        "lhs_contracting",
        &config.lhs_contracting_dims,
        lhs_rank,
    )?;
    validate_axis_list(
        OP,
        "rhs_contracting",
        &config.rhs_contracting_dims,
        rhs_rank,
    )?;
    validate_axis_list(OP, "lhs_batch", &config.lhs_batch_dims, lhs_rank)?;
    validate_axis_list(OP, "rhs_batch", &config.rhs_batch_dims, rhs_rank)?;
    validate_role_disjoint(
        OP,
        "lhs_contracting",
        &config.lhs_contracting_dims,
        "lhs_batch",
        &config.lhs_batch_dims,
    )?;
    validate_role_disjoint(
        OP,
        "rhs_contracting",
        &config.rhs_contracting_dims,
        "rhs_batch",
        &config.rhs_batch_dims,
    )?;

    for (&lhs_axis, &rhs_axis) in config
        .lhs_contracting_dims
        .iter()
        .zip(&config.rhs_contracting_dims)
    {
        if lhs.shape[lhs_axis] != rhs.shape[rhs_axis] {
            return Err(Error::InvalidConfig {
                op: OP,
                message: format!(
                    "contracting dim size mismatch: lhs axis {lhs_axis}={} rhs axis {rhs_axis}={}",
                    lhs.shape[lhs_axis], rhs.shape[rhs_axis]
                ),
            });
        }
    }
    for (&lhs_axis, &rhs_axis) in config.lhs_batch_dims.iter().zip(&config.rhs_batch_dims) {
        if lhs.shape[lhs_axis] != rhs.shape[rhs_axis] {
            return Err(Error::InvalidConfig {
                op: OP,
                message: format!(
                    "batch dim size mismatch: lhs axis {lhs_axis}={} rhs axis {rhs_axis}={}",
                    lhs.shape[lhs_axis], rhs.shape[rhs_axis]
                ),
            });
        }
    }

    Ok(())
}

fn try_fuse_dims(shapes: &[usize], strides: &[isize]) -> Option<(usize, isize)> {
    if shapes.is_empty() {
        return Some((1, 0));
    }
    if shapes.len() == 1 {
        return Some((shapes[0], strides[0]));
    }
    let mut dims: SmallVec<[(usize, isize); 8]> = shapes
        .iter()
        .copied()
        .zip(strides.iter().copied())
        .collect();
    dims.sort_by_key(|&(_, stride)| stride.unsigned_abs());
    let base_stride = dims[0].1;
    let mut expected = base_stride;
    for (shape, stride) in dims {
        if stride != expected {
            return None;
        }
        expected = stride.checked_mul(shape as isize)?;
    }
    Some((shapes.iter().product(), base_stride))
}

fn stride_sort_order(strides: &[isize]) -> SmallVec<[usize; 8]> {
    let mut order: SmallVec<[usize; 8]> = (0..strides.len()).collect();
    order.sort_by_key(|&idx| strides[idx].unsigned_abs());
    order
}

fn is_identity_order(order: &[usize]) -> bool {
    order.iter().enumerate().all(|(idx, &value)| idx == value)
}

/// Compute permutations that reorder lhs/rhs into canonical GEMM layout.
///
/// Canonical col-major layouts (batch trailing):
/// - lhs: `[free..., contract..., batch...]`
/// - rhs: `[contract..., free..., batch...]`
fn canonical_gemm_layout(
    config: &DotGeneralConfig,
    lhs_rank: usize,
    rhs_rank: usize,
) -> (SmallVec<[usize; 8]>, SmallVec<[usize; 8]>, DotGeneralConfig) {
    let lhs_free: SmallVec<[usize; 8]> = (0..lhs_rank)
        .filter(|d| !config.lhs_contracting_dims.contains(d) && !config.lhs_batch_dims.contains(d))
        .collect();
    let rhs_free: SmallVec<[usize; 8]> = (0..rhs_rank)
        .filter(|d| !config.rhs_contracting_dims.contains(d) && !config.rhs_batch_dims.contains(d))
        .collect();

    let mut lhs_perm = SmallVec::<[usize; 8]>::with_capacity(lhs_rank);
    lhs_perm.extend_from_slice(&lhs_free);
    lhs_perm.extend_from_slice(&config.lhs_contracting_dims);
    lhs_perm.extend_from_slice(&config.lhs_batch_dims);

    let mut rhs_perm = SmallVec::<[usize; 8]>::with_capacity(rhs_rank);
    rhs_perm.extend_from_slice(&config.rhs_contracting_dims);
    rhs_perm.extend_from_slice(&rhs_free);
    rhs_perm.extend_from_slice(&config.rhs_batch_dims);

    let nf_lhs = lhs_free.len();
    let nc = config.lhs_contracting_dims.len();
    let nb = config.lhs_batch_dims.len();
    let nf_rhs = rhs_free.len();

    let new_config = DotGeneralConfig {
        lhs_contracting_dims: (nf_lhs..nf_lhs + nc).collect(),
        rhs_contracting_dims: (0..nc).collect(),
        lhs_batch_dims: (nf_lhs + nc..nf_lhs + nc + nb).collect(),
        rhs_batch_dims: (nc + nf_rhs..nc + nf_rhs + nb).collect(),
        lhs_rank,
        rhs_rank,
    };

    (lhs_perm, rhs_perm, new_config)
}

fn is_identity_perm(perm: &[usize]) -> bool {
    perm.iter().enumerate().all(|(i, &p)| i == p)
}

fn analyse_gemm<T>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
) -> Option<GemmDims> {
    let lhs_rank = lhs.shape.len();
    let rhs_rank = rhs.shape.len();

    let lhs_free: SmallVec<[usize; 8]> = (0..lhs_rank)
        .filter(|d| !config.lhs_contracting_dims.contains(d) && !config.lhs_batch_dims.contains(d))
        .collect();
    let rhs_free: SmallVec<[usize; 8]> = (0..rhs_rank)
        .filter(|d| !config.rhs_contracting_dims.contains(d) && !config.rhs_batch_dims.contains(d))
        .collect();

    let lhs_strides: SmallVec<[isize; 8]> = col_major_strides(&lhs.shape).into_iter().collect();
    let rhs_strides: SmallVec<[isize; 8]> = col_major_strides(&rhs.shape).into_iter().collect();

    let batch_shapes: SmallVec<[usize; 8]> = config
        .lhs_batch_dims
        .iter()
        .map(|&d| lhs.shape[d])
        .collect();
    let batch_total: usize = batch_shapes.iter().product();

    let lhs_free_shapes: SmallVec<[usize; 8]> = lhs_free.iter().map(|&d| lhs.shape[d]).collect();
    let rhs_free_shapes: SmallVec<[usize; 8]> = rhs_free.iter().map(|&d| rhs.shape[d]).collect();
    let contract_shapes: SmallVec<[usize; 8]> = config
        .lhs_contracting_dims
        .iter()
        .map(|&d| lhs.shape[d])
        .collect();

    let m: usize = lhs_free_shapes.iter().product();
    let n: usize = rhs_free_shapes.iter().product();
    let k: usize = contract_shapes.iter().product();

    let lhs_free_strides: SmallVec<[isize; 8]> = lhs_free.iter().map(|&d| lhs_strides[d]).collect();
    let rhs_free_strides: SmallVec<[isize; 8]> = rhs_free.iter().map(|&d| rhs_strides[d]).collect();
    let lhs_contract_strides: SmallVec<[isize; 8]> = config
        .lhs_contracting_dims
        .iter()
        .map(|&d| lhs_strides[d])
        .collect();
    let rhs_contract_strides: SmallVec<[isize; 8]> = config
        .rhs_contracting_dims
        .iter()
        .map(|&d| rhs_strides[d])
        .collect();
    let lhs_batch_strides: SmallVec<[isize; 8]> = config
        .lhs_batch_dims
        .iter()
        .map(|&d| lhs_strides[d])
        .collect();
    let rhs_batch_strides: SmallVec<[isize; 8]> = config
        .rhs_batch_dims
        .iter()
        .map(|&d| rhs_strides[d])
        .collect();

    if !is_identity_order(&stride_sort_order(&lhs_free_strides))
        || !is_identity_order(&stride_sort_order(&rhs_free_strides))
        || !is_identity_order(&stride_sort_order(&lhs_batch_strides))
        || !is_identity_order(&stride_sort_order(&rhs_batch_strides))
        || stride_sort_order(&lhs_contract_strides) != stride_sort_order(&rhs_contract_strides)
    {
        return None;
    }

    let (_, a_rs) = try_fuse_dims(&lhs_free_shapes, &lhs_free_strides)?;
    let (_, a_cs) = try_fuse_dims(&contract_shapes, &lhs_contract_strides)?;
    let (_, b_rs) = try_fuse_dims(&contract_shapes, &rhs_contract_strides)?;
    let (_, b_cs) = try_fuse_dims(&rhs_free_shapes, &rhs_free_strides)?;
    let (_, a_bs) = try_fuse_dims(&batch_shapes, &lhs_batch_strides)?;
    let (_, b_bs) = try_fuse_dims(&batch_shapes, &rhs_batch_strides)?;

    let mut out_shape = SmallVec::<[usize; 8]>::new();
    out_shape.extend_from_slice(&lhs_free_shapes);
    out_shape.extend_from_slice(&rhs_free_shapes);
    out_shape.extend_from_slice(&batch_shapes);

    let out_strides: SmallVec<[isize; 8]> = col_major_strides(&out_shape).into_iter().collect();
    let nm = lhs_free_shapes.len();
    let nn = rhs_free_shapes.len();
    let out_m_shapes = &out_shape[..nm];
    let out_m_strides = &out_strides[..nm];
    let out_n_shapes = &out_shape[nm..nm + nn];
    let out_n_strides = &out_strides[nm..nm + nn];
    let out_b_shapes = &out_shape[nm + nn..];
    let out_b_strides = &out_strides[nm + nn..];

    let (_, c_rs) = try_fuse_dims(out_m_shapes, out_m_strides)?;
    let (_, c_cs) = try_fuse_dims(out_n_shapes, out_n_strides)?;
    let (_, c_bs) = try_fuse_dims(out_b_shapes, out_b_strides)?;

    Some(GemmDims {
        m,
        n,
        k,
        batch_total,
        a_rs,
        a_cs,
        a_bs,
        b_rs,
        b_cs,
        b_bs,
        c_rs,
        c_cs,
        c_bs,
        out_shape,
    })
}

#[cfg(feature = "cpu-faer")]
pub(crate) fn dot_general<T>(
    buffers: &mut BufferPool,
    ctx: &crate::cpu::CpuContext,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
) -> crate::Result<TypedTensor<T>>
where
    T: FaerGemm + PoolScalar + Copy + Clone + Zero + One + PartialEq,
{
    validate_dot_general(lhs, rhs, config)?;
    if let Some(result) = typed_faer_gemm(buffers, ctx, lhs, rhs, config) {
        return Ok(result);
    }
    let (lhs_perm, rhs_perm, new_config) =
        canonical_gemm_layout(config, lhs.shape.len(), rhs.shape.len());
    let lhs_canon = if is_identity_perm(&lhs_perm) {
        std::borrow::Cow::Borrowed(lhs)
    } else {
        std::borrow::Cow::Owned(typed_transpose(lhs, &lhs_perm)?)
    };
    let rhs_canon = if is_identity_perm(&rhs_perm) {
        std::borrow::Cow::Borrowed(rhs)
    } else {
        std::borrow::Cow::Owned(typed_transpose(rhs, &rhs_perm)?)
    };
    typed_faer_gemm(buffers, ctx, &lhs_canon, &rhs_canon, &new_config).ok_or_else(|| {
        Error::BackendFailure {
            op: "dot_general",
            message: "CPU GEMM requires host-backed canonical inputs".into(),
        }
    })
}

#[cfg(feature = "cpu-faer")]
fn typed_faer_gemm<T>(
    buffers: &mut BufferPool,
    ctx: &crate::cpu::CpuContext,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
) -> Option<TypedTensor<T>>
where
    T: FaerGemm + PoolScalar + Copy + Clone + Zero + One + PartialEq,
{
    let dims = analyse_gemm(lhs, rhs, config)?;
    let out_n: usize = dims.out_shape.iter().product();
    if dims.m == 0 || dims.n == 0 || dims.k == 0 || dims.batch_total == 0 {
        return Some(TypedTensor {
            buffer: Buffer::Host(vec![T::zero(); out_n]),
            shape: dims.out_shape.into_vec(),
            placement: lhs.placement.clone(),
        });
    }

    let a_data = match &lhs.buffer {
        Buffer::Host(v) => v.as_ptr(),
        Buffer::Backend(_) => return None,
        #[cfg(feature = "cubecl")]
        Buffer::Cubecl(_) => panic!("GPU tensor (Buffer::Cubecl) passed to CPU backend. Use cubecl::download_tensor() to transfer to CPU first."),
    };
    let b_data = match &rhs.buffer {
        Buffer::Host(v) => v.as_ptr(),
        Buffer::Backend(_) => return None,
        #[cfg(feature = "cubecl")]
        Buffer::Cubecl(_) => panic!("GPU tensor (Buffer::Cubecl) passed to CPU backend. Use cubecl::download_tensor() to transfer to CPU first."),
    };

    // SAFETY: this GEMM path uses beta = 0 and overwrites every output element.
    let mut out_data: Vec<T> = unsafe { T::pool_acquire(buffers, out_n) };
    let c_ptr = out_data.as_mut_ptr();

    for batch in 0..dims.batch_total {
        let a_off = batch as isize * dims.a_bs;
        let b_off = batch as isize * dims.b_bs;
        let c_off = batch as isize * dims.c_bs;
        unsafe {
            T::strided_gemm(
                ctx,
                T::one(),
                a_data.offset(a_off),
                dims.m,
                dims.k,
                dims.a_rs,
                dims.a_cs,
                b_data.offset(b_off),
                dims.n,
                dims.b_rs,
                dims.b_cs,
                T::zero(),
                c_ptr.offset(c_off),
                dims.c_rs,
                dims.c_cs,
            );
        }
    }

    Some(TypedTensor {
        buffer: Buffer::Host(out_data),
        shape: dims.out_shape.into_vec(),
        placement: lhs.placement.clone(),
    })
}

#[cfg(feature = "cpu-blas")]
pub(crate) fn dot_general<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
) -> crate::Result<TypedTensor<T>>
where
    T: BlasGemm + PoolScalar + Copy + Clone + Zero + One,
{
    validate_dot_general(lhs, rhs, config)?;
    if let Some(result) = typed_blas_gemm(buffers, lhs, rhs, config) {
        return Ok(result);
    }
    let (lhs_perm, rhs_perm, new_config) =
        canonical_gemm_layout(config, lhs.shape.len(), rhs.shape.len());
    let lhs_canon = if is_identity_perm(&lhs_perm) {
        std::borrow::Cow::Borrowed(lhs)
    } else {
        std::borrow::Cow::Owned(typed_transpose(lhs, &lhs_perm)?)
    };
    let rhs_canon = if is_identity_perm(&rhs_perm) {
        std::borrow::Cow::Borrowed(rhs)
    } else {
        std::borrow::Cow::Owned(typed_transpose(rhs, &rhs_perm)?)
    };
    typed_blas_gemm(buffers, &lhs_canon, &rhs_canon, &new_config).ok_or_else(|| {
        Error::BackendFailure {
            op: "dot_general",
            message: "CPU GEMM requires host-backed canonical inputs".into(),
        }
    })
}

#[cfg(feature = "cpu-blas")]
fn typed_blas_gemm<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
) -> Option<TypedTensor<T>>
where
    T: BlasGemm + PoolScalar + Copy + Clone + Zero + One,
{
    let dims = analyse_gemm(lhs, rhs, config)?;
    let out_n: usize = dims.out_shape.iter().product();
    if dims.m == 0 || dims.n == 0 || dims.k == 0 || dims.batch_total == 0 {
        return Some(TypedTensor {
            buffer: Buffer::Host(vec![T::zero(); out_n]),
            shape: dims.out_shape.into_vec(),
            placement: lhs.placement.clone(),
        });
    }

    let a_ok = dims.a_rs == 1 || dims.a_cs == 1;
    let b_ok = dims.b_rs == 1 || dims.b_cs == 1;
    let c_ok = dims.c_rs == 1;
    if !a_ok || !b_ok || !c_ok {
        return None;
    }

    let a_data = match &lhs.buffer {
        Buffer::Host(v) => v.as_ptr(),
        Buffer::Backend(_) => return None,
        #[cfg(feature = "cubecl")]
        Buffer::Cubecl(_) => panic!("GPU tensor (Buffer::Cubecl) passed to CPU backend. Use cubecl::download_tensor() to transfer to CPU first."),
    };
    let b_data = match &rhs.buffer {
        Buffer::Host(v) => v.as_ptr(),
        Buffer::Backend(_) => return None,
        #[cfg(feature = "cubecl")]
        Buffer::Cubecl(_) => panic!("GPU tensor (Buffer::Cubecl) passed to CPU backend. Use cubecl::download_tensor() to transfer to CPU first."),
    };

    // SAFETY: each batch GEMM writes its full output block with beta = 0.
    let mut out: Vec<T> = unsafe { T::pool_acquire(buffers, out_n) };
    let c_ptr = out.as_mut_ptr();

    for batch in 0..dims.batch_total {
        let a_off = batch as isize * dims.a_bs;
        let b_off = batch as isize * dims.b_bs;
        let c_off = batch as isize * dims.c_bs;
        unsafe {
            T::strided_gemm(
                T::one(),
                a_data.offset(a_off),
                dims.m,
                dims.k,
                dims.a_rs,
                dims.a_cs,
                b_data.offset(b_off),
                dims.n,
                dims.b_rs,
                dims.b_cs,
                T::zero(),
                c_ptr.offset(c_off),
                dims.c_rs,
                dims.c_cs,
            );
        }
    }

    Some(TypedTensor {
        buffer: Buffer::Host(out),
        shape: dims.out_shape.into_vec(),
        placement: lhs.placement.clone(),
    })
}

#[cfg(test)]
mod tests;
