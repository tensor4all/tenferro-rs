use num_traits::{One, Zero};

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
    #[cfg(feature = "cpu-faer")]
    a_cs: isize,
    a_bs: isize,
    b_rs: isize,
    #[cfg(feature = "cpu-faer")]
    b_cs: isize,
    b_bs: isize,
    c_rs: isize,
    #[cfg(feature = "cpu-faer")]
    c_cs: isize,
    c_bs: isize,
    out_shape: Vec<usize>,
}

fn validate_axis_list(
    op: &'static str,
    role: &'static str,
    axes: &[usize],
    rank: usize,
) -> crate::Result<()> {
    let mut seen = vec![false; rank];
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
    if config.lhs_rank != lhs_rank {
        return Err(Error::RankMismatch {
            op: OP,
            expected: config.lhs_rank,
            actual: lhs_rank,
        });
    }
    if config.rhs_rank != rhs_rank {
        return Err(Error::RankMismatch {
            op: OP,
            expected: config.rhs_rank,
            actual: rhs_rank,
        });
    }
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
    let base_stride = strides[0];
    let mut expected = base_stride;
    for i in 0..shapes.len() {
        if strides[i] != expected {
            return None;
        }
        expected = strides[i].checked_mul(shapes[i] as isize)?;
    }
    Some((shapes.iter().product(), base_stride))
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
) -> (Vec<usize>, Vec<usize>, DotGeneralConfig) {
    let lhs_free: Vec<usize> = (0..lhs_rank)
        .filter(|d| !config.lhs_contracting_dims.contains(d) && !config.lhs_batch_dims.contains(d))
        .collect();
    let rhs_free: Vec<usize> = (0..rhs_rank)
        .filter(|d| !config.rhs_contracting_dims.contains(d) && !config.rhs_batch_dims.contains(d))
        .collect();

    let mut lhs_perm = Vec::with_capacity(lhs_rank);
    lhs_perm.extend_from_slice(&lhs_free);
    lhs_perm.extend_from_slice(&config.lhs_contracting_dims);
    lhs_perm.extend_from_slice(&config.lhs_batch_dims);

    let mut rhs_perm = Vec::with_capacity(rhs_rank);
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

    let lhs_free: Vec<usize> = (0..lhs_rank)
        .filter(|d| !config.lhs_contracting_dims.contains(d) && !config.lhs_batch_dims.contains(d))
        .collect();
    let rhs_free: Vec<usize> = (0..rhs_rank)
        .filter(|d| !config.rhs_contracting_dims.contains(d) && !config.rhs_batch_dims.contains(d))
        .collect();

    let lhs_strides = col_major_strides(&lhs.shape);
    let rhs_strides = col_major_strides(&rhs.shape);

    let batch_shapes: Vec<usize> = config
        .lhs_batch_dims
        .iter()
        .map(|&d| lhs.shape[d])
        .collect();
    let batch_total: usize = batch_shapes.iter().product();

    let lhs_free_shapes: Vec<usize> = lhs_free.iter().map(|&d| lhs.shape[d]).collect();
    let rhs_free_shapes: Vec<usize> = rhs_free.iter().map(|&d| rhs.shape[d]).collect();
    let contract_shapes: Vec<usize> = config
        .lhs_contracting_dims
        .iter()
        .map(|&d| lhs.shape[d])
        .collect();

    let m: usize = lhs_free_shapes.iter().product();
    let n: usize = rhs_free_shapes.iter().product();
    let k: usize = contract_shapes.iter().product();

    let lhs_free_strides: Vec<isize> = lhs_free.iter().map(|&d| lhs_strides[d]).collect();
    #[cfg(feature = "cpu-faer")]
    let rhs_free_strides: Vec<isize> = rhs_free.iter().map(|&d| rhs_strides[d]).collect();
    #[cfg(feature = "cpu-faer")]
    let lhs_contract_strides: Vec<isize> = config
        .lhs_contracting_dims
        .iter()
        .map(|&d| lhs_strides[d])
        .collect();
    let rhs_contract_strides: Vec<isize> = config
        .rhs_contracting_dims
        .iter()
        .map(|&d| rhs_strides[d])
        .collect();
    let lhs_batch_strides: Vec<isize> = config
        .lhs_batch_dims
        .iter()
        .map(|&d| lhs_strides[d])
        .collect();
    let rhs_batch_strides: Vec<isize> = config
        .rhs_batch_dims
        .iter()
        .map(|&d| rhs_strides[d])
        .collect();

    let (_, a_rs) = try_fuse_dims(&lhs_free_shapes, &lhs_free_strides)?;
    #[cfg(feature = "cpu-faer")]
    let (_, a_cs) = try_fuse_dims(&contract_shapes, &lhs_contract_strides)?;
    let (_, b_rs) = try_fuse_dims(&contract_shapes, &rhs_contract_strides)?;
    #[cfg(feature = "cpu-faer")]
    let (_, b_cs) = try_fuse_dims(&rhs_free_shapes, &rhs_free_strides)?;
    let (_, a_bs) = try_fuse_dims(&batch_shapes, &lhs_batch_strides)?;
    let (_, b_bs) = try_fuse_dims(&batch_shapes, &rhs_batch_strides)?;

    let mut out_shape = Vec::new();
    out_shape.extend_from_slice(&lhs_free_shapes);
    out_shape.extend_from_slice(&rhs_free_shapes);
    out_shape.extend_from_slice(&batch_shapes);

    let out_strides = col_major_strides(&out_shape);
    let nm = lhs_free_shapes.len();
    let nn = rhs_free_shapes.len();
    let out_m_shapes = &out_shape[..nm];
    let out_m_strides = &out_strides[..nm];
    #[cfg(feature = "cpu-faer")]
    let out_n_shapes = &out_shape[nm..nm + nn];
    #[cfg(feature = "cpu-faer")]
    let out_n_strides = &out_strides[nm..nm + nn];
    let out_b_shapes = &out_shape[nm + nn..];
    let out_b_strides = &out_strides[nm + nn..];

    let (_, c_rs) = try_fuse_dims(out_m_shapes, out_m_strides)?;
    #[cfg(feature = "cpu-faer")]
    let (_, c_cs) = try_fuse_dims(out_n_shapes, out_n_strides)?;
    let (_, c_bs) = try_fuse_dims(out_b_shapes, out_b_strides)?;

    Some(GemmDims {
        m,
        n,
        k,
        batch_total,
        a_rs,
        #[cfg(feature = "cpu-faer")]
        a_cs,
        a_bs,
        b_rs,
        #[cfg(feature = "cpu-faer")]
        b_cs,
        b_bs,
        c_rs,
        #[cfg(feature = "cpu-faer")]
        c_cs,
        c_bs,
        out_shape,
    })
}

#[cfg(feature = "cpu-faer")]
pub(crate) fn dot_general<T>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
) -> crate::Result<TypedTensor<T>>
where
    T: FaerGemm + Copy + Clone + Zero + One + PartialEq,
{
    validate_dot_general(lhs, rhs, config)?;
    if let Some(result) = typed_faer_gemm(lhs, rhs, config) {
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
    typed_faer_gemm(&lhs_canon, &rhs_canon, &new_config).ok_or_else(|| Error::BackendFailure {
        op: "dot_general",
        message: "CPU GEMM requires host-backed canonical inputs".into(),
    })
}

#[cfg(feature = "cpu-faer")]
fn typed_faer_gemm<T>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
) -> Option<TypedTensor<T>>
where
    T: FaerGemm + Copy + Clone + Zero + One + PartialEq,
{
    let dims = analyse_gemm(lhs, rhs, config)?;
    let out_n: usize = dims.out_shape.iter().product();
    if dims.m == 0 || dims.n == 0 || dims.k == 0 || dims.batch_total == 0 {
        return Some(TypedTensor {
            buffer: Buffer::Host(vec![T::zero(); out_n]),
            shape: dims.out_shape,
            placement: lhs.placement.clone(),
        });
    }

    let a_data = match &lhs.buffer {
        Buffer::Host(v) => v.as_ptr(),
        Buffer::Backend(_) => return None,
    };
    let b_data = match &rhs.buffer {
        Buffer::Host(v) => v.as_ptr(),
        Buffer::Backend(_) => return None,
    };

    let mut out_data = vec![T::zero(); out_n];
    let c_ptr = out_data.as_mut_ptr();

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
        buffer: Buffer::Host(out_data),
        shape: dims.out_shape,
        placement: lhs.placement.clone(),
    })
}

#[cfg(feature = "cpu-blas")]
pub(crate) fn dot_general<T>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
) -> crate::Result<TypedTensor<T>>
where
    T: BlasGemm + Copy + Clone + Zero + One,
{
    validate_dot_general(lhs, rhs, config)?;
    if let Some(result) = typed_blas_gemm(lhs, rhs, config) {
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
    typed_blas_gemm(&lhs_canon, &rhs_canon, &new_config).ok_or_else(|| Error::BackendFailure {
        op: "dot_general",
        message: "CPU GEMM requires host-backed canonical inputs".into(),
    })
}

#[cfg(feature = "cpu-blas")]
fn typed_blas_gemm<T>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
) -> Option<TypedTensor<T>>
where
    T: BlasGemm + Copy + Clone + Zero + One,
{
    let dims = analyse_gemm(lhs, rhs, config)?;
    let out_n: usize = dims.out_shape.iter().product();
    if dims.m == 0 || dims.n == 0 || dims.k == 0 || dims.batch_total == 0 {
        return Some(TypedTensor {
            buffer: Buffer::Host(vec![T::zero(); out_n]),
            shape: dims.out_shape,
            placement: lhs.placement.clone(),
        });
    }

    if dims.a_rs != 1 || dims.b_rs != 1 || dims.c_rs != 1 {
        return None;
    }

    let a_data = match &lhs.buffer {
        Buffer::Host(v) => v,
        Buffer::Backend(_) => return None,
    };
    let b_data = match &rhs.buffer {
        Buffer::Host(v) => v,
        Buffer::Backend(_) => return None,
    };

    let mut out = vec![T::zero(); dims.out_shape.iter().product()];
    let a_block = dims.m * dims.k;
    let b_block = dims.k * dims.n;
    let c_block = dims.m * dims.n;

    for batch in 0..dims.batch_total {
        let a_start = (batch as isize * dims.a_bs) as usize;
        let b_start = (batch as isize * dims.b_bs) as usize;
        let c_start = (batch as isize * dims.c_bs) as usize;
        T::contiguous_gemm(
            T::one(),
            &a_data[a_start..a_start + a_block],
            &b_data[b_start..b_start + b_block],
            T::zero(),
            &mut out[c_start..c_start + c_block],
            dims.m,
            dims.n,
            dims.k,
        );
    }

    Some(TypedTensor {
        buffer: Buffer::Host(out),
        shape: dims.out_shape,
        placement: lhs.placement.clone(),
    })
}

#[cfg(test)]
mod tests;
