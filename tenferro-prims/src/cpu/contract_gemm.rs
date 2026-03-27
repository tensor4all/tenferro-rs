use strided_view::{StridedView, StridedViewMut};
use tenferro_device::Result;

use crate::CpuContext;

#[cfg(feature = "gemm-blas")]
use super::gemm_support::BlasGemm;
#[cfg(feature = "gemm-faer")]
use super::gemm_support::FaerGemm;
use super::layout_fusion::try_fuse_group_in_order;
use super::plan::ContractGemmSpec;
#[cfg(feature = "gemm-blas")]
use tenferro_algebra::Scalar;

#[derive(Debug, Clone)]
pub(super) struct GemmLayout {
    pub(super) batch_total: usize,
    pub(super) m: usize,
    pub(super) n: usize,
    pub(super) k: usize,
    pub(super) a_ms: isize,
    pub(super) a_ks: isize,
    pub(super) b_ks: isize,
    pub(super) b_ns: isize,
    pub(super) c_ms: isize,
    pub(super) c_ns: isize,
    pub(super) a_bs: isize,
    pub(super) b_bs: isize,
    pub(super) c_bs: isize,
}

pub(super) fn reordered_dims_strides(
    modes_src: &[u32],
    dims_src: &[usize],
    strides_src: &[isize],
    target: &[u32],
) -> Option<(Vec<usize>, Vec<isize>)> {
    let perm: Vec<usize> = target
        .iter()
        .map(|mode| modes_src.iter().position(|candidate| candidate == mode))
        .collect::<Option<_>>()?;
    let dims = perm.iter().map(|&p| dims_src[p]).collect();
    let strides = perm.iter().map(|&p| strides_src[p]).collect();
    Some((dims, strides))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn compute_layout(
    a_dims_src: &[usize],
    a_strides_src: &[isize],
    b_dims_src: &[usize],
    b_strides_src: &[isize],
    c_dims_src: &[usize],
    c_strides_src: &[isize],
    modes_a: &[u32],
    modes_b: &[u32],
    modes_c: &[u32],
    spec: &ContractGemmSpec,
) -> Option<GemmLayout> {
    let (a_dims, a_strides) =
        reordered_dims_strides(modes_a, a_dims_src, a_strides_src, &spec.a_target)?;
    let (b_dims, b_strides) =
        reordered_dims_strides(modes_b, b_dims_src, b_strides_src, &spec.b_target)?;
    let (c_dims, c_strides) =
        reordered_dims_strides(modes_c, c_dims_src, c_strides_src, &spec.c_target)?;

    let nb = spec.batch_modes.len();
    let nm = spec.m_modes.len();
    let nk = spec.k_modes.len();
    let nn = spec.n_modes.len();

    let (batch_total, a_bs, b_bs, c_bs) = if nb == 0 {
        (1usize, 0isize, 0isize, 0isize)
    } else {
        let (ta, sa) = try_fuse_group_in_order(&a_dims[..nb], &a_strides[..nb])?;
        let (tb, sb) = try_fuse_group_in_order(&b_dims[..nb], &b_strides[..nb])?;
        let (tc, sc) = try_fuse_group_in_order(&c_dims[..nb], &c_strides[..nb])?;
        if ta != tb || ta != tc {
            return None;
        }
        (ta, sa, sb, sc)
    };

    let (m_raw, a_ms) = if nm == 0 {
        (1usize, 0isize)
    } else {
        try_fuse_group_in_order(&a_dims[nb..nb + nm], &a_strides[nb..nb + nm])?
    };
    let (m_chk, c_ms) = if nm == 0 {
        (1usize, 0isize)
    } else {
        try_fuse_group_in_order(&c_dims[nb..nb + nm], &c_strides[nb..nb + nm])?
    };
    if m_raw != m_chk {
        return None;
    }

    let (k_raw, a_ks) = if nk == 0 {
        (1usize, 0isize)
    } else {
        try_fuse_group_in_order(
            &a_dims[nb + nm..nb + nm + nk],
            &a_strides[nb + nm..nb + nm + nk],
        )?
    };
    let (k_chk, b_ks) = if nk == 0 {
        (1usize, 0isize)
    } else {
        try_fuse_group_in_order(&b_dims[nb..nb + nk], &b_strides[nb..nb + nk])?
    };
    if k_raw != k_chk {
        return None;
    }

    let (n_raw, b_ns) = if nn == 0 {
        (1usize, 0isize)
    } else {
        try_fuse_group_in_order(
            &b_dims[nb + nk..nb + nk + nn],
            &b_strides[nb + nk..nb + nk + nn],
        )?
    };
    let (n_chk, c_ns) = if nn == 0 {
        (1usize, 0isize)
    } else {
        try_fuse_group_in_order(
            &c_dims[nb + nm..nb + nm + nn],
            &c_strides[nb + nm..nb + nm + nn],
        )?
    };
    if n_raw != n_chk {
        return None;
    }

    Some(GemmLayout {
        batch_total,
        m: m_raw.max(1),
        n: n_raw.max(1),
        k: k_raw.max(1),
        a_ms,
        a_ks,
        b_ks,
        b_ns,
        c_ms,
        c_ns,
        a_bs,
        b_bs,
        c_bs,
    })
}

#[cfg(feature = "gemm-faer")]
pub(super) fn run_strided<U: FaerGemm>(
    ctx: &CpuContext,
    alpha: U,
    a: &StridedView<U>,
    b: &StridedView<U>,
    beta: U,
    c: &mut StridedViewMut<U>,
    layout: &GemmLayout,
) -> Result<()> {
    let a_ptr = a.ptr();
    let b_ptr = b.ptr();
    let c_ptr = c.as_mut_ptr();
    let a_addr = a_ptr as usize;
    let b_addr = b_ptr as usize;
    let c_addr = c_ptr as usize;
    let mut a_off = 0isize;
    let mut b_off = 0isize;
    let mut c_off = 0isize;
    ctx.install(move || {
        let a_ptr = a_addr as *const U;
        let b_ptr = b_addr as *const U;
        let c_ptr = c_addr as *mut U;
        for _ in 0..layout.batch_total {
            unsafe {
                U::strided_gemm(
                    alpha,
                    a_ptr.offset(a_off),
                    layout.m,
                    layout.k,
                    layout.a_ms,
                    layout.a_ks,
                    b_ptr.offset(b_off),
                    layout.n,
                    layout.b_ks,
                    layout.b_ns,
                    beta,
                    c_ptr.offset(c_off),
                    layout.c_ms,
                    layout.c_ns,
                );
            }
            a_off += layout.a_bs;
            b_off += layout.b_bs;
            c_off += layout.c_bs;
        }
    });
    Ok(())
}

#[cfg(feature = "gemm-blas")]
pub(super) fn run_dense<U: Scalar>(
    alpha: U,
    a: &StridedView<U>,
    b: &StridedView<U>,
    beta: U,
    c: &mut StridedViewMut<U>,
    layout: &GemmLayout,
    gemm_fn: fn(U, &[U], &[U], U, &mut [U], usize, usize, usize) -> Result<()>,
) -> Result<()> {
    let GemmLayout {
        batch_total,
        m,
        n,
        k,
        a_ms,
        a_ks,
        b_ks,
        b_ns,
        c_ms,
        c_ns,
        a_bs,
        b_bs,
        c_bs,
    } = *layout;

    let mut a_mat = vec![U::zero(); m * k];
    let mut b_mat = vec![U::zero(); k * n];
    let mut c_mat = vec![U::zero(); m * n];

    let a_ptr = a.ptr();
    let b_ptr = b.ptr();
    let c_ptr = c.as_mut_ptr();
    let mut a_off = 0isize;
    let mut b_off = 0isize;
    let mut c_off = 0isize;

    for _ in 0..batch_total {
        for kk in 0..k {
            for i in 0..m {
                let off = a_off + i as isize * a_ms + kk as isize * a_ks;
                a_mat[i + kk * m] = unsafe { *a_ptr.offset(off) };
            }
        }
        for j in 0..n {
            for kk in 0..k {
                let off = b_off + kk as isize * b_ks + j as isize * b_ns;
                b_mat[kk + j * k] = unsafe { *b_ptr.offset(off) };
            }
        }
        if beta == U::zero() {
            c_mat.fill(U::zero());
        } else {
            for j in 0..n {
                for i in 0..m {
                    let off = c_off + i as isize * c_ms + j as isize * c_ns;
                    c_mat[i + j * m] = unsafe { *c_ptr.offset(off) };
                }
            }
        }

        gemm_fn(alpha, &a_mat, &b_mat, beta, &mut c_mat, m, n, k)?;

        for j in 0..n {
            for i in 0..m {
                let off = c_off + i as isize * c_ms + j as isize * c_ns;
                unsafe {
                    *c_ptr.offset(off) = c_mat[i + j * m];
                }
            }
        }

        a_off += a_bs;
        b_off += b_bs;
        c_off += c_bs;
    }
    Ok(())
}
