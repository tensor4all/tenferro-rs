use strided_perm::try_fuse_group;
use strided_view::{StridedView, StridedViewMut};
use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};

use crate::infra::typed_dispatch::{
    cast_scalar_value, cast_strided_view, cast_strided_view_mut, dispatch_standard_scalar_type,
};
use crate::SemiringBinaryOp;

use super::plan::{build_contract_gemm_spec, ContractGemmSpec};

#[cfg(feature = "gemm-faer")]
use super::gemm_support::FaerGemm;

#[cfg(feature = "gemm-blas")]
use super::gemm_support::BlasGemm;

pub(super) fn execute_elementwise_binary<T: Scalar>(
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
    op: SemiringBinaryOp,
) -> Result<()> {
    let a = inputs[0];
    let b = inputs[1];

    if beta == T::zero() {
        let alpha_val = alpha;
        strided_kernel::zip_map2_into(output, a, b, move |a_val, b_val| {
            let fused = match op {
                SemiringBinaryOp::Add => a_val + b_val,
                SemiringBinaryOp::Mul => a_val * b_val,
            };
            alpha_val * fused
        })
        .map_err(|e| Error::DeviceError(e.to_string()))?;
    } else {
        let dims = output.dims().to_vec();
        crate::for_each_index(&dims, |idx| {
            let fused = match op {
                SemiringBinaryOp::Add => a.get(idx) + b.get(idx),
                SemiringBinaryOp::Mul => a.get(idx) * b.get(idx),
            };
            let val = alpha * fused;
            output.set(idx, val + beta * output.get(idx));
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_contract<T: Scalar>(
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
    modes_a: &[u32],
    modes_b: &[u32],
    modes_c: &[u32],
    cached_gemm_spec: Option<&ContractGemmSpec>,
) -> Result<()> {
    if let Some(done) = try_execute_contract_gemm(
        alpha,
        inputs,
        beta,
        output,
        modes_a,
        modes_b,
        modes_c,
        cached_gemm_spec,
    )? {
        return Ok(done);
    }

    let mut contracted_modes = Vec::new();
    for &mode in modes_a.iter().chain(modes_b.iter()) {
        if !modes_c.contains(&mode) && !contracted_modes.contains(&mode) {
            contracted_modes.push(mode);
        }
    }
    let mut contracted_dims = Vec::with_capacity(contracted_modes.len());
    for &mode in &contracted_modes {
        if let Some(a_pos) = modes_a.iter().position(|&mm| mm == mode) {
            contracted_dims.push(inputs[0].dims()[a_pos]);
        } else if let Some(b_pos) = modes_b.iter().position(|&mm| mm == mode) {
            contracted_dims.push(inputs[1].dims()[b_pos]);
        } else {
            return Err(Error::InvalidArgument(format!(
                "Contract reduction mode {mode} appears in neither input operand"
            )));
        }
    }
    let contracted_total: usize = if contracted_dims.is_empty() {
        1
    } else {
        contracted_dims.iter().product()
    };

    let a_axis_map: Vec<(u8, usize)> = modes_a
        .iter()
        .map(|&mode| {
            if let Some(c_pos) = modes_c.iter().position(|&m| m == mode) {
                (0, c_pos)
            } else if let Some(k_pos) = contracted_modes.iter().position(|&m| m == mode) {
                (1, k_pos)
            } else {
                unreachable!("every A-only mode absent from C must be reduced")
            }
        })
        .collect();
    let b_axis_map: Vec<(u8, usize)> = modes_b
        .iter()
        .map(|&mode| {
            if let Some(c_pos) = modes_c.iter().position(|&m| m == mode) {
                (0, c_pos)
            } else if let Some(k_pos) = contracted_modes.iter().position(|&m| m == mode) {
                (1, k_pos)
            } else {
                unreachable!("every B-only mode absent from C must be reduced")
            }
        })
        .collect();

    fn unflatten_into(mut flat: usize, dims: &[usize], out: &mut [usize]) {
        debug_assert_eq!(dims.len(), out.len());
        debug_assert!(dims.iter().all(|&d| d > 0));
        debug_assert!(flat < dims.iter().product::<usize>());
        for (i, &d) in dims.iter().enumerate() {
            out[i] = flat % d;
            flat /= d;
        }
    }

    let out_dims = output.dims().to_vec();
    let mut a_idx = vec![0usize; modes_a.len()];
    let mut b_idx = vec![0usize; modes_b.len()];
    let mut k_idx = vec![0usize; contracted_dims.len()];

    crate::for_each_index(&out_dims, |c_idx| {
        let mut sum = T::zero();
        for k_flat in 0..contracted_total {
            if !contracted_dims.is_empty() {
                unflatten_into(k_flat, &contracted_dims, &mut k_idx);
            }
            for (ax, &(src, pos)) in a_axis_map.iter().enumerate() {
                a_idx[ax] = if src == 0 { c_idx[pos] } else { k_idx[pos] };
            }
            for (ax, &(src, pos)) in b_axis_map.iter().enumerate() {
                b_idx[ax] = if src == 0 { c_idx[pos] } else { k_idx[pos] };
            }
            sum = sum + inputs[0].get(&a_idx) * inputs[1].get(&b_idx);
        }
        let old = if beta == T::zero() {
            T::zero()
        } else {
            beta * output.get(c_idx)
        };
        output.set(c_idx, alpha * sum + old);
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn try_execute_contract_gemm<T: Scalar + 'static>(
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
    modes_a: &[u32],
    modes_b: &[u32],
    modes_c: &[u32],
    cached_spec: Option<&ContractGemmSpec>,
) -> Result<Option<()>> {
    fn perm_for(target: &[u32], source: &[u32]) -> Option<Vec<usize>> {
        target
            .iter()
            .map(|m| source.iter().position(|x| x == m))
            .collect()
    }

    fn reordered_dims_strides(
        modes_src: &[u32],
        dims_src: &[usize],
        strides_src: &[isize],
        target: &[u32],
    ) -> Option<(Vec<usize>, Vec<isize>)> {
        let perm = perm_for(target, modes_src)?;
        let dims = perm.iter().map(|&p| dims_src[p]).collect();
        let strides = perm.iter().map(|&p| strides_src[p]).collect();
        Some((dims, strides))
    }

    struct GemmLayout {
        batch_total: usize,
        m: usize,
        n: usize,
        k: usize,
        a_ms: isize,
        a_ks: isize,
        b_ks: isize,
        b_ns: isize,
        c_ms: isize,
        c_ns: isize,
        a_bs: isize,
        b_bs: isize,
        c_bs: isize,
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_layout(
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
            let (ta, sa) = try_fuse_group(&a_dims[..nb], &a_strides[..nb])?;
            let (tb, sb) = try_fuse_group(&b_dims[..nb], &b_strides[..nb])?;
            let (tc, sc) = try_fuse_group(&c_dims[..nb], &c_strides[..nb])?;
            if ta != tb || ta != tc {
                return None;
            }
            (ta, sa, sb, sc)
        };

        let (m_raw, a_ms) = if nm == 0 {
            (1usize, 0isize)
        } else {
            try_fuse_group(&a_dims[nb..nb + nm], &a_strides[nb..nb + nm])?
        };
        let (m_chk, c_ms) = if nm == 0 {
            (1usize, 0isize)
        } else {
            try_fuse_group(&c_dims[nb..nb + nm], &c_strides[nb..nb + nm])?
        };
        if m_raw != m_chk {
            return None;
        }

        let (k_raw, a_ks) = if nk == 0 {
            (1usize, 0isize)
        } else {
            try_fuse_group(
                &a_dims[nb + nm..nb + nm + nk],
                &a_strides[nb + nm..nb + nm + nk],
            )?
        };
        let (k_chk, b_ks) = if nk == 0 {
            (1usize, 0isize)
        } else {
            try_fuse_group(&b_dims[nb..nb + nk], &b_strides[nb..nb + nk])?
        };
        if k_raw != k_chk {
            return None;
        }

        let (n_raw, b_ns) = if nn == 0 {
            (1usize, 0isize)
        } else {
            try_fuse_group(
                &b_dims[nb + nk..nb + nk + nn],
                &b_strides[nb + nk..nb + nk + nn],
            )?
        };
        let (n_chk, c_ns) = if nn == 0 {
            (1usize, 0isize)
        } else {
            try_fuse_group(
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
    fn run_strided<U: FaerGemm>(
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
        let mut a_off = 0isize;
        let mut b_off = 0isize;
        let mut c_off = 0isize;
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
        Ok(())
    }

    #[cfg(feature = "gemm-blas")]
    fn run_dense<U: Scalar>(
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
                c_mat.iter_mut().for_each(|v| *v = U::zero());
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

    let spec = if let Some(cached) = cached_spec {
        cached.clone()
    } else {
        match build_contract_gemm_spec(modes_a, modes_b, modes_c) {
            Some(spec) => spec,
            None => return Ok(None),
        }
    };

    let layout = match compute_layout(
        inputs[0].dims(),
        inputs[0].strides(),
        inputs[1].dims(),
        inputs[1].strides(),
        output.dims(),
        output.strides(),
        modes_a,
        modes_b,
        modes_c,
        &spec,
    ) {
        Some(layout) => layout,
        None => return Ok(None),
    };

    dispatch_standard_scalar_type!(T, Concrete, {
        let a = cast_strided_view!(inputs[0], T, Concrete);
        let b = cast_strided_view!(inputs[1], T, Concrete);
        let c = cast_strided_view_mut!(output, T, Concrete);
        let alpha = cast_scalar_value!(alpha, T, Concrete);
        let beta = cast_scalar_value!(beta, T, Concrete);

        #[cfg(feature = "gemm-faer")]
        {
            run_strided(alpha, a, b, beta, c, &layout)?;
            return Ok(Some(()));
        }
        #[cfg(all(feature = "gemm-blas", not(feature = "gemm-faer")))]
        {
            run_dense(
                alpha,
                a,
                b,
                beta,
                c,
                &layout,
                <Concrete as BlasGemm>::contiguous_gemm,
            )?;
            return Ok(Some(()));
        }
    });
    Ok(None)
}
