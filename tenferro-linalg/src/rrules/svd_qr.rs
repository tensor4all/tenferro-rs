use super::*;

/// Reverse-mode AD rule for SVD (VJP / pullback).
///
/// Computes the gradient of the input given cotangents for the SVD outputs.
/// Uses the F-matrix approach (Mathieu 2019).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{svd, svd_rrule, SvdCotangent};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col).unwrap();
///
/// let cotangent = SvdCotangent {
///     u: None,
///     s: Some(Tensor::ones(&[3], mem, col).unwrap()),
///     vt: None,
/// };
/// let grad_a = svd_rrule(&mut ctx, &a, &cotangent, None).unwrap();
/// ```
pub fn svd_rrule<T, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &SvdCotangent<T, T::Real>,
    options: Option<&SvdOptions>,
) -> AdResult<Tensor<T>>
where
    T: KernelLinalgScalar,
    T::Real: num_traits::Float,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T::Real>>,
    C::Backend: 'static,
    T::Real: tenferro_tensor::KeepCountScalar,
{
    let result = svd(ctx, tensor, options)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let k = m.min(n);
    let bc = batch_count(batch_dims);
    // Regularization for the F-matrix: prevents division by zero when two
    // singular values are (nearly) equal.  We use max(1e-40, T::epsilon())
    // so that on f32 (where 1e-40 underflows to 0) we still get a safe floor.
    let eta: T::Real = {
        let raw: T::Real = scalar_from(1e-40).map_err(to_ad_err)?;
        let eps = T::real_epsilon();
        if raw < eps {
            eps
        } else {
            raw
        }
    };

    let (u_data, _) = extract_data(&result.u)?;
    let s_data = extract_data_scalar(&result.s)?;
    let (vt_data, _) = extract_data(&result.vt)?;
    let du_data = cotangent
        .u
        .as_ref()
        .map(extract_data)
        .transpose()?
        .map(|(data, _)| data);
    let ds_data = cotangent.s.as_ref().map(extract_data_scalar).transpose()?;
    let dvt_data = cotangent
        .vt
        .as_ref()
        .map(extract_data)
        .transpose()?
        .map(|(data, _)| data);

    let mut grad_a = vec![T::zero(); m * n * bc];

    for b in 0..bc {
        let u_b = &u_data[b * m * k..(b + 1) * m * k];
        let s_b = &s_data[b * k..(b + 1) * k];
        let vt_b = &vt_data[b * k * n..(b + 1) * k * n];
        let v_b = adjoint_transpose(vt_b, k, n);

        let mut gamma = vec![T::zero(); k * k];

        if let Some(ds_data) = ds_data.as_ref() {
            let ds_b = &ds_data[b * k..(b + 1) * k];
            for i in 0..k {
                gamma[i + i * k] = gamma[i + i * k] + T::from_real(ds_b[i]);
            }
        }

        if let Some(du_data) = du_data.as_ref() {
            let du_b = &du_data[b * m * k..(b + 1) * m * k];
            let uh_du = backend_mat_mul(ctx, &adjoint_transpose(u_b, m, k), k, m, du_b, k)?;
            for i in 0..k {
                let s_inv = stable_inverse_sigma(s_b[i], eta);
                gamma[i + i * k] =
                    gamma[i + i * k] + imag_axis_component(uh_du[i + i * k])? * T::from_real(s_inv);
                for j in 0..k {
                    if i == j {
                        continue;
                    }
                    let gap_inv = T::from_real(stable_inverse_gap(s_b[i], s_b[j], eta));
                    let skew = uh_du[i + j * k] - uh_du[j + i * k].conj();
                    gamma[i + j * k] = gamma[i + j * k] + gap_inv * skew * T::from_real(s_b[j]);
                }
            }
        }

        if let Some(dvt_data) = dvt_data.as_ref() {
            let dvt_b = &dvt_data[b * k * n..(b + 1) * k * n];
            let dv_b = adjoint_transpose(dvt_b, k, n);
            let vh_dv = backend_mat_mul(ctx, vt_b, k, n, &dv_b, k)?;
            for i in 0..k {
                for j in 0..k {
                    if i == j {
                        continue;
                    }
                    let gap_inv = T::from_real(stable_inverse_gap(s_b[i], s_b[j], eta));
                    let skew = vh_dv[i + j * k] - vh_dv[j + i * k].conj();
                    gamma[i + j * k] = gamma[i + j * k] + T::from_real(s_b[i]) * gap_inv * skew;
                }
            }
        }

        let u_gamma = backend_mat_mul(ctx, u_b, m, k, &gamma, k)?;
        let da_core = backend_mat_mul(ctx, &u_gamma, m, k, vt_b, n)?;

        for i in 0..m * n {
            grad_a[b * m * n + i] = da_core[i];
        }

        if m > k {
            if let Some(du_data) = du_data.as_ref() {
                let du_b = &du_data[b * m * k..(b + 1) * m * k];
                let mut du_sinv = vec![T::zero(); m * k];
                for j in 0..k {
                    let sinv = T::from_real(stable_inverse_sigma(s_b[j], eta));
                    for i in 0..m {
                        du_sinv[i + j * m] = du_b[i + j * m] * sinv;
                    }
                }
                let inner = backend_mat_mul(ctx, &adjoint_transpose(u_b, m, k), k, m, &du_sinv, k)?;
                let uut_du = backend_mat_mul(ctx, u_b, m, k, &inner, k)?;
                let proj = sub_vec(&du_sinv, &uut_du);
                let correction = backend_mat_mul(ctx, &proj, m, k, vt_b, n)?;
                for i in 0..m * n {
                    grad_a[b * m * n + i] = grad_a[b * m * n + i] + correction[i];
                }
            }
        }

        if n > k {
            if let Some(dvt_data) = dvt_data.as_ref() {
                let dvt_b = &dvt_data[b * k * n..(b + 1) * k * n];
                let dv_b = adjoint_transpose(dvt_b, k, n);
                let inner = backend_mat_mul(ctx, vt_b, k, n, &dv_b, k)?;
                let vvt_dv = backend_mat_mul(ctx, &v_b, n, k, &inner, k)?;
                let proj_dv = sub_vec(&dv_b, &vvt_dv);
                let mut u_sinv = vec![T::zero(); m * k];
                for j in 0..k {
                    let sinv = T::from_real(stable_inverse_sigma(s_b[j], eta));
                    for i in 0..m {
                        u_sinv[i + j * m] = u_b[i + j * m] * sinv;
                    }
                }
                let correction =
                    backend_mat_mul(ctx, &u_sinv, m, k, &adjoint_transpose(&proj_dv, n, k), n)?;
                for i in 0..m * n {
                    grad_a[b * m * n + i] = grad_a[b * m * n + i] + correction[i];
                }
            }
        }
    }

    let dims = output_dims(&[m, n], batch_dims);
    tensor_from_data(grad_a, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))
}

/// Reverse-mode AD rule for QR (VJP / pullback).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{qr_rrule, QrCotangent};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(
///     &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
///     &[4, 3],
///     col,
/// ).unwrap();
/// let cotangent = QrCotangent {
///     q: Some(Tensor::ones(&[4, 3], mem, col).unwrap()),
///     r: None,
/// };
/// let grad_a = qr_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn qr_rrule<T, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &QrCotangent<T>,
) -> AdResult<Tensor<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    let result = qr(ctx, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let k = m.min(n);
    let bc = batch_count(batch_dims);

    let (q_data, _) = extract_data(&result.q)?;
    let (r_data, _) = extract_data(&result.r)?;

    let mut grad_a = vec![T::zero(); m * n * bc];

    for b in 0..bc {
        let q_b = &q_data[b * m * k..(b + 1) * m * k];
        let r_b = &r_data[b * k * n..(b + 1) * k * n];

        // Initialize dQ and dR from cotangents (zero if not provided)
        let dq_b: Vec<T> = if let Some(ref dq) = cotangent.q {
            let (dq_data, _) = extract_data(dq)?;
            dq_data[b * m * k..(b + 1) * m * k].to_vec()
        } else {
            vec![T::zero(); m * k]
        };
        let dr_b: Vec<T> = if let Some(ref dr) = cotangent.r {
            let (dr_data, _) = extract_data(dr)?;
            dr_data[b * k * n..(b + 1) * k * n].to_vec()
        } else {
            vec![T::zero(); k * n]
        };

        if m >= n {
            let r_drh = backend_mat_mul(ctx, r_b, k, n, &adjoint_transpose(&dr_b, k, n), k)?;
            let dqh_q = backend_mat_mul(ctx, &adjoint_transpose(&dq_b, m, k), k, m, q_b, k)?;
            let w = sub_vec(&r_drh, &dqh_q);

            let h = copyltu(&w, k);
            let qh = backend_mat_mul(ctx, q_b, m, k, &h, k)?;
            let rhs = add_vec(&dq_b, &qh);

            let r_square = r_b[..k * n].to_vec();
            let rhs_h = adjoint_transpose(&rhs, m, k);
            let da_h = backend_solve_tri(ctx, &r_square, &rhs_h, k, m, true)?;
            let da_first_k = adjoint_transpose(&da_h, k, m);

            for j in 0..k.min(n) {
                for i in 0..m {
                    grad_a[b * m * n + i + j * m] = da_first_k[i + j * m];
                }
            }
        } else {
            let qhgq = backend_mat_mul(ctx, &adjoint_transpose(q_b, m, k), k, m, &dq_b, k)?;
            let gr_rh = backend_mat_mul(ctx, &dr_b, k, n, &adjoint_transpose(r_b, k, n), k)?;
            let wide_inner = sub_vec(&qhgq, &gr_rh);
            let lower_skew = tril_im_inv_adj_skew(&wide_inner, k)?;

            let q_lower = backend_mat_mul(ctx, q_b, m, k, &lower_skew, k)?;
            let mut r1 = vec![T::zero(); k * k];
            for j in 0..k {
                for i in 0..k {
                    r1[i + j * k] = r_b[i + j * k];
                }
            }
            let q_lower_h = adjoint_transpose(&q_lower, m, k);
            let leading_h = backend_solve_tri(ctx, &r1, &q_lower_h, k, m, true)?;
            let leading = adjoint_transpose(&leading_h, k, m);

            for j in 0..k {
                for i in 0..m {
                    grad_a[b * m * n + i + j * m] = leading[i + j * m];
                }
            }

            let qgr = backend_mat_mul(ctx, q_b, m, k, &dr_b, n)?;
            for j in 0..n {
                for i in 0..m {
                    grad_a[b * m * n + i + j * m] = grad_a[b * m * n + i + j * m] + qgr[i + j * m];
                }
            }
        }
    }

    let dims = output_dims(&[m, n], batch_dims);
    tensor_from_data(grad_a, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))
}
