use super::*;

// ============================================================================
// AD functions: frule (forward-mode, stateless)
// ============================================================================

/// Forward-mode AD rule for SVD (JVP / pushforward).
///
/// Computes the JVP of all SVD outputs given a tangent for the input.
/// Uses batched matrix operations that broadcast over `*`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::svd_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col).unwrap();
/// let da = Tensor::<f64>::ones(&[3, 4], mem, col).unwrap();
/// let (result, dresult) = svd_frule(&mut ctx, &a, &da, None).unwrap();
/// ```
pub fn svd_frule<T, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
    options: Option<&SvdOptions>,
) -> AdResult<(SvdResult<T, T::Real>, SvdResult<T, T::Real>)>
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
    let (da_data, _) = extract_data(tangent)?;

    let mut du_data = vec![T::zero(); m * k * bc];
    let mut ds_data = vec![T::Real::zero(); k * bc];
    let mut dvt_data = vec![T::zero(); k * n * bc];
    let half = scalar_from::<T::Real>(0.5).map_err(to_ad_err)?;

    for b in 0..bc {
        let u_b = &u_data[b * m * k..(b + 1) * m * k];
        let s_b = &s_data[b * k..(b + 1) * k];
        let vt_b = &vt_data[b * k * n..(b + 1) * k * n];
        let da_b = &da_data[b * m * n..(b + 1) * m * n];
        let v_b = adjoint_transpose(vt_b, k, n);

        let uh_da = backend_mat_mul(ctx, &adjoint_transpose(u_b, m, k), k, m, da_b, n)?;
        let c = backend_mat_mul(ctx, &uh_da, k, n, &v_b, k)?;

        let mut x = c.clone();
        for i in 0..k {
            ds_data[b * k + i] = real_diagonal_from_scalar(c[i + i * k]);
            x[i + i * k] = x[i + i * k] - T::from_real(ds_data[b * k + i]);
        }

        let mut du_inner = vec![T::zero(); k * k];
        let mut dv_inner = vec![T::zero(); k * k];
        for i in 0..k {
            let diag_term = imag_axis_component(x[i + i * k])?
                * T::from_real(half * stable_inverse_sigma(s_b[i], eta));
            du_inner[i + i * k] = du_inner[i + i * k] + diag_term;
            dv_inner[i + i * k] = dv_inner[i + i * k] - diag_term;
            for j in 0..k {
                if i == j {
                    continue;
                }
                let gap_inv = T::from_real(stable_inverse_gap(s_b[i], s_b[j], eta));
                du_inner[i + j * k] = gap_inv
                    * (T::from_real(s_b[i]) * x[j + i * k].conj()
                        + x[i + j * k] * T::from_real(s_b[j]));
                dv_inner[i + j * k] = gap_inv
                    * (T::from_real(s_b[i]) * x[i + j * k]
                        + x[j + i * k].conj() * T::from_real(s_b[j]));
            }
        }
        let du_core = backend_mat_mul(ctx, u_b, m, k, &du_inner, k)?;
        let mut du_b_vec = du_core;

        if m > k {
            let da_v = backend_mat_mul(ctx, da_b, m, n, &v_b, k)?;
            let inner = backend_mat_mul(ctx, &adjoint_transpose(u_b, m, k), k, m, &da_v, k)?;
            let uu_h_da_v = backend_mat_mul(ctx, u_b, m, k, &inner, k)?;
            let proj_da_v = sub_vec(&da_v, &uu_h_da_v);
            for j in 0..k {
                let sinv = T::from_real(stable_inverse_sigma(s_b[j], eta));
                for i in 0..m {
                    du_b_vec[i + j * m] = du_b_vec[i + j * m] + proj_da_v[i + j * m] * sinv;
                }
            }
        }
        du_data[b * m * k..(b + 1) * m * k].copy_from_slice(&du_b_vec);

        let dv_core = backend_mat_mul(ctx, &v_b, n, k, &dv_inner, k)?;
        let mut dv_b_vec = dv_core;

        if n > k {
            let da_h = adjoint_transpose(da_b, m, n);
            let da_h_u = backend_mat_mul(ctx, &da_h, n, m, u_b, k)?;
            let inner = backend_mat_mul(ctx, vt_b, k, n, &da_h_u, k)?;
            let vv_h_da_h_u = backend_mat_mul(ctx, &v_b, n, k, &inner, k)?;
            let proj = sub_vec(&da_h_u, &vv_h_da_h_u);
            for j in 0..k {
                let sinv = T::from_real(stable_inverse_sigma(s_b[j], eta));
                for i in 0..n {
                    dv_b_vec[i + j * n] = dv_b_vec[i + j * n] + proj[i + j * n] * sinv;
                }
            }
        }
        let dvt_b_vec = adjoint_transpose(&dv_b_vec, n, k);
        dvt_data[b * k * n..(b + 1) * k * n].copy_from_slice(&dvt_b_vec);
    }

    let u_dims = output_dims(&[m, k], batch_dims);
    let s_dims = output_dims(&[k], batch_dims);
    let vt_dims = output_dims(&[k, n], batch_dims);

    let dresult = SvdResult {
        u: tensor_from_data(du_data, &u_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
        s: tensor_from_data_scalar(ds_data, &s_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
        vt: tensor_from_data(dvt_data, &vt_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
    };

    Ok((result, dresult))
}

/// Forward-mode AD rule for QR (JVP / pushforward).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::qr_frule;
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
/// let da = Tensor::<f64>::ones(&[4, 3], mem, col).unwrap();
/// let (result, dresult) = qr_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn qr_frule<T, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(QrResult<T>, QrResult<T>)>
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
    let half: T::Real = scalar_from(0.5).map_err(to_ad_err)?;

    let (q_data, _) = extract_data(&result.q)?;
    let (r_data, _) = extract_data(&result.r)?;
    let (da_data, _) = extract_data(tangent)?;

    let mut dq_data = vec![T::zero(); m * k * bc];
    let mut dr_data = vec![T::zero(); k * n * bc];

    for b in 0..bc {
        let q_b = &q_data[b * m * k..(b + 1) * m * k];
        let r_b = &r_data[b * k * n..(b + 1) * k * n];
        let da_b = &da_data[b * m * n..(b + 1) * m * n];

        let (dq_b_vec, dr_b_vec) = if m >= n {
            let r_sq = r_b[..n * n].to_vec();
            let darinv_h = backend_solve_tri(
                ctx,
                &adjoint_transpose(&r_sq, n, n),
                &adjoint_transpose(da_b, m, n),
                n,
                m,
                false,
            )?;
            let darinv = adjoint_transpose(&darinv_h, n, m);
            let qhdarinv = backend_mat_mul(ctx, &adjoint_transpose(q_b, m, n), n, m, &darinv, n)?;
            let sym = add_vec(&qhdarinv, &adjoint_transpose(&qhdarinv, n, n));

            let mut dr_hat = vec![T::zero(); n * n];
            for j in 0..n {
                for i in 0..=j {
                    let mut val = sym[i + j * n];
                    if i == j {
                        val = T::from_real(val.real_part() * half);
                    }
                    dr_hat[i + j * n] = val;
                }
            }

            let dq = sub_vec(&darinv, &backend_mat_mul(ctx, q_b, m, n, &dr_hat, n)?);
            let dr = backend_mat_mul(ctx, &dr_hat, n, n, &r_sq, n)?;
            (dq, dr)
        } else {
            let qhda = backend_mat_mul(ctx, &adjoint_transpose(q_b, m, k), k, m, da_b, n)?;
            let r1 = r_b[..k * k].to_vec();
            let qhda1 = qhda[..k * k].to_vec();
            let qhda1_rinv_h = backend_solve_tri(
                ctx,
                &adjoint_transpose(&r1, k, k),
                &adjoint_transpose(&qhda1, k, k),
                k,
                k,
                false,
            )?;
            let qhda1_rinv = adjoint_transpose(&qhda1_rinv_h, k, k);
            let lower = tril_im(&qhda1_rinv, k)?;
            let dq_hat = tril_im_inv(&lower, k)?;

            let dr = sub_vec(&qhda, &backend_mat_mul(ctx, &dq_hat, k, k, r_b, n)?);
            let dq = backend_mat_mul(ctx, q_b, m, k, &dq_hat, k)?;
            (dq, dr)
        };

        dq_data[b * m * k..(b + 1) * m * k].copy_from_slice(&dq_b_vec);
        dr_data[b * k * n..(b + 1) * k * n].copy_from_slice(&dr_b_vec);
    }

    let q_dims = output_dims(&[m, k], batch_dims);
    let r_dims = output_dims(&[k, n], batch_dims);
    let dresult = QrResult {
        q: tensor_from_data(dq_data, &q_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
        r: tensor_from_data(dr_data, &r_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
    };
    Ok((result, dresult))
}
