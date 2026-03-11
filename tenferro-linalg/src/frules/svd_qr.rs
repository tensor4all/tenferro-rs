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
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 4], mem, col);
/// let (result, dresult) = svd_frule(&mut ctx, &a, &da, None).unwrap();
/// ```
pub fn svd_frule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
    options: Option<&SvdOptions>,
) -> AdResult<(SvdResult<T>, SvdResult<T>)>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
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
    let eta: T = {
        let raw: T = scalar_from(1e-40).map_err(to_ad_err)?;
        let eps = T::epsilon();
        if raw < eps {
            eps
        } else {
            raw
        }
    };

    let (u_data, _) = extract_data(&result.u)?;
    let (s_data, _) = extract_data(&result.s)?;
    let (vt_data, _) = extract_data(&result.vt)?;
    let (da_data, _) = extract_data(tangent)?;

    let mut du_data = vec![T::zero(); m * k * bc];
    let mut ds_data = vec![T::zero(); k * bc];
    let mut dvt_data = vec![T::zero(); k * n * bc];

    for b in 0..bc {
        let u_b = &u_data[b * m * k..(b + 1) * m * k];
        let s_b = &s_data[b * k..(b + 1) * k];
        let vt_b = &vt_data[b * k * n..(b + 1) * k * n];
        let da_b = &da_data[b * m * n..(b + 1) * m * n];

        // C = U^T dA V (k×k)
        let ut_da = backend_mat_mul(ctx, &transpose(u_b, m, k), k, m, da_b, n)?;
        let v_b = transpose(vt_b, k, n);
        let c = backend_mat_mul(ctx, &ut_da, k, n, &v_b, k)?;

        // dS = diag(C)
        for i in 0..k {
            ds_data[b * k + i] = c[i + i * k];
        }

        // F-matrix
        let mut f_mat = vec![T::zero(); k * k];
        for i in 0..k {
            for j in 0..k {
                if i != j {
                    let denom = s_b[j] * s_b[j] - s_b[i] * s_b[i];
                    f_mat[i + j * k] = T::one()
                        / (denom
                            + eta
                                * if denom >= T::zero() {
                                    T::one()
                                } else {
                                    -T::one()
                                });
                }
            }
        }

        // dU = U (F ⊙ (S C^T + C S)) + (I_m - U U^T) dA V S^{-1}
        let mut sc_t_plus_cs = vec![T::zero(); k * k];
        for i in 0..k {
            for j in 0..k {
                sc_t_plus_cs[i + j * k] = s_b[i] * c[j + i * k] + c[i + j * k] * s_b[j];
            }
        }
        let f_inner = hadamard(&f_mat, &sc_t_plus_cs);
        let du_core = backend_mat_mul(ctx, u_b, m, k, &f_inner, k)?;

        // Projector term for dU
        if m > k {
            let inner = backend_mat_mul(ctx, &transpose(u_b, m, k), k, m, da_b, n)?;
            let uut_da = backend_mat_mul(ctx, u_b, m, k, &inner, n)?;
            let proj_da: Vec<T> = da_b
                .iter()
                .zip(uut_da.iter())
                .map(|(&a, &b)| a - b)
                .collect();
            let proj_da_v = backend_mat_mul(ctx, &proj_da, m, n, &v_b, k)?;
            for j in 0..k {
                let sinv = if s_b[j].abs() > eta {
                    T::one() / s_b[j]
                } else {
                    T::zero()
                };
                for i in 0..m {
                    du_data[b * m * k + i + j * m] =
                        du_core[i + j * m] + proj_da_v[i + j * m] * sinv;
                }
            }
        } else {
            du_data[b * m * k..(b + 1) * m * k].copy_from_slice(&du_core);
        }

        // dVt = (F ⊙ (S^T C + C^T S)) V^T + S^{-1} U^T dA (I_n - V V^T)
        let mut st_c_plus_ct_s = vec![T::zero(); k * k];
        for i in 0..k {
            for j in 0..k {
                st_c_plus_ct_s[i + j * k] = -(s_b[i] * c[i + j * k] + c[j + i * k] * s_b[j]);
            }
        }
        let f_inner2 = hadamard(&f_mat, &st_c_plus_ct_s);
        let dvt_core = backend_mat_mul(ctx, &f_inner2, k, k, vt_b, n)?;

        if n > k {
            let vvt = backend_mat_mul(ctx, &v_b, n, k, vt_b, n)?;
            let i_n = eye::<T>(n);
            let i_vvt = sub_vec(&i_n, &vvt);
            let ut_da = backend_mat_mul(ctx, &transpose(u_b, m, k), k, m, da_b, n)?;
            let sinv_ut_da = {
                let mut r = vec![T::zero(); k * n];
                for i in 0..k {
                    let sinv = if s_b[i].abs() > eta {
                        T::one() / s_b[i]
                    } else {
                        T::zero()
                    };
                    for j in 0..n {
                        r[i + j * k] = sinv * ut_da[i + j * k];
                    }
                }
                r
            };
            let proj = backend_mat_mul(ctx, &sinv_ut_da, k, n, &i_vvt, n)?;
            dvt_data[b * k * n..(b + 1) * k * n].copy_from_slice(&add_vec(&dvt_core, &proj));
        } else {
            dvt_data[b * k * n..(b + 1) * k * n].copy_from_slice(&dvt_core);
        }
    }

    let u_dims = output_dims(&[m, k], batch_dims);
    let s_dims = output_dims(&[k], batch_dims);
    let vt_dims = output_dims(&[k, n], batch_dims);

    let dresult = SvdResult {
        u: tensor_from_data(du_data, &u_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
        s: tensor_from_data(ds_data, &s_dims)
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
/// let da = Tensor::<f64>::ones(&[4, 3], mem, col);
/// let (result, dresult) = qr_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn qr_frule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(QrResult<T>, QrResult<T>)>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    let result = qr(ctx, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let k = m.min(n);
    let bc = batch_count(batch_dims);
    let half: T = scalar_from(0.5).map_err(to_ad_err)?;

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
            let darinv_t = backend_solve_tri(
                ctx,
                &transpose(&r_sq, n, n),
                &transpose(da_b, m, n),
                n,
                m,
                false,
            )?;
            let darinv = transpose(&darinv_t, n, m);
            let qhdarinv = backend_mat_mul(ctx, &transpose(q_b, m, n), n, m, &darinv, n)?;
            let sym = add_vec(&qhdarinv, &transpose(&qhdarinv, n, n));

            let mut dr_hat = vec![T::zero(); n * n];
            for j in 0..n {
                for i in 0..=j {
                    let mut val = sym[i + j * n];
                    if i == j {
                        val = val * half;
                    }
                    dr_hat[i + j * n] = val;
                }
            }

            let dq = sub_vec(&darinv, &backend_mat_mul(ctx, q_b, m, n, &dr_hat, n)?);
            let dr = backend_mat_mul(ctx, &dr_hat, n, n, &r_sq, n)?;
            (dq, dr)
        } else {
            let qhda = backend_mat_mul(ctx, &transpose(q_b, m, k), k, m, da_b, n)?;
            // k = min(m,n) so k*n == k*k when n == k (the only case reaching here)
            let r1 = r_b[..k * n].to_vec();

            let mut qhda1 = vec![T::zero(); k * k];
            for j in 0..k {
                for i in 0..k {
                    qhda1[i + j * k] = qhda[i + j * k];
                }
            }
            let qhda1_rinv_t = backend_solve_tri(
                ctx,
                &transpose(&r1, k, k),
                &transpose(&qhda1, k, k),
                k,
                k,
                false,
            )?;
            let qhda1_rinv = transpose(&qhda1_rinv_t, k, k);
            let lower = tril_strict(&qhda1_rinv, k);
            let dq_hat = sub_vec(&lower, &transpose(&lower, k, k));

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
