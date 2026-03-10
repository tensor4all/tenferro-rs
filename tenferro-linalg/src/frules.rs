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

/// Forward-mode AD rule for LU (JVP / pushforward).
///
/// The `pivot` argument must match the pivoting strategy used in the forward pass.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{lu_frule, LuPivot};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], &[3, 3], col)
///     .unwrap();
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let (result, dresult) = lu_frule(&mut ctx, &a, &da, LuPivot::Partial).unwrap();
/// ```
pub fn lu_frule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
    pivot: LuPivot,
) -> AdResult<(LuResult<T>, LuResult<T>)>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    let result = lu(ctx, tensor, pivot)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let k = m.min(n);
    let bc = batch_count(batch_dims);

    let (l_data, _) = extract_data(&result.l)?;
    let (u_data, _) = extract_data(&result.u)?;
    let p_vec = result.p.as_ref();
    let (da_data, _) = extract_data(tangent)?;

    let mut dl_data = vec![T::zero(); m * k * bc];
    let mut du_data = vec![T::zero(); k * n * bc];

    for b in 0..bc {
        let l_b = &l_data[b * m * k..(b + 1) * m * k];
        let u_b = &u_data[b * k * n..(b + 1) * k * n];
        let da_b = &da_data[b * m * n..(b + 1) * m * n];

        // Apply permutation: P dA (m×n)
        let mut pda = vec![T::zero(); m * n];
        if let Some(pv) = p_vec {
            let p_b = &pv[b * m..(b + 1) * m];
            for i in 0..m {
                for j in 0..n {
                    pda[i + j * m] = da_b[p_b[i] + j * m];
                }
            }
        } else {
            pda.copy_from_slice(da_b);
        }

        // F = L^{-1} P dA U^{-1} (k×k for square part)
        // First: L^{-1} PdA → solve L x = PdA
        let l_sq: Vec<T> = {
            let mut s = vec![T::zero(); k * k];
            for j in 0..k {
                for i in 0..k {
                    s[i + j * k] = l_b[i + j * m];
                }
            }
            s
        };
        let pda_sq: Vec<T> = {
            let mut s = vec![T::zero(); k * n];
            for j in 0..n {
                for i in 0..k {
                    s[i + j * k] = pda[i + j * m];
                }
            }
            s
        };
        let linv_pda = backend_solve_tri(ctx, &l_sq, &pda_sq, k, n, false)?;

        // Then: (L^{-1} PdA) U^{-1} → solve (result) U = linv_pda
        let u_sq: Vec<T> = {
            let mut s = vec![T::zero(); k * k];
            for j in 0..k {
                for i in 0..k {
                    s[i + j * k] = u_b[i + j * k];
                }
            }
            s
        };
        // Solve x U = linv_pda → U^T x^T = linv_pda^T
        let f_t = backend_solve_tri(
            ctx,
            &transpose(&u_sq, k, k),
            &transpose(&linv_pda, k, n),
            k,
            k,
            false,
        )?;
        let f = transpose(&f_t, k, k);

        // dL = L tril_strict(F) (m×k)
        let tril_f = tril_strict(&f, k);
        let dl_b_vec = backend_mat_mul(ctx, &l_sq, k, k, &tril_f, k)?;
        for j in 0..k {
            for i in 0..k {
                dl_data[b * m * k + i + j * m] = dl_b_vec[i + j * k];
            }
        }

        // dU = triu(F) U (k×n)
        let triu_f = triu(&f, k);
        let du_b_vec = backend_mat_mul(ctx, &triu_f, k, k, &u_sq, k)?;
        for j in 0..k {
            for i in 0..k {
                du_data[b * k * n + i + j * k] = du_b_vec[i + j * k];
            }
        }
    }

    let l_dims = output_dims(&[m, k], batch_dims);
    let u_dims = output_dims(&[k, n], batch_dims);
    let dresult = LuResult {
        p: None, // permutation has no derivative
        l: tensor_from_data(dl_data, &l_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
        u: tensor_from_data(du_data, &u_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
    };
    Ok((result, dresult))
}

/// Forward-mode AD rule for eigendecomposition (JVP / pushforward).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::eigen_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let (result, dresult) = eigen_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn eigen_frule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(EigenResult<T>, EigenResult<T>)>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    let result = eigen(ctx, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
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

    let (v_data, _) = extract_data(&result.vectors)?;
    let (e_data, _) = extract_data(&result.values)?;
    let (da_data, _) = extract_data(tangent)?;

    let mut de_data = vec![T::zero(); n * bc];
    let mut dv_data = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let v_b = &v_data[b * n * n..(b + 1) * n * n];
        let e_b = &e_data[b * n..(b + 1) * n];
        let da_b = &da_data[b * n * n..(b + 1) * n * n];

        // C = V^T dA V (n×n)
        let vt_da = backend_mat_mul(ctx, &transpose(v_b, n, n), n, n, da_b, n)?;
        let c = backend_mat_mul(ctx, &vt_da, n, n, v_b, n)?;

        // dE = diag(C)
        for i in 0..n {
            de_data[b * n + i] = c[i + i * n];
        }

        // dV = V F ⊙ C where F_ij = 1/(e_i - e_j) for i≠j, 0 diagonal
        let mut fc = vec![T::zero(); n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let denom = e_b[j] - e_b[i];
                    let f_ij = T::one()
                        / (denom
                            + eta
                                * if denom >= T::zero() {
                                    T::one()
                                } else {
                                    -T::one()
                                });
                    fc[i + j * n] = f_ij * c[i + j * n];
                }
            }
        }
        let dv_b_vec = backend_mat_mul(ctx, v_b, n, n, &fc, n)?;
        dv_data[b * n * n..(b + 1) * n * n].copy_from_slice(&dv_b_vec);
    }

    let val_dims = output_dims(&[n], batch_dims);
    let vec_dims = output_dims(&[n, n], batch_dims);
    let dresult = EigenResult {
        values: tensor_from_data(de_data, &val_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
        vectors: tensor_from_data(dv_data, &vec_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
    };
    Ok((result, dresult))
}

/// Forward-mode AD rule for least squares (JVP / pushforward).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::lstsq_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[1.0, 0.0, 1.0, 0.0, 1.0, 1.0], &[3, 2], col).unwrap();
/// let b = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3], col).unwrap();
/// let da = Tensor::<f64>::ones(&[3, 2], mem, col);
/// let db = Tensor::<f64>::ones(&[3], mem, col);
/// let (result, dresult) = lstsq_frule(&mut ctx, &a, &b, &da, &db).unwrap();
/// ```
pub fn lstsq_frule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
    tangent_a: &Tensor<T>,
    tangent_b: &Tensor<T>,
) -> AdResult<(LstsqResult<T>, LstsqResult<T>)>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("lstsq_frule").map_err(to_ad_err)?;

    // dx = A^+ (db - dA x), where A^+ = (A^T A)^{-1} A^T
    let result = lstsq(ctx, a, b)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(a)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(a)?;
    let (x_data, _) = extract_data(&result.x)?;
    let (da_data, _) = extract_data(tangent_a)?;
    let (db_data, _) = extract_data(tangent_b)?;

    let mut dx_data = vec![T::zero(); n * bc];
    let mut dres_data = vec![T::zero(); m * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
        let x_b = &x_data[batch * n..(batch + 1) * n];
        let da_b = &da_data[batch * m * n..(batch + 1) * m * n];
        let db_b = &db_data[batch * m..(batch + 1) * m];

        // dA x (m×1)
        let da_x = backend_mat_mul(ctx, da_b, m, n, x_b, 1)?;
        // db - dA x
        let rhs: Vec<T> = db_b.iter().zip(da_x.iter()).map(|(&a, &b)| a - b).collect();

        // A^+ rhs = (A^T A)^{-1} A^T rhs
        let at_rhs = backend_mat_mul(ctx, &transpose(a_b, m, n), n, m, &rhs, 1)?;
        let ata = backend_mat_mul(ctx, &transpose(a_b, m, n), n, m, a_b, n)?;
        let dx_b_vec = backend_solve(ctx, &ata, &at_rhs, n, 1)?;
        dx_data[batch * n..(batch + 1) * n].copy_from_slice(&dx_b_vec);

        // d(residual) = db - dA x - A dx
        let a_dx = backend_mat_mul(ctx, a_b, m, n, &dx_b_vec, 1)?;
        for i in 0..m {
            dres_data[batch * m + i] = rhs[i] - a_dx[i];
        }
    }

    let x_dims = output_dims(&[n], batch_dims);
    let res_dims = output_dims(&[m], batch_dims);
    let dresult = LstsqResult {
        x: tensor_from_data(dx_data, &x_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
        residual: tensor_from_data(dres_data, &res_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
    };
    Ok((result, dresult))
}

/// Forward-mode AD rule for Cholesky (JVP / pushforward).
///
/// # Examples
///
/// ```no_run
/// use tenferro_linalg::cholesky_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let (l, dl) = cholesky_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn cholesky_frule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    // dL = L phi(L^{-1} dA L^{-T})
    let l = cholesky(ctx, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (l_data, _) = extract_data(&l)?;
    let (da_data, _) = extract_data(tangent)?;

    let mut dl_data = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let l_b = &l_data[b * n * n..(b + 1) * n * n];
        let da_b = &da_data[b * n * n..(b + 1) * n * n];

        // L^{-1} dA: solve L x = dA
        let linv_da = backend_solve_tri(ctx, l_b, da_b, n, n, false)?;
        // (L^{-1} dA) L^{-T}: solve (result) L^T = linv_da → L x^T = linv_da^T
        let linv_da_linvt_t = backend_solve_tri(ctx, l_b, &transpose(&linv_da, n, n), n, n, false)?;
        let inner = transpose(&linv_da_linvt_t, n, n);

        // phi(inner) = tril with diagonal halved
        let phi_inner = phi(&inner, n)?;

        // dL = L phi(inner)
        let dl_b_vec = backend_mat_mul(ctx, l_b, n, n, &phi_inner, n)?;
        dl_data[b * n * n..(b + 1) * n * n].copy_from_slice(&dl_b_vec);
    }

    let dims = output_dims(&[n, n], batch_dims);
    let dl = tensor_from_data(dl_data, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    Ok((l, dl))
}

/// Forward-mode AD rule for linear solve (JVP / pushforward).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::solve_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col);
/// let b = Tensor::<f64>::ones(&[3], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let db = Tensor::<f64>::ones(&[3], mem, col);
/// let (x, dx) = solve_frule(&mut ctx, &a, &b, &da, &db).unwrap();
/// ```
pub fn solve_frule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
    tangent_a: &Tensor<T>,
    tangent_b: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    // dx = A^{-1} (db - dA x)
    let x = solve(ctx, a, b)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(a)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);
    let rhs = backend::tensor_helpers::validate_solve_rhs_shape(b, n, batch_dims, "solve_frule")
        .map_err(to_ad_err)?;
    let nrhs = rhs.nrhs;

    let (a_data, _) = extract_data(a)?;
    let (x_data, _) = extract_data(&x)?;
    let (da_data, _) = extract_data(tangent_a)?;
    let (db_data, _) = extract_data(tangent_b)?;

    let mut dx_data = vec![T::zero(); n * nrhs * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
        let x_b = &x_data[batch * n * nrhs..(batch + 1) * n * nrhs];
        let da_b = &da_data[batch * n * n..(batch + 1) * n * n];
        let db_b = &db_data[batch * n * nrhs..(batch + 1) * n * nrhs];

        // dA x (n×nrhs)
        let da_x = backend_mat_mul(ctx, da_b, n, n, x_b, nrhs)?;
        // db - dA x
        let rhs = sub_vec(db_b, &da_x);
        // A^{-1} (db - dA x)
        let dx_b_vec = backend_solve(ctx, a_b, &rhs, n, nrhs)?;
        dx_data[batch * n * nrhs..(batch + 1) * n * nrhs].copy_from_slice(&dx_b_vec);
    }

    let dims = rhs.output_dims;
    let dx = tensor_from_data(dx_data, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    Ok((x, dx))
}

/// Forward-mode AD rule for triangular solve (JVP / pushforward).
///
/// Computes:
/// - `x = solve_triangular(a, b, upper)`
/// - `dx = solve_triangular(a, db - proj(dA) * x, upper)`
///
/// where `proj(dA)` keeps only the active triangular part
/// (`triu` when `upper=true`, `tril` when `upper=false`).
pub fn solve_triangular_frule<T: LinalgScalar, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
    tangent_a: &Tensor<T>,
    tangent_b: &Tensor<T>,
    upper: bool,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    if tangent_a.dims() != a.dims() {
        return Err(chainrules_core::AutodiffError::InvalidArgument(format!(
            "solve_triangular_frule: tangent_a shape mismatch: expected {:?}, got {:?}",
            a.dims(),
            tangent_a.dims()
        )));
    }
    if tangent_b.dims() != b.dims() {
        return Err(chainrules_core::AutodiffError::InvalidArgument(format!(
            "solve_triangular_frule: tangent_b shape mismatch: expected {:?}, got {:?}",
            b.dims(),
            tangent_b.dims()
        )));
    }

    // dX = A^{-1} (dB - proj(dA) X), with projection to the triangular tangent space.
    let x = solve_triangular(ctx, a, b, upper)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;

    let (n, batch_dims) = validate_square(a)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);
    let rhs = backend::tensor_helpers::validate_solve_rhs_shape(
        b,
        n,
        batch_dims,
        "solve_triangular_frule",
    )
    .map_err(to_ad_err)?;
    let nrhs = rhs.nrhs;

    let (a_data, _) = extract_data(a)?;
    let (x_data, _) = extract_data(&x)?;
    let (da_data, _) = extract_data(tangent_a)?;
    let (db_data, _) = extract_data(tangent_b)?;

    let mut dx_data = vec![T::zero(); n * nrhs * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
        let x_b = &x_data[batch * n * nrhs..(batch + 1) * n * nrhs];
        let da_b = &da_data[batch * n * n..(batch + 1) * n * n];
        let db_b = &db_data[batch * n * nrhs..(batch + 1) * n * nrhs];

        // Project dA onto the same triangular structure as A.
        let da_proj = if upper { triu(da_b, n) } else { tril(da_b, n) };

        // dA * x, treating x as n x nrhs in column-major layout.
        let da_x =
            prims_bridge::batched_gemm_via_prims(&da_proj, n, n, x_b, nrhs).map_err(to_ad_err)?;

        // RHS tangent: dB - dA * x
        let rhs = sub_vec(db_b, &da_x);

        // dX from triangular solve with the same structure.
        let mut dx_b = vec![T::zero(); n * nrhs];
        backend::cpu::solve_triangular_slices(a_b, &rhs, n, nrhs, upper, &mut dx_b)
            .map_err(to_ad_err)?;

        dx_data[batch * n * nrhs..(batch + 1) * n * nrhs].copy_from_slice(&dx_b);
    }

    let dims = rhs.output_dims;
    let dx = tensor_from_data(dx_data, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    Ok((x, dx))
}

/// Forward-mode AD rule for matrix inverse (JVP / pushforward).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::inv_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let (a_inv, da_inv) = inv_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn inv_frule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("inv_frule").map_err(to_ad_err)?;

    // dB = -B dA B where B = A^{-1}
    let b_inv = inv(ctx, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (binv_data, _) = extract_data(&b_inv)?;
    let (da_data, _) = extract_data(tangent)?;

    let mut db_data = vec![T::zero(); n * n * bc];

    for batch in 0..bc {
        let b_b = &binv_data[batch * n * n..(batch + 1) * n * n];
        let da_b = &da_data[batch * n * n..(batch + 1) * n * n];

        let b_da = backend_mat_mul(ctx, b_b, n, n, da_b, n)?;
        let b_da_b = backend_mat_mul(ctx, &b_da, n, n, b_b, n)?;
        let neg = scale_vec(&b_da_b, -T::one());
        db_data[batch * n * n..(batch + 1) * n * n].copy_from_slice(&neg);
    }

    let dims = output_dims(&[n, n], batch_dims);
    let db = tensor_from_data(db_data, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    Ok((b_inv, db))
}

/// Forward-mode AD rule for determinant (JVP / pushforward).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::det_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let (d, dd) = det_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn det_frule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("det_frule").map_err(to_ad_err)?;

    // d(det) = det(A) * tr(A^{-1} dA)
    let d = det(ctx, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor)?;
    let (d_data, _) = extract_data(&d)?;
    let (da_data, _) = extract_data(tangent)?;

    let mut dd_data = vec![T::zero(); bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
        let da_b = &da_data[batch * n * n..(batch + 1) * n * n];

        let a_inv = backend_solve(ctx, a_b, &eye::<T>(n), n, n)?;
        let a_inv_da = backend_mat_mul(ctx, &a_inv, n, n, da_b, n)?;
        let mut trace = T::zero();
        for i in 0..n {
            trace = trace + a_inv_da[i + i * n];
        }
        dd_data[batch] = d_data[batch] * trace;
    }

    let dims = output_dims(&[], batch_dims);
    let dd = tensor_from_data(dd_data, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    Ok((d, dd))
}

/// Forward-mode AD rule for slogdet (JVP / pushforward).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::slogdet_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let (result, dresult) = slogdet_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn slogdet_frule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(SlogdetResult<T>, SlogdetResult<T>)>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("slogdet_frule").map_err(to_ad_err)?;

    // d(logabsdet) = Re(tr(A^{-1} dA)), d(sign) = 0 (for real)
    let result = slogdet(ctx, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor)?;
    let (da_data, _) = extract_data(tangent)?;

    let mut dlog_data = vec![T::zero(); bc];
    let dsign_data = vec![T::zero(); bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
        let da_b = &da_data[batch * n * n..(batch + 1) * n * n];

        let a_inv = backend_solve(ctx, a_b, &eye::<T>(n), n, n)?;
        let a_inv_da = backend_mat_mul(ctx, &a_inv, n, n, da_b, n)?;
        let mut trace = T::zero();
        for i in 0..n {
            trace = trace + a_inv_da[i + i * n];
        }
        dlog_data[batch] = trace;
    }

    let dims = output_dims(&[], batch_dims);
    let dresult = SlogdetResult {
        sign: tensor_from_data(dsign_data, &dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
        logabsdet: tensor_from_data(dlog_data, &dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
    };
    Ok((result, dresult))
}

/// Forward-mode AD rule for general eigendecomposition (JVP / pushforward).
///
/// Given eigendecomposition `A V = V diag(lambda)`, computes the tangents
/// of eigenvalues and eigenvectors from a real tangent `dA` using the
/// Mike Giles formulas.
///
/// Returns `(primal, tangent)` where both are [`EigResult`] with complex
/// eigenvalues and eigenvectors.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::eig_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let (result, dresult) = eig_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn eig_frule<
    T: LinalgScalar<Real = T, Complex = num_complex::Complex<T>> + num_traits::Float,
    C,
>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(EigResult<T>, EigResult<T>)>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    // Forward pass
    let eig_result = eig(ctx, tensor).map_err(to_ad_err)?;

    let (n, batch_dims) = validate_square(tensor).map_err(to_ad_err)?;
    let bc = batch_count(batch_dims);

    let val_data = extract_data_scalar(&eig_result.values)?;
    let vec_data = extract_data_scalar(&eig_result.vectors)?;
    let (tang_data, _) = extract_data(tangent)?;

    let zero_c = Cx::new(T::zero(), T::zero());
    let one_c = Cx::new(T::one(), T::zero());

    let mut dval_data = vec![zero_c; n * bc];
    let mut dvec_data = vec![zero_c; n * n * bc];

    for b in 0..bc {
        let lambda = &val_data[b * n..(b + 1) * n];
        let v = &vec_data[b * n * n..(b + 1) * n * n];
        let da = &tang_data[b * n * n..(b + 1) * n * n];

        // Convert real dA to complex
        let da_complex: Vec<Cx<T>> = da.iter().map(|&x| Cx::new(x, T::zero())).collect();

        // W = V^{-1} dA V = solve(V, dA_c @ V)
        let da_v = complex_mat_mul_nn(&da_complex, v, n);
        let w = complex_solve_nn(ctx, v, &da_v, n)?;

        // d_lambda = diag(W)
        for i in 0..n {
            dval_data[b * n + i] = w[i + i * n];
        }

        // F matrix: F[i,j] = 1/(lambda_j - lambda_i) for i != j, 0 on diagonal
        let mut f_mat = vec![zero_c; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let diff = lambda[j] - lambda[i];
                    f_mat[i + j * n] = one_c / diff;
                }
            }
        }

        // dV = V * (F .* W)
        let mut fw = vec![zero_c; n * n];
        for k in 0..n * n {
            fw[k] = f_mat[k] * w[k];
        }
        let dv = complex_mat_mul_nn(v, &fw, n);
        dvec_data[b * n * n..(b + 1) * n * n].copy_from_slice(&dv);
    }

    // Build tangent EigResult
    let val_dims = output_dims(&[n], batch_dims);
    let vec_dims = output_dims(&[n, n], batch_dims);

    let d_result = EigResult {
        values: tensor_from_data_scalar(dval_data, &val_dims).map_err(to_ad_err)?,
        vectors: tensor_from_data_scalar(dvec_data, &vec_dims).map_err(to_ad_err)?,
    };

    Ok((eig_result, d_result))
}

/// Forward-mode AD rule for pseudoinverse (JVP / pushforward).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::pinv_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 4], mem, col);
/// let (pinv_a, dpinv_a) = pinv_frule(&mut ctx, &a, &da, None).unwrap();
/// ```
pub fn pinv_frule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
    rcond: Option<f64>,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("pinv_frule").map_err(to_ad_err)?;

    // dA+ = -A+ dA A+ + (I - A+A) dA^T (A+)^T A+ + A+ (A+)^T dA^T (I - AA+)
    let ap = pinv(ctx, tensor, rcond)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor)?;
    let (ap_data, _) = extract_data(&ap)?;
    let (da_data, _) = extract_data(tangent)?;

    let mut dap_data = vec![T::zero(); n * m * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
        let ap_b = &ap_data[batch * n * m..(batch + 1) * n * m];
        let da_b = &da_data[batch * m * n..(batch + 1) * m * n];

        let dat = transpose(da_b, m, n); // n×m
        let apt = transpose(ap_b, n, m); // m×n

        // Term 1: -A+ dA A+ (n×m × m×n × n×m = n×m)
        let ap_da = backend_mat_mul(ctx, ap_b, n, m, da_b, n)?;
        let ap_da_ap = backend_mat_mul(ctx, &ap_da, n, n, ap_b, m)?;
        let t1 = scale_vec(&ap_da_ap, -T::one());

        // Term 2: (I - A+A) dA^T (A+)^T A+
        let apa = backend_mat_mul(ctx, ap_b, n, m, a_b, n)?; // n×n
        let i_n = eye::<T>(n);
        let i_apa = sub_vec(&i_n, &apa);
        let dat_apt = backend_mat_mul(ctx, &dat, n, m, &apt, n)?; // n×n
        let dat_apt_ap = backend_mat_mul(ctx, &dat_apt, n, n, ap_b, m)?; // n×m
        let t2 = backend_mat_mul(ctx, &i_apa, n, n, &dat_apt_ap, m)?;

        // Term 3: A+ (A+)^T dA^T (I - AA+)
        let aap = backend_mat_mul(ctx, a_b, m, n, ap_b, m)?; // m×m
        let i_m = eye::<T>(m);
        let i_aap = sub_vec(&i_m, &aap);
        let ap_apt = backend_mat_mul(ctx, ap_b, n, m, &apt, n)?; // n×n
        let ap_apt_dat = backend_mat_mul(ctx, &ap_apt, n, n, &dat, m)?; // n×m
        let t3 = backend_mat_mul(ctx, &ap_apt_dat, n, m, &i_aap, m)?;

        let dap_b_vec = add_vec(&t1, &add_vec(&t2, &t3));
        dap_data[batch * n * m..(batch + 1) * n * m].copy_from_slice(&dap_b_vec);
    }

    let dims = output_dims(&[n, m], batch_dims);
    let dap = tensor_from_data(dap_data, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    Ok((ap, dap))
}

/// Forward-mode AD rule for matrix exponential (JVP / pushforward).
///
/// Computes `exp(A)` and the Frechet derivative `d(exp(A))` in the direction `dA`.
/// Uses the auxiliary 2n x 2n matrix trick (PyTorch approach):
///
/// ```text
/// M = [[A, dA], [0, A]]
/// exp(A)    = top-left  n×n block of exp(M)
/// d(exp(A)) = top-right n×n block of exp(M)
/// ```
///
/// # Examples
///
/// ```
/// use tenferro_linalg::matrix_exp_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let (exp_a, dexp_a) = matrix_exp_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn matrix_exp_frule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("matrix_exp_frule").map_err(to_ad_err)?;

    let (n, batch_dims) = validate_square(tensor).map_err(to_ad_err)?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor)?;
    let (da_data, _) = extract_data(tangent)?;

    let nn = 2 * n;
    let mut result_data = vec![T::zero(); n * n * bc];
    let mut tangent_data = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let a = &a_data[b * n * n..(b + 1) * n * n];
        let da = &da_data[b * n * n..(b + 1) * n * n];

        // Build 2n×2n auxiliary matrix M = [[A, dA], [0, A]]
        let mut m = vec![T::zero(); nn * nn];
        for j in 0..n {
            for i in 0..n {
                // Top-left: A
                m[i + j * nn] = a[i + j * n];
                // Top-right: dA
                m[i + (j + n) * nn] = da[i + j * n];
                // Bottom-right: A
                m[(i + n) + (j + n) * nn] = a[i + j * n];
                // Bottom-left: already zero
            }
        }

        // Compute exp(M) — call matrix_exp_single with the 2n×2n matrix
        let exp_m = matrix_exp_single(ctx, &m, nn).map_err(to_ad_err)?;

        // Extract top-left block → exp(A)
        for j in 0..n {
            for i in 0..n {
                result_data[b * n * n + i + j * n] = exp_m[i + j * nn];
            }
        }

        // Extract top-right block → d(exp(A))
        for j in 0..n {
            for i in 0..n {
                tangent_data[b * n * n + i + j * n] = exp_m[i + (j + n) * nn];
            }
        }
    }

    let dims = output_dims(&[n, n], batch_dims);
    let result = tensor_from_data(result_data, &dims).map_err(to_ad_err)?;
    let tang = tensor_from_data(tangent_data, &dims).map_err(to_ad_err)?;
    Ok((result, tang))
}

/// Forward-mode AD rule for norm (JVP / pushforward).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{norm_frule, NormKind};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
/// let da = Tensor::<f64>::ones(&[3, 4], mem, col);
/// let (n, dn) = norm_frule(&mut ctx, &a, &da, NormKind::Fro).unwrap();
/// ```
pub fn norm_frule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
    kind: NormKind,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("norm_frule").map_err(to_ad_err)?;

    let nrm = norm(ctx, tensor, kind)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;

    if tensor.ndim() == 1 {
        let (a_data, _) = extract_data(tensor)?;
        let (nrm_data, _) = extract_data(&nrm)?;
        let (da_data, _) = extract_data(tangent)?;
        let len = tensor.dims()[0];
        let mut dnrm = T::zero();

        match kind {
            NormKind::Fro => {
                let nv = nrm_data[0];
                if nv > T::zero() {
                    for i in 0..len {
                        dnrm = dnrm + a_data[i] * da_data[i];
                    }
                    dnrm = dnrm / nv;
                }
            }
            NormKind::L1 => {
                for i in 0..len {
                    let v = a_data[i];
                    let sign = if v > T::zero() {
                        T::one()
                    } else if v < T::zero() {
                        -T::one()
                    } else {
                        T::zero()
                    };
                    dnrm = dnrm + sign * da_data[i];
                }
            }
            NormKind::Inf => {
                let max_abs = a_data.iter().fold(T::zero(), |acc, &v| acc.max(v.abs()));
                let active: Vec<usize> = a_data
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| if v.abs() == max_abs { Some(i) } else { None })
                    .collect();
                if !active.is_empty() {
                    for i in active.iter().copied() {
                        let v = a_data[i];
                        let sign = if v > T::zero() {
                            T::one()
                        } else if v < T::zero() {
                            -T::one()
                        } else {
                            T::zero()
                        };
                        dnrm = dnrm + sign * da_data[i];
                    }
                    let active_count = scalar_from::<T>(active.len() as f64).map_err(to_ad_err)?;
                    dnrm = dnrm / active_count;
                }
            }
            NormKind::Lp(p) => {
                if p < 1.0 {
                    return Err(invalid_vector_lp_exponent_ad_error(p));
                }
                if p == 1.0 {
                    for i in 0..len {
                        let v = a_data[i];
                        let sign = if v > T::zero() {
                            T::one()
                        } else if v < T::zero() {
                            -T::one()
                        } else {
                            T::zero()
                        };
                        dnrm = dnrm + sign * da_data[i];
                    }
                } else {
                    let nv = nrm_data[0];
                    if nv > T::zero() {
                        let p_minus_one = scalar_from::<T>(p - 1.0).map_err(to_ad_err)?;
                        for i in 0..len {
                            let v = a_data[i];
                            let sign = if v > T::zero() {
                                T::one()
                            } else if v < T::zero() {
                                -T::one()
                            } else {
                                T::zero()
                            };
                            dnrm = dnrm + sign * v.abs().powf(p_minus_one) * da_data[i];
                        }
                        dnrm = dnrm / nv.powf(p_minus_one);
                    }
                }
            }
            NormKind::Nuclear | NormKind::Spectral => {
                return Err(matrix_only_norm_kind_ad_error(kind));
            }
        }

        let dnrm = tensor_from_data(vec![dnrm], &[]).map_err(to_ad_err)?;
        return Ok((nrm, dnrm));
    }

    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor)?;
    let (nrm_data, _) = extract_data(&nrm)?;
    let (da_data, _) = extract_data(tangent)?;

    let mut dnrm_data = vec![T::zero(); bc];

    match kind {
        NormKind::Fro => {
            // d||A||_F = tr(A^T dA) / ||A||_F
            for batch in 0..bc {
                let nv = nrm_data[batch];
                if nv > T::zero() {
                    let mut dot = T::zero();
                    for i in 0..m * n {
                        dot = dot + a_data[batch * m * n + i] * da_data[batch * m * n + i];
                    }
                    dnrm_data[batch] = dot / nv;
                }
            }
        }
        NormKind::Nuclear => {
            // d||A||_* = tr(U^T dA V)
            for batch in 0..bc {
                let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
                let da_b = &da_data[batch * m * n..(batch + 1) * m * n];
                let (u, _s, v) = backend_thin_svd(ctx, a_b, m, n)?;
                let k = m.min(n);
                let ut_da = backend_mat_mul(ctx, &transpose(&u, m, k), k, m, da_b, n)?;
                let ut_da_v = backend_mat_mul(ctx, &ut_da, k, n, &v, k)?;
                let mut trace = T::zero();
                for i in 0..k {
                    trace = trace + ut_da_v[i + i * k];
                }
                dnrm_data[batch] = trace;
            }
        }
        NormKind::Spectral => {
            // d||A||_2 = u1^T dA v1
            for batch in 0..bc {
                let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
                let da_b = &da_data[batch * m * n..(batch + 1) * m * n];
                let (u, _s, v) = backend_thin_svd(ctx, a_b, m, n)?;
                let mut val = T::zero();
                for i in 0..m {
                    for j in 0..n {
                        val = val + u[i] * da_b[i + j * m] * v[j];
                    }
                }
                dnrm_data[batch] = val;
            }
        }
        NormKind::L1 => {
            // d||A||_1 = sum_i sign(A_ij) dA_ij on active max columns.
            // At ties, average uniformly over active columns.
            for (batch, dn_slot) in dnrm_data.iter_mut().enumerate().take(bc) {
                if m == 0 || n == 0 {
                    continue;
                }
                let base = batch * m * n;
                let mut col_sums = vec![T::zero(); n];
                for j in 0..n {
                    let mut sum = T::zero();
                    for i in 0..m {
                        sum = sum + a_data[base + i + j * m].abs();
                    }
                    col_sums[j] = sum;
                }
                let mut max_sum = T::neg_infinity();
                for &sum in &col_sums {
                    if sum > max_sum {
                        max_sum = sum;
                    }
                }
                let active_cols: Vec<usize> = col_sums
                    .iter()
                    .enumerate()
                    .filter_map(|(j, &sum)| if sum == max_sum { Some(j) } else { None })
                    .collect();
                if active_cols.is_empty() {
                    continue;
                }
                let mut accum = T::zero();
                for j in active_cols.iter().copied() {
                    for i in 0..m {
                        let v = a_data[base + i + j * m];
                        let sign = if v > T::zero() {
                            T::one()
                        } else if v < T::zero() {
                            -T::one()
                        } else {
                            T::zero()
                        };
                        accum = accum + sign * da_data[base + i + j * m];
                    }
                }
                let active_count = scalar_from::<T>(active_cols.len() as f64).map_err(to_ad_err)?;
                *dn_slot = accum / active_count;
            }
        }
        NormKind::Inf => {
            // d||A||_inf = sum_j sign(A_ij) dA_ij on active max rows.
            // At ties, average uniformly over active rows.
            for (batch, dn_slot) in dnrm_data.iter_mut().enumerate().take(bc) {
                if m == 0 || n == 0 {
                    continue;
                }
                let base = batch * m * n;
                let mut row_sums = vec![T::zero(); m];
                for i in 0..m {
                    let mut sum = T::zero();
                    for j in 0..n {
                        sum = sum + a_data[base + i + j * m].abs();
                    }
                    row_sums[i] = sum;
                }
                let mut max_sum = T::neg_infinity();
                for &sum in &row_sums {
                    if sum > max_sum {
                        max_sum = sum;
                    }
                }
                let active_rows: Vec<usize> = row_sums
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &sum)| if sum == max_sum { Some(i) } else { None })
                    .collect();
                if active_rows.is_empty() {
                    continue;
                }
                let mut accum = T::zero();
                for i in active_rows.iter().copied() {
                    for j in 0..n {
                        let v = a_data[base + i + j * m];
                        let sign = if v > T::zero() {
                            T::one()
                        } else if v < T::zero() {
                            -T::one()
                        } else {
                            T::zero()
                        };
                        accum = accum + sign * da_data[base + i + j * m];
                    }
                }
                let active_count = scalar_from::<T>(active_rows.len() as f64).map_err(to_ad_err)?;
                *dn_slot = accum / active_count;
            }
        }
        _ => {
            return Err(chainrules_core::AutodiffError::ModeNotSupported {
                mode: "norm_frule".into(),
                reason: format!("norm kind {kind:?} AD not yet implemented"),
            });
        }
    }

    let dims = output_dims(&[], batch_dims);
    let dnrm = tensor_from_data(dnrm_data, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    Ok((nrm, dnrm))
}
