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
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
///
/// let cotangent = SvdCotangent {
///     u: None,
///     s: Some(Tensor::ones(&[3], mem, col)),
///     vt: None,
/// };
/// let grad_a = svd_rrule(&mut ctx, &a, &cotangent, None).unwrap();
/// ```
pub fn svd_rrule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &SvdCotangent<T>,
    options: Option<&SvdOptions>,
) -> AdResult<Tensor<T>>
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

    let mut grad_a = vec![T::zero(); m * n * bc];

    for b in 0..bc {
        let u_b = &u_data[b * m * k..(b + 1) * m * k];
        let s_b = &s_data[b * k..(b + 1) * k];
        let vt_b = &vt_data[b * k * n..(b + 1) * k * n];
        // V = Vt^T: n×k
        let v_b = transpose(vt_b, k, n);

        // Build F-matrix (k×k): F_ij = 1/(s_j² - s_i²) for i≠j, 0 diagonal
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

        // Start building inner matrix Gamma (k×k)
        let mut gamma = vec![T::zero(); k * k];

        // From dS cotangent: add diag(dS)
        if let Some(ref ds) = cotangent.s {
            let (ds_data, _) = extract_data(ds)?;
            let ds_b = &ds_data[b * k..(b + 1) * k];
            for i in 0..k {
                gamma[i + i * k] = gamma[i + i * k] + ds_b[i];
            }
        }

        // From dU cotangent: F ⊙ (U^T dU + (U^T dU)^T) * S
        if let Some(ref du) = cotangent.u {
            let (du_data, _) = extract_data(du)?;
            let du_b = &du_data[b * m * k..(b + 1) * m * k];
            // U^T dU (k×k)
            let ut_du = backend_mat_mul(ctx, &transpose(u_b, m, k), k, m, du_b, k)?;
            for i in 0..k {
                for j in 0..k {
                    let skew = ut_du[i + j * k] - ut_du[j + i * k];
                    gamma[i + j * k] = gamma[i + j * k] + f_mat[i + j * k] * skew * s_b[j];
                }
            }
        }

        // From dVt cotangent: S * F ⊙ (V^T dV + (V^T dV)^T)
        if let Some(ref dvt) = cotangent.vt {
            let (dvt_data, _) = extract_data(dvt)?;
            let dvt_b = &dvt_data[b * k * n..(b + 1) * k * n];
            // dV = dVt^T (n×k)
            let dv_b = transpose(dvt_b, k, n);
            // V^T dV (k×k)
            let vt_dv = backend_mat_mul(ctx, &transpose(&v_b, n, k), k, n, &dv_b, k)?;
            for i in 0..k {
                for j in 0..k {
                    let skew = vt_dv[i + j * k] - vt_dv[j + i * k];
                    gamma[i + j * k] = gamma[i + j * k] + s_b[i] * f_mat[i + j * k] * skew;
                }
            }
        }

        // Core: dA_core = U * Gamma * V^T (m×k × k×k × k×n = m×n)
        let u_gamma = backend_mat_mul(ctx, u_b, m, k, &gamma, k)?;
        let da_core = backend_mat_mul(ctx, &u_gamma, m, k, &transpose(&v_b, n, k), n)?;

        // Copy core to output
        for i in 0..m * n {
            grad_a[b * m * n + i] = da_core[i];
        }

        // Non-square correction: (I - UU^T) dU S_inv^T V^T when m > k
        if m > k {
            if let Some(ref du) = cotangent.u {
                let (du_data, _) = extract_data(du)?;
                let du_b = &du_data[b * m * k..(b + 1) * m * k];
                // dU * diag(1/S) (m×k)
                let mut du_sinv = vec![T::zero(); m * k];
                for j in 0..k {
                    let sinv = if s_b[j].abs() > eta {
                        T::one() / s_b[j]
                    } else {
                        T::zero()
                    };
                    for i in 0..m {
                        du_sinv[i + j * m] = du_b[i + j * m] * sinv;
                    }
                }
                // (I - UU^T) * du_sinv * V^T
                let inner = backend_mat_mul(ctx, &transpose(u_b, m, k), k, m, &du_sinv, k)?;
                let uut_du = backend_mat_mul(ctx, u_b, m, k, &inner, k)?;
                let proj = sub_vec(&du_sinv, &uut_du);
                let correction = backend_mat_mul(ctx, &proj, m, k, &transpose(&v_b, n, k), n)?;
                for i in 0..m * n {
                    grad_a[b * m * n + i] = grad_a[b * m * n + i] + correction[i];
                }
            }
        }

        // Non-square correction for n > k: U S_inv^T (I - VV^T) dV^T
        if n > k {
            if let Some(ref dvt) = cotangent.vt {
                let (dvt_data, _) = extract_data(dvt)?;
                let dvt_b = &dvt_data[b * k * n..(b + 1) * k * n];
                let dv_b = transpose(dvt_b, k, n);
                // diag(1/S) * dV^T (k×n) = diag(1/S) * Vt_cotangent
                // But we need dV (n×k), so: (I - VV^T) dV → project
                let inner = backend_mat_mul(ctx, &transpose(&v_b, n, k), k, n, &dv_b, k)?;
                let vvt_dv = backend_mat_mul(ctx, &v_b, n, k, &inner, k)?;
                let proj_dv = sub_vec(&dv_b, &vvt_dv);
                // U * diag(1/S) * proj_dv^T
                let mut u_sinv = vec![T::zero(); m * k];
                for j in 0..k {
                    let sinv = if s_b[j].abs() > eta {
                        T::one() / s_b[j]
                    } else {
                        T::zero()
                    };
                    for i in 0..m {
                        u_sinv[i + j * m] = u_b[i + j * m] * sinv;
                    }
                }
                let correction =
                    backend_mat_mul(ctx, &u_sinv, m, k, &transpose(&proj_dv, n, k), n)?;
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
///     q: Some(Tensor::ones(&[4, 3], mem, col)),
///     r: None,
/// };
/// let grad_a = qr_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn qr_rrule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &QrCotangent<T>,
) -> AdResult<Tensor<T>>
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
            // For thin QR (m >= n): A = QR where Q is m×k, R is k×n.
            // Match PyTorch's reduced-QR backward for the real case.
            let r_drt = backend_mat_mul(ctx, r_b, k, n, &transpose(&dr_b, k, n), k)?;
            let dqt_q = backend_mat_mul(ctx, &transpose(&dq_b, m, k), k, m, q_b, k)?;
            let w = sub_vec(&r_drt, &dqt_q);

            let h = copyltu(&w, k);
            let qh = backend_mat_mul(ctx, q_b, m, k, &h, k)?;
            let rhs = add_vec(&dq_b, &qh);

            let r_square = r_b[..k * n].to_vec();
            let rhs_t = transpose(&rhs, m, k);
            let da_t = backend_solve_tri(ctx, &r_square, &rhs_t, k, m, true)?;
            let da_first_k = transpose(&da_t, k, m);

            for j in 0..k.min(n) {
                for i in 0..m {
                    grad_a[b * m * n + i + j * m] = da_first_k[i + j * m];
                }
            }
        } else {
            // Wide reduced QR follows the PyTorch backward:
            // gA = pi*(Q trilImInvAdjSkew(Q^T gQ - gR R^T) R1^{-T}) + Q gR
            let qtgq = backend_mat_mul(ctx, &transpose(q_b, m, k), k, m, &dq_b, k)?;
            let gr_rt = backend_mat_mul(ctx, &dr_b, k, n, &transpose(r_b, k, n), k)?;
            let wide_inner = sub_vec(&qtgq, &gr_rt);

            let mut lower_skew = vec![T::zero(); k * k];
            for j in 0..k {
                for i in j..k {
                    lower_skew[i + j * k] = wide_inner[i + j * k] - wide_inner[j + i * k];
                }
            }

            let q_lower = backend_mat_mul(ctx, q_b, m, k, &lower_skew, k)?;
            let q_lower_t = transpose(&q_lower, m, k);
            let mut r1 = vec![T::zero(); k * k];
            for j in 0..k {
                for i in 0..k {
                    r1[i + j * k] = r_b[i + j * k];
                }
            }
            let leading_t = backend_solve_tri(ctx, &r1, &q_lower_t, k, m, true)?;
            let leading = transpose(&leading_t, k, m);

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

/// Reverse-mode AD rule for LU (VJP / pullback).
///
/// The `pivot` argument must match the pivoting strategy used in the forward pass.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{lu_rrule, LuCotangent, LuPivot};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], &[3, 3], col)
///     .unwrap();
/// let cotangent = LuCotangent {
///     l: Some(Tensor::ones(&[3, 3], mem, col)),
///     u: None,
/// };
/// let grad_a = lu_rrule(&mut ctx, &a, &cotangent, LuPivot::Partial).unwrap();
/// ```
pub fn lu_rrule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &LuCotangent<T>,
    pivot: LuPivot,
) -> AdResult<Tensor<T>>
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

    if let Some(ref dl) = cotangent.l {
        if dl.dims() != result.l.dims() {
            return Err(to_ad_err(Error::InvalidArgument(format!(
                "lu_rrule L cotangent shape mismatch: expected {:?}, got {:?}",
                result.l.dims(),
                dl.dims()
            ))));
        }
    }
    if let Some(ref du) = cotangent.u {
        if du.dims() != result.u.dims() {
            return Err(to_ad_err(Error::InvalidArgument(format!(
                "lu_rrule U cotangent shape mismatch: expected {:?}, got {:?}",
                result.u.dims(),
                du.dims()
            ))));
        }
    }

    let (l_data, _) = extract_data(&result.l)?;
    let (u_data, _) = extract_data(&result.u)?;
    let dl_data = if let Some(ref dl) = cotangent.l {
        Some(extract_data(dl)?.0)
    } else {
        None
    };
    let du_data = if let Some(ref du) = cotangent.u {
        Some(extract_data(du)?.0)
    } else {
        None
    };
    let p_vec = result.p.as_ref();

    let mut grad_a = vec![T::zero(); m * n * bc];

    for b in 0..bc {
        let l_b = &l_data[b * m * k..(b + 1) * m * k];
        let u_b = &u_data[b * k * n..(b + 1) * k * n];
        let dl_b = dl_data
            .as_ref()
            .map(|data| &data[b * m * k..(b + 1) * m * k]);
        let du_b = du_data
            .as_ref()
            .map(|data| &data[b * k * n..(b + 1) * k * n]);

        let batch_grad = if m == n {
            let l_t = transpose(l_b, k, k);
            let mut inner = vec![T::zero(); k * k];

            if let Some(dl_b) = dl_b {
                let lt_dl = backend_mat_mul(ctx, &l_t, k, k, dl_b, k)?;
                inner = add_vec(&inner, &tril_strict(&lt_dl, k));
            }
            if let Some(du_b) = du_b {
                let du_ut = backend_mat_mul(ctx, du_b, k, k, &transpose(u_b, k, k), k)?;
                inner = add_vec(&inner, &triu(&du_ut, k));
            }

            let right_t = backend_solve_tri(ctx, u_b, &transpose(&inner, k, k), k, k, true)?;
            let right = transpose(&right_t, k, k);
            backend_solve_tri(ctx, &l_t, &right, k, k, true)?
        } else if m < n {
            let l_t = transpose(l_b, k, k);
            let u1: Vec<T> = {
                let mut out = vec![T::zero(); k * k];
                for j in 0..k {
                    for i in 0..k {
                        out[i + j * k] = u_b[i + j * k];
                    }
                }
                out
            };

            let mut core = vec![T::zero(); k * k];
            if let Some(dl_b) = dl_b {
                let lt_dl = backend_mat_mul(ctx, &l_t, k, k, dl_b, k)?;
                core = add_vec(&core, &lt_dl);
            }
            if let Some(du_b) = du_b {
                let mut du_triu = vec![T::zero(); k * n];
                for j in 0..n {
                    for i in 0..k {
                        if i <= j {
                            du_triu[i + j * k] = du_b[i + j * k];
                        }
                    }
                }
                let du_term = backend_mat_mul(ctx, &du_triu, k, n, &transpose(u_b, k, n), k)?;
                core = sub_vec(&core, &du_term);
            }

            let lower = tril_strict(&core, k);
            let lower_t = backend_solve_tri(ctx, &u1, &transpose(&lower, k, k), k, k, true)?;
            let leading = transpose(&lower_t, k, k);

            let mut pre_left = vec![T::zero(); k * n];
            for j in 0..k {
                for i in 0..k {
                    pre_left[i + j * k] = leading[i + j * k];
                }
            }
            if let Some(du_b) = du_b {
                for j in 0..k {
                    for i in 0..=j {
                        pre_left[i + j * k] = pre_left[i + j * k] + du_b[i + j * k];
                    }
                }
                for j in k..n {
                    for i in 0..k {
                        pre_left[i + j * k] = du_b[i + j * k];
                    }
                }
            }

            backend_solve_tri(ctx, &l_t, &pre_left, k, n, true)?
        } else {
            let l1: Vec<T> = {
                let mut out = vec![T::zero(); k * k];
                for j in 0..k {
                    for i in 0..k {
                        out[i + j * k] = l_b[i + j * m];
                    }
                }
                out
            };
            let l1_t = transpose(&l1, k, k);

            let mut core = vec![T::zero(); k * k];
            if let Some(du_b) = du_b {
                let du_term = backend_mat_mul(ctx, du_b, k, k, &transpose(u_b, k, k), k)?;
                core = add_vec(&core, &du_term);
            }
            if let Some(dl_b) = dl_b {
                let mut dl_tril = vec![T::zero(); m * k];
                for j in 0..k {
                    for i in (j + 1)..m {
                        dl_tril[i + j * m] = dl_b[i + j * m];
                    }
                }
                let lt_dl = backend_mat_mul(ctx, &transpose(l_b, m, k), k, m, &dl_tril, k)?;
                core = sub_vec(&core, &lt_dl);
            }

            let upper = triu(&core, k);
            let leading = backend_solve_tri(ctx, &l1_t, &upper, k, k, true)?;

            let mut pre_right = vec![T::zero(); m * k];
            for j in 0..k {
                for i in 0..k {
                    pre_right[i + j * m] = leading[i + j * k];
                }
            }
            if let Some(dl_b) = dl_b {
                for j in 0..k {
                    for i in (j + 1)..k {
                        pre_right[i + j * m] = pre_right[i + j * m] + dl_b[i + j * m];
                    }
                    for i in k..m {
                        pre_right[i + j * m] = dl_b[i + j * m];
                    }
                }
            }

            let batch_grad_t =
                backend_solve_tri(ctx, u_b, &transpose(&pre_right, m, k), k, m, true)?;
            transpose(&batch_grad_t, k, m)
        };

        let out = &mut grad_a[b * m * n..(b + 1) * m * n];
        if let Some(pv) = p_vec {
            let p_b = &pv[b * m..(b + 1) * m];
            for j in 0..n {
                for i in 0..m {
                    out[p_b[i] + j * m] = batch_grad[i + j * m];
                }
            }
        } else {
            out.copy_from_slice(&batch_grad);
        }
    }

    let dims = output_dims(&[m, n], batch_dims);
    tensor_from_data(grad_a, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))
}

/// Reverse-mode AD rule for eigendecomposition (VJP / pullback).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{eigen_rrule, EigenCotangent};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let cotangent = EigenCotangent {
///     values: Some(Tensor::ones(&[3], mem, col)),
///     vectors: None,
/// };
/// let grad_a = eigen_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn eigen_rrule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &EigenCotangent<T>,
) -> AdResult<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    // Symmetric eigendecomposition: A = V diag(E) V^T
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

    let mut grad_a = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let v_b = &v_data[b * n * n..(b + 1) * n * n];
        let e_b = &e_data[b * n..(b + 1) * n];

        // Build F-matrix (n×n): F_ij = 1/(e_j - e_i) for i≠j, 0 diagonal
        let mut f_mat = vec![T::zero(); n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let denom = e_b[j] - e_b[i];
                    f_mat[i + j * n] = T::one()
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

        // Inner matrix D = diag(dE) + F ⊙ (V^T dV + (V^T dV)^T) / 2
        let mut d_mat = vec![T::zero(); n * n];

        if let Some(ref de) = cotangent.values {
            let (de_data, _) = extract_data(de)?;
            let de_b = &de_data[b * n..(b + 1) * n];
            for i in 0..n {
                d_mat[i + i * n] = de_b[i];
            }
        }

        if let Some(ref dv) = cotangent.vectors {
            let (dv_data, _) = extract_data(dv)?;
            let dv_b = &dv_data[b * n * n..(b + 1) * n * n];
            let vt_dv = backend_mat_mul(ctx, &transpose(v_b, n, n), n, n, dv_b, n)?;
            let half: T = scalar_from(0.5).map_err(to_ad_err)?;
            for i in 0..n {
                for j in 0..n {
                    let skew = half * (vt_dv[i + j * n] - vt_dv[j + i * n]);
                    d_mat[i + j * n] = d_mat[i + j * n] + f_mat[i + j * n] * skew;
                }
            }
        }

        // dA = V D V^T
        let vd = backend_mat_mul(ctx, v_b, n, n, &d_mat, n)?;
        let da_b = backend_mat_mul(ctx, &vd, n, n, &transpose(v_b, n, n), n)?;

        grad_a[b * n * n..(b + 1) * n * n].copy_from_slice(&da_b);
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(grad_a, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))
}

/// Reverse-mode AD rule for least squares (VJP / pullback).
///
/// Returns cotangents for both `A` and `b`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::lstsq_rrule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[1.0, 0.0, 1.0, 0.0, 1.0, 1.0], &[3, 2], col).unwrap();
/// let b = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3], col).unwrap();
/// let dx = Tensor::<f64>::ones(&[2], mem, col);
/// let grad = lstsq_rrule(&mut ctx, &a, &b, &dx).unwrap();
/// // grad.a: cotangent for A, grad.b: cotangent for b
/// ```
pub fn lstsq_rrule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
    cotangent_x: &Tensor<T>,
) -> AdResult<LstsqGrad<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("lstsq_rrule").map_err(to_ad_err)?;

    let result = lstsq(ctx, a, b)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(a)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    if cotangent_x.dims() != result.x.dims() {
        return Err(to_ad_err(Error::InvalidArgument(format!(
            "lstsq_rrule cotangent shape mismatch: expected {:?}, got {:?}",
            result.x.dims(),
            cotangent_x.dims()
        ))));
    }

    let (a_data, _) = extract_data(a)?;
    let (x_data, _) = extract_data(&result.x)?;
    let (r_data, _) = extract_data(&result.residual)?;
    let (dx_data, _) = extract_data(cotangent_x)?;

    let mut grad_a_data = vec![T::zero(); m * n * bc];
    let mut grad_b_data = vec![T::zero(); m * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
        let x_b = &x_data[batch * n..(batch + 1) * n];
        let r_b = &r_data[batch * m..(batch + 1) * m];
        let dx_b = &dx_data[batch * n..(batch + 1) * n];

        let (q_d, r_d) = backend_qr(ctx, a_b, m, n)?;
        let y = backend_solve_tri(ctx, &transpose(&r_d, n, n), dx_b, n, 1, false)?;
        let z = backend_solve_tri(ctx, &r_d, &y, n, 1, true)?;
        let grad_b = backend_mat_mul(ctx, &q_d, m, n, &y, 1)?;

        for j in 0..n {
            for i in 0..m {
                grad_a_data[batch * m * n + i + j * m] = r_b[i] * z[j] - grad_b[i] * x_b[j];
            }
        }
        grad_b_data[batch * m..(batch + 1) * m].copy_from_slice(&grad_b);
    }

    let a_dims = output_dims(&[m, n], batch_dims);
    let b_dims = output_dims(&[m], batch_dims);
    Ok(LstsqGrad {
        a: tensor_from_data(grad_a_data, &a_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
        b: tensor_from_data(grad_b_data, &b_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
    })
}

/// Reverse-mode AD rule for Cholesky (VJP / pullback).
///
/// Given `A = L L†` and cotangent `L̄`, computes `Ā`.
///
/// # Examples
///
/// ```no_run
/// use tenferro_linalg::cholesky_rrule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let cotangent = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let grad_a = cholesky_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn cholesky_rrule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &Tensor<T>,
) -> AdResult<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    // A = L L^T, dA = L^{-T} phi*(tril(L^T dL)) L^{-1}
    let l = cholesky(ctx, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (l_data, _) = extract_data(&l)?;
    let (dl_data, _) = extract_data(cotangent)?;

    let mut grad_a = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let l_b = &l_data[b * n * n..(b + 1) * n * n];
        let dl_b = &dl_data[b * n * n..(b + 1) * n * n];

        // S = tril(L^T dL)
        let lt_dl = backend_mat_mul(ctx, &transpose(l_b, n, n), n, n, dl_b, n)?;
        let s = tril(&lt_dl, n);

        // Apply phi*: symmetrize S → (S + S^T) / 2
        let s_sym = phi_star(&s, n)?;

        // Solve L^T x = S_sym → x = L^{-T} S_sym
        let x = backend_solve_tri(ctx, &transpose(l_b, n, n), &s_sym, n, n, true)?;

        // Solve x L = result → result = x L^{-1} → L^T result^T = x^T → result^T = L^{-T} x^T
        let xt = transpose(&x, n, n);
        let result_t = backend_solve_tri(ctx, &transpose(l_b, n, n), &xt, n, n, true)?;
        let da_b = transpose(&result_t, n, n);

        grad_a[b * n * n..(b + 1) * n * n].copy_from_slice(&da_b);
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(grad_a, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))
}

/// Reverse-mode AD rule for linear solve (VJP / pullback).
///
/// Given `Ax = b` and cotangent `x̄`, computes `(Ā, b̄)`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::solve_rrule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col);
/// let b = Tensor::<f64>::ones(&[3], mem, col);
/// let cotangent = Tensor::<f64>::ones(&[3], mem, col);
/// let grad = solve_rrule(&mut ctx, &a, &b, &cotangent).unwrap();
/// ```
pub fn solve_rrule<T: LinalgScalar, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
    cotangent: &Tensor<T>,
) -> AdResult<SolveGrad<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    // Ax = b → G = A^{-H} dx, dB = G, dA = -G x^H
    let x = solve(ctx, a, b)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(a)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);
    let rhs = backend::tensor_helpers::validate_solve_rhs_shape(b, n, batch_dims, "solve_rrule")
        .map_err(to_ad_err)?;
    let nrhs = rhs.nrhs;

    let (a_data, _) = extract_data(a)?;
    let (x_data, _) = extract_data(&x)?;
    let (dx_data, _) = extract_data(cotangent)?;

    let mut grad_a_data = vec![T::zero(); n * n * bc];
    let mut grad_b_data = vec![T::zero(); n * nrhs * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
        let x_b = &x_data[batch * n * nrhs..(batch + 1) * n * nrhs];
        let dx_b = &dx_data[batch * n * nrhs..(batch + 1) * n * nrhs];

        // G = A^{-H} dx = solve(A^H, dx)
        let at = adjoint_transpose(a_b, n, n);
        let g = backend_solve(ctx, &at, dx_b, n, nrhs)?;

        // dB = G
        grad_b_data[batch * n * nrhs..(batch + 1) * n * nrhs].copy_from_slice(&g);

        // dA = -G x^H (n×nrhs × nrhs×n = n×n)
        let x_h = adjoint_transpose(x_b, n, nrhs);
        let g_xh = backend_mat_mul(ctx, &g, n, nrhs, &x_h, n)?;
        let neg_g_xh = scale_vec(&g_xh, -T::one());
        grad_a_data[batch * n * n..(batch + 1) * n * n].copy_from_slice(&neg_g_xh);
    }

    let a_dims = output_dims(&[n, n], batch_dims);
    let b_dims = rhs.output_dims;
    Ok(SolveGrad {
        a: tensor_from_data(grad_a_data, &a_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
        b: tensor_from_data(grad_b_data, &b_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
    })
}

/// Reverse-mode AD rule for triangular solve (VJP / pullback).
///
/// Given `A x = b` with triangular `A` and cotangent `x̄`, computes `(Ā, b̄)`.
///
/// - `G = A^{-H} x̄` solved with conjugate-transposed triangular structure
/// - `b̄ = G`
/// - `Ā = proj(-G x^H)` where `proj = triu` for upper, `tril` for lower
pub fn solve_triangular_rrule<T: LinalgScalar, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
    cotangent: &Tensor<T>,
    upper: bool,
) -> AdResult<SolveGrad<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    let x = solve_triangular(ctx, a, b, upper)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(a)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);
    let rhs = backend::tensor_helpers::validate_solve_rhs_shape(
        b,
        n,
        batch_dims,
        "solve_triangular_rrule",
    )
    .map_err(to_ad_err)?;
    let nrhs = rhs.nrhs;

    let (a_data, _) = extract_data(a)?;
    let (x_data, _) = extract_data(&x)?;
    let (dx_data, _) = extract_data(cotangent)?;

    let mut grad_a_data = vec![T::zero(); n * n * bc];
    let mut grad_b_data = vec![T::zero(); n * nrhs * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
        let x_b = &x_data[batch * n * nrhs..(batch + 1) * n * nrhs];
        let dx_b = &dx_data[batch * n * nrhs..(batch + 1) * n * nrhs];

        // G = A^{-H} dX, where A^H flips upper/lower.
        let at = adjoint_transpose(a_b, n, n);
        let g = backend_solve_tri(ctx, &at, dx_b, n, nrhs, !upper)?;

        // dB = G
        grad_b_data[batch * n * nrhs..(batch + 1) * n * nrhs].copy_from_slice(&g);

        // dA = proj(-G x^H)
        let x_h = adjoint_transpose(x_b, n, nrhs);
        let g_xh = backend_mat_mul(ctx, &g, n, nrhs, &x_h, n)?;
        let neg_g_xh = scale_vec(&g_xh, -T::one());
        let projected = if upper {
            triu(&neg_g_xh, n)
        } else {
            tril(&neg_g_xh, n)
        };
        grad_a_data[batch * n * n..(batch + 1) * n * n].copy_from_slice(&projected);
    }

    let a_dims = output_dims(&[n, n], batch_dims);
    let b_dims = rhs.output_dims;
    Ok(SolveGrad {
        a: tensor_from_data(grad_a_data, &a_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
        b: tensor_from_data(grad_b_data, &b_dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
    })
}

/// Reverse-mode AD rule for matrix inverse (VJP / pullback).
///
/// `Ā = -A⁻ᴴ · cotangent · A⁻ᴴ`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::inv_rrule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col);
/// let cotangent = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let grad_a = inv_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn inv_rrule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &Tensor<T>,
) -> AdResult<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("inv_rrule").map_err(to_ad_err)?;

    // dA = -B^T dB B^T where B = A^{-1}
    let b_inv = inv(ctx, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (binv_data, _) = extract_data(&b_inv)?;
    let (db_data, _) = extract_data(cotangent)?;

    let mut grad_a = vec![T::zero(); n * n * bc];

    for batch in 0..bc {
        let b_b = &binv_data[batch * n * n..(batch + 1) * n * n];
        let db_b = &db_data[batch * n * n..(batch + 1) * n * n];

        let bt = transpose(b_b, n, n);
        let bt_db = backend_mat_mul(ctx, &bt, n, n, db_b, n)?;
        let bt_db_bt = backend_mat_mul(ctx, &bt_db, n, n, &bt, n)?;
        let neg = scale_vec(&bt_db_bt, -T::one());
        grad_a[batch * n * n..(batch + 1) * n * n].copy_from_slice(&neg);
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(grad_a, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))
}

/// Reverse-mode AD rule for determinant (VJP / pullback).
///
/// `Ā = det(A) · cotangent · A⁻ᵀ`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::det_rrule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col);
/// let cotangent = Tensor::<f64>::ones(&[], mem, col);
/// let grad_a = det_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn det_rrule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &Tensor<T>,
) -> AdResult<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("det_rrule").map_err(to_ad_err)?;

    // dA = ddet * det(A) * A^{-T}
    let det_val = det(ctx, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor)?;
    let (det_data, _) = extract_data(&det_val)?;
    let (ddet_data, _) = extract_data(cotangent)?;

    let mut grad_a = vec![T::zero(); n * n * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
        let d = det_data[batch];
        let dd = ddet_data[batch];

        // A^{-T}
        let a_inv = backend_solve(ctx, a_b, &eye::<T>(n), n, n)?;
        let a_inv_t = transpose(&a_inv, n, n);

        let scale = dd * d;
        let da_b = scale_vec(&a_inv_t, scale);
        grad_a[batch * n * n..(batch + 1) * n * n].copy_from_slice(&da_b);
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(grad_a, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))
}

/// Reverse-mode AD rule for slogdet (VJP / pullback).
///
/// `Ā = cotangent_logabsdet · A⁻ᵀ`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{slogdet_rrule, SlogdetCotangent};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col);
/// let cotangent = SlogdetCotangent {
///     logabsdet: Some(Tensor::ones(&[], mem, col)),
/// };
/// let grad_a = slogdet_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn slogdet_rrule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &SlogdetCotangent<T>,
) -> AdResult<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("slogdet_rrule").map_err(to_ad_err)?;

    // dA = d_logabsdet * A^{-T}
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor)?;

    let mut grad_a = vec![T::zero(); n * n * bc];

    if let Some(ref dlog) = cotangent.logabsdet {
        let (dlog_data, _) = extract_data(dlog)?;
        for batch in 0..bc {
            let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
            let dl = dlog_data[batch];

            let a_inv = backend_solve(ctx, a_b, &eye::<T>(n), n, n)?;
            let a_inv_t = transpose(&a_inv, n, n);
            let da_b = scale_vec(&a_inv_t, dl);
            grad_a[batch * n * n..(batch + 1) * n * n].copy_from_slice(&da_b);
        }
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(grad_a, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))
}

/// Reverse-mode AD rule for general eigendecomposition (VJP / pullback).
///
/// Given eigendecomposition `A V = V diag(lambda)`, computes the gradient
/// of the input `A` from complex-valued cotangents for eigenvalues and
/// eigenvectors using the Mike Giles formulas.
///
/// The cotangent uses [`EigCotangent`] with complex-valued tensors
/// because `eig()` returns complex output even for real inputs.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{eig_rrule, EigCotangent};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
/// use num_complex::Complex64;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let cotangent = EigCotangent::<f64> {
///     values: None,
///     vectors: None,
/// };
/// let grad_a = eig_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn eig_rrule<
    T: LinalgScalar<Real = T, Complex = num_complex::Complex<T>> + num_traits::Float,
    C,
>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &EigCotangent<T>,
) -> AdResult<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    let (n, batch_dims) = validate_square(tensor).map_err(to_ad_err)?;
    let bc = batch_count(batch_dims);

    // Compute eigendecomposition
    let eig_result = eig(ctx, tensor).map_err(to_ad_err)?;
    let val_data = extract_data_scalar(&eig_result.values)?;
    let vec_data = extract_data_scalar(&eig_result.vectors)?;

    let zero_c = Cx::new(T::zero(), T::zero());
    let one_c = Cx::new(T::one(), T::zero());

    let mut grad_data = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let lambda = &val_data[b * n..(b + 1) * n];
        let v = &vec_data[b * n * n..(b + 1) * n * n];

        // Compute F matrix: F[i,j] = 1/(lambda_j - lambda_i) for i != j, 0 on diagonal
        let mut f_mat = vec![zero_c; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let diff = lambda[j] - lambda[i];
                    f_mat[i + j * n] = one_c / diff;
                }
            }
        }

        // V^H (conjugate transpose of V)
        let vh = complex_conj_transpose(v, n);

        // Build M_bar = diag(d_bar_lambda) + F .* (V^H d_bar_V)
        let mut m_bar = vec![zero_c; n * n];

        if let Some(ref dv_bar) = cotangent.vectors {
            let dv_bar_data = extract_data_scalar(dv_bar)?;
            let dv_bar_b = &dv_bar_data[b * n * n..(b + 1) * n * n];
            let vh_dv = complex_mat_mul_nn(&vh, dv_bar_b, n);
            for k in 0..n * n {
                m_bar[k] = f_mat[k] * vh_dv[k];
            }
        }

        if let Some(ref dlam) = cotangent.values {
            let dlam_data = extract_data_scalar(dlam)?;
            for i in 0..n {
                m_bar[i + i * n] = m_bar[i + i * n] + dlam_data[b * n + i];
            }
        }

        // d_bar_A = V^{-H} M_bar V^H = solve(V^H, M_bar @ V^H)
        let m_vh = complex_mat_mul_nn(&m_bar, &vh, n);
        let da_complex = complex_solve_nn(ctx, &vh, &m_vh, n)?;

        // Take real part (since input A was real)
        for k in 0..n * n {
            grad_data[b * n * n + k] = da_complex[k].re;
        }
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(grad_data, &dims).map_err(to_ad_err)
}

/// Reverse-mode AD rule for pseudoinverse (VJP / pullback).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::pinv_rrule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
/// let cotangent = Tensor::<f64>::ones(&[4, 3], mem, col);
/// let grad_a = pinv_rrule(&mut ctx, &a, &cotangent, None).unwrap();
/// ```
pub fn pinv_rrule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &Tensor<T>,
    rcond: Option<f64>,
) -> AdResult<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("pinv_rrule").map_err(to_ad_err)?;

    // dA = -(A+)^T dA+ (A+)^T + (I - AA+)(dA+)^T A+(A+)^T + (A+)^T A+ (dA+)^T (I - A+A)
    let ap = pinv(ctx, tensor, rcond)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor)?;
    let (ap_data, _) = extract_data(&ap)?;
    let (dap_data, _) = extract_data(cotangent)?;

    let mut grad_a = vec![T::zero(); m * n * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
        let ap_b = &ap_data[batch * n * m..(batch + 1) * n * m];
        let dap_b = &dap_data[batch * n * m..(batch + 1) * n * m];

        let apt = transpose(ap_b, n, m); // m×n
        let dapt = transpose(dap_b, n, m); // m×n

        // Term 1: -(A+)^T dA+ (A+)^T = -apt * dap * apt^T
        // apt: m×n, dap: n×m, apt: m×n → m×n * n×m * m×n = m×n
        let t1 = backend_mat_mul(ctx, &apt, m, n, dap_b, m)?;
        let t1 = backend_mat_mul(ctx, &t1, m, m, &apt, n)?;
        let t1 = scale_vec(&t1, -T::one());

        // Term 2: (I - AA+)(dA+)^T A+ (A+)^T
        // AA+ (m×m)
        let aap = backend_mat_mul(ctx, a_b, m, n, ap_b, m)?;
        let i_m = eye::<T>(m);
        let i_aap = sub_vec(&i_m, &aap);
        // (dA+)^T A+ = dapt * ap (m×n * n×m = m×m)
        let dapt_ap = backend_mat_mul(ctx, &dapt, m, n, ap_b, m)?;
        // * (A+)^T = * apt (m×m * m×n = m×n)
        let dapt_ap_apt = backend_mat_mul(ctx, &dapt_ap, m, m, &apt, n)?;
        let t2 = backend_mat_mul(ctx, &i_aap, m, m, &dapt_ap_apt, n)?;

        // Term 3: (A+)^T A+ (dA+)^T (I - A+A)
        // A+A (n×n)
        let apa = backend_mat_mul(ctx, ap_b, n, m, a_b, n)?;
        let i_n = eye::<T>(n);
        let i_apa = sub_vec(&i_n, &apa);
        // (A+)^T A+ = apt * ap (m×n * n×m = m×m)
        let apt_ap = backend_mat_mul(ctx, &apt, m, n, ap_b, m)?;
        // * (dA+)^T = * dapt (m×m * m×n = m×n)
        let apt_ap_dapt = backend_mat_mul(ctx, &apt_ap, m, m, &dapt, n)?;
        let t3 = backend_mat_mul(ctx, &apt_ap_dapt, m, n, &i_apa, n)?;

        let da_b = add_vec(&t1, &add_vec(&t2, &t3));
        grad_a[batch * m * n..(batch + 1) * m * n].copy_from_slice(&da_b);
    }

    let dims = output_dims(&[m, n], batch_dims);
    tensor_from_data(grad_a, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))
}

/// Reverse-mode AD rule for matrix exponential (VJP / pullback).
///
/// Computes the gradient of the input given a cotangent for `exp(A)`.
/// Uses the auxiliary 2n x 2n matrix trick (PyTorch approach):
///
/// ```text
/// M = [[A^T, cotangent], [0, A^T]]
/// grad_A = top-right n×n block of exp(M)
/// ```
///
/// # Examples
///
/// ```
/// use tenferro_linalg::matrix_exp_rrule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let cotangent = Tensor::<f64>::ones(&[3, 3], mem, col);
/// let grad_a = matrix_exp_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn matrix_exp_rrule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &Tensor<T>,
) -> AdResult<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("matrix_exp_rrule").map_err(to_ad_err)?;

    let (n, batch_dims) = validate_square(tensor).map_err(to_ad_err)?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor)?;
    let (co_data, _) = extract_data(cotangent)?;

    let nn = 2 * n;
    let mut grad_data = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let a = &a_data[b * n * n..(b + 1) * n * n];
        let co = &co_data[b * n * n..(b + 1) * n * n];

        // Build 2n×2n auxiliary matrix M = [[A^T, cotangent], [0, A^T]]
        let mut m = vec![T::zero(); nn * nn];
        for j in 0..n {
            for i in 0..n {
                // A^T: transpose of A — a^T[i,j] = a[j,i] = a[j + i*n]
                let a_t_ij = a[j + i * n];
                // Top-left: A^T
                m[i + j * nn] = a_t_ij;
                // Top-right: cotangent
                m[i + (j + n) * nn] = co[i + j * n];
                // Bottom-right: A^T
                m[(i + n) + (j + n) * nn] = a_t_ij;
                // Bottom-left: already zero
            }
        }

        // Compute exp(M)
        let exp_m = matrix_exp_single(ctx, &m, nn).map_err(to_ad_err)?;

        // Extract top-right block → gradient d̄A
        for j in 0..n {
            for i in 0..n {
                grad_data[b * n * n + i + j * n] = exp_m[i + (j + n) * nn];
            }
        }
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(grad_data, &dims).map_err(to_ad_err)
}

/// Reverse-mode AD rule for norm (VJP / pullback).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{norm_rrule, NormKind};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
/// let cotangent = Tensor::<f64>::ones(&[], mem, col);
/// let grad_a = norm_rrule(&mut ctx, &a, &cotangent, NormKind::Fro).unwrap();
/// ```
pub fn norm_rrule<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &Tensor<T>,
    kind: NormKind,
) -> AdResult<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("norm_rrule").map_err(to_ad_err)?;

    if tensor.ndim() == 1 {
        validate_norm_cotangent(cotangent, &[]).map_err(to_ad_err)?;
        let (a_data, _) = extract_data(tensor)?;
        let (dn_data, _) = extract_data(cotangent)?;
        let dn = dn_data[0];
        let len = tensor.dims()[0];
        let mut grad_a = vec![T::zero(); len];

        match kind {
            NormKind::Fro => {
                let nrm = norm(ctx, tensor, NormKind::Fro).map_err(to_ad_err)?;
                let (nrm_data, _) = extract_data(&nrm)?;
                let nv = nrm_data[0];
                let scale = if nv > T::zero() { dn / nv } else { T::zero() };
                for i in 0..len {
                    grad_a[i] = scale * a_data[i];
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
                    grad_a[i] = dn * sign;
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
                    let active_count = scalar_from::<T>(active.len() as f64).map_err(to_ad_err)?;
                    let scale = dn / active_count;
                    for i in active {
                        let v = a_data[i];
                        let sign = if v > T::zero() {
                            T::one()
                        } else if v < T::zero() {
                            -T::one()
                        } else {
                            T::zero()
                        };
                        grad_a[i] = scale * sign;
                    }
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
                        grad_a[i] = dn * sign;
                    }
                } else {
                    let nrm = norm(ctx, tensor, kind).map_err(to_ad_err)?;
                    let (nrm_data, _) = extract_data(&nrm)?;
                    let nv = nrm_data[0];
                    if nv > T::zero() {
                        let p_minus_one = scalar_from::<T>(p - 1.0).map_err(to_ad_err)?;
                        let scale = dn / nv.powf(p_minus_one);
                        for i in 0..len {
                            let v = a_data[i];
                            let sign = if v > T::zero() {
                                T::one()
                            } else if v < T::zero() {
                                -T::one()
                            } else {
                                T::zero()
                            };
                            grad_a[i] = scale * sign * v.abs().powf(p_minus_one);
                        }
                    }
                }
            }
            NormKind::Nuclear | NormKind::Spectral => {
                return Err(matrix_only_norm_kind_ad_error(kind));
            }
        }

        return tensor_from_data(grad_a, &[len]).map_err(to_ad_err);
    }

    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);
    validate_norm_cotangent(cotangent, batch_dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;

    let (a_data, _) = extract_data(tensor)?;
    let (dn_data, _) = extract_data(cotangent)?;

    let mut grad_a = vec![T::zero(); m * n * bc];

    match kind {
        NormKind::Fro => {
            // dA = dn * A / ||A||_F
            let nrm = norm(ctx, tensor, NormKind::Fro)
                .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
            let (nrm_data, _) = extract_data(&nrm)?;
            for batch in 0..bc {
                let dn = dn_data[batch];
                let nv = nrm_data[batch];
                let scale = if nv > T::zero() { dn / nv } else { T::zero() };
                for i in 0..m * n {
                    grad_a[batch * m * n + i] = scale * a_data[batch * m * n + i];
                }
            }
        }
        NormKind::Nuclear => {
            // dA = dn * U V^T
            for batch in 0..bc {
                let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
                let (u, _s, v) = backend_thin_svd(ctx, a_b, m, n)?;
                let k = m.min(n);
                let uv = backend_mat_mul(ctx, &u, m, k, &transpose(&v, n, k), n)?;
                let dn = dn_data[batch];
                for i in 0..m * n {
                    grad_a[batch * m * n + i] = dn * uv[i];
                }
            }
        }
        NormKind::Spectral => {
            // dA = dn * u1 v1^T (rank-1 outer product of leading singular vectors)
            for batch in 0..bc {
                let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
                let (u, _s, v) = backend_thin_svd(ctx, a_b, m, n)?;
                let dn = dn_data[batch];
                for j in 0..n {
                    for i in 0..m {
                        grad_a[batch * m * n + i + j * m] = dn * u[i] * v[j];
                    }
                }
            }
        }
        NormKind::L1 => {
            // dA = dn * sign(A) on columns that attain max absolute column sum.
            // At ties, average uniformly over active columns.
            for batch in 0..bc {
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
                let active_count = scalar_from::<T>(active_cols.len() as f64).map_err(to_ad_err)?;
                let dn = dn_data[batch] / active_count;
                for j in active_cols {
                    for i in 0..m {
                        let v = a_data[base + i + j * m];
                        let sign = if v > T::zero() {
                            T::one()
                        } else if v < T::zero() {
                            -T::one()
                        } else {
                            T::zero()
                        };
                        grad_a[base + i + j * m] = grad_a[base + i + j * m] + dn * sign;
                    }
                }
            }
        }
        NormKind::Inf => {
            // dA = dn * sign(A) on rows that attain max absolute row sum.
            // At ties, average uniformly over active rows.
            for batch in 0..bc {
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
                let active_count = scalar_from::<T>(active_rows.len() as f64).map_err(to_ad_err)?;
                let dn = dn_data[batch] / active_count;
                for i in active_rows {
                    for j in 0..n {
                        let v = a_data[base + i + j * m];
                        let sign = if v > T::zero() {
                            T::one()
                        } else if v < T::zero() {
                            -T::one()
                        } else {
                            T::zero()
                        };
                        grad_a[base + i + j * m] = grad_a[base + i + j * m] + dn * sign;
                    }
                }
            }
        }
        _ => {
            return Err(chainrules_core::AutodiffError::ModeNotSupported {
                mode: "norm_rrule".into(),
                reason: format!("norm kind {kind:?} AD not yet implemented"),
            });
        }
    }

    let dims = output_dims(&[m, n], batch_dims);
    tensor_from_data(grad_a, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))
}
