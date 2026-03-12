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
pub fn svd_rrule<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &SvdCotangent<T>,
    options: Option<&SvdOptions>,
) -> AdResult<Tensor<T>>
where
    T: KernelLinalgScalar,
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
pub fn qr_rrule<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
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
