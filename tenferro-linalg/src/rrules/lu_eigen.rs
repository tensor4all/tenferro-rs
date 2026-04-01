use super::*;
use num_traits::Float;
use tenferro_algebra::Conjugate;

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
///     l: Some(Tensor::ones(&[3, 3], mem, col).unwrap()),
///     u: None,
/// };
/// let grad_a = lu_rrule(&mut ctx, &a, &cotangent, LuPivot::Partial).unwrap();
/// ```
pub fn lu_rrule<T, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &LuCotangent<T>,
    pivot: LuPivot,
) -> AdResult<Tensor<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorMetadataContextFor
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T::Real>>,
    C::MetadataBackend: tenferro_prims::TensorMetadataPrims<Context = C>,
    <C as tenferro_prims::TensorScalarContextFor<
        tenferro_algebra::Standard<T::Real>,
    >>::ScalarBackend: tenferro_prims::TensorMetadataCastPrims<T::Real, Context = C>,
    T: crate::primal::LiftPermutationMatrixTensor<C>,
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
    let p_vec = crate::forward_perm_from_permutation_matrix(&result.p, m, bc)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;

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
            let l_h = adjoint_transpose(l_b, k, k);
            let mut inner = vec![T::zero(); k * k];

            if let Some(dl_b) = dl_b {
                let lt_dl = backend_mat_mul(ctx, &l_h, k, k, dl_b, k)?;
                inner = add_vec(&inner, &tril_strict(&lt_dl, k));
            }
            if let Some(du_b) = du_b {
                let du_ut = backend_mat_mul(ctx, du_b, k, k, &adjoint_transpose(u_b, k, k), k)?;
                inner = add_vec(&inner, &triu(&du_ut, k));
            }

            let left = backend_solve_tri(ctx, &l_h, &inner, k, k, true)?;
            let grad_h = backend_solve_tri(ctx, u_b, &adjoint_transpose(&left, k, k), k, k, true)?;
            adjoint_transpose(&grad_h, k, k)
        } else if m < n {
            let l_h = adjoint_transpose(l_b, k, k);
            let u1 = u_b[..k * k].to_vec();
            let u2 = u_b[k * k..].to_vec();
            let mut lower_source = vec![T::zero(); k * k];
            if let Some(dl_b) = dl_b {
                let lt_dl = backend_mat_mul(ctx, &l_h, k, k, dl_b, k)?;
                lower_source = add_vec(&lower_source, &lt_dl);
            }
            if let Some(du_b) = du_b.filter(|_| n > k) {
                let du2 = &du_b[k * k..];
                let du2_u2h =
                    backend_mat_mul(ctx, du2, k, n - k, &adjoint_transpose(&u2, k, n - k), k)?;
                lower_source = sub_vec(&lower_source, &du2_u2h);
            }

            let mut inner = tril_strict(&lower_source, k);
            if let Some(du_b) = du_b {
                let du1 = &du_b[..k * k];
                let du1_u1h = backend_mat_mul(ctx, du1, k, k, &adjoint_transpose(&u1, k, k), k)?;
                inner = add_vec(&inner, &triu(&du1_u1h, k));
            }

            let leading_h = backend_solve_tri(
                ctx,
                u1.as_slice(),
                &adjoint_transpose(&inner, k, k),
                k,
                k,
                true,
            )?;
            let leading = adjoint_transpose(&leading_h, k, k);

            let mut pre_left = vec![T::zero(); k * n];
            pre_left[..k * k].copy_from_slice(&leading);
            if let Some(du_b) = du_b.filter(|_| n > k) {
                pre_left[k * k..].copy_from_slice(&du_b[k * k..]);
            }

            backend_solve_tri(ctx, &l_h, &pre_left, k, n, true)?
        } else {
            let mut l1 = vec![T::zero(); k * k];
            let mut l2 = vec![T::zero(); (m - k) * k];
            for j in 0..k {
                for i in 0..k {
                    l1[i + j * k] = l_b[i + j * m];
                }
                for i in k..m {
                    l2[(i - k) + j * (m - k)] = l_b[i + j * m];
                }
            }
            let l1_h = adjoint_transpose(&l1, k, k);

            let mut inner = vec![T::zero(); k * k];
            if let Some(dl_b) = dl_b {
                let mut dl1 = vec![T::zero(); k * k];
                let mut dl2 = vec![T::zero(); (m - k) * k];
                for j in 0..k {
                    for i in 0..k {
                        dl1[i + j * k] = dl_b[i + j * m];
                    }
                    for i in k..m {
                        dl2[(i - k) + j * (m - k)] = dl_b[i + j * m];
                    }
                }
                let l1h_dl1 = backend_mat_mul(ctx, &l1_h, k, k, &dl1, k)?;
                inner = add_vec(&inner, &tril_strict(&l1h_dl1, k));
                if m > k {
                    let l2h_dl2 =
                        backend_mat_mul(ctx, &adjoint_transpose(&l2, m - k, k), k, m - k, &dl2, k)?;
                    inner = sub_vec(&inner, &triu(&l2h_dl2, k));
                }
            }
            if let Some(du_b) = du_b {
                let du_term = backend_mat_mul(ctx, du_b, k, k, &adjoint_transpose(u_b, k, k), k)?;
                inner = add_vec(&inner, &triu(&du_term, k));
            }

            let leading = backend_solve_tri(ctx, &l1_h, &inner, k, k, true)?;

            let mut pre_right = vec![T::zero(); m * k];
            for j in 0..k {
                for i in 0..k {
                    pre_right[i + j * m] = leading[i + j * k];
                }
            }
            if let Some(dl_b) = dl_b {
                for j in 0..k {
                    for i in k..m {
                        pre_right[i + j * m] = dl_b[i + j * m];
                    }
                }
            }

            let batch_grad_h =
                backend_solve_tri(ctx, u_b, &adjoint_transpose(&pre_right, m, k), k, m, true)?;
            adjoint_transpose(&batch_grad_h, k, m)
        };

        let out = &mut grad_a[b * m * n..(b + 1) * m * n];
        let p_b = &p_vec[b * m..(b + 1) * m];
        for j in 0..n {
            for i in 0..m {
                out[p_b[i] + j * m] = batch_grad[i + j * m];
            }
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
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col).unwrap();
/// let cotangent = EigenCotangent {
///     values: Some(Tensor::ones(&[3], mem, col).unwrap()),
///     vectors: None,
/// };
/// let grad_a = eigen_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn eigen_rrule<T, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &EigenCotangent<T, T::Real>,
) -> AdResult<Tensor<T>>
where
    T: KernelLinalgScalar + Conjugate,
    T::Real: KernelLinalgScalar<Real = T::Real> + num_traits::Float,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    // Hermitian eigendecomposition: A = V diag(E) V^H
    let result = eigen(ctx, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);
    // Regularization for the F-matrix: prevents division by zero when two
    // singular values are (nearly) equal.  We use max(1e-40, T::epsilon())
    // so that on f32 (where 1e-40 underflows to 0) we still get a safe floor.
    let eta: T::Real = {
        let raw: T::Real = scalar_from(1e-40).map_err(to_ad_err)?;
        let eps = T::Real::epsilon();
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

        // Build F-matrix (n×n): F_ij = (e_i - e_j)/((e_i - e_j)^2 + eta), 0 diagonal.
        let mut f_mat = vec![T::Real::zero(); n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let gap = e_b[i] - e_b[j];
                    f_mat[i + j * n] = gap / (gap * gap + eta);
                }
            }
        }

        // Inner matrix D = diag(dE) + 1/2 * (H + H^H),
        // where H = F ⊙ (V^H dV).
        let mut d_mat = vec![T::zero(); n * n];

        if let Some(ref de) = cotangent.values {
            let (de_data, _) = extract_data(de)?;
            let de_b = &de_data[b * n..(b + 1) * n];
            for i in 0..n {
                d_mat[i + i * n] = T::from_real(de_b[i]);
            }
        }

        if let Some(ref dv) = cotangent.vectors {
            let (dv_data, _) = extract_data(dv)?;
            let dv_b = &dv_data[b * n * n..(b + 1) * n * n];
            let dv_h_v = backend_mat_mul(ctx, &adjoint_transpose(dv_b, n, n), n, n, v_b, n)?;
            let half: T::Real = scalar_from(0.5).map_err(to_ad_err)?;
            for i in 0..n {
                for j in 0..n {
                    let h_ij = T::from_real(f_mat[i + j * n]) * dv_h_v[i + j * n];
                    let h_h_ij = (T::from_real(f_mat[j + i * n]) * dv_h_v[j + i * n]).conj();
                    d_mat[i + j * n] = d_mat[i + j * n] + (h_ij + h_h_ij) * T::from_real(half);
                }
            }
        }

        // dA = V D V^H
        let vd = backend_mat_mul(ctx, v_b, n, n, &d_mat, n)?;
        let da_b = backend_mat_mul(ctx, &vd, n, n, &adjoint_transpose(v_b, n, n), n)?;

        grad_a[b * n * n..(b + 1) * n * n].copy_from_slice(&da_b);
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(grad_a, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))
}
