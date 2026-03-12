use super::*;

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
pub fn lu_frule<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
    pivot: LuPivot,
) -> AdResult<(LuResult<T>, LuResult<T>)>
where
    T: KernelLinalgScalar,
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
pub fn eigen_frule<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(EigenResult<T>, EigenResult<T>)>
where
    T: KernelLinalgScalar,
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
