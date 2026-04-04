use super::*;
use num_traits::{Float, One};
use tenferro_algebra::Conjugate;

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
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col).unwrap();
/// let (result, dresult) = lu_frule(&mut ctx, &a, &da, LuPivot::Partial).unwrap();
/// ```
pub fn lu_frule<T, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
    pivot: LuPivot,
) -> AdResult<(LuResult<T>, LuResult<T>)>
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

    let (l_data, _) = extract_data(&result.l)?;
    let (u_data, _) = extract_data(&result.u)?;
    let p_vec = crate::forward_perm_from_permutation_matrix(&result.p, m, bc)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (da_data, _) = extract_data(tangent)?;

    let mut dl_data = vec![T::zero(); m * k * bc];
    let mut du_data = vec![T::zero(); k * n * bc];

    for b in 0..bc {
        let l_b = &l_data[b * m * k..(b + 1) * m * k];
        let u_b = &u_data[b * k * n..(b + 1) * k * n];
        let da_b = &da_data[b * m * n..(b + 1) * m * n];

        // Apply permutation: P dA (m×n)
        let mut pda = vec![T::zero(); m * n];
        let p_b = &p_vec[b * m..(b + 1) * m];
        for i in 0..m {
            for j in 0..n {
                pda[i + j * m] = da_b[p_b[i] + j * m];
            }
        }

        if m == n {
            let l_sq = l_b.to_vec();
            let u_sq = u_b.to_vec();
            let linv_pda = backend_solve_tri(ctx, &l_sq, &pda, k, k, false)?;
            let f_h = backend_solve_tri(
                ctx,
                &adjoint_transpose(&u_sq, k, k),
                &adjoint_transpose(&linv_pda, k, k),
                k,
                k,
                false,
            )?;
            let f = adjoint_transpose(&f_h, k, k);
            let lower_f = tril_strict(&f, k);
            let upper_f = triu(&f, k);

            let dl_b_vec = backend_mat_mul(ctx, &l_sq, k, k, &lower_f, k)?;
            let du_b_vec = backend_mat_mul(ctx, &upper_f, k, k, &u_sq, k)?;
            dl_data[b * m * k..(b + 1) * m * k].copy_from_slice(&dl_b_vec);
            du_data[b * k * n..(b + 1) * k * n].copy_from_slice(&du_b_vec);
        } else if m < n {
            let l_sq = l_b.to_vec();
            let u1 = u_b[..k * k].to_vec();
            let u2 = u_b[k * k..].to_vec();
            let pda1 = pda[..k * k].to_vec();
            let pda2 = pda[k * k..].to_vec();

            let linv_pda1 = backend_solve_tri(ctx, &l_sq, &pda1, k, k, false)?;
            let f_h = backend_solve_tri(
                ctx,
                &adjoint_transpose(&u1, k, k),
                &adjoint_transpose(&linv_pda1, k, k),
                k,
                k,
                false,
            )?;
            let f = adjoint_transpose(&f_h, k, k);
            let lower_f = tril_strict(&f, k);
            let upper_f = triu(&f, k);

            let dl_b_vec = backend_mat_mul(ctx, &l_sq, k, k, &lower_f, k)?;
            let du1 = backend_mat_mul(ctx, &upper_f, k, k, &u1, k)?;
            let du2 = if n > k {
                let linv_pda2 = backend_solve_tri(ctx, &l_sq, &pda2, k, n - k, false)?;
                let correction = backend_mat_mul(ctx, &lower_f, k, k, &u2, n - k)?;
                sub_vec(&linv_pda2, &correction)
            } else {
                Vec::new()
            };

            dl_data[b * m * k..(b + 1) * m * k].copy_from_slice(&dl_b_vec);
            du_data[b * k * n..b * k * n + k * k].copy_from_slice(&du1);
            if n > k {
                du_data[b * k * n + k * k..(b + 1) * k * n].copy_from_slice(&du2);
            }
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
            let u_sq = u_b.to_vec();

            let mut pda1 = vec![T::zero(); k * k];
            let mut pda2 = vec![T::zero(); (m - k) * k];
            for j in 0..k {
                for i in 0..k {
                    pda1[i + j * k] = pda[i + j * m];
                }
                for i in k..m {
                    pda2[(i - k) + j * (m - k)] = pda[i + j * m];
                }
            }

            let linv_pda1 = backend_solve_tri(ctx, &l1, &pda1, k, k, false)?;
            let f_h = backend_solve_tri(
                ctx,
                &adjoint_transpose(&u_sq, k, k),
                &adjoint_transpose(&linv_pda1, k, k),
                k,
                k,
                false,
            )?;
            let f = adjoint_transpose(&f_h, k, k);
            let lower_f = tril_strict(&f, k);
            let upper_f = triu(&f, k);

            let dl1 = backend_mat_mul(ctx, &l1, k, k, &lower_f, k)?;
            let du_b_vec = backend_mat_mul(ctx, &upper_f, k, k, &u_sq, k)?;
            let dl2 = if m > k {
                let pda2_uinv_h = backend_solve_tri(
                    ctx,
                    &adjoint_transpose(&u_sq, k, k),
                    &adjoint_transpose(&pda2, m - k, k),
                    k,
                    m - k,
                    false,
                )?;
                let pda2_uinv = adjoint_transpose(&pda2_uinv_h, k, m - k);
                let correction = backend_mat_mul(ctx, &l2, m - k, k, &upper_f, k)?;
                sub_vec(&pda2_uinv, &correction)
            } else {
                Vec::new()
            };

            for j in 0..k {
                for i in 0..k {
                    dl_data[b * m * k + i + j * m] = dl1[i + j * k];
                }
                for i in k..m {
                    dl_data[b * m * k + i + j * m] = dl2[(i - k) + j * (m - k)];
                }
            }
            du_data[b * k * n..(b + 1) * k * n].copy_from_slice(&du_b_vec);
        }
    }

    let l_dims = output_dims(&[m, k], batch_dims);
    let u_dims = output_dims(&[k, n], batch_dims);
    let dresult = LuResult {
        p: Tensor::zeros(
            result.p.dims(),
            result.p.logical_memory_space(),
            MemoryOrder::ColumnMajor,
        )
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
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
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col).unwrap();
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col).unwrap();
/// let (result, dresult) = eigen_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn eigen_frule<T, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(EigenResult<T, T::Real>, EigenResult<T, T::Real>)>
where
    T: KernelLinalgScalar + Conjugate,
    T::Real: KernelLinalgScalar<Real = T::Real> + num_traits::Float,
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
    let (da_data, _) = extract_data(tangent)?;

    let mut de_data = vec![T::Real::zero(); n * bc];
    let mut dv_data = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let v_b = &v_data[b * n * n..(b + 1) * n * n];
        let e_b = &e_data[b * n..(b + 1) * n];
        let da_b = &da_data[b * n * n..(b + 1) * n * n];

        // C = V^H dA V (n×n)
        let vh_da = backend_mat_mul(ctx, &adjoint_transpose(v_b, n, n), n, n, da_b, n)?;
        let c = backend_mat_mul(ctx, &vh_da, n, n, v_b, n)?;

        // dE = diag(C)
        for i in 0..n {
            de_data[b * n + i] = c[i + i * n].real_part();
        }

        // dV = V * (F ⊙ (C - diag(dE))) where F_ij = 1/(e_j - e_i) for i≠j.
        let mut fc = vec![T::zero(); n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let denom = e_b[j] - e_b[i];
                    let f_ij = T::Real::one()
                        / (denom
                            + eta
                                * if denom >= T::Real::zero() {
                                    T::Real::one()
                                } else {
                                    -T::Real::one()
                                });
                    fc[i + j * n] = T::from_real(f_ij) * c[i + j * n];
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
