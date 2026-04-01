use super::*;

fn rhs_output_dims(core_rows: usize, nrhs: usize, batch_dims: &[usize]) -> Vec<usize> {
    let core_dims = if nrhs == 1 {
        vec![core_rows]
    } else {
        vec![core_rows, nrhs]
    };
    output_dims(&core_dims, batch_dims)
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
/// let dx = Tensor::<f64>::ones(&[2], mem, col).unwrap();
/// let grad = lstsq_rrule(&mut ctx, &a, &b, Some(&dx), None).unwrap();
/// // grad.a: cotangent for A, grad.b: cotangent for b
/// ```
pub fn lstsq_rrule<
    T: KernelLinalgScalar<Real = T>
        + num_traits::Float
        + tenferro_algebra::Conjugate
        + crate::prims_bridge::ScaleTensorByRealSameShape<C>
        + tenferro_tensor::KeepCountScalar,
    C,
>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
    cotangent_solution: Option<&Tensor<T>>,
    cotangent_residuals: Option<&Tensor<T::Real>>,
) -> AdResult<LstsqGrad<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorResolveConjContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>
        + tenferro_prims::TensorSemiringContextFor<tenferro_algebra::Standard<T>>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Lstsq, "lstsq_rrule")
        .map_err(to_ad_err)?;

    let result = lstsq(ctx, a, b)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(a)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);
    let nrhs = if b.ndim() == 1 + batch_dims.len() {
        1
    } else {
        b.dims()[1]
    };
    let rhs_is_vector = nrhs == 1 && b.ndim() == 1 + batch_dims.len();

    if let Some(cotangent_solution) = cotangent_solution {
        if cotangent_solution.dims() != result.solution.dims() {
            return Err(to_ad_err(Error::InvalidArgument(format!(
                "lstsq_rrule solution cotangent shape mismatch: expected {:?}, got {:?}",
                result.solution.dims(),
                cotangent_solution.dims()
            ))));
        }
    }
    if let Some(cotangent_residuals) = cotangent_residuals {
        if cotangent_residuals.dims() != result.residuals.dims() {
            return Err(to_ad_err(Error::InvalidArgument(format!(
                "lstsq_rrule residual cotangent shape mismatch: expected {:?}, got {:?}",
                result.residuals.dims(),
                cotangent_residuals.dims()
            ))));
        }
    }
    if cotangent_solution.is_none() && cotangent_residuals.is_none() {
        let a_dims = output_dims(&[m, n], batch_dims);
        let b_dims = rhs_output_dims(m, nrhs, batch_dims);
        return Ok(LstsqGrad {
            a: Tensor::<T>::zeros(
                &a_dims,
                a.logical_memory_space(),
                tenferro_tensor::MemoryOrder::ColumnMajor,
            )
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
            b: Tensor::<T>::zeros(
                &b_dims,
                b.logical_memory_space(),
                tenferro_tensor::MemoryOrder::ColumnMajor,
            )
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
        });
    }

    let (a_data, _) = extract_data(a)?;
    let (b_data, _) = extract_data(b)?;
    let (x_data, _) = extract_data(&result.solution)?;
    let dx_data = cotangent_solution.map(|tensor| extract_data(tensor).map(|(data, _)| data));
    let dx_data = dx_data.transpose()?;
    let dresidual_data =
        cotangent_residuals.map(|tensor| extract_data(tensor).map(|(data, _)| data));
    let dresidual_data = dresidual_data.transpose()?;
    let two = scalar_from::<T>(2.0)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let mut grad_a_data = vec![T::zero(); m * n * bc];
    let mut grad_b_data = vec![T::zero(); m * nrhs * bc];

    if let Some(dx_data) = dx_data.as_ref() {
        let pinv_a = pinv(ctx, a, None)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
        let (ap_data, _) = extract_data(&pinv_a)?;
        let mut cotangent_pinv_data = vec![T::zero(); n * m * bc];

        for batch in 0..bc {
            let ap_b = &ap_data[batch * n * m..(batch + 1) * n * m];
            let b_b = &b_data[batch * m * nrhs..(batch + 1) * m * nrhs];
            let dx_b = &dx_data[batch * n * nrhs..(batch + 1) * n * nrhs];
            let cotangent_pinv_b =
                backend_mat_mul(ctx, dx_b, n, nrhs, &transpose(b_b, m, nrhs), m)?;
            cotangent_pinv_data[batch * n * m..(batch + 1) * n * m]
                .copy_from_slice(&cotangent_pinv_b);

            let grad_b_solution =
                backend_mat_mul(ctx, &adjoint_transpose(ap_b, n, m), m, n, dx_b, nrhs)?;
            for i in 0..m * nrhs {
                grad_b_data[batch * m * nrhs + i] =
                    grad_b_data[batch * m * nrhs + i] + grad_b_solution[i];
            }
        }

        let cotangent_pinv =
            tensor_from_data(cotangent_pinv_data, &output_dims(&[n, m], batch_dims))
                .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
        let grad_a_solution = pinv_rrule(ctx, a, &cotangent_pinv, None)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
        let (grad_a_solution_data, _) = extract_data(&grad_a_solution)?;
        for (slot, value) in grad_a_data.iter_mut().zip(grad_a_solution_data.into_iter()) {
            *slot = *slot + value;
        }
    }

    for batch in 0..bc {
        let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
        let x_b = &x_data[batch * n * nrhs..(batch + 1) * n * nrhs];
        let b_b = &b_data[batch * m * nrhs..(batch + 1) * m * nrhs];

        if let Some(dresidual_data) = dresidual_data.as_ref().filter(|data| !data.is_empty()) {
            let ax = backend_mat_mul(ctx, a_b, m, n, x_b, nrhs)?;
            for col in 0..nrhs {
                let weight = if rhs_is_vector {
                    dresidual_data[batch]
                } else {
                    dresidual_data[batch * nrhs + col]
                };
                for row in 0..m {
                    let rhs_idx = row + col * m;
                    let residual = ax[rhs_idx] - b_b[rhs_idx];
                    grad_b_data[batch * m * nrhs + rhs_idx] =
                        grad_b_data[batch * m * nrhs + rhs_idx] - two * weight * residual;
                    for k in 0..n {
                        let a_idx = batch * m * n + row + k * m;
                        let x_idx = if nrhs == 1 { k } else { k + col * n };
                        grad_a_data[a_idx] =
                            grad_a_data[a_idx] + two * weight * residual * x_b[x_idx];
                    }
                }
            }
        }
    }

    let a_dims = output_dims(&[m, n], batch_dims);
    let b_dims = rhs_output_dims(m, nrhs, batch_dims);
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
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col).unwrap();
/// let cotangent = Tensor::<f64>::ones(&[3, 3], mem, col).unwrap();
/// let grad_a = cholesky_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn cholesky_rrule<T: KernelLinalgScalar + tenferro_algebra::Conjugate, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &Tensor<T>,
) -> AdResult<Tensor<T>>
where
    T: KernelLinalgScalar + tenferro_algebra::Conjugate,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    // A = L L^H, dA = L^{-H} phi*(tril(L^H dL)) L^{-1}
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

        // S = tril(L^H dL)
        let lt_dl = backend_mat_mul(ctx, &adjoint_transpose(l_b, n, n), n, n, dl_b, n)?;
        let s = tril(&lt_dl, n);

        // Apply phi*: symmetrize S → (S + S^H - diag(S)) / 2
        let s_sym = phi_star(&s, n)?;

        // Solve L^H x = S_sym → x = L^{-H} S_sym
        let x = backend_solve_tri(ctx, &adjoint_transpose(l_b, n, n), &s_sym, n, n, true)?;

        // Solve x L = result → result^H = L^{-H} x^H
        let xh = adjoint_transpose(&x, n, n);
        let result_h = backend_solve_tri(ctx, &adjoint_transpose(l_b, n, n), &xh, n, n, true)?;
        let da_b = adjoint_transpose(&result_h, n, n);

        grad_a[b * n * n..(b + 1) * n * n].copy_from_slice(&da_b);
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(grad_a, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))
}
