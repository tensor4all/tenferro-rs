use super::*;

fn rhs_output_dims(core_rows: usize, nrhs: usize, batch_dims: &[usize]) -> Vec<usize> {
    let core_dims = if nrhs == 1 {
        vec![core_rows]
    } else {
        vec![core_rows, nrhs]
    };
    output_dims(&core_dims, batch_dims)
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
/// let da = Tensor::<f64>::ones(&[3, 2], mem, col).unwrap();
/// let db = Tensor::<f64>::ones(&[3], mem, col).unwrap();
/// let (result, dresult) = lstsq_frule(&mut ctx, &a, &b, &da, &db).unwrap();
/// ```
pub fn lstsq_frule<
    T: KernelLinalgScalar<Real = T>
        + num_traits::Float
        + tenferro_algebra::Conjugate
        + crate::prims_bridge::ScaleTensorByRealSameShape<C>,
    C,
>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
    tangent_a: &Tensor<T>,
    tangent_b: &Tensor<T>,
) -> AdResult<(LstsqResult<T, T::Real>, LstsqResult<T, T::Real>)>
where
    T: KernelLinalgScalar,
    T::Real: LinalgScalar<Real = T::Real> + num_traits::Float + tenferro_tensor::KeepCountScalar,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorResolveConjContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>
        + tenferro_prims::TensorSemiringContextFor<tenferro_algebra::Standard<T>>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T::Real>>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Lstsq, "lstsq_frule")
        .map_err(to_ad_err)?;

    // dx = dA^+ * b + A^+ * db
    // d residual_summaries = 2 * sum(real((A x - b) * conj(dA x - db))), per RHS
    let result = lstsq(ctx, a, b)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (pinv_a, dpinv_a) = pinv_frule(ctx, a, tangent_a, None)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(a)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (x_data, _) = extract_data(&result.solution)?;
    let (da_data, _) = extract_data(tangent_a)?;
    let (ap_data, _) = extract_data(&pinv_a)?;
    let (dap_data, _) = extract_data(&dpinv_a)?;
    let (b_data, _) = extract_data(b)?;
    let (db_data, _) = extract_data(tangent_b)?;
    let nrhs = if b.ndim() == 1 + batch_dims.len() {
        1
    } else {
        b.dims()[1]
    };
    let rhs_is_vector = nrhs == 1 && b.ndim() == 1 + batch_dims.len();
    let aux = lstsq_aux(ctx, a)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let summarize_residuals = m > n
        && crate::primal::lstsq_has_full_rank(&aux.rank, n)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let two = scalar_from::<T::Real>(2.0)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;

    let mut dx_data = vec![T::zero(); n * nrhs * bc];
    let mut dresidual_data = vec![T::Real::zero(); bc * nrhs];
    let (a_data, _) = extract_data(a)?;

    for batch in 0..bc {
        let x_b = &x_data[batch * n * nrhs..(batch + 1) * n * nrhs];
        let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
        let ap_b = &ap_data[batch * n * m..(batch + 1) * n * m];
        let dap_b = &dap_data[batch * n * m..(batch + 1) * n * m];
        let b_b = &b_data[batch * m * nrhs..(batch + 1) * m * nrhs];
        let da_b = &da_data[batch * m * n..(batch + 1) * m * n];
        let db_b = &db_data[batch * m * nrhs..(batch + 1) * m * nrhs];

        let dpinv_b = backend_mat_mul(ctx, dap_b, n, m, b_b, nrhs)?;
        let pinv_db = backend_mat_mul(ctx, ap_b, n, m, db_b, nrhs)?;
        let dx_b_vec = add_vec(&dpinv_b, &pinv_db);
        dx_data[batch * n * nrhs..(batch + 1) * n * nrhs].copy_from_slice(&dx_b_vec);

        if summarize_residuals {
            let ax = backend_mat_mul(ctx, a_b, m, n, x_b, nrhs)?;
            let da_x = backend_mat_mul(ctx, da_b, m, n, x_b, nrhs)?;
            for col in 0..nrhs {
                let mut acc = T::Real::zero();
                for row in 0..m {
                    let idx = row + col * m;
                    let residual = ax[idx] - b_b[idx];
                    let dresidual = da_x[idx] - db_b[idx];
                    acc = acc + residual * dresidual;
                }
                dresidual_data[batch * nrhs + col] = two * acc;
            }
        }
    }

    let x_dims = rhs_output_dims(n, nrhs, batch_dims);
    let dx = tensor_from_data(dx_data, &x_dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let dresiduals = if summarize_residuals {
        let dims = crate::primal::residual_summary_output_dims(batch_dims, nrhs, rhs_is_vector);
        tensor_from_data(dresidual_data, &dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?
    } else {
        crate::primal::empty_residual_summary::<T::Real>(a.logical_memory_space())
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?
    };
    let dresult = LstsqResult {
        solution: dx,
        residuals: dresiduals,
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
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col).unwrap();
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col).unwrap();
/// let (l, dl) = cholesky_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn cholesky_frule<T: KernelLinalgScalar + tenferro_algebra::Conjugate, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: KernelLinalgScalar + tenferro_algebra::Conjugate,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    // dL = L phi(L^{-1} dA L^{-H})
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
        // (L^{-1} dA) L^{-H}: solve L x = (L^{-1} dA)^H, then adjoint back
        let linv_da_linvh_h =
            backend_solve_tri(ctx, l_b, &adjoint_transpose(&linv_da, n, n), n, n, false)?;
        let inner = adjoint_transpose(&linv_da_linvh_h, n, n);

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
