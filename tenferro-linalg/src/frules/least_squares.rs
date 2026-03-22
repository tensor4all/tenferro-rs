use super::*;

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
pub fn lstsq_frule<
    T: KernelLinalgScalar<Real = T> + num_traits::Float + tenferro_algebra::Conjugate,
    C,
>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
    tangent_a: &Tensor<T>,
    tangent_b: &Tensor<T>,
) -> AdResult<(LstsqResult<T>, LstsqResult<T>)>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>
        + tenferro_prims::TensorSemiringContextFor<tenferro_algebra::Standard<T>>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Lstsq, "lstsq_frule")
        .map_err(to_ad_err)?;

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
pub fn cholesky_frule<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: KernelLinalgScalar,
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
