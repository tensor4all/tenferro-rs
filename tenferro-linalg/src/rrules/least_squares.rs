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
/// let grad = lstsq_rrule(&mut ctx, &a, &b, &dx).unwrap();
/// // grad.a: cotangent for A, grad.b: cotangent for b
/// ```
pub fn lstsq_rrule<
    T: KernelLinalgScalar<Real = T> + num_traits::Float + tenferro_algebra::Conjugate,
    C,
>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
    cotangent_x: &Tensor<T>,
) -> AdResult<LstsqGrad<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>
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
    let nrhs = if b.ndim() == 1 + batch_dims.len() {
        1
    } else {
        b.dims()[1]
    };

    let mut grad_a_data = vec![T::zero(); m * n * bc];
    let mut grad_b_data = vec![T::zero(); m * nrhs * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
        let x_b = &x_data[batch * n * nrhs..(batch + 1) * n * nrhs];
        let r_b = &r_data[batch * m * nrhs..(batch + 1) * m * nrhs];
        let dx_b = &dx_data[batch * n * nrhs..(batch + 1) * n * nrhs];

        let (q_d, r_d) = backend_qr(ctx, a_b, m, n)?;
        let y = backend_solve_tri(ctx, &transpose(&r_d, n, n), dx_b, n, nrhs, false)?;
        let z = backend_solve_tri(ctx, &r_d, &y, n, nrhs, true)?;
        let grad_b = backend_mat_mul(ctx, &q_d, m, n, &y, nrhs)?;
        let residual_term = backend_mat_mul(ctx, r_b, m, nrhs, &transpose(&z, n, nrhs), n)?;
        let x_term = backend_mat_mul(ctx, &grad_b, m, nrhs, &transpose(x_b, n, nrhs), n)?;

        for i in 0..m * n {
            grad_a_data[batch * m * n + i] = residual_term[i] - x_term[i];
        }
        grad_b_data[batch * m * nrhs..(batch + 1) * m * nrhs].copy_from_slice(&grad_b);
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
