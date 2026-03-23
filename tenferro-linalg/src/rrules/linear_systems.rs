use super::*;

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
/// let a = Tensor::<f64>::eye(3, mem, col).unwrap();
/// let b = Tensor::<f64>::ones(&[3], mem, col).unwrap();
/// let cotangent = Tensor::<f64>::ones(&[3], mem, col).unwrap();
/// let grad = solve_rrule(&mut ctx, &a, &b, &cotangent).unwrap();
/// ```
pub fn solve_rrule<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
    cotangent: &Tensor<T>,
) -> AdResult<SolveGrad<T>>
where
    T: KernelLinalgScalar,
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
pub fn solve_triangular_rrule<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
    cotangent: &Tensor<T>,
    upper: bool,
) -> AdResult<SolveGrad<T>>
where
    T: KernelLinalgScalar,
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
/// let a = Tensor::<f64>::eye(3, mem, col).unwrap();
/// let cotangent = Tensor::<f64>::ones(&[3, 3], mem, col).unwrap();
/// let grad_a = inv_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn inv_rrule<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &Tensor<T>,
) -> AdResult<Tensor<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Inv, "inv_rrule")
        .map_err(to_ad_err)?;

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
/// let a = Tensor::<f64>::eye(3, mem, col).unwrap();
/// let cotangent = Tensor::<f64>::ones(&[], mem, col).unwrap();
/// let grad_a = det_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn det_rrule<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &Tensor<T>,
) -> AdResult<Tensor<T>>
where
    T: KernelLinalgScalar + crate::prims_bridge::ScaleTensorByRealSameShape<C>,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>
        + tenferro_prims::TensorMetadataContextFor,
    C::Backend: 'static,
    C::MetadataBackend: tenferro_prims::TensorMetadataPrims<Context = C>,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>>::ScalarBackend:
        tenferro_prims::TensorMetadataCastPrims<T, Context = C>,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Det, "det_rrule")
        .map_err(to_ad_err)?;

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
/// let a = Tensor::<f64>::eye(3, mem, col).unwrap();
/// let cotangent = SlogdetCotangent {
///     logabsdet: Some(Tensor::ones(&[], mem, col).unwrap()),
/// };
/// let grad_a = slogdet_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn slogdet_rrule<
    T: KernelLinalgScalar<Real = T> + num_traits::Float + crate::SlogdetDispatch<C>,
    C,
>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &SlogdetCotangent<T>,
) -> AdResult<Tensor<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Slogdet, "slogdet_rrule")
        .map_err(to_ad_err)?;

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
