use super::*;

/// Forward-mode AD rule for linear solve (JVP / pushforward).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::solve_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col).unwrap();
/// let b = Tensor::<f64>::ones(&[3], mem, col).unwrap();
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col).unwrap();
/// let db = Tensor::<f64>::ones(&[3], mem, col).unwrap();
/// let (x, dx) = solve_frule(&mut ctx, &a, &b, &da, &db).unwrap();
/// ```
pub fn solve_frule<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
    tangent_a: &Tensor<T>,
    tangent_b: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>,
    C::Backend: 'static,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>>::ScalarBackend:
        'static + tenferro_prims::TensorAnalyticPrims<tenferro_algebra::Standard<T>, Context = C>,
{
    // dx = A^{-1} (db - dA x)
    let x = solve(ctx, a, b)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(a)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);
    let rhs = backend::tensor_helpers::validate_solve_rhs_shape(b, n, batch_dims, "solve_frule")
        .map_err(to_ad_err)?;
    let nrhs = rhs.nrhs;

    let (a_data, _) = extract_data(a)?;
    let (x_data, _) = extract_data(&x)?;
    let (da_data, _) = extract_data(tangent_a)?;
    let (db_data, _) = extract_data(tangent_b)?;

    let mut dx_data = vec![T::zero(); n * nrhs * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
        let x_b = &x_data[batch * n * nrhs..(batch + 1) * n * nrhs];
        let da_b = &da_data[batch * n * n..(batch + 1) * n * n];
        let db_b = &db_data[batch * n * nrhs..(batch + 1) * n * nrhs];

        // dA x (n×nrhs)
        let da_x = backend_mat_mul(ctx, da_b, n, n, x_b, nrhs)?;
        // db - dA x
        let rhs = sub_vec(db_b, &da_x);
        // A^{-1} (db - dA x)
        let dx_b_vec = backend_solve(ctx, a_b, &rhs, n, nrhs)?;
        dx_data[batch * n * nrhs..(batch + 1) * n * nrhs].copy_from_slice(&dx_b_vec);
    }

    let dims = rhs.output_dims;
    let dx = tensor_from_data(dx_data, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    Ok((x, dx))
}

/// Forward-mode AD rule for triangular solve (JVP / pushforward).
///
/// Computes:
/// - `x = solve_triangular(a, b, upper)`
/// - `dx = solve_triangular(a, db - proj(dA) * x, upper)`
///
/// where `proj(dA)` keeps only the active triangular part
/// (`triu` when `upper=true`, `tril` when `upper=false`).
pub fn solve_triangular_frule<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
    tangent_a: &Tensor<T>,
    tangent_b: &Tensor<T>,
    upper: bool,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>
        + tenferro_prims::TensorMetadataContextFor,
    C::Backend: 'static,
    C::MetadataBackend: tenferro_prims::TensorMetadataPrims<Context = C>,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>>::ScalarBackend:
        'static
            + tenferro_prims::TensorAnalyticPrims<tenferro_algebra::Standard<T>, Context = C>
            + tenferro_prims::TensorMetadataCastPrims<T, Context = C>,
{
    if tangent_a.dims() != a.dims() {
        return Err(chainrules_core::AutodiffError::InvalidArgument(format!(
            "solve_triangular_frule: tangent_a shape mismatch: expected {:?}, got {:?}",
            a.dims(),
            tangent_a.dims()
        )));
    }
    if tangent_b.dims() != b.dims() {
        return Err(chainrules_core::AutodiffError::InvalidArgument(format!(
            "solve_triangular_frule: tangent_b shape mismatch: expected {:?}, got {:?}",
            b.dims(),
            tangent_b.dims()
        )));
    }

    // dX = A^{-1} (dB - proj(dA) X), with projection to the triangular tangent space.
    let x = solve_triangular(ctx, a, b, upper)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;

    let (n, batch_dims) = validate_square(a)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);
    let rhs = backend::tensor_helpers::validate_solve_rhs_shape(
        b,
        n,
        batch_dims,
        "solve_triangular_frule",
    )
    .map_err(to_ad_err)?;
    let nrhs = rhs.nrhs;

    let (a_data, _) = extract_data(a)?;
    let (x_data, _) = extract_data(&x)?;
    let (da_data, _) = extract_data(tangent_a)?;
    let (db_data, _) = extract_data(tangent_b)?;

    let mut dx_data = vec![T::zero(); n * nrhs * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
        let x_b = &x_data[batch * n * nrhs..(batch + 1) * n * nrhs];
        let da_b = &da_data[batch * n * n..(batch + 1) * n * n];
        let db_b = &db_data[batch * n * nrhs..(batch + 1) * n * nrhs];

        // Project dA onto the same triangular structure as A.
        let da_proj = if upper { triu(da_b, n) } else { tril(da_b, n) };

        // dA * x, treating x as n x nrhs in column-major layout.
        let da_x = prims_bridge::batched_gemm_with_semiring_context(ctx, &da_proj, n, n, x_b, nrhs)
            .map_err(to_ad_err)?;

        // RHS tangent: dB - dA * x
        let rhs = sub_vec(db_b, &da_x);

        // dX from triangular solve with the same structure.
        let dx_b = backend::slice_bridge::solve_triangular_vec(ctx, a_b, &rhs, n, nrhs, upper)
            .map_err(to_ad_err)?;

        dx_data[batch * n * nrhs..(batch + 1) * n * nrhs].copy_from_slice(&dx_b);
    }

    let dims = rhs.output_dims;
    let dx = tensor_from_data(dx_data, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    Ok((x, dx))
}

/// Forward-mode AD rule for matrix inverse (JVP / pushforward).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::inv_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col).unwrap();
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col).unwrap();
/// let (a_inv, da_inv) = inv_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn inv_frule<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>,
    C::Backend: 'static,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>>::ScalarBackend:
        'static + tenferro_prims::TensorAnalyticPrims<tenferro_algebra::Standard<T>, Context = C>,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Inv, "inv_frule")
        .map_err(to_ad_err)?;

    // dB = -B dA B where B = A^{-1}
    let b_inv = inv(ctx, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (binv_data, _) = extract_data(&b_inv)?;
    let (da_data, _) = extract_data(tangent)?;

    let mut db_data = vec![T::zero(); n * n * bc];

    for batch in 0..bc {
        let b_b = &binv_data[batch * n * n..(batch + 1) * n * n];
        let da_b = &da_data[batch * n * n..(batch + 1) * n * n];

        let b_da = backend_mat_mul(ctx, b_b, n, n, da_b, n)?;
        let b_da_b = backend_mat_mul(ctx, &b_da, n, n, b_b, n)?;
        let neg = scale_vec(&b_da_b, -T::one());
        db_data[batch * n * n..(batch + 1) * n * n].copy_from_slice(&neg);
    }

    let dims = output_dims(&[n, n], batch_dims);
    let db = tensor_from_data(db_data, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    Ok((b_inv, db))
}

/// Forward-mode AD rule for determinant (JVP / pushforward).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::det_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col).unwrap();
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col).unwrap();
/// let (d, dd) = det_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn det_frule<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(Tensor<T>, Tensor<T>)>
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
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Det, "det_frule")
        .map_err(to_ad_err)?;

    // d(det) = det(A) * tr(A^{-1} dA)
    let d = det(ctx, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor)?;
    let (d_data, _) = extract_data(&d)?;
    let (da_data, _) = extract_data(tangent)?;

    let mut dd_data = vec![T::zero(); bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
        let da_b = &da_data[batch * n * n..(batch + 1) * n * n];

        let a_inv = backend_solve(ctx, a_b, &eye::<T>(n), n, n)?;
        let a_inv_da = backend_mat_mul(ctx, &a_inv, n, n, da_b, n)?;
        let mut trace = T::zero();
        for i in 0..n {
            trace = trace + a_inv_da[i + i * n];
        }
        dd_data[batch] = d_data[batch] * trace;
    }

    let dims = output_dims(&[], batch_dims);
    let dd = tensor_from_data(dd_data, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    Ok((d, dd))
}

/// Forward-mode AD rule for slogdet (JVP / pushforward).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::slogdet_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col).unwrap();
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col).unwrap();
/// let (result, dresult) = slogdet_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn slogdet_frule<
    T: KernelLinalgScalar<Real = T> + num_traits::Float + crate::SlogdetDispatch<C>,
    C,
>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(SlogdetResult<T>, SlogdetResult<T>)>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>,
    C::Backend: 'static,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>>::ScalarBackend:
        'static + tenferro_prims::TensorAnalyticPrims<tenferro_algebra::Standard<T>, Context = C>,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Slogdet, "slogdet_frule")
        .map_err(to_ad_err)?;

    // d(logabsdet) = Re(tr(A^{-1} dA)), d(sign) = 0 (for real)
    let result = slogdet(ctx, tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (n, batch_dims) = validate_square(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor)?;
    let (da_data, _) = extract_data(tangent)?;

    let mut dlog_data = vec![T::zero(); bc];
    let dsign_data = vec![T::zero(); bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * n * n..(batch + 1) * n * n];
        let da_b = &da_data[batch * n * n..(batch + 1) * n * n];

        let a_inv = backend_solve(ctx, a_b, &eye::<T>(n), n, n)?;
        let a_inv_da = backend_mat_mul(ctx, &a_inv, n, n, da_b, n)?;
        let mut trace = T::zero();
        for i in 0..n {
            trace = trace + a_inv_da[i + i * n];
        }
        dlog_data[batch] = trace;
    }

    let dims = output_dims(&[], batch_dims);
    let dresult = SlogdetResult {
        sign: tensor_from_data(dsign_data, &dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
        logabsdet: tensor_from_data(dlog_data, &dims)
            .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?,
    };
    Ok((result, dresult))
}
