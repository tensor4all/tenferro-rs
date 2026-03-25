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
    let x = solve(ctx, a, b).map_err(to_ad_err)?;
    let (n, batch_dims) = validate_square(a).map_err(to_ad_err)?;
    let rhs = backend::tensor_helpers::validate_solve_rhs_shape(b, n, batch_dims, "solve_frule")
        .map_err(to_ad_err)?;
    let nrhs = rhs.nrhs;
    let sr = rhs.structural_rank;

    // Promote x and tangent_b to matrix form [n, nrhs, batch...]
    let x_mat = rhs_to_matrix(&x, sr).map_err(to_ad_err)?;
    let db_mat = rhs_to_matrix(tangent_b, sr).map_err(to_ad_err)?;

    // dA @ x  (n×n @ n×nrhs = n×nrhs)
    let da_x = prims_bridge::batched_gemm_with_semiring_tensors(ctx, tangent_a, &x_mat, n, n, nrhs)
        .map_err(to_ad_err)?;

    // db - dA @ x
    let rhs_tangent = prims_bridge::scalar_binary_same_shape(
        ctx,
        &db_mat,
        &da_x,
        tenferro_prims::ScalarBinaryOp::Sub,
    )
    .map_err(to_ad_err)?;

    // dx = A^{-1} (db - dA x)
    let dx_mat = solve(ctx, a, &rhs_tangent).map_err(to_ad_err)?;
    let dx = matrix_to_rhs(dx_mat, sr).map_err(to_ad_err)?;

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
    let x = solve_triangular(ctx, a, b, upper).map_err(to_ad_err)?;
    let (n, _) = validate_square(a).map_err(to_ad_err)?;
    let rhs = backend::tensor_helpers::validate_solve_rhs_shape(
        b,
        n,
        &a.dims()[2..],
        "solve_triangular_frule",
    )
    .map_err(to_ad_err)?;
    let nrhs = rhs.nrhs;
    let sr = rhs.structural_rank;

    let x_mat = rhs_to_matrix(&x, sr).map_err(to_ad_err)?;
    let db_mat = rhs_to_matrix(tangent_b, sr).map_err(to_ad_err)?;

    // Project dA onto the same triangular structure as A.
    let da_proj = if upper {
        tangent_a.triu(0)
    } else {
        tangent_a.tril(0)
    };

    // proj(dA) @ x
    let da_x = prims_bridge::batched_gemm_with_semiring_tensors(ctx, &da_proj, &x_mat, n, n, nrhs)
        .map_err(to_ad_err)?;

    // dB - proj(dA) @ x
    let rhs_tangent = prims_bridge::scalar_binary_same_shape(
        ctx,
        &db_mat,
        &da_x,
        tenferro_prims::ScalarBinaryOp::Sub,
    )
    .map_err(to_ad_err)?;

    // dX = solve_triangular(A, rhs_tangent, upper)
    let dx_mat = solve_triangular(ctx, a, &rhs_tangent, upper).map_err(to_ad_err)?;
    let dx = matrix_to_rhs(dx_mat, sr).map_err(to_ad_err)?;

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
    let b_inv = inv(ctx, tensor).map_err(to_ad_err)?;
    let (n, _) = validate_square(tensor).map_err(to_ad_err)?;

    let b_da = prims_bridge::batched_gemm_with_semiring_tensors(ctx, &b_inv, tangent, n, n, n)
        .map_err(to_ad_err)?;
    let db = prims_bridge::batched_gemm_alpha_tensors(ctx, &b_da, &b_inv, n, n, n, -T::one())
        .map_err(to_ad_err)?;

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
    let d = det(ctx, tensor).map_err(to_ad_err)?;
    let (n, _) = validate_square(tensor).map_err(to_ad_err)?;

    let a_inv = inv(ctx, tensor).map_err(to_ad_err)?;
    let a_inv_da = prims_bridge::batched_gemm_with_semiring_tensors(ctx, &a_inv, tangent, n, n, n)
        .map_err(to_ad_err)?;
    let trace = trace_tensor(ctx, &a_inv_da).map_err(to_ad_err)?;

    // dd = det(A) * trace
    let dd = prims_bridge::scalar_binary_same_shape(
        ctx,
        &d,
        &trace,
        tenferro_prims::ScalarBinaryOp::Mul,
    )
    .map_err(to_ad_err)?;

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
    let result = slogdet(ctx, tensor).map_err(to_ad_err)?;
    let (n, batch_dims) = validate_square(tensor).map_err(to_ad_err)?;

    let a_inv = inv(ctx, tensor).map_err(to_ad_err)?;
    let a_inv_da = prims_bridge::batched_gemm_with_semiring_tensors(ctx, &a_inv, tangent, n, n, n)
        .map_err(to_ad_err)?;
    let dlog = trace_tensor(ctx, &a_inv_da).map_err(to_ad_err)?;

    let dsign_dims = output_dims(&[], batch_dims);
    let dsign = Tensor::zeros(
        &dsign_dims,
        tensor.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    )
    .map_err(to_ad_err)?;

    let dresult = SlogdetResult {
        sign: dsign,
        logabsdet: dlog,
    };
    Ok((result, dresult))
}
