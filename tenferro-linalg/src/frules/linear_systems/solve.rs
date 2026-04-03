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
        tenferro_prims::tensor_ops::triu(ctx, tangent_a, 0).map_err(to_ad_err)?
    } else {
        tenferro_prims::tensor_ops::tril(ctx, tangent_a, 0).map_err(to_ad_err)?
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
