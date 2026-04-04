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
pub fn solve_rrule<T, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
    cotangent: &Tensor<T>,
) -> AdResult<SolveGrad<T>>
where
    T: KernelLinalgScalar + tenferro_algebra::Conjugate,
    C: backend::TensorLinalgContextFor<T> + tenferro_prims::TensorResolveConjContextFor<T>,
    C::Backend: 'static,
{
    // Ax = b → G = A^{-H} dx̄, dB = G, dA = -G x^H
    let x = solve(ctx, a, b).map_err(to_ad_err)?;
    let (n, batch_dims) = validate_square(a).map_err(to_ad_err)?;
    let rhs = backend::tensor_helpers::validate_solve_rhs_shape(b, n, batch_dims, "solve_rrule")
        .map_err(to_ad_err)?;
    let nrhs = rhs.nrhs;
    let sr = rhs.structural_rank;

    // Promote cotangent and x to matrix form [n, nrhs, batch...]
    let dx_mat = rhs_to_matrix(cotangent, sr).map_err(to_ad_err)?;
    let x_mat = rhs_to_matrix(&x, sr).map_err(to_ad_err)?;

    // G = solve(A^H, cotangent)
    let a_h = matrix_adjoint_eager(ctx, a).map_err(to_ad_err)?;
    let g = solve(ctx, &a_h, &dx_mat).map_err(to_ad_err)?;

    // dB = G (convert back to original RHS shape)
    let grad_b = matrix_to_rhs(g.clone(), sr).map_err(to_ad_err)?;

    // dA = -G x^H  (use alpha=-1 in GEMM)
    let x_h = matrix_adjoint_eager(ctx, &x_mat).map_err(to_ad_err)?;
    let grad_a = prims_bridge::batched_gemm_alpha_tensors(ctx, &g, &x_h, n, nrhs, n, -T::one())
        .map_err(to_ad_err)?;

    Ok(SolveGrad {
        a: grad_a,
        b: grad_b,
    })
}

/// Reverse-mode AD rule for triangular solve (VJP / pullback).
///
/// Given `A x = b` with triangular `A` and cotangent `x̄`, computes `(Ā, b̄)`.
///
/// - `G = A^{-H} x̄` solved with conjugate-transposed triangular structure
/// - `b̄ = G`
/// - `Ā = proj(-G x^H)` where `proj = triu` for upper, `tril` for lower
pub fn solve_triangular_rrule<T, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
    cotangent: &Tensor<T>,
    upper: bool,
) -> AdResult<SolveGrad<T>>
where
    T: KernelLinalgScalar + tenferro_algebra::Conjugate,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorResolveConjContextFor<T>
        + tenferro_prims::TensorMetadataContextFor,
    C::Backend: 'static,
{
    let x = solve_triangular(ctx, a, b, upper).map_err(to_ad_err)?;
    let (n, _) = validate_square(a).map_err(to_ad_err)?;
    let rhs = backend::tensor_helpers::validate_solve_rhs_shape(
        b,
        n,
        &a.dims()[2..],
        "solve_triangular_rrule",
    )
    .map_err(to_ad_err)?;
    let nrhs = rhs.nrhs;
    let sr = rhs.structural_rank;

    // Promote cotangent and x to matrix form
    let dx_mat = rhs_to_matrix(cotangent, sr).map_err(to_ad_err)?;
    let x_mat = rhs_to_matrix(&x, sr).map_err(to_ad_err)?;

    // G = solve_triangular(A^H, dx, !upper)  (A^H flips upper/lower)
    let a_h = matrix_adjoint_eager(ctx, a).map_err(to_ad_err)?;
    let g = solve_triangular(ctx, &a_h, &dx_mat, !upper).map_err(to_ad_err)?;

    // dB = G
    let grad_b = matrix_to_rhs(g.clone(), sr).map_err(to_ad_err)?;

    // dA = proj(-G x^H)
    let x_h = matrix_adjoint_eager(ctx, &x_mat).map_err(to_ad_err)?;
    let neg_g_xh = prims_bridge::batched_gemm_alpha_tensors(ctx, &g, &x_h, n, nrhs, n, -T::one())
        .map_err(to_ad_err)?;
    let grad_a = if upper {
        tenferro_prims::tensor_ops::triu(ctx, &neg_g_xh, 0).map_err(to_ad_err)?
    } else {
        tenferro_prims::tensor_ops::tril(ctx, &neg_g_xh, 0).map_err(to_ad_err)?
    };

    Ok(SolveGrad {
        a: grad_a,
        b: grad_b,
    })
}
