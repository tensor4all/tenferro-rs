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
    C: backend::TensorLinalgContextFor<T> + tenferro_prims::TensorResolveConjContextFor<T>,
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
        neg_g_xh.triu(0)
    } else {
        neg_g_xh.tril(0)
    };

    Ok(SolveGrad {
        a: grad_a,
        b: grad_b,
    })
}

/// Reverse-mode AD rule for matrix inverse (VJP / pullback).
///
/// `Ā = -A⁻ᵀ · cotangent · A⁻ᵀ`.
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

    // dA = -B^T dB B^T where B = A^{-1}  (real-only, so transpose = adjoint)
    let b_inv = inv(ctx, tensor).map_err(to_ad_err)?;
    let (n, _) = validate_square(tensor).map_err(to_ad_err)?;

    let bt = matrix_transpose(&b_inv).map_err(to_ad_err)?;
    let bt_db = prims_bridge::batched_gemm_with_semiring_tensors(ctx, &bt, cotangent, n, n, n)
        .map_err(to_ad_err)?;
    let grad_a = prims_bridge::batched_gemm_alpha_tensors(ctx, &bt_db, &bt, n, n, n, -T::one())
        .map_err(to_ad_err)?;

    Ok(grad_a)
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
    let det_val = det(ctx, tensor).map_err(to_ad_err)?;
    let a_inv = inv(ctx, tensor).map_err(to_ad_err)?;
    let a_inv_t = matrix_transpose(&a_inv).map_err(to_ad_err)?;

    // scale = cotangent * det(A), shape [batch...]
    let scale = prims_bridge::scalar_binary_same_shape(
        ctx,
        cotangent,
        &det_val,
        tenferro_prims::ScalarBinaryOp::Mul,
    )
    .map_err(to_ad_err)?;

    // broadcast scale [batch...] → [1, 1, batch...] → [n, n, batch...]
    let scale_expanded = scale
        .unsqueeze(0)
        .map_err(to_ad_err)?
        .unsqueeze(0)
        .map_err(to_ad_err)?
        .broadcast(a_inv_t.dims())
        .map_err(to_ad_err)?;
    let grad_a = prims_bridge::scalar_binary_same_shape(
        ctx,
        &scale_expanded,
        &a_inv_t,
        tenferro_prims::ScalarBinaryOp::Mul,
    )
    .map_err(to_ad_err)?;

    Ok(grad_a)
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
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Slogdet, "slogdet_rrule")
        .map_err(to_ad_err)?;

    // dA = d_logabsdet * A^{-T}
    let (n, batch_dims) = validate_square(tensor).map_err(to_ad_err)?;

    if let Some(ref dlog) = cotangent.logabsdet {
        let a_inv = inv(ctx, tensor).map_err(to_ad_err)?;
        let a_inv_t = matrix_transpose(&a_inv).map_err(to_ad_err)?;

        // broadcast dlog [batch...] → [1, 1, batch...] → [n, n, batch...]
        let dlog_expanded = dlog
            .unsqueeze(0)
            .map_err(to_ad_err)?
            .unsqueeze(0)
            .map_err(to_ad_err)?
            .broadcast(a_inv_t.dims())
            .map_err(to_ad_err)?;
        let grad_a = prims_bridge::scalar_binary_same_shape(
            ctx,
            &dlog_expanded,
            &a_inv_t,
            tenferro_prims::ScalarBinaryOp::Mul,
        )
        .map_err(to_ad_err)?;
        Ok(grad_a)
    } else {
        let dims = output_dims(&[n, n], batch_dims);
        Tensor::zeros(
            &dims,
            tensor.logical_memory_space(),
            MemoryOrder::ColumnMajor,
        )
        .map_err(to_ad_err)
    }
}
