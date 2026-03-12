use super::*;

/// Reverse-mode AD rule for general eigendecomposition (VJP / pullback).
///
/// Given eigendecomposition `A V = V diag(lambda)`, computes the gradient
/// of the input `A` from complex-valued cotangents for eigenvalues and
/// eigenvectors using the Mike Giles formulas.
///
/// The cotangent uses [`EigCotangent`] with complex-valued tensors
/// because `eig()` returns complex output even for real inputs.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{eig_rrule, EigCotangent};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
/// use num_complex::Complex64;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col);
/// let cotangent = EigCotangent::<f64> {
///     values: None,
///     vectors: None,
/// };
/// let grad_a = eig_rrule(&mut ctx, &a, &cotangent).unwrap();
/// ```
pub fn eig_rrule<
    T: KernelLinalgScalar<Real = T, Complex = num_complex::Complex<T>> + num_traits::Float,
    C,
>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &EigCotangent<T>,
) -> AdResult<Tensor<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    let (n, batch_dims) = validate_square(tensor).map_err(to_ad_err)?;
    let bc = batch_count(batch_dims);

    // Compute eigendecomposition
    let eig_result = eig(ctx, tensor).map_err(to_ad_err)?;
    let val_data = extract_data_scalar(&eig_result.values)?;
    let vec_data = extract_data_scalar(&eig_result.vectors)?;

    let zero_c = Cx::new(T::zero(), T::zero());
    let one_c = Cx::new(T::one(), T::zero());

    let mut grad_data = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let lambda = &val_data[b * n..(b + 1) * n];
        let v = &vec_data[b * n * n..(b + 1) * n * n];

        // Compute F matrix: F[i,j] = 1/(lambda_j - lambda_i) for i != j, 0 on diagonal
        let mut f_mat = vec![zero_c; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let diff = lambda[j] - lambda[i];
                    f_mat[i + j * n] = one_c / diff;
                }
            }
        }

        // V^H (conjugate transpose of V)
        let vh = complex_conj_transpose(v, n);

        // Build M_bar = diag(d_bar_lambda) + F .* (V^H d_bar_V)
        let mut m_bar = vec![zero_c; n * n];

        if let Some(ref dv_bar) = cotangent.vectors {
            let dv_bar_data = extract_data_scalar(dv_bar)?;
            let dv_bar_b = &dv_bar_data[b * n * n..(b + 1) * n * n];
            let vh_dv = complex_mat_mul_nn(&vh, dv_bar_b, n);
            for k in 0..n * n {
                m_bar[k] = f_mat[k] * vh_dv[k];
            }
        }

        if let Some(ref dlam) = cotangent.values {
            let dlam_data = extract_data_scalar(dlam)?;
            for i in 0..n {
                m_bar[i + i * n] = m_bar[i + i * n] + dlam_data[b * n + i];
            }
        }

        // d_bar_A = V^{-H} M_bar V^H = solve(V^H, M_bar @ V^H)
        let m_vh = complex_mat_mul_nn(&m_bar, &vh, n);
        let da_complex = complex_solve_nn(ctx, &vh, &m_vh, n)?;

        // Take real part (since input A was real)
        for k in 0..n * n {
            grad_data[b * n * n + k] = da_complex[k].re;
        }
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(grad_data, &dims).map_err(to_ad_err)
}

/// Reverse-mode AD rule for pseudoinverse (VJP / pullback).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::pinv_rrule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
/// let cotangent = Tensor::<f64>::ones(&[4, 3], mem, col);
/// let grad_a = pinv_rrule(&mut ctx, &a, &cotangent, None).unwrap();
/// ```
pub fn pinv_rrule<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    cotangent: &Tensor<T>,
    rcond: Option<f64>,
) -> AdResult<Tensor<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Pinv, "pinv_rrule")
        .map_err(to_ad_err)?;

    // dA = -(A+)^T dA+ (A+)^T + (I - AA+)(dA+)^T A+(A+)^T + (A+)^T A+ (dA+)^T (I - A+A)
    let ap = pinv(ctx, tensor, rcond)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor)?;
    let (ap_data, _) = extract_data(&ap)?;
    let (dap_data, _) = extract_data(cotangent)?;

    let mut grad_a = vec![T::zero(); m * n * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
        let ap_b = &ap_data[batch * n * m..(batch + 1) * n * m];
        let dap_b = &dap_data[batch * n * m..(batch + 1) * n * m];

        let apt = transpose(ap_b, n, m); // m×n
        let dapt = transpose(dap_b, n, m); // m×n

        // Term 1: -(A+)^T dA+ (A+)^T = -apt * dap * apt^T
        // apt: m×n, dap: n×m, apt: m×n → m×n * n×m * m×n = m×n
        let t1 = backend_mat_mul(ctx, &apt, m, n, dap_b, m)?;
        let t1 = backend_mat_mul(ctx, &t1, m, m, &apt, n)?;
        let t1 = scale_vec(&t1, -T::one());

        // Term 2: (I - AA+)(dA+)^T A+ (A+)^T
        // AA+ (m×m)
        let aap = backend_mat_mul(ctx, a_b, m, n, ap_b, m)?;
        let i_m = eye::<T>(m);
        let i_aap = sub_vec(&i_m, &aap);
        // (dA+)^T A+ = dapt * ap (m×n * n×m = m×m)
        let dapt_ap = backend_mat_mul(ctx, &dapt, m, n, ap_b, m)?;
        // * (A+)^T = * apt (m×m * m×n = m×n)
        let dapt_ap_apt = backend_mat_mul(ctx, &dapt_ap, m, m, &apt, n)?;
        let t2 = backend_mat_mul(ctx, &i_aap, m, m, &dapt_ap_apt, n)?;

        // Term 3: (A+)^T A+ (dA+)^T (I - A+A)
        // A+A (n×n)
        let apa = backend_mat_mul(ctx, ap_b, n, m, a_b, n)?;
        let i_n = eye::<T>(n);
        let i_apa = sub_vec(&i_n, &apa);
        // (A+)^T A+ = apt * ap (m×n * n×m = m×m)
        let apt_ap = backend_mat_mul(ctx, &apt, m, n, ap_b, m)?;
        // * (dA+)^T = * dapt (m×m * m×n = m×n)
        let apt_ap_dapt = backend_mat_mul(ctx, &apt_ap, m, m, &dapt, n)?;
        let t3 = backend_mat_mul(ctx, &apt_ap_dapt, m, n, &i_apa, n)?;

        let da_b = add_vec(&t1, &add_vec(&t2, &t3));
        grad_a[batch * m * n..(batch + 1) * m * n].copy_from_slice(&da_b);
    }

    let dims = output_dims(&[m, n], batch_dims);
    tensor_from_data(grad_a, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))
}
