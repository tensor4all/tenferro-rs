use super::*;

/// Forward-mode AD rule for general eigendecomposition (JVP / pushforward).
///
/// Given eigendecomposition `A V = V diag(lambda)`, computes the tangents
/// of eigenvalues and eigenvectors from a real tangent `dA` using the
/// Mike Giles formulas.
///
/// Returns `(primal, tangent)` where both are [`EigResult`] with complex
/// eigenvalues and eigenvectors.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::eig_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3], mem, col).unwrap();
/// let da = Tensor::<f64>::ones(&[3, 3], mem, col).unwrap();
/// let (result, dresult) = eig_frule(&mut ctx, &a, &da).unwrap();
/// ```
pub fn eig_frule<
    T: KernelLinalgScalar<Real = T, Complex = num_complex::Complex<T>> + num_traits::Float,
    C,
>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
) -> AdResult<(EigResult<T>, EigResult<T>)>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T::Real>>,
    C::Backend: 'static,
    T::Real: tenferro_tensor::KeepCountScalar,
{
    // Forward pass
    let eig_result = eig(ctx, tensor).map_err(to_ad_err)?;

    let (n, batch_dims) = validate_square(tensor).map_err(to_ad_err)?;
    let bc = batch_count(batch_dims);

    let val_data = extract_data_scalar(&eig_result.values)?;
    let vec_data = extract_data_scalar(&eig_result.vectors)?;
    let (tang_data, _) = extract_data(tangent)?;

    let zero_c = Cx::new(T::zero(), T::zero());
    let one_c = Cx::new(T::one(), T::zero());

    let mut dval_data = vec![zero_c; n * bc];
    let mut dvec_data = vec![zero_c; n * n * bc];

    for b in 0..bc {
        let lambda = &val_data[b * n..(b + 1) * n];
        let v = &vec_data[b * n * n..(b + 1) * n * n];
        let da = &tang_data[b * n * n..(b + 1) * n * n];

        // Convert real dA to complex
        let da_complex: Vec<Cx<T>> = da.iter().map(|&x| Cx::new(x, T::zero())).collect();

        // W = V^{-1} dA V = solve(V, dA_c @ V)
        let da_v = complex_mat_mul_nn(&da_complex, v, n);
        let w = complex_solve_nn(ctx, v, &da_v, n)?;

        // d_lambda = diag(W)
        for i in 0..n {
            dval_data[b * n + i] = w[i + i * n];
        }

        // F matrix: F[i,j] = 1/(lambda_j - lambda_i) for i != j, 0 on diagonal
        let mut f_mat = vec![zero_c; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let diff = lambda[j] - lambda[i];
                    f_mat[i + j * n] = one_c / diff;
                }
            }
        }

        // dV = V * (F .* W)
        let mut fw = vec![zero_c; n * n];
        for k in 0..n * n {
            fw[k] = f_mat[k] * w[k];
        }
        let dv = complex_mat_mul_nn(v, &fw, n);
        dvec_data[b * n * n..(b + 1) * n * n].copy_from_slice(&dv);
    }

    // Build tangent EigResult
    let val_dims = output_dims(&[n], batch_dims);
    let vec_dims = output_dims(&[n, n], batch_dims);

    let d_result = EigResult {
        values: tensor_from_data_scalar(dval_data, &val_dims).map_err(to_ad_err)?,
        vectors: tensor_from_data_scalar(dvec_data, &vec_dims).map_err(to_ad_err)?,
    };

    Ok((eig_result, d_result))
}

/// Forward-mode AD rule for pseudoinverse (JVP / pushforward).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::pinv_frule;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 4], mem, col).unwrap();
/// let da = Tensor::<f64>::ones(&[3, 4], mem, col).unwrap();
/// let (pinv_a, dpinv_a) = pinv_frule(&mut ctx, &a, &da, None).unwrap();
/// ```
pub fn pinv_frule<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    tangent: &Tensor<T>,
    rcond: Option<f64>,
) -> AdResult<(Tensor<T>, Tensor<T>)>
where
    T: KernelLinalgScalar
        + crate::prims_bridge::ScaleTensorByRealSameShape<C>
        + tenferro_algebra::Conjugate,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorResolveConjContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T::Real>>,
    C::Backend: 'static,
    T::Real: tenferro_tensor::KeepCountScalar,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Pinv, "pinv_frule")
        .map_err(to_ad_err)?;

    // dA+ = -A+ dA A+ + (I - A+A) dA^T (A+)^T A+ + A+ (A+)^T dA^T (I - AA+)
    let ap = pinv(ctx, tensor, rcond)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let (m, n, batch_dims) = validate_2d(tensor)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    let bc = batch_count(batch_dims);

    let (a_data, _) = extract_data(tensor)?;
    let (ap_data, _) = extract_data(&ap)?;
    let (da_data, _) = extract_data(tangent)?;

    let mut dap_data = vec![T::zero(); n * m * bc];

    for batch in 0..bc {
        let a_b = &a_data[batch * m * n..(batch + 1) * m * n];
        let ap_b = &ap_data[batch * n * m..(batch + 1) * n * m];
        let da_b = &da_data[batch * m * n..(batch + 1) * m * n];

        let dat = transpose(da_b, m, n); // n×m
        let apt = transpose(ap_b, n, m); // m×n

        // Term 1: -A+ dA A+ (n×m × m×n × n×m = n×m)
        let ap_da = backend_mat_mul(ctx, ap_b, n, m, da_b, n)?;
        let ap_da_ap = backend_mat_mul(ctx, &ap_da, n, n, ap_b, m)?;
        let t1 = scale_vec(&ap_da_ap, -T::one());

        // Term 2: (I - A+A) dA^T (A+)^T A+
        let apa = backend_mat_mul(ctx, ap_b, n, m, a_b, n)?; // n×n
        let i_n = eye::<T>(n);
        let i_apa = sub_vec(&i_n, &apa);
        let dat_apt = backend_mat_mul(ctx, &dat, n, m, &apt, n)?; // n×n
        let dat_apt_ap = backend_mat_mul(ctx, &dat_apt, n, n, ap_b, m)?; // n×m
        let t2 = backend_mat_mul(ctx, &i_apa, n, n, &dat_apt_ap, m)?;

        // Term 3: A+ (A+)^T dA^T (I - AA+)
        let aap = backend_mat_mul(ctx, a_b, m, n, ap_b, m)?; // m×m
        let i_m = eye::<T>(m);
        let i_aap = sub_vec(&i_m, &aap);
        let ap_apt = backend_mat_mul(ctx, ap_b, n, m, &apt, n)?; // n×n
        let ap_apt_dat = backend_mat_mul(ctx, &ap_apt, n, n, &dat, m)?; // n×m
        let t3 = backend_mat_mul(ctx, &ap_apt_dat, n, m, &i_aap, m)?;

        let dap_b_vec = add_vec(&t1, &add_vec(&t2, &t3));
        dap_data[batch * n * m..(batch + 1) * n * m].copy_from_slice(&dap_b_vec);
    }

    let dims = output_dims(&[n, m], batch_dims);
    let dap = tensor_from_data(dap_data, &dims)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    Ok((ap, dap))
}
