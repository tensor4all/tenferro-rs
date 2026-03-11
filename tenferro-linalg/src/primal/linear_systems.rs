use super::*;

/// Solve a square linear system `A x = b`.
pub fn solve<T: LinalgScalar, C>(ctx: &mut C, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    <C::Backend as backend::TensorLinalgBackend<T>>::solve(ctx, a, b)
}

/// Solve a square linear system with numerical status information.
pub fn solve_ex<T: LinalgScalar, C>(
    _ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<SolveExResult<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::SolveEx, "solve_ex")?;

    let (n, batch_dims) = validate_square(a)?;
    let rhs = backend::tensor_helpers::validate_solve_rhs_shape(b, n, batch_dims, "solve_ex")?;
    let bc = batch_count(batch_dims);

    let a_input = ensure_col_major(a);
    let b_input = ensure_col_major(b);
    let a_data = extract_slice(&a_input)?;
    let b_data = extract_slice(&b_input)?;
    let a_offset = a_input.offset() as usize;
    let b_offset = b_input.offset() as usize;

    let mat_size = n * n;
    let rhs_size = n * rhs.nrhs;
    let mut solution = vec![T::zero(); rhs_size * bc];
    let mut info = vec![0_i32; bc];

    for batch in 0..bc {
        let a_start = a_offset + batch * mat_size;
        let b_start = b_offset + batch * rhs_size;
        let a_slice = &a_data[a_start..a_start + mat_size];
        let b_slice = &b_data[b_start..b_start + rhs_size];
        let x_out = &mut solution[batch * rhs_size..(batch + 1) * rhs_size];
        if backend::cpu::solve_slices(a_slice, b_slice, n, rhs.nrhs, x_out).is_err() {
            x_out.fill(T::zero());
            info[batch] = 1;
        }
    }

    Ok(SolveExResult {
        solution: tensor_from_data(solution, &rhs.output_dims)?,
        info,
    })
}

/// Compute the inverse of a square matrix.
pub fn inv<T: LinalgScalar, C>(_ctx: &mut C, tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Inv, "inv")?;

    let (n, batch_dims) = validate_square(tensor)?;
    let input = ensure_col_major(tensor);
    let data = extract_slice(&input)?;
    let offset = input.offset() as usize;
    let bc = batch_count(batch_dims);
    let mat_size = n * n;

    let mut eye_mat = vec![T::zero(); n * n];
    for i in 0..n {
        eye_mat[i + i * n] = T::one();
    }

    let mut inv_data = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let start = offset + b * mat_size;
        let a_b = &data[start..start + mat_size];
        let x_out = &mut inv_data[b * mat_size..(b + 1) * mat_size];
        backend::cpu::solve_slices(a_b, &eye_mat, n, n, x_out)?;
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(inv_data, &dims)
}

/// Compute the inverse with numerical status information.
pub fn inv_ex<T: LinalgScalar, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<InvExResult<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    let (n, batch_dims) = validate_square(tensor)?;
    let bc = batch_count(batch_dims);
    let mut eye_data = vec![T::zero(); n * n * bc];
    let eye = identity_matrix::<T>(n);
    for batch in 0..bc {
        eye_data[batch * n * n..(batch + 1) * n * n].copy_from_slice(&eye);
    }
    let rhs = tensor_from_data(eye_data, &output_dims(&[n, n], batch_dims))?;
    let result = solve_ex(ctx, tensor, &rhs)?;
    Ok(InvExResult {
        inverse: result.solution,
        info: result.info,
    })
}

/// Compute the determinant of a square matrix.
pub fn det<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    _ctx: &mut C,
    tensor: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Det, "det")?;

    let (n, batch_dims) = validate_square(tensor)?;
    let input = ensure_col_major(tensor);
    let data = extract_slice(&input)?;
    let offset = input.offset() as usize;
    let bc = batch_count(batch_dims);
    let mat_size = n * n;

    let mut det_data = vec![T::zero(); bc];
    let mut perm = vec![0usize; n];
    let mut l_buf = vec![T::zero(); n * n];
    let mut u_buf = vec![T::zero(); n * n];

    for (b, det_slot) in det_data.iter_mut().enumerate().take(bc) {
        let start = offset + b * mat_size;
        let batch_data = &data[start..start + mat_size];

        backend::cpu::lu_slices(batch_data, n, n, &mut perm, &mut l_buf, &mut u_buf)?;

        let mut d = T::one();
        for i in 0..n {
            d = d * u_buf[i + i * n];
        }

        let mut sign = 1i32;
        let mut visited = vec![false; n];
        for i in 0..n {
            if !visited[i] {
                visited[i] = true;
                let mut j = perm[i];
                while j != i {
                    sign = -sign;
                    visited[j] = true;
                    j = perm[j];
                }
            }
        }

        if sign < 0 {
            d = T::zero() - d;
        }
        *det_slot = d;
    }

    let dims = if batch_dims.is_empty() {
        vec![]
    } else {
        batch_dims.to_vec()
    };

    if dims.is_empty() {
        Tensor::from_vec(det_data, &dims, &[], 0)
    } else {
        tensor_from_data(det_data, &dims)
    }
}

/// Compute sign and log-absolute-determinant of a square matrix.
pub fn slogdet<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    _ctx: &mut C,
    tensor: &Tensor<T>,
) -> Result<SlogdetResult<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Slogdet, "slogdet")?;

    let (n, batch_dims) = validate_square(tensor)?;
    let input = ensure_col_major(tensor);
    let data = extract_slice(&input)?;
    let offset = input.offset() as usize;
    let bc = batch_count(batch_dims);
    let mat_size = n * n;

    let mut sign_data = vec![T::zero(); bc];
    let mut logabsdet_data = vec![T::zero(); bc];
    let mut perm = vec![0usize; n];
    let mut l_buf = vec![T::zero(); n * n];
    let mut u_buf = vec![T::zero(); n * n];

    for b in 0..bc {
        let start = offset + b * mat_size;
        let batch_data = &data[start..start + mat_size];

        backend::cpu::lu_slices(batch_data, n, n, &mut perm, &mut l_buf, &mut u_buf)?;

        let mut log_abs = T::zero();
        let mut sign = T::one();
        for i in 0..n {
            let diag = u_buf[i + i * n];
            log_abs = log_abs + diag.abs().ln();
            if diag < T::zero() {
                sign = T::zero() - sign;
            }
        }

        let mut perm_sign = 1i32;
        let mut visited = vec![false; n];
        for i in 0..n {
            if !visited[i] {
                visited[i] = true;
                let mut j = perm[i];
                while j != i {
                    perm_sign = -perm_sign;
                    visited[j] = true;
                    j = perm[j];
                }
            }
        }
        if perm_sign < 0 {
            sign = T::zero() - sign;
        }

        sign_data[b] = sign;
        logabsdet_data[b] = log_abs;
    }

    let dims = if batch_dims.is_empty() {
        vec![]
    } else {
        batch_dims.to_vec()
    };

    if dims.is_empty() {
        Ok(SlogdetResult {
            sign: Tensor::from_vec(sign_data, &dims, &[], 0)?,
            logabsdet: Tensor::from_vec(logabsdet_data, &dims, &[], 0)?,
        })
    } else {
        Ok(SlogdetResult {
            sign: tensor_from_data(sign_data, &dims)?,
            logabsdet: tensor_from_data(logabsdet_data, &dims)?,
        })
    }
}
