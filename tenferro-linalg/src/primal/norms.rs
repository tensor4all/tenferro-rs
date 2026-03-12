use super::*;

/// Solve a triangular linear system `A x = b`.
pub fn solve_triangular<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
    upper: bool,
) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    <C::Backend as backend::TensorLinalgBackend<T>>::solve_triangular(ctx, a, b, upper)
}

/// Compute a norm.
pub fn norm<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    kind: NormKind,
) -> Result<Tensor<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Norm, "norm")?;

    if tensor.ndim() == 1 {
        let input = ensure_col_major(tensor);
        let offset = input.offset() as usize;
        let len = tensor.dims()[0];
        let vec_data = &extract_slice(&input)?[offset..offset + len];

        let value = match kind {
            NormKind::Fro => {
                let mut sum = T::zero();
                for &v in vec_data {
                    sum = sum + v * v;
                }
                sum.sqrt()
            }
            NormKind::L1 => vec_data.iter().fold(T::zero(), |acc, &v| acc + v.abs()),
            NormKind::Inf => vec_data.iter().fold(T::zero(), |acc, &v| acc.max(v.abs())),
            NormKind::Lp(p) => {
                if p < 1.0 {
                    return Err(invalid_vector_lp_exponent_error(p));
                }
                let (p_t, mut sum) = (scalar_from::<T>(p)?, T::zero());
                for &v in vec_data {
                    sum = sum + v.abs().powf(p_t);
                }
                sum.powf(T::one() / p_t)
            }
            NormKind::Nuclear | NormKind::Spectral => {
                return Err(matrix_only_norm_kind_error(kind));
            }
        };

        return tensor_from_data(vec![value], &[]);
    }

    let (m, n, batch_dims) = validate_2d(tensor)?;
    let bc = batch_count(batch_dims);
    let mat_size = m * n;
    let out_dims = if batch_dims.is_empty() {
        vec![]
    } else {
        batch_dims.to_vec()
    };

    let input = ensure_col_major(tensor);
    let data = extract_slice(&input)?;
    let offset = input.offset() as usize;

    match kind {
        NormKind::Fro => {
            let mut out = vec![T::zero(); bc];
            for (batch, out_slot) in out.iter_mut().enumerate().take(bc) {
                let start = offset + batch * mat_size;
                let mut sum = T::zero();
                for i in 0..mat_size {
                    let v = data[start + i];
                    sum = sum + v * v;
                }
                *out_slot = sum.sqrt();
            }
            tensor_from_data(out, &out_dims)
        }
        NormKind::Nuclear => {
            let svd_result = svd(ctx, tensor, None)?;
            let s_data = extract_slice(&svd_result.s)?;
            let s_off = svd_result.s.offset() as usize;
            let k = m.min(n);
            let mut out = vec![T::zero(); bc];
            for (batch, out_slot) in out.iter_mut().enumerate().take(bc) {
                let mut sum = T::zero();
                let start = s_off + batch * k;
                for i in 0..k {
                    sum = sum + s_data[start + i];
                }
                *out_slot = sum;
            }
            tensor_from_data(out, &out_dims)
        }
        NormKind::Spectral => {
            let svd_result = svd(ctx, tensor, None)?;
            let s_data = extract_slice(&svd_result.s)?;
            let s_off = svd_result.s.offset() as usize;
            let k = m.min(n);
            let mut out = vec![T::zero(); bc];
            for (batch, out_slot) in out.iter_mut().enumerate().take(bc) {
                *out_slot = s_data[s_off + batch * k];
            }
            tensor_from_data(out, &out_dims)
        }
        NormKind::L1 => {
            let mut out = vec![T::zero(); bc];
            for (batch, out_slot) in out.iter_mut().enumerate().take(bc) {
                if m == 0 || n == 0 {
                    *out_slot = T::zero();
                    continue;
                }
                let start = offset + batch * mat_size;
                let mut max_col_sum = T::zero();
                for j in 0..n {
                    let mut col_sum = T::zero();
                    for i in 0..m {
                        col_sum = col_sum + data[start + i + j * m].abs();
                    }
                    if j == 0 || col_sum > max_col_sum {
                        max_col_sum = col_sum;
                    }
                }
                *out_slot = max_col_sum;
            }
            tensor_from_data(out, &out_dims)
        }
        NormKind::Inf => {
            let mut out = vec![T::zero(); bc];
            for (batch, out_slot) in out.iter_mut().enumerate().take(bc) {
                if m == 0 || n == 0 {
                    *out_slot = T::zero();
                    continue;
                }
                let start = offset + batch * mat_size;
                let mut max_row_sum = T::zero();
                for i in 0..m {
                    let mut row_sum = T::zero();
                    for j in 0..n {
                        row_sum = row_sum + data[start + i + j * m].abs();
                    }
                    if i == 0 || row_sum > max_row_sum {
                        max_row_sum = row_sum;
                    }
                }
                *out_slot = max_row_sum;
            }
            tensor_from_data(out, &out_dims)
        }
        _ => Err(Error::InvalidArgument(format!(
            "norm kind {kind:?} not yet implemented"
        ))),
    }
}

/// Compute the matrix condition number with a selected norm convention.
pub fn cond<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    kind: NormKind,
) -> Result<Tensor<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    match kind {
        NormKind::Fro | NormKind::L1 | NormKind::Inf | NormKind::Spectral => {}
        _ => {
            return Err(Error::InvalidArgument(format!(
                "cond only supports Fro, L1, Inf, and Spectral norms, got {kind:?}"
            )));
        }
    }

    validate_square(tensor)?;
    let lhs = norm(ctx, tensor, kind)?;
    let inverse = inv(ctx, tensor)?;
    let rhs = norm(ctx, &inverse, kind)?;
    let lhs_data = extract_slice(&lhs)?;
    let rhs_data = extract_slice(&rhs)?;
    let lhs_offset = lhs.offset() as usize;
    let rhs_offset = rhs.offset() as usize;
    let len = lhs.dims().iter().product::<usize>().max(1);
    let mut out = vec![T::zero(); len];
    for i in 0..len {
        out[i] = lhs_data[lhs_offset + i] * rhs_data[rhs_offset + i];
    }
    tensor_from_data(out, lhs.dims())
}
