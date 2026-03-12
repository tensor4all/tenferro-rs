use super::*;

/// Compute the eigendecomposition of a general (non-symmetric) square matrix.
pub fn eig<
    T: KernelLinalgScalar<Real = T, Complex = num_complex::Complex<T>> + num_traits::Float,
    C,
>(
    ctx: &mut C,
    tensor: &Tensor<T>,
) -> Result<EigResult<T>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    let result = <C::Backend as backend::TensorLinalgBackend<T>>::eig(ctx, tensor)?;
    Ok(EigResult {
        values: result.values,
        vectors: result.vectors,
    })
}

pub(crate) fn require_linalg_support<T: KernelLinalgScalar, C>(
    capability: backend::LinalgCapabilityOp,
    op: &str,
) -> Result<()>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    if <C::Backend as backend::TensorLinalgBackend<T>>::has_linalg_support(capability) {
        return Ok(());
    }

    Err(Error::DeviceError(format!(
        "{op} is not supported on the current linalg backend"
    )))
}

/// Compute the Moore-Penrose pseudoinverse of a matrix.
pub fn pinv<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    rcond: Option<f64>,
) -> Result<Tensor<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Pinv, "pinv")?;

    let (m, n, batch_dims) = validate_2d(tensor)?;
    let svd_result = svd(ctx, tensor, None)?;
    let u_input = ensure_col_major(&svd_result.u);
    let s_input = ensure_col_major(&svd_result.s);
    let vt_input = ensure_col_major(&svd_result.vt);

    let u_data = extract_slice(&u_input)?;
    let s_data = extract_slice(&s_input)?;
    let vt_data = extract_slice(&vt_input)?;
    let u_off = u_input.offset() as usize;
    let s_off = s_input.offset() as usize;
    let vt_off = vt_input.offset() as usize;

    let k = m.min(n);
    let bc = batch_count(batch_dims);
    let threshold: T = scalar_from(rcond.unwrap_or(1e-15))?;

    let mut result_data = vec![T::zero(); n * m * bc];

    for b in 0..bc {
        let s_b = &s_data[s_off + b * k..s_off + (b + 1) * k];
        let u_b = &u_data[u_off + b * m * k..u_off + (b + 1) * m * k];
        let vt_b = &vt_data[vt_off + b * k * n..vt_off + (b + 1) * k * n];

        let s_max = s_b
            .iter()
            .copied()
            .fold(T::zero(), |a, b| if a > b { a } else { b });
        let cutoff = s_max * threshold;

        let mut sinv_ut = vec![T::zero(); k * m];
        for i in 0..k {
            if s_b[i] > cutoff {
                let sinv = T::one() / s_b[i];
                for j in 0..m {
                    sinv_ut[i + j * k] = sinv * u_b[j + i * m];
                }
            }
        }

        for j in 0..m {
            for i in 0..n {
                let mut sum = T::zero();
                for p in 0..k {
                    sum = sum + vt_b[p + i * k] * sinv_ut[p + j * k];
                }
                result_data[b * n * m + i + j * n] = sum;
            }
        }
    }

    let dims = output_dims(&[n, m], batch_dims);
    tensor_from_data(result_data, &dims)
}

/// Compute the matrix exponential `exp(A)` of a square matrix.
pub fn matrix_exp<T: KernelLinalgScalar, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::MatrixExp, "matrix_exp")?;

    let (n, batch_dims) = validate_square(tensor)?;
    let input = ensure_col_major(tensor);
    let data = extract_slice(&input)?;
    let offset = input.offset() as usize;
    let bc = batch_count(batch_dims);
    let mat_size = n * n;

    let mut result_data = vec![T::zero(); mat_size * bc];

    for b in 0..bc {
        let start = offset + b * mat_size;
        let a_slice = &data[start..start + mat_size];
        let exp_a = matrix_exp_single(ctx, a_slice, n)?;
        result_data[b * mat_size..(b + 1) * mat_size].copy_from_slice(&exp_a);
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(result_data, &dims)
}
