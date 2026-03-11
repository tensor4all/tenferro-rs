use super::*;

/// Raise a square matrix to an integer power.
pub fn matrix_power<T: LinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    exponent: i64,
) -> Result<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::MatrixPower, "matrix_power")?;

    let (n, batch_dims) = validate_square(tensor)?;
    let bc = batch_count(batch_dims);
    let dims = output_dims(&[n, n], batch_dims);

    if exponent == 0 {
        let eye = identity_matrix::<T>(n);
        let mut data = vec![T::zero(); n * n * bc];
        for batch in 0..bc {
            data[batch * n * n..(batch + 1) * n * n].copy_from_slice(&eye);
        }
        return tensor_from_data(data, &dims);
    }

    let positive_exponent = if exponent < 0 {
        let abs = exponent.checked_abs().ok_or_else(|| {
            Error::InvalidArgument("matrix_power does not support i64::MIN exponent".into())
        })?;
        let inverse = inv(ctx, tensor)?;
        return matrix_power(ctx, &inverse, abs);
    } else {
        exponent as u64
    };

    let input = ensure_col_major(tensor);
    let data = extract_slice(&input)?;
    let offset = input.offset() as usize;
    let mat_size = n * n;
    let mut out = vec![T::zero(); mat_size * bc];

    for batch in 0..bc {
        let start = offset + batch * mat_size;
        let a_slice = &data[start..start + mat_size];
        let powered = matrix_power_single(ctx, a_slice, n, positive_exponent)?;
        out[batch * mat_size..(batch + 1) * mat_size].copy_from_slice(&powered);
    }

    tensor_from_data(out, &dims)
}
