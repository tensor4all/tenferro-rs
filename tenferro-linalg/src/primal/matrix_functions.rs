use super::*;

fn batched_identity<T: KernelLinalgScalar>(
    n: usize,
    batch_dims: &[usize],
    logical_memory_space: tenferro_device::LogicalMemorySpace,
) -> Result<Tensor<T>> {
    let mut reshape_dims = vec![n, n];
    reshape_dims.extend(std::iter::repeat(1).take(batch_dims.len()));
    let eye = crate::prims_bridge::identity_matrix(n, logical_memory_space)?;
    let eye = eye.reshape(&reshape_dims)?;
    eye.broadcast(&output_dims(&[n, n], batch_dims))
}

/// Raise a square matrix to an integer power.
pub fn matrix_power<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    exponent: i64,
) -> Result<Tensor<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C: tenferro_prims::TensorSemiringContextFor<tenferro_algebra::Standard<T>>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::MatrixPower, "matrix_power")?;

    let (n, batch_dims) = validate_square(tensor)?;

    if exponent == 0 {
        return batched_identity::<T>(n, batch_dims, tensor.logical_memory_space());
    }
    if exponent == 1 {
        return Ok(tensor.clone());
    }
    if exponent == -1 {
        return inv(ctx, tensor);
    }

    let mut positive_exponent = if exponent < 0 {
        exponent.checked_abs().ok_or_else(|| {
            Error::InvalidArgument("matrix_power does not support i64::MIN exponent".into())
        })? as u64
    } else {
        exponent as u64
    };
    let mut base = if exponent < 0 {
        inv(ctx, tensor)?
    } else {
        tensor.clone()
    };

    if positive_exponent == 2 {
        return crate::prims_bridge::batched_gemm_with_semiring_tensors(ctx, &base, &base, n, n, n);
    }
    if positive_exponent == 3 {
        let base_squared =
            crate::prims_bridge::batched_gemm_with_semiring_tensors(ctx, &base, &base, n, n, n)?;
        return crate::prims_bridge::batched_gemm_with_semiring_tensors(
            ctx,
            &base_squared,
            &base,
            n,
            n,
            n,
        );
    }

    let mut result = batched_identity::<T>(n, batch_dims, tensor.logical_memory_space())?;

    while positive_exponent > 0 {
        if positive_exponent & 1 == 1 {
            result = crate::prims_bridge::batched_gemm_with_semiring_tensors(
                ctx, &result, &base, n, n, n,
            )?;
        }
        let next_exponent = positive_exponent >> 1;
        if next_exponent > 0 {
            base = crate::prims_bridge::batched_gemm_with_semiring_tensors(
                ctx, &base, &base, n, n, n,
            )?;
        }
        positive_exponent = next_exponent;
    }

    Ok(result)
}

#[cfg(test)]
mod tests;
