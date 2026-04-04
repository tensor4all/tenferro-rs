use super::*;

fn tensor_from_col_major_data<T: LinalgScalar>(data: &[T], dims: &[usize]) -> AdResult<Tensor<T>> {
    tensor_from_data(data.to_vec(), dims).map_err(to_ad_err)
}

fn rhs_dims(n: usize, nrhs: usize) -> Vec<usize> {
    if nrhs == 1 {
        vec![n]
    } else {
        vec![n, nrhs]
    }
}

/// Mat mul via LinalgBackend, returning `Vec` for convenience in AD code.
pub(crate) fn backend_mat_mul<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &[T],
    m: usize,
    k: usize,
    b: &[T],
    n: usize,
) -> AdResult<Vec<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    prims_bridge::batched_gemm_with_semiring_context(ctx, a, m, k, b, n).map_err(to_ad_err)
}

/// Solve triangular via LinalgBackend, returning `Vec` for convenience in AD code.
pub(crate) fn backend_solve_tri<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &[T],
    b: &[T],
    n: usize,
    nrhs: usize,
    upper: bool,
) -> AdResult<Vec<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    if nrhs == 0 {
        return Ok(Vec::new());
    }
    let a_tensor = tensor_from_col_major_data(a, &[n, n])?;
    let b_tensor = tensor_from_col_major_data(b, &rhs_dims(n, nrhs))?;
    let x = <C::Backend as backend::TensorLinalgBackend<T>>::solve_triangular(
        ctx, &a_tensor, &b_tensor, upper,
    )
    .map_err(to_ad_err)?;
    Ok(extract_data(&x)?.0)
}

/// Thin SVD via LinalgBackend, returning `(U, S, V)` for convenience in AD code.
pub(crate) fn backend_thin_svd<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    a: &[T],
    m: usize,
    n: usize,
) -> AdResult<(Vec<T>, Vec<T>, Vec<T>)>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    let k = m.min(n);
    let a_tensor = tensor_from_col_major_data(a, &[m, n])?;
    let result = <C::Backend as backend::TensorLinalgBackend<T>>::thin_svd(ctx, &a_tensor)
        .map_err(to_ad_err)?;
    let (u, _) = extract_data(&result.u)?;
    let s = extract_data_scalar(&result.s)?;
    let (vt, _) = extract_data(&result.vt)?;
    let v = transpose(&vt, k, n);
    Ok((u, s, v))
}
