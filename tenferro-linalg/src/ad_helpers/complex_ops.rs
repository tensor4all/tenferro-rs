use super::*;

/// Complex type alias parameterized by real scalar.
pub(crate) type Cx<R> = num_complex::Complex<R>;

/// Complex matrix multiply via the linalg backend.
pub(crate) fn complex_mat_mul_nn_backend<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    a: &[Cx<T>],
    b: &[Cx<T>],
    n: usize,
) -> AdResult<Vec<Cx<T>>>
where
    T: KernelLinalgScalar,
    Cx<T>: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<Cx<T>>,
    <C as backend::TensorLinalgContextFor<Cx<T>>>::Backend: 'static,
{
    crate::prims_bridge::batched_gemm_with_semiring_context(ctx, a, n, n, b, n).map_err(to_ad_err)
}

/// Conjugate transpose of a complex column-major matrix.
pub(crate) fn complex_conj_transpose<R>(a: &[Cx<R>], n: usize) -> Vec<Cx<R>>
where
    R: num_traits::Float + num_traits::NumCast,
{
    let zero = Cx::new(R::zero(), R::zero());
    let mut result = vec![zero; n * n];
    for j in 0..n {
        for i in 0..n {
            result[i + j * n] = a[j + i * n].conj();
        }
    }
    result
}

/// Solve `A X = B` for complex square matrices.
pub(crate) fn complex_solve_nn<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    a: &[Cx<T>],
    b: &[Cx<T>],
    n: usize,
) -> AdResult<Vec<Cx<T>>>
where
    T: KernelLinalgScalar,
    Cx<T>: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<Cx<T>>,
    <C as backend::TensorLinalgContextFor<Cx<T>>>::Backend: 'static,
{
    let a_tensor = tensor_from_data(a.to_vec(), &[n, n]).map_err(to_ad_err)?;
    let b_tensor = tensor_from_data(b.to_vec(), &[n, n]).map_err(to_ad_err)?;
    let x =
        <<C as backend::TensorLinalgContextFor<Cx<T>>>::Backend as backend::TensorLinalgBackend<
            Cx<T>,
        >>::solve(ctx, &a_tensor, &b_tensor)
        .map_err(to_ad_err)?;
    Ok(extract_data(&x)?.0)
}
