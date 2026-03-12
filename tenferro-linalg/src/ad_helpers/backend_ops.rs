use super::*;

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

/// Solve via LinalgBackend, returning `Vec` for convenience in AD code.
pub(crate) fn backend_solve<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &[T],
    b: &[T],
    n: usize,
    nrhs: usize,
) -> AdResult<Vec<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    backend::slice_bridge::solve_vec(ctx, a, b, n, nrhs).map_err(to_ad_err)
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
{
    backend::slice_bridge::solve_triangular_vec(ctx, a, b, n, nrhs, upper).map_err(to_ad_err)
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
{
    let k = m.min(n);
    let (u, s, vt) = backend::slice_bridge::thin_svd_vec(ctx, a, m, n).map_err(to_ad_err)?;
    let v = transpose(&vt, k, n);
    Ok((u, s, v))
}

/// QR decomposition via LinalgBackend, returning `(Q, R)` for convenience in AD code.
pub(crate) fn backend_qr<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    a: &[T],
    m: usize,
    n: usize,
) -> AdResult<(Vec<T>, Vec<T>)>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    backend::slice_bridge::qr_vec(ctx, a, m, n).map_err(to_ad_err)
}
