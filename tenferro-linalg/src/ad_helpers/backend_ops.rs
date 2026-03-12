use super::*;

/// Mat mul via LinalgBackend, returning `Vec` for convenience in AD code.
pub(crate) fn backend_mat_mul<T: KernelLinalgScalar, C>(
    _ctx: &mut C,
    a: &[T],
    m: usize,
    k: usize,
    b: &[T],
    n: usize,
) -> AdResult<Vec<T>>
where
    T: KernelLinalgScalar,
{
    prims_bridge::batched_gemm_via_prims(a, m, k, b, n).map_err(to_ad_err)
}

/// Solve via LinalgBackend, returning `Vec` for convenience in AD code.
pub(crate) fn backend_solve<T: KernelLinalgScalar, C>(
    _ctx: &mut C,
    a: &[T],
    b: &[T],
    n: usize,
    nrhs: usize,
) -> AdResult<Vec<T>>
where
    T: KernelLinalgScalar,
{
    let mut x = vec![T::zero(); n * nrhs];
    backend::cpu::solve_slices(a, b, n, nrhs, &mut x).map_err(to_ad_err)?;
    Ok(x)
}

/// Solve triangular via LinalgBackend, returning `Vec` for convenience in AD code.
pub(crate) fn backend_solve_tri<T: KernelLinalgScalar, C>(
    _ctx: &mut C,
    a: &[T],
    b: &[T],
    n: usize,
    nrhs: usize,
    upper: bool,
) -> AdResult<Vec<T>>
where
    T: KernelLinalgScalar,
{
    let mut x = vec![T::zero(); n * nrhs];
    backend::cpu::solve_triangular_slices(a, b, n, nrhs, upper, &mut x).map_err(to_ad_err)?;
    Ok(x)
}

/// Thin SVD via LinalgBackend, returning `(U, S, V)` for convenience in AD code.
pub(crate) fn backend_thin_svd<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    _ctx: &mut C,
    a: &[T],
    m: usize,
    n: usize,
) -> AdResult<(Vec<T>, Vec<T>, Vec<T>)>
where
    T: KernelLinalgScalar,
{
    let k = m.min(n);
    let mut u = vec![T::zero(); m * k];
    let mut s = vec![T::zero(); k];
    let mut vt = vec![T::zero(); k * n];
    backend::cpu::thin_svd_slices(a, m, n, &mut u, &mut s, &mut vt).map_err(to_ad_err)?;
    let v = transpose(&vt, k, n);
    Ok((u, s, v))
}

/// QR decomposition via LinalgBackend, returning `(Q, R)` for convenience in AD code.
pub(crate) fn backend_qr<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    _ctx: &mut C,
    a: &[T],
    m: usize,
    n: usize,
) -> AdResult<(Vec<T>, Vec<T>)>
where
    T: KernelLinalgScalar,
{
    let k = m.min(n);
    let mut q = vec![T::zero(); m * k];
    let mut r = vec![T::zero(); k * n];
    backend::cpu::qr_slices(a, m, n, &mut q, &mut r).map_err(to_ad_err)?;
    Ok((q, r))
}
