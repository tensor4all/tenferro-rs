//! BLAS/LAPACK backend for linear algebra operations.
//!
//! This backend implements [`super::LinalgBackend`] for
//! `f32`, `f64`, `Complex32`, and `Complex64` using:
//! - LAPACK wrappers from the [`lapack`](https://crates.io/crates/lapack) crate
//! - CBLAS symbols from the [`cblas-sys`](https://crates.io/crates/cblas-sys) crate
//!
//! Symbol providers are selected at compile time via crate features
//! (`provider-src` / `provider-inject` and `src-*`).

use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use tenferro_device::{Error, Result};

use super::LinalgBackend;

/// LAPACK/CBLAS backend with compile-time selectable symbol provider.
///
/// This backend is stateless; `&mut self` is accepted for API uniformity
/// and future workspace reuse.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::backend::{BlasLapackBackend, LinalgBackend};
///
/// let mut backend = BlasLapackBackend::new();
/// let a = [1.0_f64, 0.0, 0.0, 1.0]; // 2x2 identity, col-major
/// let mut u = [0.0; 4];
/// let mut s = [0.0; 2];
/// let mut vt = [0.0; 4];
/// backend.thin_svd(&a, 2, 2, &mut u, &mut s, &mut vt).unwrap();
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct BlasLapackBackend;

impl BlasLapackBackend {
    /// Create a new backend instance.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_linalg::backend::BlasLapackBackend;
    ///
    /// let backend = BlasLapackBackend::new();
    /// let _ = backend;
    /// ```
    pub fn new() -> Self {
        Self
    }
}

fn as_i32(name: &str, value: usize) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| Error::InvalidArgument(format!("{name} is too large for LAPACK i32 ABI")))
}

fn check_len(op: &str, field: &str, got: usize, need: usize) -> Result<()> {
    if got < need {
        return Err(Error::InvalidArgument(format!(
            "{op}: {field} slice length {got} < required {need}"
        )));
    }
    Ok(())
}

fn lwork_from_query_f64(op: &str, query: f64) -> Result<i32> {
    if !query.is_finite() {
        return Err(Error::InvalidArgument(format!(
            "{op}: invalid LAPACK work query result {query}"
        )));
    }
    let lwork = query.max(1.0).ceil();
    if lwork > i32::MAX as f64 {
        return Err(Error::InvalidArgument(format!(
            "{op}: requested work size {lwork} exceeds i32::MAX"
        )));
    }
    Ok(lwork as i32)
}

fn lwork_from_query_f32(op: &str, query: f32) -> Result<i32> {
    if !query.is_finite() {
        return Err(Error::InvalidArgument(format!(
            "{op}: invalid LAPACK work query result {query}"
        )));
    }
    let lwork = query.max(1.0).ceil();
    if lwork > i32::MAX as f32 {
        return Err(Error::InvalidArgument(format!(
            "{op}: requested work size {lwork} exceeds i32::MAX"
        )));
    }
    Ok(lwork as i32)
}

fn lwork_from_query_c64(op: &str, query: Complex64) -> Result<i32> {
    lwork_from_query_f64(op, query.re)
}

fn lwork_from_query_c32(op: &str, query: Complex32) -> Result<i32> {
    lwork_from_query_f32(op, query.re)
}

fn check_info_nonnegative(op: &str, info: i32) -> Result<()> {
    if info < 0 {
        return Err(Error::InvalidArgument(format!(
            "{op}: LAPACK reported invalid argument at position {}",
            -info
        )));
    }
    Ok(())
}

fn check_info_success(op: &str, info: i32) -> Result<()> {
    check_info_nonnegative(op, info)?;
    if info > 0 {
        return Err(Error::InvalidArgument(format!(
            "{op}: LAPACK failed with info={info}"
        )));
    }
    Ok(())
}

fn check_info_cholesky(op: &str, info: i32) -> Result<()> {
    check_info_nonnegative(op, info)?;
    if info > 0 {
        return Err(Error::InvalidArgument(format!(
            "{op}: matrix is not positive definite (minor {info})"
        )));
    }
    Ok(())
}

fn pivots_to_forward_perm(m: usize, pivots: &[i32]) -> Result<Vec<usize>> {
    let mut perm: Vec<usize> = (0..m).collect();
    for (i, &p) in pivots.iter().enumerate() {
        if p <= 0 {
            return Err(Error::InvalidArgument(
                "lu: LAPACK returned non-positive pivot index".into(),
            ));
        }
        let j = (p - 1) as usize;
        if j >= m {
            return Err(Error::InvalidArgument(format!(
                "lu: LAPACK pivot index {p} out of range for m={m}"
            )));
        }
        perm.swap(i, j);
    }
    Ok(perm)
}

fn split_lu<T: Copy + Zero + One>(lu: &[T], m: usize, n: usize, l: &mut [T], u_out: &mut [T]) {
    let k = m.min(n);

    // L is m x k, unit lower-triangular.
    for j in 0..k {
        for i in 0..m {
            l[i + j * m] = if i > j {
                lu[i + j * m]
            } else if i == j {
                T::one()
            } else {
                T::zero()
            };
        }
    }

    // U is k x n, upper-triangular.
    for j in 0..n {
        for i in 0..k {
            u_out[i + j * k] = if i <= j { lu[i + j * m] } else { T::zero() };
        }
    }
}

fn fill_zero_upper<T: Copy + Zero>(mat: &mut [T], n: usize) {
    for j in 0..n {
        for i in 0..j {
            mat[i + j * n] = T::zero();
        }
    }
}

macro_rules! impl_lapack_backend_real {
    (
        $ty:ty,
        $gesvd:ident,
        $geqrf:ident,
        $orgqr:ident,
        $getrf:ident,
        $potrf:ident,
        $syev:ident,
        $gesv:ident,
        $trtrs:ident,
        $geev:ident,
        $gemm:path,
        $lwork_from_query:ident
    ) => {
        impl LinalgBackend<$ty> for BlasLapackBackend {
            type Real = $ty;

            fn thin_svd(
                &mut self,
                a: &[$ty],
                m: usize,
                n: usize,
                u: &mut [$ty],
                s: &mut [Self::Real],
                vt: &mut [$ty],
            ) -> Result<()> {
                let k = m.min(n);
                check_len("thin_svd", "a", a.len(), m * n)?;
                check_len("thin_svd", "u", u.len(), m * k)?;
                check_len("thin_svd", "s", s.len(), k)?;
                check_len("thin_svd", "vt", vt.len(), k * n)?;

                let m_i32 = as_i32("thin_svd m", m)?;
                let n_i32 = as_i32("thin_svd n", n)?;
                let k_i32 = as_i32("thin_svd k", k)?;

                let mut a_work = a[..m * n].to_vec();
                let mut info = 0;

                let mut work_query = [0 as $ty; 1];
                unsafe {
                    lapack::$gesvd(
                        b'S',
                        b'S',
                        m_i32,
                        n_i32,
                        &mut a_work,
                        m_i32,
                        &mut s[..k],
                        &mut u[..m * k],
                        m_i32,
                        &mut vt[..k * n],
                        k_i32,
                        &mut work_query,
                        -1,
                        &mut info,
                    );
                }
                check_info_nonnegative("thin_svd(work query)", info)?;

                let lwork = $lwork_from_query("thin_svd", work_query[0])?;
                let mut work = vec![0 as $ty; lwork as usize];

                unsafe {
                    lapack::$gesvd(
                        b'S',
                        b'S',
                        m_i32,
                        n_i32,
                        &mut a_work,
                        m_i32,
                        &mut s[..k],
                        &mut u[..m * k],
                        m_i32,
                        &mut vt[..k * n],
                        k_i32,
                        &mut work,
                        lwork,
                        &mut info,
                    );
                }
                check_info_success("thin_svd", info)
            }

            fn qr(
                &mut self,
                a: &[$ty],
                m: usize,
                n: usize,
                q: &mut [$ty],
                r: &mut [$ty],
            ) -> Result<()> {
                let k = m.min(n);
                check_len("qr", "a", a.len(), m * n)?;
                check_len("qr", "q", q.len(), m * k)?;
                check_len("qr", "r", r.len(), k * n)?;

                if k == 0 {
                    return Ok(());
                }

                let m_i32 = as_i32("qr m", m)?;
                let n_i32 = as_i32("qr n", n)?;
                let k_i32 = as_i32("qr k", k)?;

                let mut a_fact = a[..m * n].to_vec();
                let mut tau = vec![0 as $ty; k];
                let mut info = 0;

                let mut work_query = [0 as $ty; 1];
                unsafe {
                    lapack::$geqrf(
                        m_i32,
                        n_i32,
                        &mut a_fact,
                        m_i32,
                        &mut tau,
                        &mut work_query,
                        -1,
                        &mut info,
                    );
                }
                check_info_nonnegative("qr(geqrf work query)", info)?;

                let geqrf_lwork = $lwork_from_query("qr(geqrf)", work_query[0])?;
                let mut geqrf_work = vec![0 as $ty; geqrf_lwork as usize];

                unsafe {
                    lapack::$geqrf(
                        m_i32,
                        n_i32,
                        &mut a_fact,
                        m_i32,
                        &mut tau,
                        &mut geqrf_work,
                        geqrf_lwork,
                        &mut info,
                    );
                }
                check_info_success("qr(geqrf)", info)?;

                for j in 0..n {
                    for i in 0..k {
                        r[i + j * k] = if i <= j { a_fact[i + j * m] } else { 0 as $ty };
                    }
                }

                let mut q_data = vec![0 as $ty; m * k];
                for j in 0..k {
                    for i in 0..m {
                        q_data[i + j * m] = a_fact[i + j * m];
                    }
                }

                let mut q_work_query = [0 as $ty; 1];
                unsafe {
                    lapack::$orgqr(
                        m_i32,
                        k_i32,
                        k_i32,
                        &mut q_data,
                        m_i32,
                        &tau,
                        &mut q_work_query,
                        -1,
                        &mut info,
                    );
                }
                check_info_nonnegative("qr(orgqr work query)", info)?;

                let orgqr_lwork = $lwork_from_query("qr(orgqr)", q_work_query[0])?;
                let mut orgqr_work = vec![0 as $ty; orgqr_lwork as usize];

                unsafe {
                    lapack::$orgqr(
                        m_i32,
                        k_i32,
                        k_i32,
                        &mut q_data,
                        m_i32,
                        &tau,
                        &mut orgqr_work,
                        orgqr_lwork,
                        &mut info,
                    );
                }
                check_info_success("qr(orgqr)", info)?;

                q[..m * k].copy_from_slice(&q_data);
                Ok(())
            }

            fn lu(
                &mut self,
                a: &[$ty],
                m: usize,
                n: usize,
                perm: &mut [usize],
                l: &mut [$ty],
                u_out: &mut [$ty],
            ) -> Result<()> {
                let k = m.min(n);
                check_len("lu", "a", a.len(), m * n)?;
                check_len("lu", "perm", perm.len(), m)?;
                check_len("lu", "l", l.len(), m * k)?;
                check_len("lu", "u_out", u_out.len(), k * n)?;

                if m == 0 || n == 0 {
                    for (i, p) in perm.iter_mut().take(m).enumerate() {
                        *p = i;
                    }
                    return Ok(());
                }

                let m_i32 = as_i32("lu m", m)?;
                let n_i32 = as_i32("lu n", n)?;

                let mut lu = a[..m * n].to_vec();
                let mut piv = vec![0i32; k];
                let mut info = 0;

                unsafe {
                    lapack::$getrf(m_i32, n_i32, &mut lu, m_i32, &mut piv, &mut info);
                }
                check_info_nonnegative("lu(getrf)", info)?;

                let p = pivots_to_forward_perm(m, &piv)?;
                perm[..m].copy_from_slice(&p);
                split_lu(&lu, m, n, l, u_out);

                Ok(())
            }

            fn cholesky(&mut self, a: &[$ty], n: usize, l: &mut [$ty]) -> Result<()> {
                check_len("cholesky", "a", a.len(), n * n)?;
                check_len("cholesky", "l", l.len(), n * n)?;

                if n == 0 {
                    return Ok(());
                }

                let n_i32 = as_i32("cholesky n", n)?;
                l[..n * n].copy_from_slice(&a[..n * n]);

                let mut info = 0;
                unsafe {
                    lapack::$potrf(b'L', n_i32, &mut l[..n * n], n_i32, &mut info);
                }
                check_info_cholesky("cholesky", info)?;
                fill_zero_upper(&mut l[..n * n], n);
                Ok(())
            }

            fn eigen_sym(
                &mut self,
                a: &[$ty],
                n: usize,
                values: &mut [Self::Real],
                vectors: &mut [$ty],
            ) -> Result<()> {
                check_len("eigen_sym", "a", a.len(), n * n)?;
                check_len("eigen_sym", "values", values.len(), n)?;
                check_len("eigen_sym", "vectors", vectors.len(), n * n)?;

                if n == 0 {
                    return Ok(());
                }

                let n_i32 = as_i32("eigen_sym n", n)?;
                vectors[..n * n].copy_from_slice(&a[..n * n]);

                let mut info = 0;
                let mut work_query = [0 as $ty; 1];
                unsafe {
                    lapack::$syev(
                        b'V',
                        b'L',
                        n_i32,
                        &mut vectors[..n * n],
                        n_i32,
                        &mut values[..n],
                        &mut work_query,
                        -1,
                        &mut info,
                    );
                }
                check_info_nonnegative("eigen_sym(work query)", info)?;

                let lwork = $lwork_from_query("eigen_sym", work_query[0])?;
                let mut work = vec![0 as $ty; lwork as usize];
                unsafe {
                    lapack::$syev(
                        b'V',
                        b'L',
                        n_i32,
                        &mut vectors[..n * n],
                        n_i32,
                        &mut values[..n],
                        &mut work,
                        lwork,
                        &mut info,
                    );
                }
                check_info_success("eigen_sym", info)
            }

            fn mat_mul(
                &mut self,
                a: &[$ty],
                m: usize,
                k: usize,
                b: &[$ty],
                n: usize,
                c: &mut [$ty],
            ) -> Result<()> {
                check_len("mat_mul", "a", a.len(), m * k)?;
                check_len("mat_mul", "b", b.len(), k * n)?;
                check_len("mat_mul", "c", c.len(), m * n)?;

                let m_i32 = as_i32("mat_mul m", m)?;
                let k_i32 = as_i32("mat_mul k", k)?;
                let n_i32 = as_i32("mat_mul n", n)?;

                unsafe {
                    $gemm(
                        cblas_sys::CBLAS_LAYOUT::CblasColMajor,
                        cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                        cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                        m_i32,
                        n_i32,
                        k_i32,
                        1 as $ty,
                        a.as_ptr(),
                        m_i32,
                        b.as_ptr(),
                        k_i32,
                        0 as $ty,
                        c.as_mut_ptr(),
                        m_i32,
                    );
                }
                Ok(())
            }

            fn solve(
                &mut self,
                a: &[$ty],
                b: &[$ty],
                n: usize,
                nrhs: usize,
                x: &mut [$ty],
            ) -> Result<()> {
                check_len("solve", "a", a.len(), n * n)?;
                check_len("solve", "b", b.len(), n * nrhs)?;
                check_len("solve", "x", x.len(), n * nrhs)?;

                let n_i32 = as_i32("solve n", n)?;
                let nrhs_i32 = as_i32("solve nrhs", nrhs)?;

                let mut a_work = a[..n * n].to_vec();
                x[..n * nrhs].copy_from_slice(&b[..n * nrhs]);
                let mut ipiv = vec![0i32; n];
                let mut info = 0;

                unsafe {
                    lapack::$gesv(
                        n_i32,
                        nrhs_i32,
                        &mut a_work,
                        n_i32,
                        &mut ipiv,
                        &mut x[..n * nrhs],
                        n_i32,
                        &mut info,
                    );
                }
                check_info_success("solve", info)
            }

            fn solve_triangular(
                &mut self,
                a: &[$ty],
                b: &[$ty],
                n: usize,
                nrhs: usize,
                upper: bool,
                x: &mut [$ty],
            ) -> Result<()> {
                check_len("solve_triangular", "a", a.len(), n * n)?;
                check_len("solve_triangular", "b", b.len(), n * nrhs)?;
                check_len("solve_triangular", "x", x.len(), n * nrhs)?;

                let n_i32 = as_i32("solve_triangular n", n)?;
                let nrhs_i32 = as_i32("solve_triangular nrhs", nrhs)?;

                x[..n * nrhs].copy_from_slice(&b[..n * nrhs]);
                let mut info = 0;
                let uplo = if upper { b'U' } else { b'L' };

                unsafe {
                    lapack::$trtrs(
                        uplo,
                        b'N',
                        b'N',
                        n_i32,
                        nrhs_i32,
                        &a[..n * n],
                        n_i32,
                        &mut x[..n * nrhs],
                        n_i32,
                        &mut info,
                    );
                }
                check_info_success("solve_triangular", info)
            }

            fn eig_general(
                &mut self,
                a: &[$ty],
                n: usize,
                values_ri: &mut [$ty],
                vectors_ri: &mut [$ty],
            ) -> Result<()> {
                check_len("eig_general", "a", a.len(), n * n)?;
                check_len("eig_general", "values_ri", values_ri.len(), 2 * n)?;
                check_len("eig_general", "vectors_ri", vectors_ri.len(), 2 * n * n)?;

                if n == 0 {
                    return Ok(());
                }

                let n_i32 = as_i32("eig_general n", n)?;
                let mut a_work = a[..n * n].to_vec();
                let mut wr = vec![0 as $ty; n];
                let mut wi = vec![0 as $ty; n];
                let mut vr = vec![0 as $ty; n * n];
                let mut vl = vec![0 as $ty; 1];
                let mut info = 0;

                let mut work_query = [0 as $ty; 1];
                unsafe {
                    lapack::$geev(
                        b'N',
                        b'V',
                        n_i32,
                        &mut a_work,
                        n_i32,
                        &mut wr,
                        &mut wi,
                        &mut vl,
                        1,
                        &mut vr,
                        n_i32,
                        &mut work_query,
                        -1,
                        &mut info,
                    );
                }
                check_info_nonnegative("eig_general(work query)", info)?;

                let lwork = $lwork_from_query("eig_general", work_query[0])?;
                let mut work = vec![0 as $ty; lwork as usize];
                unsafe {
                    lapack::$geev(
                        b'N',
                        b'V',
                        n_i32,
                        &mut a_work,
                        n_i32,
                        &mut wr,
                        &mut wi,
                        &mut vl,
                        1,
                        &mut vr,
                        n_i32,
                        &mut work,
                        lwork,
                        &mut info,
                    );
                }
                check_info_success("eig_general", info)?;

                for i in 0..n {
                    values_ri[2 * i] = wr[i];
                    values_ri[2 * i + 1] = wi[i];
                }

                for v in vectors_ri.iter_mut() {
                    *v = 0 as $ty;
                }

                let mut j = 0usize;
                while j < n {
                    if wi[j] == 0 as $ty {
                        for i in 0..n {
                            vectors_ri[2 * (i + j * n)] = vr[i + j * n];
                            vectors_ri[2 * (i + j * n) + 1] = 0 as $ty;
                        }
                        j += 1;
                    } else if wi[j] > 0 as $ty {
                        for i in 0..n {
                            let re = vr[i + j * n];
                            let im = vr[i + (j + 1) * n];
                            vectors_ri[2 * (i + j * n)] = re;
                            vectors_ri[2 * (i + j * n) + 1] = im;
                            vectors_ri[2 * (i + (j + 1) * n)] = re;
                            vectors_ri[2 * (i + (j + 1) * n) + 1] = -im;
                        }
                        j += 2;
                    } else {
                        // Paired conjugate column already handled by the preceding wi>0 column.
                        j += 1;
                    }
                }

                Ok(())
            }
        }
    };
}

macro_rules! impl_lapack_backend_complex {
    (
        $complex_ty:ty,
        $real_ty:ty,
        $gesvd:ident,
        $geqrf:ident,
        $ungqr:ident,
        $getrf:ident,
        $potrf:ident,
        $heev:ident,
        $gesv:ident,
        $trtrs:ident,
        $geev:ident,
        $gemm:path,
        $lwork_from_query:ident
    ) => {
        impl LinalgBackend<$complex_ty> for BlasLapackBackend {
            type Real = $real_ty;

            fn thin_svd(
                &mut self,
                a: &[$complex_ty],
                m: usize,
                n: usize,
                u: &mut [$complex_ty],
                s: &mut [Self::Real],
                vt: &mut [$complex_ty],
            ) -> Result<()> {
                let k = m.min(n);
                check_len("thin_svd", "a", a.len(), m * n)?;
                check_len("thin_svd", "u", u.len(), m * k)?;
                check_len("thin_svd", "s", s.len(), k)?;
                check_len("thin_svd", "vt", vt.len(), k * n)?;

                let m_i32 = as_i32("thin_svd m", m)?;
                let n_i32 = as_i32("thin_svd n", n)?;
                let k_i32 = as_i32("thin_svd k", k)?;

                let mut a_work = a[..m * n].to_vec();
                let mut info = 0;
                let mut work_query = [<$complex_ty>::new(0 as $real_ty, 0 as $real_ty); 1];
                let mut rwork = vec![0 as $real_ty; (5 * k).max(1)];

                unsafe {
                    lapack::$gesvd(
                        b'S',
                        b'S',
                        m_i32,
                        n_i32,
                        &mut a_work,
                        m_i32,
                        &mut s[..k],
                        &mut u[..m * k],
                        m_i32,
                        &mut vt[..k * n],
                        k_i32,
                        &mut work_query,
                        -1,
                        &mut rwork,
                        &mut info,
                    );
                }
                check_info_nonnegative("thin_svd(work query)", info)?;

                let lwork = $lwork_from_query("thin_svd", work_query[0])?;
                let mut work =
                    vec![<$complex_ty>::new(0 as $real_ty, 0 as $real_ty); lwork as usize];

                unsafe {
                    lapack::$gesvd(
                        b'S',
                        b'S',
                        m_i32,
                        n_i32,
                        &mut a_work,
                        m_i32,
                        &mut s[..k],
                        &mut u[..m * k],
                        m_i32,
                        &mut vt[..k * n],
                        k_i32,
                        &mut work,
                        lwork,
                        &mut rwork,
                        &mut info,
                    );
                }
                check_info_success("thin_svd", info)
            }

            fn qr(
                &mut self,
                a: &[$complex_ty],
                m: usize,
                n: usize,
                q: &mut [$complex_ty],
                r: &mut [$complex_ty],
            ) -> Result<()> {
                let k = m.min(n);
                check_len("qr", "a", a.len(), m * n)?;
                check_len("qr", "q", q.len(), m * k)?;
                check_len("qr", "r", r.len(), k * n)?;

                if k == 0 {
                    return Ok(());
                }

                let m_i32 = as_i32("qr m", m)?;
                let n_i32 = as_i32("qr n", n)?;
                let k_i32 = as_i32("qr k", k)?;

                let mut a_fact = a[..m * n].to_vec();
                let mut tau = vec![<$complex_ty>::new(0 as $real_ty, 0 as $real_ty); k];
                let mut info = 0;

                let mut work_query = [<$complex_ty>::new(0 as $real_ty, 0 as $real_ty); 1];
                unsafe {
                    lapack::$geqrf(
                        m_i32,
                        n_i32,
                        &mut a_fact,
                        m_i32,
                        &mut tau,
                        &mut work_query,
                        -1,
                        &mut info,
                    );
                }
                check_info_nonnegative("qr(geqrf work query)", info)?;

                let geqrf_lwork = $lwork_from_query("qr(geqrf)", work_query[0])?;
                let mut geqrf_work =
                    vec![<$complex_ty>::new(0 as $real_ty, 0 as $real_ty); geqrf_lwork as usize];

                unsafe {
                    lapack::$geqrf(
                        m_i32,
                        n_i32,
                        &mut a_fact,
                        m_i32,
                        &mut tau,
                        &mut geqrf_work,
                        geqrf_lwork,
                        &mut info,
                    );
                }
                check_info_success("qr(geqrf)", info)?;

                for j in 0..n {
                    for i in 0..k {
                        r[i + j * k] = if i <= j {
                            a_fact[i + j * m]
                        } else {
                            <$complex_ty>::new(0 as $real_ty, 0 as $real_ty)
                        };
                    }
                }

                let mut q_data = vec![<$complex_ty>::new(0 as $real_ty, 0 as $real_ty); m * k];
                for j in 0..k {
                    for i in 0..m {
                        q_data[i + j * m] = a_fact[i + j * m];
                    }
                }

                let mut q_work_query = [<$complex_ty>::new(0 as $real_ty, 0 as $real_ty); 1];
                unsafe {
                    lapack::$ungqr(
                        m_i32,
                        k_i32,
                        k_i32,
                        &mut q_data,
                        m_i32,
                        &tau,
                        &mut q_work_query,
                        -1,
                        &mut info,
                    );
                }
                check_info_nonnegative("qr(ungqr work query)", info)?;

                let ungqr_lwork = $lwork_from_query("qr(ungqr)", q_work_query[0])?;
                let mut ungqr_work =
                    vec![<$complex_ty>::new(0 as $real_ty, 0 as $real_ty); ungqr_lwork as usize];

                unsafe {
                    lapack::$ungqr(
                        m_i32,
                        k_i32,
                        k_i32,
                        &mut q_data,
                        m_i32,
                        &tau,
                        &mut ungqr_work,
                        ungqr_lwork,
                        &mut info,
                    );
                }
                check_info_success("qr(ungqr)", info)?;

                q[..m * k].copy_from_slice(&q_data);
                Ok(())
            }

            fn lu(
                &mut self,
                a: &[$complex_ty],
                m: usize,
                n: usize,
                perm: &mut [usize],
                l: &mut [$complex_ty],
                u_out: &mut [$complex_ty],
            ) -> Result<()> {
                let k = m.min(n);
                check_len("lu", "a", a.len(), m * n)?;
                check_len("lu", "perm", perm.len(), m)?;
                check_len("lu", "l", l.len(), m * k)?;
                check_len("lu", "u_out", u_out.len(), k * n)?;

                if m == 0 || n == 0 {
                    for (i, p) in perm.iter_mut().take(m).enumerate() {
                        *p = i;
                    }
                    return Ok(());
                }

                let m_i32 = as_i32("lu m", m)?;
                let n_i32 = as_i32("lu n", n)?;

                let mut lu = a[..m * n].to_vec();
                let mut piv = vec![0i32; k];
                let mut info = 0;
                unsafe {
                    lapack::$getrf(m_i32, n_i32, &mut lu, m_i32, &mut piv, &mut info);
                }
                check_info_nonnegative("lu(getrf)", info)?;

                let p = pivots_to_forward_perm(m, &piv)?;
                perm[..m].copy_from_slice(&p);
                split_lu(&lu, m, n, l, u_out);
                Ok(())
            }

            fn cholesky(
                &mut self,
                a: &[$complex_ty],
                n: usize,
                l: &mut [$complex_ty],
            ) -> Result<()> {
                check_len("cholesky", "a", a.len(), n * n)?;
                check_len("cholesky", "l", l.len(), n * n)?;

                if n == 0 {
                    return Ok(());
                }

                let n_i32 = as_i32("cholesky n", n)?;
                l[..n * n].copy_from_slice(&a[..n * n]);
                let mut info = 0;
                unsafe {
                    lapack::$potrf(b'L', n_i32, &mut l[..n * n], n_i32, &mut info);
                }
                check_info_cholesky("cholesky", info)?;
                fill_zero_upper(&mut l[..n * n], n);
                Ok(())
            }

            fn eigen_sym(
                &mut self,
                a: &[$complex_ty],
                n: usize,
                values: &mut [Self::Real],
                vectors: &mut [$complex_ty],
            ) -> Result<()> {
                check_len("eigen_sym", "a", a.len(), n * n)?;
                check_len("eigen_sym", "values", values.len(), n)?;
                check_len("eigen_sym", "vectors", vectors.len(), n * n)?;

                if n == 0 {
                    return Ok(());
                }

                let n_i32 = as_i32("eigen_sym n", n)?;
                vectors[..n * n].copy_from_slice(&a[..n * n]);

                let mut info = 0;
                let mut work_query = [<$complex_ty>::new(0 as $real_ty, 0 as $real_ty); 1];
                let mut rwork = vec![0 as $real_ty; (3 * n).saturating_sub(2).max(1)];

                unsafe {
                    lapack::$heev(
                        b'V',
                        b'L',
                        n_i32,
                        &mut vectors[..n * n],
                        n_i32,
                        &mut values[..n],
                        &mut work_query,
                        -1,
                        &mut rwork,
                        &mut info,
                    );
                }
                check_info_nonnegative("eigen_sym(work query)", info)?;

                let lwork = $lwork_from_query("eigen_sym", work_query[0])?;
                let mut work =
                    vec![<$complex_ty>::new(0 as $real_ty, 0 as $real_ty); lwork as usize];
                unsafe {
                    lapack::$heev(
                        b'V',
                        b'L',
                        n_i32,
                        &mut vectors[..n * n],
                        n_i32,
                        &mut values[..n],
                        &mut work,
                        lwork,
                        &mut rwork,
                        &mut info,
                    );
                }
                check_info_success("eigen_sym", info)
            }

            fn mat_mul(
                &mut self,
                a: &[$complex_ty],
                m: usize,
                k: usize,
                b: &[$complex_ty],
                n: usize,
                c: &mut [$complex_ty],
            ) -> Result<()> {
                check_len("mat_mul", "a", a.len(), m * k)?;
                check_len("mat_mul", "b", b.len(), k * n)?;
                check_len("mat_mul", "c", c.len(), m * n)?;

                let m_i32 = as_i32("mat_mul m", m)?;
                let k_i32 = as_i32("mat_mul k", k)?;
                let n_i32 = as_i32("mat_mul n", n)?;

                let alpha = [1 as $real_ty, 0 as $real_ty];
                let beta = [0 as $real_ty, 0 as $real_ty];

                unsafe {
                    $gemm(
                        cblas_sys::CBLAS_LAYOUT::CblasColMajor,
                        cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                        cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                        m_i32,
                        n_i32,
                        k_i32,
                        &alpha,
                        a.as_ptr() as *const _,
                        m_i32,
                        b.as_ptr() as *const _,
                        k_i32,
                        &beta,
                        c.as_mut_ptr() as *mut _,
                        m_i32,
                    );
                }
                Ok(())
            }

            fn solve(
                &mut self,
                a: &[$complex_ty],
                b: &[$complex_ty],
                n: usize,
                nrhs: usize,
                x: &mut [$complex_ty],
            ) -> Result<()> {
                check_len("solve", "a", a.len(), n * n)?;
                check_len("solve", "b", b.len(), n * nrhs)?;
                check_len("solve", "x", x.len(), n * nrhs)?;

                let n_i32 = as_i32("solve n", n)?;
                let nrhs_i32 = as_i32("solve nrhs", nrhs)?;

                let mut a_work = a[..n * n].to_vec();
                x[..n * nrhs].copy_from_slice(&b[..n * nrhs]);
                let mut ipiv = vec![0i32; n];
                let mut info = 0;

                unsafe {
                    lapack::$gesv(
                        n_i32,
                        nrhs_i32,
                        &mut a_work,
                        n_i32,
                        &mut ipiv,
                        &mut x[..n * nrhs],
                        n_i32,
                        &mut info,
                    );
                }
                check_info_success("solve", info)
            }

            fn solve_triangular(
                &mut self,
                a: &[$complex_ty],
                b: &[$complex_ty],
                n: usize,
                nrhs: usize,
                upper: bool,
                x: &mut [$complex_ty],
            ) -> Result<()> {
                check_len("solve_triangular", "a", a.len(), n * n)?;
                check_len("solve_triangular", "b", b.len(), n * nrhs)?;
                check_len("solve_triangular", "x", x.len(), n * nrhs)?;

                let n_i32 = as_i32("solve_triangular n", n)?;
                let nrhs_i32 = as_i32("solve_triangular nrhs", nrhs)?;

                x[..n * nrhs].copy_from_slice(&b[..n * nrhs]);
                let mut info = 0;
                let uplo = if upper { b'U' } else { b'L' };

                unsafe {
                    lapack::$trtrs(
                        uplo,
                        b'N',
                        b'N',
                        n_i32,
                        nrhs_i32,
                        &a[..n * n],
                        n_i32,
                        &mut x[..n * nrhs],
                        n_i32,
                        &mut info,
                    );
                }
                check_info_success("solve_triangular", info)
            }

            fn eig_general(
                &mut self,
                a: &[$complex_ty],
                n: usize,
                values_ri: &mut [$complex_ty],
                vectors_ri: &mut [$complex_ty],
            ) -> Result<()> {
                check_len("eig_general", "a", a.len(), n * n)?;
                check_len("eig_general", "values_ri", values_ri.len(), n)?;
                check_len("eig_general", "vectors_ri", vectors_ri.len(), n * n)?;

                if n == 0 {
                    return Ok(());
                }

                let n_i32 = as_i32("eig_general n", n)?;
                let mut a_work = a[..n * n].to_vec();
                let mut w = vec![<$complex_ty>::new(0 as $real_ty, 0 as $real_ty); n];
                let mut vr = vec![<$complex_ty>::new(0 as $real_ty, 0 as $real_ty); n * n];
                let mut vl = vec![<$complex_ty>::new(0 as $real_ty, 0 as $real_ty); 1];
                let mut rwork = vec![0 as $real_ty; (2 * n).max(1)];
                let mut info = 0;

                let mut work_query = [<$complex_ty>::new(0 as $real_ty, 0 as $real_ty); 1];
                unsafe {
                    lapack::$geev(
                        b'N',
                        b'V',
                        n_i32,
                        &mut a_work,
                        n_i32,
                        &mut w,
                        &mut vl,
                        1,
                        &mut vr,
                        n_i32,
                        &mut work_query,
                        -1,
                        &mut rwork,
                        &mut info,
                    );
                }
                check_info_nonnegative("eig_general(work query)", info)?;

                let lwork = $lwork_from_query("eig_general", work_query[0])?;
                let mut work =
                    vec![<$complex_ty>::new(0 as $real_ty, 0 as $real_ty); lwork as usize];
                unsafe {
                    lapack::$geev(
                        b'N',
                        b'V',
                        n_i32,
                        &mut a_work,
                        n_i32,
                        &mut w,
                        &mut vl,
                        1,
                        &mut vr,
                        n_i32,
                        &mut work,
                        lwork,
                        &mut rwork,
                        &mut info,
                    );
                }
                check_info_success("eig_general", info)?;

                values_ri[..n].copy_from_slice(&w[..n]);
                vectors_ri[..n * n].copy_from_slice(&vr[..n * n]);
                Ok(())
            }
        }
    };
}

impl_lapack_backend_real!(
    f64,
    dgesvd,
    dgeqrf,
    dorgqr,
    dgetrf,
    dpotrf,
    dsyev,
    dgesv,
    dtrtrs,
    dgeev,
    cblas_sys::cblas_dgemm,
    lwork_from_query_f64
);

impl_lapack_backend_real!(
    f32,
    sgesvd,
    sgeqrf,
    sorgqr,
    sgetrf,
    spotrf,
    ssyev,
    sgesv,
    strtrs,
    sgeev,
    cblas_sys::cblas_sgemm,
    lwork_from_query_f32
);

impl_lapack_backend_complex!(
    Complex64,
    f64,
    zgesvd,
    zgeqrf,
    zungqr,
    zgetrf,
    zpotrf,
    zheev,
    zgesv,
    ztrtrs,
    zgeev,
    cblas_sys::cblas_zgemm,
    lwork_from_query_c64
);

impl_lapack_backend_complex!(
    Complex32,
    f32,
    cgesvd,
    cgeqrf,
    cungqr,
    cgetrf,
    cpotrf,
    cheev,
    cgesv,
    ctrtrs,
    cgeev,
    cblas_sys::cblas_cgemm,
    lwork_from_query_c32
);

#[cfg(test)]
mod tests;
