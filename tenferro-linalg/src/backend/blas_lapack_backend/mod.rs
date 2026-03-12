//! BLAS/LAPACK backend for linear algebra operations.
//!
//! This backend implements [`super::LinalgBackend`] for
//! `f32`, `f64`, `Complex32`, and `Complex64` using:
//! - LAPACK wrappers from the [`lapack`](https://crates.io/crates/lapack) crate
//! - CBLAS symbols from the [`cblas-sys`](https://crates.io/crates/cblas-sys) crate
//!
//! Symbol providers are selected at compile time via crate features
//! (`provider-src` / `provider-inject` and `src-*`).

use super::LinalgBackend;
use num_complex::{Complex32, Complex64};
use tenferro_device::Result;

mod complex;
mod helpers;
mod real;

pub(crate) use helpers::{
    as_i32, check_info_cholesky, check_info_nonnegative, check_info_success, check_len,
    fill_zero_upper, lwork_from_query_c32, lwork_from_query_c64, lwork_from_query_f32,
    lwork_from_query_f64, pivots_to_forward_perm, split_lu, write_real_eig_general_output,
};

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

impl LinalgBackend<f64> for BlasLapackBackend {
    type Real = f64;

    real::decompositions::impl_real_decompositions!(
        f64,
        dgesvd,
        dgeqrf,
        dorgqr,
        dgetrf,
        dpotrf,
        lwork_from_query_f64
    );
    real::linear_systems::impl_real_linear_systems!(f64, dgesv, dtrtrs, cblas_sys::cblas_dgemm);
    real::spectral::impl_real_spectral!(f64, dsyev, dgeev, lwork_from_query_f64);
}

impl LinalgBackend<f32> for BlasLapackBackend {
    type Real = f32;

    real::decompositions::impl_real_decompositions!(
        f32,
        sgesvd,
        sgeqrf,
        sorgqr,
        sgetrf,
        spotrf,
        lwork_from_query_f32
    );
    real::linear_systems::impl_real_linear_systems!(f32, sgesv, strtrs, cblas_sys::cblas_sgemm);
    real::spectral::impl_real_spectral!(f32, ssyev, sgeev, lwork_from_query_f32);
}

impl LinalgBackend<Complex64> for BlasLapackBackend {
    type Real = f64;

    complex::decompositions::impl_complex_decompositions!(
        Complex64,
        f64,
        zgesvd,
        zgeqrf,
        zungqr,
        zgetrf,
        zpotrf,
        lwork_from_query_c64
    );
    complex::linear_systems::impl_complex_linear_systems!(
        Complex64,
        f64,
        zgesv,
        ztrtrs,
        cblas_sys::cblas_zgemm
    );
    complex::spectral::impl_complex_spectral!(Complex64, f64, zheev, zgeev, lwork_from_query_c64);
}

impl LinalgBackend<Complex32> for BlasLapackBackend {
    type Real = f32;

    complex::decompositions::impl_complex_decompositions!(
        Complex32,
        f32,
        cgesvd,
        cgeqrf,
        cungqr,
        cgetrf,
        cpotrf,
        lwork_from_query_c32
    );
    complex::linear_systems::impl_complex_linear_systems!(
        Complex32,
        f32,
        cgesv,
        ctrtrs,
        cblas_sys::cblas_cgemm
    );
    complex::spectral::impl_complex_spectral!(Complex32, f32, cheev, cgeev, lwork_from_query_c32);
}

#[cfg(test)]
mod tests;
