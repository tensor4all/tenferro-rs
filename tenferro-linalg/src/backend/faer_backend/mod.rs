//! Faer backend for linear algebra operations.
//!
//! This module provides the [`FaerBackend`] struct implementing
//! [`LinalgBackend`] for `f64`, `f32`, `Complex64`, and `Complex32`.

use faer::linalg::solvers::Solve;
use num_complex::{Complex32, Complex64};
use tenferro_device::{Error, Result};

use super::LinalgBackend;

mod complex;
mod conversion;
mod helpers;
mod real;

use conversion::{from_faer_c32_mat, from_faer_c64_mat, to_faer_c32, to_faer_c64};

pub(crate) use helpers::{
    check_len, complex_is_finite, non_finite_result_error, singular_matrix_error,
    zero_diagonal_error,
};

/// Pure-Rust linear algebra backend powered by [faer](https://crates.io/crates/faer).
///
/// This struct is stateless; `&mut self` is accepted for future workspace reuse.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::backend::{FaerBackend, LinalgBackend};
///
/// let mut backend = FaerBackend::new();
/// let a = [1.0_f64, 0.0, 0.0, 1.0]; // 2x2 identity, col-major
/// let mut u = [0.0; 4];
/// let mut s = [0.0; 2];
/// let mut vt = [0.0; 4];
/// backend.thin_svd(&a, 2, 2, &mut u, &mut s, &mut vt).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct FaerBackend;

impl FaerBackend {
    /// Create a new `FaerBackend`.
    pub fn new() -> Self {
        Self
    }
}

impl Default for FaerBackend {
    fn default() -> Self {
        Self::new()
    }
}

real::impl_linalg_backend!(f64);
real::impl_linalg_backend!(f32);
complex::impl_complex_linalg_backend!(Complex64, f64, to_faer_c64, from_faer_c64_mat);
complex::impl_complex_linalg_backend!(Complex32, f32, to_faer_c32, from_faer_c32_mat);

#[cfg(test)]
mod tests;
