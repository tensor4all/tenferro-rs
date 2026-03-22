//! Faer backend for linear algebra operations.

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

/// Pure-Rust linear algebra backend powered by faer.
#[derive(Debug, Clone)]
pub struct FaerBackend;

impl FaerBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for FaerBackend {
    fn default() -> Self {
        Self::new()
    }
}

real::upstream::impl_linalg_backend!(f64);
real::upstream::impl_linalg_backend!(f32);
complex::upstream::impl_complex_linalg_backend!(Complex64, f64, to_faer_c64, from_faer_c64_mat);
complex::upstream::impl_complex_linalg_backend!(Complex32, f32, to_faer_c32, from_faer_c32_mat);
