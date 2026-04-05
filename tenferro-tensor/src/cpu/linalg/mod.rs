#[cfg(feature = "cpu-faer")]
pub mod faer_linalg;

#[cfg(feature = "cpu-blas")]
pub mod lapack_linalg;

#[cfg(feature = "cpu-faer")]
pub(crate) use faer_linalg::{cholesky, eigh, qr, solve, svd};

#[cfg(feature = "cpu-blas")]
pub(crate) use lapack_linalg::{cholesky, eigh, qr, solve, svd};
