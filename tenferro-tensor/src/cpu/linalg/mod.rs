#[cfg(feature = "cpu-faer")]
pub mod faer_linalg;

#[cfg(feature = "cpu-blas")]
pub mod lapack_linalg;
