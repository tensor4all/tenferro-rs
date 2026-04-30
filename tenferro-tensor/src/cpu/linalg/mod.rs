#[cfg(feature = "cpu-faer")]
pub mod faer_linalg;

#[cfg(feature = "cpu-blas")]
pub mod lapack_linalg;

#[cfg(feature = "cpu-faer")]
pub(crate) use faer_linalg::{cholesky, eig, eigh, lu, qr, svd, triangular_solve};
#[cfg(feature = "cpu-faer")]
pub(crate) use faer_linalg::{full_piv_lu, full_piv_lu_solve};

#[cfg(feature = "cpu-blas")]
pub(crate) use lapack_linalg::{cholesky, eig, eigh, lu, qr, svd, triangular_solve};
#[cfg(feature = "cpu-blas")]
pub(crate) use lapack_linalg::{full_piv_lu, full_piv_lu_solve};
