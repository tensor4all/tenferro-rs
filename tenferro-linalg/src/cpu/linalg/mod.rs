#[cfg(feature = "cpu-faer")]
pub mod faer_linalg;

#[cfg(feature = "cpu-blas")]
#[cfg_attr(feature = "cpu-faer", allow(dead_code, unused_imports))]
pub mod lapack_linalg;

#[cfg(feature = "cpu-faer")]
pub(crate) use faer_linalg as faer;
#[cfg(feature = "cpu-blas")]
pub(crate) use lapack_linalg as blas;
