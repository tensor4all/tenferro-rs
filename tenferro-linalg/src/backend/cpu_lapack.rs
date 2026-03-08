//! LAPACK-backed tensor linalg dispatch.

pub(crate) type LapackBackend = super::blas_lapack_backend::BlasLapackBackend;

pub(crate) use super::cpu_tensor_impl::{
    cholesky, eig, eigen_sym, lu_factor, qr, solve, solve_triangular, thin_svd,
};
