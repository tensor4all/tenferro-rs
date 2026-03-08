//! Faer-backed tensor linalg dispatch.

pub(crate) use super::cpu_tensor_impl::{
    cholesky, eig, eigen_sym, lu_factor, qr, solve, solve_triangular, thin_svd,
};
