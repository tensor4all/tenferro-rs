//! Traced tensor linear algebra operations.
//!
//! This module is the canonical traced tensor namespace for the linalg
//! extension crate.

pub use crate::traced::{
    cholesky, det, eig, eigh, eigh_with_eps, eigvals, eigvalsh, full_piv_lu, full_piv_lu_solve,
    inv, lu, norm, pinv, pinv_with_rtol, qr, slogdet, solve, svd, svd_with_eps, triangular_solve,
};
