//! Linear algebra extension operations for tenferro.
//!
//! This crate owns the graph-facing linalg op payloads and runtime
//! registration. Traced helpers live under `tenferro_linalg::traced_tensor`.
//! Backend kernels remain in `tenferro-internal-tensor`.

#[cfg(feature = "autodiff")]
mod ad;
#[cfg(feature = "autodiff")]
pub mod eager_tensor;
mod extension;
mod traced;
pub mod traced_tensor;

pub use extension::{register_runtime, LINALG_EXTENSION_FAMILY_ID};
pub use traced::{
    cholesky, det, eig, eigh, eigh_with_eps, eigvals, eigvalsh, full_piv_lu, full_piv_lu_solve,
    inv, lu, norm, pinv, pinv_with_rtol, qr, slogdet, solve, svd, svd_with_eps, triangular_solve,
};
