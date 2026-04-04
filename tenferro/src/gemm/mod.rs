//! GEMM dispatch for the v2 CPU backend.
//!
//! Provides zero-copy strided GEMM via faer (default) or contiguous
//! GEMM via CBLAS, selected at compile time through feature flags.

#[cfg(feature = "cpu-faer")]
pub(crate) mod faer_gemm;

#[cfg(feature = "cpu-blas")]
pub(crate) mod blas_gemm;
