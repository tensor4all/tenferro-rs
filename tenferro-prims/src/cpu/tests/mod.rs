mod organization;

#[cfg(feature = "gemm-faer")]
mod gemm_support;

#[cfg(feature = "gemm-blas")]
mod scratch_pool;
