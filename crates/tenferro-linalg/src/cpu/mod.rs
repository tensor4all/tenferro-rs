pub(crate) mod backend;
mod linalg;

#[cfg(feature = "native")]
mod tlinalg;

#[cfg(feature = "blas")]
mod tlinalg_blas;

#[cfg(feature = "blas")]
mod tlinalg_workspace;

#[cfg(any(feature = "native", feature = "blas"))]
mod tlinalg_error;

#[cfg(test)]
mod tests;
