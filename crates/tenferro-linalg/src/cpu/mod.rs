pub(crate) mod backend;
mod linalg;

// Both CPU builds: the `blas` build also takes complete-pivoting LU from tlinalg.
#[cfg_attr(not(feature = "native"), allow(dead_code, unused_imports))]
mod tlinalg;

#[cfg(feature = "blas")]
mod tlinalg_blas;

#[cfg(feature = "blas")]
mod tlinalg_workspace;

#[cfg(any(feature = "native", feature = "blas"))]
mod tlinalg_error;

#[cfg(test)]
mod tests;
