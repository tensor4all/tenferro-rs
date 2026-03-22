//! Backend-owned linalg context bindings and marker types.
//!
//! This module is the long-term home of the tensor-level linalg ownership
//! boundary. High-level crates should depend on these bindings instead of
//! owning backend markers themselves.

mod context;
pub(crate) mod linalg_utils;
mod tensor_api;
mod tensor_helpers;

#[cfg(feature = "linalg-lapack")]
#[path = "../../../tenferro-linalg/src/backend/blas_lapack_backend/mod.rs"]
mod blas_lapack_backend;
mod cpu;
#[cfg(feature = "linalg-faer")]
#[path = "../../../tenferro-linalg/src/backend/cpu_faer.rs"]
mod cpu_faer;
#[cfg(feature = "linalg-lapack")]
#[path = "../../../tenferro-linalg/src/backend/cpu_lapack.rs"]
mod cpu_lapack;
#[path = "../../../tenferro-linalg/src/backend/cpu_tensor_impl.rs"]
mod cpu_tensor_impl;
mod cuda;
#[cfg(feature = "linalg-faer")]
mod faer_backend;
mod hip;

use tenferro_device::Result;

#[cfg(feature = "linalg-lapack")]
pub use blas_lapack_backend::BlasLapackBackend;
pub use context::TensorLinalgContextFor;
pub use cpu::CpuTensorLinalgBackend;
pub use cuda::{CudaDataType, CudaLinalgScalar, CudaTensorLinalgBackend};
#[cfg(feature = "linalg-faer")]
pub use faer_backend::FaerBackend;
pub use hip::HipTensorLinalgBackend;

#[doc(hidden)]
pub use tensor_helpers::{
    batch_count, broadcast_batch_dims, ensure_col_major, extract_contiguous_slice,
    materialize_broadcasted_batches, validate_matrix_shape, validate_solve_rhs_shape,
    validate_square, zero_trailing_by_counts, BroadcastBatchIndexer, SolveRhsLayout,
};

/// Slice-level backend interface for matrix linear algebra operations.
#[allow(dead_code)]
pub(crate) trait LinalgBackend<T: Copy + 'static> {
    type Real: Copy + 'static;

    fn thin_svd(
        &mut self,
        a: &[T],
        m: usize,
        n: usize,
        u: &mut [T],
        s: &mut [Self::Real],
        vt: &mut [T],
    ) -> Result<()>;

    fn qr(&mut self, a: &[T], m: usize, n: usize, q: &mut [T], r: &mut [T]) -> Result<()>;

    fn lu(
        &mut self,
        a: &[T],
        m: usize,
        n: usize,
        perm: &mut [usize],
        l: &mut [T],
        u_out: &mut [T],
    ) -> Result<()>;

    fn cholesky(&mut self, a: &[T], n: usize, l: &mut [T]) -> Result<()>;

    fn eigen_sym(
        &mut self,
        a: &[T],
        n: usize,
        values: &mut [Self::Real],
        vectors: &mut [T],
    ) -> Result<()>;

    fn mat_mul(
        &mut self,
        a: &[T],
        m: usize,
        k: usize,
        b: &[T],
        n: usize,
        c: &mut [T],
    ) -> Result<()>;

    fn solve(&mut self, a: &[T], b: &[T], n: usize, nrhs: usize, x: &mut [T]) -> Result<()>;

    fn solve_triangular(
        &mut self,
        a: &[T],
        b: &[T],
        n: usize,
        nrhs: usize,
        upper: bool,
        x: &mut [T],
    ) -> Result<()>;

    fn eig_general(
        &mut self,
        a: &[T],
        n: usize,
        values_ri: &mut [T],
        vectors_ri: &mut [T],
    ) -> Result<()>;
}

/// Compute column-major strides for the provided dimensions.
#[doc(hidden)]
pub fn col_major_strides(dims: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; dims.len()];
    if dims.is_empty() {
        return strides;
    }
    strides[0] = 1;
    for i in 1..dims.len() {
        strides[i] = strides[i - 1] * dims[i - 1] as isize;
    }
    strides
}

#[cfg(test)]
mod tests;
