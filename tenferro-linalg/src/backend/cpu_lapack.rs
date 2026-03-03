//! LAPACK-backed CPU implementation of tensor linalg operations.
//!
//! This module is a placeholder for the `linalg-lapack` feature.
//! When `linalg-lapack` is enabled, it will provide CPU linalg
//! operations backed by an external LAPACK library.

use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use super::tensor_api::{
    EigTensorResult, EigenTensorResult, LuTensorResult, QrTensorResult, SvdTensorResult,
};
use super::LinalgBackend;
use crate::LinalgScalar;

/// Placeholder LAPACK backend that compiles but returns a stable error.
pub struct LapackBackend;

impl LapackBackend {
    pub fn new() -> Self {
        Self
    }
}

fn unsupported<T>() -> Result<T> {
    Err(Error::DeviceError(
        "CPU linalg-lapack backend is not yet implemented".into(),
    ))
}

impl<T: LinalgScalar> LinalgBackend<T> for LapackBackend {
    type Real = T::Real;

    fn thin_svd(
        &mut self,
        _a: &[T],
        _m: usize,
        _n: usize,
        _u: &mut [T],
        _s: &mut [Self::Real],
        _vt: &mut [T],
    ) -> Result<()> {
        unsupported()
    }

    fn qr(&mut self, _a: &[T], _m: usize, _n: usize, _q: &mut [T], _r: &mut [T]) -> Result<()> {
        unsupported()
    }

    fn lu(
        &mut self,
        _a: &[T],
        _m: usize,
        _n: usize,
        _perm: &mut [usize],
        _l: &mut [T],
        _u_out: &mut [T],
    ) -> Result<()> {
        unsupported()
    }

    fn cholesky(&mut self, _a: &[T], _n: usize, _l: &mut [T]) -> Result<()> {
        unsupported()
    }

    fn eigen_sym(
        &mut self,
        _a: &[T],
        _n: usize,
        _values: &mut [Self::Real],
        _vectors: &mut [T],
    ) -> Result<()> {
        unsupported()
    }

    fn mat_mul(
        &mut self,
        _a: &[T],
        _m: usize,
        _k: usize,
        _b: &[T],
        _n: usize,
        _c: &mut [T],
    ) -> Result<()> {
        unsupported()
    }

    fn solve(&mut self, _a: &[T], _b: &[T], _n: usize, _nrhs: usize, _x: &mut [T]) -> Result<()> {
        unsupported()
    }

    fn solve_triangular(
        &mut self,
        _a: &[T],
        _b: &[T],
        _n: usize,
        _nrhs: usize,
        _upper: bool,
        _x: &mut [T],
    ) -> Result<()> {
        unsupported()
    }

    fn eig_general(
        &mut self,
        _a: &[T],
        _n: usize,
        _values_ri: &mut [T],
        _vectors_ri: &mut [T],
    ) -> Result<()> {
        unsupported()
    }
}

pub(crate) fn solve<T: LinalgScalar>(
    _ctx: &mut tenferro_prims::CpuContext,
    _a: &Tensor<T>,
    _b: &Tensor<T>,
) -> Result<Tensor<T>> {
    unsupported()
}

pub(crate) fn solve_triangular<T: LinalgScalar>(
    _ctx: &mut tenferro_prims::CpuContext,
    _a: &Tensor<T>,
    _b: &Tensor<T>,
    _upper: bool,
) -> Result<Tensor<T>> {
    unsupported()
}

pub(crate) fn qr<T: LinalgScalar>(
    _ctx: &mut tenferro_prims::CpuContext,
    _a: &Tensor<T>,
) -> Result<QrTensorResult<T>> {
    unsupported()
}

pub(crate) fn thin_svd<T: LinalgScalar>(
    _ctx: &mut tenferro_prims::CpuContext,
    _a: &Tensor<T>,
) -> Result<SvdTensorResult<T>> {
    unsupported()
}

pub(crate) fn lu_factor<T: LinalgScalar>(
    _ctx: &mut tenferro_prims::CpuContext,
    _a: &Tensor<T>,
) -> Result<LuTensorResult<T>> {
    unsupported()
}

pub(crate) fn cholesky<T: LinalgScalar>(
    _ctx: &mut tenferro_prims::CpuContext,
    _a: &Tensor<T>,
) -> Result<Tensor<T>> {
    unsupported()
}

pub(crate) fn eigen_sym<T: LinalgScalar>(
    _ctx: &mut tenferro_prims::CpuContext,
    _a: &Tensor<T>,
) -> Result<EigenTensorResult<T>> {
    unsupported()
}

pub(crate) fn eig<T: LinalgScalar>(
    _ctx: &mut tenferro_prims::CpuContext,
    _a: &Tensor<T>,
) -> Result<EigTensorResult<T>> {
    unsupported()
}
