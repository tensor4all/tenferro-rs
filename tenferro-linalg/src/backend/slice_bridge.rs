//! Generic slice-to-tensor bridge helpers for linalg backends.
//!
//! These helpers keep composite and AD code generic over runtime contexts
//! without reintroducing CPU-only slice calls in higher layers.

use tenferro_device::{Error, Result};
use tenferro_tensor::{MemoryOrder, Tensor};

use super::{TensorLinalgBackend, TensorLinalgContextFor};
use crate::KernelLinalgScalar;

fn tensor_from_col_major_slice<T: KernelLinalgScalar>(
    data: &[T],
    dims: &[usize],
) -> Result<Tensor<T>> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor)
}

fn tensor_to_col_major_vec<T: KernelLinalgScalar>(tensor: Tensor<T>) -> Result<Vec<T>> {
    let contiguous = tensor.into_contiguous(MemoryOrder::ColumnMajor);
    let offset = contiguous.offset() as usize;
    let len = contiguous.len();
    let slice = contiguous
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::DeviceError("expected CPU-accessible contiguous tensor".into()))?;
    Ok(slice[offset..offset + len].to_vec())
}

fn tensor_to_col_major_vec_i32(tensor: Tensor<i32>) -> Result<Vec<i32>> {
    let contiguous = tensor.into_contiguous(MemoryOrder::ColumnMajor);
    let offset = contiguous.offset() as usize;
    let len = contiguous.len();
    let slice = contiguous
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::DeviceError("expected CPU-accessible contiguous tensor".into()))?;
    Ok(slice[offset..offset + len].to_vec())
}

pub(crate) fn solve_vec<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &[T],
    b: &[T],
    n: usize,
    nrhs: usize,
) -> Result<Vec<T>>
where
    C: TensorLinalgContextFor<T>,
{
    if nrhs == 0 {
        return Ok(Vec::new());
    }
    let rhs_dims = if nrhs == 1 { vec![n] } else { vec![n, nrhs] };
    let a_tensor = tensor_from_col_major_slice(a, &[n, n])?;
    let b_tensor = tensor_from_col_major_slice(b, &rhs_dims)?;
    let x = <C::Backend as TensorLinalgBackend<T>>::solve(ctx, &a_tensor, &b_tensor)?;
    tensor_to_col_major_vec(x)
}

pub(crate) fn solve_triangular_vec<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &[T],
    b: &[T],
    n: usize,
    nrhs: usize,
    upper: bool,
) -> Result<Vec<T>>
where
    C: TensorLinalgContextFor<T>,
{
    if nrhs == 0 {
        return Ok(Vec::new());
    }
    let rhs_dims = if nrhs == 1 { vec![n] } else { vec![n, nrhs] };
    let a_tensor = tensor_from_col_major_slice(a, &[n, n])?;
    let b_tensor = tensor_from_col_major_slice(b, &rhs_dims)?;
    let x =
        <C::Backend as TensorLinalgBackend<T>>::solve_triangular(ctx, &a_tensor, &b_tensor, upper)?;
    tensor_to_col_major_vec(x)
}

pub(crate) fn qr_vec<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &[T],
    m: usize,
    n: usize,
) -> Result<(Vec<T>, Vec<T>)>
where
    C: TensorLinalgContextFor<T>,
{
    let a_tensor = tensor_from_col_major_slice(a, &[m, n])?;
    let result = <C::Backend as TensorLinalgBackend<T>>::qr(ctx, &a_tensor)?;
    Ok((
        tensor_to_col_major_vec(result.q)?,
        tensor_to_col_major_vec(result.r)?,
    ))
}

pub(crate) fn thin_svd_vec<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    a: &[T],
    m: usize,
    n: usize,
) -> Result<(Vec<T>, Vec<T>, Vec<T>)>
where
    C: TensorLinalgContextFor<T>,
{
    let a_tensor = tensor_from_col_major_slice(a, &[m, n])?;
    let result = <C::Backend as TensorLinalgBackend<T>>::thin_svd(ctx, &a_tensor)?;
    Ok((
        tensor_to_col_major_vec(result.u)?,
        tensor_to_col_major_vec(result.s)?,
        tensor_to_col_major_vec(result.vt)?,
    ))
}

pub(crate) fn lu_factor_vec<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &[T],
    m: usize,
    n: usize,
) -> Result<(Vec<i32>, Vec<T>, Vec<T>)>
where
    C: TensorLinalgContextFor<T>,
{
    let a_tensor = tensor_from_col_major_slice(a, &[m, n])?;
    let result = <C::Backend as TensorLinalgBackend<T>>::lu_factor(ctx, &a_tensor)?;
    Ok((
        tensor_to_col_major_vec_i32(result.pivots)?,
        tensor_to_col_major_vec(result.l)?,
        tensor_to_col_major_vec(result.u)?,
    ))
}

pub(crate) fn cholesky_vec<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &[T],
    n: usize,
) -> Result<Vec<T>>
where
    C: TensorLinalgContextFor<T>,
{
    let a_tensor = tensor_from_col_major_slice(a, &[n, n])?;
    let result = <C::Backend as TensorLinalgBackend<T>>::cholesky(ctx, &a_tensor)?;
    tensor_to_col_major_vec(result)
}

#[cfg(test)]
mod tests;
