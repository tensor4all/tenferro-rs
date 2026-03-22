//! Torch-aligned tensor layout helpers for linalg backends.

use tenferro_algebra::{Conjugate, Standard};
use tenferro_device::{Error, Result};
use tenferro_prims::{
    SemiringCoreDescriptor, TensorResolveConjContextFor, TensorSemiringContextFor,
    TensorSemiringCore,
};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::LinalgScalar;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MatrixOperandTransform {
    None,
    TransposeFirstTwoAxes,
}

fn transpose_first_two_axes<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: LinalgScalar,
{
    if tensor.ndim() < 2 {
        return Err(Error::InvalidArgument(format!(
            "TransposeFirstTwoAxes expects at least 2D tensor, got {}D",
            tensor.ndim()
        )));
    }
    let mut perm: Vec<usize> = (0..tensor.ndim()).collect();
    perm.swap(0, 1);
    tensor.permute(&perm)
}

pub(crate) fn prepare_matrix_operand<T, C>(
    ctx: &mut C,
    src: &Tensor<T>,
    transform: MatrixOperandTransform,
) -> Result<Tensor<T>>
where
    T: LinalgScalar + Conjugate,
    C: TensorSemiringContextFor<Standard<T>> + TensorResolveConjContextFor<T>,
{
    let transformed = match transform {
        MatrixOperandTransform::None => src.clone(),
        MatrixOperandTransform::TransposeFirstTwoAxes => transpose_first_two_axes(src)?,
    };
    let resolved = <C as TensorResolveConjContextFor<T>>::resolve_conj(ctx, &transformed);
    clone_batched_column_major(ctx, &resolved)
}

pub(crate) fn matrix_stride(dims: &[usize]) -> Result<usize> {
    if dims.len() < 2 {
        return Err(Error::InvalidArgument(format!(
            "matrix_stride expects at least 2D shape, got {:?}",
            dims
        )));
    }

    dims[0]
        .checked_mul(dims[1])
        .ok_or_else(|| Error::InvalidArgument(format!("matrix stride overflow for dims {dims:?}")))
}

pub(crate) fn clone_batched_column_major<T, C>(ctx: &mut C, src: &Tensor<T>) -> Result<Tensor<T>>
where
    T: LinalgScalar,
    C: TensorSemiringContextFor<Standard<T>>,
{
    if src.is_col_major_contiguous() && src.offset() == 0 && !src.is_conjugated() {
        return Ok(src.clone());
    }

    let mut output = Tensor::zeros(
        src.dims(),
        src.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    );
    output.set_preferred_compute_device(src.preferred_compute_device());
    copy_batched_column_major(ctx, src, &mut output)?;
    Ok(output)
}

pub(crate) fn copy_batched_column_major<T, C>(
    ctx: &mut C,
    src: &Tensor<T>,
    dst: &mut Tensor<T>,
) -> Result<()>
where
    T: LinalgScalar,
    C: TensorSemiringContextFor<Standard<T>>,
{
    if src.dims() != dst.dims() {
        return Err(Error::ShapeMismatch {
            expected: src.dims().to_vec(),
            got: dst.dims().to_vec(),
        });
    }
    if src.logical_memory_space() != dst.logical_memory_space() {
        return Err(Error::InvalidArgument(format!(
            "copy_batched_column_major requires matching memory spaces, got {:?} and {:?}",
            src.logical_memory_space(),
            dst.logical_memory_space()
        )));
    }
    if src.is_conjugated() || dst.is_conjugated() {
        return Err(Error::InvalidArgument(
            "copy_batched_column_major requires resolved (non-conjugated) tensors".into(),
        ));
    }

    let desc = SemiringCoreDescriptor::MakeContiguous;
    let plan = <C::SemiringBackend as TensorSemiringCore<Standard<T>>>::plan(
        ctx,
        &desc,
        &[src.dims(), dst.dims()],
    )?;
    <C::SemiringBackend as TensorSemiringCore<Standard<T>>>::execute(
        ctx,
        &plan,
        T::one(),
        &[src],
        T::zero(),
        dst,
    )
}
