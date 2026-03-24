use tenferro_algebra::Conjugate;
use tenferro_prims::TensorResolveConjContextFor;

use super::*;

fn swap_leading_matrix_axes_perm(ndim: usize) -> Result<Vec<usize>> {
    if ndim < 2 {
        return Err(Error::InvalidArgument(format!(
            "expected at least 2 dimensions for matrix transpose, got {ndim}"
        )));
    }

    let mut perm: Vec<usize> = (0..ndim).collect();
    perm.swap(0, 1);
    Ok(perm)
}

pub(crate) fn matrix_transpose<T: LinalgScalar>(tensor: &Tensor<T>) -> Result<Tensor<T>> {
    tensor.permute(&swap_leading_matrix_axes_perm(tensor.ndim())?)
}

/// Eagerly compute the conjugate transpose of a matrix tensor.
///
/// Swaps the leading two axes and resolves any lazy conjugation so the result
/// is safe to pass to backend operations that reject conjugated inputs.
pub(crate) fn matrix_adjoint_eager<T, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: LinalgScalar + Conjugate,
    C: TensorResolveConjContextFor<T>,
{
    let transposed = matrix_transpose(tensor)?;
    Ok(prims_bridge::resolve_conj(ctx, &transposed.conj()))
}

pub(crate) fn rhs_to_matrix<T: LinalgScalar>(
    tensor: &Tensor<T>,
    structural_rank: usize,
) -> Result<Tensor<T>> {
    match structural_rank {
        1 => tensor.unsqueeze(1),
        2 => Ok(tensor.clone()),
        _ => Err(Error::InvalidArgument(format!(
            "rhs_to_matrix expects structural_rank 1 or 2, got {structural_rank}"
        ))),
    }
}

pub(crate) fn matrix_to_rhs<T: LinalgScalar>(
    tensor: Tensor<T>,
    structural_rank: usize,
) -> Result<Tensor<T>> {
    match structural_rank {
        1 => tensor.squeeze_dim(1),
        2 => Ok(tensor),
        _ => Err(Error::InvalidArgument(format!(
            "matrix_to_rhs expects structural_rank 1 or 2, got {structural_rank}"
        ))),
    }
}
