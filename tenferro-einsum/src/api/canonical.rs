use tenferro_algebra::{Conjugate, Scalar};
use tenferro_tensor::{MemoryOrder, Tensor};

/// Materialize a tensor as a col-major contiguous buffer, applying conjugation if needed.
///
/// Returns `None` if the tensor is already in canonical form (col-major, zero offset,
/// unconjugated), allowing callers to avoid an allocation in the fast path.
fn try_make_col_major_data<T: Scalar + Conjugate>(tensor: &Tensor<T>) -> Option<Vec<T>> {
    let contiguous = tensor.contiguous(MemoryOrder::ColumnMajor);
    let offset = usize::try_from(contiguous.offset()).unwrap_or(0);
    let len = contiguous.len();
    contiguous
        .buffer()
        .as_slice()
        .and_then(|values| values.get(offset..offset + len))
        .map(|values| {
            if contiguous.is_conjugated() {
                values.iter().copied().map(Conjugate::conj).collect()
            } else {
                values.to_vec()
            }
        })
}

pub(super) fn materialize_logical_col_major<T: Scalar + Conjugate>(
    tensor: &Tensor<T>,
) -> Tensor<T> {
    if tensor.is_col_major_contiguous() && tensor.offset() == 0 && !tensor.is_conjugated() {
        return tensor.clone();
    }
    try_make_col_major_data(tensor)
        .and_then(|values| {
            Tensor::from_slice(&values, tensor.dims(), MemoryOrder::ColumnMajor).ok()
        })
        .unwrap_or_else(|| tensor.clone().into_contiguous(MemoryOrder::ColumnMajor))
}

fn into_logical_col_major<T: Scalar + Conjugate>(tensor: Tensor<T>) -> Tensor<T> {
    if tensor.is_col_major_contiguous() && tensor.offset() == 0 && !tensor.is_conjugated() {
        return tensor;
    }
    let dims = tensor.dims().to_vec();
    try_make_col_major_data(&tensor)
        .and_then(|values| Tensor::from_slice(&values, &dims, MemoryOrder::ColumnMajor).ok())
        .unwrap_or_else(|| tensor.into_contiguous(MemoryOrder::ColumnMajor))
}

pub(super) fn canonicalize_col_major_operands<T: Scalar + Conjugate>(
    operands: &[&Tensor<T>],
) -> Vec<Tensor<T>> {
    operands
        .iter()
        .map(|tensor| materialize_logical_col_major(tensor))
        .collect()
}

pub(super) fn canonicalize_col_major_operands_owned<T: Scalar + Conjugate>(
    operands: Vec<Tensor<T>>,
) -> Vec<Tensor<T>> {
    operands.into_iter().map(into_logical_col_major).collect()
}
