use super::*;
use num_traits::Zero;

/// Ensure tensor is column-major contiguous. Returns a contiguous tensor.
pub(crate) fn ensure_col_major<T: LinalgScalar>(tensor: &Tensor<T>) -> Tensor<T> {
    tensor.contiguous(MemoryOrder::ColumnMajor)
}

/// Extract the raw data slice from a tensor.
pub(crate) fn extract_slice<T: LinalgScalar>(tensor: &Tensor<T>) -> Result<&[T]> {
    tensor
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::InvalidArgument("tensor buffer is not a contiguous CPU slice".into()))
}

/// Convert an f64 constant to scalar type `T`.
pub(crate) fn scalar_from<T: LinalgScalar>(val: f64) -> Result<T> {
    T::from(val).ok_or_else(|| {
        Error::InvalidArgument(format!("cannot convert {val} to target scalar type"))
    })
}

/// Convert a device error into an autodiff error.
pub(crate) fn to_ad_err(e: Error) -> chainrules_core::AutodiffError {
    chainrules_core::AutodiffError::InvalidArgument(e.to_string())
}

/// Create a tensor from raw column-major data with the given dims.
pub(crate) fn tensor_from_data<T: LinalgScalar>(data: Vec<T>, dims: &[usize]) -> Result<Tensor<T>> {
    let strides = backend::col_major_strides(dims);
    Tensor::from_vec(data, dims, &strides, 0)
}

/// Create a tensor from raw column-major data with the given dims.
pub(crate) fn tensor_from_data_scalar<T: Scalar>(
    data: Vec<T>,
    dims: &[usize],
) -> Result<Tensor<T>> {
    let strides = backend::col_major_strides(dims);
    Tensor::from_vec(data, dims, &strides, 0)
}

/// Extract owned contiguous data from a tensor.
pub(crate) fn extract_data<T: LinalgScalar>(tensor: &Tensor<T>) -> AdResult<(Vec<T>, usize)> {
    let t = ensure_col_major(tensor);
    let offset = t.offset() as usize;
    let slice = extract_slice(&t).map_err(to_ad_err)?;
    let total_len = tensor.dims().iter().product::<usize>();
    Ok((slice[offset..offset + total_len].to_vec(), 0))
}

/// Extract owned contiguous data from a scalar-backed tensor.
pub(crate) fn extract_data_scalar<T: Scalar>(tensor: &Tensor<T>) -> AdResult<Vec<T>> {
    let t = tensor.contiguous(MemoryOrder::ColumnMajor);
    let offset = t.offset() as usize;
    let slice = t.buffer().as_slice().ok_or_else(|| {
        chainrules_core::AutodiffError::InvalidArgument(
            "tensor buffer is not a contiguous CPU slice".into(),
        )
    })?;
    let total_len: usize = tensor.dims().iter().product();
    Ok(slice[offset..offset + total_len].to_vec())
}

/// Extract a forward permutation vector from a permutation matrix tensor.
pub(crate) fn forward_perm_from_permutation_matrix<T: LinalgScalar>(
    tensor: &Tensor<T>,
    row_count: usize,
    batch_count: usize,
) -> AdResult<Vec<usize>>
where
    T: PartialEq + Zero,
{
    let dims = tensor.dims();
    if dims == [0] {
        let identity: Vec<usize> = (0..row_count).collect();
        let mut out = Vec::with_capacity(row_count * batch_count);
        for _ in 0..batch_count {
            out.extend_from_slice(&identity);
        }
        return Ok(out);
    }

    if dims.len() < 2 {
        return Err(chainrules_core::AutodiffError::InvalidArgument(format!(
            "LU permutation matrix expects rank >= 2 or an empty tensor, got dims={dims:?}"
        )));
    }
    if dims[0] != row_count || dims[1] != row_count {
        return Err(chainrules_core::AutodiffError::InvalidArgument(format!(
            "LU permutation matrix shape mismatch: expected [{row_count}, {row_count}, *], got {dims:?}"
        )));
    }

    let actual_batch_count = if dims.len() <= 2 {
        1
    } else {
        dims[2..].iter().product()
    };
    if actual_batch_count != batch_count {
        return Err(chainrules_core::AutodiffError::InvalidArgument(format!(
            "LU permutation matrix batch count {actual_batch_count} does not match expected {batch_count}"
        )));
    }

    let (data, _) = extract_data(tensor)?;
    let mut out = Vec::with_capacity(row_count * batch_count);
    let batch_stride = row_count * row_count;
    for batch in 0..batch_count {
        let batch_offset = batch * batch_stride;
        for row in 0..row_count {
            let mut found = None;
            for col in 0..row_count {
                let value = data[batch_offset + row + col * row_count];
                if value != T::zero() {
                    found = Some(col);
                    break;
                }
            }
            let col = found.ok_or_else(|| {
                chainrules_core::AutodiffError::InvalidArgument(format!(
                    "LU permutation matrix row {row} in batch {batch} has no nonzero entry"
                ))
            })?;
            out.push(col);
        }
    }
    Ok(out)
}
