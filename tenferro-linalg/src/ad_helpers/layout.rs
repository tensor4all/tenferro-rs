use super::*;

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

/// Require a main-memory tensor before entering host-slice algorithms.
pub(crate) fn require_main_memory_tensor<T: LinalgScalar>(
    tensor: &Tensor<T>,
    op: &str,
) -> Result<()> {
    if tensor.logical_memory_space() == tenferro_device::LogicalMemorySpace::MainMemory {
        Ok(())
    } else {
        Err(Error::DeviceError(format!(
            "{op} is only implemented for main-memory tensors"
        )))
    }
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
