use tenferro_device::{Error, Result};
use tenferro_tensor::{MemoryOrder, Tensor};

use super::TropicalScalar;

/// Promote a standard-real tensor to a tropical scalar tensor.
pub fn promote_to_tropical<T: TropicalScalar>(tensor: &Tensor<T::Inner>) -> Result<Tensor<T>> {
    tensor
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?;
    let contiguous = tensor.contiguous(MemoryOrder::ColumnMajor);
    let data = contiguous.buffer().as_slice().ok_or_else(|| {
        Error::DeviceError("tensor materialization produced a non-CPU buffer".into())
    })?;
    let tropical_data: Vec<T> = data.iter().map(|&v| T::from_inner(v)).collect();
    Tensor::<T>::from_slice(&tropical_data, tensor.dims(), MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("{e}")))
}

/// Extract the inner real values from a tropical tensor.
pub fn extract_inner<T: TropicalScalar>(tensor: &Tensor<T>) -> Result<Tensor<T::Inner>> {
    tensor
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?;
    let contiguous = tensor.contiguous(MemoryOrder::ColumnMajor);
    let data = contiguous.buffer().as_slice().ok_or_else(|| {
        Error::DeviceError("tensor materialization produced a non-CPU buffer".into())
    })?;
    let inner_data: Vec<T::Inner> = data.iter().map(|value| value.inner()).collect();
    Tensor::<T::Inner>::from_slice(&inner_data, tensor.dims(), MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("{e}")))
}
