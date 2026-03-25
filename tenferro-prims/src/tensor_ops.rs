//! GPU-generic free functions for tensor data operations.
//!
//! These functions expose `triu`, `tril`, `cat`, and `stack` as context-bearing
//! free functions in `tenferro-prims`, which is the correct layer for
//! GPU-dispatch-aware tensor operations.
//!
//! **Design note**: `tenferro-tensor` does NOT depend on `tenferro-prims`
//! (the dependency goes the other way), so these must be free functions here,
//! not methods on `Tensor`.
//!
//! **Current implementation**: Delegates to existing `Tensor` methods for CPU
//! (the context parameter is unused in the current CPU path). When GPU backends
//! are wired, these functions will dispatch through the context to the
//! appropriate backend kernels.
//!
//! # Examples
//!
//! ```ignore
//! use tenferro_device::LogicalMemorySpace;
//! use tenferro_prims::{tensor_ops, CpuContext};
//! use tenferro_tensor::{MemoryOrder, Tensor};
//!
//! let mut ctx = CpuContext::new(1);
//! let a = Tensor::<f64>::ones(
//!     &[3, 3],
//!     LogicalMemorySpace::MainMemory,
//!     MemoryOrder::ColumnMajor,
//! ).unwrap();
//!
//! let upper = tensor_ops::triu(&mut ctx, &a, 0).unwrap();
//! let lower = tensor_ops::tril(&mut ctx, &a, 0).unwrap();
//! ```

use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::TensorMetadataContextFor;

/// Extract the upper triangular part of a matrix (or batch of matrices).
///
/// Elements below the `diagonal`-th diagonal are set to zero. The context
/// parameter enables future GPU-backend dispatch.
///
/// The `diagonal` parameter controls which diagonal is the boundary:
/// - `0` is the main diagonal
/// - positive values move above the main diagonal
/// - negative values move below the main diagonal
///
/// # Errors
///
/// Returns an error if the tensor has fewer than 2 dimensions or if backend
/// dispatch fails.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_prims::{tensor_ops, CpuContext};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::ones(
///     &[3, 3],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// ).unwrap();
/// let upper = tensor_ops::triu(&mut ctx, &a, 0).unwrap();
/// assert_eq!(upper.dims(), &[3, 3]);
/// ```
pub fn triu<T, C>(_ctx: &mut C, tensor: &Tensor<T>, diagonal: isize) -> Result<Tensor<T>>
where
    T: Scalar,
    C: TensorMetadataContextFor,
{
    if tensor.ndim() < 2 {
        return Err(Error::InvalidArgument(format!(
            "triu requires at least 2 dimensions (got {})",
            tensor.ndim()
        )));
    }
    Ok(tensor.triu(diagonal))
}

/// Extract the lower triangular part of a matrix (or batch of matrices).
///
/// Elements above the `diagonal`-th diagonal are set to zero. The context
/// parameter enables future GPU-backend dispatch.
///
/// The `diagonal` parameter controls which diagonal is the boundary:
/// - `0` is the main diagonal
/// - positive values move above the main diagonal
/// - negative values move below the main diagonal
///
/// # Errors
///
/// Returns an error if the tensor has fewer than 2 dimensions or if backend
/// dispatch fails.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_prims::{tensor_ops, CpuContext};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::ones(
///     &[3, 3],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// ).unwrap();
/// let lower = tensor_ops::tril(&mut ctx, &a, 0).unwrap();
/// assert_eq!(lower.dims(), &[3, 3]);
/// ```
pub fn tril<T, C>(_ctx: &mut C, tensor: &Tensor<T>, diagonal: isize) -> Result<Tensor<T>>
where
    T: Scalar,
    C: TensorMetadataContextFor,
{
    if tensor.ndim() < 2 {
        return Err(Error::InvalidArgument(format!(
            "tril requires at least 2 dimensions (got {})",
            tensor.ndim()
        )));
    }
    Ok(tensor.tril(diagonal))
}

/// Concatenate tensors along an existing dimension.
///
/// The context parameter enables future GPU-backend dispatch. Currently
/// delegates to `Tensor::cat`.
///
/// # Errors
///
/// Returns an error if tensors have incompatible shapes, the axis is out of
/// range, or the tensor list is empty.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_prims::{tensor_ops, CpuContext};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(
///     &[2, 3],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// ).unwrap();
/// let b = Tensor::<f64>::zeros(
///     &[2, 4],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// ).unwrap();
/// let result = tensor_ops::cat(&mut ctx, &[&a, &b], 1).unwrap();
/// assert_eq!(result.dims(), &[2, 7]);
/// ```
pub fn cat<T, C>(_ctx: &mut C, tensors: &[&Tensor<T>], axis: usize) -> Result<Tensor<T>>
where
    T: Scalar,
    C: TensorMetadataContextFor,
{
    Tensor::cat(tensors, axis as isize)
}

/// Stack tensors along a new dimension.
///
/// Creates a new dimension at `axis` and concatenates the input tensors
/// along it. All input tensors must have the same shape.
///
/// The context parameter enables future GPU-backend dispatch. Currently
/// delegates to `Tensor::stack`.
///
/// # Errors
///
/// Returns an error if tensors have different shapes, the axis is out of
/// range, or the tensor list is empty.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_prims::{tensor_ops, CpuContext};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(
///     &[2, 3],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// ).unwrap();
/// let b = Tensor::<f64>::zeros(
///     &[2, 3],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// ).unwrap();
/// let result = tensor_ops::stack(&mut ctx, &[&a, &b], 0).unwrap();
/// assert_eq!(result.dims(), &[2, 2, 3]);
/// ```
pub fn stack<T, C>(_ctx: &mut C, tensors: &[&Tensor<T>], axis: usize) -> Result<Tensor<T>>
where
    T: Scalar,
    C: TensorMetadataContextFor,
{
    Tensor::stack(tensors, axis as isize)
}
