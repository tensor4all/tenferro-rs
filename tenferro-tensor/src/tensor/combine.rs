use std::sync::Arc;

use tenferro_algebra::Scalar;
use tenferro_device::{Error, LogicalMemorySpace, Result};

use super::Tensor;
use crate::layout::compute_contiguous_strides;
use crate::MemoryOrder;

impl<T: Scalar> Tensor<T> {
    /// Stack tensors along a new dimension.
    ///
    /// Creates a new dimension and concatenates the input tensors along it.
    /// All input tensors must have the same shape. Negative dimensions are
    /// supported and count from the end.
    ///
    /// This is a dense materialization operation that allocates a new buffer.
    /// Phase 1 only supports main-memory tensors.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Slice of input tensors to stack. Must not be empty.
    /// * `dim` - Position to insert the new dimension. Must be in range `[-ndim-1, ndim]`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The input list is empty
    /// - Tensors have different shapes
    /// - Tensors have different memory spaces
    /// - The dimension is out of range
    /// - Non-main-memory tensors are provided (Phase 1 limitation)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::LogicalMemorySpace;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let a = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// let b = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    ///
    /// let stacked = Tensor::stack(&[&a, &b], 0).unwrap();
    /// assert_eq!(stacked.dims(), &[2, 2, 3]);
    /// ```
    pub fn stack(tensors: &[&Tensor<T>], dim: isize) -> Result<Tensor<T>> {
        if tensors.is_empty() {
            return Err(Error::InvalidArgument(
                "stack requires at least one tensor".to_string(),
            ));
        }

        for t in tensors {
            t.wait();
        }

        let first = tensors[0];
        let ndim = first.ndim();

        let dim = if dim < 0 {
            let wrapped = dim + (ndim as isize) + 1;
            if wrapped < 0 {
                return Err(Error::InvalidArgument(format!(
                    "stack dim {dim} out of range for tensors with {ndim} dimensions (valid: [{}, {}])",
                    -(ndim as isize) - 1,
                    ndim
                )));
            }
            wrapped as usize
        } else if dim as usize > ndim {
            return Err(Error::InvalidArgument(format!(
                "stack dim {dim} out of range for tensors with {ndim} dimensions (valid: [{}, {}])",
                -(ndim as isize) - 1,
                ndim
            )));
        } else {
            dim as usize
        };

        let memory_space = first.logical_memory_space();
        if memory_space != LogicalMemorySpace::MainMemory {
            return Err(Error::InvalidArgument(
                "stack only supports main-memory tensors in Phase 1".to_string(),
            ));
        }

        for (i, t) in tensors.iter().enumerate() {
            if t.dims() != first.dims() {
                return Err(Error::ShapeMismatch {
                    expected: first.dims.to_vec(),
                    got: t.dims.to_vec(),
                });
            }
            if t.logical_memory_space() != memory_space {
                return Err(Error::InvalidArgument(format!(
                    "tensor {} has different memory space {:?} (expected {:?})",
                    i, t.logical_memory_space, memory_space
                )));
            }
        }

        let n = tensors.len();
        let mut result_dims: Vec<usize> = first.dims.to_vec();
        result_dims.insert(dim, n);

        let result_strides = compute_contiguous_strides(&result_dims, MemoryOrder::ColumnMajor);
        let result_len: usize = result_dims.iter().product();
        let mut result_data = vec![T::zero(); result_len];

        let src_strides = compute_contiguous_strides(&first.dims, MemoryOrder::ColumnMajor);

        for (stack_idx, tensor) in tensors.iter().enumerate() {
            let contiguous_tensor = tensor.contiguous(MemoryOrder::ColumnMajor);
            let src = contiguous_tensor.buffer().as_slice().unwrap();

            let mut index = vec![0usize; ndim];
            let n_elements: usize = first.dims.iter().product();

            if n_elements > 0 {
                for _ in 0..n_elements {
                    let src_pos: isize = index
                        .iter()
                        .zip(src_strides.iter())
                        .map(|(&i, &s)| (i as isize) * s)
                        .sum();

                    let mut result_index = Vec::with_capacity(ndim + 1);
                    for (axis, &idx) in index.iter().enumerate() {
                        if axis == dim {
                            result_index.push(stack_idx);
                        }
                        result_index.push(idx);
                    }
                    if dim == ndim {
                        result_index.push(stack_idx);
                    }

                    let dst_pos: isize = result_index
                        .iter()
                        .zip(result_strides.iter())
                        .map(|(&i, &s)| (i as isize) * s)
                        .sum();

                    result_data[dst_pos as usize] = src[src_pos as usize];

                    for axis in (0..ndim).rev() {
                        index[axis] += 1;
                        if index[axis] < first.dims[axis] {
                            break;
                        }
                        index[axis] = 0;
                    }
                }
            }
        }

        Ok(Tensor::from_parts(
            crate::DataBuffer::from_vec(result_data),
            Arc::from(result_dims),
            Arc::from(result_strides),
            0,
            memory_space,
            first.preferred_compute_device,
            None,
            false,
            None,
        ))
    }

    /// Concatenate tensors along an existing dimension.
    ///
    /// Joins tensors along the specified dimension. All tensors must have
    /// the same rank and matching sizes on non-concatenated dimensions.
    /// Negative dimensions are supported and count from the end.
    ///
    /// This is a dense materialization operation that allocates a new buffer.
    /// Phase 1 only supports main-memory tensors.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Slice of input tensors to concatenate. Must not be empty.
    /// * `dim` - Dimension along which to concatenate. Must be in range `[-ndim, ndim-1]`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The input list is empty
    /// - Any tensor is rank-0 (scalars cannot be concatenated)
    /// - Tensors have different ranks
    /// - Tensors have mismatched sizes on non-concatenated dimensions
    /// - Tensors have different memory spaces
    /// - The dimension is out of range
    /// - Non-main-memory tensors are provided (Phase 1 limitation)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::LogicalMemorySpace;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let a = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    /// let b = Tensor::<f64>::zeros(&[2, 4], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    ///
    /// let concatenated = Tensor::cat(&[&a, &b], 1).unwrap();
    /// assert_eq!(concatenated.dims(), &[2, 7]);
    /// ```
    pub fn cat(tensors: &[&Tensor<T>], dim: isize) -> Result<Tensor<T>> {
        if tensors.is_empty() {
            return Err(Error::InvalidArgument(
                "cat requires at least one tensor".to_string(),
            ));
        }

        for t in tensors {
            t.wait();
        }

        let first = tensors[0];
        let ndim = first.ndim();

        if ndim == 0 {
            return Err(Error::InvalidArgument(
                "cat cannot concatenate rank-0 tensors (use stack to pack scalars)".to_string(),
            ));
        }

        let dim = if dim < 0 {
            let wrapped = dim + (ndim as isize);
            if wrapped < 0 {
                return Err(Error::InvalidArgument(format!(
                    "cat dim {dim} out of range for tensors with {ndim} dimensions (valid: [{}, {}])",
                    -(ndim as isize),
                    ndim - 1
                )));
            }
            wrapped as usize
        } else if dim as usize >= ndim {
            return Err(Error::InvalidArgument(format!(
                "cat dim {dim} out of range for tensors with {ndim} dimensions (valid: [{}, {}])",
                -(ndim as isize),
                ndim - 1
            )));
        } else {
            dim as usize
        };

        let memory_space = first.logical_memory_space();
        if memory_space != LogicalMemorySpace::MainMemory {
            return Err(Error::InvalidArgument(
                "cat only supports main-memory tensors in Phase 1".to_string(),
            ));
        }

        let mut total_cat_dim = 0usize;
        for (i, t) in tensors.iter().enumerate() {
            if t.ndim() != ndim {
                return Err(Error::InvalidArgument(format!(
                    "tensor {} has rank {} but expected rank {}",
                    i,
                    t.ndim(),
                    ndim
                )));
            }
            if t.logical_memory_space() != memory_space {
                return Err(Error::InvalidArgument(format!(
                    "tensor {} has different memory space {:?} (expected {:?})",
                    i, t.logical_memory_space, memory_space
                )));
            }
            for (axis, (&d1, &d2)) in first.dims.iter().zip(t.dims.iter()).enumerate() {
                if axis != dim && d1 != d2 {
                    return Err(Error::ShapeMismatch {
                        expected: first.dims.to_vec(),
                        got: t.dims.to_vec(),
                    });
                }
            }
            total_cat_dim = total_cat_dim.checked_add(t.dims[dim]).ok_or_else(|| {
                Error::InvalidArgument("cat: dimension size overflow".to_string())
            })?;
        }

        let mut result_dims: Vec<usize> = first.dims.to_vec();
        result_dims[dim] = total_cat_dim;

        let result_strides = compute_contiguous_strides(&result_dims, MemoryOrder::ColumnMajor);
        let result_len: usize = result_dims.iter().product();
        let mut result_data = vec![T::zero(); result_len];

        let mut cat_offset: usize = 0;
        for tensor in tensors {
            let contiguous_tensor = tensor.contiguous(MemoryOrder::ColumnMajor);
            let src = contiguous_tensor.buffer().as_slice().unwrap();
            let src_strides = compute_contiguous_strides(&tensor.dims, MemoryOrder::ColumnMajor);

            let mut index = vec![0usize; ndim];
            let n_elements: usize = tensor.dims.iter().product();

            if n_elements > 0 {
                for _ in 0..n_elements {
                    let src_pos: usize = index
                        .iter()
                        .zip(src_strides.iter())
                        .map(|(&i, &s)| (i as isize) * s)
                        .sum::<isize>() as usize;

                    let dst_pos: usize = index
                        .iter()
                        .enumerate()
                        .zip(result_strides.iter())
                        .map(|((axis, &i), &s)| {
                            let adjusted_i = if axis == dim { i + cat_offset } else { i };
                            (adjusted_i as isize) * s
                        })
                        .sum::<isize>() as usize;

                    result_data[dst_pos] = src[src_pos];

                    for axis in (0..ndim).rev() {
                        index[axis] += 1;
                        if index[axis] < tensor.dims[axis] {
                            break;
                        }
                        index[axis] = 0;
                    }
                }
            }

            cat_offset += tensor.dims[dim];
        }

        Ok(Tensor::from_parts(
            crate::DataBuffer::from_vec(result_data),
            Arc::from(result_dims),
            Arc::from(result_strides),
            0,
            memory_space,
            first.preferred_compute_device,
            None,
            false,
            None,
        ))
    }
}
