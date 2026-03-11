use super::*;

pub(crate) fn increment_col_major_index(index: &mut [usize], dims: &[usize]) {
    for axis in 0..dims.len() {
        index[axis] += 1;
        if index[axis] < dims[axis] {
            return;
        }
        index[axis] = 0;
    }
}

pub(crate) fn tensor_value_at<T: Scalar>(tensor: &Tensor<T>, indices: &[usize]) -> Result<T> {
    if indices.len() != tensor.dims().len() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "index rank mismatch: indices has rank {}, tensor has rank {}",
                indices.len(),
                tensor.dims().len()
            ),
        });
    }

    let data = tensor
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::InvalidAdTensor {
            message: "reverse cotangent normalization requires CPU-backed tensors".to_string(),
        })?;

    let mut offset = tensor.offset();
    for (axis, &idx) in indices.iter().enumerate() {
        let dim = tensor.dims()[axis];
        if idx >= dim {
            return Err(Error::InvalidAdTensor {
                message: format!(
                    "index out of bounds on axis {}: idx={} >= dim={}",
                    axis, idx, dim
                ),
            });
        }
        let step = (idx as isize)
            .checked_mul(tensor.strides()[axis])
            .ok_or_else(|| Error::InvalidAdTensor {
                message: format!(
                    "offset overflow on axis {}: idx={} * stride={}",
                    axis,
                    idx,
                    tensor.strides()[axis]
                ),
            })?;
        offset = offset
            .checked_add(step)
            .ok_or_else(|| Error::InvalidAdTensor {
                message: format!(
                    "offset overflow while indexing tensor at {:?}",
                    tensor.dims()
                ),
            })?;
    }

    data.get(offset as usize)
        .copied()
        .ok_or_else(|| Error::InvalidAdTensor {
            message: format!(
                "computed offset {} out of bounds for backing buffer length {}",
                offset,
                data.len()
            ),
        })
}

pub(crate) fn structured_to_dense_payload<T: Scalar>(
    layout: &StructuredTensor<T>,
) -> Result<Tensor<T>> {
    let dense_dims = layout.logical_dims();
    let total: usize = dense_dims.iter().product();
    let mut dense_data = Vec::with_capacity(total);
    let mut logical_idx = vec![0usize; dense_dims.len()];

    for _ in 0..total {
        let mut payload_idx = vec![0usize; layout.class_count()];
        for (axis, &class_id) in layout.axis_classes().iter().enumerate() {
            payload_idx[class_id] = logical_idx[axis];
        }
        dense_data.push(tensor_value_at(layout.payload(), &payload_idx)?);
        increment_col_major_index(&mut logical_idx, dense_dims);
    }

    Tensor::from_slice(&dense_data, dense_dims, MemoryOrder::ColumnMajor).map_err(Error::from)
}

pub(crate) fn compress_dense_payload_to_layout<T: Scalar>(
    dense: &Tensor<T>,
    layout: &StructuredTensor<T>,
) -> Result<Tensor<T>> {
    if dense.dims() != layout.logical_dims() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "structured compression shape mismatch: expected {:?}, got {:?}",
                layout.logical_dims(),
                dense.dims()
            ),
        });
    }

    let payload_dims = layout.payload().dims();
    let total: usize = payload_dims.iter().product();
    let mut payload_data = Vec::with_capacity(total);
    let mut payload_idx = vec![0usize; payload_dims.len()];

    for _ in 0..total {
        let mut logical_idx = vec![0usize; layout.logical_dims().len()];
        for (axis, &class_id) in layout.axis_classes().iter().enumerate() {
            logical_idx[axis] = payload_idx[class_id];
        }
        payload_data.push(tensor_value_at(dense, &logical_idx)?);
        increment_col_major_index(&mut payload_idx, payload_dims);
    }

    Tensor::from_slice(&payload_data, payload_dims, MemoryOrder::ColumnMajor).map_err(Error::from)
}

pub(crate) fn normalize_cotangent_payload<T: Scalar>(
    output: &AdTensor<T>,
    cotangent: &AdTensor<T>,
    op_name: &'static str,
) -> Result<Tensor<T>> {
    if cotangent.dims() != output.dims() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "{op_name} cotangent shape mismatch: expected {:?}, got {:?}",
                output.dims(),
                cotangent.dims()
            ),
        });
    }

    let output_layout = output.structured_primal();
    let cotangent_layout = cotangent.structured_primal();
    if output_layout.axis_classes() == cotangent_layout.axis_classes()
        && output.primal().dims() == cotangent.primal().dims()
    {
        return Ok(cotangent.primal().clone());
    }

    let dense = if cotangent.is_dense() {
        cotangent.primal().clone()
    } else {
        structured_to_dense_payload(cotangent_layout)?
    };

    if output.is_dense() {
        Ok(dense)
    } else {
        compress_dense_payload_to_layout(&dense, output_layout)
    }
}
