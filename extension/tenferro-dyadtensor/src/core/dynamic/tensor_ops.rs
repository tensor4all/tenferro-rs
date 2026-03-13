use core::ops::Sub;

use tenferro_algebra::Scalar;
use tenferro_tensor::{MemoryOrder, Tensor};

use super::dyn_scalar::AbsAsF64;
use crate::{Error, Result};

pub(super) fn unflatten_index_column_major(mut flat: usize, dims: &[usize], out: &mut [usize]) {
    for (axis, &dim) in dims.iter().enumerate() {
        out[axis] = flat % dim;
        flat /= dim;
    }
}

pub(super) fn tensor_element<T: Scalar + Copy>(tensor: &Tensor<T>, indices: &[usize]) -> Result<T> {
    if indices.len() != tensor.dims().len() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "index rank mismatch: indices has rank {}, tensor has rank {}",
                indices.len(),
                tensor.dims().len()
            ),
        });
    }

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
        let stride = tensor.strides()[axis];
        let step = (idx as isize)
            .checked_mul(stride)
            .ok_or_else(|| Error::InvalidAdTensor {
                message: format!(
                    "offset overflow on axis {}: idx={} * stride={}",
                    axis, idx, stride
                ),
            })?;
        offset = offset
            .checked_add(step)
            .ok_or_else(|| Error::InvalidAdTensor {
                message: format!("offset overflow: {} + {} on axis {}", offset, step, axis),
            })?;
    }

    let buffer = tensor
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::InvalidAdTensor {
            message: "tensor buffer is not host-accessible".to_string(),
        })?;
    let pos = usize::try_from(offset).map_err(|_| Error::InvalidAdTensor {
        message: format!("negative tensor offset computed: {}", offset),
    })?;
    buffer
        .get(pos)
        .copied()
        .ok_or_else(|| Error::InvalidAdTensor {
            message: format!("computed offset {} is out of buffer bounds", pos),
        })
}

pub(super) fn tensor_max_abs_diff_typed<T>(lhs: &Tensor<T>, rhs: &Tensor<T>) -> Result<f64>
where
    T: Scalar + Copy + Sub<Output = T> + AbsAsF64,
{
    if lhs.dims() != rhs.dims() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "shape mismatch in max_abs_diff: lhs={:?}, rhs={:?}",
                lhs.dims(),
                rhs.dims()
            ),
        });
    }

    let dims = lhs.dims();
    let total: usize = dims.iter().product();
    if total == 0 {
        return Ok(0.0);
    }

    let mut idx = vec![0usize; dims.len()];
    let mut max_diff = 0.0_f64;
    for flat in 0..total {
        unflatten_index_column_major(flat, dims, &mut idx);
        let lv = tensor_element(lhs, &idx)?;
        let rv = tensor_element(rhs, &idx)?;
        let d = (lv - rv).abs_as_f64();
        if d > max_diff {
            max_diff = d;
        }
    }
    Ok(max_diff)
}

pub(super) fn tensor_map_binary_typed<T>(
    lhs: &Tensor<T>,
    rhs: &Tensor<T>,
    mut f: impl FnMut(T, T) -> T,
) -> Result<Tensor<T>>
where
    T: Scalar + Copy,
{
    if lhs.dims() != rhs.dims() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "shape mismatch in binary map: lhs={:?}, rhs={:?}",
                lhs.dims(),
                rhs.dims()
            ),
        });
    }

    let dims = lhs.dims().to_vec();
    let total: usize = dims.iter().product();
    let mut idx = vec![0usize; dims.len()];
    let mut out = Vec::with_capacity(total);
    for flat in 0..total {
        unflatten_index_column_major(flat, &dims, &mut idx);
        let lv = tensor_element(lhs, &idx)?;
        let rv = tensor_element(rhs, &idx)?;
        out.push(f(lv, rv));
    }

    Tensor::from_slice(&out, &dims, MemoryOrder::ColumnMajor).map_err(Error::from)
}

pub(super) fn tensor_map_unary_typed<T, U>(
    input: &Tensor<T>,
    mut f: impl FnMut(T) -> U,
) -> Result<Tensor<U>>
where
    T: Scalar + Copy,
    U: Scalar + Copy,
{
    let dims = input.dims().to_vec();
    let total: usize = dims.iter().product();
    let mut idx = vec![0usize; dims.len()];
    let mut out = Vec::with_capacity(total);
    for flat in 0..total {
        unflatten_index_column_major(flat, &dims, &mut idx);
        let v = tensor_element(input, &idx)?;
        out.push(f(v));
    }

    Tensor::from_slice(&out, &dims, MemoryOrder::ColumnMajor).map_err(Error::from)
}

pub(super) fn tensor_max_typed<T>(input: &Tensor<T>) -> Result<T>
where
    T: Scalar + Copy + PartialOrd,
{
    if input.is_empty() {
        return Err(Error::InvalidAdTensor {
            message: "max is undefined for empty tensor".to_string(),
        });
    }

    let dims = input.dims().to_vec();
    let total: usize = dims.iter().product();
    let mut idx = vec![0usize; dims.len()];
    unflatten_index_column_major(0, &dims, &mut idx);
    let mut best = tensor_element(input, &idx)?;
    for flat in 1..total {
        unflatten_index_column_major(flat, &dims, &mut idx);
        let v = tensor_element(input, &idx)?;
        if v > best {
            best = v;
        }
    }
    Ok(best)
}
