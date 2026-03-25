use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::validate_shape_count;

pub(crate) fn validate_pointwise_broadcast_shapes(
    shapes: &[&[usize]],
    arity: usize,
    op_name: &str,
) -> Result<()> {
    validate_shape_count(shapes, arity + 1, op_name)?;
    let output_shape = shapes[arity];
    for shape in &shapes[..arity] {
        validate_broadcastable_to_output(shape, output_shape)?;
    }
    Ok(())
}

pub(crate) fn broadcast_tensors_to_output<T: Scalar>(
    inputs: &[&Tensor<T>],
    output_dims: &[usize],
) -> Result<Vec<Tensor<T>>> {
    inputs
        .iter()
        .map(|tensor| broadcast_tensor_to_output(tensor, output_dims))
        .collect()
}

pub(crate) fn broadcast_tensor_to_output<T: Scalar>(
    tensor: &Tensor<T>,
    output_dims: &[usize],
) -> Result<Tensor<T>> {
    if tensor.ndim() > output_dims.len() {
        return Err(Error::RankMismatch {
            expected: output_dims.len(),
            got: tensor.ndim(),
        });
    }

    let mut view = tensor.clone();
    while view.ndim() < output_dims.len() {
        view = view.unsqueeze(0)?;
    }
    view.broadcast(output_dims)
}

fn validate_broadcastable_to_output(input_dims: &[usize], output_dims: &[usize]) -> Result<()> {
    if input_dims.len() > output_dims.len() {
        return Err(Error::RankMismatch {
            expected: output_dims.len(),
            got: input_dims.len(),
        });
    }

    let offset = output_dims.len() - input_dims.len();
    for (&input, &output) in input_dims.iter().zip(&output_dims[offset..]) {
        if input != 1 && input != output {
            return Err(Error::ShapeMismatch {
                expected: output_dims.to_vec(),
                got: input_dims.to_vec(),
            });
        }
    }
    Ok(())
}
