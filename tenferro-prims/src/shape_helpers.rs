use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

pub(crate) fn validate_shape_broadcastable(
    got: &[usize],
    target: &[usize],
    _operand_name: &str,
) -> Result<()> {
    if got.len() > target.len() {
        return Err(Error::ShapeMismatch {
            expected: target.to_vec(),
            got: got.to_vec(),
        });
    }

    let leading = target.len() - got.len();
    for (axis, (&got_dim, &target_dim)) in got.iter().zip(&target[leading..]).enumerate() {
        if got_dim != 1 && got_dim != target_dim {
            return Err(Error::ShapeMismatch {
                expected: target.to_vec(),
                got: got.to_vec(),
            });
        }
        let _ = axis;
    }

    Ok(())
}

pub(crate) fn broadcast_tensor_to_shape<T: Scalar>(
    tensor: &Tensor<T>,
    target: &[usize],
    label: &str,
) -> Result<Tensor<T>> {
    validate_shape_broadcastable(tensor.dims(), target, label)?;

    let mut expanded = tensor.clone();
    for _ in 0..target.len().saturating_sub(expanded.ndim()) {
        expanded = expanded.unsqueeze(0)?;
    }
    expanded.broadcast(target)
}
