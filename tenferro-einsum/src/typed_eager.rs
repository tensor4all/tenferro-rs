use tenferro_tensor::{Error, Result, Tensor, TensorBackend, TensorScalar, TypedTensor};

use crate::eager::eager_einsum;

/// Execute eager einsum over typed tensors and return a typed result.
///
pub(crate) fn typed_eager_einsum<T: TensorScalar>(
    ctx: &mut impl TensorBackend,
    inputs: &[&TypedTensor<T>],
    subscripts: &str,
) -> Result<TypedTensor<T>> {
    let tensors: Vec<Tensor> = inputs
        .iter()
        .map(|tensor| T::into_tensor(tensor.shape.clone(), tensor.host_data().to_vec()))
        .collect();
    let refs: Vec<&Tensor> = tensors.iter().collect();
    let result = eager_einsum(ctx, &refs, subscripts)?;
    let actual = result.dtype();
    T::try_into_typed(result).ok_or_else(|| Error::DTypeMismatch {
        op: "typed_eager_einsum",
        lhs: actual,
        rhs: T::dtype(),
    })
}
