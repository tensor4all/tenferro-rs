use tenferro_tensor::{Error, Result, TensorBackend, TensorScalar, TypedTensor};

use crate::eager::eager_einsum_read_subscripts;
use crate::Subscripts;

/// Execute eager einsum over typed tensors and return a typed result.
///
pub(crate) fn typed_eager_einsum<T: TensorScalar>(
    ctx: &mut impl TensorBackend,
    inputs: &[&TypedTensor<T>],
    subscripts: &str,
) -> Result<TypedTensor<T>> {
    let subscripts = Subscripts::parse(subscripts).map_err(|err| Error::InvalidConfig {
        op: "typed_eager_einsum",
        message: format!("invalid subscripts: {err}"),
    })?;
    let reads: Vec<_> = inputs.iter().map(|tensor| T::tensor_read(tensor)).collect();
    let result = eager_einsum_read_subscripts(ctx, &reads, &subscripts)?;
    let actual = result.dtype();
    T::try_into_typed(result).ok_or_else(|| Error::DTypeMismatch {
        op: "typed_eager_einsum",
        lhs: actual,
        rhs: T::dtype(),
    })
}
