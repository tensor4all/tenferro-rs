use tenferro_tensor::{Engine, Error, Result, TensorScalar, TypedTensor};

use crate::eager::eager_einsum_read_subscripts;
use crate::Subscripts;

/// Execute eager einsum over typed tensors and return a typed result.
///
pub(crate) fn typed_eager_einsum<T: TensorScalar>(
    engine: &mut dyn Engine,
    inputs: &[&TypedTensor<T>],
    subscripts: &str,
) -> Result<TypedTensor<T>> {
    let subscripts = Subscripts::parse(subscripts).map_err(|err| {
        Error::extension(
            "typed_eager_einsum",
            crate::EINSUM_EXTENSION_FAMILY_ID,
            err.kind(),
            err,
        )
    })?;
    let reads: Vec<_> = inputs.iter().map(|tensor| T::tensor_read(tensor)).collect();
    let result = eager_einsum_read_subscripts(engine, &reads, &subscripts)?;
    let actual = result.dtype();
    T::into_typed(result)
        .map_err(|_| Error::dtype_mismatch("typed_eager_einsum", T::dtype(), actual))
}
