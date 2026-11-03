//! Internal support surface used by `tenferro-ad`.

use tenferro_tensor::{DType, Tensor, TypedTensor};

use crate::error::Result;

pub use crate::checkpoint::RetainedValue;
pub use crate::metadata::tensor_meta_from_tensor;

/// Allocate one fresh traced leaf key.
pub fn next_input_key() -> tenferro_ops::input_key::TensorInputKey {
    crate::traced::next_input_key()
}

/// Build a core identity tensor of ones with the requested dtype and shape.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::ad_support::ones_tensor;
/// use tenferro_tensor::DType;
///
/// let ones = ones_tensor(DType::F64, vec![2])?;
/// assert_eq!(ones.as_slice::<f64>()?, &[1.0, 1.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns [`crate::Error::TensorRuntime`] with a typed validation error when the
/// shape product overflows, or the typed tensor's validation source when the
/// tensor cannot be constructed from `shape`. An externally defined scalar has no
/// core identity tensor and is rejected with [`crate::Error::Validation`].
pub fn ones_tensor(dtype: DType, shape: Vec<usize>) -> Result<Tensor> {
    match dtype {
        DType::F32 => Ok(Tensor::from_typed::<f32>(TypedTensor::ones(shape)?)),
        DType::F64 => Ok(Tensor::from_typed::<f64>(TypedTensor::ones(shape)?)),
        DType::I32 => Ok(Tensor::from_typed::<i32>(TypedTensor::ones(shape)?)),
        DType::I64 => Ok(Tensor::from_typed::<i64>(TypedTensor::ones(shape)?)),
        DType::Bool => {
            let len =
                tenferro_tensor::validate::checked_shape_product("ones_tensor", "shape", &shape)?;
            Ok(Tensor::from_typed::<bool>(TypedTensor::from_vec_col_major(
                shape,
                vec![true; len],
            )?))
        }
        DType::C32 => Ok(Tensor::from_typed::<tenferro_tensor::Complex32>(
            TypedTensor::ones(shape)?,
        )),
        DType::C64 => Ok(Tensor::from_typed::<tenferro_tensor::Complex64>(
            TypedTensor::ones(shape)?,
        )),
        // An externally defined scalar is caller-owned and has no core identity
        // tensor, so the runtime cannot build one for it.
        DType::External(_) => Err(crate::Error::invalid_argument(
            "ones_tensor",
            crate::ErrorPhase::GraphBuild,
            "dtype",
            "an externally defined scalar has no core runtime tensor",
        )),
    }
}
