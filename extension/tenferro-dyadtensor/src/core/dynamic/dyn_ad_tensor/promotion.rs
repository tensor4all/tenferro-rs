use num_complex::{Complex32, Complex64};

use super::merge::map_ad_tensor_mixed_linear_typed;
use super::DynAdTensor;
use crate::{AdTensor, Error, Result, ScalarType};

fn unsupported_promotion(from: ScalarType, to: ScalarType) -> Error {
    Error::InvalidAdTensor {
        message: format!("unsupported promotion from {from:?} to {to:?}"),
    }
}

pub(super) fn join_scalar_types(types: &[ScalarType]) -> Result<ScalarType> {
    let mut saw_c32 = false;
    let mut saw_c64 = false;
    let mut saw_f32 = false;
    let mut saw_f64 = false;

    for ty in types {
        match ty {
            ScalarType::F32 => saw_f32 = true,
            ScalarType::F64 => saw_f64 = true,
            ScalarType::C32 => saw_c32 = true,
            ScalarType::C64 => saw_c64 = true,
        }
    }

    if saw_c32 && (saw_f64 || saw_c64) {
        return Err(unsupported_promotion(ScalarType::C32, ScalarType::C64));
    }
    if saw_c64 && (saw_f32 || saw_c32) {
        return Err(unsupported_promotion(ScalarType::C64, ScalarType::C32));
    }
    if saw_f32 && saw_f64 {
        return Err(unsupported_promotion(ScalarType::F32, ScalarType::F64));
    }

    if saw_c64 {
        return Ok(ScalarType::C64);
    }
    if saw_c32 {
        return Ok(ScalarType::C32);
    }
    if saw_f64 {
        return Ok(ScalarType::F64);
    }
    if saw_f32 {
        return Ok(ScalarType::F32);
    }
    Err(Error::InvalidAdTensor {
        message: "cannot join an empty scalar type set".to_string(),
    })
}

fn promote_f32_ad_tensor_to_c32(tensor: &AdTensor<f32>) -> Result<AdTensor<Complex32>> {
    map_ad_tensor_mixed_linear_typed(tensor, |x| Complex32::new(x, 0.0), |z| z.re)
}

fn promote_f64_ad_tensor_to_c64(tensor: &AdTensor<f64>) -> Result<AdTensor<Complex64>> {
    map_ad_tensor_mixed_linear_typed(tensor, |x| Complex64::new(x, 0.0), |z| z.re)
}

impl DynAdTensor {
    /// Promotes the tensor to a target runtime scalar type while preserving AD
    /// metadata when the promotion stays within the supported algebraic
    /// promotion matrix.
    ///
    /// Supported promotions are:
    /// - identity (`T -> T`)
    /// - same-precision real-to-complex (`F32 -> C32`, `F64 -> C64`)
    ///
    /// Mixed-dtype reverse promotion remains unsupported under the current
    /// homogeneous `Tape<StructuredTensor<T>>` model and returns
    /// [`Error::UnsupportedAdOp`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex64;
    /// use tenferro_dyadtensor::{AdTensor, DynAdTensor, ScalarType};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let x: DynAdTensor = AdTensor::new_primal(
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// )
    /// .into();
    ///
    /// let y = x.promote_to(ScalarType::C64).unwrap();
    /// assert_eq!(y.scalar_type(), ScalarType::C64);
    /// assert_eq!(
    ///     y.as_c64().unwrap().primal().buffer().as_slice().unwrap(),
    ///     &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]
    /// );
    /// ```
    pub fn promote_to(&self, target: ScalarType) -> Result<Self> {
        match (self, target) {
            (Self::F32(value), ScalarType::F32) => Ok(Self::F32(value.clone())),
            (Self::F64(value), ScalarType::F64) => Ok(Self::F64(value.clone())),
            (Self::C32(value), ScalarType::C32) => Ok(Self::C32(value.clone())),
            (Self::C64(value), ScalarType::C64) => Ok(Self::C64(value.clone())),
            (Self::F32(value), ScalarType::C32) => {
                Ok(Self::C32(promote_f32_ad_tensor_to_c32(value)?))
            }
            (Self::F64(value), ScalarType::C64) => {
                Ok(Self::C64(promote_f64_ad_tensor_to_c64(value)?))
            }
            _ => Err(unsupported_promotion(self.scalar_type(), target)),
        }
    }
}
