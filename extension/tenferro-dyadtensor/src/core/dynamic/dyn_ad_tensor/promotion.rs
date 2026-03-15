use num_complex::{Complex32, Complex64};

use super::merge::map_ad_tensor_mixed_linear_typed;
use super::DynAdTensor;
use crate::{AdTensor, Error, Result, ScalarType};

pub(super) fn join_scalar_types(types: &[ScalarType]) -> Result<ScalarType> {
    if types.is_empty() {
        return Err(Error::InvalidAdTensor {
            message: "cannot join an empty scalar type set".to_string(),
        });
    }

    let mut saw_complex = false;
    let mut saw_64 = false;
    for ty in types {
        match ty {
            ScalarType::F32 => {}
            ScalarType::F64 => saw_64 = true,
            ScalarType::C32 => saw_complex = true,
            ScalarType::C64 => {
                saw_complex = true;
                saw_64 = true;
            }
        }
    }

    match (saw_complex, saw_64) {
        (false, false) => Ok(ScalarType::F32),
        (false, true) => Ok(ScalarType::F64),
        (true, false) => Ok(ScalarType::C32),
        (true, true) => Ok(ScalarType::C64),
    }
}

fn promote_f32_ad_tensor_to_c32(tensor: &AdTensor<f32>) -> Result<AdTensor<Complex32>> {
    map_ad_tensor_mixed_linear_typed(tensor, |x| Complex32::new(x, 0.0), |z| z.re)
}

fn promote_f64_ad_tensor_to_c64(tensor: &AdTensor<f64>) -> Result<AdTensor<Complex64>> {
    map_ad_tensor_mixed_linear_typed(tensor, |x| Complex64::new(x, 0.0), |z| z.re)
}

fn cast_f32_to_c64(tensor: &AdTensor<f32>) -> Result<AdTensor<Complex64>> {
    map_ad_tensor_mixed_linear_typed(tensor, |x| Complex64::new(x as f64, 0.0), |z| z.re as f32)
}

fn cast_f64_to_c32(tensor: &AdTensor<f64>) -> Result<AdTensor<Complex32>> {
    map_ad_tensor_mixed_linear_typed(tensor, |x| Complex32::new(x as f32, 0.0), |z| z.re as f64)
}

fn cast_f32_to_f64(tensor: &AdTensor<f32>) -> Result<AdTensor<f64>> {
    map_ad_tensor_mixed_linear_typed(tensor, |x| x as f64, |y| y as f32)
}

fn cast_f64_to_f32(tensor: &AdTensor<f64>) -> Result<AdTensor<f32>> {
    map_ad_tensor_mixed_linear_typed(tensor, |x| x as f32, |y| y as f64)
}

fn cast_c32_to_c64(tensor: &AdTensor<Complex32>) -> Result<AdTensor<Complex64>> {
    map_ad_tensor_mixed_linear_typed(
        tensor,
        |z| Complex64::new(z.re as f64, z.im as f64),
        |z| Complex32::new(z.re as f32, z.im as f32),
    )
}

fn cast_c64_to_c32(tensor: &AdTensor<Complex64>) -> Result<AdTensor<Complex32>> {
    map_ad_tensor_mixed_linear_typed(
        tensor,
        |z| Complex32::new(z.re as f32, z.im as f32),
        |z| Complex64::new(z.re as f64, z.im as f64),
    )
}

fn cast_c32_to_f32(tensor: &AdTensor<Complex32>) -> Result<AdTensor<f32>> {
    map_ad_tensor_mixed_linear_typed(tensor, |z| z.re, |x| Complex32::new(x, 0.0))
}

fn cast_c32_to_f64(tensor: &AdTensor<Complex32>) -> Result<AdTensor<f64>> {
    map_ad_tensor_mixed_linear_typed(tensor, |z| z.re as f64, |x| Complex32::new(x as f32, 0.0))
}

fn cast_c64_to_f64(tensor: &AdTensor<Complex64>) -> Result<AdTensor<f64>> {
    map_ad_tensor_mixed_linear_typed(tensor, |z| z.re, |x| Complex64::new(x, 0.0))
}

fn cast_c64_to_f32(tensor: &AdTensor<Complex64>) -> Result<AdTensor<f32>> {
    map_ad_tensor_mixed_linear_typed(tensor, |z| z.re as f32, |x| Complex64::new(x as f64, 0.0))
}

fn cast_dynadtensor(value: &DynAdTensor, target: ScalarType) -> Result<DynAdTensor> {
    match (value, target) {
        (DynAdTensor::F32(value), ScalarType::F32) => Ok(DynAdTensor::F32(value.clone())),
        (DynAdTensor::F32(value), ScalarType::F64) => Ok(DynAdTensor::F64(cast_f32_to_f64(value)?)),
        (DynAdTensor::F32(value), ScalarType::C32) => {
            Ok(DynAdTensor::C32(promote_f32_ad_tensor_to_c32(value)?))
        }
        (DynAdTensor::F32(value), ScalarType::C64) => Ok(DynAdTensor::C64(cast_f32_to_c64(value)?)),
        (DynAdTensor::F64(value), ScalarType::F32) => Ok(DynAdTensor::F32(cast_f64_to_f32(value)?)),
        (DynAdTensor::F64(value), ScalarType::F64) => Ok(DynAdTensor::F64(value.clone())),
        (DynAdTensor::F64(value), ScalarType::C32) => Ok(DynAdTensor::C32(cast_f64_to_c32(value)?)),
        (DynAdTensor::F64(value), ScalarType::C64) => {
            Ok(DynAdTensor::C64(promote_f64_ad_tensor_to_c64(value)?))
        }
        (DynAdTensor::C32(value), ScalarType::F32) => Ok(DynAdTensor::F32(cast_c32_to_f32(value)?)),
        (DynAdTensor::C32(value), ScalarType::F64) => Ok(DynAdTensor::F64(cast_c32_to_f64(value)?)),
        (DynAdTensor::C32(value), ScalarType::C32) => Ok(DynAdTensor::C32(value.clone())),
        (DynAdTensor::C32(value), ScalarType::C64) => Ok(DynAdTensor::C64(cast_c32_to_c64(value)?)),
        (DynAdTensor::C64(value), ScalarType::F32) => Ok(DynAdTensor::F32(cast_c64_to_f32(value)?)),
        (DynAdTensor::C64(value), ScalarType::F64) => Ok(DynAdTensor::F64(cast_c64_to_f64(value)?)),
        (DynAdTensor::C64(value), ScalarType::C32) => Ok(DynAdTensor::C32(cast_c64_to_c32(value)?)),
        (DynAdTensor::C64(value), ScalarType::C64) => Ok(DynAdTensor::C64(value.clone())),
    }
}

pub(super) fn promote_pair_to_common(
    lhs: &DynAdTensor,
    rhs: &DynAdTensor,
) -> Result<(ScalarType, DynAdTensor, DynAdTensor)> {
    let target = join_scalar_types(&[lhs.scalar_type(), rhs.scalar_type()])?;
    Ok((target, lhs.promote_to(target)?, rhs.promote_to(target)?))
}

pub(super) fn promote_many_to_common(
    operands: &[&DynAdTensor],
) -> Result<(ScalarType, Vec<DynAdTensor>)> {
    let target = join_scalar_types(
        &operands
            .iter()
            .map(|operand| operand.scalar_type())
            .collect::<Vec<_>>(),
    )?;
    let promoted = operands
        .iter()
        .map(|operand| operand.promote_to(target))
        .collect::<Result<Vec<_>>>()?;
    Ok((target, promoted))
}

impl DynAdTensor {
    /// Explicitly casts the tensor to `target`, similar to PyTorch `tensor.to(dtype)`.
    ///
    /// This is distinct from implicit operation-local promotion. Explicit casts
    /// may change precision and may also convert between real and complex
    /// dtypes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{DynAdTensor, ScalarType};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let x = DynAdTensor::new_primal(
    ///     Tensor::<f32>::from_slice(&[1.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    /// let y = x.to_scalar_type(ScalarType::F64).unwrap();
    /// assert_eq!(y.scalar_type(), ScalarType::F64);
    /// ```
    pub fn to_scalar_type(&self, target: ScalarType) -> Result<Self> {
        cast_dynadtensor(self, target)
    }

    pub(crate) fn promote_to(&self, target: ScalarType) -> Result<Self> {
        cast_dynadtensor(self, target)
    }
}
