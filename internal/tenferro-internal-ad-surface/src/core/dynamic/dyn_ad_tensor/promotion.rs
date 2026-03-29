use num_complex::{Complex32, Complex64};

use super::basics::ensure_common_reverse_tape;
use super::merge::map_ad_tensor_mixed_linear_typed;
use super::Tensor;
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

fn cast_dynadtensor(value: &Tensor, target: ScalarType) -> Result<Tensor> {
    ensure_common_reverse_tape(&[value])?;
    match (value, target) {
        (Tensor::F32(value), ScalarType::F32) => Ok(Tensor::F32(value.clone())),
        (Tensor::F32(value), ScalarType::F64) => Ok(Tensor::F64(cast_f32_to_f64(value)?)),
        (Tensor::F32(value), ScalarType::C32) => {
            Ok(Tensor::C32(promote_f32_ad_tensor_to_c32(value)?))
        }
        (Tensor::F32(value), ScalarType::C64) => Ok(Tensor::C64(cast_f32_to_c64(value)?)),
        (Tensor::F64(value), ScalarType::F32) => Ok(Tensor::F32(cast_f64_to_f32(value)?)),
        (Tensor::F64(value), ScalarType::F64) => Ok(Tensor::F64(value.clone())),
        (Tensor::F64(value), ScalarType::C32) => Ok(Tensor::C32(cast_f64_to_c32(value)?)),
        (Tensor::F64(value), ScalarType::C64) => {
            Ok(Tensor::C64(promote_f64_ad_tensor_to_c64(value)?))
        }
        (Tensor::C32(value), ScalarType::F32) => Ok(Tensor::F32(cast_c32_to_f32(value)?)),
        (Tensor::C32(value), ScalarType::F64) => Ok(Tensor::F64(cast_c32_to_f64(value)?)),
        (Tensor::C32(value), ScalarType::C32) => Ok(Tensor::C32(value.clone())),
        (Tensor::C32(value), ScalarType::C64) => Ok(Tensor::C64(cast_c32_to_c64(value)?)),
        (Tensor::C64(value), ScalarType::F32) => Ok(Tensor::F32(cast_c64_to_f32(value)?)),
        (Tensor::C64(value), ScalarType::F64) => Ok(Tensor::F64(cast_c64_to_f64(value)?)),
        (Tensor::C64(value), ScalarType::C32) => Ok(Tensor::C32(cast_c64_to_c32(value)?)),
        (Tensor::C64(value), ScalarType::C64) => Ok(Tensor::C64(value.clone())),
    }
}

pub(super) fn promote_pair_to_common(
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<(ScalarType, Tensor, Tensor)> {
    ensure_common_reverse_tape(&[lhs, rhs])?;
    let target = join_scalar_types(&[lhs.scalar_type(), rhs.scalar_type()])?;
    Ok((target, lhs.promote_to(target)?, rhs.promote_to(target)?))
}

pub(super) fn promote_many_to_common(operands: &[&Tensor]) -> Result<(ScalarType, Vec<Tensor>)> {
    ensure_common_reverse_tape(operands)?;
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

fn cast_dynadtensor_owned(value: Tensor, target: ScalarType) -> Result<Tensor> {
    match (value, target) {
        (Tensor::F32(value), ScalarType::F32) => Ok(Tensor::F32(value)),
        (Tensor::F32(value), ScalarType::F64) => Ok(Tensor::F64(cast_f32_to_f64(&value)?)),
        (Tensor::F32(value), ScalarType::C32) => {
            Ok(Tensor::C32(promote_f32_ad_tensor_to_c32(&value)?))
        }
        (Tensor::F32(value), ScalarType::C64) => Ok(Tensor::C64(cast_f32_to_c64(&value)?)),
        (Tensor::F64(value), ScalarType::F32) => Ok(Tensor::F32(cast_f64_to_f32(&value)?)),
        (Tensor::F64(value), ScalarType::F64) => Ok(Tensor::F64(value)),
        (Tensor::F64(value), ScalarType::C32) => Ok(Tensor::C32(cast_f64_to_c32(&value)?)),
        (Tensor::F64(value), ScalarType::C64) => {
            Ok(Tensor::C64(promote_f64_ad_tensor_to_c64(&value)?))
        }
        (Tensor::C32(value), ScalarType::F32) => Ok(Tensor::F32(cast_c32_to_f32(&value)?)),
        (Tensor::C32(value), ScalarType::F64) => Ok(Tensor::F64(cast_c32_to_f64(&value)?)),
        (Tensor::C32(value), ScalarType::C32) => Ok(Tensor::C32(value)),
        (Tensor::C32(value), ScalarType::C64) => Ok(Tensor::C64(cast_c32_to_c64(&value)?)),
        (Tensor::C64(value), ScalarType::F32) => Ok(Tensor::F32(cast_c64_to_f32(&value)?)),
        (Tensor::C64(value), ScalarType::F64) => Ok(Tensor::F64(cast_c64_to_f64(&value)?)),
        (Tensor::C64(value), ScalarType::C32) => Ok(Tensor::C32(cast_c64_to_c32(&value)?)),
        (Tensor::C64(value), ScalarType::C64) => Ok(Tensor::C64(value)),
    }
}

pub(super) fn promote_many_to_common_owned(
    operands: Vec<Tensor>,
) -> Result<(ScalarType, Vec<Tensor>)> {
    let operand_refs: Vec<&Tensor> = operands.iter().collect();
    ensure_common_reverse_tape(&operand_refs)?;
    let target = join_scalar_types(
        &operand_refs
            .iter()
            .map(|operand| operand.scalar_type())
            .collect::<Vec<_>>(),
    )?;
    let promoted = operands
        .into_iter()
        .map(|operand| cast_dynadtensor_owned(operand, target))
        .collect::<Result<Vec<_>>>()?;
    Ok((target, promoted))
}

impl Tensor {
    /// Explicitly casts the tensor to `target`, similar to PyTorch `tensor.to(dtype)`.
    ///
    /// This is distinct from implicit operation-local promotion. Explicit casts
    /// may change precision and may also convert between real and complex
    /// dtypes.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::{ScalarType, Tensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};
    ///
    /// let x = Tensor::from_tensor(
    ///     DenseTensor::<f32>::from_slice(&[1.0], &[], MemoryOrder::ColumnMajor).unwrap(),
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
