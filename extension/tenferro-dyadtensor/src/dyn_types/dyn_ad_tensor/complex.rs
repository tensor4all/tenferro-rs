use num_complex::{Complex32, Complex64};

use super::merge::{
    map_ad_tensor_mixed_linear_typed, map_ad_tensor_same_type_linear_typed, merge_add_ad_tensors,
};
use super::DynAdTensor;
use crate::{AdTensor, Error, Result};

impl DynAdTensor {
    /// AD-preserving extraction of the real component.
    pub fn real_part(&self) -> Result<Self> {
        match self {
            Self::F32(v) => Ok(Self::F32(v.clone())),
            Self::F64(v) => Ok(Self::F64(v.clone())),
            Self::C32(v) => Ok(Self::F32(map_ad_tensor_mixed_linear_typed(
                v,
                "real_part",
                |z| z.re,
                |cotangent| Complex32::new(cotangent, 0.0),
            )?)),
            Self::C64(v) => Ok(Self::F64(map_ad_tensor_mixed_linear_typed(
                v,
                "real_part",
                |z| z.re,
                |cotangent| Complex64::new(cotangent, 0.0),
            )?)),
        }
    }

    /// AD-preserving extraction of the imaginary component.
    pub fn imag_part(&self) -> Result<Self> {
        match self {
            Self::F32(v) => Ok(Self::F32(map_ad_tensor_same_type_linear_typed(
                v,
                "imag_part",
                |_| 0.0_f32,
            )?)),
            Self::F64(v) => Ok(Self::F64(map_ad_tensor_same_type_linear_typed(
                v,
                "imag_part",
                |_| 0.0_f64,
            )?)),
            Self::C32(v) => Ok(Self::F32(map_ad_tensor_mixed_linear_typed(
                v,
                "imag_part",
                |z| z.im,
                |cotangent| Complex32::new(0.0, cotangent),
            )?)),
            Self::C64(v) => Ok(Self::F64(map_ad_tensor_mixed_linear_typed(
                v,
                "imag_part",
                |z| z.im,
                |cotangent| Complex64::new(0.0, cotangent),
            )?)),
        }
    }

    /// Compose a complex AD tensor from real/imaginary AD tensors.
    pub fn compose_complex(real: Self, imag: Self) -> Result<Self> {
        match (real, imag) {
            (Self::F32(re), Self::F32(im)) => {
                let re_c = map_ad_tensor_mixed_linear_typed(
                    &re,
                    "compose_complex",
                    |x| Complex32::new(x, 0.0),
                    |cotangent| cotangent.re,
                )?;
                let im_c = map_ad_tensor_mixed_linear_typed(
                    &im,
                    "compose_complex",
                    |y| Complex32::new(0.0, y),
                    |cotangent| cotangent.im,
                )?;
                let merged = merge_add_ad_tensors(re_c.into_value(), im_c.into_value())?;
                Ok(Self::C32(AdTensor::try_from(merged)?))
            }
            (Self::F64(re), Self::F64(im)) => {
                let re_c = map_ad_tensor_mixed_linear_typed(
                    &re,
                    "compose_complex",
                    |x| Complex64::new(x, 0.0),
                    |cotangent| cotangent.re,
                )?;
                let im_c = map_ad_tensor_mixed_linear_typed(
                    &im,
                    "compose_complex",
                    |y| Complex64::new(0.0, y),
                    |cotangent| cotangent.im,
                )?;
                let merged = merge_add_ad_tensors(re_c.into_value(), im_c.into_value())?;
                Ok(Self::C64(AdTensor::try_from(merged)?))
            }
            (lhs, rhs) => Err(Error::InvalidAdTensor {
                message: format!(
                    "compose_complex requires matching real dtypes, got lhs={:?}, rhs={:?}",
                    lhs.scalar_type(),
                    rhs.scalar_type()
                ),
            }),
        }
    }
}
