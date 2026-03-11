use num_complex::{Complex32, Complex64};

use super::binary::{
    checked_apply_binary_ad, embed_f32_to_c32_imag, embed_f64_to_c64_imag, promote_f32_to_c32,
    promote_f64_to_c64, try_binary_dyn, BinaryOp,
};
use super::DynAdScalar;
use crate::{AdScalar, Error, Result};

impl DynAdScalar {
    /// Complex conjugation with AD propagation.
    pub fn conj(&self) -> Self {
        match self.clone() {
            Self::F32(v) => Self::F32(v),
            Self::F64(v) => Self::F64(v),
            Self::C32(v) => Self::C32(AdScalar::from(v).conj().into_value()),
            Self::C64(v) => Self::C64(AdScalar::from(v).conj().into_value()),
        }
    }

    /// Square root with AD propagation.
    pub fn sqrt(&self) -> Self {
        match self.clone() {
            Self::F32(v) => {
                if *v.primal_ref() >= 0.0 {
                    Self::F32(AdScalar::from(v).sqrt().into_value())
                } else {
                    Self::C32(
                        AdScalar::from(promote_f32_to_c32(v, "sqrt"))
                            .sqrt()
                            .into_value(),
                    )
                }
            }
            Self::F64(v) => {
                if *v.primal_ref() >= 0.0 {
                    Self::F64(AdScalar::from(v).sqrt().into_value())
                } else {
                    Self::C64(
                        AdScalar::from(promote_f64_to_c64(v, "sqrt"))
                            .sqrt()
                            .into_value(),
                    )
                }
            }
            Self::C32(v) => Self::C32(AdScalar::from(v).sqrt().into_value()),
            Self::C64(v) => Self::C64(AdScalar::from(v).sqrt().into_value()),
        }
    }

    /// Power with real exponent (`f64`) and AD propagation.
    pub fn powf(&self, exponent: f64) -> Self {
        match self.clone() {
            Self::F32(v) => {
                if *v.primal_ref() >= 0.0 {
                    Self::F32(AdScalar::from(v).powf(exponent as f32).into_value())
                } else {
                    Self::C32(
                        AdScalar::from(promote_f32_to_c32(v, "powf"))
                            .powf(exponent as f32)
                            .into_value(),
                    )
                }
            }
            Self::F64(v) => {
                if *v.primal_ref() >= 0.0 {
                    Self::F64(AdScalar::from(v).powf(exponent).into_value())
                } else {
                    Self::C64(
                        AdScalar::from(promote_f64_to_c64(v, "powf"))
                            .powf(exponent)
                            .into_value(),
                    )
                }
            }
            Self::C32(v) => Self::C32(AdScalar::from(v).powf(exponent as f32).into_value()),
            Self::C64(v) => Self::C64(AdScalar::from(v).powf(exponent).into_value()),
        }
    }

    /// Power with integer exponent and AD propagation.
    pub fn powi(&self, exponent: i32) -> Self {
        match self.clone() {
            Self::F32(v) => Self::F32(AdScalar::from(v).powi(exponent).into_value()),
            Self::F64(v) => Self::F64(AdScalar::from(v).powi(exponent).into_value()),
            Self::C32(v) => Self::C32(AdScalar::from(v).powi(exponent).into_value()),
            Self::C64(v) => Self::C64(AdScalar::from(v).powi(exponent).into_value()),
        }
    }

    /// AD-preserving extraction of the real component.
    pub fn real_part(&self) -> Self {
        match self.clone() {
            Self::F32(v) => Self::F32(v),
            Self::F64(v) => Self::F64(v),
            Self::C32(v) => Self::F32(crate::ad_value::map_ad_value_mixed_linear(
                v,
                "real_part",
                |z| z.re,
                |cotangent| Complex32::new(cotangent, 0.0),
            )),
            Self::C64(v) => Self::F64(crate::ad_value::map_ad_value_mixed_linear(
                v,
                "real_part",
                |z| z.re,
                |cotangent| Complex64::new(cotangent, 0.0),
            )),
        }
    }

    /// AD-preserving extraction of the imaginary component.
    pub fn imag_part(&self) -> Self {
        match self.clone() {
            Self::F32(v) => Self::F32(crate::ad_value::map_ad_value_same_type_linear(
                v,
                "imag_part",
                |_| 0.0_f32,
            )),
            Self::F64(v) => Self::F64(crate::ad_value::map_ad_value_same_type_linear(
                v,
                "imag_part",
                |_| 0.0_f64,
            )),
            Self::C32(v) => Self::F32(crate::ad_value::map_ad_value_mixed_linear(
                v,
                "imag_part",
                |z| z.im,
                |cotangent| Complex32::new(0.0, cotangent),
            )),
            Self::C64(v) => Self::F64(crate::ad_value::map_ad_value_mixed_linear(
                v,
                "imag_part",
                |z| z.im,
                |cotangent| Complex64::new(0.0, cotangent),
            )),
        }
    }

    /// Compose a complex AD scalar from real/imaginary AD scalars.
    pub fn compose_complex(real: Self, imag: Self) -> Result<Self> {
        match (real, imag) {
            (Self::F32(re), Self::F32(im)) => Ok(Self::C32(checked_apply_binary_ad(
                promote_f32_to_c32(re, "compose_complex"),
                embed_f32_to_c32_imag(im, "compose_complex"),
                BinaryOp::Add,
            )?)),
            (Self::F64(re), Self::F64(im)) => Ok(Self::C64(checked_apply_binary_ad(
                promote_f64_to_c64(re, "compose_complex"),
                embed_f64_to_c64_imag(im, "compose_complex"),
                BinaryOp::Add,
            )?)),
            (lhs, rhs) => Err(Error::InvalidAdScalar {
                message: format!(
                    "compose_complex requires matching real dtypes, got lhs={:?}, rhs={:?}",
                    lhs.scalar_type(),
                    rhs.scalar_type()
                ),
            }),
        }
    }

    /// Checked addition with runtime dtype validation and promotion.
    pub fn try_add(self, rhs: Self) -> Result<Self> {
        try_binary_dyn(self, rhs, BinaryOp::Add)
    }

    /// Checked subtraction with runtime dtype validation and promotion.
    pub fn try_sub(self, rhs: Self) -> Result<Self> {
        try_binary_dyn(self, rhs, BinaryOp::Sub)
    }

    /// Checked multiplication with runtime dtype validation and promotion.
    pub fn try_mul(self, rhs: Self) -> Result<Self> {
        try_binary_dyn(self, rhs, BinaryOp::Mul)
    }

    /// Checked division with runtime dtype validation and promotion.
    pub fn try_div(self, rhs: Self) -> Result<Self> {
        try_binary_dyn(self, rhs, BinaryOp::Div)
    }
}
