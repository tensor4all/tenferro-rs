use num_complex::{Complex32, Complex64};
use tenferro_algebra::{Conjugate, Scalar};

use super::basics::ensure_common_reverse_tape;
use super::merge::{
    map_ad_tensor_mixed_linear_typed, map_ad_tensor_same_type_linear_typed, merge_add_ad_tensors,
};
use super::Tensor;
use crate::core::{AdTensorSnapshot, DynTensorTyped};
use crate::{tape, AdTensor, Error, Result};
use tidu::Tape;

fn ensure_reverse_leaf_attached<T>(input: &AdTensor<T>) -> Result<()>
where
    T: Scalar + DynTensorTyped + 'static,
{
    if input.requires_grad() && input.reverse_tape().is_none() {
        let tape = Tape::new();
        input.ensure_reverse_leaf_on(&tape)?;
    }
    Ok(())
}

fn conj_ad_tensor_typed<T>(input: &AdTensor<T>) -> Result<AdTensor<T>>
where
    T: Scalar + Conjugate + DynTensorTyped + 'static,
{
    ensure_reverse_leaf_attached(input)?;

    match input.snapshot()? {
        AdTensorSnapshot::Primal(primal) => Ok(AdTensor::new_primal(primal.conj())),
        AdTensorSnapshot::Forward { primal, tangent } => {
            AdTensor::new_forward(primal.conj(), tangent.conj())
        }
        AdTensorSnapshot::Reverse {
            primal,
            node: input_node,
            tape,
            tangent,
        } => {
            let output_primal = primal.conj();
            let output_tangent = tangent.map(|value| value.conj());
            let out = AdTensor::new_reverse_output(output_primal, &tape, output_tangent)?;
            let output_node = out
                .reverse_node_id()
                .ok_or_else(|| Error::InvalidAdTensor {
                    message: "conj reverse output is missing a tape node".to_string(),
                })?;
            tape::register_closure_rule::<T>(
                &tape,
                output_node,
                vec![input_node],
                Box::new(move |cotangent| Ok(vec![(input_node, cotangent.conj())])),
            );
            Ok(out)
        }
    }
}

impl Tensor {
    /// Returns the complex conjugate while preserving AD mode and layout.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use num_complex::Complex64;
    /// use tenferro::{ScalarValue, Tensor};
    ///
    /// let x = Tensor::from_slice(&[Complex64::new(1.0, 2.0)], &[]).unwrap();
    /// let y = x.conj();
    /// assert_eq!(y.try_scalar_value().unwrap(), ScalarValue::C64(Complex64::new(1.0, -2.0)));
    /// ```
    pub fn conj(&self) -> Self {
        match self {
            Self::F32(v) => Self::F32(v.clone()),
            Self::F64(v) => Self::F64(v.clone()),
            Self::C32(v) => match conj_ad_tensor_typed(v) {
                Ok(value) => Self::C32(value),
                Err(err) => panic!("Tensor::conj should preserve valid AD invariants: {err}"),
            },
            Self::C64(v) => match conj_ad_tensor_typed(v) {
                Ok(value) => Self::C64(value),
                Err(err) => panic!("Tensor::conj should preserve valid AD invariants: {err}"),
            },
        }
    }

    /// AD-preserving extraction of the real component.
    pub fn real_part(&self) -> Result<Self> {
        match self {
            Self::F32(v) => Ok(Self::F32(v.clone())),
            Self::F64(v) => Ok(Self::F64(v.clone())),
            Self::C32(v) => {
                if v.requires_grad() {
                    return Err(Error::UnsupportedAdOp {
                        op: "real_part_reverse",
                    });
                }
                Ok(Self::F32(map_ad_tensor_mixed_linear_typed(
                    v,
                    |z| z.re,
                    |cotangent| Complex32::new(cotangent, 0.0),
                )?))
            }
            Self::C64(v) => {
                if v.requires_grad() {
                    return Err(Error::UnsupportedAdOp {
                        op: "real_part_reverse",
                    });
                }
                Ok(Self::F64(map_ad_tensor_mixed_linear_typed(
                    v,
                    |z| z.re,
                    |cotangent| Complex64::new(cotangent, 0.0),
                )?))
            }
        }
    }

    /// AD-preserving extraction of the imaginary component.
    pub fn imag_part(&self) -> Result<Self> {
        match self {
            Self::F32(v) => Ok(Self::F32(map_ad_tensor_same_type_linear_typed(v, |_| {
                0.0_f32
            })?)),
            Self::F64(v) => Ok(Self::F64(map_ad_tensor_same_type_linear_typed(v, |_| {
                0.0_f64
            })?)),
            Self::C32(v) => {
                if v.requires_grad() {
                    return Err(Error::UnsupportedAdOp {
                        op: "imag_part_reverse",
                    });
                }
                Ok(Self::F32(map_ad_tensor_mixed_linear_typed(
                    v,
                    |z| z.im,
                    |cotangent| Complex32::new(0.0, cotangent),
                )?))
            }
            Self::C64(v) => {
                if v.requires_grad() {
                    return Err(Error::UnsupportedAdOp {
                        op: "imag_part_reverse",
                    });
                }
                Ok(Self::F64(map_ad_tensor_mixed_linear_typed(
                    v,
                    |z| z.im,
                    |cotangent| Complex64::new(0.0, cotangent),
                )?))
            }
        }
    }

    /// Compose a complex AD tensor from real/imaginary AD tensors.
    pub fn compose_complex(real: Self, imag: Self) -> Result<Self> {
        ensure_common_reverse_tape(&[&real, &imag])?;
        match (real, imag) {
            (Self::F32(re), Self::F32(im)) => {
                let re_c = map_ad_tensor_mixed_linear_typed(
                    &re,
                    |x| Complex32::new(x, 0.0),
                    |cotangent| cotangent.re,
                )?;
                let im_c = map_ad_tensor_mixed_linear_typed(
                    &im,
                    |y| Complex32::new(0.0, y),
                    |cotangent| cotangent.im,
                )?;
                let merged = merge_add_ad_tensors(re_c.snapshot()?, im_c.snapshot()?)?;
                Ok(Self::C32(AdTensor::try_from(merged)?))
            }
            (Self::F64(re), Self::F64(im)) => {
                let re_c = map_ad_tensor_mixed_linear_typed(
                    &re,
                    |x| Complex64::new(x, 0.0),
                    |cotangent| cotangent.re,
                )?;
                let im_c = map_ad_tensor_mixed_linear_typed(
                    &im,
                    |y| Complex64::new(0.0, y),
                    |cotangent| cotangent.im,
                )?;
                let merged = merge_add_ad_tensors(re_c.snapshot()?, im_c.snapshot()?)?;
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
