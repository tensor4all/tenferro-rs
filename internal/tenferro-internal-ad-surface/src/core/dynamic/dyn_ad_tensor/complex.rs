use chainrules::ScalarAd;
use num_complex::{Complex32, Complex64};
use tenferro_algebra::{Conjugate, Scalar};
use tenferro_internal_ad_core::{AdTensor, DynAdTensor, DynAdTensorRef};

use super::basics::ensure_common_reverse_tape_impl;
use super::merge::{
    map_ad_tensor_mixed_linear_typed, map_ad_tensor_same_type_linear_typed, merge_add_ad_tensors,
};
use super::Tensor;
use crate::core::{AdTensorSnapshot, DynTensorTyped};
use crate::{tape, Error, Result};
use tidu::expert::Tape;

macro_rules! match_dyn_ad_tensor_ref {
    ($tensor:expr, {
        F32($f32:ident) => $f32_body:expr,
        F64($f64:ident) => $f64_body:expr,
        C32($c32:ident) => $c32_body:expr,
        C64($c64:ident) => $c64_body:expr,
    }) => {{
        match $tensor.as_dyn_ad_ref() {
            DynAdTensorRef::F32($f32) => $f32_body,
            DynAdTensorRef::F64($f64) => $f64_body,
            DynAdTensorRef::C32($c32) => $c32_body,
            DynAdTensorRef::C64($c64) => $c64_body,
        }
    }};
}

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

fn map_complex_component_typed<TIn, TOut, P, R>(
    input: &AdTensor<TIn>,
    op: &'static str,
    primal_map: P,
    reverse_map: R,
) -> Result<AdTensor<TOut>>
where
    TIn: Scalar + ScalarAd + Copy + DynTensorTyped + 'static,
    TOut: Scalar + ScalarAd + Copy + DynTensorTyped + 'static,
    P: Fn(TIn) -> TOut + Copy,
    R: Fn(TOut) -> TIn + Copy + Send + Sync + 'static,
{
    if input.requires_grad() {
        return Err(Error::UnsupportedAdOp { op });
    }
    map_ad_tensor_mixed_linear_typed(input, primal_map, reverse_map)
}

fn compose_complex_typed<TIn, TOut, PR, RR, PI, RI>(
    real: &AdTensor<TIn>,
    imag: &AdTensor<TIn>,
    real_primal_map: PR,
    real_reverse_map: RR,
    imag_primal_map: PI,
    imag_reverse_map: RI,
) -> Result<AdTensor<TOut>>
where
    TIn: Scalar + ScalarAd + Copy + DynTensorTyped + 'static,
    TOut: Scalar + ScalarAd + Copy + DynTensorTyped + 'static,
    PR: Fn(TIn) -> TOut + Copy,
    RR: Fn(TOut) -> TIn + Copy + Send + Sync + 'static,
    PI: Fn(TIn) -> TOut + Copy,
    RI: Fn(TOut) -> TIn + Copy + Send + Sync + 'static,
{
    let re_c = map_ad_tensor_mixed_linear_typed(real, real_primal_map, real_reverse_map)?;
    let im_c = map_ad_tensor_mixed_linear_typed(imag, imag_primal_map, imag_reverse_map)?;
    let merged = merge_add_ad_tensors(re_c.snapshot()?, im_c.snapshot()?)?;
    AdTensor::try_from(merged)
}

fn conj_ad_tensor_typed<T>(input: &AdTensor<T>) -> Result<AdTensor<T>>
where
    T: Scalar + Conjugate + DynTensorTyped + 'static,
{
    ensure_reverse_leaf_attached(input)?;

    match input.snapshot()? {
        AdTensorSnapshot::Primal(primal) => {
            AdTensor::try_from(AdTensorSnapshot::Primal(primal.conj()))
        }
        AdTensorSnapshot::Forward { primal, tangent } => {
            AdTensor::try_from(AdTensorSnapshot::Forward {
                primal: primal.conj(),
                tangent: tangent.conj(),
            })
        }
        AdTensorSnapshot::Reverse {
            primal,
            node: input_node,
            tape,
            tangent,
        } => {
            let output_primal = primal.conj();
            let output_tangent = tangent.map(|value| value.conj());
            let out = AdTensor::from_reverse_output(output_primal, &tape, output_tangent)?;
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
        match_dyn_ad_tensor_ref!(self, {
            F32(v) => Self::from(v.clone()),
            F64(v) => Self::from(v.clone()),
            C32(v) => match conj_ad_tensor_typed(v) {
                Ok(value) => Self::from(value),
                Err(err) => panic!("Tensor::conj should preserve valid AD invariants: {err}"),
            },
            C64(v) => match conj_ad_tensor_typed(v) {
                Ok(value) => Self::from(value),
                Err(err) => panic!("Tensor::conj should preserve valid AD invariants: {err}"),
            },
        })
    }

    /// AD-preserving extraction of the real component.
    pub fn real_part(&self) -> Result<Self> {
        match_dyn_ad_tensor_ref!(self, {
            F32(v) => Ok(Self::from(v.clone())),
            F64(v) => Ok(Self::from(v.clone())),
            C32(v) => Ok(Self::from(map_complex_component_typed(
                v,
                "real_part_reverse",
                |z| z.re,
                |cotangent| Complex32::new(cotangent, 0.0),
            )?)),
            C64(v) => Ok(Self::from(map_complex_component_typed(
                v,
                "real_part_reverse",
                |z| z.re,
                |cotangent| Complex64::new(cotangent, 0.0),
            )?)),
        })
    }

    /// AD-preserving extraction of the imaginary component.
    pub fn imag_part(&self) -> Result<Self> {
        match_dyn_ad_tensor_ref!(self, {
            F32(v) => Ok(Self::from(map_ad_tensor_same_type_linear_typed(v, |_| {
                0.0_f32
            })?)),
            F64(v) => Ok(Self::from(map_ad_tensor_same_type_linear_typed(v, |_| {
                0.0_f64
            })?)),
            C32(v) => Ok(Self::from(map_complex_component_typed(
                v,
                "imag_part_reverse",
                |z| z.im,
                |cotangent| Complex32::new(0.0, cotangent),
            )?)),
            C64(v) => Ok(Self::from(map_complex_component_typed(
                v,
                "imag_part_reverse",
                |z| z.im,
                |cotangent| Complex64::new(0.0, cotangent),
            )?)),
        })
    }

    /// Compose a complex AD tensor from real/imaginary AD tensors.
    pub fn compose_complex(real: Self, imag: Self) -> Result<Self> {
        ensure_common_reverse_tape_impl(&[&real, &imag])?;
        match (real.0, imag.0) {
            (DynAdTensor::F32(re), DynAdTensor::F32(im)) => Ok(Self::from(compose_complex_typed(
                &re,
                &im,
                |x| Complex32::new(x, 0.0),
                |cotangent| cotangent.re,
                |y| Complex32::new(0.0, y),
                |cotangent| cotangent.im,
            )?)),
            (DynAdTensor::F64(re), DynAdTensor::F64(im)) => Ok(Self::from(compose_complex_typed(
                &re,
                &im,
                |x| Complex64::new(x, 0.0),
                |cotangent| cotangent.re,
                |y| Complex64::new(0.0, y),
                |cotangent| cotangent.im,
            )?)),
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
