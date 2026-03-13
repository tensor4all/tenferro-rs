use chainrules_scalarops::{self, ScalarAd};
use num_complex::{Complex32, Complex64};
use tenferro_algebra::Scalar;
use tenferro_tensor::Tensor;

use super::super::dyn_ad_scalar::{promote_f32_to_c32, promote_f64_to_c64, DynAdScalar};
use super::super::tensor_ops::{
    tensor_element, tensor_map_binary_typed, tensor_map_unary_typed, tensor_max_abs_diff_typed,
    unflatten_index_column_major,
};
use super::merge::merge_add_ad_tensors;
use super::DynAdTensor;
use crate::{AdTensor, AdValue, Error, Result};

fn tensor_scalar_rrule_typed<T>(
    tensor_primal: &Tensor<T>,
    scalar_primal: T,
    cotangent: &Tensor<T>,
    rrule: fn(T, T, T) -> (T, T),
) -> Result<(Tensor<T>, T)>
where
    T: Scalar + ScalarAd + Copy,
{
    if tensor_primal.dims() != cotangent.dims() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "shape mismatch in mixed reverse pullback: primal={:?}, cotangent={:?}",
                tensor_primal.dims(),
                cotangent.dims()
            ),
        });
    }

    let dims = tensor_primal.dims().to_vec();
    let total: usize = dims.iter().product();
    let mut idx = vec![0usize; dims.len()];
    let mut tensor_grad = Vec::with_capacity(total);
    let mut scalar_grad = T::from_i32(0);

    for flat in 0..total {
        unflatten_index_column_major(flat, &dims, &mut idx);
        let x = tensor_element(tensor_primal, &idx)?;
        let dy = tensor_element(cotangent, &idx)?;
        let (dx, da) = rrule(x, scalar_primal, dy);
        tensor_grad.push(dx);
        scalar_grad = scalar_grad + da;
    }

    Ok((
        Tensor::from_slice(
            &tensor_grad,
            &dims,
            tenferro_tensor::MemoryOrder::ColumnMajor,
        )
        .map_err(Error::from)?,
        scalar_grad,
    ))
}

fn tensor_binary_scalar_ad_typed<T>(
    primal: &Tensor<T>,
    tensor_tangent: Option<&Tensor<T>>,
    scalar_primal: T,
    scalar_tangent: Option<T>,
    primal_rule: fn(T, T) -> T,
    frule: fn(T, T, T, T) -> (T, T),
) -> Result<(Tensor<T>, Option<Tensor<T>>)>
where
    T: Scalar + ScalarAd + Copy,
{
    let primal_out = tensor_map_unary_typed(primal, |x| primal_rule(x, scalar_primal))?;
    let tangent_out = match (tensor_tangent, scalar_tangent) {
        (None, None) => None,
        (Some(dt), maybe_ds) => Some(tensor_map_binary_typed(primal, dt, |x, dx| {
            let (_, tangent) = frule(
                x,
                scalar_primal,
                dx,
                maybe_ds.unwrap_or_else(|| T::from_i32(0)),
            );
            tangent
        })?),
        (None, Some(ds)) => Some(tensor_map_unary_typed(primal, |x| {
            let (_, tangent) = frule(x, scalar_primal, T::from_i32(0), ds);
            tangent
        })?),
    };
    Ok((primal_out, tangent_out))
}

fn merge_tensor_scalar_output<T>(
    tensor: &AdTensor<T>,
    scalar: &AdValue<T>,
    primal: Tensor<T>,
    tangent: Option<Tensor<T>>,
    rrule: fn(T, T, T) -> (T, T),
) -> Result<AdTensor<T>>
where
    T: Scalar + ScalarAd + Copy + 'static,
{
    let tensor_reverse = match tensor.as_value() {
        AdValue::Reverse { node, tape, .. } => Some((*node, *tape)),
        _ => None,
    };
    let scalar_reverse = match scalar {
        AdValue::Reverse { node, tape, .. } => Some((*node, *tape)),
        _ => None,
    };

    let reverse = match (tensor_reverse, scalar_reverse) {
        (Some((_lhs_node, lhs_tape)), Some((_, rhs_tape))) if lhs_tape != rhs_tape => {
            return Err(Error::MixedReverseTape {
                expected: lhs_tape.0,
                found: rhs_tape.0,
            });
        }
        (Some((node, tape)), Some(_)) => Some((node, tape)),
        (Some((node, tape)), None) => Some((node, tape)),
        (None, Some((node, tape))) => Some((node, tape)),
        (None, None) => None,
    };

    let structured_primal = tensor.structured_primal().with_payload_like(primal)?;
    let structured_tangent = tangent
        .map(|payload| tensor.structured_primal().with_payload_like(payload))
        .transpose()?;

    if let Some((_, tape)) = reverse {
        let output_node = super::layout::fresh_ad_tensor_node_id();
        let tensor_node = tensor_reverse.map(|(node, _)| node);
        let scalar_node = scalar_reverse.map(|(node, _)| node);
        let tensor_primal = tensor.primal().clone();
        let tensor_primal_for_scalar = tensor_primal.clone();
        let scalar_primal = *scalar.primal_ref();

        crate::tape::register_rule::<T>(
            tape,
            output_node,
            Box::new(move |cotangent| {
                let mut input_grads = Vec::new();
                if let Some(node) = tensor_node {
                    let (tensor_grad, _) =
                        tensor_scalar_rrule_typed(&tensor_primal, scalar_primal, cotangent, rrule)?;
                    input_grads.push((node, tensor_grad));
                }
                Ok(input_grads)
            }),
        );

        if let Some(node) = scalar_node {
            crate::tape::register_scalar_bridge_rule::<T, T>(
                tape,
                output_node,
                Box::new(move |cotangent| {
                    let (_, scalar_grad) = tensor_scalar_rrule_typed(
                        &tensor_primal_for_scalar,
                        scalar_primal,
                        cotangent,
                        rrule,
                    )?;
                    Ok(vec![(node, scalar_grad)])
                }),
            );
        }

        return AdTensor::new_reverse(structured_primal, output_node, tape, structured_tangent);
    }
    if let Some(tangent) = structured_tangent {
        return AdTensor::new_forward(structured_primal, tangent);
    }
    Ok(AdTensor::new_primal(structured_primal))
}

fn scale_ad_tensor_typed<T>(tensor: &AdTensor<T>, scalar: &AdValue<T>) -> Result<AdTensor<T>>
where
    T: Scalar + ScalarAd + Copy + 'static,
{
    let (primal, tangent) = tensor_binary_scalar_ad_typed(
        tensor.primal(),
        tensor.tangent(),
        *scalar.primal_ref(),
        scalar.tangent_ref().copied(),
        chainrules_scalarops::mul,
        chainrules_scalarops::mul_frule,
    )?;
    merge_tensor_scalar_output(
        tensor,
        scalar,
        primal,
        tangent,
        chainrules_scalarops::mul_rrule,
    )
}

fn div_ad_tensor_typed<T>(tensor: &AdTensor<T>, scalar: &AdValue<T>) -> Result<AdTensor<T>>
where
    T: Scalar + ScalarAd + Copy + 'static,
{
    let (primal, tangent) = tensor_binary_scalar_ad_typed(
        tensor.primal(),
        tensor.tangent(),
        *scalar.primal_ref(),
        scalar.tangent_ref().copied(),
        chainrules_scalarops::div,
        chainrules_scalarops::div_frule,
    )?;
    merge_tensor_scalar_output(
        tensor,
        scalar,
        primal,
        tangent,
        chainrules_scalarops::div_rrule,
    )
}

impl DynAdTensor {
    /// Scalar multiply with AD preservation for scalar and tensor inputs.
    pub fn scale(&self, scalar: &DynAdScalar) -> Result<Self> {
        match (self, scalar) {
            (Self::F32(tensor), DynAdScalar::F32(alpha)) => {
                Ok(Self::F32(scale_ad_tensor_typed(tensor, alpha)?))
            }
            (Self::F64(tensor), DynAdScalar::F64(alpha)) => {
                Ok(Self::F64(scale_ad_tensor_typed(tensor, alpha)?))
            }
            (Self::C32(tensor), DynAdScalar::C32(alpha)) => {
                Ok(Self::C32(scale_ad_tensor_typed(tensor, alpha)?))
            }
            (Self::C32(tensor), DynAdScalar::F32(alpha)) => {
                let promoted = promote_f32_to_c32(alpha.clone(), "scale");
                Ok(Self::C32(scale_ad_tensor_typed(tensor, &promoted)?))
            }
            (Self::C64(tensor), DynAdScalar::C64(alpha)) => {
                Ok(Self::C64(scale_ad_tensor_typed(tensor, alpha)?))
            }
            (Self::C64(tensor), DynAdScalar::F64(alpha)) => {
                let promoted = promote_f64_to_c64(alpha.clone(), "scale");
                Ok(Self::C64(scale_ad_tensor_typed(tensor, &promoted)?))
            }
            _ => Err(Error::InvalidAdTensor {
                message: format!(
                    "dtype mismatch in scale: tensor={:?}, scalar={:?}",
                    self.scalar_type(),
                    scalar.scalar_type()
                ),
            }),
        }
    }

    /// Affine combination `a * self + b * other`.
    pub fn axpby(&self, a: &DynAdScalar, other: &Self, b: &DynAdScalar) -> Result<Self> {
        match (self.scale(a)?, other.scale(b)?) {
            (Self::F32(lhs), Self::F32(rhs)) => Ok(Self::F32(AdTensor::try_from(
                merge_add_ad_tensors(lhs.into_value(), rhs.into_value())?,
            )?)),
            (Self::F64(lhs), Self::F64(rhs)) => Ok(Self::F64(AdTensor::try_from(
                merge_add_ad_tensors(lhs.into_value(), rhs.into_value())?,
            )?)),
            (Self::C32(lhs), Self::C32(rhs)) => Ok(Self::C32(AdTensor::try_from(
                merge_add_ad_tensors(lhs.into_value(), rhs.into_value())?,
            )?)),
            (Self::C64(lhs), Self::C64(rhs)) => Ok(Self::C64(AdTensor::try_from(
                merge_add_ad_tensors(lhs.into_value(), rhs.into_value())?,
            )?)),
            (lhs, rhs) => Err(Error::InvalidAdTensor {
                message: format!(
                    "dtype mismatch in axpby after scaling: lhs={:?}, rhs={:?}",
                    lhs.scalar_type(),
                    rhs.scalar_type()
                ),
            }),
        }
    }

    /// Division by an AD-aware scalar.
    pub fn div_scalar(&self, scalar: &DynAdScalar) -> Result<Self> {
        match (self, scalar) {
            (Self::F32(tensor), DynAdScalar::F32(alpha)) => {
                Ok(Self::F32(div_ad_tensor_typed(tensor, alpha)?))
            }
            (Self::F64(tensor), DynAdScalar::F64(alpha)) => {
                Ok(Self::F64(div_ad_tensor_typed(tensor, alpha)?))
            }
            (Self::C32(tensor), DynAdScalar::C32(alpha)) => {
                Ok(Self::C32(div_ad_tensor_typed(tensor, alpha)?))
            }
            (Self::C32(tensor), DynAdScalar::F32(alpha)) => {
                let promoted = promote_f32_to_c32(alpha.clone(), "div_scalar");
                Ok(Self::C32(div_ad_tensor_typed(tensor, &promoted)?))
            }
            (Self::C64(tensor), DynAdScalar::C64(alpha)) => {
                Ok(Self::C64(div_ad_tensor_typed(tensor, alpha)?))
            }
            (Self::C64(tensor), DynAdScalar::F64(alpha)) => {
                let promoted = promote_f64_to_c64(alpha.clone(), "div_scalar");
                Ok(Self::C64(div_ad_tensor_typed(tensor, &promoted)?))
            }
            _ => Err(Error::InvalidAdTensor {
                message: format!(
                    "dtype mismatch in div_scalar: tensor={:?}, scalar={:?}",
                    self.scalar_type(),
                    scalar.scalar_type()
                ),
            }),
        }
    }

    /// Computes `max(abs(primal(self) - primal(rhs)))`.
    pub fn max_abs_diff_primal(&self, rhs: &Self) -> Result<f64> {
        match (self, rhs) {
            (Self::F32(a), Self::F32(b)) => tensor_max_abs_diff_typed(a.primal(), b.primal()),
            (Self::F64(a), Self::F64(b)) => tensor_max_abs_diff_typed(a.primal(), b.primal()),
            (Self::C32(a), Self::C32(b)) => tensor_max_abs_diff_typed(a.primal(), b.primal()),
            (Self::C64(a), Self::C64(b)) => tensor_max_abs_diff_typed(a.primal(), b.primal()),
            _ => Err(Error::InvalidAdTensor {
                message: format!(
                    "dtype mismatch in max_abs_diff_primal: lhs={:?}, rhs={:?}",
                    self.scalar_type(),
                    rhs.scalar_type()
                ),
            }),
        }
    }
}

macro_rules! impl_dyn_ad_tensor_from {
    ($variant:ident, $ty:ty) => {
        impl From<AdTensor<$ty>> for DynAdTensor {
            fn from(value: AdTensor<$ty>) -> Self {
                Self::$variant(value)
            }
        }
    };
}

impl_dyn_ad_tensor_from!(F32, f32);
impl_dyn_ad_tensor_from!(F64, f64);
impl_dyn_ad_tensor_from!(C32, Complex32);
impl_dyn_ad_tensor_from!(C64, Complex64);
