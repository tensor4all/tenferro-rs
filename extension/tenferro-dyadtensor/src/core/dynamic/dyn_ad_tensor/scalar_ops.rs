use chainrules::Tape;
use chainrules_scalarops::{self, ScalarAd};
use num_complex::{Complex32, Complex64};
use tenferro_algebra::Scalar;
use tenferro_tensor::{MemoryOrder, Tensor};

use super::super::tensor_ops::{
    tensor_element, tensor_map_binary_typed, tensor_map_unary_typed, tensor_max_abs_diff_typed,
    unflatten_index_column_major,
};
use super::merge::merge_add_ad_tensors;
use super::promotion::join_scalar_types;
use super::DynAdTensor;
use crate::core::DynTensorTyped;
use crate::{AdTensor, DynTensor, Error, Result};

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

fn rank0_tensor<T>(value: T) -> Result<Tensor<T>>
where
    T: Scalar + Copy,
{
    Tensor::from_slice(&[value], &[], MemoryOrder::ColumnMajor).map_err(Error::from)
}

fn extract_rank0_scalar<T>(scalar: &AdTensor<T>, op_name: &'static str) -> Result<T>
where
    T: Scalar + Copy,
{
    if scalar.dims() != [] {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "{op_name} requires a rank-0 scalar tensor, got dims={:?}",
                scalar.dims()
            ),
        });
    }
    tensor_element(scalar.primal(), &[])
}

fn extract_rank0_scalar_tangent<T>(scalar: &AdTensor<T>, op_name: &'static str) -> Result<Option<T>>
where
    T: Scalar + Copy,
{
    scalar
        .tangent()
        .map(|tangent| {
            if tangent.dims() != [] {
                return Err(Error::InvalidAdTensor {
                    message: format!(
                        "{op_name} requires a rank-0 scalar tangent tensor, got dims={:?}",
                        tangent.dims()
                    ),
                });
            }
            tensor_element(tangent, &[])
        })
        .transpose()
}

fn merge_tensor_scalar_output<T>(
    tensor: &AdTensor<T>,
    scalar: &AdTensor<T>,
    primal: Tensor<T>,
    tangent: Option<Tensor<T>>,
    rrule: fn(T, T, T) -> (T, T),
) -> Result<AdTensor<T>>
where
    T: Scalar + ScalarAd + Copy + DynTensorTyped + 'static,
{
    let tensor_reverse = tensor.reverse_handle();
    let scalar_reverse = scalar.reverse_handle();

    let reverse = match (tensor_reverse.clone(), scalar_reverse.clone()) {
        (Some((_lhs_node, lhs_tape)), Some((_, rhs_tape)))
            if !lhs_tape.same_tape(&rhs_tape as &Tape<DynTensor>) =>
        {
            return Err(Error::MixedReverseTape {
                expected: lhs_tape.id() as u64,
                found: rhs_tape.id() as u64,
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
        let tensor_node = tensor_reverse.map(|(node, _)| node);
        let scalar_node = scalar_reverse.map(|(node, _)| node);
        let tensor_layout = tensor.structured_primal().clone();
        let scalar_layout = scalar.structured_primal().clone();
        let tensor_primal = tensor.primal().clone();
        let scalar_primal = extract_rank0_scalar(scalar, "tensor_scalar_reverse")?;
        let out = AdTensor::new_reverse_output(structured_primal, &tape, structured_tangent)?;
        let output_node = out
            .reverse_node_id()
            .ok_or_else(|| Error::InvalidAdTensor {
                message: "tensor-scalar reverse output is missing a tape node".to_string(),
            })?;

        crate::tape::register_rule::<T>(
            &tape,
            output_node,
            Box::new(move |cotangent| {
                let mut input_grads = Vec::new();
                if let Some(node) = tensor_node {
                    let (tensor_grad, _) = tensor_scalar_rrule_typed(
                        &tensor_primal,
                        scalar_primal,
                        cotangent.payload(),
                        rrule,
                    )?;
                    input_grads.push((node, tensor_layout.with_payload_like(tensor_grad)?));
                }
                if let Some(node) = scalar_node {
                    let (_, scalar_grad) = tensor_scalar_rrule_typed(
                        &tensor_primal,
                        scalar_primal,
                        cotangent.payload(),
                        rrule,
                    )?;
                    input_grads.push((
                        node,
                        scalar_layout.with_payload_like(rank0_tensor(scalar_grad)?)?,
                    ));
                }
                Ok(input_grads)
            }),
        );

        return Ok(out);
    }
    if let Some(tangent) = structured_tangent {
        return AdTensor::new_forward(structured_primal, tangent);
    }
    Ok(AdTensor::new_primal(structured_primal))
}

fn scale_ad_tensor_typed<T>(tensor: &AdTensor<T>, scalar: &AdTensor<T>) -> Result<AdTensor<T>>
where
    T: Scalar + ScalarAd + Copy + DynTensorTyped + 'static,
{
    let (primal, tangent) = tensor_binary_scalar_ad_typed(
        tensor.primal(),
        tensor.tangent(),
        extract_rank0_scalar(scalar, "scale")?,
        extract_rank0_scalar_tangent(scalar, "scale")?,
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

fn div_ad_tensor_typed<T>(tensor: &AdTensor<T>, scalar: &AdTensor<T>) -> Result<AdTensor<T>>
where
    T: Scalar + ScalarAd + Copy + DynTensorTyped + 'static,
{
    let (primal, tangent) = tensor_binary_scalar_ad_typed(
        tensor.primal(),
        tensor.tangent(),
        extract_rank0_scalar(scalar, "div_scalar")?,
        extract_rank0_scalar_tangent(scalar, "div_scalar")?,
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
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex64;
    /// use tenferro_dyadtensor::DynAdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let x = DynAdTensor::new_primal(
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    /// let alpha = DynAdTensor::new_primal(
    ///     Tensor::<Complex64>::from_slice(&[Complex64::new(0.0, 2.0)], &[], MemoryOrder::ColumnMajor)
    ///         .unwrap(),
    /// );
    ///
    /// let y = x.scale(&alpha).unwrap();
    /// assert_eq!(y.scalar_type(), tenferro_dyadtensor::ScalarType::C64);
    /// ```
    pub fn scale(&self, scalar: &DynAdTensor) -> Result<Self> {
        let target = join_scalar_types(&[self.scalar_type(), scalar.scalar_type()])?;
        let tensor = self.promote_to(target)?;
        let alpha = scalar.promote_to(target)?;
        match (&tensor, &alpha) {
            (Self::F32(tensor), Self::F32(alpha)) => {
                Ok(Self::F32(scale_ad_tensor_typed(tensor, alpha)?))
            }
            (Self::F64(tensor), Self::F64(alpha)) => {
                Ok(Self::F64(scale_ad_tensor_typed(tensor, alpha)?))
            }
            (Self::C32(tensor), Self::C32(alpha)) => {
                Ok(Self::C32(scale_ad_tensor_typed(tensor, alpha)?))
            }
            (Self::C64(tensor), Self::C64(alpha)) => {
                Ok(Self::C64(scale_ad_tensor_typed(tensor, alpha)?))
            }
            _ => Err(Error::InvalidAdTensor {
                message: format!(
                    "dtype mismatch in scale after promotion: tensor={:?}, scalar={:?}",
                    tensor.scalar_type(),
                    alpha.scalar_type()
                ),
            }),
        }
    }

    /// Affine combination `a * self + b * other`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::DynAdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let x = DynAdTensor::new_primal(
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    /// let y = DynAdTensor::new_primal(
    ///     Tensor::<f64>::from_slice(&[3.0, 4.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    /// let a = DynAdTensor::new_primal(
    ///     Tensor::<f64>::from_slice(&[2.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    /// let b = DynAdTensor::new_primal(
    ///     Tensor::<f64>::from_slice(&[-1.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    ///
    /// let out = x.axpby(&a, &y, &b).unwrap();
    /// assert_eq!(out.scalar_type(), tenferro_dyadtensor::ScalarType::F64);
    /// ```
    pub fn axpby(&self, a: &DynAdTensor, other: &Self, b: &DynAdTensor) -> Result<Self> {
        let _ = join_scalar_types(&[
            self.scalar_type(),
            other.scalar_type(),
            a.scalar_type(),
            b.scalar_type(),
        ])?;
        match (self.scale(a)?, other.scale(b)?) {
            (Self::F32(lhs), Self::F32(rhs)) => Ok(Self::F32(AdTensor::try_from(
                merge_add_ad_tensors(lhs.snapshot(), rhs.snapshot())?,
            )?)),
            (Self::F64(lhs), Self::F64(rhs)) => Ok(Self::F64(AdTensor::try_from(
                merge_add_ad_tensors(lhs.snapshot(), rhs.snapshot())?,
            )?)),
            (Self::C32(lhs), Self::C32(rhs)) => Ok(Self::C32(AdTensor::try_from(
                merge_add_ad_tensors(lhs.snapshot(), rhs.snapshot())?,
            )?)),
            (Self::C64(lhs), Self::C64(rhs)) => Ok(Self::C64(AdTensor::try_from(
                merge_add_ad_tensors(lhs.snapshot(), rhs.snapshot())?,
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
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::DynAdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let x = DynAdTensor::new_primal(
    ///     Tensor::<f64>::from_slice(&[2.0, 4.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    /// let alpha = DynAdTensor::new_primal(
    ///     Tensor::<f64>::from_slice(&[2.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    ///
    /// let y = x.div_scalar(&alpha).unwrap();
    /// assert_eq!(y.scalar_type(), tenferro_dyadtensor::ScalarType::F64);
    /// ```
    pub fn div_scalar(&self, scalar: &DynAdTensor) -> Result<Self> {
        let target = join_scalar_types(&[self.scalar_type(), scalar.scalar_type()])?;
        let tensor = self.promote_to(target)?;
        let alpha = scalar.promote_to(target)?;
        match (&tensor, &alpha) {
            (Self::F32(tensor), Self::F32(alpha)) => {
                Ok(Self::F32(div_ad_tensor_typed(tensor, alpha)?))
            }
            (Self::F64(tensor), Self::F64(alpha)) => {
                Ok(Self::F64(div_ad_tensor_typed(tensor, alpha)?))
            }
            (Self::C32(tensor), Self::C32(alpha)) => {
                Ok(Self::C32(div_ad_tensor_typed(tensor, alpha)?))
            }
            (Self::C64(tensor), Self::C64(alpha)) => {
                Ok(Self::C64(div_ad_tensor_typed(tensor, alpha)?))
            }
            _ => Err(Error::InvalidAdTensor {
                message: format!(
                    "dtype mismatch in div_scalar after promotion: tensor={:?}, scalar={:?}",
                    tensor.scalar_type(),
                    alpha.scalar_type()
                ),
            }),
        }
    }

    /// Computes `max(abs(primal(self) - primal(rhs)))`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::DynAdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let x = DynAdTensor::new_primal(
    ///     Tensor::<f64>::from_slice(&[1.0, 3.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    /// let y = DynAdTensor::new_primal(
    ///     Tensor::<f64>::from_slice(&[2.5, 1.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    ///
    /// assert_eq!(x.max_abs_diff_primal(&y).unwrap(), 2.0);
    /// ```
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
