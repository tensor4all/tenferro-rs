use num_complex::{Complex32, Complex64};

use super::*;
use crate::structured::StructuredTensor;

impl Tensor {
    /// Scalar multiply with AD preservation for scalar and tensor inputs.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use num_complex::Complex64;
    /// use tenferro::Tensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};
    ///
    /// let x = Tensor::from_tensor(
    ///     DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    /// let alpha = Tensor::from_tensor(
    ///     DenseTensor::<Complex64>::from_slice(
    ///         &[Complex64::new(0.0, 2.0)],
    ///         &[],
    ///         MemoryOrder::ColumnMajor,
    ///     )
    ///         .unwrap(),
    /// );
    ///
    /// let y = x.scale(&alpha).unwrap();
    /// assert_eq!(y.scalar_type(), tenferro::ScalarType::C64);
    /// ```
    pub fn scale(&self, scalar: &Tensor) -> Result<Self> {
        let (_, tensor, alpha) = promote_pair_to_common(self, scalar)?;
        Ok(scale_dyn(tensor.as_dyn_ad_ref(), alpha.as_dyn_ad_ref())?.into())
    }

    /// Affine combination `a * self + b * other`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::Tensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};
    ///
    /// let x = Tensor::from_tensor(
    ///     DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    /// let y = Tensor::from_tensor(
    ///     DenseTensor::<f64>::from_slice(&[3.0, 4.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    /// let a = Tensor::from_tensor(
    ///     DenseTensor::<f64>::from_slice(&[2.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    /// let b = Tensor::from_tensor(
    ///     DenseTensor::<f64>::from_slice(&[-1.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    ///
    /// let out = x.axpby(&a, &y, &b).unwrap();
    /// assert_eq!(out.scalar_type(), tenferro::ScalarType::F64);
    /// ```
    pub fn axpby(&self, a: &Tensor, other: &Self, b: &Tensor) -> Result<Self> {
        let (_, promoted) = promote_many_to_common(&[self, a, other, b])?;
        ensure_common_reverse_tape_impl(&[&promoted[0], &promoted[1], &promoted[2], &promoted[3]])?;
        Ok(axpby_dyn(
            promoted[0].as_dyn_ad_ref(),
            promoted[1].as_dyn_ad_ref(),
            promoted[2].as_dyn_ad_ref(),
            promoted[3].as_dyn_ad_ref(),
        )?
        .into())
    }

    /// Division by an AD-aware scalar.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::Tensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};
    ///
    /// let x = Tensor::from_tensor(
    ///     DenseTensor::<f64>::from_slice(&[2.0, 4.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    /// let alpha = Tensor::from_tensor(
    ///     DenseTensor::<f64>::from_slice(&[2.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    ///
    /// let y = x.div_scalar(&alpha).unwrap();
    /// assert_eq!(y.scalar_type(), tenferro::ScalarType::F64);
    /// ```
    pub fn div_scalar(&self, scalar: &Tensor) -> Result<Self> {
        let (_, tensor, alpha) = promote_pair_to_common(self, scalar)?;
        Ok(div_scalar_dyn(tensor.as_dyn_ad_ref(), alpha.as_dyn_ad_ref())?.into())
    }

    /// Computes `max(abs(primal(self) - primal(rhs)))`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::Tensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};
    ///
    /// let x = Tensor::from_tensor(
    ///     DenseTensor::<f64>::from_slice(&[1.0, 3.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    /// let y = Tensor::from_tensor(
    ///     DenseTensor::<f64>::from_slice(&[2.5, 1.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    ///
    /// assert_eq!(x.max_abs_diff_primal(&y).unwrap(), 2.0);
    /// ```
    pub fn max_abs_diff_primal(&self, rhs: &Self) -> Result<f64> {
        match_dyn_ad_tensor_ref_pair!(
            "max_abs_diff_primal",
            self.as_dyn_ad_ref(),
            rhs.as_dyn_ad_ref(),
            |lhs, rhs| { tensor_max_abs_diff_typed(lhs.primal(), rhs.primal()) }
        )
    }
}

macro_rules! impl_dyn_ad_tensor_from {
    ($variant:ident, $ty:ty) => {
        impl From<AdTensor<$ty>> for Tensor {
            fn from(value: AdTensor<$ty>) -> Self {
                Self::from(DynAdTensor::from(value))
            }
        }
    };
}

impl_dyn_ad_tensor_from!(F32, f32);
impl_dyn_ad_tensor_from!(F64, f64);
impl_dyn_ad_tensor_from!(C32, Complex32);
impl_dyn_ad_tensor_from!(C64, Complex64);

impl From<DynAdTensor> for Tensor {
    fn from(value: DynAdTensor) -> Self {
        Self(value)
    }
}

impl From<DynTensor> for Tensor {
    fn from(value: DynTensor) -> Self {
        match value {
            DynTensor::F32(value) => Self::from_structured(value),
            DynTensor::F64(value) => Self::from_structured(value),
            DynTensor::C32(value) => Self::from_structured(value),
            DynTensor::C64(value) => Self::from_structured(value),
        }
    }
}

impl<T> From<tenferro_tensor::Tensor<T>> for Tensor
where
    T: tenferro_algebra::Scalar
        + crate::DynTensorTyped
        + tenferro_internal_ad_core::DynAdTensorTyped
        + 'static,
{
    fn from(value: tenferro_tensor::Tensor<T>) -> Self {
        Self::from_tensor(value)
    }
}

impl<T> From<StructuredTensor<T>> for Tensor
where
    T: tenferro_algebra::Scalar
        + crate::DynTensorTyped
        + tenferro_internal_ad_core::DynAdTensorTyped
        + 'static,
{
    fn from(value: StructuredTensor<T>) -> Self {
        Self::from_structured(value)
    }
}
