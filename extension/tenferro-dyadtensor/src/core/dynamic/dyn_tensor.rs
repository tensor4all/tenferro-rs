use num_complex::{Complex32, Complex64};
use tenferro_tensor::Tensor;

use super::dyn_scalar::{DynScalar, ScalarType};
use super::tensor_ops::{tensor_map_binary_typed, tensor_map_unary_typed, tensor_max_typed};
use crate::{Error, Result, StructuredTensor};

/// Runtime tensor wrapper for a fixed supported dtype set.
///
/// `DynTensor` is the canonical dynamic primal tensor type in
/// `tenferro-dyadtensor`. Each variant carries a `StructuredTensor<T>`, so dense
/// tensors and structured special cases such as `Diag` share the same public
/// container.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{DynTensor, ScalarType};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
/// let x: DynTensor = t.into();
/// assert_eq!(x.scalar_type(), ScalarType::F64);
/// ```
#[derive(Clone)]
pub enum DynTensor {
    F32(StructuredTensor<f32>),
    F64(StructuredTensor<f64>),
    C32(StructuredTensor<Complex32>),
    C64(StructuredTensor<Complex64>),
}

impl DynTensor {
    /// Returns runtime scalar type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{DynTensor, ScalarType};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f32>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynTensor = t.into();
    /// assert_eq!(x.scalar_type(), ScalarType::F32);
    /// ```
    pub fn scalar_type(&self) -> ScalarType {
        match self {
            Self::F32(_) => ScalarType::F32,
            Self::F64(_) => ScalarType::F64,
            Self::C32(_) => ScalarType::C32,
            Self::C64(_) => ScalarType::C64,
        }
    }

    /// Returns logical dimensions of the underlying tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynTensor = t.into();
    /// assert_eq!(x.dims(), &[2]);
    /// ```
    pub fn dims(&self) -> &[usize] {
        match self {
            Self::F32(t) => t.logical_dims(),
            Self::F64(t) => t.logical_dims(),
            Self::C32(t) => t.logical_dims(),
            Self::C64(t) => t.logical_dims(),
        }
    }

    /// Returns axis equivalence classes of the structured layout.
    pub fn axis_classes(&self) -> &[usize] {
        match self {
            Self::F32(t) => t.axis_classes(),
            Self::F64(t) => t.axis_classes(),
            Self::C32(t) => t.axis_classes(),
            Self::C64(t) => t.axis_classes(),
        }
    }

    /// Returns `true` when the structured payload is dense.
    pub fn is_dense(&self) -> bool {
        match self {
            Self::F32(t) => t.is_dense(),
            Self::F64(t) => t.is_dense(),
            Self::C32(t) => t.is_dense(),
            Self::C64(t) => t.is_dense(),
        }
    }

    /// Returns `true` when the structured payload is diagonal.
    pub fn is_diag(&self) -> bool {
        match self {
            Self::F32(t) => t.is_diag(),
            Self::F64(t) => t.is_diag(),
            Self::C32(t) => t.is_diag(),
            Self::C64(t) => t.is_diag(),
        }
    }

    /// Returns rank.
    pub fn ndim(&self) -> usize {
        self.dims().len()
    }

    /// Returns number of elements.
    pub fn len(&self) -> usize {
        self.dims().iter().product()
    }

    /// Returns true when tensor has zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns typed structured tensor ref when dtype is `f32`.
    pub fn as_f32(&self) -> Option<&StructuredTensor<f32>> {
        if let Self::F32(t) = self {
            Some(t)
        } else {
            None
        }
    }

    /// Returns typed structured tensor ref when dtype is `f64`.
    pub fn as_f64(&self) -> Option<&StructuredTensor<f64>> {
        if let Self::F64(t) = self {
            Some(t)
        } else {
            None
        }
    }

    /// Returns typed structured tensor ref when dtype is `Complex32`.
    pub fn as_c32(&self) -> Option<&StructuredTensor<Complex32>> {
        if let Self::C32(t) = self {
            Some(t)
        } else {
            None
        }
    }

    /// Returns typed structured tensor ref when dtype is `Complex64`.
    pub fn as_c64(&self) -> Option<&StructuredTensor<Complex64>> {
        if let Self::C64(t) = self {
            Some(t)
        } else {
            None
        }
    }

    /// Returns typed payload ref when dtype is `f32`.
    pub fn payload_f32(&self) -> Option<&Tensor<f32>> {
        self.as_f32().map(StructuredTensor::payload)
    }

    /// Returns typed payload ref when dtype is `f64`.
    pub fn payload_f64(&self) -> Option<&Tensor<f64>> {
        self.as_f64().map(StructuredTensor::payload)
    }

    /// Returns typed payload ref when dtype is `Complex32`.
    pub fn payload_c32(&self) -> Option<&Tensor<Complex32>> {
        self.as_c32().map(StructuredTensor::payload)
    }

    /// Returns typed payload ref when dtype is `Complex64`.
    pub fn payload_c64(&self) -> Option<&Tensor<Complex64>> {
        self.as_c64().map(StructuredTensor::payload)
    }

    /// Element-wise subtraction with dtype/shape checks.
    pub fn try_sub(&self, rhs: &Self) -> Result<Self> {
        match (self, rhs) {
            (Self::F32(a), Self::F32(b)) => {
                ensure_same_layout("try_sub", a, b)?;
                Ok(Self::F32(a.with_payload_like(tensor_map_binary_typed(
                    a.payload(),
                    b.payload(),
                    |x, y| x - y,
                )?)?))
            }
            (Self::F64(a), Self::F64(b)) => {
                ensure_same_layout("try_sub", a, b)?;
                Ok(Self::F64(a.with_payload_like(tensor_map_binary_typed(
                    a.payload(),
                    b.payload(),
                    |x, y| x - y,
                )?)?))
            }
            (Self::C32(a), Self::C32(b)) => {
                ensure_same_layout("try_sub", a, b)?;
                Ok(Self::C32(a.with_payload_like(tensor_map_binary_typed(
                    a.payload(),
                    b.payload(),
                    |x, y| x - y,
                )?)?))
            }
            (Self::C64(a), Self::C64(b)) => {
                ensure_same_layout("try_sub", a, b)?;
                Ok(Self::C64(a.with_payload_like(tensor_map_binary_typed(
                    a.payload(),
                    b.payload(),
                    |x, y| x - y,
                )?)?))
            }
            _ => Err(Error::InvalidAdTensor {
                message: format!(
                    "dtype mismatch in try_sub: lhs={:?}, rhs={:?}",
                    self.scalar_type(),
                    rhs.scalar_type()
                ),
            }),
        }
    }

    /// Element-wise absolute value.
    pub fn abs_tensor(&self) -> Result<Self> {
        match self {
            Self::F32(a) => Ok(Self::F32(
                a.with_payload_like(tensor_map_unary_typed(a.payload(), |x| x.abs())?)?,
            )),
            Self::F64(a) => Ok(Self::F64(
                a.with_payload_like(tensor_map_unary_typed(a.payload(), |x| x.abs())?)?,
            )),
            Self::C32(a) => Ok(Self::F32(StructuredTensor::new(
                a.logical_dims().to_vec(),
                a.axis_classes().to_vec(),
                tensor_map_unary_typed(a.payload(), |z| z.norm())?,
            )?)),
            Self::C64(a) => Ok(Self::F64(StructuredTensor::new(
                a.logical_dims().to_vec(),
                a.axis_classes().to_vec(),
                tensor_map_unary_typed(a.payload(), |z| z.norm())?,
            )?)),
        }
    }

    /// Maximum element value.
    pub fn max(&self) -> Result<DynScalar> {
        match self {
            Self::F32(t) => Ok(DynScalar::F32(tensor_max_typed(t.payload())?)),
            Self::F64(t) => Ok(DynScalar::F64(tensor_max_typed(t.payload())?)),
            Self::C32(_) | Self::C64(_) => Err(Error::InvalidAdTensor {
                message: "max is undefined for complex tensors; call abs_tensor() first"
                    .to_string(),
            }),
        }
    }

    /// Maximum element value as `f64` (real tensors only).
    pub fn max_as_f64(&self) -> Result<f64> {
        match self.max()? {
            DynScalar::F32(v) => Ok(v as f64),
            DynScalar::F64(v) => Ok(v),
            DynScalar::C32(_) | DynScalar::C64(_) => Err(Error::InvalidAdTensor {
                message: "max_as_f64 expects a real tensor".to_string(),
            }),
        }
    }

    /// Computes `max(abs(self - rhs))` without flattening to raw slices.
    pub fn max_abs_diff(&self, rhs: &Self) -> Result<f64> {
        self.try_sub(rhs)?.abs_tensor()?.max_as_f64()
    }
}

fn ensure_same_layout<T>(
    op_name: &'static str,
    lhs: &StructuredTensor<T>,
    rhs: &StructuredTensor<T>,
) -> Result<()>
where
    T: tenferro_algebra::Scalar,
{
    if lhs.logical_dims() != rhs.logical_dims() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "{op_name} requires matching logical_dims, got lhs={:?}, rhs={:?}",
                lhs.logical_dims(),
                rhs.logical_dims()
            ),
        });
    }
    if lhs.axis_classes() != rhs.axis_classes() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "{op_name} requires matching axis_classes, got lhs={:?}, rhs={:?}",
                lhs.axis_classes(),
                rhs.axis_classes()
            ),
        });
    }
    Ok(())
}

macro_rules! impl_dyn_tensor_from {
    ($variant:ident, $ty:ty) => {
        impl From<Tensor<$ty>> for DynTensor {
            fn from(value: Tensor<$ty>) -> Self {
                Self::$variant(StructuredTensor::from_dense(value))
            }
        }

        impl From<StructuredTensor<$ty>> for DynTensor {
            fn from(value: StructuredTensor<$ty>) -> Self {
                Self::$variant(value)
            }
        }
    };
}

impl_dyn_tensor_from!(F32, f32);
impl_dyn_tensor_from!(F64, f64);
impl_dyn_tensor_from!(C32, Complex32);
impl_dyn_tensor_from!(C64, Complex64);
