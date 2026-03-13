use num_complex::{Complex32, Complex64};
use tenferro_tensor::Tensor;

use super::dyn_scalar::{DynScalar, ScalarType};
use super::tensor_ops::{tensor_map_binary_typed, tensor_map_unary_typed, tensor_max_typed};
use crate::{Error, Result};

/// Runtime tensor wrapper for a fixed supported dtype set.
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
    F32(Tensor<f32>),
    F64(Tensor<f64>),
    C32(Tensor<Complex32>),
    C64(Tensor<Complex64>),
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

    /// Returns dimensions of the underlying tensor.
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
            Self::F32(t) => t.dims(),
            Self::F64(t) => t.dims(),
            Self::C32(t) => t.dims(),
            Self::C64(t) => t.dims(),
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

    /// Returns typed tensor ref when dtype is `f32`.
    pub fn as_f32(&self) -> Option<&Tensor<f32>> {
        if let Self::F32(t) = self {
            Some(t)
        } else {
            None
        }
    }

    /// Returns typed tensor ref when dtype is `f64`.
    pub fn as_f64(&self) -> Option<&Tensor<f64>> {
        if let Self::F64(t) = self {
            Some(t)
        } else {
            None
        }
    }

    /// Returns typed tensor ref when dtype is `Complex32`.
    pub fn as_c32(&self) -> Option<&Tensor<Complex32>> {
        if let Self::C32(t) = self {
            Some(t)
        } else {
            None
        }
    }

    /// Returns typed tensor ref when dtype is `Complex64`.
    pub fn as_c64(&self) -> Option<&Tensor<Complex64>> {
        if let Self::C64(t) = self {
            Some(t)
        } else {
            None
        }
    }

    /// Element-wise subtraction with dtype/shape checks.
    pub fn try_sub(&self, rhs: &Self) -> Result<Self> {
        match (self, rhs) {
            (Self::F32(a), Self::F32(b)) => {
                Ok(Self::F32(tensor_map_binary_typed(a, b, |x, y| x - y)?))
            }
            (Self::F64(a), Self::F64(b)) => {
                Ok(Self::F64(tensor_map_binary_typed(a, b, |x, y| x - y)?))
            }
            (Self::C32(a), Self::C32(b)) => {
                Ok(Self::C32(tensor_map_binary_typed(a, b, |x, y| x - y)?))
            }
            (Self::C64(a), Self::C64(b)) => {
                Ok(Self::C64(tensor_map_binary_typed(a, b, |x, y| x - y)?))
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
            Self::F32(a) => Ok(Self::F32(tensor_map_unary_typed(a, |x| x.abs())?)),
            Self::F64(a) => Ok(Self::F64(tensor_map_unary_typed(a, |x| x.abs())?)),
            Self::C32(a) => Ok(Self::F32(tensor_map_unary_typed(a, |z| z.norm())?)),
            Self::C64(a) => Ok(Self::F64(tensor_map_unary_typed(a, |z| z.norm())?)),
        }
    }

    /// Maximum element value.
    pub fn max(&self) -> Result<DynScalar> {
        match self {
            Self::F32(t) => Ok(DynScalar::F32(tensor_max_typed(t)?)),
            Self::F64(t) => Ok(DynScalar::F64(tensor_max_typed(t)?)),
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

macro_rules! impl_dyn_tensor_from {
    ($variant:ident, $ty:ty) => {
        impl From<Tensor<$ty>> for DynTensor {
            fn from(value: Tensor<$ty>) -> Self {
                Self::$variant(value)
            }
        }
    };
}

impl_dyn_tensor_from!(F32, f32);
impl_dyn_tensor_from!(F64, f64);
impl_dyn_tensor_from!(C32, Complex32);
impl_dyn_tensor_from!(C64, Complex64);
