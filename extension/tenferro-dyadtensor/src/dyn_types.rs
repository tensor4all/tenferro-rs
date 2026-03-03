use num_complex::{Complex32, Complex64};
use tenferro_tensor::Tensor;

use crate::{AdMode, AdTensor, AdValue};

/// Runtime scalar type tag used by all `Dyn*` wrappers.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::ScalarType;
///
/// assert_eq!(ScalarType::F64, ScalarType::F64);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    /// `f32`
    F32,
    /// `f64`
    F64,
    /// `Complex32`
    C32,
    /// `Complex64`
    C64,
}

/// Runtime scalar wrapper for a fixed supported dtype set.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{DynScalar, ScalarType};
///
/// let x: DynScalar = 2.0_f32.into();
/// assert_eq!(x.scalar_type(), ScalarType::F32);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DynScalar {
    F32(f32),
    F64(f64),
    C32(Complex32),
    C64(Complex64),
}

impl DynScalar {
    /// Returns runtime scalar type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{DynScalar, ScalarType};
    ///
    /// let x = DynScalar::F64(1.0);
    /// assert_eq!(x.scalar_type(), ScalarType::F64);
    /// ```
    pub fn scalar_type(&self) -> ScalarType {
        match self {
            Self::F32(_) => ScalarType::F32,
            Self::F64(_) => ScalarType::F64,
            Self::C32(_) => ScalarType::C32,
            Self::C64(_) => ScalarType::C64,
        }
    }

    /// Returns the `f32` value when this scalar is `F32`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::DynScalar;
    ///
    /// let x = DynScalar::F32(3.0);
    /// assert_eq!(x.as_f32(), Some(3.0));
    /// ```
    pub fn as_f32(&self) -> Option<f32> {
        if let Self::F32(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Returns the `f64` value when this scalar is `F64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::DynScalar;
    ///
    /// let x = DynScalar::F64(3.0);
    /// assert_eq!(x.as_f64(), Some(3.0));
    /// ```
    pub fn as_f64(&self) -> Option<f64> {
        if let Self::F64(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Returns the `Complex32` value when this scalar is `C32`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::DynScalar;
    /// use num_complex::Complex32;
    ///
    /// let x = DynScalar::C32(Complex32::new(1.0, 2.0));
    /// assert_eq!(x.as_c32(), Some(Complex32::new(1.0, 2.0)));
    /// ```
    pub fn as_c32(&self) -> Option<Complex32> {
        if let Self::C32(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Returns the `Complex64` value when this scalar is `C64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::DynScalar;
    /// use num_complex::Complex64;
    ///
    /// let x = DynScalar::C64(Complex64::new(1.0, 2.0));
    /// assert_eq!(x.as_c64(), Some(Complex64::new(1.0, 2.0)));
    /// ```
    pub fn as_c64(&self) -> Option<Complex64> {
        if let Self::C64(v) = self {
            Some(*v)
        } else {
            None
        }
    }
}

impl From<f32> for DynScalar {
    fn from(value: f32) -> Self {
        Self::F32(value)
    }
}

impl From<f64> for DynScalar {
    fn from(value: f64) -> Self {
        Self::F64(value)
    }
}

impl From<Complex32> for DynScalar {
    fn from(value: Complex32) -> Self {
        Self::C32(value)
    }
}

impl From<Complex64> for DynScalar {
    fn from(value: Complex64) -> Self {
        Self::C64(value)
    }
}

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
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynTensor = t.into();
    /// assert_eq!(x.ndim(), 1);
    /// ```
    pub fn ndim(&self) -> usize {
        self.dims().len()
    }

    /// Returns number of elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynTensor = t.into();
    /// assert_eq!(x.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.dims().iter().product()
    }

    /// Returns true when tensor has zero elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[], &[0], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynTensor = t.into();
    /// assert!(x.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns typed tensor ref when dtype is `f32`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f32>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynTensor = t.into();
    /// assert!(x.as_f32().is_some());
    /// ```
    pub fn as_f32(&self) -> Option<&Tensor<f32>> {
        if let Self::F32(t) = self {
            Some(t)
        } else {
            None
        }
    }

    /// Returns typed tensor ref when dtype is `f64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynTensor = t.into();
    /// assert!(x.as_f64().is_some());
    /// ```
    pub fn as_f64(&self) -> Option<&Tensor<f64>> {
        if let Self::F64(t) = self {
            Some(t)
        } else {
            None
        }
    }

    /// Returns typed tensor ref when dtype is `Complex32`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::DynTensor;
    /// use num_complex::Complex32;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<Complex32>::from_slice(
    ///     &[Complex32::new(1.0, 0.0)],
    ///     &[1],
    ///     MemoryOrder::ColumnMajor,
    /// )
    /// .unwrap();
    /// let x: DynTensor = t.into();
    /// assert!(x.as_c32().is_some());
    /// ```
    pub fn as_c32(&self) -> Option<&Tensor<Complex32>> {
        if let Self::C32(t) = self {
            Some(t)
        } else {
            None
        }
    }

    /// Returns typed tensor ref when dtype is `Complex64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::DynTensor;
    /// use num_complex::Complex64;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<Complex64>::from_slice(
    ///     &[Complex64::new(1.0, 0.0)],
    ///     &[1],
    ///     MemoryOrder::ColumnMajor,
    /// )
    /// .unwrap();
    /// let x: DynTensor = t.into();
    /// assert!(x.as_c64().is_some());
    /// ```
    pub fn as_c64(&self) -> Option<&Tensor<Complex64>> {
        if let Self::C64(t) = self {
            Some(t)
        } else {
            None
        }
    }
}

impl From<Tensor<f32>> for DynTensor {
    fn from(value: Tensor<f32>) -> Self {
        Self::F32(value)
    }
}

impl From<Tensor<f64>> for DynTensor {
    fn from(value: Tensor<f64>) -> Self {
        Self::F64(value)
    }
}

impl From<Tensor<Complex32>> for DynTensor {
    fn from(value: Tensor<Complex32>) -> Self {
        Self::C32(value)
    }
}

impl From<Tensor<Complex64>> for DynTensor {
    fn from(value: Tensor<Complex64>) -> Self {
        Self::C64(value)
    }
}

/// Runtime AD scalar value wrapper.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{AdMode, AdValue, DynAdValue};
///
/// let x: DynAdValue = AdValue::forward(2.0_f64, 1.0_f64).into();
/// assert_eq!(x.mode(), AdMode::Forward);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum DynAdValue {
    F32(AdValue<f32>),
    F64(AdValue<f64>),
    C32(AdValue<Complex32>),
    C64(AdValue<Complex64>),
}

impl DynAdValue {
    /// Returns runtime scalar type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdValue, DynAdValue, ScalarType};
    ///
    /// let x: DynAdValue = AdValue::primal(1.0_f32).into();
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

    /// Returns AD mode.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdMode, AdValue, DynAdValue};
    ///
    /// let x: DynAdValue = AdValue::forward(2.0_f64, 1.0_f64).into();
    /// assert_eq!(x.mode(), AdMode::Forward);
    /// ```
    pub fn mode(&self) -> AdMode {
        match self {
            Self::F32(v) => v.mode(),
            Self::F64(v) => v.mode(),
            Self::C32(v) => v.mode(),
            Self::C64(v) => v.mode(),
        }
    }

    /// Returns primal part as dynamic scalar.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdValue, DynAdValue, DynScalar};
    ///
    /// let x: DynAdValue = AdValue::primal(3.0_f64).into();
    /// assert_eq!(x.primal(), DynScalar::F64(3.0));
    /// ```
    pub fn primal(&self) -> DynScalar {
        match self {
            Self::F32(v) => DynScalar::F32(*v.primal_ref()),
            Self::F64(v) => DynScalar::F64(*v.primal_ref()),
            Self::C32(v) => DynScalar::C32(*v.primal_ref()),
            Self::C64(v) => DynScalar::C64(*v.primal_ref()),
        }
    }

    /// Returns tangent part as dynamic scalar when available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdValue, DynAdValue, DynScalar};
    ///
    /// let x: DynAdValue = AdValue::forward(3.0_f64, 0.5_f64).into();
    /// assert_eq!(x.tangent(), Some(DynScalar::F64(0.5)));
    /// ```
    pub fn tangent(&self) -> Option<DynScalar> {
        match self {
            Self::F32(v) => v.tangent_ref().copied().map(DynScalar::F32),
            Self::F64(v) => v.tangent_ref().copied().map(DynScalar::F64),
            Self::C32(v) => v.tangent_ref().copied().map(DynScalar::C32),
            Self::C64(v) => v.tangent_ref().copied().map(DynScalar::C64),
        }
    }

    /// Returns typed AD value ref when dtype is `f32`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdValue, DynAdValue};
    ///
    /// let x: DynAdValue = AdValue::primal(1.0_f32).into();
    /// assert!(x.as_f32().is_some());
    /// ```
    pub fn as_f32(&self) -> Option<&AdValue<f32>> {
        if let Self::F32(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns typed AD value ref when dtype is `f64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdValue, DynAdValue};
    ///
    /// let x: DynAdValue = AdValue::primal(1.0_f64).into();
    /// assert!(x.as_f64().is_some());
    /// ```
    pub fn as_f64(&self) -> Option<&AdValue<f64>> {
        if let Self::F64(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns typed AD value ref when dtype is `Complex32`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdValue, DynAdValue};
    /// use num_complex::Complex32;
    ///
    /// let x: DynAdValue = AdValue::primal(Complex32::new(1.0, 0.0)).into();
    /// assert!(x.as_c32().is_some());
    /// ```
    pub fn as_c32(&self) -> Option<&AdValue<Complex32>> {
        if let Self::C32(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns typed AD value ref when dtype is `Complex64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdValue, DynAdValue};
    /// use num_complex::Complex64;
    ///
    /// let x: DynAdValue = AdValue::primal(Complex64::new(1.0, 0.0)).into();
    /// assert!(x.as_c64().is_some());
    /// ```
    pub fn as_c64(&self) -> Option<&AdValue<Complex64>> {
        if let Self::C64(v) = self {
            Some(v)
        } else {
            None
        }
    }
}

impl From<AdValue<f32>> for DynAdValue {
    fn from(value: AdValue<f32>) -> Self {
        Self::F32(value)
    }
}

impl From<AdValue<f64>> for DynAdValue {
    fn from(value: AdValue<f64>) -> Self {
        Self::F64(value)
    }
}

impl From<AdValue<Complex32>> for DynAdValue {
    fn from(value: AdValue<Complex32>) -> Self {
        Self::C32(value)
    }
}

impl From<AdValue<Complex64>> for DynAdValue {
    fn from(value: AdValue<Complex64>) -> Self {
        Self::C64(value)
    }
}

/// Runtime AD tensor wrapper.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{AdTensor, DynAdTensor, ScalarType};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let t = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
/// let x: DynAdTensor = AdTensor::new_primal(t).into();
/// assert_eq!(x.scalar_type(), ScalarType::F64);
/// ```
#[derive(Clone)]
pub enum DynAdTensor {
    F32(AdTensor<f32>),
    F64(AdTensor<f64>),
    C32(AdTensor<Complex32>),
    C64(AdTensor<Complex64>),
}

impl DynAdTensor {
    /// Returns runtime scalar type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdTensor, DynAdTensor, ScalarType};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f32>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynAdTensor = AdTensor::new_primal(t).into();
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

    /// Returns AD mode.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdMode, AdTensor, DynAdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynAdTensor = AdTensor::new_primal(t).into();
    /// assert_eq!(x.mode(), AdMode::Primal);
    /// ```
    pub fn mode(&self) -> AdMode {
        match self {
            Self::F32(v) => v.mode(),
            Self::F64(v) => v.mode(),
            Self::C32(v) => v.mode(),
            Self::C64(v) => v.mode(),
        }
    }

    /// Returns primal tensor dimensions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdTensor, DynAdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynAdTensor = AdTensor::new_primal(t).into();
    /// assert_eq!(x.dims(), &[2]);
    /// ```
    pub fn dims(&self) -> &[usize] {
        match self {
            Self::F32(v) => v.dims(),
            Self::F64(v) => v.dims(),
            Self::C32(v) => v.dims(),
            Self::C64(v) => v.dims(),
        }
    }

    /// Returns primal tensor rank.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdTensor, DynAdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynAdTensor = AdTensor::new_primal(t).into();
    /// assert_eq!(x.ndim(), 1);
    /// ```
    pub fn ndim(&self) -> usize {
        self.dims().len()
    }

    /// Returns primal tensor element count.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdTensor, DynAdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynAdTensor = AdTensor::new_primal(t).into();
    /// assert_eq!(x.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.dims().iter().product()
    }

    /// Returns true when primal tensor has zero elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdTensor, DynAdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[], &[0], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynAdTensor = AdTensor::new_primal(t).into();
    /// assert!(x.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns typed AD tensor ref when dtype is `f32`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdTensor, DynAdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f32>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynAdTensor = AdTensor::new_primal(t).into();
    /// assert!(x.as_f32().is_some());
    /// ```
    pub fn as_f32(&self) -> Option<&AdTensor<f32>> {
        if let Self::F32(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns typed AD tensor ref when dtype is `f64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdTensor, DynAdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynAdTensor = AdTensor::new_primal(t).into();
    /// assert!(x.as_f64().is_some());
    /// ```
    pub fn as_f64(&self) -> Option<&AdTensor<f64>> {
        if let Self::F64(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns typed AD tensor ref when dtype is `Complex32`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdTensor, DynAdTensor};
    /// use num_complex::Complex32;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<Complex32>::from_slice(
    ///     &[Complex32::new(1.0, 0.0)],
    ///     &[1],
    ///     MemoryOrder::ColumnMajor,
    /// )
    /// .unwrap();
    /// let x: DynAdTensor = AdTensor::new_primal(t).into();
    /// assert!(x.as_c32().is_some());
    /// ```
    pub fn as_c32(&self) -> Option<&AdTensor<Complex32>> {
        if let Self::C32(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns typed AD tensor ref when dtype is `Complex64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdTensor, DynAdTensor};
    /// use num_complex::Complex64;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<Complex64>::from_slice(
    ///     &[Complex64::new(1.0, 0.0)],
    ///     &[1],
    ///     MemoryOrder::ColumnMajor,
    /// )
    /// .unwrap();
    /// let x: DynAdTensor = AdTensor::new_primal(t).into();
    /// assert!(x.as_c64().is_some());
    /// ```
    pub fn as_c64(&self) -> Option<&AdTensor<Complex64>> {
        if let Self::C64(v) = self {
            Some(v)
        } else {
            None
        }
    }
}

impl From<AdTensor<f32>> for DynAdTensor {
    fn from(value: AdTensor<f32>) -> Self {
        Self::F32(value)
    }
}

impl From<AdTensor<f64>> for DynAdTensor {
    fn from(value: AdTensor<f64>) -> Self {
        Self::F64(value)
    }
}

impl From<AdTensor<Complex32>> for DynAdTensor {
    fn from(value: AdTensor<Complex32>) -> Self {
        Self::C32(value)
    }
}

impl From<AdTensor<Complex64>> for DynAdTensor {
    fn from(value: AdTensor<Complex64>) -> Self {
        Self::C64(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_tensor::MemoryOrder;

    #[test]
    fn dyn_scalar_metadata() {
        let x: DynScalar = 1.0_f64.into();
        assert_eq!(x.scalar_type(), ScalarType::F64);
        assert_eq!(x.as_f64(), Some(1.0));
    }

    #[test]
    fn dyn_ad_value_mode_and_tangent() {
        let x: DynAdValue = AdValue::forward(2.0_f32, 0.5_f32).into();
        assert_eq!(x.scalar_type(), ScalarType::F32);
        assert_eq!(x.mode(), AdMode::Forward);
        assert_eq!(x.tangent(), Some(DynScalar::F32(0.5)));
    }

    #[test]
    fn dyn_tensor_and_dyn_ad_tensor_dims() {
        let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        let d: DynTensor = t.clone().into();
        assert_eq!(d.dims(), &[2]);

        let ad = AdTensor::new_primal(t);
        let dad: DynAdTensor = ad.into();
        assert_eq!(dad.dims(), &[2]);
        assert_eq!(dad.mode(), AdMode::Primal);
    }
}
