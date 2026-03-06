use chainrules_scalarops::{self, ScalarAd};
use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};
use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use std::sync::atomic::{AtomicU64, Ordering};
use tenferro_algebra::Scalar;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::ad_value::{map_ad_value_mixed_linear, map_ad_value_same_type_linear};
use crate::structured::StructuredTensor;
use crate::{reverse_tape, AdMode, AdScalar, AdTensor, AdValue, Error, NodeId, Result, TapeId};

static NEXT_AD_TENSOR_NODE_ID: AtomicU64 = AtomicU64::new(1_u64 << 61);

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

trait AbsAsF64 {
    fn abs_as_f64(self) -> f64;
}

impl AbsAsF64 for f32 {
    fn abs_as_f64(self) -> f64 {
        self.abs() as f64
    }
}

impl AbsAsF64 for f64 {
    fn abs_as_f64(self) -> f64 {
        self.abs()
    }
}

impl AbsAsF64 for Complex32 {
    fn abs_as_f64(self) -> f64 {
        self.norm() as f64
    }
}

impl AbsAsF64 for Complex64 {
    fn abs_as_f64(self) -> f64 {
        self.norm()
    }
}

fn unflatten_index_column_major(mut flat: usize, dims: &[usize], out: &mut [usize]) {
    for (axis, &dim) in dims.iter().enumerate() {
        out[axis] = flat % dim;
        flat /= dim;
    }
}

fn tensor_element<T: Scalar + Copy>(tensor: &Tensor<T>, indices: &[usize]) -> Result<T> {
    if indices.len() != tensor.dims().len() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "index rank mismatch: indices has rank {}, tensor has rank {}",
                indices.len(),
                tensor.dims().len()
            ),
        });
    }

    let mut offset = tensor.offset();
    for (axis, &idx) in indices.iter().enumerate() {
        let dim = tensor.dims()[axis];
        if idx >= dim {
            return Err(Error::InvalidAdTensor {
                message: format!(
                    "index out of bounds on axis {}: idx={} >= dim={}",
                    axis, idx, dim
                ),
            });
        }
        offset += (idx as isize) * tensor.strides()[axis];
    }

    let buffer = tensor
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::InvalidAdTensor {
            message: "tensor buffer is not host-accessible".to_string(),
        })?;
    let pos = usize::try_from(offset).map_err(|_| Error::InvalidAdTensor {
        message: format!("negative tensor offset computed: {}", offset),
    })?;
    buffer
        .get(pos)
        .copied()
        .ok_or_else(|| Error::InvalidAdTensor {
            message: format!("computed offset {} is out of buffer bounds", pos),
        })
}

fn tensor_max_abs_diff_typed<T>(lhs: &Tensor<T>, rhs: &Tensor<T>) -> Result<f64>
where
    T: Scalar + Copy + Sub<Output = T> + AbsAsF64,
{
    if lhs.dims() != rhs.dims() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "shape mismatch in max_abs_diff: lhs={:?}, rhs={:?}",
                lhs.dims(),
                rhs.dims()
            ),
        });
    }

    let dims = lhs.dims();
    let total: usize = dims.iter().product();
    if total == 0 {
        return Ok(0.0);
    }

    let mut idx = vec![0usize; dims.len()];
    let mut max_diff = 0.0_f64;
    for flat in 0..total {
        unflatten_index_column_major(flat, dims, &mut idx);
        let lv = tensor_element(lhs, &idx)?;
        let rv = tensor_element(rhs, &idx)?;
        let d = (lv - rv).abs_as_f64();
        if d > max_diff {
            max_diff = d;
        }
    }
    Ok(max_diff)
}

fn tensor_map_binary_typed<T>(
    lhs: &Tensor<T>,
    rhs: &Tensor<T>,
    mut f: impl FnMut(T, T) -> T,
) -> Result<Tensor<T>>
where
    T: Scalar + Copy,
{
    if lhs.dims() != rhs.dims() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "shape mismatch in binary map: lhs={:?}, rhs={:?}",
                lhs.dims(),
                rhs.dims()
            ),
        });
    }

    let dims = lhs.dims().to_vec();
    let total: usize = dims.iter().product();
    let mut idx = vec![0usize; dims.len()];
    let mut out = Vec::with_capacity(total);
    for flat in 0..total {
        unflatten_index_column_major(flat, &dims, &mut idx);
        let lv = tensor_element(lhs, &idx)?;
        let rv = tensor_element(rhs, &idx)?;
        out.push(f(lv, rv));
    }

    Tensor::from_slice(&out, &dims, MemoryOrder::ColumnMajor).map_err(Error::from)
}

fn tensor_map_unary_typed<T, U>(input: &Tensor<T>, mut f: impl FnMut(T) -> U) -> Result<Tensor<U>>
where
    T: Scalar + Copy,
    U: Scalar + Copy,
{
    let dims = input.dims().to_vec();
    let total: usize = dims.iter().product();
    let mut idx = vec![0usize; dims.len()];
    let mut out = Vec::with_capacity(total);
    for flat in 0..total {
        unflatten_index_column_major(flat, &dims, &mut idx);
        let v = tensor_element(input, &idx)?;
        out.push(f(v));
    }

    Tensor::from_slice(&out, &dims, MemoryOrder::ColumnMajor).map_err(Error::from)
}

fn tensor_max_typed<T>(input: &Tensor<T>) -> Result<T>
where
    T: Scalar + Copy + PartialOrd,
{
    if input.is_empty() {
        return Err(Error::InvalidAdTensor {
            message: "max is undefined for empty tensor".to_string(),
        });
    }

    let dims = input.dims().to_vec();
    let total: usize = dims.iter().product();
    let mut idx = vec![0usize; dims.len()];
    unflatten_index_column_major(0, &dims, &mut idx);
    let mut best = tensor_element(input, &idx)?;
    for flat in 1..total {
        unflatten_index_column_major(flat, &dims, &mut idx);
        let v = tensor_element(input, &idx)?;
        if v > best {
            best = v;
        }
    }
    Ok(best)
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

    /// Element-wise subtraction with dtype/shape checks.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let a: DynTensor =
    ///     Tensor::<f64>::from_slice(&[3.0, 1.0], &[2], MemoryOrder::ColumnMajor).unwrap().into();
    /// let b: DynTensor =
    ///     Tensor::<f64>::from_slice(&[1.0, 1.5], &[2], MemoryOrder::ColumnMajor).unwrap().into();
    /// let c = a.try_sub(&b).unwrap();
    /// assert_eq!(c.scalar_type(), tenferro_dyadtensor::ScalarType::F64);
    /// ```
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
    ///
    /// For complex tensors, returns a real tensor (`C32 -> F32`, `C64 -> F64`)
    /// containing magnitudes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{DynTensor, ScalarType};
    /// use num_complex::Complex64;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let x: DynTensor = Tensor::<Complex64>::from_slice(
    ///     &[Complex64::new(3.0, 4.0)],
    ///     &[1],
    ///     MemoryOrder::ColumnMajor,
    /// )
    /// .unwrap()
    /// .into();
    /// let y = x.abs_tensor().unwrap();
    /// assert_eq!(y.scalar_type(), ScalarType::F64);
    /// ```
    pub fn abs_tensor(&self) -> Result<Self> {
        match self {
            Self::F32(a) => Ok(Self::F32(tensor_map_unary_typed(a, |x| x.abs())?)),
            Self::F64(a) => Ok(Self::F64(tensor_map_unary_typed(a, |x| x.abs())?)),
            Self::C32(a) => Ok(Self::F32(tensor_map_unary_typed(a, |z| z.norm())?)),
            Self::C64(a) => Ok(Self::F64(tensor_map_unary_typed(a, |z| z.norm())?)),
        }
    }

    /// Maximum element value.
    ///
    /// This operation is defined only for real dtypes.
    /// For complex tensors, call [`Self::abs_tensor`] first.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{DynScalar, DynTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let x: DynTensor =
    ///     Tensor::<f64>::from_slice(&[1.0, 3.0, 2.0], &[3], MemoryOrder::ColumnMajor).unwrap().into();
    /// assert_eq!(x.max().unwrap(), DynScalar::F64(3.0));
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let x: DynTensor =
    ///     Tensor::<f64>::from_slice(&[1.0, 3.0, 2.0], &[3], MemoryOrder::ColumnMajor).unwrap().into();
    /// assert!((x.max_as_f64().unwrap() - 3.0).abs() < 1e-12);
    /// ```
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
    ///
    /// The computation follows tensor logical indexing (`dims` + `strides`),
    /// so it is robust to non-contiguous views and memory-order differences.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let a = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let b = Tensor::<f64>::from_slice(&[1.0, 1.5], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let da: DynTensor = a.into();
    /// let db: DynTensor = b.into();
    /// let diff = da.max_abs_diff(&db).unwrap();
    /// assert!((diff - 0.5).abs() < 1e-12);
    /// ```
    pub fn max_abs_diff(&self, rhs: &Self) -> Result<f64> {
        self.try_sub(rhs)?.abs_tensor()?.max_as_f64()
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
/// use tenferro_dyadtensor::{AdMode, AdValue, DynAdScalar};
///
/// let x: DynAdScalar = AdValue::forward(2.0_f64, 1.0_f64).into();
/// assert_eq!(x.mode(), AdMode::Forward);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum DynAdScalar {
    F32(AdValue<f32>),
    F64(AdValue<f64>),
    C32(AdValue<Complex32>),
    C64(AdValue<Complex64>),
}

#[derive(Clone, Copy)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinaryOp {
    fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
        }
    }
}

fn promote_f32_to_c32(value: AdValue<f32>, op_name: &'static str) -> AdValue<Complex32> {
    map_ad_value_mixed_linear(value, op_name, |x| Complex32::new(x, 0.0), |z| z.re)
}

fn promote_f64_to_c64(value: AdValue<f64>, op_name: &'static str) -> AdValue<Complex64> {
    map_ad_value_mixed_linear(value, op_name, |x| Complex64::new(x, 0.0), |z| z.re)
}

fn embed_f32_to_c32_imag(value: AdValue<f32>, op_name: &'static str) -> AdValue<Complex32> {
    map_ad_value_mixed_linear(value, op_name, |y| Complex32::new(0.0, y), |z| z.im)
}

fn embed_f64_to_c64_imag(value: AdValue<f64>, op_name: &'static str) -> AdValue<Complex64> {
    map_ad_value_mixed_linear(value, op_name, |y| Complex64::new(0.0, y), |z| z.im)
}

fn apply_binary_ad<T: ScalarAd + 'static>(
    lhs: AdValue<T>,
    rhs: AdValue<T>,
    op: BinaryOp,
) -> AdValue<T> {
    let lhs = AdScalar::from(lhs);
    let rhs = AdScalar::from(rhs);
    match op {
        BinaryOp::Add => (lhs + rhs).into_value(),
        BinaryOp::Sub => (lhs - rhs).into_value(),
        BinaryOp::Mul => (lhs * rhs).into_value(),
        BinaryOp::Div => (lhs / rhs).into_value(),
    }
}

fn check_reverse_tape_compatibility<T>(
    lhs: &AdValue<T>,
    rhs: &AdValue<T>,
    op: BinaryOp,
) -> Result<()> {
    match (lhs.tape_id(), rhs.tape_id()) {
        (Some(lhs_tape), Some(rhs_tape)) if lhs_tape != rhs_tape => Err(Error::InvalidAdScalar {
            message: format!(
                "{}: reverse-mode tape mismatch (lhs={}, rhs={})",
                op.name(),
                lhs_tape.0,
                rhs_tape.0
            ),
        }),
        _ => Ok(()),
    }
}

fn checked_apply_binary_ad<T: ScalarAd + 'static>(
    lhs: AdValue<T>,
    rhs: AdValue<T>,
    op: BinaryOp,
) -> Result<AdValue<T>> {
    check_reverse_tape_compatibility(&lhs, &rhs, op)?;
    Ok(apply_binary_ad(lhs, rhs, op))
}

fn unsupported_binary_pair(op: BinaryOp, lhs: ScalarType, rhs: ScalarType) -> Error {
    Error::InvalidAdScalar {
        message: format!(
            "unsupported dtype pair for `{}`: lhs={lhs:?}, rhs={rhs:?}",
            op.name()
        ),
    }
}

fn try_binary_dyn(lhs: DynAdScalar, rhs: DynAdScalar, op: BinaryOp) -> Result<DynAdScalar> {
    let lhs_ty = lhs.scalar_type();
    let rhs_ty = rhs.scalar_type();
    match (lhs, rhs) {
        (DynAdScalar::F32(a), DynAdScalar::F32(b)) => {
            Ok(DynAdScalar::F32(checked_apply_binary_ad(a, b, op)?))
        }
        (DynAdScalar::F64(a), DynAdScalar::F64(b)) => {
            Ok(DynAdScalar::F64(checked_apply_binary_ad(a, b, op)?))
        }
        (DynAdScalar::C32(a), DynAdScalar::C32(b)) => {
            Ok(DynAdScalar::C32(checked_apply_binary_ad(a, b, op)?))
        }
        (DynAdScalar::C64(a), DynAdScalar::C64(b)) => {
            Ok(DynAdScalar::C64(checked_apply_binary_ad(a, b, op)?))
        }
        (DynAdScalar::F32(a), DynAdScalar::C32(b)) => Ok(DynAdScalar::C32(
            checked_apply_binary_ad(promote_f32_to_c32(a, op.name()), b, op)?,
        )),
        (DynAdScalar::C32(a), DynAdScalar::F32(b)) => Ok(DynAdScalar::C32(
            checked_apply_binary_ad(a, promote_f32_to_c32(b, op.name()), op)?,
        )),
        (DynAdScalar::F64(a), DynAdScalar::C64(b)) => Ok(DynAdScalar::C64(
            checked_apply_binary_ad(promote_f64_to_c64(a, op.name()), b, op)?,
        )),
        (DynAdScalar::C64(a), DynAdScalar::F64(b)) => Ok(DynAdScalar::C64(
            checked_apply_binary_ad(a, promote_f64_to_c64(b, op.name()), op)?,
        )),
        _ => Err(unsupported_binary_pair(op, lhs_ty, rhs_ty)),
    }
}

impl DynAdScalar {
    /// Creates a real scalar (`f64`) in primal mode.
    pub fn new_real(x: f64) -> Self {
        Self::from(x)
    }

    /// Creates a complex scalar (`Complex64`) in primal mode.
    pub fn new_complex(re: f64, im: f64) -> Self {
        Self::from(Complex64::new(re, im))
    }

    /// Returns runtime scalar type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdValue, DynAdScalar, ScalarType};
    ///
    /// let x: DynAdScalar = AdValue::primal(1.0_f32).into();
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
    /// use tenferro_dyadtensor::{AdMode, AdValue, DynAdScalar};
    ///
    /// let x: DynAdScalar = AdValue::forward(2.0_f64, 1.0_f64).into();
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

    /// Returns reverse-mode node id when available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdValue, DynAdScalar, NodeId, TapeId};
    ///
    /// let x: DynAdScalar = AdValue::reverse(1.0_f64, NodeId(4), TapeId(6), None).into();
    /// assert_eq!(x.node_id(), Some(NodeId(4)));
    /// ```
    pub fn node_id(&self) -> Option<NodeId> {
        match self {
            Self::F32(v) => v.node_id(),
            Self::F64(v) => v.node_id(),
            Self::C32(v) => v.node_id(),
            Self::C64(v) => v.node_id(),
        }
    }

    /// Returns reverse-mode tape id when available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdValue, DynAdScalar, NodeId, TapeId};
    ///
    /// let x: DynAdScalar = AdValue::reverse(1.0_f64, NodeId(4), TapeId(6), None).into();
    /// assert_eq!(x.tape_id(), Some(TapeId(6)));
    /// ```
    pub fn tape_id(&self) -> Option<TapeId> {
        match self {
            Self::F32(v) => v.tape_id(),
            Self::F64(v) => v.tape_id(),
            Self::C32(v) => v.tape_id(),
            Self::C64(v) => v.tape_id(),
        }
    }

    /// Returns primal part as dynamic scalar.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdValue, DynAdScalar, DynScalar};
    ///
    /// let x: DynAdScalar = AdValue::primal(3.0_f64).into();
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

    /// Consumes this scalar and returns the primal value, explicitly dropping AD metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdValue, DynAdScalar, DynScalar};
    ///
    /// let x: DynAdScalar = AdValue::forward(2.0_f64, 0.5_f64).into();
    /// assert_eq!(x.primal_into(), DynScalar::F64(2.0));
    /// ```
    pub fn primal_into(self) -> DynScalar {
        match self {
            Self::F32(v) => DynScalar::F32(match v {
                AdValue::Primal(primal) => primal,
                AdValue::Forward { primal, .. } => primal,
                AdValue::Reverse { primal, .. } => primal,
            }),
            Self::F64(v) => DynScalar::F64(match v {
                AdValue::Primal(primal) => primal,
                AdValue::Forward { primal, .. } => primal,
                AdValue::Reverse { primal, .. } => primal,
            }),
            Self::C32(v) => DynScalar::C32(match v {
                AdValue::Primal(primal) => primal,
                AdValue::Forward { primal, .. } => primal,
                AdValue::Reverse { primal, .. } => primal,
            }),
            Self::C64(v) => DynScalar::C64(match v {
                AdValue::Primal(primal) => primal,
                AdValue::Forward { primal, .. } => primal,
                AdValue::Reverse { primal, .. } => primal,
            }),
        }
    }

    /// Returns tangent part as dynamic scalar when available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdValue, DynAdScalar, DynScalar};
    ///
    /// let x: DynAdScalar = AdValue::forward(3.0_f64, 0.5_f64).into();
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

    /// Returns the primal value while intentionally dropping AD metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdValue, DynAdScalar, DynScalar};
    ///
    /// let x: DynAdScalar = AdValue::reverse(3.0_f64, tenferro_dyadtensor::NodeId(1), tenferro_dyadtensor::TapeId(2), None).into();
    /// assert_eq!(x.detach(), DynScalar::F64(3.0));
    /// ```
    pub fn detach(&self) -> DynScalar {
        self.primal()
    }

    /// Returns typed AD value ref when dtype is `f32`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdValue, DynAdScalar};
    ///
    /// let x: DynAdScalar = AdValue::primal(1.0_f32).into();
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
    /// use tenferro_dyadtensor::{AdValue, DynAdScalar};
    ///
    /// let x: DynAdScalar = AdValue::primal(1.0_f64).into();
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
    /// use tenferro_dyadtensor::{AdValue, DynAdScalar};
    /// use num_complex::Complex32;
    ///
    /// let x: DynAdScalar = AdValue::primal(Complex32::new(1.0, 0.0)).into();
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
    /// use tenferro_dyadtensor::{AdValue, DynAdScalar};
    /// use num_complex::Complex64;
    ///
    /// let x: DynAdScalar = AdValue::primal(Complex64::new(1.0, 0.0)).into();
    /// assert!(x.as_c64().is_some());
    /// ```
    pub fn as_c64(&self) -> Option<&AdValue<Complex64>> {
        if let Self::C64(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns true when scalar dtype is complex.
    pub fn is_complex(&self) -> bool {
        matches!(self, Self::C32(_) | Self::C64(_))
    }

    /// Returns true when scalar dtype is real.
    pub fn is_real(&self) -> bool {
        matches!(self, Self::F32(_) | Self::F64(_))
    }

    /// Extracts the real part as `f64`, intentionally dropping AD metadata.
    pub fn real(&self) -> f64 {
        match self {
            Self::F32(v) => *v.primal_ref() as f64,
            Self::F64(v) => *v.primal_ref(),
            Self::C32(v) => v.primal_ref().re as f64,
            Self::C64(v) => v.primal_ref().re,
        }
    }

    /// Extracts the imaginary part as `f64`, intentionally dropping AD metadata.
    pub fn imag(&self) -> f64 {
        match self {
            Self::F32(_) | Self::F64(_) => 0.0,
            Self::C32(v) => v.primal_ref().im as f64,
            Self::C64(v) => v.primal_ref().im,
        }
    }

    /// Extracts the magnitude as `f64`, intentionally dropping AD metadata.
    pub fn abs(&self) -> f64 {
        match self {
            Self::F32(v) => v.primal_ref().abs() as f64,
            Self::F64(v) => v.primal_ref().abs(),
            Self::C32(v) => v.primal_ref().norm() as f64,
            Self::C64(v) => v.primal_ref().norm(),
        }
    }

    /// Returns true when the primal scalar value is zero.
    pub fn is_zero(&self) -> bool {
        match self {
            Self::F32(v) => v.primal_ref().is_zero(),
            Self::F64(v) => v.primal_ref().is_zero(),
            Self::C32(v) => v.primal_ref().is_zero(),
            Self::C64(v) => v.primal_ref().is_zero(),
        }
    }

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
    ///
    /// For negative real primals, promotes to complex before applying sqrt.
    pub fn sqrt(&self) -> Self {
        match self.clone() {
            Self::F32(v) => {
                if *v.primal_ref() >= 0.0 {
                    Self::F32(AdScalar::from(v).sqrt().into_value())
                } else {
                    let promoted = promote_f32_to_c32(v, "sqrt");
                    Self::C32(AdScalar::from(promoted).sqrt().into_value())
                }
            }
            Self::F64(v) => {
                if *v.primal_ref() >= 0.0 {
                    Self::F64(AdScalar::from(v).sqrt().into_value())
                } else {
                    let promoted = promote_f64_to_c64(v, "sqrt");
                    Self::C64(AdScalar::from(promoted).sqrt().into_value())
                }
            }
            Self::C32(v) => Self::C32(AdScalar::from(v).sqrt().into_value()),
            Self::C64(v) => Self::C64(AdScalar::from(v).sqrt().into_value()),
        }
    }

    /// Power with real exponent (`f64`) and AD propagation.
    ///
    /// For negative real primals, promotes to complex before applying powf.
    pub fn powf(&self, exponent: f64) -> Self {
        match self.clone() {
            Self::F32(v) => {
                if *v.primal_ref() >= 0.0 {
                    Self::F32(AdScalar::from(v).powf(exponent as f32).into_value())
                } else {
                    let promoted = promote_f32_to_c32(v, "powf");
                    Self::C32(AdScalar::from(promoted).powf(exponent as f32).into_value())
                }
            }
            Self::F64(v) => {
                if *v.primal_ref() >= 0.0 {
                    Self::F64(AdScalar::from(v).powf(exponent).into_value())
                } else {
                    let promoted = promote_f64_to_c64(v, "powf");
                    Self::C64(AdScalar::from(promoted).powf(exponent).into_value())
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
    ///
    /// - Real dtype: returns itself.
    /// - Complex dtype: returns real dtype (`C32->F32`, `C64->F64`).
    pub fn real_part(&self) -> Self {
        match self.clone() {
            Self::F32(v) => Self::F32(v),
            Self::F64(v) => Self::F64(v),
            Self::C32(v) => Self::F32(map_ad_value_mixed_linear(
                v,
                "real_part",
                |z| z.re,
                |cotangent| Complex32::new(cotangent, 0.0),
            )),
            Self::C64(v) => Self::F64(map_ad_value_mixed_linear(
                v,
                "real_part",
                |z| z.re,
                |cotangent| Complex64::new(cotangent, 0.0),
            )),
        }
    }

    /// AD-preserving extraction of the imaginary component.
    ///
    /// - Real dtype: returns zero with matching real dtype.
    /// - Complex dtype: returns real dtype (`C32->F32`, `C64->F64`).
    pub fn imag_part(&self) -> Self {
        match self.clone() {
            Self::F32(v) => Self::F32(map_ad_value_same_type_linear(v, "imag_part", |_| 0.0_f32)),
            Self::F64(v) => Self::F64(map_ad_value_same_type_linear(v, "imag_part", |_| 0.0_f64)),
            Self::C32(v) => Self::F32(map_ad_value_mixed_linear(
                v,
                "imag_part",
                |z| z.im,
                |cotangent| Complex32::new(0.0, cotangent),
            )),
            Self::C64(v) => Self::F64(map_ad_value_mixed_linear(
                v,
                "imag_part",
                |z| z.im,
                |cotangent| Complex64::new(0.0, cotangent),
            )),
        }
    }

    /// Compose a complex AD scalar from real/imaginary AD scalars.
    ///
    /// Both inputs must be real and have matching precision (`F32/F32` or `F64/F64`).
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
    ///
    /// Supported mixed-type pairs in v1 are:
    /// - `F32` + `C32` -> `C32`
    /// - `F64` + `C64` -> `C64`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex64;
    /// use tenferro_dyadtensor::{AdValue, DynAdScalar};
    ///
    /// let x: DynAdScalar = AdValue::primal(2.0_f64).into();
    /// let y: DynAdScalar = AdValue::primal(Complex64::new(1.0, -3.0)).into();
    /// let z = x.try_add(y).unwrap();
    /// assert_eq!(z.scalar_type(), tenferro_dyadtensor::ScalarType::C64);
    /// ```
    pub fn try_add(self, rhs: Self) -> Result<Self> {
        try_binary_dyn(self, rhs, BinaryOp::Add)
    }

    /// Checked subtraction with runtime dtype validation and promotion.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdValue, DynAdScalar};
    ///
    /// let x: DynAdScalar = AdValue::primal(5.0_f64).into();
    /// let y: DynAdScalar = AdValue::primal(2.0_f64).into();
    /// let z = x.try_sub(y).unwrap();
    /// assert_eq!(z.primal(), tenferro_dyadtensor::DynScalar::F64(3.0));
    /// ```
    pub fn try_sub(self, rhs: Self) -> Result<Self> {
        try_binary_dyn(self, rhs, BinaryOp::Sub)
    }

    /// Checked multiplication with runtime dtype validation and promotion.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex64;
    /// use tenferro_dyadtensor::{AdValue, DynAdScalar};
    ///
    /// let x: DynAdScalar = AdValue::primal(2.0_f64).into();
    /// let y: DynAdScalar = AdValue::primal(Complex64::new(1.0, 2.0)).into();
    /// let z = x.try_mul(y).unwrap();
    /// assert_eq!(z.scalar_type(), tenferro_dyadtensor::ScalarType::C64);
    /// ```
    pub fn try_mul(self, rhs: Self) -> Result<Self> {
        try_binary_dyn(self, rhs, BinaryOp::Mul)
    }

    /// Checked division with runtime dtype validation and promotion.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdValue, DynAdScalar};
    ///
    /// let x: DynAdScalar = AdValue::primal(8.0_f64).into();
    /// let y: DynAdScalar = AdValue::primal(2.0_f64).into();
    /// let z = x.try_div(y).unwrap();
    /// assert_eq!(z.primal(), tenferro_dyadtensor::DynScalar::F64(4.0));
    /// ```
    pub fn try_div(self, rhs: Self) -> Result<Self> {
        try_binary_dyn(self, rhs, BinaryOp::Div)
    }
}

impl From<AdValue<f32>> for DynAdScalar {
    fn from(value: AdValue<f32>) -> Self {
        Self::F32(value)
    }
}

impl From<AdValue<f64>> for DynAdScalar {
    fn from(value: AdValue<f64>) -> Self {
        Self::F64(value)
    }
}

impl From<AdValue<Complex32>> for DynAdScalar {
    fn from(value: AdValue<Complex32>) -> Self {
        Self::C32(value)
    }
}

impl From<AdValue<Complex64>> for DynAdScalar {
    fn from(value: AdValue<Complex64>) -> Self {
        Self::C64(value)
    }
}

impl From<f32> for DynAdScalar {
    fn from(value: f32) -> Self {
        Self::F32(AdValue::primal(value))
    }
}

impl From<f64> for DynAdScalar {
    fn from(value: f64) -> Self {
        Self::F64(AdValue::primal(value))
    }
}

impl From<Complex32> for DynAdScalar {
    fn from(value: Complex32) -> Self {
        Self::C32(AdValue::primal(value))
    }
}

impl From<Complex64> for DynAdScalar {
    fn from(value: Complex64) -> Self {
        Self::C64(AdValue::primal(value))
    }
}

impl Add for DynAdScalar {
    type Output = DynAdScalar;

    fn add(self, rhs: Self) -> Self::Output {
        self.try_add(rhs)
            .unwrap_or_else(|e| panic!("DynAdScalar add failed: {e}"))
    }
}

impl Sub for DynAdScalar {
    type Output = DynAdScalar;

    fn sub(self, rhs: Self) -> Self::Output {
        self.try_sub(rhs)
            .unwrap_or_else(|e| panic!("DynAdScalar sub failed: {e}"))
    }
}

impl Mul for DynAdScalar {
    type Output = DynAdScalar;

    fn mul(self, rhs: Self) -> Self::Output {
        self.try_mul(rhs)
            .unwrap_or_else(|e| panic!("DynAdScalar mul failed: {e}"))
    }
}

impl Div for DynAdScalar {
    type Output = DynAdScalar;

    fn div(self, rhs: Self) -> Self::Output {
        self.try_div(rhs)
            .unwrap_or_else(|e| panic!("DynAdScalar div failed: {e}"))
    }
}

impl Neg for DynAdScalar {
    type Output = DynAdScalar;

    fn neg(self) -> Self::Output {
        match self {
            DynAdScalar::F32(v) => DynAdScalar::F32((-AdScalar::from(v)).into_value()),
            DynAdScalar::F64(v) => DynAdScalar::F64((-AdScalar::from(v)).into_value()),
            DynAdScalar::C32(v) => DynAdScalar::C32((-AdScalar::from(v)).into_value()),
            DynAdScalar::C64(v) => DynAdScalar::C64((-AdScalar::from(v)).into_value()),
        }
    }
}

macro_rules! impl_dynadvalue_scalar_binop {
    ($trait:ident, $method:ident, $scalar:ty) => {
        impl $trait<$scalar> for DynAdScalar {
            type Output = DynAdScalar;

            fn $method(self, rhs: $scalar) -> Self::Output {
                $trait::$method(self, DynAdScalar::from(rhs))
            }
        }

        impl $trait<DynAdScalar> for $scalar {
            type Output = DynAdScalar;

            fn $method(self, rhs: DynAdScalar) -> Self::Output {
                $trait::$method(DynAdScalar::from(self), rhs)
            }
        }
    };
}

impl_dynadvalue_scalar_binop!(Add, add, f32);
impl_dynadvalue_scalar_binop!(Add, add, f64);
impl_dynadvalue_scalar_binop!(Add, add, Complex32);
impl_dynadvalue_scalar_binop!(Add, add, Complex64);
impl_dynadvalue_scalar_binop!(Sub, sub, f32);
impl_dynadvalue_scalar_binop!(Sub, sub, f64);
impl_dynadvalue_scalar_binop!(Sub, sub, Complex32);
impl_dynadvalue_scalar_binop!(Sub, sub, Complex64);
impl_dynadvalue_scalar_binop!(Mul, mul, f32);
impl_dynadvalue_scalar_binop!(Mul, mul, f64);
impl_dynadvalue_scalar_binop!(Mul, mul, Complex32);
impl_dynadvalue_scalar_binop!(Mul, mul, Complex64);
impl_dynadvalue_scalar_binop!(Div, div, f32);
impl_dynadvalue_scalar_binop!(Div, div, f64);
impl_dynadvalue_scalar_binop!(Div, div, Complex32);
impl_dynadvalue_scalar_binop!(Div, div, Complex64);

impl TryFrom<DynAdScalar> for f64 {
    type Error = &'static str;

    fn try_from(value: DynAdScalar) -> core::result::Result<Self, Self::Error> {
        match value {
            DynAdScalar::F32(v) => Ok(*v.primal_ref() as f64),
            DynAdScalar::F64(v) => Ok(*v.primal_ref()),
            DynAdScalar::C32(_) | DynAdScalar::C64(_) => {
                Err("Cannot convert complex DynAdScalar to f64")
            }
        }
    }
}

impl From<DynAdScalar> for Complex64 {
    fn from(value: DynAdScalar) -> Self {
        match value {
            DynAdScalar::F32(v) => Complex64::new(*v.primal_ref() as f64, 0.0),
            DynAdScalar::F64(v) => Complex64::new(*v.primal_ref(), 0.0),
            DynAdScalar::C32(v) => {
                let z = v.primal_ref();
                Complex64::new(z.re as f64, z.im as f64)
            }
            DynAdScalar::C64(v) => *v.primal_ref(),
        }
    }
}

impl Default for DynAdScalar {
    fn default() -> Self {
        Self::new_real(0.0)
    }
}

impl Zero for DynAdScalar {
    fn zero() -> Self {
        Self::new_real(0.0)
    }

    fn is_zero(&self) -> bool {
        DynAdScalar::is_zero(self)
    }
}

impl One for DynAdScalar {
    fn one() -> Self {
        Self::new_real(1.0)
    }
}

impl PartialOrd for DynAdScalar {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        match (self, other) {
            (DynAdScalar::F32(a), DynAdScalar::F32(b)) => {
                a.primal_ref().partial_cmp(b.primal_ref())
            }
            (DynAdScalar::F64(a), DynAdScalar::F64(b)) => {
                a.primal_ref().partial_cmp(b.primal_ref())
            }
            (DynAdScalar::F32(a), DynAdScalar::F64(b)) => {
                (*a.primal_ref() as f64).partial_cmp(b.primal_ref())
            }
            (DynAdScalar::F64(a), DynAdScalar::F32(b)) => {
                a.primal_ref().partial_cmp(&(*b.primal_ref() as f64))
            }
            _ => None,
        }
    }
}

impl fmt::Display for DynAdScalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DynAdScalar::F32(v) => write!(f, "{}", v.primal_ref()),
            DynAdScalar::F64(v) => write!(f, "{}", v.primal_ref()),
            DynAdScalar::C32(v) => write!(f, "{}", v.primal_ref()),
            DynAdScalar::C64(v) => write!(f, "{}", v.primal_ref()),
        }
    }
}

impl Mul<&DynAdTensor> for &DynAdScalar {
    type Output = Result<DynAdTensor>;

    fn mul(self, rhs: &DynAdTensor) -> Self::Output {
        rhs.scale(self)
    }
}

impl Div<&DynAdScalar> for &DynAdTensor {
    type Output = Result<DynAdTensor>;

    fn div(self, rhs: &DynAdScalar) -> Self::Output {
        self.div_scalar(rhs)
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

fn map_ad_tensor_same_type_linear_typed<T, F>(
    input: &AdTensor<T>,
    op_name: &'static str,
    map: F,
) -> Result<AdTensor<T>>
where
    T: Scalar + ScalarAd + Copy + 'static,
    F: Fn(T) -> T + Copy + 'static,
{
    let mapped = match input.as_value().clone() {
        AdValue::Primal(primal) => {
            AdValue::Primal(primal.with_payload_like(tensor_map_unary_typed(primal.payload(), map)?)?)
        }
        AdValue::Forward { primal, tangent } => AdValue::Forward {
            primal: primal.with_payload_like(tensor_map_unary_typed(primal.payload(), map)?)?,
            tangent: tangent.with_payload_like(tensor_map_unary_typed(tangent.payload(), map)?)?,
        },
        AdValue::Reverse {
            primal,
            node: input_node,
            tape,
            tangent,
        } => {
            let output_primal =
                primal.with_payload_like(tensor_map_unary_typed(primal.payload(), map)?)?;
            let output_tangent = tangent
                .as_ref()
                .map(|t| t.with_payload_like(tensor_map_unary_typed(t.payload(), map)?))
                .transpose()?;
            let output_node = NodeId(NEXT_AD_TENSOR_NODE_ID.fetch_add(1, Ordering::Relaxed));
            reverse_tape::register_rule::<T>(
                tape,
                output_node,
                Box::new(move |cotangent| {
                    Ok(vec![(input_node, tensor_map_unary_typed(cotangent, map)?)])
                }),
            )
            .unwrap_or_else(|e| panic!("{op_name}: {e}"));
            AdValue::Reverse {
                primal: output_primal,
                node: output_node,
                tape,
                tangent: output_tangent,
            }
        }
    };
    Ok(AdTensor(mapped))
}

fn map_ad_tensor_mixed_linear_typed<TIn, TOut, P, R>(
    input: &AdTensor<TIn>,
    op_name: &'static str,
    primal_map: P,
    reverse_map: R,
) -> Result<AdTensor<TOut>>
where
    TIn: Scalar + ScalarAd + Copy + 'static,
    TOut: Scalar + ScalarAd + Copy + 'static,
    P: Fn(TIn) -> TOut + Copy,
    R: Fn(TOut) -> TIn + Copy + 'static,
{
    let mapped = match input.as_value().clone() {
        AdValue::Primal(primal) => AdValue::Primal(
            StructuredTensor::new(
                primal.logical_dims().to_vec(),
                primal.axis_classes().to_vec(),
                tensor_map_unary_typed(primal.payload(), primal_map)?,
            )?,
        ),
        AdValue::Forward { primal, tangent } => AdValue::Forward {
            primal: StructuredTensor::new(
                primal.logical_dims().to_vec(),
                primal.axis_classes().to_vec(),
                tensor_map_unary_typed(primal.payload(), primal_map)?,
            )?,
            tangent: StructuredTensor::new(
                tangent.logical_dims().to_vec(),
                tangent.axis_classes().to_vec(),
                tensor_map_unary_typed(tangent.payload(), primal_map)?,
            )?,
        },
        AdValue::Reverse {
            primal,
            node: input_node,
            tape,
            tangent,
        } => {
            let output_primal = StructuredTensor::new(
                primal.logical_dims().to_vec(),
                primal.axis_classes().to_vec(),
                tensor_map_unary_typed(primal.payload(), primal_map)?,
            )?;
            let output_tangent = tangent
                .as_ref()
                .map(|t| {
                    StructuredTensor::new(
                        t.logical_dims().to_vec(),
                        t.axis_classes().to_vec(),
                        tensor_map_unary_typed(t.payload(), primal_map)?,
                    )
                })
                .transpose()?;
            let output_node = NodeId(NEXT_AD_TENSOR_NODE_ID.fetch_add(1, Ordering::Relaxed));
            reverse_tape::register_bridge_rule::<TOut, TIn>(
                tape,
                output_node,
                Box::new(move |cotangent| {
                    Ok(vec![(
                        input_node,
                        tensor_map_unary_typed(cotangent, reverse_map)?,
                    )])
                }),
            )
            .unwrap_or_else(|e| panic!("{op_name}: {e}"));
            AdValue::Reverse {
                primal: output_primal,
                node: output_node,
                tape,
                tangent: output_tangent,
            }
        }
    };
    Ok(AdTensor(mapped))
}

fn tensor_add_typed<T>(lhs: &Tensor<T>, rhs: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Scalar + Copy + Add<Output = T>,
{
    tensor_map_binary_typed(lhs, rhs, |x, y| x + y)
}

struct AdTensorBinaryState<T: Scalar> {
    primal: StructuredTensor<T>,
    tangent: Option<StructuredTensor<T>>,
    reverse: Option<(NodeId, TapeId)>,
}

fn split_ad_tensor_state<T: Scalar>(
    value: AdValue<StructuredTensor<T>>,
) -> AdTensorBinaryState<T> {
    match value {
        AdValue::Primal(primal) => AdTensorBinaryState {
            primal,
            tangent: None,
            reverse: None,
        },
        AdValue::Forward { primal, tangent } => AdTensorBinaryState {
            primal,
            tangent: Some(tangent),
            reverse: None,
        },
        AdValue::Reverse {
            primal,
            node,
            tape,
            tangent,
        } => AdTensorBinaryState {
            primal,
            tangent,
            reverse: Some((node, tape)),
        },
    }
}

fn merge_add_ad_tensors<T>(
    lhs: AdValue<StructuredTensor<T>>,
    rhs: AdValue<StructuredTensor<T>>,
) -> Result<AdValue<StructuredTensor<T>>>
where
    T: Scalar + Copy + Add<Output = T> + 'static,
{
    let lhs_state = split_ad_tensor_state(lhs);
    let rhs_state = split_ad_tensor_state(rhs);

    let primal = StructuredTensor::new(
        lhs_state.primal.logical_dims().to_vec(),
        lhs_state.primal.axis_classes().to_vec(),
        tensor_add_typed(lhs_state.primal.payload(), rhs_state.primal.payload())?,
    )?;
    let tangent = match (lhs_state.tangent, rhs_state.tangent) {
        (Some(a), Some(b)) => Some(StructuredTensor::new(
            a.logical_dims().to_vec(),
            a.axis_classes().to_vec(),
            tensor_add_typed(a.payload(), b.payload())?,
        )?),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };

    match (lhs_state.reverse, rhs_state.reverse) {
        (None, None) => match tangent {
            Some(tangent) => Ok(AdValue::Forward { primal, tangent }),
            None => Ok(AdValue::Primal(primal)),
        },
        (Some((lhs_node, lhs_tape)), rhs_reverse) => {
            if let Some((_, rhs_tape)) = rhs_reverse {
                if lhs_tape != rhs_tape {
                    return Err(Error::InvalidAdTensor {
                        message: format!(
                            "reverse-mode tape mismatch in tensor add (lhs={}, rhs={})",
                            lhs_tape.0, rhs_tape.0
                        ),
                    });
                }
            }
            let rhs_node = rhs_reverse.map(|(node, _)| node);
            let output_node = NodeId(NEXT_AD_TENSOR_NODE_ID.fetch_add(1, Ordering::Relaxed));
            reverse_tape::register_rule::<T>(
                lhs_tape,
                output_node,
                Box::new(move |cotangent: &Tensor<T>| {
                    let mut input_grads = Vec::new();
                    input_grads.push((lhs_node, cotangent.clone()));
                    if let Some(node) = rhs_node {
                        input_grads.push((node, cotangent.clone()));
                    }
                    Ok(input_grads)
                }),
            )?;
            Ok(AdValue::Reverse {
                primal,
                node: output_node,
                tape: lhs_tape,
                tangent,
            })
        }
        (None, Some((rhs_node, rhs_tape))) => {
            let output_node = NodeId(NEXT_AD_TENSOR_NODE_ID.fetch_add(1, Ordering::Relaxed));
            reverse_tape::register_rule::<T>(
                rhs_tape,
                output_node,
                Box::new(move |cotangent: &Tensor<T>| Ok(vec![(rhs_node, cotangent.clone())])),
            )?;
            Ok(AdValue::Reverse {
                primal,
                node: output_node,
                tape: rhs_tape,
                tangent,
            })
        }
    }
}

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
        Tensor::from_slice(&tensor_grad, &dims, MemoryOrder::ColumnMajor).map_err(Error::from)?,
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
        (Some((lhs_node, lhs_tape)), Some((_, rhs_tape))) if lhs_tape != rhs_tape => {
            return Err(Error::MixedReverseTape {
                expected: lhs_tape.0,
                found: rhs_tape.0,
            })
        }
        (Some((node, tape)), Some(_)) => Some((node, tape)),
        (Some((node, tape)), None) => Some((node, tape)),
        (None, Some((node, tape))) => Some((node, tape)),
        (None, None) => None,
    };

    if let Some((_, tape)) = reverse {
        let output_node = NodeId(NEXT_AD_TENSOR_NODE_ID.fetch_add(1, Ordering::Relaxed));
        let tensor_node = tensor_reverse.map(|(node, _)| node);
        let scalar_node = scalar_reverse.map(|(node, _)| node);
        let tensor_primal = tensor.primal().clone();
        let tensor_primal_for_scalar = tensor_primal.clone();
        let scalar_primal = *scalar.primal_ref();

        reverse_tape::register_rule::<T>(
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
        )?;

        if let Some(node) = scalar_node {
            reverse_tape::register_scalar_bridge_rule::<T, T>(
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
            )?;
        }

        return Ok(AdTensor::new_reverse(primal, output_node, tape, tangent));
    }
    if let Some(tangent) = tangent {
        return Ok(AdTensor::new_forward(primal, tangent));
    }
    Ok(AdTensor::new_primal(primal))
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

    /// Returns axis classes of the structured primal.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdTensor, DynAdTensor, StructuredTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let payload =
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynAdTensor =
    ///     AdTensor::new_primal(StructuredTensor::from_diagonal_vector(payload, 2).unwrap()).into();
    /// assert_eq!(x.axis_classes(), &[0, 0]);
    /// ```
    pub fn axis_classes(&self) -> &[usize] {
        match self {
            Self::F32(v) => v.axis_classes(),
            Self::F64(v) => v.axis_classes(),
            Self::C32(v) => v.axis_classes(),
            Self::C64(v) => v.axis_classes(),
        }
    }

    /// Returns `true` when the structured primal is dense.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdTensor, DynAdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynAdTensor = AdTensor::new_primal(t).into();
    /// assert!(x.is_dense());
    /// ```
    pub fn is_dense(&self) -> bool {
        match self {
            Self::F32(v) => v.is_dense(),
            Self::F64(v) => v.is_dense(),
            Self::C32(v) => v.is_dense(),
            Self::C64(v) => v.is_dense(),
        }
    }

    /// Returns `true` when the structured primal is diagonal.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdTensor, DynAdTensor, StructuredTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let payload =
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x: DynAdTensor =
    ///     AdTensor::new_primal(StructuredTensor::from_diagonal_vector(payload, 2).unwrap()).into();
    /// assert!(x.is_diag());
    /// ```
    pub fn is_diag(&self) -> bool {
        match self {
            Self::F32(v) => v.is_diag(),
            Self::F64(v) => v.is_diag(),
            Self::C32(v) => v.is_diag(),
            Self::C64(v) => v.is_diag(),
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

    /// Returns true when scalar dtype is complex.
    pub fn is_complex(&self) -> bool {
        matches!(self, Self::C32(_) | Self::C64(_))
    }

    /// Returns true when scalar dtype is real.
    pub fn is_real(&self) -> bool {
        matches!(self, Self::F32(_) | Self::F64(_))
    }

    /// AD-preserving extraction of the real component.
    ///
    /// - Real dtype: returns itself.
    /// - Complex dtype: returns real dtype (`C32->F32`, `C64->F64`).
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
    ///
    /// - Real dtype: returns zero with matching real dtype.
    /// - Complex dtype: returns real dtype (`C32->F32`, `C64->F64`).
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
    ///
    /// Both inputs must be real and have matching precision (`F32/F32` or `F64/F64`).
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
                Ok(Self::C32(AdTensor(merged)))
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
                Ok(Self::C64(AdTensor(merged)))
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

    /// Scalar multiply with AD preservation for scalar and tensor inputs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdTensor, DynAdScalar, DynAdTensor, ScalarType};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let x: DynAdTensor = AdTensor::new_primal(
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// )
    /// .into();
    /// let a = DynAdScalar::from(2.0_f64);
    /// let y = x.scale(&a).unwrap();
    /// assert_eq!(y.scalar_type(), ScalarType::F64);
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdTensor, DynAdScalar, DynAdTensor, ScalarType};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let x: DynAdTensor = AdTensor::new_primal(
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// )
    /// .into();
    /// let y = x.clone();
    /// let a = DynAdScalar::from(2.0_f64);
    /// let b = DynAdScalar::from(-1.0_f64);
    /// let z = x.axpby(&a, &y, &b).unwrap();
    /// assert_eq!(z.scalar_type(), ScalarType::F64);
    /// ```
    pub fn axpby(&self, a: &DynAdScalar, other: &Self, b: &DynAdScalar) -> Result<Self> {
        match (self.scale(a)?, other.scale(b)?) {
            (Self::F32(lhs), Self::F32(rhs)) => Ok(Self::F32(AdTensor(merge_add_ad_tensors(
                lhs.into_value(),
                rhs.into_value(),
            )?))),
            (Self::F64(lhs), Self::F64(rhs)) => Ok(Self::F64(AdTensor(merge_add_ad_tensors(
                lhs.into_value(),
                rhs.into_value(),
            )?))),
            (Self::C32(lhs), Self::C32(rhs)) => Ok(Self::C32(AdTensor(merge_add_ad_tensors(
                lhs.into_value(),
                rhs.into_value(),
            )?))),
            (Self::C64(lhs), Self::C64(rhs)) => Ok(Self::C64(AdTensor(merge_add_ad_tensors(
                lhs.into_value(),
                rhs.into_value(),
            )?))),
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
    /// use tenferro_dyadtensor::{AdTensor, DynAdScalar, DynAdTensor, ScalarType};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let x: DynAdTensor = AdTensor::new_primal(
    ///     Tensor::<f64>::from_slice(&[2.0, 4.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// )
    /// .into();
    /// let a = DynAdScalar::from(2.0_f64);
    /// let y = x.div_scalar(&a).unwrap();
    /// assert_eq!(y.scalar_type(), ScalarType::F64);
    /// ```
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
    ///
    /// AD metadata is preserved in the operands and not modified; this utility
    /// only inspects primal tensors for comparison.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdTensor, DynAdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let a = AdTensor::new_primal(
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    /// let b = AdTensor::new_primal(
    ///     Tensor::<f64>::from_slice(&[1.0, 1.5], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    /// let da: DynAdTensor = a.into();
    /// let db: DynAdTensor = b.into();
    /// let diff = da.max_abs_diff_primal(&db).unwrap();
    /// assert!((diff - 0.5).abs() < 1e-12);
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
    use crate::{Error, TapeId};
    use tenferro_tensor::MemoryOrder;

    #[test]
    fn dyn_scalar_metadata() {
        let x: DynScalar = 1.0_f64.into();
        assert_eq!(x.scalar_type(), ScalarType::F64);
        assert_eq!(x.as_f64(), Some(1.0));
    }

    #[test]
    fn dyn_ad_value_mode_and_tangent() {
        let x: DynAdScalar = AdValue::forward(2.0_f32, 0.5_f32).into();
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

    #[test]
    fn dyn_ad_value_mul_mixed_real_complex_promotes_to_complex() {
        let lhs = DynAdScalar::from(2.0_f64);
        let rhs = DynAdScalar::from(Complex64::new(1.0, -3.0));
        let out = lhs * rhs;
        assert_eq!(out.scalar_type(), ScalarType::C64);
        assert_eq!(out.primal(), DynScalar::C64(Complex64::new(2.0, -6.0)));
    }

    #[test]
    fn dyn_ad_value_div_with_scalar_lhs_is_supported() {
        let rhs = DynAdScalar::from(2.0_f64);
        let out = Complex64::new(4.0, -2.0) / rhs;
        assert_eq!(out.scalar_type(), ScalarType::C64);
        assert_eq!(out.primal(), DynScalar::C64(Complex64::new(2.0, -1.0)));
    }

    #[test]
    fn dyn_ad_value_try_add_rejects_cross_precision_pairs() {
        let lhs = DynAdScalar::from(1.0_f32);
        let rhs = DynAdScalar::from(2.0_f64);
        let err = lhs.try_add(rhs).unwrap_err();
        assert!(matches!(err, Error::InvalidAdScalar { .. }));
    }

    #[test]
    fn dyn_ad_value_try_mul_checks_reverse_tape_compatibility() {
        let lhs: DynAdScalar = AdValue::reverse(2.0_f64, crate::NodeId(1), TapeId(7), None).into();
        let rhs: DynAdScalar = AdValue::reverse(3.0_f64, crate::NodeId(2), TapeId(8), None).into();
        let err = lhs.try_mul(rhs).unwrap_err();
        assert!(
            matches!(err, Error::InvalidAdScalar { message } if message.contains("reverse-mode tape mismatch"))
        );
    }

    #[test]
    fn dyn_tensor_max_abs_diff_is_zero_for_same_logical_tensor_with_different_memory_order() {
        let base = Tensor::<f64>::from_slice(
            &(0..12).map(|x| x as f64).collect::<Vec<_>>(),
            &[2, 3, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let row_major = base.contiguous(MemoryOrder::RowMajor);

        let lhs: DynTensor = base.into();
        let rhs: DynTensor = row_major.into();
        let diff = lhs.max_abs_diff(&rhs).unwrap();
        assert!(diff < 1e-12, "expected zero diff, got {diff}");
    }

    #[test]
    fn dyn_tensor_sub_abs_max_pipeline_matches_expected() {
        let lhs_t = Tensor::<f64>::from_slice(
            &(0..12).map(|x| x as f64).collect::<Vec<_>>(),
            &[2, 3, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let rhs_t = Tensor::<f64>::from_slice(
            &(0..12)
                .map(|x| if x == 7 { (x as f64) + 4.0 } else { x as f64 })
                .collect::<Vec<_>>(),
            &[2, 3, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap()
        .contiguous(MemoryOrder::RowMajor);

        let lhs: DynTensor = lhs_t.into();
        let rhs: DynTensor = rhs_t.into();

        let diff = lhs
            .try_sub(&rhs)
            .unwrap()
            .abs_tensor()
            .unwrap()
            .max_as_f64()
            .unwrap();
        assert!((diff - 4.0).abs() < 1e-12, "expected diff=4, got {diff}");
    }

    #[test]
    fn dyn_tensor_abs_tensor_on_complex_returns_real_dtype() {
        let t = Tensor::<Complex64>::from_slice(
            &[Complex64::new(3.0, 4.0), Complex64::new(0.0, -2.0)],
            &[2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let x: DynTensor = t.into();
        let y = x.abs_tensor().unwrap();
        assert_eq!(y.scalar_type(), ScalarType::F64);
        let yr = y.as_f64().unwrap();
        let data = yr.buffer().as_slice().unwrap();
        assert!((data[0] - 5.0).abs() < 1e-12);
        assert!((data[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn dyn_tensor_max_on_complex_requires_abs_first() {
        let t = Tensor::<Complex64>::from_slice(
            &[Complex64::new(1.0, 2.0)],
            &[1],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let x: DynTensor = t.into();
        let err = x.max().unwrap_err();
        assert!(
            matches!(err, Error::InvalidAdTensor { message } if message.contains("abs_tensor"))
        );
    }

    #[test]
    fn dyn_tensor_max_abs_diff_detects_value_difference() {
        let lhs_t = Tensor::<f64>::from_slice(
            &(0..12).map(|x| x as f64).collect::<Vec<_>>(),
            &[2, 3, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let rhs_t = Tensor::<f64>::from_slice(
            &(0..12)
                .map(|x| if x == 7 { (x as f64) + 4.0 } else { x as f64 })
                .collect::<Vec<_>>(),
            &[2, 3, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let lhs: DynTensor = lhs_t.into();
        let rhs: DynTensor = rhs_t.into();

        let diff = lhs.max_abs_diff(&rhs).unwrap();
        assert!((diff - 4.0).abs() < 1e-12, "expected diff=4, got {diff}");
    }

    #[test]
    fn dyn_tensor_max_abs_diff_rejects_dtype_mismatch() {
        let lhs = Tensor::<f32>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        let rhs = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        let lhs: DynTensor = lhs.into();
        let rhs: DynTensor = rhs.into();

        let err = lhs.max_abs_diff(&rhs).unwrap_err();
        assert!(matches!(err, Error::InvalidAdTensor { .. }));
    }

    #[test]
    fn dyn_ad_tensor_max_abs_diff_primal_uses_primal_values() {
        let lhs =
            Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();
        let rhs =
            Tensor::<f64>::from_slice(&[1.0, 1.5, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();
        let lhs: DynAdTensor = AdTensor::new_primal(lhs).into();
        let rhs: DynAdTensor = AdTensor::new_primal(rhs).into();

        let diff = lhs.max_abs_diff_primal(&rhs).unwrap();
        assert!((diff - 0.5).abs() < 1e-12, "expected diff=0.5, got {diff}");
    }

    #[test]
    fn dyn_ad_tensor_real_imag_part_preserve_forward_mode() {
        let primal = Tensor::<Complex64>::from_slice(
            &[Complex64::new(2.5, -1.25)],
            &[1],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let tangent = Tensor::<Complex64>::from_slice(
            &[Complex64::new(0.5, 0.75)],
            &[1],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let x: DynAdTensor = AdTensor::new_forward(primal, tangent).into();
        assert!(x.is_complex());
        assert!(!x.is_real());

        let xr = x.real_part().unwrap();
        let xi = x.imag_part().unwrap();
        assert!(xr.is_real());
        assert!(xi.is_real());
        assert_eq!(xr.scalar_type(), ScalarType::F64);
        assert_eq!(xi.scalar_type(), ScalarType::F64);
        assert_eq!(xr.mode(), AdMode::Forward);
        assert_eq!(xi.mode(), AdMode::Forward);

        let xr_t = xr.as_f64().unwrap();
        let xi_t = xi.as_f64().unwrap();
        let xr_primal = xr_t.primal().buffer().as_slice().unwrap()[0];
        let xr_tangent = xr_t.tangent().unwrap().buffer().as_slice().unwrap()[0];
        let xi_primal = xi_t.primal().buffer().as_slice().unwrap()[0];
        let xi_tangent = xi_t.tangent().unwrap().buffer().as_slice().unwrap()[0];

        assert!((xr_primal - 2.5).abs() < 1e-12);
        assert!((xr_tangent - 0.5).abs() < 1e-12);
        assert!((xi_primal - (-1.25)).abs() < 1e-12);
        assert!((xi_tangent - 0.75).abs() < 1e-12);
    }

    #[test]
    fn dyn_ad_tensor_compose_complex_roundtrip_forward() {
        let re = AdTensor::new_forward(
            Tensor::<f64>::from_slice(&[1.5], &[1], MemoryOrder::ColumnMajor).unwrap(),
            Tensor::<f64>::from_slice(&[0.25], &[1], MemoryOrder::ColumnMajor).unwrap(),
        );
        let im = AdTensor::new_forward(
            Tensor::<f64>::from_slice(&[-2.0], &[1], MemoryOrder::ColumnMajor).unwrap(),
            Tensor::<f64>::from_slice(&[0.75], &[1], MemoryOrder::ColumnMajor).unwrap(),
        );
        let z = DynAdTensor::compose_complex(DynAdTensor::F64(re), DynAdTensor::F64(im)).unwrap();
        assert_eq!(z.scalar_type(), ScalarType::C64);
        assert_eq!(z.mode(), AdMode::Forward);

        let zc = z.as_c64().unwrap();
        let primal = zc.primal().buffer().as_slice().unwrap()[0];
        let tangent = zc.tangent().unwrap().buffer().as_slice().unwrap()[0];
        assert!((primal - Complex64::new(1.5, -2.0)).norm() < 1e-12);
        assert!((tangent - Complex64::new(0.25, 0.75)).norm() < 1e-12);
    }

    #[test]
    fn dyn_ad_tensor_compose_complex_rejects_non_real_inputs() {
        let re = AdTensor::new_primal(
            Tensor::<Complex64>::from_slice(
                &[Complex64::new(1.0, 0.0)],
                &[1],
                MemoryOrder::ColumnMajor,
            )
            .unwrap(),
        );
        let im = AdTensor::new_primal(
            Tensor::<f64>::from_slice(&[2.0], &[1], MemoryOrder::ColumnMajor).unwrap(),
        );
        let err = match DynAdTensor::compose_complex(DynAdTensor::C64(re), DynAdTensor::F64(im)) {
            Ok(_) => panic!("compose_complex should reject non-real input"),
            Err(err) => err,
        };
        assert!(matches!(err, Error::InvalidAdTensor { .. }));
    }

    #[test]
    fn dyn_ad_tensor_compose_complex_checks_reverse_tape_compatibility() {
        let re = AdTensor::new_reverse(
            Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap(),
            crate::NodeId(1),
            TapeId(7),
            None,
        );
        let im = AdTensor::new_reverse(
            Tensor::<f64>::from_slice(&[2.0], &[1], MemoryOrder::ColumnMajor).unwrap(),
            crate::NodeId(2),
            TapeId(8),
            None,
        );
        let err = match DynAdTensor::compose_complex(DynAdTensor::F64(re), DynAdTensor::F64(im)) {
            Ok(_) => panic!("compose_complex should reject mixed reverse tapes"),
            Err(err) => err,
        };
        assert!(
            matches!(err, Error::InvalidAdTensor { message } if message.contains("reverse-mode tape mismatch"))
        );
    }
}
