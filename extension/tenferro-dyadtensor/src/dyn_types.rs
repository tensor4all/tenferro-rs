use chainrules_scalarops::ScalarAd;
use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};
use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use tenferro_algebra::Scalar;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{AdMode, AdScalar, AdTensor, AdValue, Error, NodeId, Result, TapeId};

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

fn promote_f32_to_c32(value: AdValue<f32>) -> AdValue<Complex32> {
    value.map(|x| Complex32::new(x, 0.0))
}

fn promote_f64_to_c64(value: AdValue<f64>) -> AdValue<Complex64> {
    value.map(|x| Complex64::new(x, 0.0))
}

fn apply_binary_ad<T: ScalarAd>(lhs: AdValue<T>, rhs: AdValue<T>, op: BinaryOp) -> AdValue<T> {
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

fn checked_apply_binary_ad<T: ScalarAd>(
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

fn try_binary_dyn(lhs: DynAdValue, rhs: DynAdValue, op: BinaryOp) -> Result<DynAdValue> {
    let lhs_ty = lhs.scalar_type();
    let rhs_ty = rhs.scalar_type();
    match (lhs, rhs) {
        (DynAdValue::F32(a), DynAdValue::F32(b)) => {
            Ok(DynAdValue::F32(checked_apply_binary_ad(a, b, op)?))
        }
        (DynAdValue::F64(a), DynAdValue::F64(b)) => {
            Ok(DynAdValue::F64(checked_apply_binary_ad(a, b, op)?))
        }
        (DynAdValue::C32(a), DynAdValue::C32(b)) => {
            Ok(DynAdValue::C32(checked_apply_binary_ad(a, b, op)?))
        }
        (DynAdValue::C64(a), DynAdValue::C64(b)) => {
            Ok(DynAdValue::C64(checked_apply_binary_ad(a, b, op)?))
        }
        (DynAdValue::F32(a), DynAdValue::C32(b)) => Ok(DynAdValue::C32(checked_apply_binary_ad(
            promote_f32_to_c32(a),
            b,
            op,
        )?)),
        (DynAdValue::C32(a), DynAdValue::F32(b)) => Ok(DynAdValue::C32(checked_apply_binary_ad(
            a,
            promote_f32_to_c32(b),
            op,
        )?)),
        (DynAdValue::F64(a), DynAdValue::C64(b)) => Ok(DynAdValue::C64(checked_apply_binary_ad(
            promote_f64_to_c64(a),
            b,
            op,
        )?)),
        (DynAdValue::C64(a), DynAdValue::F64(b)) => Ok(DynAdValue::C64(checked_apply_binary_ad(
            a,
            promote_f64_to_c64(b),
            op,
        )?)),
        _ => Err(unsupported_binary_pair(op, lhs_ty, rhs_ty)),
    }
}

impl DynAdValue {
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
                    let promoted = v.map(|x| Complex32::new(x, 0.0));
                    Self::C32(AdScalar::from(promoted).sqrt().into_value())
                }
            }
            Self::F64(v) => {
                if *v.primal_ref() >= 0.0 {
                    Self::F64(AdScalar::from(v).sqrt().into_value())
                } else {
                    let promoted = v.map(|x| Complex64::new(x, 0.0));
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
                    let promoted = v.map(|x| Complex32::new(x, 0.0));
                    Self::C32(AdScalar::from(promoted).powf(exponent as f32).into_value())
                }
            }
            Self::F64(v) => {
                if *v.primal_ref() >= 0.0 {
                    Self::F64(AdScalar::from(v).powf(exponent).into_value())
                } else {
                    let promoted = v.map(|x| Complex64::new(x, 0.0));
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
            Self::C32(v) => Self::F32(v.map(|z| z.re)),
            Self::C64(v) => Self::F64(v.map(|z| z.re)),
        }
    }

    /// AD-preserving extraction of the imaginary component.
    ///
    /// - Real dtype: returns zero with matching real dtype.
    /// - Complex dtype: returns real dtype (`C32->F32`, `C64->F64`).
    pub fn imag_part(&self) -> Self {
        match self.clone() {
            Self::F32(v) => Self::F32(v.map(|_| 0.0_f32)),
            Self::F64(v) => Self::F64(v.map(|_| 0.0_f64)),
            Self::C32(v) => Self::F32(v.map(|z| z.im)),
            Self::C64(v) => Self::F64(v.map(|z| z.im)),
        }
    }

    /// Compose a complex AD scalar from real/imaginary AD scalars.
    ///
    /// Both inputs must be real and have matching precision (`F32/F32` or `F64/F64`).
    pub fn compose_complex(real: Self, imag: Self) -> Result<Self> {
        match (real, imag) {
            (Self::F32(re), Self::F32(im)) => Ok(Self::C32(checked_apply_binary_ad(
                re.map(|x| Complex32::new(x, 0.0)),
                im.map(|y| Complex32::new(0.0, y)),
                BinaryOp::Add,
            )?)),
            (Self::F64(re), Self::F64(im)) => Ok(Self::C64(checked_apply_binary_ad(
                re.map(|x| Complex64::new(x, 0.0)),
                im.map(|y| Complex64::new(0.0, y)),
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
    /// use tenferro_dyadtensor::{AdValue, DynAdValue};
    ///
    /// let x: DynAdValue = AdValue::primal(2.0_f64).into();
    /// let y: DynAdValue = AdValue::primal(Complex64::new(1.0, -3.0)).into();
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
    /// use tenferro_dyadtensor::{AdValue, DynAdValue};
    ///
    /// let x: DynAdValue = AdValue::primal(5.0_f64).into();
    /// let y: DynAdValue = AdValue::primal(2.0_f64).into();
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
    /// use tenferro_dyadtensor::{AdValue, DynAdValue};
    ///
    /// let x: DynAdValue = AdValue::primal(2.0_f64).into();
    /// let y: DynAdValue = AdValue::primal(Complex64::new(1.0, 2.0)).into();
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
    /// use tenferro_dyadtensor::{AdValue, DynAdValue};
    ///
    /// let x: DynAdValue = AdValue::primal(8.0_f64).into();
    /// let y: DynAdValue = AdValue::primal(2.0_f64).into();
    /// let z = x.try_div(y).unwrap();
    /// assert_eq!(z.primal(), tenferro_dyadtensor::DynScalar::F64(4.0));
    /// ```
    pub fn try_div(self, rhs: Self) -> Result<Self> {
        try_binary_dyn(self, rhs, BinaryOp::Div)
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

impl From<f32> for DynAdValue {
    fn from(value: f32) -> Self {
        Self::F32(AdValue::primal(value))
    }
}

impl From<f64> for DynAdValue {
    fn from(value: f64) -> Self {
        Self::F64(AdValue::primal(value))
    }
}

impl From<Complex32> for DynAdValue {
    fn from(value: Complex32) -> Self {
        Self::C32(AdValue::primal(value))
    }
}

impl From<Complex64> for DynAdValue {
    fn from(value: Complex64) -> Self {
        Self::C64(AdValue::primal(value))
    }
}

impl Add for DynAdValue {
    type Output = DynAdValue;

    fn add(self, rhs: Self) -> Self::Output {
        self.try_add(rhs)
            .unwrap_or_else(|e| panic!("DynAdValue add failed: {e}"))
    }
}

impl Sub for DynAdValue {
    type Output = DynAdValue;

    fn sub(self, rhs: Self) -> Self::Output {
        self.try_sub(rhs)
            .unwrap_or_else(|e| panic!("DynAdValue sub failed: {e}"))
    }
}

impl Mul for DynAdValue {
    type Output = DynAdValue;

    fn mul(self, rhs: Self) -> Self::Output {
        self.try_mul(rhs)
            .unwrap_or_else(|e| panic!("DynAdValue mul failed: {e}"))
    }
}

impl Div for DynAdValue {
    type Output = DynAdValue;

    fn div(self, rhs: Self) -> Self::Output {
        self.try_div(rhs)
            .unwrap_or_else(|e| panic!("DynAdValue div failed: {e}"))
    }
}

impl Neg for DynAdValue {
    type Output = DynAdValue;

    fn neg(self) -> Self::Output {
        match self {
            DynAdValue::F32(v) => DynAdValue::F32(v.map(|x| -x)),
            DynAdValue::F64(v) => DynAdValue::F64(v.map(|x| -x)),
            DynAdValue::C32(v) => DynAdValue::C32(v.map(|x| -x)),
            DynAdValue::C64(v) => DynAdValue::C64(v.map(|x| -x)),
        }
    }
}

macro_rules! impl_dynadvalue_scalar_binop {
    ($trait:ident, $method:ident, $scalar:ty) => {
        impl $trait<$scalar> for DynAdValue {
            type Output = DynAdValue;

            fn $method(self, rhs: $scalar) -> Self::Output {
                $trait::$method(self, DynAdValue::from(rhs))
            }
        }

        impl $trait<DynAdValue> for $scalar {
            type Output = DynAdValue;

            fn $method(self, rhs: DynAdValue) -> Self::Output {
                $trait::$method(DynAdValue::from(self), rhs)
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

impl TryFrom<DynAdValue> for f64 {
    type Error = &'static str;

    fn try_from(value: DynAdValue) -> core::result::Result<Self, Self::Error> {
        match value {
            DynAdValue::F32(v) => Ok(*v.primal_ref() as f64),
            DynAdValue::F64(v) => Ok(*v.primal_ref()),
            DynAdValue::C32(_) | DynAdValue::C64(_) => {
                Err("Cannot convert complex DynAdValue to f64")
            }
        }
    }
}

impl From<DynAdValue> for Complex64 {
    fn from(value: DynAdValue) -> Self {
        match value {
            DynAdValue::F32(v) => Complex64::new(*v.primal_ref() as f64, 0.0),
            DynAdValue::F64(v) => Complex64::new(*v.primal_ref(), 0.0),
            DynAdValue::C32(v) => {
                let z = v.primal_ref();
                Complex64::new(z.re as f64, z.im as f64)
            }
            DynAdValue::C64(v) => *v.primal_ref(),
        }
    }
}

impl Default for DynAdValue {
    fn default() -> Self {
        Self::new_real(0.0)
    }
}

impl Zero for DynAdValue {
    fn zero() -> Self {
        Self::new_real(0.0)
    }

    fn is_zero(&self) -> bool {
        DynAdValue::is_zero(self)
    }
}

impl One for DynAdValue {
    fn one() -> Self {
        Self::new_real(1.0)
    }
}

impl PartialOrd for DynAdValue {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        match (self, other) {
            (DynAdValue::F32(a), DynAdValue::F32(b)) => a.primal_ref().partial_cmp(b.primal_ref()),
            (DynAdValue::F64(a), DynAdValue::F64(b)) => a.primal_ref().partial_cmp(b.primal_ref()),
            (DynAdValue::F32(a), DynAdValue::F64(b)) => {
                (*a.primal_ref() as f64).partial_cmp(b.primal_ref())
            }
            (DynAdValue::F64(a), DynAdValue::F32(b)) => {
                a.primal_ref().partial_cmp(&(*b.primal_ref() as f64))
            }
            _ => None,
        }
    }
}

impl fmt::Display for DynAdValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DynAdValue::F32(v) => write!(f, "{}", v.primal_ref()),
            DynAdValue::F64(v) => write!(f, "{}", v.primal_ref()),
            DynAdValue::C32(v) => write!(f, "{}", v.primal_ref()),
            DynAdValue::C64(v) => write!(f, "{}", v.primal_ref()),
        }
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

fn map_ad_tensor_unary_typed<T, U, F>(input: &AdTensor<T>, f: F) -> Result<AdTensor<U>>
where
    T: Scalar + Copy,
    U: Scalar + Copy,
    F: Fn(T) -> U + Copy,
{
    let mapped = match input.as_value().clone() {
        AdValue::Primal(primal) => AdValue::Primal(tensor_map_unary_typed(&primal, f)?),
        AdValue::Forward { primal, tangent } => AdValue::Forward {
            primal: tensor_map_unary_typed(&primal, f)?,
            tangent: tensor_map_unary_typed(&tangent, f)?,
        },
        AdValue::Reverse {
            primal,
            node,
            tape,
            tangent,
        } => AdValue::Reverse {
            primal: tensor_map_unary_typed(&primal, f)?,
            node,
            tape,
            tangent: tangent
                .as_ref()
                .map(|t| tensor_map_unary_typed(t, f))
                .transpose()?,
        },
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
    primal: Tensor<T>,
    tangent: Option<Tensor<T>>,
    reverse: Option<(NodeId, TapeId)>,
}

fn split_ad_tensor_state<T: Scalar>(value: AdValue<Tensor<T>>) -> AdTensorBinaryState<T> {
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
    lhs: AdValue<Tensor<T>>,
    rhs: AdValue<Tensor<T>>,
) -> Result<AdValue<Tensor<T>>>
where
    T: Scalar + Copy + Add<Output = T>,
{
    let lhs_state = split_ad_tensor_state(lhs);
    let rhs_state = split_ad_tensor_state(rhs);

    let primal = tensor_add_typed(&lhs_state.primal, &rhs_state.primal)?;
    let tangent = match (lhs_state.tangent, rhs_state.tangent) {
        (Some(a), Some(b)) => Some(tensor_add_typed(&a, &b)?),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };

    match (lhs_state.reverse, rhs_state.reverse) {
        (None, None) => match tangent {
            Some(tangent) => Ok(AdValue::Forward { primal, tangent }),
            None => Ok(AdValue::Primal(primal)),
        },
        (Some((node, tape)), None) => Ok(AdValue::Reverse {
            primal,
            node,
            tape,
            tangent,
        }),
        (None, Some((node, tape))) => Ok(AdValue::Reverse {
            primal,
            node,
            tape,
            tangent,
        }),
        (Some((lhs_node, lhs_tape)), Some((_, rhs_tape))) => {
            if lhs_tape != rhs_tape {
                return Err(Error::InvalidAdTensor {
                    message: format!(
                        "compose_complex: reverse-mode tape mismatch (lhs={}, rhs={})",
                        lhs_tape.0, rhs_tape.0
                    ),
                });
            }
            Ok(AdValue::Reverse {
                primal,
                node: lhs_node,
                tape: lhs_tape,
                tangent,
            })
        }
    }
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
            Self::C32(v) => Ok(Self::F32(map_ad_tensor_unary_typed(v, |z| z.re)?)),
            Self::C64(v) => Ok(Self::F64(map_ad_tensor_unary_typed(v, |z| z.re)?)),
        }
    }

    /// AD-preserving extraction of the imaginary component.
    ///
    /// - Real dtype: returns zero with matching real dtype.
    /// - Complex dtype: returns real dtype (`C32->F32`, `C64->F64`).
    pub fn imag_part(&self) -> Result<Self> {
        match self {
            Self::F32(v) => Ok(Self::F32(map_ad_tensor_unary_typed(v, |_| 0.0_f32)?)),
            Self::F64(v) => Ok(Self::F64(map_ad_tensor_unary_typed(v, |_| 0.0_f64)?)),
            Self::C32(v) => Ok(Self::F32(map_ad_tensor_unary_typed(v, |z| z.im)?)),
            Self::C64(v) => Ok(Self::F64(map_ad_tensor_unary_typed(v, |z| z.im)?)),
        }
    }

    /// Compose a complex AD tensor from real/imaginary AD tensors.
    ///
    /// Both inputs must be real and have matching precision (`F32/F32` or `F64/F64`).
    pub fn compose_complex(real: Self, imag: Self) -> Result<Self> {
        match (real, imag) {
            (Self::F32(re), Self::F32(im)) => {
                let re_c = map_ad_tensor_unary_typed(&re, |x| Complex32::new(x, 0.0))?;
                let im_c = map_ad_tensor_unary_typed(&im, |y| Complex32::new(0.0, y))?;
                let merged = merge_add_ad_tensors(re_c.into_value(), im_c.into_value())?;
                Ok(Self::C32(AdTensor(merged)))
            }
            (Self::F64(re), Self::F64(im)) => {
                let re_c = map_ad_tensor_unary_typed(&re, |x| Complex64::new(x, 0.0))?;
                let im_c = map_ad_tensor_unary_typed(&im, |y| Complex64::new(0.0, y))?;
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

    #[test]
    fn dyn_ad_value_mul_mixed_real_complex_promotes_to_complex() {
        let lhs = DynAdValue::from(2.0_f64);
        let rhs = DynAdValue::from(Complex64::new(1.0, -3.0));
        let out = lhs * rhs;
        assert_eq!(out.scalar_type(), ScalarType::C64);
        assert_eq!(out.primal(), DynScalar::C64(Complex64::new(2.0, -6.0)));
    }

    #[test]
    fn dyn_ad_value_div_with_scalar_lhs_is_supported() {
        let rhs = DynAdValue::from(2.0_f64);
        let out = Complex64::new(4.0, -2.0) / rhs;
        assert_eq!(out.scalar_type(), ScalarType::C64);
        assert_eq!(out.primal(), DynScalar::C64(Complex64::new(2.0, -1.0)));
    }

    #[test]
    fn dyn_ad_value_try_add_rejects_cross_precision_pairs() {
        let lhs = DynAdValue::from(1.0_f32);
        let rhs = DynAdValue::from(2.0_f64);
        let err = lhs.try_add(rhs).unwrap_err();
        assert!(matches!(err, Error::InvalidAdScalar { .. }));
    }

    #[test]
    fn dyn_ad_value_try_mul_checks_reverse_tape_compatibility() {
        let lhs: DynAdValue = AdValue::reverse(2.0_f64, crate::NodeId(1), TapeId(7), None).into();
        let rhs: DynAdValue = AdValue::reverse(3.0_f64, crate::NodeId(2), TapeId(8), None).into();
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
