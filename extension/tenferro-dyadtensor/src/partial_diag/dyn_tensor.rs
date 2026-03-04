use tenferro_einsum::Subscripts;
use tenferro_tensor::Tensor;

use crate::dyn_types::{DynTensor, ScalarType};
use crate::partial_diag::typed::AdTensor;
use crate::{Error, Result};

/// Runtime-dispatched PartialDiagonal tensor.
///
/// This is the dynamic counterpart of [`AdTensor<T>`].
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::partial_diag::DynAdTensor;
/// use tenferro_dyadtensor::DynTensor;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let dense = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
/// let x = DynAdTensor::from_dense(DynTensor::from(dense));
/// assert_eq!(x.scalar_type(), tenferro_dyadtensor::ScalarType::F64);
/// ```
#[derive(Debug, Clone)]
pub enum DynAdTensor {
    /// f32 payload.
    F32(AdTensor<f32>),
    /// f64 payload.
    F64(AdTensor<f64>),
    /// Complex32 payload.
    C32(AdTensor<num_complex::Complex32>),
    /// Complex64 payload.
    C64(AdTensor<num_complex::Complex64>),
}

impl DynAdTensor {
    /// Construct from dense runtime tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::partial_diag::DynAdTensor;
    /// use tenferro_dyadtensor::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let dense = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = DynAdTensor::from_dense(DynTensor::from(dense));
    /// assert_eq!(x.scalar_type(), tenferro_dyadtensor::ScalarType::F64);
    /// ```
    pub fn from_dense(payload: DynTensor) -> Self {
        match payload {
            DynTensor::F32(t) => Self::F32(AdTensor::from_dense(t)),
            DynTensor::F64(t) => Self::F64(AdTensor::from_dense(t)),
            DynTensor::C32(t) => Self::C32(AdTensor::from_dense(t)),
            DynTensor::C64(t) => Self::C64(AdTensor::from_dense(t)),
        }
    }

    /// Construct from logical metadata and compressed runtime payload.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::partial_diag::DynAdTensor;
    /// use tenferro_dyadtensor::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let payload = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = DynAdTensor::new(vec![2, 2], vec![0, 0], DynTensor::from(payload)).unwrap();
    /// assert_eq!(x.axis_classes(), &[0, 0]);
    /// ```
    pub fn new(
        logical_dims: Vec<usize>,
        axis_classes: Vec<usize>,
        payload: DynTensor,
    ) -> Result<Self> {
        match payload {
            DynTensor::F32(t) => Ok(Self::F32(AdTensor::new(logical_dims, axis_classes, t)?)),
            DynTensor::F64(t) => Ok(Self::F64(AdTensor::new(logical_dims, axis_classes, t)?)),
            DynTensor::C32(t) => Ok(Self::C32(AdTensor::new(logical_dims, axis_classes, t)?)),
            DynTensor::C64(t) => Ok(Self::C64(AdTensor::new(logical_dims, axis_classes, t)?)),
        }
    }

    /// Construct a diagonal-like dynamic PartialDiagonal tensor from vector payload.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::partial_diag::DynAdTensor;
    /// use tenferro_dyadtensor::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let payload = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = DynAdTensor::from_diagonal_vector(DynTensor::from(payload), 2).unwrap();
    /// assert_eq!(x.logical_dims(), &[2, 2]);
    /// ```
    pub fn from_diagonal_vector(payload: DynTensor, logical_rank: usize) -> Result<Self> {
        match payload {
            DynTensor::F32(t) => Ok(Self::F32(AdTensor::from_diagonal_vector(t, logical_rank)?)),
            DynTensor::F64(t) => Ok(Self::F64(AdTensor::from_diagonal_vector(t, logical_rank)?)),
            DynTensor::C32(t) => Ok(Self::C32(AdTensor::from_diagonal_vector(t, logical_rank)?)),
            DynTensor::C64(t) => Ok(Self::C64(AdTensor::from_diagonal_vector(t, logical_rank)?)),
        }
    }

    /// Runtime scalar type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::partial_diag::DynAdTensor;
    /// use tenferro_dyadtensor::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let dense = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = DynAdTensor::from_dense(DynTensor::from(dense));
    /// assert_eq!(x.scalar_type(), tenferro_dyadtensor::ScalarType::F64);
    /// ```
    pub fn scalar_type(&self) -> ScalarType {
        match self {
            Self::F32(_) => ScalarType::F32,
            Self::F64(_) => ScalarType::F64,
            Self::C32(_) => ScalarType::C32,
            Self::C64(_) => ScalarType::C64,
        }
    }

    /// Logical dimensions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::partial_diag::DynAdTensor;
    /// use tenferro_dyadtensor::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let payload = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = DynAdTensor::from_diagonal_vector(DynTensor::from(payload), 2).unwrap();
    /// assert_eq!(x.logical_dims(), &[2, 2]);
    /// ```
    pub fn logical_dims(&self) -> &[usize] {
        match self {
            Self::F32(t) => t.logical_dims(),
            Self::F64(t) => t.logical_dims(),
            Self::C32(t) => t.logical_dims(),
            Self::C64(t) => t.logical_dims(),
        }
    }

    /// Axis classes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::partial_diag::DynAdTensor;
    /// use tenferro_dyadtensor::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let payload = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = DynAdTensor::from_diagonal_vector(DynTensor::from(payload), 2).unwrap();
    /// assert_eq!(x.axis_classes(), &[0, 0]);
    /// ```
    pub fn axis_classes(&self) -> &[usize] {
        match self {
            Self::F32(t) => t.axis_classes(),
            Self::F64(t) => t.axis_classes(),
            Self::C32(t) => t.axis_classes(),
            Self::C64(t) => t.axis_classes(),
        }
    }

    /// Returns `true` when the underlying layout is dense.
    pub fn is_dense(&self) -> bool {
        match self {
            Self::F32(t) => t.is_dense(),
            Self::F64(t) => t.is_dense(),
            Self::C32(t) => t.is_dense(),
            Self::C64(t) => t.is_dense(),
        }
    }

    /// Returns `true` when the underlying layout is diagonal.
    pub fn is_diag(&self) -> bool {
        match self {
            Self::F32(t) => t.is_diag(),
            Self::F64(t) => t.is_diag(),
            Self::C32(t) => t.is_diag(),
            Self::C64(t) => t.is_diag(),
        }
    }

    /// Borrow compressed payload as runtime tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::partial_diag::DynAdTensor;
    /// use tenferro_dyadtensor::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let dense = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = DynAdTensor::from_dense(DynTensor::from(dense));
    /// let p = x.payload();
    /// assert_eq!(p.scalar_type(), tenferro_dyadtensor::ScalarType::F64);
    /// ```
    pub fn payload(&self) -> DynTensor {
        match self {
            Self::F32(t) => DynTensor::from(t.payload().clone()),
            Self::F64(t) => DynTensor::from(t.payload().clone()),
            Self::C32(t) => DynTensor::from(t.payload().clone()),
            Self::C64(t) => DynTensor::from(t.payload().clone()),
        }
    }

    /// Materialize into dense runtime tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Requires default runtime to be configured.
    /// let dense = x.to_dense()?;
    /// ```
    pub fn to_dense(&self) -> Result<DynTensor> {
        match self {
            Self::F32(t) => Ok(DynTensor::from(t.to_dense()?)),
            Self::F64(t) => Ok(DynTensor::from(t.to_dense()?)),
            Self::C32(t) => Ok(DynTensor::from(t.to_dense()?)),
            Self::C64(t) => Ok(DynTensor::from(t.to_dense()?)),
        }
    }

    /// Runtime-dispatched PartialDiagonal einsum/contract.
    ///
    /// All operands must have the same scalar type.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Requires default runtime to be configured.
    /// let out = DynAdTensor::einsum_with_subscripts(&subs, &[&a, &b])?;
    /// ```
    pub fn einsum_with_subscripts(subscripts: &Subscripts, operands: &[&Self]) -> Result<Self> {
        let Some(first) = operands.first() else {
            return Err(Error::InvalidAdTensor {
                message: "DynAdTensor::einsum_with_subscripts requires at least one operand"
                    .to_string(),
            });
        };
        let scalar_type = first.scalar_type();
        if operands.iter().any(|op| op.scalar_type() != scalar_type) {
            return Err(Error::InvalidAdTensor {
                message: "mixed scalar types are not supported in DynAdTensor einsum".to_string(),
            });
        }

        match scalar_type {
            ScalarType::F32 => {
                let typed = collect_typed_refs_f32(operands)?;
                Ok(Self::F32(AdTensor::einsum_with_subscripts(
                    subscripts, &typed,
                )?))
            }
            ScalarType::F64 => {
                let typed = collect_typed_refs_f64(operands)?;
                Ok(Self::F64(AdTensor::einsum_with_subscripts(
                    subscripts, &typed,
                )?))
            }
            ScalarType::C32 => {
                let typed = collect_typed_refs_c32(operands)?;
                Ok(Self::C32(AdTensor::einsum_with_subscripts(
                    subscripts, &typed,
                )?))
            }
            ScalarType::C64 => {
                let typed = collect_typed_refs_c64(operands)?;
                Ok(Self::C64(AdTensor::einsum_with_subscripts(
                    subscripts, &typed,
                )?))
            }
        }
    }
}

fn collect_typed_refs_f32<'a>(operands: &'a [&'a DynAdTensor]) -> Result<Vec<&'a AdTensor<f32>>> {
    operands
        .iter()
        .map(|op| match op {
            DynAdTensor::F32(v) => Ok(v),
            _ => Err(Error::InvalidAdTensor {
                message: "expected f32 operand".to_string(),
            }),
        })
        .collect()
}

fn collect_typed_refs_f64<'a>(operands: &'a [&'a DynAdTensor]) -> Result<Vec<&'a AdTensor<f64>>> {
    operands
        .iter()
        .map(|op| match op {
            DynAdTensor::F64(v) => Ok(v),
            _ => Err(Error::InvalidAdTensor {
                message: "expected f64 operand".to_string(),
            }),
        })
        .collect()
}

fn collect_typed_refs_c32<'a>(
    operands: &'a [&'a DynAdTensor],
) -> Result<Vec<&'a AdTensor<num_complex::Complex32>>> {
    operands
        .iter()
        .map(|op| match op {
            DynAdTensor::C32(v) => Ok(v),
            _ => Err(Error::InvalidAdTensor {
                message: "expected Complex32 operand".to_string(),
            }),
        })
        .collect()
}

fn collect_typed_refs_c64<'a>(
    operands: &'a [&'a DynAdTensor],
) -> Result<Vec<&'a AdTensor<num_complex::Complex64>>> {
    operands
        .iter()
        .map(|op| match op {
            DynAdTensor::C64(v) => Ok(v),
            _ => Err(Error::InvalidAdTensor {
                message: "expected Complex64 operand".to_string(),
            }),
        })
        .collect()
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

impl From<AdTensor<num_complex::Complex32>> for DynAdTensor {
    fn from(value: AdTensor<num_complex::Complex32>) -> Self {
        Self::C32(value)
    }
}

impl From<AdTensor<num_complex::Complex64>> for DynAdTensor {
    fn from(value: AdTensor<num_complex::Complex64>) -> Self {
        Self::C64(value)
    }
}

impl From<DynAdTensor> for DynTensor {
    fn from(value: DynAdTensor) -> Self {
        match value {
            DynAdTensor::F32(v) => DynTensor::from(v.into_payload()),
            DynAdTensor::F64(v) => DynTensor::from(v.into_payload()),
            DynAdTensor::C32(v) => DynTensor::from(v.into_payload()),
            DynAdTensor::C64(v) => DynTensor::from(v.into_payload()),
        }
    }
}

impl From<&DynAdTensor> for DynTensor {
    fn from(value: &DynAdTensor) -> Self {
        value.payload()
    }
}

impl DynAdTensor {
    /// Access typed f64 payload view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::partial_diag::DynAdTensor;
    /// use tenferro_dyadtensor::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let dense = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = DynAdTensor::from_dense(DynTensor::from(dense));
    /// assert!(x.as_f64_payload().is_some());
    /// ```
    pub fn as_f64_payload(&self) -> Option<&Tensor<f64>> {
        if let Self::F64(v) = self {
            Some(v.payload())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{set_default_runtime, RuntimeContext};
    use tenferro_prims::CpuContext;
    use tenferro_tensor::{MemoryOrder, Tensor};

    fn vector(data: &[f64]) -> Tensor<f64> {
        Tensor::<f64>::from_slice(data, &[data.len()], MemoryOrder::ColumnMajor).unwrap()
    }

    #[test]
    fn dyn_diag_roundtrip_dense() {
        let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
        let x = DynAdTensor::from_diagonal_vector(DynTensor::from(vector(&[1.0, 2.0, 3.0])), 2)
            .unwrap();
        assert!(!x.is_dense());
        assert!(x.is_diag());
        let dense = x.to_dense().unwrap();
        match dense {
            DynTensor::F64(t) => assert_eq!(t.dims(), &[3, 3]),
            _ => panic!("expected f64 tensor"),
        }
    }

    #[test]
    fn dyn_einsum_diag_chain() {
        let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
        let a = DynAdTensor::from_diagonal_vector(DynTensor::from(vector(&[1.0, 2.0, 3.0])), 2)
            .unwrap();
        let b = DynAdTensor::from_diagonal_vector(DynTensor::from(vector(&[4.0, 5.0, 6.0])), 2)
            .unwrap();
        let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
        let c = DynAdTensor::einsum_with_subscripts(&subs, &[&a, &b]).unwrap();
        assert_eq!(c.axis_classes(), &[0, 0]);
        match c.payload() {
            DynTensor::F64(t) => assert_eq!(t.dims(), &[3]),
            _ => panic!("expected f64 payload"),
        }
    }
}
