use tenferro_einsum::Subscripts;
use tenferro_tensor::Tensor;

use crate::dyn_types::{DynTensor, ScalarType};
use crate::partial_diag::typed::PartialDiagTensor;
use crate::{Error, Result};

/// Runtime-dispatched PartialDiagonal tensor.
///
/// This is the dynamic counterpart of [`PartialDiagTensor<T>`].
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::partial_diag::DynPartialDiagTensor;
/// use tenferro_dyadtensor::DynTensor;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let dense = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
/// let x = DynPartialDiagTensor::from_dense(DynTensor::from(dense));
/// assert_eq!(x.scalar_type(), tenferro_dyadtensor::ScalarType::F64);
/// ```
#[derive(Debug, Clone)]
pub enum DynPartialDiagTensor {
    /// f32 payload.
    F32(PartialDiagTensor<f32>),
    /// f64 payload.
    F64(PartialDiagTensor<f64>),
    /// Complex32 payload.
    C32(PartialDiagTensor<num_complex::Complex32>),
    /// Complex64 payload.
    C64(PartialDiagTensor<num_complex::Complex64>),
}

impl DynPartialDiagTensor {
    /// Construct from dense runtime tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::partial_diag::DynPartialDiagTensor;
    /// use tenferro_dyadtensor::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let dense = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = DynPartialDiagTensor::from_dense(DynTensor::from(dense));
    /// assert_eq!(x.scalar_type(), tenferro_dyadtensor::ScalarType::F64);
    /// ```
    pub fn from_dense(payload: DynTensor) -> Self {
        match payload {
            DynTensor::F32(t) => Self::F32(PartialDiagTensor::from_dense(t)),
            DynTensor::F64(t) => Self::F64(PartialDiagTensor::from_dense(t)),
            DynTensor::C32(t) => Self::C32(PartialDiagTensor::from_dense(t)),
            DynTensor::C64(t) => Self::C64(PartialDiagTensor::from_dense(t)),
        }
    }

    /// Construct from logical metadata and compressed runtime payload.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::partial_diag::DynPartialDiagTensor;
    /// use tenferro_dyadtensor::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let payload = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = DynPartialDiagTensor::new(vec![2, 2], vec![0, 0], DynTensor::from(payload)).unwrap();
    /// assert_eq!(x.axis_classes(), &[0, 0]);
    /// ```
    pub fn new(
        logical_dims: Vec<usize>,
        axis_classes: Vec<usize>,
        payload: DynTensor,
    ) -> Result<Self> {
        match payload {
            DynTensor::F32(t) => Ok(Self::F32(PartialDiagTensor::new(
                logical_dims,
                axis_classes,
                t,
            )?)),
            DynTensor::F64(t) => Ok(Self::F64(PartialDiagTensor::new(
                logical_dims,
                axis_classes,
                t,
            )?)),
            DynTensor::C32(t) => Ok(Self::C32(PartialDiagTensor::new(
                logical_dims,
                axis_classes,
                t,
            )?)),
            DynTensor::C64(t) => Ok(Self::C64(PartialDiagTensor::new(
                logical_dims,
                axis_classes,
                t,
            )?)),
        }
    }

    /// Construct a diagonal-like dynamic PartialDiagonal tensor from vector payload.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::partial_diag::DynPartialDiagTensor;
    /// use tenferro_dyadtensor::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let payload = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = DynPartialDiagTensor::from_diagonal_vector(DynTensor::from(payload), 2).unwrap();
    /// assert_eq!(x.logical_dims(), &[2, 2]);
    /// ```
    pub fn from_diagonal_vector(payload: DynTensor, logical_rank: usize) -> Result<Self> {
        match payload {
            DynTensor::F32(t) => Ok(Self::F32(PartialDiagTensor::from_diagonal_vector(
                t,
                logical_rank,
            )?)),
            DynTensor::F64(t) => Ok(Self::F64(PartialDiagTensor::from_diagonal_vector(
                t,
                logical_rank,
            )?)),
            DynTensor::C32(t) => Ok(Self::C32(PartialDiagTensor::from_diagonal_vector(
                t,
                logical_rank,
            )?)),
            DynTensor::C64(t) => Ok(Self::C64(PartialDiagTensor::from_diagonal_vector(
                t,
                logical_rank,
            )?)),
        }
    }

    /// Runtime scalar type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::partial_diag::DynPartialDiagTensor;
    /// use tenferro_dyadtensor::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let dense = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = DynPartialDiagTensor::from_dense(DynTensor::from(dense));
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
    /// use tenferro_dyadtensor::partial_diag::DynPartialDiagTensor;
    /// use tenferro_dyadtensor::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let payload = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = DynPartialDiagTensor::from_diagonal_vector(DynTensor::from(payload), 2).unwrap();
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
    /// use tenferro_dyadtensor::partial_diag::DynPartialDiagTensor;
    /// use tenferro_dyadtensor::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let payload = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = DynPartialDiagTensor::from_diagonal_vector(DynTensor::from(payload), 2).unwrap();
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

    /// Borrow compressed payload as runtime tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::partial_diag::DynPartialDiagTensor;
    /// use tenferro_dyadtensor::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let dense = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = DynPartialDiagTensor::from_dense(DynTensor::from(dense));
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
    /// let out = DynPartialDiagTensor::einsum_with_subscripts(&subs, &[&a, &b])?;
    /// ```
    pub fn einsum_with_subscripts(subscripts: &Subscripts, operands: &[&Self]) -> Result<Self> {
        let Some(first) = operands.first() else {
            return Err(Error::InvalidAdTensor {
                message:
                    "DynPartialDiagTensor::einsum_with_subscripts requires at least one operand"
                        .to_string(),
            });
        };
        let scalar_type = first.scalar_type();
        if operands.iter().any(|op| op.scalar_type() != scalar_type) {
            return Err(Error::InvalidAdTensor {
                message: "mixed scalar types are not supported in DynPartialDiagTensor einsum"
                    .to_string(),
            });
        }

        match scalar_type {
            ScalarType::F32 => {
                let typed = collect_typed_refs_f32(operands)?;
                Ok(Self::F32(PartialDiagTensor::einsum_with_subscripts(
                    subscripts, &typed,
                )?))
            }
            ScalarType::F64 => {
                let typed = collect_typed_refs_f64(operands)?;
                Ok(Self::F64(PartialDiagTensor::einsum_with_subscripts(
                    subscripts, &typed,
                )?))
            }
            ScalarType::C32 => {
                let typed = collect_typed_refs_c32(operands)?;
                Ok(Self::C32(PartialDiagTensor::einsum_with_subscripts(
                    subscripts, &typed,
                )?))
            }
            ScalarType::C64 => {
                let typed = collect_typed_refs_c64(operands)?;
                Ok(Self::C64(PartialDiagTensor::einsum_with_subscripts(
                    subscripts, &typed,
                )?))
            }
        }
    }
}

fn collect_typed_refs_f32<'a>(
    operands: &'a [&'a DynPartialDiagTensor],
) -> Result<Vec<&'a PartialDiagTensor<f32>>> {
    operands
        .iter()
        .map(|op| match op {
            DynPartialDiagTensor::F32(v) => Ok(v),
            _ => Err(Error::InvalidAdTensor {
                message: "expected f32 operand".to_string(),
            }),
        })
        .collect()
}

fn collect_typed_refs_f64<'a>(
    operands: &'a [&'a DynPartialDiagTensor],
) -> Result<Vec<&'a PartialDiagTensor<f64>>> {
    operands
        .iter()
        .map(|op| match op {
            DynPartialDiagTensor::F64(v) => Ok(v),
            _ => Err(Error::InvalidAdTensor {
                message: "expected f64 operand".to_string(),
            }),
        })
        .collect()
}

fn collect_typed_refs_c32<'a>(
    operands: &'a [&'a DynPartialDiagTensor],
) -> Result<Vec<&'a PartialDiagTensor<num_complex::Complex32>>> {
    operands
        .iter()
        .map(|op| match op {
            DynPartialDiagTensor::C32(v) => Ok(v),
            _ => Err(Error::InvalidAdTensor {
                message: "expected Complex32 operand".to_string(),
            }),
        })
        .collect()
}

fn collect_typed_refs_c64<'a>(
    operands: &'a [&'a DynPartialDiagTensor],
) -> Result<Vec<&'a PartialDiagTensor<num_complex::Complex64>>> {
    operands
        .iter()
        .map(|op| match op {
            DynPartialDiagTensor::C64(v) => Ok(v),
            _ => Err(Error::InvalidAdTensor {
                message: "expected Complex64 operand".to_string(),
            }),
        })
        .collect()
}

impl From<PartialDiagTensor<f32>> for DynPartialDiagTensor {
    fn from(value: PartialDiagTensor<f32>) -> Self {
        Self::F32(value)
    }
}

impl From<PartialDiagTensor<f64>> for DynPartialDiagTensor {
    fn from(value: PartialDiagTensor<f64>) -> Self {
        Self::F64(value)
    }
}

impl From<PartialDiagTensor<num_complex::Complex32>> for DynPartialDiagTensor {
    fn from(value: PartialDiagTensor<num_complex::Complex32>) -> Self {
        Self::C32(value)
    }
}

impl From<PartialDiagTensor<num_complex::Complex64>> for DynPartialDiagTensor {
    fn from(value: PartialDiagTensor<num_complex::Complex64>) -> Self {
        Self::C64(value)
    }
}

impl From<DynPartialDiagTensor> for DynTensor {
    fn from(value: DynPartialDiagTensor) -> Self {
        match value {
            DynPartialDiagTensor::F32(v) => DynTensor::from(v.into_payload()),
            DynPartialDiagTensor::F64(v) => DynTensor::from(v.into_payload()),
            DynPartialDiagTensor::C32(v) => DynTensor::from(v.into_payload()),
            DynPartialDiagTensor::C64(v) => DynTensor::from(v.into_payload()),
        }
    }
}

impl From<&DynPartialDiagTensor> for DynTensor {
    fn from(value: &DynPartialDiagTensor) -> Self {
        value.payload()
    }
}

impl DynPartialDiagTensor {
    /// Access typed f64 payload view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::partial_diag::DynPartialDiagTensor;
    /// use tenferro_dyadtensor::DynTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let dense = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = DynPartialDiagTensor::from_dense(DynTensor::from(dense));
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
        let x = DynPartialDiagTensor::from_diagonal_vector(
            DynTensor::from(vector(&[1.0, 2.0, 3.0])),
            2,
        )
        .unwrap();
        let dense = x.to_dense().unwrap();
        match dense {
            DynTensor::F64(t) => assert_eq!(t.dims(), &[3, 3]),
            _ => panic!("expected f64 tensor"),
        }
    }

    #[test]
    fn dyn_einsum_diag_chain() {
        let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
        let a = DynPartialDiagTensor::from_diagonal_vector(
            DynTensor::from(vector(&[1.0, 2.0, 3.0])),
            2,
        )
        .unwrap();
        let b = DynPartialDiagTensor::from_diagonal_vector(
            DynTensor::from(vector(&[4.0, 5.0, 6.0])),
            2,
        )
        .unwrap();
        let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
        let c = DynPartialDiagTensor::einsum_with_subscripts(&subs, &[&a, &b]).unwrap();
        assert_eq!(c.axis_classes(), &[0, 0]);
        match c.payload() {
            DynTensor::F64(t) => assert_eq!(t.dims(), &[3]),
            _ => panic!("expected f64 payload"),
        }
    }
}
