use super::promotion::join_scalar_types;
use super::DynAdTensor;
use crate::{ad, AdTensor, Error, Result};

impl DynAdTensor {
    /// Runs eager AD full `sum` reduction on a dynamic tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{set_default_runtime, DynAdTensor, RuntimeContext};
    /// use tenferro_prims::CpuContext;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    /// let x = DynAdTensor::new_primal(
    ///     Tensor::<f64>::from_slice(&[1.0, 3.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    /// let y = x.sum().unwrap();
    /// assert_eq!(y.dims(), &[]);
    /// ```
    pub fn sum(&self) -> Result<Self> {
        match self {
            Self::F32(value) => Ok(Self::F32(ad::sum(value)?)),
            Self::F64(value) => Ok(Self::F64(ad::sum(value)?)),
            Self::C32(value) => Ok(Self::C32(ad::sum(value)?)),
            Self::C64(value) => Ok(Self::C64(ad::sum(value)?)),
        }
    }

    /// Runs eager AD einsum on dynamic tensors after applying the standard
    /// dynamic promotion join used by dyadtensor scalar ops.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex64;
    /// use tenferro_dyadtensor::{set_default_runtime, DynAdTensor, RuntimeContext};
    /// use tenferro_prims::CpuContext;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    /// let a = DynAdTensor::new_primal(
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    /// let b = DynAdTensor::new_primal(
    ///     Tensor::<Complex64>::from_slice(
    ///         &[Complex64::new(1.0, 0.5), Complex64::new(-2.0, 1.0)],
    ///         &[2],
    ///         MemoryOrder::ColumnMajor,
    ///     )
    ///     .unwrap(),
    /// );
    /// let out = DynAdTensor::einsum("i,i->", &[&a, &b]).unwrap();
    /// assert_eq!(out.dims(), &[]);
    /// ```
    pub fn einsum(subscripts: &str, operands: &[&Self]) -> Result<Self> {
        if operands.is_empty() {
            return Err(Error::InvalidAdTensor {
                message: "einsum requires at least one operand".to_string(),
            });
        }
        let target = join_scalar_types(
            &operands
                .iter()
                .map(|operand| operand.scalar_type())
                .collect::<Vec<_>>(),
        )?;
        let promoted = operands
            .iter()
            .map(|operand| operand.promote_to(target))
            .collect::<Result<Vec<_>>>()?;

        match target {
            crate::ScalarType::F32 => {
                let refs: Vec<&AdTensor<f32>> = promoted
                    .iter()
                    .map(|operand| match operand {
                        Self::F32(value) => value,
                        _ => unreachable!("promotion join should normalize all operands to f32"),
                    })
                    .collect();
                Ok(Self::F32(ad::einsum(subscripts, &refs)?))
            }
            crate::ScalarType::F64 => {
                let refs: Vec<&AdTensor<f64>> = promoted
                    .iter()
                    .map(|operand| match operand {
                        Self::F64(value) => value,
                        _ => unreachable!("promotion join should normalize all operands to f64"),
                    })
                    .collect();
                Ok(Self::F64(ad::einsum(subscripts, &refs)?))
            }
            crate::ScalarType::C32 => {
                let refs: Vec<&AdTensor<num_complex::Complex32>> = promoted
                    .iter()
                    .map(|operand| match operand {
                        Self::C32(value) => value,
                        _ => unreachable!("promotion join should normalize all operands to c32"),
                    })
                    .collect();
                Ok(Self::C32(ad::einsum(subscripts, &refs)?))
            }
            crate::ScalarType::C64 => {
                let refs: Vec<&AdTensor<num_complex::Complex64>> = promoted
                    .iter()
                    .map(|operand| match operand {
                        Self::C64(value) => value,
                        _ => unreachable!("promotion join should normalize all operands to c64"),
                    })
                    .collect();
                Ok(Self::C64(ad::einsum(subscripts, &refs)?))
            }
        }
    }
}
