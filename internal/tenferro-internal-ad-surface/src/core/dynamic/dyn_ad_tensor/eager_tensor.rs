use super::promotion::{promote_many_to_common, promote_many_to_common_owned};
use super::Tensor;
use crate::ops::ad;
use crate::{Error, Result};

impl Tensor {
    fn einsum_with_refs(subscripts: &str, operands: &[&Self]) -> Result<Self> {
        if operands.is_empty() {
            return Err(Error::InvalidAdTensor {
                message: "einsum requires at least one operand".to_string(),
            });
        }
        let (target, promoted) = promote_many_to_common(operands)?;

        match target {
            crate::ScalarType::F32
            | crate::ScalarType::F64
            | crate::ScalarType::C32
            | crate::ScalarType::C64 => {
                let refs: Vec<_> = promoted.iter().map(Tensor::as_dyn_ad_ref).collect();
                Ok(ad::einsum_dyn(subscripts, &refs)?.into())
            }
        }
    }

    fn einsum_with_owned(subscripts: &str, operands: Vec<Self>) -> Result<Self> {
        if operands.is_empty() {
            return Err(Error::InvalidAdTensor {
                message: "einsum requires at least one operand".to_string(),
            });
        }
        let (target, promoted) = promote_many_to_common_owned(operands)?;

        match target {
            crate::ScalarType::F32
            | crate::ScalarType::F64
            | crate::ScalarType::C32
            | crate::ScalarType::C64 => {
                let refs: Vec<_> = promoted.iter().map(Tensor::as_dyn_ad_ref).collect();
                Ok(ad::einsum_dyn(subscripts, &refs)?.into())
            }
        }
    }

    /// Runs eager AD full `sum` reduction on a dynamic tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::{set_default_runtime, Tensor, RuntimeContext};
    /// use tenferro_prims::CpuContext;
    /// use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};
    ///
    /// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    /// let x = Tensor::from_tensor(
    ///     DenseTensor::<f64>::from_slice(&[1.0, 3.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    /// let y = x.sum().unwrap();
    /// assert!(y.dims().is_empty());
    /// ```
    pub fn sum(&self) -> Result<Self> {
        Ok(ad::sum_dyn(self.as_dyn_ad_ref())?.into())
    }

    /// Runs eager AD einsum on dynamic tensors after applying the standard
    /// dynamic promotion join used by frontend scalar ops.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use num_complex::Complex64;
    /// use tenferro::{set_default_runtime, Tensor, RuntimeContext};
    /// use tenferro_prims::CpuContext;
    /// use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};
    ///
    /// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    /// let a = Tensor::from_tensor(
    ///     DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    /// let b = Tensor::from_tensor(
    ///     DenseTensor::<Complex64>::from_slice(
    ///         &[Complex64::new(1.0, 0.5), Complex64::new(-2.0, 1.0)],
    ///         &[2],
    ///         MemoryOrder::ColumnMajor,
    ///     )
    ///     .unwrap(),
    /// );
    /// let out = Tensor::einsum("i,i->", &[&a, &b]).unwrap();
    /// assert!(out.dims().is_empty());
    /// ```
    pub fn einsum(subscripts: &str, operands: &[&Self]) -> Result<Self> {
        Self::einsum_with_refs(subscripts, operands)
    }

    /// Runs direct AD einsum on owned dynamic tensors.
    ///
    /// This is the ownership-oriented variant of [`Tensor::einsum`]. It keeps
    /// promotion on the owned path and only borrows at the final typed AD
    /// execution boundary.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use num_complex::Complex64;
    /// use tenferro::{set_default_runtime, Tensor, RuntimeContext};
    /// use tenferro_prims::CpuContext;
    /// use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};
    ///
    /// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    /// let operands = vec![
    ///     Tensor::from_tensor(
    ///         DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor)
    ///             .unwrap(),
    ///     ),
    ///     Tensor::from_tensor(
    ///         DenseTensor::<Complex64>::from_slice(
    ///             &[Complex64::new(1.0, 0.5), Complex64::new(-2.0, 1.0)],
    ///             &[2],
    ///             MemoryOrder::ColumnMajor,
    ///         )
    ///         .unwrap(),
    ///     ),
    /// ];
    /// let out = Tensor::einsum_owned("i,i->", operands).unwrap();
    /// assert!(out.dims().is_empty());
    /// ```
    pub fn einsum_owned(subscripts: &str, operands: Vec<Self>) -> Result<Self> {
        Self::einsum_with_owned(subscripts, operands)
    }
}
