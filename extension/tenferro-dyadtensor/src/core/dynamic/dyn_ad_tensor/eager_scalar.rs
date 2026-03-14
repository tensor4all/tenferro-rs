use super::promotion::join_scalar_types;
use super::DynAdTensor;
use crate::{ad, Result};

macro_rules! define_dyn_unary_method {
    ($fn_name:ident, $typed_fn:path, $doc_label:literal) => {
        #[doc = concat!("Runs eager AD `", $doc_label, "` on a dynamic tensor.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = "```rust"]
        #[doc = "use tenferro_dyadtensor::{set_default_runtime, DynAdTensor, RuntimeContext};"]
        #[doc = "use tenferro_prims::CpuContext;"]
        #[doc = "use tenferro_tensor::{MemoryOrder, Tensor};"]
        #[doc = ""]
        #[doc = "let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));"]
        #[doc = "let x = DynAdTensor::new_primal("]
        #[doc = "    Tensor::<f64>::from_slice(&[0.0], &[], MemoryOrder::ColumnMajor).unwrap(),"]
        #[doc = ");"]
        #[doc = concat!("let y = x.", stringify!($fn_name), "().unwrap();")]
        #[doc = "assert_eq!(y.dims(), &[]);"]
        #[doc = "```"]
        pub fn $fn_name(&self) -> Result<Self> {
            match self {
                Self::F32(value) => Ok(Self::F32($typed_fn(value)?)),
                Self::F64(value) => Ok(Self::F64($typed_fn(value)?)),
                Self::C32(value) => Ok(Self::C32($typed_fn(value)?)),
                Self::C64(value) => Ok(Self::C64($typed_fn(value)?)),
            }
        }
    };
}

macro_rules! define_dyn_reduction_method {
    ($fn_name:ident, $typed_fn:path, $doc_label:literal) => {
        #[doc = concat!("Runs eager AD full `", $doc_label, "` reduction on a dynamic tensor.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = "```rust"]
        #[doc = "use tenferro_dyadtensor::{set_default_runtime, DynAdTensor, RuntimeContext};"]
        #[doc = "use tenferro_prims::CpuContext;"]
        #[doc = "use tenferro_tensor::{MemoryOrder, Tensor};"]
        #[doc = ""]
        #[doc = "let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));"]
        #[doc = "let x = DynAdTensor::new_primal("]
        #[doc = "    Tensor::<f64>::from_slice(&[1.0, 3.0], &[2], MemoryOrder::ColumnMajor).unwrap(),"]
        #[doc = ");"]
        #[doc = concat!("let y = x.", stringify!($fn_name), "().unwrap();")]
        #[doc = "assert_eq!(y.dims(), &[]);"]
        #[doc = "```"]
        pub fn $fn_name(&self) -> Result<Self> {
            match self {
                Self::F32(value) => Ok(Self::F32($typed_fn(value)?)),
                Self::F64(value) => Ok(Self::F64($typed_fn(value)?)),
                Self::C32(value) => Ok(Self::C32($typed_fn(value)?)),
                Self::C64(value) => Ok(Self::C64($typed_fn(value)?)),
            }
        }
    };
}

impl DynAdTensor {
    define_dyn_unary_method!(exp, ad::exp, "exp");
    define_dyn_reduction_method!(mean, ad::mean, "mean");

    /// Runs eager AD `add` on two dynamic tensors after applying the standard
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
    /// let lhs = DynAdTensor::new_primal(
    ///     Tensor::<f64>::from_slice(&[2.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    /// let rhs = DynAdTensor::new_primal(
    ///     Tensor::<Complex64>::from_slice(&[Complex64::new(0.0, 1.0)], &[], MemoryOrder::ColumnMajor)
    ///         .unwrap(),
    /// );
    /// let out = lhs.add(&rhs).unwrap();
    /// assert_eq!(out.scalar_type(), tenferro_dyadtensor::ScalarType::C64);
    /// ```
    pub fn add(&self, rhs: &Self) -> Result<Self> {
        let target = join_scalar_types(&[self.scalar_type(), rhs.scalar_type()])?;
        let lhs = self.promote_to(target)?;
        let rhs = rhs.promote_to(target)?;
        match (&lhs, &rhs) {
            (Self::F32(lhs), Self::F32(rhs)) => Ok(Self::F32(ad::add(lhs, rhs)?)),
            (Self::F64(lhs), Self::F64(rhs)) => Ok(Self::F64(ad::add(lhs, rhs)?)),
            (Self::C32(lhs), Self::C32(rhs)) => Ok(Self::C32(ad::add(lhs, rhs)?)),
            (Self::C64(lhs), Self::C64(rhs)) => Ok(Self::C64(ad::add(lhs, rhs)?)),
            _ => unreachable!("promotion join should normalize both operands to the same dtype"),
        }
    }
}
