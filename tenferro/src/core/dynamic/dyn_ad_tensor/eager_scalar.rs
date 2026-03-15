use super::promotion::promote_pair_to_common;
use super::Tensor;
use crate::ops::ad;
use crate::{Error, Result};

macro_rules! define_dyn_unary_method {
    ($fn_name:ident, $typed_fn:path, $doc_label:literal) => {
        #[doc = concat!("Runs eager AD `", $doc_label, "` on a dynamic tensor.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = "```rust"]
        #[doc = "use tenferro::{set_default_runtime, Tensor, RuntimeContext};"]
        #[doc = "use tenferro_prims::CpuContext;"]
        #[doc = "use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};"]
        #[doc = ""]
        #[doc = "let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));"]
        #[doc = "let x = Tensor::from_tensor("]
        #[doc = "    DenseTensor::<f64>::from_slice(&[0.0], &[], MemoryOrder::ColumnMajor).unwrap(),"]
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
        #[doc = "use tenferro::{set_default_runtime, Tensor, RuntimeContext};"]
        #[doc = "use tenferro_prims::CpuContext;"]
        #[doc = "use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};"]
        #[doc = ""]
        #[doc = "let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));"]
        #[doc = "let x = Tensor::from_tensor("]
        #[doc = "    DenseTensor::<f64>::from_slice(&[1.0, 3.0], &[2], MemoryOrder::ColumnMajor).unwrap(),"]
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
    ($fn_name:ident, $typed_fn:path, $doc_label:literal, real) => {
        #[doc = concat!("Runs eager AD full `", $doc_label, "` reduction on a dynamic tensor.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = "```rust"]
        #[doc = "use tenferro::{set_default_runtime, Tensor, RuntimeContext};"]
        #[doc = "use tenferro_prims::CpuContext;"]
        #[doc = "use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};"]
        #[doc = ""]
        #[doc = "let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));"]
        #[doc = "let x = Tensor::from_tensor("]
        #[doc = "    DenseTensor::<f64>::from_slice(&[1.0, 3.0], &[2], MemoryOrder::ColumnMajor).unwrap(),"]
        #[doc = ");"]
        #[doc = concat!("let y = x.", stringify!($fn_name), "().unwrap();")]
        #[doc = "assert_eq!(y.dims(), &[]);"]
        #[doc = "```"]
        pub fn $fn_name(&self) -> Result<Self> {
            match self {
                Self::F32(value) => Ok(Self::F32($typed_fn(value)?)),
                Self::F64(value) => Ok(Self::F64($typed_fn(value)?)),
                Self::C32(_) | Self::C64(_) => Err(Error::InvalidAdTensor {
                    message: format!("{} requires real-valued input", stringify!($fn_name)),
                }),
            }
        }
    };
}

macro_rules! define_dyn_binary_method {
    ($fn_name:ident, $typed_fn:path, $doc_label:literal, generic) => {
        #[doc = concat!(
            "Runs eager AD `",
            $doc_label,
            "` on two dynamic tensors after applying the dynamic result-type promotion rule."
        )]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = "```rust"]
        #[doc = "use tenferro::{set_default_runtime, Tensor, RuntimeContext};"]
        #[doc = "use tenferro_prims::CpuContext;"]
        #[doc = "use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};"]
        #[doc = ""]
        #[doc = "let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));"]
        #[doc = "let lhs = Tensor::from_tensor("]
        #[doc = "    DenseTensor::<f64>::from_slice(&[2.0], &[], MemoryOrder::ColumnMajor).unwrap(),"]
        #[doc = ");"]
        #[doc = "let rhs = Tensor::from_tensor("]
        #[doc = "    DenseTensor::<f64>::from_slice(&[3.0], &[], MemoryOrder::ColumnMajor).unwrap(),"]
        #[doc = ");"]
        #[doc = concat!("let out = lhs.", stringify!($fn_name), "(&rhs).unwrap();")]
        #[doc = "assert_eq!(out.dims(), &[]);"]
        #[doc = "```"]
        pub fn $fn_name(&self, rhs: &Self) -> Result<Self> {
            let (_, lhs, rhs) = promote_pair_to_common(self, rhs)?;
            match (&lhs, &rhs) {
                (Self::F32(lhs), Self::F32(rhs)) => Ok(Self::F32($typed_fn(lhs, rhs)?)),
                (Self::F64(lhs), Self::F64(rhs)) => Ok(Self::F64($typed_fn(lhs, rhs)?)),
                (Self::C32(lhs), Self::C32(rhs)) => Ok(Self::C32($typed_fn(lhs, rhs)?)),
                (Self::C64(lhs), Self::C64(rhs)) => Ok(Self::C64($typed_fn(lhs, rhs)?)),
                _ => unreachable!("promotion join should normalize both operands to the same dtype"),
            }
        }
    };
    ($fn_name:ident, $typed_fn:path, $doc_label:literal, real) => {
        #[doc = concat!(
            "Runs eager AD `",
            $doc_label,
            "` on two real-valued dynamic tensors after applying the dynamic result-type promotion rule."
        )]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = "```rust"]
        #[doc = "use tenferro::{set_default_runtime, Tensor, RuntimeContext};"]
        #[doc = "use tenferro_prims::CpuContext;"]
        #[doc = "use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};"]
        #[doc = ""]
        #[doc = "let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));"]
        #[doc = "let lhs = Tensor::from_tensor("]
        #[doc = "    DenseTensor::<f64>::from_slice(&[2.0], &[], MemoryOrder::ColumnMajor).unwrap(),"]
        #[doc = ");"]
        #[doc = "let rhs = Tensor::from_tensor("]
        #[doc = "    DenseTensor::<f64>::from_slice(&[3.0], &[], MemoryOrder::ColumnMajor).unwrap(),"]
        #[doc = ");"]
        #[doc = concat!("let out = lhs.", stringify!($fn_name), "(&rhs).unwrap();")]
        #[doc = "assert_eq!(out.dims(), &[]);"]
        #[doc = "```"]
        pub fn $fn_name(&self, rhs: &Self) -> Result<Self> {
            let (_, lhs, rhs) = promote_pair_to_common(self, rhs)?;
            match (&lhs, &rhs) {
                (Self::F32(lhs), Self::F32(rhs)) => Ok(Self::F32($typed_fn(lhs, rhs)?)),
                (Self::F64(lhs), Self::F64(rhs)) => Ok(Self::F64($typed_fn(lhs, rhs)?)),
                _ => Err(Error::InvalidAdTensor {
                    message: format!(
                        "{} requires real-valued operands, got lhs={:?}, rhs={:?}",
                        stringify!($fn_name),
                        lhs.scalar_type(),
                        rhs.scalar_type()
                    ),
                }),
            }
        }
    };
}

impl Tensor {
    define_dyn_unary_method!(acos, ad::acos, "acos");
    define_dyn_unary_method!(acosh, ad::acosh, "acosh");
    define_dyn_unary_method!(asin, ad::asin, "asin");
    define_dyn_unary_method!(asinh, ad::asinh, "asinh");
    define_dyn_unary_method!(atan, ad::atan, "atan");
    define_dyn_unary_method!(atanh, ad::atanh, "atanh");
    define_dyn_unary_method!(cos, ad::cos, "cos");
    define_dyn_unary_method!(cosh, ad::cosh, "cosh");
    define_dyn_unary_method!(exp, ad::exp, "exp");
    define_dyn_unary_method!(expm1, ad::expm1, "expm1");
    define_dyn_unary_method!(log, ad::log, "log");
    define_dyn_unary_method!(log1p, ad::log1p, "log1p");
    define_dyn_unary_method!(sin, ad::sin, "sin");
    define_dyn_unary_method!(sinh, ad::sinh, "sinh");
    define_dyn_unary_method!(sqrt, ad::sqrt, "sqrt");
    define_dyn_unary_method!(tanh, ad::tanh, "tanh");

    define_dyn_reduction_method!(mean, ad::mean, "mean");
    define_dyn_reduction_method!(std, ad::std, "standard deviation", real);
    define_dyn_reduction_method!(var, ad::var, "variance", real);

    define_dyn_binary_method!(add, ad::add, "add", generic);
    define_dyn_binary_method!(atan2, ad::atan2, "atan2", real);
    define_dyn_binary_method!(hypot, ad::hypot, "hypot", real);
    define_dyn_binary_method!(pow, ad::pow, "pow", generic);
}
