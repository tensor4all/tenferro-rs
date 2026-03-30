use super::promotion::promote_pair_to_common;
use super::Tensor;
use crate::ops::ad;
use crate::Result;

macro_rules! define_dyn_unary_method {
    ($fn_name:ident, $dyn_fn:path, $doc_label:literal) => {
        #[doc = concat!("Runs eager AD `", $doc_label, "` on a dynamic tensor.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = "```ignore"]
        #[doc = "use tenferro::{set_default_runtime, Tensor, RuntimeContext};"]
        #[doc = "use tenferro_prims::CpuContext;"]
        #[doc = "use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};"]
        #[doc = ""]
        #[doc = "let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));"]
        #[doc = "let x = Tensor::from_tensor("]
        #[doc = "    DenseTensor::<f64>::from_slice(&[0.0], &[], MemoryOrder::ColumnMajor).unwrap(),"]
        #[doc = ");"]
        #[doc = concat!("let y = x.", stringify!($fn_name), "().unwrap();")]
        #[doc = "assert!(y.dims().is_empty());"]
        #[doc = "```"]
        pub fn $fn_name(&self) -> Result<Self> {
            Ok($dyn_fn(self.as_dyn_ad_ref())?.into())
        }
    };
}

macro_rules! define_dyn_reduction_method {
    ($fn_name:ident, $dyn_fn:path, $doc_label:literal) => {
        #[doc = concat!("Runs eager AD full `", $doc_label, "` reduction on a dynamic tensor.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = "```ignore"]
        #[doc = "use tenferro::{set_default_runtime, Tensor, RuntimeContext};"]
        #[doc = "use tenferro_prims::CpuContext;"]
        #[doc = "use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};"]
        #[doc = ""]
        #[doc = "let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));"]
        #[doc = "let x = Tensor::from_tensor("]
        #[doc = "    DenseTensor::<f64>::from_slice(&[1.0, 3.0], &[2], MemoryOrder::ColumnMajor).unwrap(),"]
        #[doc = ");"]
        #[doc = concat!("let y = x.", stringify!($fn_name), "().unwrap();")]
        #[doc = "assert!(y.dims().is_empty());"]
        #[doc = "```"]
        pub fn $fn_name(&self) -> Result<Self> {
            Ok($dyn_fn(self.as_dyn_ad_ref())?.into())
        }
    };
    ($fn_name:ident, $dyn_fn:path, $doc_label:literal, real) => {
        #[doc = concat!("Runs eager AD full `", $doc_label, "` reduction on a dynamic tensor.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = "```ignore"]
        #[doc = "use tenferro::{set_default_runtime, Tensor, RuntimeContext};"]
        #[doc = "use tenferro_prims::CpuContext;"]
        #[doc = "use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};"]
        #[doc = ""]
        #[doc = "let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));"]
        #[doc = "let x = Tensor::from_tensor("]
        #[doc = "    DenseTensor::<f64>::from_slice(&[1.0, 3.0], &[2], MemoryOrder::ColumnMajor).unwrap(),"]
        #[doc = ");"]
        #[doc = concat!("let y = x.", stringify!($fn_name), "().unwrap();")]
        #[doc = "assert!(y.dims().is_empty());"]
        #[doc = "```"]
        pub fn $fn_name(&self) -> Result<Self> {
            Ok($dyn_fn(self.as_dyn_ad_ref())?.into())
        }
    };
}

macro_rules! define_dyn_binary_method {
    ($fn_name:ident, $dyn_fn:path, $doc_label:literal, generic) => {
        #[doc = concat!(
            "Runs eager AD `",
            $doc_label,
            "` on two dynamic tensors after applying the dynamic result-type promotion rule."
        )]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = "```ignore"]
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
        #[doc = "assert!(out.dims().is_empty());"]
        #[doc = "```"]
        pub fn $fn_name(&self, rhs: &Self) -> Result<Self> {
            let (_, lhs, rhs) = promote_pair_to_common(self, rhs)?;
            Ok($dyn_fn(lhs.as_dyn_ad_ref(), rhs.as_dyn_ad_ref())?.into())
        }
    };
    ($fn_name:ident, $dyn_fn:path, $doc_label:literal, real) => {
        #[doc = concat!(
            "Runs eager AD `",
            $doc_label,
            "` on two real-valued dynamic tensors after applying the dynamic result-type promotion rule."
        )]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = "```ignore"]
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
        #[doc = "assert!(out.dims().is_empty());"]
        #[doc = "```"]
        pub fn $fn_name(&self, rhs: &Self) -> Result<Self> {
            let (_, lhs, rhs) = promote_pair_to_common(self, rhs)?;
            Ok($dyn_fn(lhs.as_dyn_ad_ref(), rhs.as_dyn_ad_ref())?.into())
        }
    };
}

impl Tensor {
    define_dyn_unary_method!(acos, ad::acos_dyn, "acos");
    define_dyn_unary_method!(acosh, ad::acosh_dyn, "acosh");
    define_dyn_unary_method!(asin, ad::asin_dyn, "asin");
    define_dyn_unary_method!(asinh, ad::asinh_dyn, "asinh");
    define_dyn_unary_method!(atan, ad::atan_dyn, "atan");
    define_dyn_unary_method!(atanh, ad::atanh_dyn, "atanh");
    define_dyn_unary_method!(cos, ad::cos_dyn, "cos");
    define_dyn_unary_method!(cosh, ad::cosh_dyn, "cosh");
    define_dyn_unary_method!(expm1, ad::expm1_dyn, "expm1");
    define_dyn_unary_method!(log, ad::log_dyn, "log");
    define_dyn_unary_method!(log1p, ad::log1p_dyn, "log1p");
    define_dyn_unary_method!(sin, ad::sin_dyn, "sin");
    define_dyn_unary_method!(sinh, ad::sinh_dyn, "sinh");
    define_dyn_unary_method!(sqrt, ad::sqrt_dyn, "sqrt");
    define_dyn_unary_method!(tanh, ad::tanh_dyn, "tanh");

    define_dyn_reduction_method!(std, ad::std_dyn, "standard deviation", real);
    define_dyn_reduction_method!(var, ad::var_dyn, "variance", real);

    define_dyn_binary_method!(atan2, ad::atan2_dyn, "atan2", real);
    define_dyn_binary_method!(hypot, ad::hypot_dyn, "hypot", real);
    define_dyn_binary_method!(pow, ad::pow_dyn, "pow", generic);

    #[doc = "Runs eager AD `exp` on a dynamic tensor."]
    #[doc = ""]
    #[doc = "# Examples"]
    #[doc = ""]
    #[doc = "```ignore"]
    #[doc = "use tenferro::{set_default_runtime, Tensor, RuntimeContext};"]
    #[doc = "use tenferro_prims::CpuContext;"]
    #[doc = "use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};"]
    #[doc = ""]
    #[doc = "let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));"]
    #[doc = "let x = Tensor::from_tensor("]
    #[doc = "    DenseTensor::<f64>::from_slice(&[0.0], &[], MemoryOrder::ColumnMajor).unwrap(),"]
    #[doc = ");"]
    #[doc = "let y = x.exp().unwrap();"]
    #[doc = "assert!(y.dims().is_empty());"]
    #[doc = "```"]
    pub fn exp(&self) -> Result<Self> {
        Ok(ad::exp_dyn(self.as_dyn_ad_ref())?.into())
    }

    #[doc = "Runs eager AD full `mean` reduction on a dynamic tensor."]
    #[doc = ""]
    #[doc = "# Examples"]
    #[doc = ""]
    #[doc = "```ignore"]
    #[doc = "use tenferro::{set_default_runtime, Tensor, RuntimeContext};"]
    #[doc = "use tenferro_prims::CpuContext;"]
    #[doc = "use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};"]
    #[doc = ""]
    #[doc = "let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));"]
    #[doc = "let x = Tensor::from_tensor("]
    #[doc = "    DenseTensor::<f64>::from_slice(&[1.0, 3.0], &[2], MemoryOrder::ColumnMajor).unwrap(),"]
    #[doc = ");"]
    #[doc = "let y = x.mean().unwrap();"]
    #[doc = "assert!(y.dims().is_empty());"]
    #[doc = "```"]
    pub fn mean(&self) -> Result<Self> {
        Ok(ad::mean_dyn(self.as_dyn_ad_ref())?.into())
    }

    #[doc = "Runs eager AD `add` on two dynamic tensors after applying the dynamic result-type promotion rule."]
    #[doc = ""]
    #[doc = "# Examples"]
    #[doc = ""]
    #[doc = "```ignore"]
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
    #[doc = "let out = lhs.add(&rhs).unwrap();"]
    #[doc = "assert!(out.dims().is_empty());"]
    #[doc = "```"]
    pub fn add(&self, rhs: &Self) -> Result<Self> {
        let (_, lhs, rhs) = promote_pair_to_common(self, rhs)?;
        Ok(ad::add_dyn(lhs.as_dyn_ad_ref(), rhs.as_dyn_ad_ref())?.into())
    }
}
