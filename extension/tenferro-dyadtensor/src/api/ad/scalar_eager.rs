use super::*;

macro_rules! define_scalar_unary_eager_ad_fn {
    ($fn_name:ident, $builder_fn:ident, $doc_op:literal, generic) => {
        #[doc = concat!("Eager AD `", $doc_op, "`.")]
        #[doc = ""]
        #[doc = concat!("Equivalent to `crate::", stringify!($builder_fn), "(...).run()`.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro_dyadtensor::ad::", stringify!($fn_name), "(&x)?;\n```")]
        pub fn $fn_name<T>(tensor: &AdTensor<T>) -> Result<AdTensor<T>>
        where
            T: Scalar + HasAlgebra<Algebra = Standard<T>> + chainrules_scalarops::ScalarAd + Copy + 'static,
            CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
            CpuBackend: tenferro_prims::TensorScalarPrims<Standard<T>, Context = CpuContext>,
            CpuBackend: tenferro_prims::TensorAnalyticPrims<Standard<T>, Context = CpuContext>,
            tenferro_prims::CudaBackend:
                tenferro_prims::TensorScalarPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
            tenferro_prims::CudaBackend:
                tenferro_prims::TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
            tenferro_prims::RocmBackend:
                tenferro_prims::TensorScalarPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
            tenferro_prims::RocmBackend:
                tenferro_prims::TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
        {
            super::super::$builder_fn(tensor).run()
        }
    };
    ($fn_name:ident, $builder_fn:ident, $doc_op:literal, real) => {
        #[doc = concat!("Eager AD `", $doc_op, "`.")]
        #[doc = ""]
        #[doc = concat!("Equivalent to `crate::", stringify!($builder_fn), "(...).run()`.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro_dyadtensor::ad::", stringify!($fn_name), "(&x)?;\n```")]
        pub fn $fn_name<T>(tensor: &AdTensor<T>) -> Result<AdTensor<T>>
        where
            T: Scalar
                + HasAlgebra<Algebra = Standard<T>>
                + chainrules_scalarops::ScalarAd<Real = T>
                + Float
                + Copy
                + 'static,
            CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
            CpuBackend: tenferro_prims::TensorScalarPrims<Standard<T>, Context = CpuContext>,
            CpuBackend: tenferro_prims::TensorAnalyticPrims<Standard<T>, Context = CpuContext>,
            tenferro_prims::CudaBackend:
                tenferro_prims::TensorScalarPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
            tenferro_prims::CudaBackend:
                tenferro_prims::TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
            tenferro_prims::RocmBackend:
                tenferro_prims::TensorScalarPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
            tenferro_prims::RocmBackend:
                tenferro_prims::TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
        {
            super::super::$builder_fn(tensor).run()
        }
    };
}

macro_rules! define_scalar_binary_eager_ad_fn {
    ($fn_name:ident, $builder_fn:ident, $doc_op:literal, generic) => {
        #[doc = concat!("Eager AD `", $doc_op, "`.")]
        #[doc = ""]
        #[doc = concat!("Equivalent to `crate::", stringify!($builder_fn), "(...).run()`.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro_dyadtensor::ad::", stringify!($fn_name), "(&a, &b)?;\n```")]
        pub fn $fn_name<T>(lhs: &AdTensor<T>, rhs: &AdTensor<T>) -> Result<AdTensor<T>>
        where
            T: Scalar + HasAlgebra<Algebra = Standard<T>> + chainrules_scalarops::ScalarAd + Copy + 'static,
            CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
            CpuBackend: tenferro_prims::TensorScalarPrims<Standard<T>, Context = CpuContext>,
            CpuBackend: tenferro_prims::TensorAnalyticPrims<Standard<T>, Context = CpuContext>,
            tenferro_prims::CudaBackend:
                tenferro_prims::TensorScalarPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
            tenferro_prims::CudaBackend:
                tenferro_prims::TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
            tenferro_prims::RocmBackend:
                tenferro_prims::TensorScalarPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
            tenferro_prims::RocmBackend:
                tenferro_prims::TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
        {
            super::super::$builder_fn(lhs, rhs).run()
        }
    };
    ($fn_name:ident, $builder_fn:ident, $doc_op:literal, real) => {
        #[doc = concat!("Eager AD `", $doc_op, "`.")]
        #[doc = ""]
        #[doc = concat!("Equivalent to `crate::", stringify!($builder_fn), "(...).run()`.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro_dyadtensor::ad::", stringify!($fn_name), "(&a, &b)?;\n```")]
        pub fn $fn_name<T>(lhs: &AdTensor<T>, rhs: &AdTensor<T>) -> Result<AdTensor<T>>
        where
            T: Scalar
                + HasAlgebra<Algebra = Standard<T>>
                + chainrules_scalarops::ScalarAd<Real = T>
                + Float
                + Copy
                + 'static,
            CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
            CpuBackend: tenferro_prims::TensorScalarPrims<Standard<T>, Context = CpuContext>,
            CpuBackend: tenferro_prims::TensorAnalyticPrims<Standard<T>, Context = CpuContext>,
            tenferro_prims::CudaBackend:
                tenferro_prims::TensorScalarPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
            tenferro_prims::CudaBackend:
                tenferro_prims::TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
            tenferro_prims::RocmBackend:
                tenferro_prims::TensorScalarPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
            tenferro_prims::RocmBackend:
                tenferro_prims::TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
        {
            super::super::$builder_fn(lhs, rhs).run()
        }
    };
}

macro_rules! define_scalar_reduction_eager_ad_fn {
    ($fn_name:ident, $builder_fn:ident, $doc_label:literal, generic) => {
        #[doc = concat!("Eager AD full `", $doc_label, "` reduction.")]
        #[doc = ""]
        #[doc = concat!("Equivalent to `crate::", stringify!($builder_fn), "(...).run()`.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro_dyadtensor::ad::", stringify!($fn_name), "(&x)?;\n```")]
        pub fn $fn_name<T>(tensor: &AdTensor<T>) -> Result<AdTensor<T>>
        where
            T: Scalar + HasAlgebra<Algebra = Standard<T>> + chainrules_scalarops::ScalarAd + Copy + 'static,
            CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
            CpuBackend: tenferro_prims::TensorScalarPrims<Standard<T>, Context = CpuContext>,
            CpuBackend: tenferro_prims::TensorAnalyticPrims<Standard<T>, Context = CpuContext>,
            tenferro_prims::CudaBackend:
                tenferro_prims::TensorScalarPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
            tenferro_prims::CudaBackend:
                tenferro_prims::TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
            tenferro_prims::RocmBackend:
                tenferro_prims::TensorScalarPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
            tenferro_prims::RocmBackend:
                tenferro_prims::TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
        {
            super::super::$builder_fn(tensor).run()
        }
    };
    ($fn_name:ident, $builder_fn:ident, $doc_label:literal, real) => {
        #[doc = concat!("Eager AD full `", $doc_label, "` reduction.")]
        #[doc = ""]
        #[doc = concat!("Equivalent to `crate::", stringify!($builder_fn), "(...).run()`.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro_dyadtensor::ad::", stringify!($fn_name), "(&x)?;\n```")]
        pub fn $fn_name<T>(tensor: &AdTensor<T>) -> Result<AdTensor<T>>
        where
            T: Scalar
                + HasAlgebra<Algebra = Standard<T>>
                + chainrules_scalarops::ScalarAd<Real = T>
                + Float
                + Copy
                + 'static,
            CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
            CpuBackend: tenferro_prims::TensorScalarPrims<Standard<T>, Context = CpuContext>,
            CpuBackend: tenferro_prims::TensorAnalyticPrims<Standard<T>, Context = CpuContext>,
            tenferro_prims::CudaBackend:
                tenferro_prims::TensorScalarPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
            tenferro_prims::CudaBackend:
                tenferro_prims::TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
            tenferro_prims::RocmBackend:
                tenferro_prims::TensorScalarPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
            tenferro_prims::RocmBackend:
                tenferro_prims::TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
        {
            super::super::$builder_fn(tensor).run()
        }
    };
}

define_scalar_unary_eager_ad_fn!(sqrt, sqrt_ad, "sqrt", generic);
define_scalar_unary_eager_ad_fn!(exp, exp_ad, "exp", generic);
define_scalar_unary_eager_ad_fn!(expm1, expm1_ad, "expm1", generic);
define_scalar_unary_eager_ad_fn!(log, log_ad, "log", generic);
define_scalar_unary_eager_ad_fn!(log1p, log1p_ad, "log1p", generic);
define_scalar_unary_eager_ad_fn!(sin, sin_ad, "sin", generic);
define_scalar_unary_eager_ad_fn!(cos, cos_ad, "cos", generic);
define_scalar_unary_eager_ad_fn!(tanh, tanh_ad, "tanh", generic);
define_scalar_unary_eager_ad_fn!(asin, asin_ad, "asin", generic);
define_scalar_unary_eager_ad_fn!(acos, acos_ad, "acos", generic);
define_scalar_unary_eager_ad_fn!(atan, atan_ad, "atan", generic);
define_scalar_unary_eager_ad_fn!(sinh, sinh_ad, "sinh", generic);
define_scalar_unary_eager_ad_fn!(cosh, cosh_ad, "cosh", generic);
define_scalar_unary_eager_ad_fn!(asinh, asinh_ad, "asinh", generic);
define_scalar_unary_eager_ad_fn!(acosh, acosh_ad, "acosh", generic);
define_scalar_unary_eager_ad_fn!(atanh, atanh_ad, "atanh", generic);

define_scalar_binary_eager_ad_fn!(add, add_ad, "add", generic);
define_scalar_binary_eager_ad_fn!(atan2, atan2_ad, "atan2", real);
define_scalar_binary_eager_ad_fn!(pow, pow_ad, "pow", generic);
define_scalar_binary_eager_ad_fn!(hypot, hypot_ad, "hypot", real);

define_scalar_reduction_eager_ad_fn!(mean, mean_ad, "mean", generic);
define_scalar_reduction_eager_ad_fn!(var, var_ad, "variance", real);
define_scalar_reduction_eager_ad_fn!(std, std_ad, "standard deviation", real);
