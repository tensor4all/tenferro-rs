use super::super::super::*;
use tenferro_internal_ad_core::DynAdTensorTyped;
use tenferro_linalg::NormKind;

macro_rules! define_stub_unary_ad_builder {
    ($builder:ident, $ctor:ident) => {
        pub struct $builder<'a, T: Scalar> {
            tensor: &'a AdTensor<T>,
        }
        impl<'a, T: Scalar + DynAdTensorTyped> $builder<'a, T> {
            pub fn run(self) -> Result<AdTensor<T>> {
                Err(Error::InvalidAdTensor {
                    message: concat!(stringify!($ctor), " is not yet implemented").to_string(),
                })
            }
        }
        pub fn $ctor<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> $builder<'a, T> {
            $builder { tensor }
        }
    };
}

macro_rules! define_stub_binary_ad_builder {
    ($builder:ident, $ctor:ident) => {
        pub struct $builder<'a, T: Scalar> {
            a: &'a AdTensor<T>,
            b: &'a AdTensor<T>,
        }
        impl<'a, T: Scalar + DynAdTensorTyped> $builder<'a, T> {
            pub fn run(self) -> Result<AdTensor<T>> {
                Err(Error::InvalidAdTensor {
                    message: concat!(stringify!($ctor), " is not yet implemented").to_string(),
                })
            }
        }
        pub fn $ctor<'a, T: Scalar>(a: &'a AdTensor<T>, b: &'a AdTensor<T>) -> $builder<'a, T> {
            $builder { a, b }
        }
    };
}

define_stub_binary_ad_builder!(SolveAdBuilder, solve_ad);
define_stub_unary_ad_builder!(PinvAdBuilder, pinv_ad);
define_stub_unary_ad_builder!(MatrixExpAdBuilder, matrix_exp_ad);
define_stub_binary_ad_builder!(SolveTriangularAdBuilder, solve_triangular_ad);

pub struct NormAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T: Scalar> NormAdBuilder<'a, T> {
    pub fn kind(self, _norm_kind: NormKind) -> Self {
        self
    }

    pub fn run(self) -> Result<AdTensor<T>> {
        Err(Error::InvalidAdTensor {
            message: "norm_ad is not yet implemented".to_string(),
        })
    }
}

pub fn norm_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> NormAdBuilder<'a, T> {
    NormAdBuilder { tensor }
}
