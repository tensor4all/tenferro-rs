macro_rules! define_binary_ad_builder {
    ($builder:ident, $ctor:ident, $doc_op:literal, $dtype:path, |$builder_var:ident| $body:expr) => {
        pub struct $builder<'a, T: tenferro_algebra::Scalar> {
            lhs: &'a crate::AdTensor<T>,
            rhs: &'a crate::AdTensor<T>,
        }
        impl<'a, T: tenferro_algebra::Scalar> $builder<'a, T> {
            pub fn run(self) -> crate::Result<crate::AdTensor<T>> {
                unimplemented!(concat!(stringify!($ctor), " not yet implemented"))
            }
        }
        pub fn $ctor<'a, T: tenferro_algebra::Scalar>(
            lhs: &'a crate::AdTensor<T>,
            rhs: &'a crate::AdTensor<T>,
        ) -> $builder<'a, T> {
            $builder { lhs, rhs }
        }
    };
}

macro_rules! define_unary_ad_builder {
    ($builder:ident, $ctor:ident, $doc_op:literal, $dtype:path, |$builder_var:ident| $body:expr) => {
        pub struct $builder<'a, T: tenferro_algebra::Scalar> {
            tensor: &'a crate::AdTensor<T>,
        }
        impl<'a, T: tenferro_algebra::Scalar> $builder<'a, T> {
            pub fn run(self) -> crate::Result<crate::AdTensor<T>> {
                unimplemented!(concat!(stringify!($ctor), " not yet implemented"))
            }
        }
        pub fn $ctor<'a, T: tenferro_algebra::Scalar>(
            tensor: &'a crate::AdTensor<T>,
        ) -> $builder<'a, T> {
            $builder { tensor }
        }
    };
}

pub(crate) use define_binary_ad_builder;
pub(crate) use define_unary_ad_builder;
