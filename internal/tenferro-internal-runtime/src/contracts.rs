//! Hidden runtime contract aliases shared by tenferro frontend crates.

use chainrules::ScalarAd;
use num_complex::Complex;
use num_traits::Float;
use tenferro_algebra::{Conjugate, HasAlgebra, Scalar, Standard};
use tenferro_internal_frontend_core::DynTensorTyped;
use tenferro_linalg::backend::CudaLinalgScalar;
use tenferro_linalg::{KernelLinalgScalar, LinalgScalar};
use tenferro_tensor::KeepCountScalar;

#[doc(hidden)]
pub trait StandardRuntimeValue:
    Scalar + DynTensorTyped + HasAlgebra<Algebra = Standard<Self>> + 'static
{
}

impl<T> StandardRuntimeValue for T where
    T: Scalar + DynTensorTyped + HasAlgebra<Algebra = Standard<T>> + 'static
{
}

#[doc(hidden)]
pub trait EinsumRuntimeValue: StandardRuntimeValue + Conjugate {}

impl<T> EinsumRuntimeValue for T where T: StandardRuntimeValue + Conjugate {}

#[doc(hidden)]
pub trait ScalarRuntimeValue: StandardRuntimeValue + Copy {}

impl<T> ScalarRuntimeValue for T where T: StandardRuntimeValue + Copy {}

#[doc(hidden)]
pub trait AnalyticRuntimeValue: ScalarRuntimeValue {}

impl<T> AnalyticRuntimeValue for T where T: ScalarRuntimeValue {}

#[doc(hidden)]
pub trait ScalarAnalyticRuntimeValue:
    EinsumRuntimeValue + ScalarRuntimeValue + AnalyticRuntimeValue
{
}

impl<T> ScalarAnalyticRuntimeValue for T where
    T: EinsumRuntimeValue + ScalarRuntimeValue + AnalyticRuntimeValue
{
}

#[doc(hidden)]
pub trait GenericAdRuntimeValue: ScalarAnalyticRuntimeValue + ScalarAd {}

impl<T> GenericAdRuntimeValue for T where T: ScalarAnalyticRuntimeValue + ScalarAd {}

#[doc(hidden)]
pub trait RealAdRuntimeValue: GenericAdRuntimeValue + ScalarAd<Real = Self> + Float {}

impl<T> RealAdRuntimeValue for T where T: GenericAdRuntimeValue + ScalarAd<Real = T> + Float {}

#[doc(hidden)]
pub trait LinalgRuntimeValue:
    EinsumRuntimeValue + KernelLinalgScalar + CudaLinalgScalar + 'static
{
}

impl<T> LinalgRuntimeValue for T where
    T: EinsumRuntimeValue + KernelLinalgScalar + CudaLinalgScalar + 'static
{
}

#[doc(hidden)]
pub trait RealLinalgRuntimeValue:
    LinalgRuntimeValue
    + HasAlgebra<Algebra = Standard<Self>>
    + LinalgScalar<Real = Self>
    + Float
    + KeepCountScalar
{
}

impl<T> RealLinalgRuntimeValue for T where
    T: LinalgRuntimeValue
        + HasAlgebra<Algebra = Standard<T>>
        + LinalgScalar<Real = T>
        + Float
        + KeepCountScalar
{
}

#[doc(hidden)]
pub trait ComplexLinalgRuntimeValue:
    RealLinalgRuntimeValue + LinalgScalar<Complex = Complex<Self>>
where
    Complex<Self>: Scalar + DynTensorTyped,
{
}

impl<T> ComplexLinalgRuntimeValue for T
where
    T: RealLinalgRuntimeValue + LinalgScalar<Complex = Complex<T>>,
    Complex<T>: Scalar + DynTensorTyped,
{
}
