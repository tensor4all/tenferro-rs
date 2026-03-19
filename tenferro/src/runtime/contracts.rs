//! Hidden runtime contract aliases for the public tenferro frontend surface.

use chainrules::ScalarAd;
use num_complex::Complex;
use num_traits::Float;
use tenferro_algebra::{Conjugate, HasAlgebra, Scalar, Standard};
use tenferro_linalg::{KernelLinalgScalar, LinalgScalar};

use crate::core::DynTensorTyped;

#[doc(hidden)]
/// Hidden bound for values that participate in the standard tenferro runtime.
///
/// # Examples
///
/// ```ignore
/// fn require_standard<T: tenferro::runtime::contracts::StandardRuntimeValue>() {}
/// require_standard::<f64>();
/// ```
pub trait StandardRuntimeValue:
    Scalar + DynTensorTyped + HasAlgebra<Algebra = Standard<Self>> + 'static
{
}

impl<T> StandardRuntimeValue for T where
    T: Scalar + DynTensorTyped + HasAlgebra<Algebra = Standard<T>> + 'static
{
}

#[doc(hidden)]
/// Hidden bound for values that can run through the einsum runtime dispatch layer.
///
/// # Examples
///
/// ```ignore
/// fn require_einsum<T: tenferro::runtime::contracts::EinsumRuntimeValue>() {}
/// require_einsum::<f64>();
/// ```
pub trait EinsumRuntimeValue: StandardRuntimeValue + Conjugate {}

impl<T> EinsumRuntimeValue for T where T: StandardRuntimeValue + Conjugate {}

#[doc(hidden)]
/// Hidden bound for values that support scalar pointwise and reduction primitives.
///
/// # Examples
///
/// ```ignore
/// fn require_scalar<T: tenferro::runtime::contracts::ScalarRuntimeValue>() {}
/// require_scalar::<f64>();
/// ```
pub trait ScalarRuntimeValue: StandardRuntimeValue + Copy {}

impl<T> ScalarRuntimeValue for T where T: StandardRuntimeValue + Copy {}

#[doc(hidden)]
/// Hidden bound for values that support analytic pointwise and reduction primitives.
///
/// # Examples
///
/// ```ignore
/// fn require_analytic<T: tenferro::runtime::contracts::AnalyticRuntimeValue>() {}
/// require_analytic::<f64>();
/// ```
pub trait AnalyticRuntimeValue: ScalarRuntimeValue {}

impl<T> AnalyticRuntimeValue for T where T: ScalarRuntimeValue {}

#[doc(hidden)]
/// Hidden shorthand for values that support einsum, scalar, and analytic runtime families.
///
/// # Examples
///
/// ```ignore
/// fn require_all<T: tenferro::runtime::contracts::ScalarAnalyticRuntimeValue>() {}
/// require_all::<f64>();
/// ```
pub trait ScalarAnalyticRuntimeValue:
    EinsumRuntimeValue + ScalarRuntimeValue + AnalyticRuntimeValue
{
}

impl<T> ScalarAnalyticRuntimeValue for T where
    T: EinsumRuntimeValue + ScalarRuntimeValue + AnalyticRuntimeValue
{
}

#[doc(hidden)]
/// Hidden shorthand for scalar/analytic runtime values with scalar AD formulas attached.
///
/// # Examples
///
/// ```ignore
/// fn require_generic_ad<T: tenferro::runtime::contracts::GenericAdRuntimeValue>() {}
/// require_generic_ad::<f64>();
/// ```
pub trait GenericAdRuntimeValue: ScalarAnalyticRuntimeValue + ScalarAd {}

impl<T> GenericAdRuntimeValue for T where T: ScalarAnalyticRuntimeValue + ScalarAd {}

#[doc(hidden)]
/// Hidden shorthand for real-valued generic AD runtime values.
///
/// # Examples
///
/// ```ignore
/// fn require_real_ad<T: tenferro::runtime::contracts::RealAdRuntimeValue>() {}
/// require_real_ad::<f64>();
/// ```
pub trait RealAdRuntimeValue: GenericAdRuntimeValue + ScalarAd<Real = Self> + Float {}

impl<T> RealAdRuntimeValue for T where T: GenericAdRuntimeValue + ScalarAd<Real = T> + Float {}

#[doc(hidden)]
/// Hidden bound for values that can execute linalg families through the runtime dispatch layer.
///
/// # Examples
///
/// ```ignore
/// fn require_linalg<T: tenferro::runtime::contracts::LinalgRuntimeValue>() {}
/// require_linalg::<f64>();
/// ```
pub trait LinalgRuntimeValue: EinsumRuntimeValue + KernelLinalgScalar + 'static {}

impl<T> LinalgRuntimeValue for T where T: EinsumRuntimeValue + KernelLinalgScalar + 'static {}

#[doc(hidden)]
/// Hidden shorthand for real-valued linalg runtime values.
///
/// # Examples
///
/// ```ignore
/// fn require_real_linalg<T: tenferro::runtime::contracts::RealLinalgRuntimeValue>() {}
/// require_real_linalg::<f64>();
/// ```
pub trait RealLinalgRuntimeValue:
    LinalgRuntimeValue + HasAlgebra<Algebra = Standard<Self>> + LinalgScalar<Real = Self> + Float
{
}

impl<T> RealLinalgRuntimeValue for T where
    T: LinalgRuntimeValue + HasAlgebra<Algebra = Standard<T>> + LinalgScalar<Real = T> + Float
{
}

#[doc(hidden)]
/// Hidden shorthand for complex-capable linalg runtime values.
///
/// # Examples
///
/// ```ignore
/// fn require_complex_linalg<T: tenferro::runtime::contracts::ComplexLinalgRuntimeValue>() {}
/// require_complex_linalg::<f64>();
/// ```
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
