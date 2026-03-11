//! Hidden runtime contract aliases for the public dyadtensor API surface.

use chainrules_scalarops::ScalarAd;
use num_complex::Complex;
use num_traits::Float;
use tenferro_algebra::{HasAlgebra, Scalar, Standard};
use tenferro_linalg::backend::{TensorLinalgBackend, TensorLinalgContextFor};
use tenferro_linalg::{backend::CpuLinalgScalar, LinalgScalar};
use tenferro_prims::{
    CpuBackend, CpuContext, CudaBackend, CudaContext, RocmBackend, RocmContext,
    TensorAnalyticPrims, TensorPrims, TensorScalarPrims,
};

#[doc(hidden)]
/// Hidden bound for values that participate in the standard dyadtensor runtime.
///
/// # Examples
///
/// ```ignore
/// fn require_standard<T: tenferro_dyadtensor::api::contracts::StandardRuntimeValue>() {}
/// require_standard::<f64>();
/// ```
pub trait StandardRuntimeValue: Scalar + HasAlgebra<Algebra = Standard<Self>> + 'static {}

impl<T> StandardRuntimeValue for T where T: Scalar + HasAlgebra<Algebra = Standard<T>> + 'static {}

#[doc(hidden)]
/// Hidden bound for values that can run through the einsum runtime dispatch layer.
///
/// # Examples
///
/// ```ignore
/// fn require_einsum<T: tenferro_dyadtensor::api::contracts::EinsumRuntimeValue>() {}
/// require_einsum::<f64>();
/// ```
pub trait EinsumRuntimeValue: StandardRuntimeValue {}

impl<T> EinsumRuntimeValue for T
where
    T: StandardRuntimeValue,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
    CudaBackend: TensorPrims<Standard<T>, Context = CudaContext>,
    RocmBackend: TensorPrims<Standard<T>, Context = RocmContext>,
{
}

#[doc(hidden)]
/// Hidden bound for values that support scalar pointwise and reduction primitives.
///
/// # Examples
///
/// ```ignore
/// fn require_scalar<T: tenferro_dyadtensor::api::contracts::ScalarRuntimeValue>() {}
/// require_scalar::<f64>();
/// ```
pub trait ScalarRuntimeValue: StandardRuntimeValue + Copy {}

impl<T> ScalarRuntimeValue for T
where
    T: StandardRuntimeValue + Copy,
    CpuBackend: TensorScalarPrims<Standard<T>, Context = CpuContext>,
    CudaBackend: TensorScalarPrims<Standard<T>, Context = CudaContext>,
    RocmBackend: TensorScalarPrims<Standard<T>, Context = RocmContext>,
{
}

#[doc(hidden)]
/// Hidden bound for values that support analytic pointwise and reduction primitives.
///
/// # Examples
///
/// ```ignore
/// fn require_analytic<T: tenferro_dyadtensor::api::contracts::AnalyticRuntimeValue>() {}
/// require_analytic::<f64>();
/// ```
pub trait AnalyticRuntimeValue: ScalarRuntimeValue {}

impl<T> AnalyticRuntimeValue for T
where
    T: ScalarRuntimeValue,
    CpuBackend: TensorAnalyticPrims<Standard<T>, Context = CpuContext>,
    CudaBackend: TensorAnalyticPrims<Standard<T>, Context = CudaContext>,
    RocmBackend: TensorAnalyticPrims<Standard<T>, Context = RocmContext>,
{
}

#[doc(hidden)]
/// Hidden shorthand for values that support einsum, scalar, and analytic runtime families.
///
/// # Examples
///
/// ```ignore
/// fn require_all<T: tenferro_dyadtensor::api::contracts::ScalarAnalyticRuntimeValue>() {}
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
/// fn require_generic_ad<T: tenferro_dyadtensor::api::contracts::GenericAdRuntimeValue>() {}
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
/// fn require_real_ad<T: tenferro_dyadtensor::api::contracts::RealAdRuntimeValue>() {}
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
/// fn require_linalg<T: tenferro_dyadtensor::api::contracts::LinalgRuntimeValue>() {}
/// require_linalg::<f64>();
/// ```
pub trait LinalgRuntimeValue:
    EinsumRuntimeValue + LinalgScalar + CpuLinalgScalar + 'static
{
}

impl<T> LinalgRuntimeValue for T
where
    T: EinsumRuntimeValue + LinalgScalar + CpuLinalgScalar + 'static,
    CpuContext: TensorLinalgContextFor<T>,
    CudaContext: TensorLinalgContextFor<T>,
    RocmContext: TensorLinalgContextFor<T>,
    <CpuContext as TensorLinalgContextFor<T>>::Backend:
        TensorLinalgBackend<T, Context = CpuContext>,
    <CudaContext as TensorLinalgContextFor<T>>::Backend:
        TensorLinalgBackend<T, Context = CudaContext>,
    <RocmContext as TensorLinalgContextFor<T>>::Backend:
        TensorLinalgBackend<T, Context = RocmContext>,
{
}

#[doc(hidden)]
/// Hidden shorthand for real-valued linalg runtime values.
///
/// # Examples
///
/// ```ignore
/// fn require_real_linalg<T: tenferro_dyadtensor::api::contracts::RealLinalgRuntimeValue>() {}
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
/// fn require_complex_linalg<T: tenferro_dyadtensor::api::contracts::ComplexLinalgRuntimeValue>() {}
/// require_complex_linalg::<f64>();
/// ```
pub trait ComplexLinalgRuntimeValue:
    RealLinalgRuntimeValue + LinalgScalar<Complex = Complex<Self>>
where
    Complex<Self>: Scalar,
{
}

impl<T> ComplexLinalgRuntimeValue for T
where
    T: RealLinalgRuntimeValue + LinalgScalar<Complex = Complex<T>>,
    Complex<T>: Scalar,
{
}
