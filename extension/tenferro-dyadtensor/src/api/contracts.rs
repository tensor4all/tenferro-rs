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
pub trait StandardRuntimeValue: Scalar + HasAlgebra<Algebra = Standard<Self>> + 'static {}

impl<T> StandardRuntimeValue for T where T: Scalar + HasAlgebra<Algebra = Standard<T>> + 'static {}

#[doc(hidden)]
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
pub trait RealLinalgRuntimeValue:
    LinalgRuntimeValue + HasAlgebra<Algebra = Standard<Self>> + LinalgScalar<Real = Self> + Float
{
}

impl<T> RealLinalgRuntimeValue for T where
    T: LinalgRuntimeValue + HasAlgebra<Algebra = Standard<T>> + LinalgScalar<Real = T> + Float
{
}

#[doc(hidden)]
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
