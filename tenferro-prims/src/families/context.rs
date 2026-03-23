use num_complex::ComplexFloat;
use tenferro_algebra::{Algebra, Conjugate, Scalar, Semiring};
use tenferro_tensor::Tensor;

use crate::{
    CpuBackend, CpuContext, CudaBackend, CudaContext, RocmBackend, RocmContext,
    TensorComplexRealPrims, TensorComplexScalePrims, TensorMetadataContextFor, TensorMetadataPrims,
    TensorScalarPrims, TensorSemiringCore,
};

/// Bridge trait that binds a semiring execution context to its backend.
///
/// High-level crates use this trait to stay generic over runtime context types
/// while still dispatching semiring execution through the correct backend
/// marker type.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::{CpuContext, TensorSemiringContextFor};
///
/// fn accepts_context<C>(_: &mut C)
/// where
///     C: TensorSemiringContextFor<tenferro_algebra::Standard<f64>>,
/// {
/// }
///
/// let mut ctx = CpuContext::new(1);
/// accepts_context(&mut ctx);
/// ```
pub trait TensorSemiringContextFor<Alg: Semiring> {
    /// Backend associated with this context for the given algebra family.
    type SemiringBackend: TensorSemiringCore<Alg, Context = Self>;
}

/// Bridge trait that binds a scalar-family execution context to its backend.
///
/// High-level crates use this trait to stay generic over runtime context types
/// while dispatching pointwise and reduction scalar families through the
/// correct backend marker type.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_prims::{CpuContext, TensorScalarContextFor};
///
/// fn accepts_context<C>(_: &mut C)
/// where
///     C: TensorScalarContextFor<Standard<f64>>,
/// {
/// }
///
/// let mut ctx = CpuContext::new(1);
/// accepts_context(&mut ctx);
/// ```
pub trait TensorScalarContextFor<Alg: Algebra> {
    /// Backend associated with this context for the scalar family.
    type ScalarBackend: TensorScalarPrims<Alg, Context = Self>;
}

/// Bridge trait that binds a complex-to-real execution context to its backend.
///
/// High-level crates use this trait to stay generic over runtime context types
/// while dispatching cross-dtype complex-to-real families through the correct
/// backend marker type.
///
/// # Examples
///
/// ```ignore
/// use num_complex::Complex64;
/// use tenferro_prims::{CpuContext, TensorComplexRealContextFor};
///
/// fn accepts_context<C>(_: &mut C)
/// where
///     C: TensorComplexRealContextFor<Complex64>,
/// {
/// }
///
/// let mut ctx = CpuContext::new(1);
/// accepts_context(&mut ctx);
/// ```
pub trait TensorComplexRealContextFor<Input: ComplexFloat + Scalar> {
    /// Backend associated with this context for the complex-to-real family.
    type ComplexRealBackend: TensorComplexRealPrims<Input, Context = Self, Real = Input::Real>;
}

/// Bridge trait that binds a complex-by-real execution context to its backend.
///
/// # Examples
///
/// ```ignore
/// use num_complex::Complex64;
/// use tenferro_prims::{CpuContext, TensorComplexScaleContextFor};
///
/// fn accepts_context<C>(_: &mut C)
/// where
///     C: TensorComplexScaleContextFor<Complex64>,
/// {
/// }
///
/// let mut ctx = CpuContext::new(1);
/// accepts_context(&mut ctx);
/// ```
pub trait TensorComplexScaleContextFor<Input: ComplexFloat + Scalar>
where
    Input::Real: Scalar + Send + Sync,
{
    /// Backend associated with the complex-by-real family.
    type ComplexScaleBackend: TensorComplexScalePrims<Input, Context = Self>;
}

/// Bridge trait for backend-specific lazy-conjugation resolution.
///
/// High-level crates use this trait to stay generic over runtime context types
/// while still materializing unresolved conjugation through the correct backend
/// utility.
///
/// # Examples
///
/// ```ignore
/// use num_complex::Complex64;
/// use tenferro_prims::{CpuContext, TensorResolveConjContextFor};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let base = Tensor::from_slice(
///     &[Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)],
///     &[2],
///     MemoryOrder::ColumnMajor,
/// )
/// .unwrap();
/// let lazy = base.conj();
/// let resolved = <CpuContext as TensorResolveConjContextFor<Complex64>>::resolve_conj(
///     &mut ctx,
///     &lazy,
/// );
/// assert!(!resolved.is_conjugated());
/// ```
pub trait TensorResolveConjContextFor<T: Scalar + Conjugate> {
    /// Materialize a lazily-conjugated tensor using the backend tied to `Self`.
    fn resolve_conj(ctx: &mut Self, src: &Tensor<T>) -> Tensor<T>;
}

impl<Alg> TensorSemiringContextFor<Alg> for CpuContext
where
    Alg: Semiring,
    CpuBackend: TensorSemiringCore<Alg, Context = CpuContext>,
{
    type SemiringBackend = CpuBackend;
}

impl<Alg> TensorScalarContextFor<Alg> for CpuContext
where
    Alg: Algebra,
    CpuBackend: TensorScalarPrims<Alg, Context = CpuContext>,
{
    type ScalarBackend = CpuBackend;
}

impl<Input> TensorComplexRealContextFor<Input> for CpuContext
where
    Input: ComplexFloat + Scalar,
    Input::Real: Scalar,
    CpuBackend: TensorComplexRealPrims<Input, Context = CpuContext, Real = Input::Real>,
{
    type ComplexRealBackend = CpuBackend;
}

impl<T> TensorResolveConjContextFor<T> for CpuContext
where
    T: Scalar + Conjugate,
{
    fn resolve_conj(ctx: &mut Self, src: &Tensor<T>) -> Tensor<T> {
        CpuBackend::resolve_conj(ctx, src)
    }
}

impl TensorMetadataContextFor for CpuContext
where
    CpuBackend: TensorMetadataPrims<Context = CpuContext>,
{
    type MetadataBackend = CpuBackend;
}

impl<Input> TensorComplexScaleContextFor<Input> for CpuContext
where
    Input: ComplexFloat + Scalar,
    Input::Real: Scalar + Send + Sync,
    CpuBackend: TensorComplexScalePrims<Input, Context = CpuContext>,
{
    type ComplexScaleBackend = CpuBackend;
}

impl<Alg> TensorSemiringContextFor<Alg> for CudaContext
where
    Alg: Semiring,
    CudaBackend: TensorSemiringCore<Alg, Context = CudaContext>,
{
    type SemiringBackend = CudaBackend;
}

impl<Alg> TensorScalarContextFor<Alg> for CudaContext
where
    Alg: Algebra,
    CudaBackend: TensorScalarPrims<Alg, Context = CudaContext>,
{
    type ScalarBackend = CudaBackend;
}

impl<Input> TensorComplexRealContextFor<Input> for CudaContext
where
    Input: ComplexFloat + Scalar,
    Input::Real: Scalar,
    CudaBackend: TensorComplexRealPrims<Input, Context = CudaContext, Real = Input::Real>,
{
    type ComplexRealBackend = CudaBackend;
}

impl<T> TensorResolveConjContextFor<T> for CudaContext
where
    T: Scalar + Conjugate + 'static,
{
    fn resolve_conj(ctx: &mut Self, src: &Tensor<T>) -> Tensor<T> {
        CudaBackend::resolve_conj(ctx, src)
    }
}

#[cfg(feature = "cuda")]
impl TensorMetadataContextFor for CudaContext
where
    CudaBackend: TensorMetadataPrims<Context = CudaContext>,
{
    type MetadataBackend = CudaBackend;
}

impl<Input> TensorComplexScaleContextFor<Input> for CudaContext
where
    Input: ComplexFloat + Scalar,
    Input::Real: Scalar + Send + Sync,
    CudaBackend: TensorComplexScalePrims<Input, Context = CudaContext>,
{
    type ComplexScaleBackend = CudaBackend;
}

impl<Alg> TensorSemiringContextFor<Alg> for RocmContext
where
    Alg: Semiring,
    RocmBackend: TensorSemiringCore<Alg, Context = RocmContext>,
{
    type SemiringBackend = RocmBackend;
}

impl<Alg> TensorScalarContextFor<Alg> for RocmContext
where
    Alg: Algebra,
    RocmBackend: TensorScalarPrims<Alg, Context = RocmContext>,
{
    type ScalarBackend = RocmBackend;
}

impl<Input> TensorComplexRealContextFor<Input> for RocmContext
where
    Input: ComplexFloat + Scalar,
    Input::Real: Scalar,
    RocmBackend: TensorComplexRealPrims<Input, Context = RocmContext, Real = Input::Real>,
{
    type ComplexRealBackend = RocmBackend;
}

impl<T> TensorResolveConjContextFor<T> for RocmContext
where
    T: Scalar + Conjugate,
{
    fn resolve_conj(ctx: &mut Self, src: &Tensor<T>) -> Tensor<T> {
        RocmBackend::resolve_conj(ctx, src)
    }
}

impl<Input> TensorComplexScaleContextFor<Input> for RocmContext
where
    Input: ComplexFloat + Scalar,
    Input::Real: Scalar + Send + Sync,
    RocmBackend: TensorComplexScalePrims<Input, Context = RocmContext>,
{
    type ComplexScaleBackend = RocmBackend;
}
