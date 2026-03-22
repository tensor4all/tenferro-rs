use tenferro_algebra::{Algebra, Semiring};

use crate::{
    CpuBackend, CpuContext, CudaBackend, CudaContext, RocmBackend, RocmContext, TensorScalarPrims,
    TensorSemiringCore,
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
