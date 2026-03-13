use tenferro_algebra::Semiring;

use crate::{
    CpuBackend, CpuContext, CudaBackend, CudaContext, RocmBackend, RocmContext, TensorSemiringCore,
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

impl<Alg> TensorSemiringContextFor<Alg> for CpuContext
where
    Alg: Semiring,
    CpuBackend: TensorSemiringCore<Alg, Context = CpuContext>,
{
    type SemiringBackend = CpuBackend;
}

impl<Alg> TensorSemiringContextFor<Alg> for CudaContext
where
    Alg: Semiring,
    CudaBackend: TensorSemiringCore<Alg, Context = CudaContext>,
{
    type SemiringBackend = CudaBackend;
}

impl<Alg> TensorSemiringContextFor<Alg> for RocmContext
where
    Alg: Semiring,
    RocmBackend: TensorSemiringCore<Alg, Context = RocmContext>,
{
    type SemiringBackend = RocmBackend;
}
