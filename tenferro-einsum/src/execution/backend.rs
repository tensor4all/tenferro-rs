use tenferro_algebra::Semiring;
use tenferro_prims::{TensorSemiringCore, TensorSemiringFastPath};

/// Context type used by public einsum APIs for a backend/algebra pair.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_einsum::BackendContext;
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx: BackendContext<Standard<f64>, CpuBackend> = CpuContext::new(1);
/// let _ = &mut ctx;
/// ```
pub type BackendContext<Alg, Backend> = <Backend as TensorSemiringCore<Alg>>::Context;
pub(crate) type BackendPlan<Alg, Backend> = <Backend as TensorSemiringCore<Alg>>::Plan;

/// Public backend contract accepted by `tenferro-einsum`.
///
/// Backends must provide the semiring core operations needed for all einsum
/// execution paths, plus the optional semiring fast paths used for optimized
/// contraction and elementwise multiplication.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_einsum::EinsumBackend;
/// use tenferro_prims::CpuBackend;
///
/// fn assert_backend<B: EinsumBackend<Standard<f64>>>() {}
/// assert_backend::<CpuBackend>();
/// ```
pub trait EinsumBackend<Alg: Semiring>:
    TensorSemiringCore<Alg>
    + TensorSemiringFastPath<
        Alg,
        Plan = <Self as TensorSemiringCore<Alg>>::Plan,
        Context = <Self as TensorSemiringCore<Alg>>::Context,
    >
{
}

impl<Alg, Backend> EinsumBackend<Alg> for Backend
where
    Alg: Semiring,
    Backend: TensorSemiringCore<Alg>
        + TensorSemiringFastPath<
            Alg,
            Plan = <Backend as TensorSemiringCore<Alg>>::Plan,
            Context = <Backend as TensorSemiringCore<Alg>>::Context,
        >,
{
}
