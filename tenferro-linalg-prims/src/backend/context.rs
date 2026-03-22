use tenferro_algebra::Standard;
use tenferro_prims::TensorSemiringContextFor;

use crate::{KernelLinalgScalar, TensorLinalgPrims};

/// Maps a context type to its tensor-level linalg backend.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg_prims::backend::TensorLinalgContextFor;
///
/// fn accepts_ctx<T, C>(_ctx: &mut C)
/// where
///     T: tenferro_linalg_prims::KernelLinalgScalar,
///     C: TensorLinalgContextFor<T>,
/// {
/// }
/// ```
pub trait TensorLinalgContextFor<T: KernelLinalgScalar>:
    TensorSemiringContextFor<Standard<T>>
{
    /// The backend associated with this context type.
    type Backend: TensorLinalgPrims<T, Context = Self>;
}
