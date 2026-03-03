//! Context-to-backend bridge trait for tensor linalg operations.
//!
//! [`TensorLinalgContextFor<T>`] binds a context type to its backend,
//! enabling generic `solve(&mut ctx, ...)` calls while preserving the
//! backend-marker pattern from [`TensorLinalgBackend`].

use super::tensor_api::TensorLinalgBackend;
use crate::LinalgScalar;

/// Bridge trait that maps a context type to its backend.
///
/// Each context type implements this trait to declare which backend it
/// belongs to. This keeps public APIs generic over context while
/// preserving the backend-marker pattern.
///
/// Implemented for:
/// - [`tenferro_prims::CpuContext`] → [`CpuTensorLinalgBackend`](super::cpu::CpuTensorLinalgBackend)
/// - [`tenferro_prims::CudaContext`] (future)
/// - [`tenferro_prims::RocmContext`] (future)
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::backend::TensorLinalgContextFor;
///
/// fn do_work<T, C>(ctx: &mut C)
/// where
///     T: tenferro_linalg::LinalgScalar,
///     C: TensorLinalgContextFor<T>,
/// {
///     // dispatch through the backend
/// }
/// ```
pub trait TensorLinalgContextFor<T: LinalgScalar> {
    /// The backend type that this context is associated with.
    type Backend: TensorLinalgBackend<T, Context = Self>;
}
