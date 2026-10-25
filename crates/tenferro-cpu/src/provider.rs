//! Host-tensor helpers shared with the lower numerical libraries.
//!
//! tenferro owns tensor semantics, placement and resource lifetime; cpueinsum,
//! tcontract and tlinalg own numerical execution, lane selection and
//! scheduling. The two helpers here are the host-side steps a provider call
//! needs around its numerical work: materializing a strided read and reshaping a
//! compact tensor. Both take the operation's one [`cpueinsum::Exec`] bundle (the
//! provider token, see [`crate::CpuEngine::exec`]) so any copy they make runs on
//! the placement that operation was granted, never on an ambient pool.

use cpueinsum::Exec;
use tenferro_tensor::{Tensor, TensorRead};

use crate::buffer_pool::BufferPool;

/// Materialize a borrowed tensor view for one scoped operation and reclaim
/// its temporary host buffer before returning.
///
/// Owned tensor inputs are borrowed directly. View inputs are materialized
/// from `buffers`, passed to `operation`, and returned to the same pool on
/// both success and ordinary error.
///
/// # Errors
///
/// Returns [`tenferro_tensor::Error::RuntimeState`] when the view is not
/// accessible from CPU host memory, propagates typed view-materialization
/// errors, and otherwise returns the error produced by `operation`.
#[doc(hidden)]
pub fn with_materialized_tensor_read<R>(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    op: &'static str,
    input: TensorRead<'_>,
    operation: impl FnOnce(&Tensor, &mut BufferPool, &mut Exec<'_>) -> tenferro_tensor::Result<R>,
) -> tenferro_tensor::Result<R> {
    match input {
        TensorRead::Tensor(tensor) => operation(tensor, buffers, exec),
        TensorRead::View(view) => {
            let materialized =
                crate::materialize_tensor_read(buffers, op, TensorRead::View(view), exec.token())?;
            let result = operation(&materialized, buffers, exec);
            crate::backend::reclaim_tensor(buffers, materialized);
            result
        }
    }
}

/// Reshape a compact tensor on the placement of `exec`.
///
/// Small inputs and a serial `exec` keep the serial host copy. A large input
/// under a parallel `exec` materializes through the same pooled strided copy the
/// session's `reshape_read` uses, sharing `structural::reshape_uses_pool` and
/// its threshold rather than a second copy of either. The pooled branch takes a
/// buffer from the caller's pool.
///
/// # Errors
///
/// Returns [`tenferro_tensor::Error::Validation`] when the input and output
/// shapes have different element counts or the requested layout is invalid.
#[doc(hidden)]
pub fn reshape_tensor(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
    shape: &[usize],
) -> tenferro_tensor::Result<Tensor> {
    if crate::structural::reshape_uses_pool(input, !exec.token().is_serial())? {
        return crate::structural::reshape_read_with_pool(
            buffers,
            TensorRead::from_tensor(input),
            shape,
            exec.token(),
        );
    }
    crate::structural::reshape(input, shape)
}
