//! Pending upload/download handles: submission, readiness, waiting and the drop,
//! forget and rejection semantics.

use std::sync::Arc;

use tenferro_tensor::Tensor;

use super::pinned::PinnedHostBuffer;
use super::runtime::{CudaRuntime, CudaRuntimeState};
use super::storage::CudaStorage;

/// A submission refused before admission, with the submitted package returned.
#[derive(Debug)]
#[doc(hidden)]
pub struct RejectedTransfer {
    /// The pinned source, unchanged and reusable.
    pub buffer: PinnedHostBuffer,
    /// The destination value, unchanged and reusable.
    pub value: Tensor,
    /// Why the submission was refused.
    pub reason: &'static str,
}

/// A failure of an explicit transfer submission.
///
/// The rejected variant carries the submitted package (a pinned buffer and a tensor), so
/// it is boxed to keep the error type small.
#[derive(Debug)]
#[non_exhaustive]
#[doc(hidden)]
pub enum TransferSubmitError {
    /// The submission was refused before any work was admitted; the package is
    /// returned so the caller can retry with a valid destination.
    Rejected(Box<RejectedTransfer>),
    /// The runtime could not admit the transfer; the source keeps the typed cause.
    Runtime(tenferro_tensor::BoxError),
}

impl std::fmt::Display for TransferSubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransferSubmitError::Rejected(rejected) => {
                write!(f, "transfer rejected: {}", rejected.reason)
            }
            TransferSubmitError::Runtime(detail) => {
                write!(f, "transfer submission failed: {detail}")
            }
        }
    }
}

impl std::error::Error for TransferSubmitError {}

/// A submitted host-to-device copy whose completion has not been observed yet.
///
/// Waiting returns both the destination value and the pinned source, so the caller can
/// reuse either. Dropping the handle detaches the package into the runtime's
/// retirement queue; `core::mem::forget` leaks it deliberately.
#[doc(hidden)]
pub struct PendingUpload {
    package: Option<(PinnedHostBuffer, Tensor)>,
    completion: Option<t4a_cuda_runtime::Completion>,
    state: Arc<CudaRuntimeState>,
}

impl PendingUpload {
    pub(crate) fn new(
        buffer: PinnedHostBuffer,
        value: Tensor,
        completion: t4a_cuda_runtime::Completion,
        state: Arc<CudaRuntimeState>,
    ) -> Self {
        Self {
            package: Some((buffer, value)),
            completion: Some(completion),
            state,
        }
    }

    /// Whether the copy has completed. Never blocks.
    ///
    /// # Errors
    ///
    /// The runtime's asynchronous failure, if the stream reported one.
    #[doc(hidden)]
    pub fn is_ready(&mut self) -> crate::Result<bool> {
        let Some(completion) = self.completion.as_mut() else {
            return Ok(true);
        };
        completion
            .poll()
            .map(|ready| ready.is_some())
            .map_err(|error| crate::Error::runtime_state_source("cuda pending", error))
    }

    /// Wait for the copy and return the destination and the reusable source.
    ///
    /// # Errors
    ///
    /// The runtime's asynchronous failure, if the stream reported one.
    #[doc(hidden)]
    pub fn wait(mut self) -> crate::Result<(Tensor, PinnedHostBuffer)> {
        if let Some(completion) = self.completion.as_mut() {
            completion
                .wait()
                .map_err(|error| crate::Error::runtime_state_source("cuda pending", error))?;
        }
        let (buffer, value) = self
            .package
            .take()
            .expect("a pending upload returns its package exactly once");
        Ok((value, buffer))
    }
}

impl std::fmt::Debug for PendingUpload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingUpload")
            .field("ready", &self.completion.is_none())
            .finish()
    }
}

impl Drop for PendingUpload {
    fn drop(&mut self) {
        // Detach: the package stays with the runtime until the stream is observed, so
        // the queued copy's source and destination are neither freed nor reused early.
        if let Some(package) = self.package.take() {
            let runtime = CudaRuntime {
                inner: Arc::clone(&self.state),
            };
            runtime.detach(Box::new(package), self.completion.take());
        }
    }
}

/// A submitted device-to-host copy whose completion has not been observed yet.
///
/// The copy writes straight into the pinned buffer inside the runtime's raw scope, so
/// waiting only has to observe completion.
#[doc(hidden)]
pub struct PendingDownload {
    package: Option<(Tensor, PinnedHostBuffer)>,
    completion: Option<t4a_cuda_runtime::Completion>,
    state: Arc<CudaRuntimeState>,
}

impl PendingDownload {
    pub(crate) fn new(
        value: Tensor,
        buffer: PinnedHostBuffer,
        completion: t4a_cuda_runtime::Completion,
        state: Arc<CudaRuntimeState>,
    ) -> Self {
        Self {
            package: Some((value, buffer)),
            completion: Some(completion),
            state,
        }
    }

    /// Whether the copy has completed. Never blocks.
    ///
    /// # Errors
    ///
    /// The runtime's asynchronous failure, if the stream reported one.
    #[doc(hidden)]
    pub fn is_ready(&mut self) -> crate::Result<bool> {
        let Some(completion) = self.completion.as_mut() else {
            return Ok(true);
        };
        completion
            .poll()
            .map(|ready| ready.is_some())
            .map_err(|error| crate::Error::runtime_state_source("cuda pending", error))
    }

    /// Wait for the copy and return the value with the filled pinned buffer.
    ///
    /// # Errors
    ///
    /// The runtime's asynchronous failure, if the stream reported one.
    #[doc(hidden)]
    pub fn wait(mut self) -> crate::Result<(Tensor, PinnedHostBuffer)> {
        if let Some(completion) = self.completion.as_mut() {
            completion
                .wait()
                .map_err(|error| crate::Error::runtime_state_source("cuda pending", error))?;
        }
        Ok(self
            .package
            .take()
            .expect("a pending download returns its package exactly once"))
    }
}

impl std::fmt::Debug for PendingDownload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingDownload")
            .field("ready", &self.completion.is_none())
            .finish()
    }
}

impl Drop for PendingDownload {
    fn drop(&mut self) {
        if let Some(package) = self.package.take() {
            let runtime = CudaRuntime {
                inner: Arc::clone(&self.state),
            };
            runtime.detach(Box::new(package), self.completion.take());
        }
    }
}

/// The CUDA allocation behind `value`, for every preset dtype.
pub(crate) fn cuda_storage(value: &Tensor) -> Option<&CudaStorage> {
    use tenferro_tensor::{Complex32, Complex64, DType};
    fn pick<T: tenferro_tensor::TensorScalar + Send + Sync + 'static>(
        value: &Tensor,
    ) -> Option<&CudaStorage> {
        let typed = value.as_typed::<T>()?;
        match typed.buffer() {
            tenferro_tensor::StorageBuffer::Backend(buffer) => {
                buffer.as_ref().as_any().downcast_ref::<CudaStorage>()
            }
            tenferro_tensor::StorageBuffer::Host(_) => None,
        }
    }
    match value.dtype() {
        DType::F64 => pick::<f64>(value),
        DType::F32 => pick::<f32>(value),
        DType::I32 => pick::<i32>(value),
        DType::I64 => pick::<i64>(value),
        DType::C32 => pick::<Complex32>(value),
        DType::C64 => pick::<Complex64>(value),
        DType::Bool | DType::External(_) => None,
    }
}
