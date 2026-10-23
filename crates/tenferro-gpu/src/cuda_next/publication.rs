//! Source publication: a produced value with a frontier another engine can wait on.
//!
//! A publication retains the producing runtime, the value's allocation and one
//! completion checkpoint recorded on the producing stream. Admission never
//! synchronises; a reader either polls [`SourcePublication::frontier_ready`] or waits
//! with [`SourcePublication::await_frontier`].

use std::sync::Arc;

use tenferro_tensor::Tensor;

use super::pending::TransferSubmitError;
use super::runtime::{CudaRuntime, CudaRuntimeState};

/// A publication refused before admission, with the submitted value returned.
#[derive(Debug)]
#[doc(hidden)]
pub struct RejectedPublication {
    /// The value, unchanged and reusable.
    pub value: Tensor,
    /// Why the publication was refused.
    pub reason: &'static str,
}

/// A failure of publication admission.
#[derive(Debug)]
#[non_exhaustive]
#[doc(hidden)]
pub enum PublicationAdmissionError {
    /// The admission was refused before any state was published.
    Rejected(Box<RejectedPublication>),
    /// The runtime could not record the frontier; the source keeps the typed cause.
    Runtime(tenferro_tensor::BoxError),
}

impl std::fmt::Display for PublicationAdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PublicationAdmissionError::Rejected(rejected) => {
                write!(f, "publication rejected: {}", rejected.reason)
            }
            PublicationAdmissionError::Runtime(source) => {
                write!(f, "publication admission failed: {source}")
            }
        }
    }
}

impl std::error::Error for PublicationAdmissionError {}

/// A failure of a published value's copy.
#[derive(Debug)]
#[non_exhaustive]
#[doc(hidden)]
pub enum PublicationError {
    /// The copy could not be submitted.
    Transfer(TransferSubmitError),
    /// The frontier or the copy failed asynchronously; the source keeps the typed cause.
    Runtime(tenferro_tensor::BoxError),
}

impl std::fmt::Display for PublicationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PublicationError::Transfer(error) => write!(f, "{error}"),
            PublicationError::Runtime(source) => write!(f, "publication failure: {source}"),
        }
    }
}

impl std::error::Error for PublicationError {}

/// A produced value whose producing frontier is retained.
///
/// Waiting for the frontier never blocks admission: the checkpoint was recorded when
/// the value was published. Dropping the publication detaches it into the runtime's
/// retirement queue; `core::mem::forget` leaks it deliberately.
#[doc(hidden)]
pub struct SourcePublication {
    value: Option<Tensor>,
    frontier: Option<t4a_cuda_runtime::Completion>,
    state: Arc<CudaRuntimeState>,
}

impl SourcePublication {
    pub(crate) fn new(
        value: Tensor,
        frontier: t4a_cuda_runtime::Completion,
        state: Arc<CudaRuntimeState>,
    ) -> Self {
        Self {
            value: Some(value),
            frontier: Some(frontier),
            state,
        }
    }

    /// Payload length in bytes.
    #[doc(hidden)]
    pub fn bytes(&self) -> usize {
        super::pending::cuda_storage(self.value.as_ref().expect("published value"))
            .map(|storage| storage.bytes())
            .unwrap_or(0)
    }

    /// Whether the producing frontier has completed. Never blocks.
    ///
    /// # Errors
    ///
    /// [`PublicationError::Runtime`] when the producing stream reported a failure.
    #[doc(hidden)]
    pub fn frontier_ready(&mut self) -> Result<bool, PublicationError> {
        let Some(frontier) = self.frontier.as_mut() else {
            return Ok(true);
        };
        frontier
            .poll()
            .map(|ready| ready.is_some())
            .map_err(|error| PublicationError::Runtime(Box::new(error)))
    }

    /// Wait for the producing frontier.
    ///
    /// # Errors
    ///
    /// [`PublicationError::Runtime`] when the producing stream reported a failure.
    #[doc(hidden)]
    pub fn await_frontier(&mut self) -> Result<(), PublicationError> {
        let Some(frontier) = self.frontier.as_mut() else {
            return Ok(());
        };
        frontier
            .wait()
            .map_err(|error| PublicationError::Runtime(Box::new(error)))
    }

    /// Copy the published value into a pinned buffer, ordered behind its frontier.
    ///
    /// # Errors
    ///
    /// [`PublicationError::Transfer`] when the copy is refused, and
    /// [`PublicationError::Runtime`] when it cannot be admitted.
    #[doc(hidden)]
    pub fn download_into(
        &self,
        mut buffer: super::pinned::PinnedHostBuffer,
    ) -> Result<PublicationDownload, PublicationError> {
        let storage = super::pending::cuda_storage(self.value.as_ref().expect("published value"))
            .ok_or_else(|| {
            PublicationError::Runtime(Box::new(std::io::Error::other(
                "the published value is not a CUDA allocation",
            )))
        })?;
        let runtime = CudaRuntime {
            inner: Arc::clone(&self.state),
        };
        let completion = super::transfer::submit_download(&runtime, storage, &mut buffer)
            .map_err(|error| PublicationError::Runtime(Box::new(error)))?;
        Ok(PublicationDownload {
            buffer: Some(buffer),
            completion: Some(completion),
            state: Arc::clone(&self.state),
        })
    }
}

impl Drop for SourcePublication {
    fn drop(&mut self) {
        if let Some(value) = self.value.take() {
            let runtime = CudaRuntime {
                inner: Arc::clone(&self.state),
            };
            runtime.detach(Box::new(value), self.frontier.take());
        }
    }
}

/// A published value's copy into a pinned buffer.
///
/// The copy writes straight into the buffer, so waiting only observes completion; the
/// publication keeps owning the value, which is why the first element of the wait is
/// the unit rather than a tensor.
#[doc(hidden)]
pub struct PublicationDownload {
    buffer: Option<super::pinned::PinnedHostBuffer>,
    completion: Option<t4a_cuda_runtime::Completion>,
    state: Arc<CudaRuntimeState>,
}

impl PublicationDownload {
    /// Whether the copy has completed. Never blocks.
    ///
    /// # Errors
    ///
    /// [`PublicationError::Runtime`] when the copy failed asynchronously.
    #[doc(hidden)]
    pub fn is_ready(&mut self) -> Result<bool, PublicationError> {
        let Some(completion) = self.completion.as_mut() else {
            return Ok(true);
        };
        completion
            .poll()
            .map(|ready| ready.is_some())
            .map_err(|error| PublicationError::Runtime(Box::new(error)))
    }

    /// Wait for the copy and return the filled buffer.
    ///
    /// # Errors
    ///
    /// [`PublicationError::Runtime`] when the copy failed asynchronously.
    #[doc(hidden)]
    pub fn wait(mut self) -> Result<((), super::pinned::PinnedHostBuffer), PublicationError> {
        if let Some(completion) = self.completion.as_mut() {
            completion
                .wait()
                .map_err(|error| PublicationError::Runtime(Box::new(error)))?;
        }
        Ok((
            (),
            self.buffer
                .take()
                .expect("a publication download returns its buffer exactly once"),
        ))
    }
}

impl Drop for PublicationDownload {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            let runtime = CudaRuntime {
                inner: Arc::clone(&self.state),
            };
            runtime.detach(Box::new(buffer), self.completion.take());
        }
    }
}
