/// Errors raised while preparing or executing a CUDA FFT request.
#[derive(Debug, thiserror::Error)]
pub(crate) enum CudaFftError {
    /// A descriptor value cannot be represented or used by cuFFT.
    #[error("invalid cuFFT configuration field: {field}")]
    InvalidConfiguration { field: &'static str },
    /// A required cuFFT symbol is absent from an opened library.
    #[error("failed to load cuFFT symbol {name}: {source}")]
    SymbolLoad {
        name: String,
        #[source]
        source: libloading::Error,
    },
    /// No candidate library could be opened.
    #[error("failed to load cuFFT library (tried {paths}): {source}; attempts: {attempts}")]
    LibraryLoad {
        paths: String,
        attempts: String,
        #[source]
        source: libloading::Error,
    },
    /// The ordered loader was called without any candidates.
    #[error("no cuFFT library candidates configured")]
    NoLibraryCandidates,
    /// A cuFFT API call returned a non-success result code.
    #[error("cuFFT call {function} failed with status {status}")]
    CufftStatus { function: &'static str, status: i32 },
    /// A CUDA runtime or scoped interop operation failed.
    #[error("CUDA interop failed during {operation}: {source}")]
    Interop {
        operation: &'static str,
        #[source]
        source: tenferro_tensor::BoxError,
    },
    /// An internal construction invariant was violated.
    #[error("internal cuFFT plan invariant failed: {message}")]
    InternalInvariant { message: &'static str },
}

/// Translate a descriptor configuration failure to structured tensor validation.
pub(crate) fn into_tensor_error(op: &'static str, source: CudaFftError) -> tenferro_tensor::Error {
    match source {
        CudaFftError::InvalidConfiguration { field } => tenferro_tensor::Error::invalid_argument(
            op,
            field,
            "cuFFT descriptor configuration is invalid",
        ),
        CudaFftError::InternalInvariant { message } => {
            tenferro_tensor::Error::Internal(message.into())
        }
        source => tenferro_tensor::Error::backend_source(op, source),
    }
}

impl CudaFftError {
    pub(crate) fn interop(operation: &'static str, source: tenferro_tensor::Error) -> Self {
        Self::Interop {
            operation,
            source: Box::new(source),
        }
    }

    pub(crate) fn internal(message: &'static str) -> Self {
        Self::InternalInvariant { message }
    }

    #[cfg(test)]
    pub(crate) fn test_interop(operation: &'static str) -> Self {
        Self::Interop {
            operation,
            source: Box::new(std::io::Error::other("fake CUDA interop failure")),
        }
    }
}
