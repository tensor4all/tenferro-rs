use std::error::Error as StdError;

/// Errors raised while preparing or executing a CUDA FFT request.
#[derive(Debug, thiserror::Error)]
pub(crate) enum CudaFftError {
    /// A descriptor value cannot be represented or used by cuFFT.
    #[error("invalid cuFFT configuration field: {field}")]
    InvalidConfiguration { field: &'static str },
}

/// Translate a descriptor configuration failure to structured tensor validation.
pub(crate) fn into_tensor_error(op: &'static str, source: CudaFftError) -> tenferro_tensor::Error {
    match source {
        CudaFftError::InvalidConfiguration { field } => tenferro_tensor::Error::invalid_argument(
            op,
            field,
            "cuFFT descriptor configuration is invalid",
        ),
    }
}

/// Translate loader, vendor-status, and lifecycle failures to typed backend sources.
pub(crate) fn into_backend_source_error<E>(op: &'static str, source: E) -> tenferro_tensor::Error
where
    E: StdError + Send + Sync + 'static,
{
    tenferro_tensor::Error::backend_source(op, source)
}
