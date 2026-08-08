/// Errors raised while preparing or executing a CUDA FFT request.
#[derive(Debug, thiserror::Error)]
pub(crate) enum CudaFftError {
    /// A descriptor value cannot be represented or used by cuFFT.
    #[error("invalid cuFFT configuration field: {field}")]
    InvalidConfiguration { field: &'static str },
}

/// Preserve a CUDA FFT error as the typed source of a tensor error.
pub(crate) fn into_tensor_error(op: &'static str, source: CudaFftError) -> tenferro_tensor::Error {
    tenferro_tensor::Error::backend_source(op, source)
}
