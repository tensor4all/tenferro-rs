use tenferro_tensor::{DType, ErrorKind};
use thiserror::Error;

/// Structured errors owned by the CUDA backend.
#[derive(Debug, Error)]
#[non_exhaustive]
pub(crate) enum CudaError {
    /// The operation does not have a CUDA implementation for this dtype.
    #[error("{op} does not support dtype {dtype:?} on CUDA")]
    UnsupportedDType { op: &'static str, dtype: DType },
    /// The backend does not implement an operation or configuration.
    #[error("{op} is unsupported on CUDA: {detail}")]
    UnsupportedOperation {
        op: &'static str,
        detail: &'static str,
    },
    /// A CUDA library call returned a non-success status.
    #[error("{library} call {call} returned status {status}")]
    ProviderStatus {
        library: &'static str,
        call: &'static str,
        status: i32,
    },
    /// A CUDA provider reported a workspace size that cannot be represented
    /// by the host allocator.
    #[error("{op} workspace size {size} does not fit in usize")]
    WorkspaceSizeOverflow { op: &'static str, size: u64 },
    /// Integer arithmetic encountered a zero divisor.
    #[error("{op} detected division by zero for dtype {dtype:?} on CUDA")]
    DivisionByZero { op: &'static str, dtype: DType },
    /// Integer power received a negative exponent.
    #[error("{op} received a negative integer exponent for dtype {dtype:?} on CUDA")]
    NegativeIntegerExponent { op: &'static str, dtype: DType },
}

/// Construct an operation-level unsupported-dtype error without pretending it
/// is a conversion between two different dtypes.
pub(crate) fn unsupported_dtype(op: &'static str, dtype: DType) -> crate::Error {
    crate::Error::extension(
        op,
        "cuda",
        ErrorKind::Unsupported,
        CudaError::UnsupportedDType { op, dtype },
    )
}

/// Construct a typed CUDA numerical-domain failure.
pub(crate) fn division_by_zero(op: &'static str, dtype: DType) -> crate::Error {
    crate::Error::extension(
        op,
        "cuda",
        ErrorKind::NumericalFailure,
        CudaError::DivisionByZero { op, dtype },
    )
}

/// Construct a typed CUDA operation-capability failure.
pub(crate) fn unsupported_operation(op: &'static str, detail: &'static str) -> crate::Error {
    crate::Error::extension(
        op,
        "cuda",
        ErrorKind::Unsupported,
        CudaError::UnsupportedOperation { op, detail },
    )
}

/// Preserve a CUDA library status as a typed backend source.
pub(crate) fn provider_status(
    op: &'static str,
    library: &'static str,
    call: &'static str,
    status: i32,
) -> crate::Error {
    crate::Error::extension(
        op,
        "cuda",
        ErrorKind::BackendFailure,
        CudaError::ProviderStatus {
            library,
            call,
            status,
        },
    )
}

/// Preserve a CUDA workspace allocation overflow as a typed backend source.
pub(crate) fn workspace_size_overflow(op: &'static str, size: u64) -> crate::Error {
    crate::Error::extension(
        op,
        "cuda",
        ErrorKind::BackendFailure,
        CudaError::WorkspaceSizeOverflow { op, size },
    )
}

/// Construct a typed CUDA negative-exponent failure.
pub(crate) fn negative_integer_exponent(op: &'static str, dtype: DType) -> crate::Error {
    crate::Error::extension(
        op,
        "cuda",
        ErrorKind::NumericalFailure,
        CudaError::NegativeIntegerExponent { op, dtype },
    )
}

#[cfg(test)]
mod tests;
