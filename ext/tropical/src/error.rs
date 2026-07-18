use tenferro_einsum::Error as EinsumError;
use tenferro_tensor::{DType, Error, ErrorKind};

#[derive(Debug, thiserror::Error)]
pub(crate) enum TropicalError {
    #[error("{op} does not support dtype {dtype:?}")]
    UnsupportedDType { op: &'static str, dtype: DType },
}

pub(crate) fn unsupported_dtype(op: &'static str, dtype: DType) -> Error {
    Error::extension(
        op,
        "tropical",
        ErrorKind::Unsupported,
        TropicalError::UnsupportedDType { op, dtype },
    )
}

pub(crate) fn from_einsum_error(op: &'static str, error: EinsumError) -> Error {
    match error {
        EinsumError::Validation { source, .. } => Error::validation(op, source),
        error => Error::extension(op, "tropical", error.kind(), error),
    }
}
