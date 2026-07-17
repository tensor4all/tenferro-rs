use tenferro_tensor::{DType, ErrorKind};
use thiserror::Error;

/// Structured errors owned by the CUDA backend.
#[derive(Debug, Error)]
#[non_exhaustive]
pub(crate) enum CudaError {
    /// The operation does not have a CUDA implementation for this dtype.
    #[error("{op} does not support dtype {dtype:?} on CUDA")]
    UnsupportedDType { op: &'static str, dtype: DType },
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
mod tests {
    use std::error::Error as _;

    use super::*;

    #[test]
    fn unsupported_dtype_preserves_classification_and_source() {
        let error = unsupported_dtype("exp", DType::I32);

        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let source = error.source().expect("extension errors have a source");
        let source = source
            .downcast_ref::<CudaError>()
            .expect("CUDA errors preserve their typed source");
        assert!(matches!(
            source,
            CudaError::UnsupportedDType {
                op: "exp",
                dtype: DType::I32
            }
        ));
    }
}
