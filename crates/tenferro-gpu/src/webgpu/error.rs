use crate::{DType, Error, ErrorKind};

#[derive(Debug, thiserror::Error)]
pub(crate) enum WebGpuError {
    #[error("{op} does not support dtype {dtype:?} on WebGPU")]
    UnsupportedDType { op: &'static str, dtype: DType },
    #[error("{op} is unsupported on WebGPU: {detail}")]
    UnsupportedOperation {
        op: &'static str,
        detail: &'static str,
    },
}

pub(crate) fn unsupported_dtype(op: &'static str, dtype: DType) -> Error {
    Error::extension(
        op,
        "webgpu",
        ErrorKind::Unsupported,
        WebGpuError::UnsupportedDType { op, dtype },
    )
}

pub(crate) fn unsupported_operation(op: &'static str, detail: &'static str) -> Error {
    Error::extension(
        op,
        "webgpu",
        ErrorKind::Unsupported,
        WebGpuError::UnsupportedOperation { op, detail },
    )
}
