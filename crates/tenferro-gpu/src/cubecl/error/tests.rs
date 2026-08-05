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

#[test]
fn provider_status_preserves_classification_and_source() {
    let error = provider_status("dot_general", "cuTENSOR", "cutensorContract", 7);

    assert_eq!(error.kind(), ErrorKind::BackendFailure);
    let source = error.source().expect("CUDA errors have a source");
    let source = source
        .downcast_ref::<CudaError>()
        .expect("CUDA errors preserve their typed source");
    assert!(matches!(
        source,
        CudaError::ProviderStatus {
            library: "cuTENSOR",
            call: "cutensorContract",
            status: 7,
        }
    ));
}

#[test]
fn workspace_overflow_preserves_classification_and_source() {
    let error = workspace_size_overflow("dot_general", u64::MAX);

    assert_eq!(error.kind(), ErrorKind::BackendFailure);
    let source = error.source().expect("CUDA errors have a source");
    let source = source
        .downcast_ref::<CudaError>()
        .expect("CUDA errors preserve their typed source");
    assert!(matches!(
        source,
        CudaError::WorkspaceSizeOverflow {
            op: "dot_general",
            size: u64::MAX,
        }
    ));
}
