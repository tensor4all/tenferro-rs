use std::error::Error as _;

use tenferro_core_ops::PrimitiveOpKind;
use tenferro_tensor::{BackendId, ErrorKind, OperationCapability, TensorBackendCapability};

use super::*;

struct UnsupportedCudaCapability;

impl TensorBackendCapability for UnsupportedCudaCapability {
    fn backend_id(&self) -> BackendId {
        BackendId::Cuda
    }

    fn capabilities(&self) -> &'static [OperationCapability] {
        &[]
    }
}

#[test]
fn owned_capability_rejection_preserves_the_typed_cuda_source() {
    let error =
        require_owned_capability(&UnsupportedCudaCapability, PrimitiveOpKind::Exp, DType::C64)
            .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let source = error
        .source()
        .expect("CUDA capability failures preserve a source")
        .downcast_ref::<crate::cubecl::error::CudaError>()
        .expect("CUDA capability failures preserve CudaError");
    assert!(matches!(
        source,
        crate::cubecl::error::CudaError::UnsupportedDType {
            op: "exp",
            dtype: DType::C64,
        }
    ));
}
