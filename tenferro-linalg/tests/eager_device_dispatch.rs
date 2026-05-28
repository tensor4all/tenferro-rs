#![cfg(not(feature = "cuda"))]

use std::sync::Arc;

use tenferro_linalg::ad_support::{LinalgExtensionOp, LinalgOp};
use tenferro_runtime::extension::ExtensionOpTrait;
use tenferro_tensor::{
    Buffer, BufferHandle, ComputeDevice, DeviceKind, Error, GpuBackendKind, MemoryKind, Placement,
    Tensor, TypedTensor,
};

#[test]
fn eager_linalg_rejects_cuda_tensor_when_cuda_feature_is_disabled() {
    let tensor = Tensor::F64(TypedTensor::from_buffer_col_major(
        vec![2, 2],
        Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(7, 4))),
        Placement {
            memory_kind: MemoryKind::Device,
            device: Some(ComputeDevice {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal: 0,
            }),
        },
    ));
    let op = LinalgExtensionOp::new(LinalgOp::Cholesky);

    let err = op.eager_execute(&[&tensor]).unwrap_err();

    match err {
        Error::BackendFailure { op, message } => {
            assert_eq!(op, "linalg_eager_execute");
            assert!(message.contains("cuda feature"));
            assert!(message.contains("download"));
        }
        other => panic!("expected BackendFailure, got {other:?}"),
    }
}
