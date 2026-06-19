use std::sync::Arc;

use tenferro_runtime::extension::ExtensionOp;
use tenferro_tensor::{
    Buffer, BufferHandle, DeviceId, DeviceKind, Error, GpuBackendKind, MemoryKind, Placement,
    Tensor, TypedTensor,
};

use super::{LinalgExtensionOp, LinalgOp};

#[test]
fn eager_linalg_rejects_cuda_tensor_when_cuda_feature_is_disabled() {
    let tensor = Tensor::F64(
        TypedTensor::from_buffer_col_major(
            vec![2, 2],
            Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(7, 4))),
            Placement {
                memory_kind: MemoryKind::Device,
                device: Some(DeviceId {
                    kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                    ordinal: 0,
                }),
            },
        )
        .unwrap(),
    );
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

#[test]
fn infer_output_meta_returns_empty_on_input_count_mismatch() {
    let op = LinalgExtensionOp::new(LinalgOp::Cholesky);

    let metas = op.infer_output_meta(&[], &[]);

    assert!(metas.is_empty());
}
