use std::sync::Arc;

use tenferro_runtime::extension::ExtensionOp;
use tenferro_tensor::{
    Buffer, BufferHandle, DType, DeviceId, DeviceKind, Error, GpuBackendKind, MemoryKind,
    Placement, Tensor, TypedTensor,
};

use super::{promote_dtypes, LinalgExtensionOp, LinalgOp};

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

    let err = op
        .host_reference()
        .expect("linalg host reference")
        .execute(&[&tensor])
        .unwrap_err();

    match err {
        Error::BackendFailure { op, message } => {
            assert_eq!(op, "linalg_host_reference");
            assert!(message.contains("cuda feature"));
            assert!(message.contains("download"));
        }
        other => panic!("expected BackendFailure, got {other:?}"),
    }
}

#[test]
fn infer_output_meta_returns_error_on_input_count_mismatch() {
    let op = LinalgExtensionOp::new(LinalgOp::Cholesky);

    let err = op.infer_output_meta(&[], &[]).unwrap_err();

    assert!(matches!(
        err,
        Error::InvalidConfig {
            op: "tenferro-linalg",
            ..
        }
    ));
}

#[test]
fn extension_dtype_promotion_delegates_to_canonical_tensor_rules() {
    let source = include_str!("../extension.rs");
    assert!(
        !source.contains("fn promote_dtype("),
        "linalg extension metadata must not duplicate the canonical dtype promotion lattice"
    );

    let dtypes = [
        DType::Bool,
        DType::I32,
        DType::I64,
        DType::F32,
        DType::F64,
        DType::C32,
        DType::C64,
    ];
    for lhs in dtypes {
        for rhs in dtypes {
            assert_eq!(
                promote_dtypes(&[lhs, rhs]),
                tenferro_tensor::validate::promote_dtype(lhs, rhs),
                "promotion mismatch for {lhs:?}, {rhs:?}"
            );
        }
    }
}
