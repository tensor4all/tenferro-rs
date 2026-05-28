use std::sync::Arc;

use crate::{
    Buffer, BufferHandle, Error, MemoryKind, Placement, TensorViewCanonicalization, TypedTensor,
};

fn opaque_backend_placement() -> Placement {
    Placement {
        memory_kind: MemoryKind::Device,
        device: None,
    }
}

#[test]
fn cpu_backend_rejects_backend_view_without_download() {
    let mut backend = crate::cpu::CpuBackend::new();
    let tensor = TypedTensor::<f64>::from_buffer_col_major(
        vec![2],
        Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(7, 2))),
        opaque_backend_placement(),
    );

    let err = backend.to_contiguous(&tensor.as_view()).unwrap_err();

    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "CpuBackend::to_contiguous",
            ref message,
        } if message.contains("download")
    ));
}

#[test]
fn cpu_backend_rejects_backend_copy_back_without_download() {
    let mut backend = crate::cpu::CpuBackend::new();
    let src = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]);
    let mut dst = TypedTensor::<f64>::from_buffer_col_major(
        vec![2],
        Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(8, 2))),
        opaque_backend_placement(),
    );

    let err = backend
        .copy_from_contiguous(&src, &mut dst.as_view_mut())
        .unwrap_err();

    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "CpuBackend::copy_from_contiguous",
            ref message,
        } if message.contains("download")
    ));
}
