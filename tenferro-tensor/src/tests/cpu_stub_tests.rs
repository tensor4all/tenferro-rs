use std::sync::Arc;

use crate::{
    Buffer, BufferHandle, DotGeneralConfig, Error, MemoryKind, Placement, Tensor, TensorDot,
    TensorRead, TensorReduction, TensorView, TensorViewCanonicalization, TypedTensor,
};

fn opaque_backend_placement() -> Placement {
    Placement {
        memory_kind: MemoryKind::Device,
        device: None,
    }
}

fn backend_tensor_f64(handle_id: u64, len: usize) -> Tensor {
    Tensor::F64(TypedTensor::<f64>::from_buffer_col_major(
        vec![len],
        Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(handle_id, len))),
        opaque_backend_placement(),
    ))
}

fn assert_backend_download_error(result: crate::Result<Tensor>, expected_op: &'static str) {
    let err = result.unwrap_err();

    assert!(matches!(
        err,
        Error::BackendFailure {
            op,
            ref message,
        } if op == expected_op && message.contains("download")
    ));
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

#[test]
fn cpu_dot_general_read_rejects_backend_view_without_panic() {
    let mut backend = crate::cpu::CpuBackend::new();
    let lhs = TypedTensor::<f64>::from_buffer_col_major(
        vec![2, 2],
        Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(9, 4))),
        opaque_backend_placement(),
    );
    let rhs = Tensor::F64(TypedTensor::<f64>::from_vec_col_major(
        vec![2, 2],
        vec![1.0, 2.0, 3.0, 4.0],
    ));
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let err = backend
        .dot_general_read(
            TensorRead::from_view(TensorView::F64(lhs.as_view())),
            TensorRead::from_tensor(&rhs),
            &config,
        )
        .unwrap_err();

    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "dot_general",
            ref message,
        } if message.contains("download")
    ));
}

#[test]
fn cpu_reduce_read_rejects_backend_tensors_with_empty_axes() {
    let mut backend = crate::cpu::CpuBackend::new();
    let input = backend_tensor_f64(10, 2);

    assert_backend_download_error(
        backend.reduce_sum_read(TensorRead::from_tensor(&input), &[]),
        "reduce_sum",
    );
    assert_backend_download_error(
        backend.reduce_prod_read(TensorRead::from_tensor(&input), &[]),
        "reduce_prod",
    );
    assert_backend_download_error(
        backend.reduce_max_read(TensorRead::from_tensor(&input), &[]),
        "reduce_max",
    );
    assert_backend_download_error(
        backend.reduce_min_read(TensorRead::from_tensor(&input), &[]),
        "reduce_min",
    );
}
