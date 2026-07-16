use std::sync::Arc;

use num_complex::{Complex32, Complex64};

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
    ).unwrap())
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
    let mut backend = crate::CpuBackend::new();
    let tensor = TypedTensor::<f64>::from_buffer_col_major(
        vec![2],
        Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(7, 2))),
        opaque_backend_placement(),
    ).unwrap();

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
fn cpu_backend_copy_into_rejects_backend_destination_without_download() {
    let mut backend = crate::CpuBackend::new();
    let src = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    let mut dst = TypedTensor::<f64>::from_buffer_col_major(
        vec![2],
        Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(8, 2))),
        opaque_backend_placement(),
    ).unwrap();

    let err = backend
        .copy_into(&src.as_view(), &mut dst.as_view_mut())
        .unwrap_err();

    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "CpuBackend::copy_into",
            ref message,
        } if message.contains("download")
    ));
}

#[test]
fn cpu_dot_general_read_rejects_backend_view_without_panic() {
    let mut backend = crate::CpuBackend::new();
    let lhs = TypedTensor::<f64>::from_buffer_col_major(
        vec![2, 2],
        Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(9, 4))),
        opaque_backend_placement(),
    ).unwrap();
    let rhs = Tensor::F64(TypedTensor::<f64>::from_vec_col_major(
        vec![2, 2],
        vec![1.0, 2.0, 3.0, 4.0],
    ).unwrap());
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
    let mut backend = crate::CpuBackend::new();
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

#[test]
fn cpu_reduce_read_rejects_backend_views_without_download() {
    let mut backend = crate::CpuBackend::new();
    let input = TypedTensor::<f64>::from_buffer_col_major(
        vec![2],
        Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(11, 2))),
        opaque_backend_placement(),
    ).unwrap();

    assert_backend_download_error(
        backend.reduce_sum_read(
            TensorRead::from_view(TensorView::F64(input.as_view())),
            &[0],
        ),
        "reduce_sum",
    );
    assert_backend_download_error(
        backend.reduce_prod_read(
            TensorRead::from_view(TensorView::F64(input.as_view())),
            &[0],
        ),
        "reduce_prod",
    );
    assert_backend_download_error(
        backend.reduce_max_read(
            TensorRead::from_view(TensorView::F64(input.as_view())),
            &[0],
        ),
        "reduce_max",
    );
    assert_backend_download_error(
        backend.reduce_min_read(
            TensorRead::from_view(TensorView::F64(input.as_view())),
            &[0],
        ),
        "reduce_min",
    );
}

#[test]
fn cpu_materialize_tensor_read_covers_host_tensor_and_view_dtypes() {
    let mut buffers = crate::buffer_pool::BufferPool::new();
    let tensors = [
        Tensor::F32(TypedTensor::from_vec_col_major(vec![1], vec![1.0_f32]).unwrap()),
        Tensor::F64(TypedTensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap()),
        Tensor::I32(TypedTensor::from_vec_col_major(vec![1], vec![1_i32]).unwrap()),
        Tensor::I64(TypedTensor::from_vec_col_major(vec![1], vec![1_i64]).unwrap()),
        Tensor::Bool(TypedTensor::from_vec_col_major(vec![1], vec![true]).unwrap()),
        Tensor::C32(TypedTensor::from_vec_col_major(
            vec![1],
            vec![Complex32::new(1.0, 0.0)],
        ).unwrap()),
        Tensor::C64(TypedTensor::from_vec_col_major(
            vec![1],
            vec![Complex64::new(1.0, 0.0)],
        ).unwrap()),
    ];
    for tensor in &tensors {
        let materialized = crate::materialize_tensor_read(
            &mut buffers,
            "dot_general",
            TensorRead::from_tensor(tensor),
        )
        .unwrap();
        assert_eq!(materialized.dtype(), tensor.dtype());
        assert_eq!(materialized.shape(), tensor.shape());
    }

    let shape = [1usize];
    let f32s = [1.0_f32];
    let f64s = [1.0_f64];
    let i32s = [1_i32];
    let i64s = [1_i64];
    let bools = [true];
    let c32s = [Complex32::new(1.0, 0.0)];
    let c64s = [Complex64::new(1.0, 0.0)];
    let views = [
        TensorView::f32(&shape, &f32s).unwrap(),
        TensorView::f64(&shape, &f64s).unwrap(),
        TensorView::i32(&shape, &i32s).unwrap(),
        TensorView::i64(&shape, &i64s).unwrap(),
        TensorView::bool(&shape, &bools).unwrap(),
        TensorView::c32(&shape, &c32s).unwrap(),
        TensorView::c64(&shape, &c64s).unwrap(),
    ];
    for view in views {
        let dtype = view.dtype();
        let materialized = crate::materialize_tensor_read(
            &mut buffers,
            "dot_general",
            TensorRead::from_view(view),
        )
        .unwrap();
        assert_eq!(materialized.dtype(), dtype);
        assert_eq!(materialized.shape(), &[1]);
    }
}
