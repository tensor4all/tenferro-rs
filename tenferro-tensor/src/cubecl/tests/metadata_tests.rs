use cubecl::stream_id::StreamId;

use crate::cubecl::dispatch::{cubecl_shape_and_strides, typed_tensor_binding};
use crate::cubecl::memory::canonical_host_tensor_for_upload;
use crate::{Buffer, ComputeDevice, CubeclBuffer, MemoryKind, MemoryOrder, Placement, TypedTensor};

#[test]
fn cubecl_metadata_uses_dense_column_major_strides() {
    assert_eq!(cubecl_shape_and_strides(&[]), (vec![], vec![]));
    assert_eq!(
        cubecl_shape_and_strides(&[2, 3, 4]),
        (vec![2, 3, 4], vec![1, 2, 6])
    );
}

#[test]
fn typed_tensor_binding_rejects_shape_buffer_len_mismatch() {
    let tensor = cubecl_tensor_with_len(vec![2, 3], 5);

    let err = typed_tensor_binding(&tensor, "metadata_test").unwrap_err();

    match err {
        crate::Error::BackendFailure { op, message } => {
            assert_eq!(op, "metadata_test");
            assert!(message.contains("expected shape product 6"));
            assert!(message.contains("actual CubeclBuffer::len 5"));
        }
        other => panic!("expected BackendFailure, got {other:?}"),
    }
}

#[test]
fn typed_tensor_binding_rejects_shape_product_overflow() {
    let tensor = cubecl_tensor_with_len(vec![usize::MAX, 2], 1);

    let err = typed_tensor_binding(&tensor, "metadata_test").unwrap_err();

    match err {
        crate::Error::BackendFailure { op, message } => {
            assert_eq!(op, "metadata_test");
            assert!(message.contains("shape product overflow"));
            assert!(message.contains("["));
        }
        other => panic!("expected BackendFailure, got {other:?}"),
    }
}

#[test]
fn typed_tensor_binding_rejects_row_major_gpu_tensor() {
    let mut tensor = cubecl_tensor_with_len(vec![2, 3], 6);
    tensor.order = MemoryOrder::RowMajor;

    let err = typed_tensor_binding(&tensor, "metadata_test").unwrap_err();

    match err {
        crate::Error::BackendFailure { op, message } => {
            assert_eq!(op, "metadata_test");
            assert!(message.contains("column-major GPU tensor"));
        }
        other => panic!("expected BackendFailure, got {other:?}"),
    }
}

#[test]
fn upload_canonicalizes_row_major_host_tensor_to_col_major() {
    let tensor =
        TypedTensor::from_vec_row_major(vec![2, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let upload_source = canonical_host_tensor_for_upload(&tensor).unwrap();

    assert_eq!(upload_source.order(), MemoryOrder::ColMajor);
    assert_eq!(upload_source.host_data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn upload_canonicalization_rejects_non_host_tensors() {
    let tensor = cubecl_tensor_with_len(vec![2], 2);

    let err = canonical_host_tensor_for_upload(&tensor).unwrap_err();

    match err {
        crate::Error::BackendFailure { op, message } => {
            assert_eq!(op, "upload");
            assert!(message.contains("already backed by CubeCL"));
        }
        other => panic!("expected BackendFailure, got {other:?}"),
    }
}

fn cubecl_tensor_with_len(shape: Vec<usize>, len: usize) -> TypedTensor<f32> {
    let handle = cubecl::server::Handle::new(
        StreamId::current(),
        (len * core::mem::size_of::<f32>()) as u64,
    );
    TypedTensor {
        buffer: Buffer::Cubecl(CubeclBuffer::new(handle, len)),
        shape,
        placement: Placement {
            memory_kind: MemoryKind::Device,
            resident_device: Some(ComputeDevice {
                kind: "cuda".into(),
                ordinal: 0,
            }),
        },
        order: crate::MemoryOrder::ColMajor,
    }
}
