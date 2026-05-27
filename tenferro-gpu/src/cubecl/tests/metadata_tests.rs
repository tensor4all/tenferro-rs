use cubecl::stream_id::StreamId;
use std::sync::Arc;

use crate::cubecl::dispatch::{
    cubecl_shape_and_strides, typed_tensor_array_arg, typed_tensor_binding,
};
use crate::cubecl::CudaExtensionCache;
use crate::{
    Buffer, ComputeDevice, CubeclBuffer, DeviceKind, GpuBackendKind, MemoryKind, Placement,
    TypedTensor,
};

#[test]
fn cubecl_metadata_uses_dense_column_major_strides() {
    assert_eq!(cubecl_shape_and_strides(&[]), (vec![], vec![]));
    assert_eq!(
        cubecl_shape_and_strides(&[2, 3, 4]),
        (vec![2, 3, 4], vec![1, 2, 6])
    );
}

#[test]
fn cuda_extension_cache_is_type_indexed_and_lazy() {
    let cache = CudaExtensionCache::new();
    let mut initializers = 0usize;

    {
        let value = cache
            .get_or_try_init::<usize>(|| {
                initializers += 1;
                Ok(17)
            })
            .unwrap();
        assert_eq!(*value, 17);
    }
    {
        let value = cache
            .get_or_try_init::<usize>(|| {
                initializers += 1;
                Ok(23)
            })
            .unwrap();
        assert_eq!(*value, 17);
    }
    {
        let value = cache
            .get_or_try_init::<String>(|| Ok("gpu".to_string()))
            .unwrap();
        assert_eq!(value.as_str(), "gpu");
    }

    assert_eq!(initializers, 1);
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
fn typed_tensor_array_arg_rejects_shape_buffer_len_mismatch() {
    let tensor = cubecl_tensor_with_len(vec![2, 3], 5);

    let err = match typed_tensor_array_arg(&tensor, "metadata_test") {
        Ok(_) => panic!("expected typed_tensor_array_arg to reject buffer length mismatch"),
        Err(err) => err,
    };

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

fn cubecl_tensor_with_len(shape: Vec<usize>, len: usize) -> TypedTensor<f32> {
    let handle = cubecl::server::Handle::new(
        StreamId::current(),
        (len * core::mem::size_of::<f32>()) as u64,
    );
    TypedTensor {
        buffer: Buffer::Backend(Arc::new(CubeclBuffer::new(handle, len))),
        shape,
        placement: Placement {
            memory_kind: MemoryKind::Device,
            device: Some(ComputeDevice {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal: 0,
            }),
        },
    }
}
