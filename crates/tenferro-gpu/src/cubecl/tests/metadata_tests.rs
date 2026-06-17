use cubecl::stream_id::StreamId;
use std::num::NonZeroUsize;
use std::panic;

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
    assert_eq!(cubecl_shape_and_strides(&[]), (vec![1], vec![1]));
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
fn cuda_extension_cache_reports_stats_and_clear() {
    let cache = CudaExtensionCache::new();
    assert_eq!(cache.stats().entries, 0);
    assert_eq!(cache.stats().retained_bytes, 0);

    let _usize = cache.get_or_try_init::<usize>(|| Ok(17)).unwrap();
    drop(_usize);
    let _string = cache
        .get_or_try_init::<String>(|| Ok("gpu".to_string()))
        .unwrap();
    drop(_string);

    let stats = cache.stats();
    assert_eq!(stats.entries, 2);
    assert!(stats.retained_bytes >= std::mem::size_of::<usize>());

    cache.clear();
    assert!(cache.is_empty());
    assert_eq!(cache.stats().entries, 0);
}

#[test]
fn cuda_extension_cache_try_methods_report_poisoned_lock() {
    let cache = CudaExtensionCache::new();
    let poisoned = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let _guard = cache.inner.lock().unwrap();
        panic!("poison cuda extension cache lock");
    }));
    assert!(poisoned.is_err());

    assert!(cache.try_is_empty().is_err());
    assert!(cache.try_stats().is_err());
    assert!(cache.try_clear().is_err());
    assert!(cache.try_max_entries().is_err());
    assert!(cache.get_or_try_init::<usize>(|| Ok(17)).is_err());
}

#[test]
fn cuda_extension_cache_has_configurable_entry_bound() {
    let cache = CudaExtensionCache::with_max_entries(NonZeroUsize::new(1).unwrap());
    let mut usize_initializers = 0usize;

    let value = cache
        .get_or_try_init::<usize>(|| {
            usize_initializers += 1;
            Ok(17)
        })
        .unwrap();
    assert_eq!(*value, 17);
    drop(value);

    let value = cache
        .get_or_try_init::<String>(|| Ok("gpu".to_string()))
        .unwrap();
    assert_eq!(value.as_str(), "gpu");
    drop(value);
    assert_eq!(cache.stats().entries, 1);

    let value = cache
        .get_or_try_init::<usize>(|| {
            usize_initializers += 1;
            Ok(23)
        })
        .unwrap();
    assert_eq!(*value, 23);
    assert_eq!(usize_initializers, 2);
}

#[test]
fn typed_tensor_binding_accepts_valid_backend_metadata() {
    let tensor = cubecl_tensor_with_len(vec![2, 3], 6);

    typed_tensor_binding(&tensor, "metadata_test").unwrap();
    typed_tensor_array_arg(&tensor, "metadata_test").unwrap();
}

#[test]
fn from_buffer_col_major_rejects_backend_buffer_len_mismatch() {
    let panic = constructor_panic_message(vec![2, 3], 5);

    assert!(panic.contains("from_buffer_col_major"));
    assert!(panic.contains("data length 5 does not match shape product 6"));
}

#[test]
fn from_buffer_col_major_rejects_shape_product_overflow() {
    let panic = constructor_panic_message(vec![usize::MAX, 2], 1);

    assert!(
        panic.contains("attempt to multiply with overflow")
            || panic.contains("invalid compact tensor layout")
            || panic.contains("from_buffer_col_major: data length")
            || (panic.contains("integer overflow") && panic.contains("tensor metadata")),
        "unexpected panic message: {panic}"
    );
}

fn constructor_panic_message(shape: Vec<usize>, len: usize) -> String {
    let panic = panic::catch_unwind(|| {
        let _ = cubecl_tensor_with_len(shape, len);
    })
    .expect_err("expected TypedTensor::from_buffer_col_major to reject invalid metadata");

    if let Some(message) = panic.downcast_ref::<String>() {
        return message.clone();
    }
    if let Some(message) = panic.downcast_ref::<&'static str>() {
        return (*message).to_string();
    }
    "non-string panic payload".to_string()
}

fn cubecl_tensor_with_len(shape: Vec<usize>, len: usize) -> TypedTensor<f32> {
    let handle = cubecl::server::Handle::new(
        StreamId::current(),
        (len * core::mem::size_of::<f32>()) as u64,
    );
    TypedTensor::from_buffer_col_major(
        shape,
        Buffer::Backend(std::sync::Arc::new(CubeclBuffer::new(handle, len))),
        Placement {
            memory_kind: MemoryKind::Device,
            device: Some(ComputeDevice {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal: 0,
            }),
        },
    )
}
