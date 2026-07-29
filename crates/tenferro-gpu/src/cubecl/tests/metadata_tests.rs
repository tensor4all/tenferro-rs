use cubecl::stream_id::StreamId;
use std::num::NonZeroUsize;
use std::panic;

use crate::cubecl::dispatch::{
    cubecl_shape_and_strides, typed_tensor_array_arg, typed_tensor_binding,
};
use crate::cubecl::{CudaBackend, CudaExtensionCache};
use crate::{
    Buffer, CubeclBuffer, DeviceId, DeviceKind, GpuBackendKind, MemoryKind, Placement, TypedTensor,
};
use tenferro_tensor::{CacheStats, Error, ErrorKind, ValidationError, ValidationKind};

#[test]
fn scalar_reduction_shape_stays_separate_from_cubecl_launch_metadata() {
    assert!(crate::cubecl::reduction_output_shape(&[2, 3], &[0, 1]).is_empty());
    assert_eq!(cubecl_shape_and_strides(&[]).unwrap(), (vec![1], vec![1]));
}

#[test]
fn cubecl_metadata_uses_dense_column_major_strides() {
    assert_eq!(cubecl_shape_and_strides(&[]).unwrap(), (vec![1], vec![1]));
    assert_eq!(
        cubecl_shape_and_strides(&[2, 3, 4]).unwrap(),
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
    assert_eq!(cache.stats().unwrap().entries, 0);
    assert_eq!(cache.stats().unwrap().retained_bytes, 0);

    let _usize = cache.get_or_try_init::<usize>(|| Ok(17)).unwrap();
    drop(_usize);
    let _usize_again = cache.get_or_try_init::<usize>(|| Ok(23)).unwrap();
    assert_eq!(*_usize_again, 17);
    drop(_usize_again);
    let _string = cache
        .get_or_try_init::<String>(|| Ok("gpu".to_string()))
        .unwrap();
    drop(_string);

    let stats = cache.stats().unwrap();
    assert_eq!(stats.entries, 2);
    assert!(stats.retained_bytes >= std::mem::size_of::<usize>());
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 2);

    cache.clear().unwrap();
    assert!(cache.is_empty().unwrap());
    let stats = cache.stats().unwrap();
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.retained_bytes, 0);
    assert_eq!(stats.clears, 1);
}

#[test]
fn cuda_extension_cache_methods_report_poisoned_lock() {
    let cache = CudaExtensionCache::new();
    let poisoned = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let _guard = cache.inner.lock().unwrap();
        panic!("poison cuda extension cache lock");
    }));
    assert!(poisoned.is_err());

    assert!(cache.is_empty().is_err());
    assert!(cache.stats().is_err());
    assert!(cache.clear().is_err());
    assert!(cache.max_entries().is_err());
    assert!(cache.max_retained_bytes().is_err());
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
    assert_eq!(cache.stats().unwrap().entries, 1);

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
fn cuda_extension_cache_has_configurable_retained_byte_bound() {
    let cache = CudaExtensionCache::with_max_entries(NonZeroUsize::new(8).unwrap());
    let byte_limit = std::mem::size_of::<usize>().max(std::mem::size_of::<String>());
    cache
        .set_max_retained_bytes(NonZeroUsize::new(byte_limit).unwrap())
        .unwrap();
    assert_eq!(cache.max_retained_bytes().unwrap().get(), byte_limit,);

    let value = cache.get_or_try_init::<usize>(|| Ok(17)).unwrap();
    assert_eq!(*value, 17);
    drop(value);

    let value = cache
        .get_or_try_init::<String>(|| Ok("gpu".to_string()))
        .unwrap();
    assert_eq!(value.as_str(), "gpu");
    drop(value);

    let stats = cache.stats().unwrap();
    assert_eq!(stats.entries, 1);
    assert!(stats.retained_bytes <= byte_limit);
    assert_eq!(stats.evictions, 1);
}

#[test]
fn cuda_extension_cache_allows_dynamic_retained_byte_updates() {
    let cache = CudaExtensionCache::with_max_entries(NonZeroUsize::new(8).unwrap());
    let value = cache.get_or_try_init::<usize>(|| Ok(17)).unwrap();
    assert_eq!(*value, 17);
    drop(value);

    cache.update_retained_bytes::<usize>(4096).unwrap();
    let stats = cache.stats().unwrap();
    assert_eq!(stats.entries, 1);
    assert_eq!(stats.retained_bytes, 4096);

    cache
        .set_max_retained_bytes(NonZeroUsize::new(1024).unwrap())
        .unwrap();
    assert!(cache.is_empty().unwrap());
    assert_eq!(cache.stats().unwrap().evictions, 1);
}

#[test]
fn cuda_backend_exposes_extension_cache_retained_byte_controls() {
    let _getter: fn(&CudaBackend) -> crate::Result<NonZeroUsize> =
        CudaBackend::cuda_extension_cache_max_retained_bytes;
    let _setter: fn(&CudaBackend, NonZeroUsize) -> crate::Result<()> =
        CudaBackend::set_cuda_extension_cache_max_retained_bytes;
    let _cutensor_stats: fn(&CudaBackend) -> crate::Result<CacheStats> =
        CudaBackend::cutensor_plan_cache_stats;
    let _cutensor_getter: fn(&CudaBackend) -> crate::Result<NonZeroUsize> =
        CudaBackend::cutensor_plan_cache_max_entries;
    let _cutensor_setter: fn(&CudaBackend, NonZeroUsize) -> crate::Result<()> =
        CudaBackend::set_cutensor_plan_cache_max_entries;
    let _cutensor_permutation_stats: fn(&CudaBackend) -> crate::Result<CacheStats> =
        CudaBackend::cutensor_permutation_plan_cache_stats;
    let _cutensor_permutation_getter: fn(&CudaBackend) -> crate::Result<NonZeroUsize> =
        CudaBackend::cutensor_permutation_plan_cache_max_entries;
    let _cutensor_permutation_setter: fn(&CudaBackend, NonZeroUsize) -> crate::Result<()> =
        CudaBackend::set_cutensor_permutation_plan_cache_max_entries;
}

#[test]
fn typed_tensor_binding_accepts_valid_backend_metadata() {
    let tensor = cubecl_tensor_with_len(vec![2, 3], 6).unwrap();

    typed_tensor_binding(&tensor, "metadata_test").unwrap();
    typed_tensor_array_arg(&tensor, "metadata_test").unwrap();
}

#[test]
fn from_buffer_col_major_rejects_backend_buffer_len_mismatch() {
    let error = cubecl_tensor_with_len(vec![2, 3], 5).unwrap_err();

    assert_eq!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::ShapeMismatch)
    );
    assert!(matches!(
        error,
        Error::Validation {
            op: "from_buffer_col_major",
            source: ValidationError::ShapeDataLengthMismatch {
                expected: 6,
                actual: 5,
            },
        }
    ));
}

#[test]
fn from_buffer_col_major_rejects_shape_product_overflow() {
    let error = cubecl_tensor_with_len(vec![usize::MAX, 2], 1).unwrap_err();

    assert_eq!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::InvalidArgument)
    );
    assert!(matches!(
        error,
        Error::Validation {
            op: "from_buffer_col_major",
            source: ValidationError::IntegerOverflow,
        }
    ));
}

fn cubecl_tensor_with_len(
    shape: Vec<usize>,
    len: usize,
) -> tenferro_tensor::Result<TypedTensor<f32>> {
    let handle = cubecl::server::Handle::new(
        StreamId::current(),
        (len * core::mem::size_of::<f32>()) as u64,
    );
    TypedTensor::from_buffer_col_major(
        shape,
        Buffer::Backend(std::sync::Arc::new(CubeclBuffer::new(handle, len, 0))),
        Placement {
            memory_kind: MemoryKind::Device,
            device: Some(DeviceId {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal: 0,
            }),
            cpu_affinity: None,
        },
    )
}
