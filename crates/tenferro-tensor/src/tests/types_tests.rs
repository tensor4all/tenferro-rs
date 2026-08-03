use std::{error::Error as _, sync::Arc};

use num_complex::{Complex32, Complex64};

use crate::types::{
    col_major_strides, flat_to_multi, AllocationDomainId, AllocationId, BackendBuffer, Buffer,
    BufferHandle, CpuDomainId, DType, DeviceId, DeviceKind, GpuBackendKind, HostAccessError,
    HostReadGuard, HostWriteGuard, MemoryKind, Placement, Rank, StridedSliceSpec, Tensor,
    TensorBufferRef, TensorBufferRefMut, TensorLayout, TensorRank, TensorRead, TensorScalar,
    TensorValue, TensorView, TensorViewMut, TensorWrite, TypedTensor, TypedTensorView,
    TypedTensorViewMut, TypedTensorWrite,
};
use crate::{
    Error, ErrorKind, ShapeMismatch, ShapeVec, SliceConfig, ValidationError, ValidationKind,
};

mod strided_dynamic;

#[derive(Debug)]
struct HostAccessibleBuffer {
    data: std::sync::Mutex<Vec<f32>>,
    domain: AllocationDomainId,
    allocation: AllocationId,
}

impl BackendBuffer<f32> for HostAccessibleBuffer {
    fn backend_family(&self) -> &'static str {
        "test-host-access"
    }

    fn len(&self) -> usize {
        self.data.lock().unwrap().len()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn allocation_domain(&self) -> Option<AllocationDomainId> {
        Some(self.domain)
    }

    fn allocation_id(&self) -> Option<AllocationId> {
        Some(self.allocation)
    }

    fn map_read(&self) -> Result<HostReadGuard<'_, f32>, HostAccessError> {
        Ok(HostReadGuard::new(self.data.lock().unwrap()))
    }

    fn map_write(&self) -> Result<HostWriteGuard<'_, f32>, HostAccessError> {
        let mut guard = self.data.lock().unwrap();
        let len = guard.len();
        Ok(HostWriteGuard::new(len, move |source| {
            guard.copy_from_slice(source);
            Ok(())
        }))
    }
}

#[test]
fn backend_host_access_guards_preserve_domain_identity_and_writeback() {
    let domain = AllocationDomainId::fresh();
    let allocation = AllocationId::from_backend_id(17);
    let buffer = HostAccessibleBuffer {
        data: std::sync::Mutex::new(vec![1.0, 2.0]),
        domain,
        allocation,
    };

    assert_eq!(buffer.allocation_domain(), Some(domain));
    assert_eq!(buffer.allocation_id(), Some(allocation));
    assert_eq!(&*buffer.map_read().unwrap(), &[1.0, 2.0]);
    {
        let mut mapped = buffer.map_write().unwrap();
        mapped.copy_from_slice(&[1.0, 4.0]).unwrap();
    }
    assert_eq!(&*buffer.map_read().unwrap(), &[1.0, 4.0]);
}

#[test]
fn opaque_backend_buffers_reject_host_mapping_with_a_typed_error() {
    let buffer = BufferHandle::<f32>::new_with_len(1, 2);

    assert_eq!(buffer.allocation_domain(), None);
    assert_eq!(buffer.allocation_id(), None);
    assert!(matches!(
        buffer.map_read(),
        Err(HostAccessError::Unsupported { backend: "opaque" })
    ));
}

#[test]
fn placement_default_is_the_canonical_host_placement() {
    assert_eq!(
        Placement::default(),
        Placement {
            memory_kind: MemoryKind::UnpinnedHost,
            device: None,
            cpu_affinity: None,
        }
    );
}

#[test]
fn cpu_domain_id_is_stable_caller_metadata() {
    let id = CpuDomainId::new(17);
    assert_eq!(id.as_u64(), 17);
    assert_eq!(id, CpuDomainId::new(17));
}

#[test]
fn cpu_affinity_is_not_a_device_boundary() {
    let placement = Placement {
        memory_kind: MemoryKind::UnpinnedHost,
        device: None,
        cpu_affinity: Some(CpuDomainId::new(3)),
    };
    assert_eq!(placement.cpu_affinity, Some(CpuDomainId::new(3)));
    assert!(placement.device.is_none());
    assert!(Placement::default().cpu_affinity.is_none());
}

#[test]
fn tensor_view_mut_constructors_cover_every_dtype() {
    let shape = [1];
    let mut f32_data = [1.0_f32];
    let mut f64_data = [1.0_f64];
    let mut i32_data = [1_i32];
    let mut i64_data = [1_i64];
    let mut bool_data = [true];
    let mut c32_data = [Complex32::new(1.0, 0.0)];
    let mut c64_data = [Complex64::new(1.0, 0.0)];

    assert_eq!(
        TensorViewMut::f32(&shape, &mut f32_data).unwrap().dtype(),
        DType::F32
    );
    assert_eq!(
        TensorViewMut::f64(&shape, &mut f64_data).unwrap().dtype(),
        DType::F64
    );
    assert_eq!(
        TensorViewMut::i32(&shape, &mut i32_data).unwrap().dtype(),
        DType::I32
    );
    assert_eq!(
        TensorViewMut::i64(&shape, &mut i64_data).unwrap().dtype(),
        DType::I64
    );
    assert_eq!(
        TensorViewMut::bool(&shape, &mut bool_data).unwrap().dtype(),
        DType::Bool
    );
    assert_eq!(
        TensorViewMut::c32(&shape, &mut c32_data).unwrap().dtype(),
        DType::C32
    );
    assert_eq!(
        TensorViewMut::c64(&shape, &mut c64_data).unwrap().dtype(),
        DType::C64
    );
}

#[test]
fn tensor_error_preserves_shared_validation_source() {
    let err = Error::validation(
        "add",
        ShapeMismatch::IncompatibleShapes {
            lhs: ShapeVec::from_vec(vec![2]),
            rhs: ShapeVec::from_vec(vec![3]),
        }
        .into(),
    );

    assert_eq!(
        err.kind(),
        ErrorKind::Validation(ValidationKind::ShapeMismatch)
    );
    assert!(err.source().is_some());
    assert!(matches!(
        err,
        Error::Validation {
            op: "add",
            source: ValidationError::ShapeMismatch(_),
        }
    ));
}

#[test]
fn typed_backend_source_is_not_formatted_away() {
    let err = Error::backend_source("load", std::io::Error::other("device read failed"));
    assert_eq!(err.kind(), ErrorKind::BackendFailure);
    assert!(err.source().is_some());
}

#[test]
fn typed_io_source_is_not_classified_as_backend_failure() {
    let err = Error::io_source("load", std::io::Error::other("file read failed"));
    assert_eq!(err.kind(), ErrorKind::Io);
    assert!(err.source().is_some());
}

#[test]
fn runtime_state_source_is_not_classified_as_backend_failure() {
    let err = Error::runtime_state_source(
        "execute",
        std::io::Error::other("executor state is unavailable"),
    );
    assert_eq!(err.kind(), ErrorKind::RuntimeState);
    assert!(err.source().is_some());
}

#[test]
fn unsupported_operation_has_a_distinct_coarse_classification() {
    let err = Error::unsupported("full_piv_lu", "backend has no implementation");
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert!(matches!(
        err,
        Error::Unsupported {
            op: "full_piv_lu",
            ..
        }
    ));
}

#[derive(Debug)]
struct NonCloneElement;

fn tensor_scalar_roundtrip<T>(shape: Vec<usize>, data: Vec<T>)
where
    T: TensorScalar + PartialEq + std::fmt::Debug,
{
    let tensor = T::into_tensor(shape.clone(), data.clone()).unwrap();

    assert_eq!(tensor.shape(), shape.as_slice());
    assert_eq!(T::as_slice(&tensor).unwrap(), data.as_slice());
    assert_eq!(tensor.as_slice::<T>().unwrap(), data.as_slice());

    let mut mutable = tensor.duplicate().unwrap();
    assert_eq!(T::as_slice_mut(&mut mutable).unwrap(), data.as_slice());

    let typed = T::into_typed(tensor).unwrap();
    assert_eq!(typed.as_slice().unwrap(), data.as_slice());
}

macro_rules! tensor_scalar_roundtrip_test {
    ($name:ident, $ty:ty, $shape:expr, $data:expr) => {
        #[test]
        fn $name() {
            tensor_scalar_roundtrip::<$ty>($shape, $data);
        }
    };
}

#[test]
fn tensor_scalar_tensor_read_covers_all_variants() {
    let f64s = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![1.0]).unwrap();
    assert_eq!(f64::tensor_read(&f64s).dtype(), DType::F64);

    let f32s = TypedTensor::<f32>::from_vec_col_major(vec![1], vec![1.0]).unwrap();
    assert_eq!(f32::tensor_read(&f32s).dtype(), DType::F32);

    let i64s = TypedTensor::<i64>::from_vec_col_major(vec![1], vec![1]).unwrap();
    assert_eq!(i64::tensor_read(&i64s).dtype(), DType::I64);

    let i32s = TypedTensor::<i32>::from_vec_col_major(vec![1], vec![1]).unwrap();
    assert_eq!(i32::tensor_read(&i32s).dtype(), DType::I32);

    let bools = TypedTensor::<bool>::from_vec_col_major(vec![1], vec![true]).unwrap();
    assert_eq!(bool::tensor_read(&bools).dtype(), DType::Bool);

    let c64s =
        TypedTensor::<Complex64>::from_vec_col_major(vec![1], vec![Complex64::new(1.0, 2.0)])
            .unwrap();
    assert_eq!(Complex64::tensor_read(&c64s).dtype(), DType::C64);

    let c32s =
        TypedTensor::<Complex32>::from_vec_col_major(vec![1], vec![Complex32::new(1.0, 2.0)])
            .unwrap();
    assert_eq!(Complex32::tensor_read(&c32s).dtype(), DType::C32);
}

#[test]
fn tensor_value_keeps_owned_transpose_as_view() {
    let tensor = Arc::new(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap(),
    );
    let value = TensorValue::from_tensor((*tensor).duplicate().unwrap());
    let transposed = value.transpose_view([1, 0]).unwrap();

    assert_eq!(transposed.shape(), &[3, 2]);
    match transposed.tensor_read() {
        TensorRead::View(view) => assert_eq!(view.shape(), &[3, 2]),
        TensorRead::Tensor(_) => panic!("owned transpose should be exposed as a view"),
    }
}

#[test]
fn tensor_owned_view_and_tensor_value_cover_lazy_accessors_and_errors() {
    let base = Arc::new(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );
    let owned =
        TensorValue::from_parts((*base).duplicate().unwrap(), vec![2, 3], vec![1, 2], 0).unwrap();

    assert_eq!(owned.dtype(), DType::F64);
    assert_eq!(owned.shape(), &[2, 3]);
    assert_eq!(owned.strides(), &[1, 2]);
    assert_eq!(owned.offset(), 0);
    match owned.tensor_view() {
        TensorView::F64(view) => assert_eq!(view.get(&[1, 2]), Some(&6.0)),
        other => panic!("expected f64 view, got {other:?}"),
    }
    assert!(matches!(owned.tensor_read(), TensorRead::Tensor(_)));

    let explicit =
        TensorValue::from_parts((*base).duplicate().unwrap(), vec![3, 2], vec![2, 1], 0).unwrap();
    assert_eq!(explicit.shape(), &[3, 2]);
    assert_eq!(explicit.strides(), &[2, 1]);

    let transposed = owned.transpose_view([1, 0]).unwrap();
    assert_eq!(transposed.shape(), &[3, 2]);
    assert_eq!(transposed.strides(), &[2, 1]);

    let reshaped = owned.reshape_view([6]).unwrap();
    assert_eq!(reshaped.shape(), &[6]);

    let sliced = owned
        .slice_view(&SliceConfig {
            starts: vec![0, 1],
            limits: vec![2, 3],
            strides: vec![1, 1],
        })
        .unwrap();
    assert_eq!(sliced.shape(), &[2, 2]);
    match sliced.tensor_view() {
        TensorView::F64(view) => {
            assert_eq!(view.get(&[0, 0]), Some(&3.0));
            assert_eq!(view.get(&[1, 1]), Some(&6.0));
        }
        other => panic!("expected f64 view, got {other:?}"),
    }

    let vector =
        TensorValue::from_parts((*base).duplicate().unwrap(), vec![2], vec![1], 0).unwrap();
    let broadcast = vector.broadcast_in_dim_view([2, 3], [0]).unwrap();
    assert_eq!(broadcast.shape(), &[2, 3]);
    assert_eq!(broadcast.strides(), &[1, 0]);

    for bad_config in [
        SliceConfig {
            starts: vec![0],
            limits: vec![2, 3],
            strides: vec![1, 1],
        },
        SliceConfig {
            starts: vec![0, 0],
            limits: vec![2],
            strides: vec![1, 1],
        },
        SliceConfig {
            starts: vec![0, 0],
            limits: vec![2, 3],
            strides: vec![1],
        },
    ] {
        assert!(matches!(
            owned.slice_view(&bad_config),
            Err(Error::Validation {
                source: ValidationError::RankMismatch { .. },
                ..
            })
        ));
    }

    let value = TensorValue::from_tensor((*base).duplicate().unwrap());
    assert!(value.as_tensor().is_some());
    assert_eq!(value.dtype(), DType::F64);
    assert_eq!(value.shape(), &[2, 3]);
    assert!(matches!(value.tensor_read(), TensorRead::Tensor(_)));

    let view_value = value.transpose_view([1, 0]).unwrap();
    assert!(view_value.as_tensor().is_none());
    assert_eq!(view_value.dtype(), DType::F64);
    assert_eq!(view_value.shape(), &[3, 2]);
    let original_order = view_value.transpose_view([1, 0]).unwrap();
    assert_eq!(original_order.shape(), &[2, 3]);

    let compact_view = value.reshape_view([6]).unwrap();
    assert_eq!(compact_view.reshape_view([2, 3]).unwrap().shape(), &[2, 3]);
    let sliced_value = compact_view
        .slice_view(&SliceConfig {
            starts: vec![1],
            limits: vec![5],
            strides: vec![2],
        })
        .unwrap();
    assert_eq!(sliced_value.shape(), &[2]);
    assert_eq!(
        compact_view
            .broadcast_in_dim_view([6, 2], [0])
            .unwrap()
            .shape(),
        &[6, 2]
    );
}

#[test]
fn col_major_helpers_cover_scalar_and_higher_rank_shapes() {
    assert_eq!(col_major_strides(&[]).unwrap(), Vec::<isize>::new());
    assert_eq!(col_major_strides(&[2, 3, 4]).unwrap(), vec![1, 2, 6]);
    assert_eq!(col_major_strides(&[2, 3, 4]).unwrap(), vec![1, 2, 6]);
    assert!(matches!(
        col_major_strides(&[usize::MAX, 2]),
        Err(Error::Validation { .. })
    ));

    let mut scalar_idx = [];
    flat_to_multi(0, &[], &mut scalar_idx);

    let mut idx = [0usize; 3];
    flat_to_multi(13, &[2, 3, 4], &mut idx);
    assert_eq!(idx, [1, 0, 2]);

    let mut zero_extent_idx = [usize::MAX; 2];
    flat_to_multi(0, &[0, 5], &mut zero_extent_idx);
    assert_eq!(zero_extent_idx, [0, 0]);
}

#[test]
fn device_model_has_typed_hashable_device_ids() {
    use std::collections::HashSet;

    let cuda0 = DeviceId {
        kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
        ordinal: 0,
    };
    let cuda1 = DeviceId {
        kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
        ordinal: 1,
    };
    let cpu0 = DeviceId {
        kind: DeviceKind::Cpu,
        ordinal: 0,
    };

    let mut devices = HashSet::new();
    devices.insert(cuda0.clone());
    devices.insert(cuda1.clone());

    assert!(devices.contains(&cuda0));
    assert!(devices.contains(&cuda1));
    assert!(!devices.contains(&cpu0));
    assert_ne!(cuda0, cpu0);
}

#[test]
fn device_model_has_first_class_webgpu_backend_kind() {
    let cuda = DeviceId {
        kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
        ordinal: 0,
    };
    let webgpu = DeviceId {
        kind: DeviceKind::Gpu(GpuBackendKind::WebGpu),
        ordinal: 0,
    };

    assert_ne!(webgpu, cuda);
    assert_eq!(format!("{:?}", webgpu.kind), "Gpu(WebGpu)");
}

#[test]
fn default_placement_is_unpinned_host_without_device() {
    let tensor = TypedTensor::<f64>::zeros(vec![2]).unwrap();

    assert_eq!(tensor.placement().memory_kind, MemoryKind::UnpinnedHost);
    assert_eq!(tensor.placement().device, None);
}

#[test]
fn gpu_placement_is_metadata_only() {
    let placement = Placement {
        memory_kind: MemoryKind::Device,
        device: Some(DeviceId {
            kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
            ordinal: 0,
        }),
        cpu_affinity: None,
    };

    assert_eq!(placement.memory_kind, MemoryKind::Device);
    assert_eq!(
        placement.device.as_ref().map(|device| &device.kind),
        Some(&DeviceKind::Gpu(GpuBackendKind::Cuda))
    );
}

#[test]
fn typed_tensor_rank_zero_access_and_mutation_work() {
    let mut tensor = TypedTensor::<f64>::from_vec_col_major(vec![], vec![7.0_f64]).unwrap();

    assert_eq!(tensor.n_elements(), 1);
    assert_eq!(tensor.linear_offset(&[]).unwrap(), 0);
    assert_eq!(*tensor.get(&[]).unwrap(), 7.0);

    *tensor.get_mut(&[]).unwrap() = 9.5;
    assert_eq!(tensor.host_data().unwrap(), &[9.5]);
}

#[test]
fn typed_tensor_static_rank_constructs_compact_layout() {
    let tensor =
        TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.layout().strides(), &[1, 2]);
    assert!(tensor.layout().is_compact_col_major().unwrap());
}

#[test]
fn typed_tensor_try_into_rank_validates_rank() {
    let tensor = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![0.0; 6]).unwrap();

    let err = tensor.try_into_rank::<3>().unwrap_err();

    assert!(matches!(
        err,
        Error::Validation {
            source: ValidationError::RankMismatch {
                expected: 3,
                actual: 2,
            },
            ..
        }
    ));
}

#[test]
fn typed_tensor_try_into_rank_preserves_backend_buffer_and_placement() {
    let placement = Placement {
        memory_kind: MemoryKind::Device,
        device: Some(DeviceId {
            kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
            ordinal: 2,
        }),
        cpu_affinity: None,
    };
    let tensor = TypedTensor::<f64>::from_buffer_col_major(
        vec![2, 3],
        Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(42, 6))),
        placement.clone(),
    )
    .unwrap();

    let ranked = tensor.try_into_rank::<2>().unwrap();

    assert_eq!(ranked.shape(), &[2, 3]);
    assert_eq!(ranked.layout().strides(), &[1, 2]);
    assert_eq!(ranked.placement(), &placement);
    match ranked.buffer() {
        Buffer::Backend(buffer) => {
            assert_eq!(buffer.len(), 6);
            buffer
                .as_any()
                .downcast_ref::<BufferHandle<f64>>()
                .expect("opaque test handle");
        }
        Buffer::Host(_) => panic!("expected backend buffer"),
    }
}

#[test]
fn typed_tensor_as_view_preserves_rank_and_layout() {
    let tensor =
        TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let view = tensor.as_view();
    assert_eq!(view.shape(), &[2, 2]);
    assert_eq!(view.strides(), &[1, 2]);
    assert_eq!(view.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn typed_tensor_view_backend_as_slice_returns_runtime_state() {
    let tensor = TypedTensor::<f64>::from_buffer_col_major(
        vec![2],
        Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(77, 2))),
        Placement {
            memory_kind: MemoryKind::Device,
            device: Some(DeviceId {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal: 0,
            }),
            cpu_affinity: None,
        },
    )
    .unwrap();

    let err = tensor.as_view().as_slice().unwrap_err();

    assert!(matches!(
        err,
        Error::RuntimeState {
            op: "TypedTensorView::as_slice",
            ..
        }
    ));
    assert!(err.to_string().contains("download explicitly first"));
}

#[test]
fn typed_tensor_view_as_slice_rejects_non_contiguous_layout() {
    let data = [1_i32, 2, 3, 4];
    let view = TypedTensorView::from_slice(vec![2], vec![2], 0, &data).unwrap();

    let err = view.as_slice().unwrap_err();

    assert!(matches!(
        err,
        Error::Validation {
            op: "TypedTensorView::as_slice",
            source: ValidationError::InvalidArgument { .. },
            ..
        }
    ));
    assert!(err.to_string().contains("not contiguous column-major"));
}

#[test]
fn typed_tensor_view_transpose_is_metadata_only() {
    let tensor =
        TypedTensor::<i32, Rank<2>>::from_vec_col_major([2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
    let view = tensor.as_view().transpose_view([1, 0]).unwrap();
    assert_eq!(view.shape(), &[3, 2]);
    assert_eq!(view.strides(), &[2, 1]);
    assert_eq!(view.get(&[2, 1]), Some(&6));
}

#[test]
fn mutable_typed_tensor_view_rejects_overlapping_layout() {
    let mut data = vec![1_i32, 2, 3, 4];
    assert!(TypedTensorViewMut::from_slice(vec![2, 2], vec![1, 1], 0, &mut data).is_err());
}

#[test]
fn typed_tensor_owned_layout_is_always_compact() {
    let tensor = TypedTensor::<i32>::from_vec_col_major(vec![3], vec![1, 2, 3]).unwrap();
    assert_eq!(
        tensor.layout(),
        &TensorLayout::compact(vec![3].into()).unwrap()
    );
}

#[test]
fn typed_tensor_backend_buffer_layout_length_mismatch_returns_error() {
    let err = TypedTensor::<f64>::from_buffer_col_major(
        vec![1],
        Buffer::Backend(Arc::new(BufferHandle::<f64>::new(9))),
        Placement {
            memory_kind: MemoryKind::Device,
            device: Some(DeviceId {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal: 0,
            }),
            cpu_affinity: None,
        },
    )
    .unwrap_err();

    assert!(matches!(err, Error::Validation { .. }));
}

#[test]
fn typed_tensor_static_rank_shape_conversion_reports_rank_mismatch() {
    let err = <Rank<2> as TensorRank>::shape_from_vec(vec![2, 3, 4].into()).unwrap_err();

    assert!(matches!(
        err,
        tenferro_tensor_core::ValidationError::RankMismatch {
            expected: 2,
            actual: 3
        }
    ));
}

#[test]
fn tensor_owned_export_returns_column_major_buffer() {
    let data = vec![1.0_f64, 2.0, 3.0, 4.0];
    let ptr = data.as_ptr();
    let tensor = Tensor::from_vec_col_major(vec![2, 2], data).unwrap();

    let (shape, out) = tensor.into_vec_col_major::<f64>().unwrap();

    assert_eq!(shape, vec![2, 2]);
    assert_eq!(out.as_ptr(), ptr);
}

#[test]
fn tensor_owned_export_reports_dtype_mismatch() {
    let err = Tensor::from_vec_col_major(vec![1], vec![1.0_f64])
        .unwrap()
        .into_vec_col_major::<f32>()
        .unwrap_err();

    assert!(matches!(
        err,
        Error::Validation {
            source: ValidationError::DTypeMismatch { .. },
            ..
        }
    ));
}

#[test]
fn backend_buffer_handle_metadata_and_host_export_errors_are_explicit() {
    let handle: Arc<dyn crate::types::BackendBuffer<f64>> =
        Arc::new(BufferHandle::<f64>::new_with_len(42, 2));

    assert_eq!(format!("{handle:?}"), "BufferHandle { id: 42 }");
    assert_eq!(handle.backend_family(), "opaque");
    assert_eq!(handle.len(), 2);
    assert!(!handle.is_empty());
    assert!(handle.as_any().is::<BufferHandle<f64>>());

    let tensor = TypedTensor::<f64>::from_buffer_col_major(
        vec![2],
        Buffer::Backend(Arc::clone(&handle)),
        Placement {
            memory_kind: MemoryKind::Device,
            device: Some(DeviceId {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal: 0,
            }),
            cpu_affinity: None,
        },
    )
    .unwrap();
    let col_err = tensor.into_vec_col_major().unwrap_err();

    assert!(matches!(col_err, Error::RuntimeState { .. }));
    assert!(col_err
        .to_string()
        .contains("backend buffers cannot be exported"));
}

#[test]
fn tensor_buffer_refs_cover_backend_metadata() {
    let read_handle: Arc<dyn crate::types::BackendBuffer<f64>> =
        Arc::new(BufferHandle::<f64>::new_with_len(7, 0));
    let read_ref = TensorBufferRef::Backend(Arc::clone(&read_handle));
    let cloned_read_ref = read_ref.clone();

    assert_eq!(cloned_read_ref.len(), 0);
    assert!(cloned_read_ref.is_empty());

    let write_handle: Arc<dyn crate::types::BackendBuffer<i32>> =
        Arc::new(BufferHandle::<i32>::new_with_len(8, 2));
    let write_ref = TensorBufferRefMut::Backend(write_handle);

    assert_eq!(write_ref.len(), 2);
    assert!(!write_ref.is_empty());
}

#[test]
fn typed_tensor_col_major_constructor_matches_logical_matrix() {
    let tensor =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
            .unwrap();

    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.as_slice().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(tensor.get(&[0, 0]).unwrap(), &1.0);
    assert_eq!(tensor.get(&[1, 0]).unwrap(), &4.0);
    assert_eq!(tensor.get(&[0, 2]).unwrap(), &3.0);
    assert_eq!(tensor.get(&[1, 2]).unwrap(), &6.0);
}

#[test]
fn typed_tensor_exports_col_major_order() {
    let tensor =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
            .unwrap();

    let (shape, col) = tensor.into_vec_col_major().unwrap();
    assert_eq!(shape, vec![2, 3]);
    assert_eq!(col, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn tensor_col_major_roundtrips_dynamic_dtype() {
    let tensor = Tensor::from_vec_col_major(vec![2, 2], vec![1_i64, 3, 2, 4]).unwrap();

    assert_eq!(tensor.as_slice::<i64>().unwrap(), &[1, 3, 2, 4]);
    assert_eq!(
        tensor.into_vec_col_major::<i64>().unwrap(),
        (vec![2, 2], vec![1, 3, 2, 4]),
    );
}

#[test]
fn col_major_typed_tensor_uses_logical_indices() {
    let tensor =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0])
            .unwrap();

    assert_eq!(*tensor.get(&[0, 2]).unwrap(), 3.0);
    assert_eq!(*tensor.get(&[1, 0]).unwrap(), 4.0);
}

#[test]
fn typed_tensor_linear_offsets_cover_higher_rank_shapes() {
    let tensor = TypedTensor::<f64>::zeros(vec![2, 3, 5]).unwrap();

    assert_eq!(
        tensor.linear_offset(&[1, 2, 4]).unwrap(),
        1 + 2 * 2 + 4 * 2 * 3
    );
}

#[test]
fn typed_tensor_iterators_follow_physical_buffer_order() {
    let tensor =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0])
            .unwrap();
    assert_eq!(
        tensor.iter().unwrap().copied().collect::<Vec<_>>(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );

    let mut tensor = TypedTensor::<i64>::from_vec_col_major(vec![3], vec![1_i64, 2, 3]).unwrap();
    for value in tensor.iter_mut().unwrap() {
        *value *= 2;
    }
    assert_eq!(tensor.as_slice().unwrap(), &[2, 4, 6]);
}

#[test]
fn tensor_iterators_are_typed_by_requested_scalar() {
    let mut tensor = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();

    assert_eq!(
        tensor.iter::<f64>().unwrap().copied().collect::<Vec<_>>(),
        vec![1.0, 2.0, 3.0]
    );
    assert!(tensor.iter::<f32>().is_err());

    for value in tensor.iter_mut::<f64>().unwrap() {
        *value += 1.0;
    }

    assert_eq!(tensor.as_slice::<f64>().unwrap(), &[2.0, 3.0, 4.0]);
    assert!(tensor.as_slice_mut::<f32>().is_err());
}

#[test]
fn typed_tensor_checked_and_unchecked_accessors_work() {
    let mut tensor =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0])
            .unwrap();

    assert_eq!(tensor.host_data().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(tensor.get(&[1, 2]).unwrap(), &6.0);
    assert!(tensor.get(&[2, 0]).is_err());
    assert!(tensor.get(&[0]).is_err());

    *tensor.get_mut(&[0, 1]).unwrap() = 20.0;
    assert_eq!(tensor.host_data().unwrap()[2], 20.0);

    tensor.host_data_mut().unwrap()[0] = 10.0;
    assert_eq!(unsafe { *tensor.get_unchecked(&[0, 0]).unwrap() }, 10.0);

    unsafe {
        *tensor.get_unchecked_mut(&[1, 0]).unwrap() = 40.0;
    }
    assert_eq!(tensor.get(&[1, 0]).unwrap(), &40.0);
}

#[test]
fn typed_tensor_rank_fixed_accessors_use_column_major_offsets() {
    let mut col2 =
        TypedTensor::<i64>::from_vec_col_major(vec![2, 3], vec![1_i64, 2, 3, 4, 5, 6]).unwrap();

    assert_eq!(col2.linear_offset2(1, 2).unwrap(), 5);
    assert_eq!(*col2.get2(1, 2).unwrap(), 6);
    *col2.get_mut2(0, 1).unwrap() = 30;
    assert_eq!(col2.host_data().unwrap()[2], 30);

    let mut col3 =
        TypedTensor::<i64>::from_vec_col_major(vec![2, 3, 2], (0_i64..12).collect()).unwrap();

    assert_eq!(col3.linear_offset3(1, 2, 1).unwrap(), 11);
    assert_eq!(*col3.get3(1, 2, 1).unwrap(), 11);
    *col3.get_mut3(0, 1, 1).unwrap() = 60;
    assert_eq!(col3.host_data().unwrap()[8], 60);
}

#[test]
fn tensor_checked_accessors_are_typed_and_use_physical_order() {
    let mut tensor =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();

    assert_eq!(
        tensor.as_slice::<f64>().unwrap(),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
    assert_eq!(tensor.linear_offset(&[1, 0]).unwrap(), 1);
    assert_eq!(tensor.linear_offset2(1, 0).unwrap(), 1);
    assert_eq!(tensor.get::<f64>(&[1, 2]).unwrap(), &6.0);
    assert!(matches!(
        tensor.get::<f32>(&[1, 2]),
        Err(Error::Validation {
            source: ValidationError::DTypeMismatch { .. },
            ..
        })
    ));
    assert!(matches!(
        tensor.get::<f64>(&[2, 0]),
        Err(Error::Validation { .. })
    ));

    tensor.as_slice_mut::<f64>().unwrap()[0] = 10.0;
    *tensor.get_mut::<f64>(&[0, 1]).unwrap() = 20.0;

    assert_eq!(
        unsafe { *tensor.get_unchecked::<f64>(&[0, 0]).unwrap() },
        10.0
    );
    unsafe {
        *tensor.get_unchecked_mut::<f64>(&[1, 0]).unwrap() = 40.0;
    }

    assert_eq!(
        tensor.as_slice::<f64>().unwrap(),
        &[10.0, 40.0, 20.0, 5.0, 3.0, 6.0]
    );
}

#[test]
fn typed_tensor_accessors_report_length_and_indexing_errors() {
    assert!(matches!(
        TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0]),
        Err(Error::Validation { .. })
    ));

    let tensor = TypedTensor::<f64>::zeros(vec![2, 3]).unwrap();
    assert!(matches!(
        tensor.linear_offset(&[0]),
        Err(Error::Validation {
            source: ValidationError::RankMismatch { .. },
            ..
        })
    ));

    assert!(matches!(
        tensor.linear_offset(&[2, 0]),
        Err(Error::Validation { .. })
    ));
}

#[test]
fn typed_tensor_apis_report_shape_arithmetic_errors() {
    assert!(matches!(
        TypedTensor::<f64>::from_vec_col_major(vec![usize::MAX, 2], Vec::new()),
        Err(Error::Validation { .. })
    ));
    assert!(matches!(
        TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0]),
        Err(Error::Validation { .. })
    ));
    assert!(matches!(
        TypedTensor::<f64>::zeros(vec![usize::MAX, 2]),
        Err(Error::Validation { .. })
    ));
    assert!(matches!(
        TypedTensor::<f64>::ones(vec![usize::MAX, 2]),
        Err(Error::Validation { .. })
    ));

    let tensor = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![0.0; 6]).unwrap();
    assert_eq!(tensor.n_elements(), 6);
    assert_eq!(tensor.linear_offset(&[1, 2]).unwrap(), 5);
    assert!(matches!(
        tensor.linear_offset(&[2, 0]),
        Err(Error::Validation { .. })
    ));
}

#[test]
fn tensor_constructors_report_shape_arithmetic_errors() {
    assert!(matches!(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0]),
        Err(Error::Validation { .. })
    ));
    assert!(matches!(
        Tensor::from_vec_col_major(vec![usize::MAX, 2], Vec::<f64>::new()),
        Err(Error::Validation { .. })
    ));
}

#[test]
fn typed_tensor_ranked_accessors_return_errors_instead_of_panicking() {
    let tensor = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![0.0; 6]).unwrap();

    assert_eq!(tensor.linear_offset2(1, 2).unwrap(), 5);
    assert!(matches!(
        tensor.linear_offset2(10, 0),
        Err(Error::Validation { .. })
    ));
    assert!(matches!(tensor.get2(10, 0), Err(Error::Validation { .. })));

    let rank3 = TypedTensor::<f64>::from_vec_col_major(vec![2, 3, 2], vec![0.0; 12]).unwrap();
    assert_eq!(rank3.linear_offset3(1, 2, 1).unwrap(), 11);
    assert!(matches!(
        rank3.linear_offset3(1, 2, 99),
        Err(Error::Validation { .. })
    ));
    assert!(matches!(
        rank3.get3(1, 2, 99),
        Err(Error::Validation { .. })
    ));
}

#[test]
fn tensor_ranked_linear_offsets_return_errors_instead_of_panicking() {
    let tensor = Tensor::from_vec_col_major(vec![2, 3], vec![0.0_f64; 6]).unwrap();

    assert_eq!(tensor.linear_offset2(1, 2).unwrap(), 5);
    assert!(matches!(
        tensor.linear_offset2(10, 0),
        Err(Error::Validation { .. })
    ));

    let rank3 = Tensor::from_vec_col_major(vec![2, 3, 2], vec![0.0_f64; 12]).unwrap();
    assert_eq!(rank3.linear_offset3(1, 2, 1).unwrap(), 11);
    assert!(matches!(
        rank3.linear_offset3(1, 2, 99),
        Err(Error::Validation { .. })
    ));
}

#[test]
fn backend_buffers_return_errors_when_host_access_is_requested() {
    let placement = Placement {
        memory_kind: MemoryKind::Other("backend".to_string()),
        device: Some(DeviceId {
            kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
            ordinal: 0,
        }),
        cpu_affinity: None,
    };
    let tensor = TypedTensor::<f64>::from_buffer_col_major(
        vec![1],
        Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(7, 1))),
        placement.clone(),
    )
    .unwrap();
    assert_eq!(tensor.placement().device.as_ref().unwrap().ordinal, 0);

    assert!(tensor.host_data().is_err());
    let erased = Tensor::F64(tensor);
    assert!(erased.as_slice::<f64>().is_err());
    assert!(erased.get::<f64>(&[0]).is_err());

    let mut mutable_tensor = TypedTensor::<f64>::from_buffer_col_major(
        vec![1],
        Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(8, 1))),
        placement,
    )
    .unwrap();
    assert!(mutable_tensor.host_data_mut().is_err());
    let mut erased_mut = Tensor::F64(mutable_tensor);
    assert!(erased_mut.as_slice_mut::<f64>().is_err());
    assert!(erased_mut.get_mut::<f64>(&[0]).is_err());
}

#[test]
fn backend_mutable_views_keep_metadata_paths_without_host_access() {
    let placement = Placement {
        memory_kind: MemoryKind::Device,
        device: Some(DeviceId {
            kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
            ordinal: 1,
        }),
        cpu_affinity: None,
    };
    let mut tensor = TypedTensor::<i32>::from_buffer_col_major(
        vec![2, 2],
        Buffer::Backend(Arc::new(BufferHandle::<i32>::new_with_len(91, 4))),
        placement.clone(),
    )
    .unwrap();

    assert!(tensor.buffer().is_backend());
    assert!(!tensor.buffer().is_empty());

    {
        let mut view = tensor.as_view_mut();
        assert_eq!(view.shape(), &[2, 2]);
        assert_eq!(view.strides(), &[1, 2]);
        assert_eq!(view.n_elements(), 4);
        assert_eq!(view.placement(), &placement);
        assert!(view.backend_buffer().is_some());
        assert_eq!(view.get(&[0, 0]), None);
        assert_eq!(view.get_mut(&[0, 0]), None);
        assert!(view.host_storage().is_err());
        assert!(view.host_storage_mut().is_err());

        let read = view.as_read_only();
        assert!(read.backend_buffer().is_some());
        assert_eq!(read.get(&[0, 0]), None);
        assert!(read.host_storage().is_err());
    }

    {
        let mut view = tensor.as_view_mut();
        let sliced = view
            .try_slice(&[
                StridedSliceSpec::all(),
                StridedSliceSpec::new(0, Some(1), 1),
            ])
            .unwrap();
        assert_eq!(sliced.shape(), &[2, 1]);
        assert!(sliced.backend_buffer().is_some());
    }

    {
        let mut view = tensor.as_view_mut();
        let reshaped = view.try_reshape(&[4]).unwrap();
        assert_eq!(reshaped.shape(), &[4]);
        assert!(reshaped.backend_buffer().is_some());
    }

    {
        let view = tensor.as_view_mut();
        let transposed = view.transpose_view([1, 0]).unwrap();
        assert_eq!(transposed.strides(), &[2, 1]);
        assert!(transposed.backend_buffer().is_some());
    }

    {
        let view = tensor.as_view_mut();
        let read = view.into_read_only();
        assert_eq!(read.shape(), &[2, 2]);
        assert!(read.backend_buffer().is_some());
    }
}

#[test]
fn backend_multi_slice_mut_returns_none_instead_of_touching_host_memory() {
    let mut tensor = TypedTensor::<i32>::from_buffer_col_major(
        vec![4],
        Buffer::Backend(Arc::new(BufferHandle::<i32>::new_with_len(92, 4))),
        Placement {
            memory_kind: MemoryKind::Device,
            device: Some(DeviceId {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal: 1,
            }),
            cpu_affinity: None,
        },
    )
    .unwrap();
    let mut view = tensor.as_view_mut();

    assert!(view
        .try_multi_slice_mut(
            &[StridedSliceSpec::new(0, Some(2), 1)],
            &[StridedSliceSpec::new(2, Some(4), 1)]
        )
        .unwrap()
        .is_none());
    assert!(view
        .try_multi_slice_mut(
            &[StridedSliceSpec::new(0, Some(0), 1)],
            &[StridedSliceSpec::new(2, Some(4), 1)]
        )
        .unwrap()
        .is_none());
    assert!(view
        .try_multi_slice_mut(
            &[StridedSliceSpec::new(0, Some(2), 1)],
            &[StridedSliceSpec::new(4, Some(4), 1)]
        )
        .unwrap()
        .is_none());
}

#[test]
fn typed_tensor_metadata_accessors_accept_non_clone_elements() {
    let placement = Placement {
        memory_kind: MemoryKind::Other("backend".to_string()),
        device: None,
        cpu_affinity: None,
    };
    let tensor = TypedTensor::<NonCloneElement>::from_buffer_col_major(
        vec![2, 3],
        Buffer::Backend(Arc::new(BufferHandle::<NonCloneElement>::new_with_len(
            9, 6,
        ))),
        placement,
    )
    .unwrap();

    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.rank(), 2);
    assert_eq!(tensor.n_elements(), 6);
    assert_eq!(tensor.layout().strides(), &[1, 2]);
    assert!(tensor.into_layout().is_compact_col_major().unwrap());
}

#[test]
fn tensor_shape_and_dtype_cover_all_variants() {
    let f32_tensor =
        Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap());
    let f64_tensor = Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap());
    let i32_tensor =
        Tensor::I32(TypedTensor::from_vec_col_major(vec![2], vec![1_i32, -2]).unwrap());
    let c32_tensor = Tensor::C32(
        TypedTensor::from_vec_col_major(vec![1], vec![Complex32::new(1.0, -2.0)]).unwrap(),
    );
    let c64_tensor = Tensor::C64(
        TypedTensor::from_vec_col_major(vec![1, 1], vec![Complex64::new(-3.0, 4.0)]).unwrap(),
    );
    let bool_tensor =
        Tensor::Bool(TypedTensor::from_vec_col_major(vec![2], vec![true, false]).unwrap());

    assert_eq!(f32_tensor.shape(), &[2]);
    assert_eq!(f32_tensor.dtype(), DType::F32);
    assert_eq!(f64_tensor.shape(), &[] as &[usize]);
    assert_eq!(f64_tensor.dtype(), DType::F64);
    assert_eq!(i32_tensor.shape(), &[2]);
    assert_eq!(i32_tensor.dtype(), DType::I32);
    assert_eq!(bool_tensor.shape(), &[2]);
    assert_eq!(bool_tensor.dtype(), DType::Bool);
    assert_eq!(c32_tensor.shape(), &[1]);
    assert_eq!(c32_tensor.dtype(), DType::C32);
    assert_eq!(c64_tensor.shape(), &[1, 1]);
    assert_eq!(c64_tensor.dtype(), DType::C64);
}

#[test]
fn typed_tensor_view_validates_shape_and_exposes_slice() {
    let shape = [2, 2];
    let data = [1.0_f64, 2.0, 3.0, 4.0];
    let view = TypedTensorView::from_col_major(&shape, &data).unwrap();

    assert_eq!(view.shape(), &[2, 2]);
    assert_eq!(view.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

    let err = TypedTensorView::from_col_major(&shape, &data[..3]).unwrap_err();
    assert!(matches!(err, Error::Validation { .. }));
}

#[test]
fn tensor_view_covers_dtype_and_shape() {
    let f32_data = [1.0_f32, 2.0];
    let f64_data = [1.0_f64, 2.0];
    let i32_data = [1_i32, 2];
    let i64_data = [1_i64, 2];
    let bool_data = [true, false];
    let c32_data = [Complex32::new(1.0, -1.0), Complex32::new(2.0, 0.5)];
    let c64_data = [Complex64::new(1.0, -1.0), Complex64::new(2.0, 0.5)];
    let shape = [2usize];

    let views = [
        TensorView::f32(&shape, &f32_data).unwrap(),
        TensorView::f64(&shape, &f64_data).unwrap(),
        TensorView::i32(&shape, &i32_data).unwrap(),
        TensorView::i64(&shape, &i64_data).unwrap(),
        TensorView::bool(&shape, &bool_data).unwrap(),
        TensorView::c32(&shape, &c32_data).unwrap(),
        TensorView::c64(&shape, &c64_data).unwrap(),
    ];
    let dtypes = [
        DType::F32,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::Bool,
        DType::C32,
        DType::C64,
    ];

    for (view, dtype) in views.iter().zip(dtypes) {
        assert_eq!(view.dtype(), dtype);
        assert_eq!(view.shape(), &[2]);
    }
}

#[test]
fn typed_tensor_view_sliced_layouts_preserve_indexed_access() {
    let row_major = [1_i32, 2, 3, 4, 5, 6];
    let view = TypedTensorView::from_slice([2, 3], [3, 1], 0, &row_major).unwrap();

    assert_eq!(view.shape(), &[2, 3]);
    assert_eq!(view.strides(), &[3, 1]);
    assert_eq!(view.get(&[1, 2]), Some(&6));
    assert_eq!(view.get(&[0, 0]), Some(&1));
    assert_eq!(view.get(&[1, 2]), Some(&6));

    let transposed = view.transpose_view([1, 0]).unwrap();
    assert_eq!(transposed.shape(), &[3, 2]);
    assert_eq!(transposed.get(&[2, 1]), Some(&6));

    let reversed_cols = view.try_slice_axis(1, StridedSliceSpec::reverse()).unwrap();
    assert_eq!(reversed_cols.strides(), &[3, -1]);
    assert_eq!(reversed_cols.get(&[0, 0]), Some(&3));
    assert_eq!(reversed_cols.get(&[1, 2]), Some(&4));

    let every_other_col = view
        .try_slice(&[StridedSliceSpec::all(), StridedSliceSpec::new(0, None, 2)])
        .unwrap();
    assert_eq!(every_other_col.get(&[0, 1]), Some(&3));
    assert_eq!(every_other_col.get(&[1, 1]), Some(&6));
    assert!(view.try_reshape(&[6]).is_err());

    let col_major = [1_i32, 4, 2, 5, 3, 6];
    let contiguous = TypedTensorView::from_col_major(&[2, 3], &col_major).unwrap();
    assert_eq!(contiguous.try_reshape(&[6]).unwrap().strides(), &[1]);
}

#[test]
fn tensor_view_covers_strided_i32_and_bool() {
    let i32_data = [1_i32, 2, 3, 4];
    let i32_view =
        TensorView::I32(TypedTensorView::from_slice([2, 2], [2, 1], 0, &i32_data).unwrap());
    assert_eq!(i32_view.dtype(), DType::I32);
    assert_eq!(i32_view.strides(), &[2, 1]);

    let bool_data = [false, true, true];
    let bool_view =
        TensorView::Bool(TypedTensorView::from_slice([3], [-1], 2, &bool_data).unwrap());
    assert_eq!(bool_view.dtype(), DType::Bool);
    assert_eq!(bool_view.strides(), &[-1]);
}

#[test]
fn strided_tensor_view_mut_updates_sliced_host_layouts() {
    let mut row_major = [1_i32, 2, 3, 4, 5, 6];
    let mut view = TypedTensorViewMut::from_slice([2, 3], [3, 1], 0, &mut row_major).unwrap();

    assert_eq!(view.shape(), &[2, 3]);
    assert_eq!(view.strides(), &[3, 1]);
    assert_eq!(view.get(&[1, 2]), Some(&6));

    *view.get_mut(&[1, 2]).unwrap() = 60;
    assert_eq!(view.get(&[1, 2]), Some(&60));

    {
        let mut transposed = view.transpose_view([1, 0]).unwrap();
        *transposed.get_mut(&[2, 1]).unwrap() = 600;
    }
    let mut view = TypedTensorViewMut::from_slice([2, 3], [3, 1], 0, &mut row_major).unwrap();
    assert_eq!(view.get(&[1, 2]), Some(&600));

    {
        let mut reversed_cols = view.try_slice_axis(1, StridedSliceSpec::reverse()).unwrap();
        assert_eq!(reversed_cols.strides(), &[3, -1]);
        *reversed_cols.get_mut(&[0, 0]).unwrap() = 30;
    }
    assert_eq!(view.get(&[0, 2]), Some(&30));

    assert_eq!(view.get(&[0, 2]), Some(&30));
    assert_eq!(view.get(&[1, 2]), Some(&600));
}

#[test]
fn strided_tensor_view_mut_rejects_aliasing_layouts() {
    let data = [1_i32, 2, 3, 4];
    assert!(TypedTensorView::from_slice([2, 2], [1, 1], 0, &data).is_ok());

    let mut data = [1_i32, 2, 3, 4];
    let err = TypedTensorViewMut::from_slice([2, 2], [1, 1], 0, &mut data).unwrap_err();
    assert!(matches!(err, Error::Validation { .. }));

    let mut data = [1_i32, 2];
    assert!(TypedTensorViewMut::from_slice([2], [0], 0, &mut data).is_err());

    let mut data = [1_i32, 2, 3];
    let mut reversed = TypedTensorViewMut::from_slice([3], [-1], 2, &mut data).unwrap();
    *reversed.get_mut(&[2]).unwrap() = 10;
    assert_eq!(reversed.host_storage().unwrap(), &[10, 2, 3]);

    let mut data = [1_i32, 2];
    let singleton_zero_stride =
        TypedTensorViewMut::from_slice([1, 2], [0, 1], 0, &mut data).unwrap();
    assert_eq!(singleton_zero_stride.shape(), &[1, 2]);
}

#[test]
fn strided_tensor_view_mut_multi_slice_returns_option() {
    let mut data = [1_i32, 2, 3, 4, 5, 6];
    let mut view = TypedTensorViewMut::from_slice([6], [1], 0, &mut data).unwrap();

    {
        let (mut left, mut right) = view
            .try_multi_slice_mut(
                &[StridedSliceSpec::new(0, Some(3), 1)],
                &[StridedSliceSpec::new(3, Some(6), 1)],
            )
            .unwrap()
            .unwrap();
        *left.get_mut(&[2]).unwrap() = 30;
        *right.get_mut(&[0]).unwrap() = 40;
    }
    assert_eq!(view.host_storage().unwrap(), &[1, 2, 30, 40, 5, 6]);

    assert!(view
        .try_multi_slice_mut(
            &[StridedSliceSpec::new(0, Some(4), 1)],
            &[StridedSliceSpec::new(3, Some(6), 1)],
        )
        .unwrap()
        .is_none());

    assert!(view
        .try_multi_slice_mut(
            &[StridedSliceSpec::new(0, Some(2), 0)],
            &[StridedSliceSpec::new(3, Some(6), 1)],
        )
        .is_err());
}

#[test]
fn typed_tensor_view_mut_covers_i32_and_bool_strided_layouts() {
    let mut i32_data = [1_i32, 2, 3, 4];
    let mut i32_view = TypedTensorViewMut::from_slice([2, 2], [2, 1], 0, &mut i32_data).unwrap();
    *i32_view.get_mut(&[1, 1]).unwrap() = 40;
    assert_eq!(i32_view.get(&[0, 0]), Some(&1));
    assert_eq!(i32_view.get(&[1, 1]), Some(&40));

    let mut bool_data = [false, true, true];
    let mut bool_view = TypedTensorViewMut::from_slice([3], [-1], 2, &mut bool_data).unwrap();
    *bool_view.get_mut(&[2]).unwrap() = true;
    assert_eq!(bool_view.get(&[0]), Some(&true));
    assert_eq!(bool_view.get(&[2]), Some(&true));
}

#[test]
fn typed_tensor_view_mut_multi_slice_returns_option_for_strided_layouts() {
    let mut data = [1_i32, 2, 3, 4];
    let mut view = TypedTensorViewMut::from_slice([4], [1], 0, &mut data).unwrap();
    {
        let (left, right) = view
            .try_multi_slice_mut(
                &[StridedSliceSpec::new(0, Some(2), 1)],
                &[StridedSliceSpec::new(2, Some(4), 1)],
            )
            .unwrap()
            .unwrap();
        assert_eq!(left.shape(), &[2]);
        assert_eq!(right.shape(), &[2]);
    }

    assert!(view
        .try_multi_slice_mut(
            &[StridedSliceSpec::new(0, Some(3), 1)],
            &[StridedSliceSpec::new(2, Some(4), 1)],
        )
        .unwrap()
        .is_none());
}

#[test]
fn strided_tensor_view_validation_covers_error_edges() {
    let data = [1_i32, 2, 3];

    let empty = TypedTensorView::from_slice([0, 3], [1, 0], 3, &data).unwrap();
    assert_eq!(empty.n_elements(), 0);
    assert_eq!(empty.try_reshape(&[0]).unwrap().shape(), &[0]);
    let reversed_empty = empty
        .try_slice_axis(0, StridedSliceSpec::reverse())
        .unwrap();
    assert_eq!(reversed_empty.shape(), &[0, 3]);
    assert_eq!(empty.get(&[0, 0]), None);
    assert!(matches!(
        TypedTensorView::<i32>::from_slice([0], [1], 4, &data),
        Err(Error::Validation { .. })
    ));

    assert!(matches!(
        TypedTensorView::<i32>::from_slice([2], [1, 1], 0, &data),
        Err(Error::Validation {
            source: ValidationError::RankMismatch { .. },
            ..
        })
    ));
    assert!(matches!(
        TypedTensorView::<i32>::from_slice([2], [-1], 0, &data[..1]),
        Err(Error::Validation { .. })
    ));
    assert!(matches!(
        TypedTensorView::<i32>::from_slice([2], [2], 0, &data[..1]),
        Err(Error::Validation { .. })
    ));

    let view = TypedTensorView::from_slice([3], [1], 0, &data).unwrap();
    assert_eq!(view.get(&[1]), Some(&2));
    assert_eq!(view.linear_offset(&[0, 0]), None);
    assert_eq!(view.linear_offset(&[3]), None);

    assert!(matches!(
        TypedTensorView::<i32>::from_slice([usize::MAX, 2], [1, 1], 0, &[]),
        Err(Error::Validation { .. })
    ));
    assert!(matches!(
        TypedTensorView::<i32>::from_slice(
            [usize::MAX / 2 + 1, usize::MAX / 2 + 1],
            [0, 0],
            0,
            &[7],
        ),
        Err(Error::Validation { .. })
    ));
    let empty_after_large_prefix =
        TypedTensorView::from_slice([2, usize::MAX, 0], [0, 0, 1], 0, &[7]).unwrap();
    assert_eq!(empty_after_large_prefix.n_elements(), 0);
    assert!(matches!(
        TypedTensorView::<i32>::from_slice([3], [isize::MAX], 0, &[]),
        Err(Error::Validation { .. })
    ));
    assert!(matches!(
        TypedTensorView::<i32>::from_slice([2, 2], [isize::MAX, 1], 0, &[]),
        Err(Error::Validation { .. })
    ));
}

#[test]
fn typed_tensor_view_slice_transpose_and_reshape_cover_boundaries() {
    let data = [1_i32, 2, 3, 4, 5, 6];
    let view = TypedTensorView::from_slice([2, 3], [3, 1], 0, &data).unwrap();

    assert!(matches!(
        view.transpose_view([0]),
        Err(Error::Validation {
            source: ValidationError::InvalidPermutationLength { .. },
            ..
        })
    ));
    assert!(matches!(
        view.transpose_view([2, 0]),
        Err(Error::Validation {
            source: ValidationError::AxisOutOfBounds { .. },
            ..
        })
    ));
    assert!(matches!(
        view.transpose_view([0, 0]),
        Err(Error::Validation {
            source: ValidationError::DuplicateAxis { .. },
            ..
        })
    ));

    assert!(matches!(
        view.try_slice(&[StridedSliceSpec::all()]),
        Err(Error::Validation {
            source: ValidationError::RankMismatch { .. },
            ..
        })
    ));
    assert!(matches!(
        view.try_slice_axis(2, StridedSliceSpec::all()),
        Err(Error::Validation {
            source: ValidationError::AxisOutOfBounds { .. },
            ..
        })
    ));
    assert!(matches!(
        view.try_slice(&[StridedSliceSpec::all(), StridedSliceSpec::new(0, None, 0)]),
        Err(Error::Validation { .. })
    ));
    assert!(matches!(
        view.try_slice(&[StridedSliceSpec::all(), StridedSliceSpec::new(-4, None, 1)]),
        Err(Error::Validation { .. })
    ));
    assert!(matches!(
        view.try_slice(&[
            StridedSliceSpec::all(),
            StridedSliceSpec::new(0, Some(4), 1)
        ]),
        Err(Error::Validation { .. })
    ));

    let empty = view
        .try_slice_axis(1, StridedSliceSpec::new(2, Some(1), 1))
        .unwrap();
    assert_eq!(empty.shape(), &[2, 0]);
    assert_eq!(empty.n_elements(), 0);

    assert!(matches!(
        view.try_reshape(&[5]),
        Err(Error::Validation { .. })
    ));
    assert!(matches!(
        TypedTensorView::<i32>::from_col_major(&[isize::MAX as usize, 2, 2], &[]),
        Err(Error::Validation { .. })
    ));

    let scalar = TypedTensorView::from_col_major(&[], &data[..1]).unwrap();
    assert_eq!(scalar.shape(), &[] as &[usize]);
    assert_eq!(scalar.strides(), &[] as &[isize]);
    assert_eq!(scalar.get(&[]), Some(&1));

    let singleton_axis = TypedTensorView::from_slice([1, 3], [99, 1], 0, &data).unwrap();
    assert_eq!(singleton_axis.try_reshape(&[3]).unwrap().strides(), &[1]);
}

#[test]
fn strided_tensor_view_mut_multi_slice_covers_empty_reverse_and_conservative_cases() {
    let mut data = [0_i32, 1, 2, 3, 4, 5];
    let mut view = TypedTensorViewMut::from_slice([6], [1], 0, &mut data).unwrap();
    {
        let (mut high, mut low) = view
            .try_multi_slice_mut(
                &[StridedSliceSpec::new(4, Some(6), 1)],
                &[StridedSliceSpec::new(1, Some(3), 1)],
            )
            .unwrap()
            .unwrap();
        *high.get_mut(&[0]).unwrap() = 40;
        *low.get_mut(&[1]).unwrap() = 20;
    }
    assert_eq!(view.host_storage().unwrap(), &[0, 1, 20, 3, 40, 5]);

    let mut data = [0_i32, 1, 2, 3];
    let mut view = TypedTensorViewMut::from_slice([4], [1], 0, &mut data).unwrap();
    {
        let (empty, mut right) = view
            .try_multi_slice_mut(
                &[StridedSliceSpec::new(0, Some(0), 1)],
                &[StridedSliceSpec::new(2, Some(4), 1)],
            )
            .unwrap()
            .unwrap();
        assert_eq!(empty.n_elements(), 0);
        *right.get_mut(&[0]).unwrap() = 20;
    }
    assert_eq!(view.host_storage().unwrap(), &[0, 1, 20, 3]);

    let mut data = [0_i32, 1, 2, 3];
    let mut view = TypedTensorViewMut::from_slice([4], [1], 0, &mut data).unwrap();
    {
        let (mut left, empty) = view
            .try_multi_slice_mut(
                &[StridedSliceSpec::new(0, Some(2), 1)],
                &[StridedSliceSpec::new(4, Some(4), 1)],
            )
            .unwrap()
            .unwrap();
        assert_eq!(empty.n_elements(), 0);
        *left.get_mut(&[1]).unwrap() = 10;
    }
    assert_eq!(view.host_storage().unwrap(), &[0, 10, 2, 3]);

    let mut data = [0_i32, 1, 2, 3];
    let mut view = TypedTensorViewMut::from_slice([4], [1], 0, &mut data).unwrap();
    let (empty_left, empty_right) = view
        .try_multi_slice_mut(
            &[StridedSliceSpec::new(0, Some(0), 1)],
            &[StridedSliceSpec::new(4, Some(4), 1)],
        )
        .unwrap()
        .unwrap();
    assert_eq!(empty_left.n_elements(), 0);
    assert_eq!(empty_right.n_elements(), 0);

    let mut data = [0_i32, 1, 2, 3, 4, 5];
    let mut view = TypedTensorViewMut::from_slice([6], [1], 0, &mut data).unwrap();
    {
        let (mut reversed_high, mut low) = view
            .try_multi_slice_mut(
                &[StridedSliceSpec::new(3, Some(6), -1)],
                &[StridedSliceSpec::new(0, Some(3), 1)],
            )
            .unwrap()
            .unwrap();
        assert_eq!(reversed_high.get(&[0]), Some(&5));
        *reversed_high.get_mut(&[2]).unwrap() = 30;
        *low.get_mut(&[2]).unwrap() = 20;
    }
    assert_eq!(view.host_storage().unwrap(), &[0, 1, 20, 30, 4, 5]);

    let mut data = [0_i32, 1, 2, 3, 4, 5];
    let mut view = TypedTensorViewMut::from_slice([6], [1], 0, &mut data).unwrap();
    assert!(view
        .try_multi_slice_mut(
            &[StridedSliceSpec::new(0, Some(6), 2)],
            &[StridedSliceSpec::new(1, Some(6), 2)],
        )
        .unwrap()
        .is_none());
}

#[test]
fn tensor_read_wraps_owned_tensor_or_borrowed_view() {
    let tensor = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    let read_tensor = TensorRead::from_tensor(&tensor);

    assert_eq!(read_tensor.dtype(), DType::F64);
    assert_eq!(read_tensor.shape(), &[2]);
    assert!(read_tensor.as_tensor().is_some());

    let shape = [2usize];
    let data = [5.0_f64, 6.0];
    let read_view = TensorRead::from_view(TensorView::f64(&shape, &data).unwrap());

    assert_eq!(read_view.dtype(), DType::F64);
    assert_eq!(read_view.shape(), &[2]);
    assert!(read_view.as_tensor().is_none());
    assert_eq!(read_view.layout_linear_offset(&[1]).unwrap(), 1);
}

#[test]
fn tensor_write_wraps_owned_tensor_or_borrowed_mutable_view() {
    let mut tensor = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    {
        let write_tensor = TensorWrite::from_tensor(&mut tensor);

        assert_eq!(write_tensor.dtype(), DType::F64);
        assert_eq!(write_tensor.shape(), &[2]);
        assert_eq!(write_tensor.strides().unwrap().as_slice(), &[1]);
        assert_eq!(write_tensor.offset(), 0);
        assert!(write_tensor.is_col_major_contiguous().unwrap());
    }
    assert_eq!(tensor.as_slice::<f64>().unwrap(), &[3.0, 4.0]);

    let mut data = [0.0_f64, 10.0, 0.0, 20.0, 0.0];
    {
        let view =
            TensorViewMut::F64(TypedTensorViewMut::from_slice([2], [2], 1, &mut data).unwrap());
        let write_view = TensorWrite::from_view(view);

        assert_eq!(write_view.dtype(), DType::F64);
        assert_eq!(write_view.shape(), &[2]);
        assert_eq!(write_view.strides().unwrap().as_slice(), &[2]);
        assert_eq!(write_view.offset(), 1);
        assert!(!write_view.is_col_major_contiguous().unwrap());
    }
    assert_eq!(data, [0.0, 10.0, 0.0, 20.0, 0.0]);
}

#[test]
fn typed_tensor_write_wraps_owned_tensor_or_borrowed_mutable_view() {
    let mut tensor = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 4.0]).unwrap();
    {
        let typed_write = TypedTensorWrite::from_tensor(&mut tensor);
        let write = typed_write.into_tensor_write();

        assert_eq!(write.dtype(), DType::F64);
        assert_eq!(write.shape(), &[2]);
    }
    assert_eq!(tensor.as_slice().unwrap(), &[3.0, 4.0]);

    let mut data = [0.0_f64, 10.0, 0.0, 20.0, 0.0];
    {
        let view = TypedTensorViewMut::from_slice([2], [2], 1, &mut data).unwrap();
        let typed_write = TypedTensorWrite::from_view(view);
        let write = typed_write.into_tensor_write();

        assert_eq!(write.dtype(), DType::F64);
        assert_eq!(write.shape(), &[2]);
        assert_eq!(write.strides().unwrap(), [2]);
        assert_eq!(write.offset(), 1);
    }
    assert_eq!(data, [0.0, 10.0, 0.0, 20.0, 0.0]);
}

#[test]
fn layout_helpers_report_shape_strides_and_offset() {
    let mut data = [1_i32, 2, 3, 4, 5];
    let view = TypedTensorViewMut::from_slice([2], [2], 1, &mut data).unwrap();

    assert_eq!(view.layout_linear_offset(&[1]).unwrap(), 3);
    assert!(!view.is_col_major_contiguous().unwrap());

    let summary = view.layout_summary();
    assert!(summary.contains("shape=[2]"));
    assert!(summary.contains("strides=[2]"));
    assert!(summary.contains("offset=1"));

    let err = view.assert_col_major_contiguous().unwrap_err();
    let message = err.to_string();
    assert!(message.contains("shape=[2]"));
    assert!(message.contains("strides=[2]"));
    assert!(message.contains("offset=1"));
}

#[test]
fn tensor_view_layout_helpers_cover_all_dtype_variants() {
    macro_rules! assert_view {
        ($view:expr, $dtype:expr) => {{
            let view = $view;
            assert_eq!(view.dtype(), $dtype);
            assert_eq!(view.shape(), &[2]);
            assert_eq!(view.strides(), &[1]);
            assert_eq!(view.offset(), 0);
            assert_eq!(view.layout_linear_offset(&[1]).unwrap(), 1);
            assert!(view.is_col_major_contiguous().unwrap());
            assert!(view.layout_summary().contains("shape=[2]"));
            view.assert_col_major_contiguous().unwrap();
        }};
    }

    let f32_data = [1.0_f32, 2.0];
    assert_view!(TensorView::f32(&[2], &f32_data).unwrap(), DType::F32);

    let f64_data = [1.0_f64, 2.0];
    assert_view!(TensorView::f64(&[2], &f64_data).unwrap(), DType::F64);

    let i32_data = [1_i32, 2];
    assert_view!(TensorView::i32(&[2], &i32_data).unwrap(), DType::I32);

    let i64_data = [1_i64, 2];
    assert_view!(TensorView::i64(&[2], &i64_data).unwrap(), DType::I64);

    let bool_data = [true, false];
    assert_view!(TensorView::bool(&[2], &bool_data).unwrap(), DType::Bool);

    let c32_data = [Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
    assert_view!(TensorView::c32(&[2], &c32_data).unwrap(), DType::C32);

    let c64_data = [Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
    assert_view!(TensorView::c64(&[2], &c64_data).unwrap(), DType::C64);
}

#[test]
fn tensor_view_mut_layout_helpers_cover_all_dtype_variants() {
    macro_rules! assert_view_mut {
        ($variant:ident, $ty:ty, $dtype:expr, $initial:expr, $replacement:expr) => {{
            let mut data: [$ty; 2] = $initial;
            {
                let typed = TypedTensorViewMut::from_slice([2], [1], 0, &mut data).unwrap();
                let view = TensorViewMut::$variant(typed);
                assert_eq!(view.dtype(), $dtype);
                assert_eq!(view.shape(), &[2]);
                assert_eq!(view.strides(), &[1]);
                assert_eq!(view.offset(), 0);
                assert_eq!(view.layout_linear_offset(&[1]).unwrap(), 1);
                assert!(view.is_col_major_contiguous().unwrap());
                assert!(view.layout_summary().contains("shape=[2]"));
                view.assert_col_major_contiguous().unwrap();

                let read = view.as_read_only();
                assert_eq!(read.dtype(), $dtype);
                assert_eq!(read.shape(), &[2]);
            }
            assert_eq!(data, $initial);
        }};
    }

    assert_view_mut!(F32, f32, DType::F32, [1.0, 2.0], [3.0, 4.0]);
    assert_view_mut!(F64, f64, DType::F64, [1.0, 2.0], [3.0, 4.0]);
    assert_view_mut!(I32, i32, DType::I32, [1, 2], [3, 4]);
    assert_view_mut!(I64, i64, DType::I64, [1, 2], [3, 4]);
    assert_view_mut!(Bool, bool, DType::Bool, [true, false], [false, true]);
    assert_view_mut!(
        C32,
        Complex32,
        DType::C32,
        [Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)],
        [Complex32::new(5.0, 6.0), Complex32::new(7.0, 8.0)]
    );
    assert_view_mut!(
        C64,
        Complex64,
        DType::C64,
        [Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
        [Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)]
    );
}

#[test]
fn tensor_read_and_write_layout_helpers_cover_tensor_and_view_paths() {
    let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let read_tensor = TensorRead::from_tensor(&tensor);
    assert_eq!(read_tensor.strides().unwrap(), vec![1]);
    assert_eq!(read_tensor.offset(), 0);
    assert_eq!(read_tensor.layout_linear_offset(&[1]).unwrap(), 1);
    assert!(read_tensor.is_col_major_contiguous().unwrap());
    assert!(read_tensor.layout_summary().contains("offset=0"));
    read_tensor.assert_col_major_contiguous().unwrap();

    let view_data = [10.0_f64, 20.0, 30.0];
    let read_view = TensorRead::from_view(TensorView::F64(
        TypedTensorView::from_slice([2], [2], 0, &view_data).unwrap(),
    ));
    assert_eq!(read_view.strides().unwrap(), vec![2]);
    assert_eq!(read_view.offset(), 0);
    assert_eq!(read_view.layout_linear_offset(&[1]).unwrap(), 2);
    assert!(!read_view.is_col_major_contiguous().unwrap());
    assert!(read_view.assert_col_major_contiguous().is_err());

    let mut write_tensor_dst = Tensor::from_vec_col_major(vec![2], vec![0.0_f64, 0.0]).unwrap();
    {
        let write_tensor = TensorWrite::from_tensor(&mut write_tensor_dst);
        assert_eq!(write_tensor.strides().unwrap(), vec![1]);
        assert_eq!(write_tensor.offset(), 0);
        assert_eq!(write_tensor.layout_linear_offset(&[1]).unwrap(), 1);
        assert!(write_tensor.is_col_major_contiguous().unwrap());
        assert!(write_tensor.layout_summary().contains("shape=[2]"));
        write_tensor.assert_col_major_contiguous().unwrap();
    }
    assert_eq!(write_tensor_dst.as_slice::<f64>().unwrap(), &[0.0, 0.0]);

    let mut write_view_data = [0.0_f64, 10.0, 0.0, 20.0];
    {
        let write_view = TensorWrite::from_view(TensorViewMut::F64(
            TypedTensorViewMut::from_slice([2], [2], 1, &mut write_view_data).unwrap(),
        ));
        assert_eq!(write_view.strides().unwrap(), vec![2]);
        assert_eq!(write_view.offset(), 1);
        assert_eq!(write_view.layout_linear_offset(&[1]).unwrap(), 3);
        assert!(!write_view.is_col_major_contiguous().unwrap());
        assert!(write_view.assert_col_major_contiguous().is_err());
    }
    assert_eq!(write_view_data, [0.0, 10.0, 0.0, 20.0]);
}

tensor_scalar_roundtrip_test!(
    tensor_scalar_roundtrip_f32,
    f32,
    vec![2],
    vec![1.25_f32, -2.5_f32]
);

tensor_scalar_roundtrip_test!(
    tensor_scalar_roundtrip_f64,
    f64,
    vec![2, 1],
    vec![1.25_f64, -2.5_f64]
);

tensor_scalar_roundtrip_test!(
    tensor_scalar_roundtrip_i32,
    i32,
    vec![2],
    vec![1_i32, -2_i32]
);

tensor_scalar_roundtrip_test!(
    tensor_scalar_roundtrip_i64,
    i64,
    vec![2],
    vec![1_i64, -2_i64]
);

tensor_scalar_roundtrip_test!(
    tensor_scalar_roundtrip_bool,
    bool,
    vec![2],
    vec![true, false]
);

tensor_scalar_roundtrip_test!(
    tensor_scalar_roundtrip_c32,
    Complex32,
    vec![2],
    vec![Complex32::new(1.0, -0.5), Complex32::new(-2.0, 3.5)]
);

tensor_scalar_roundtrip_test!(
    tensor_scalar_roundtrip_c64,
    Complex64,
    vec![1, 2],
    vec![Complex64::new(1.0, -0.5), Complex64::new(-2.0, 3.5)]
);

#[test]
fn tensor_as_slice_returns_none_for_dtype_mismatch() {
    let tensor = <f64 as TensorScalar>::into_tensor(vec![2], vec![1.0, 2.0]).unwrap();

    assert!(tensor.as_slice::<f32>().is_err());
}

#[test]
fn tensor_write_as_read_borrows_tensor_and_view_outputs() {
    let mut tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let write = TensorWrite::from_tensor(&mut tensor);
    let read = write.as_read();
    assert_eq!(read.dtype(), DType::F64);
    assert_eq!(read.shape(), &[2]);
    assert_eq!(
        read.as_tensor().unwrap().as_slice::<f64>().unwrap(),
        &[1.0, 2.0]
    );

    let mut data = [0.0_f64, 3.0, 4.0, 0.0];
    let view = TensorViewMut::F64(TypedTensorViewMut::from_slice([2], [1], 1, &mut data).unwrap());
    let write = TensorWrite::from_view(view);
    let read = write.as_read();
    assert_eq!(read.dtype(), DType::F64);
    assert_eq!(read.shape(), &[2]);
    assert!(read.as_tensor().is_none());
    assert_eq!(read.layout_linear_offset(&[1]).unwrap(), 2);
}

#[test]
fn memory_kind_variants_are_constructible() {
    let kinds = [
        MemoryKind::Device,
        MemoryKind::PinnedHost,
        MemoryKind::UnpinnedHost,
        MemoryKind::Other("scratch".to_string()),
    ];

    assert_eq!(kinds.len(), 4);
    assert!(matches!(kinds[0], MemoryKind::Device));
    assert!(matches!(kinds[1], MemoryKind::PinnedHost));
    assert!(matches!(kinds[2], MemoryKind::UnpinnedHost));
    assert!(matches!(kinds[3], MemoryKind::Other(_)));
}

#[test]
fn runtime_error_formats_include_op_name() {
    let err = Error::validation(
        "dot_general",
        ValidationError::AxisOutOfBounds { axis: 2, rank: 1 },
    );

    assert!(err.to_string().contains("dot_general"));
}

#[test]
fn n_elements_invariant_checks_do_not_use_unsafe_unreachable() {
    let source = include_str!("../types.rs");
    assert!(
        !source.contains("unreachable_unchecked"),
        "n_elements invariant checks must never use unsafe unreachable_unchecked"
    );
    assert!(
        source.contains("TypedTensor compact shape is validated at construction"),
        "n_elements panic path should document the constructor-validation invariant"
    );
}

fn backend_tensor_f64(id: u64, len: usize) -> TypedTensor<f64> {
    TypedTensor::<f64>::from_buffer_col_major(
        vec![len],
        Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(id, len))),
        Placement {
            memory_kind: MemoryKind::Device,
            device: Some(DeviceId {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal: 0,
            }),
            cpu_affinity: None,
        },
    )
    .unwrap()
}

#[test]
fn backend_region_view_exposes_layout_and_shared_buffer() {
    let tensor = backend_tensor_f64(90, 16);
    let view = tensor
        .backend_region_view(vec![2, 3], vec![1, 4], 5)
        .unwrap();

    assert_eq!(view.shape(), &[2, 3]);
    assert_eq!(view.strides(), &[1, 4]);
    assert_eq!(view.offset(), 5);
    assert_eq!(view.placement(), tensor.placement());
    let buffer = view.backend_buffer().expect("backend buffer");
    assert_eq!(buffer.len(), 16);
}

#[test]
fn backend_region_view_mut_exposes_layout_and_shared_buffer() {
    let mut tensor = backend_tensor_f64(91, 16);
    let placement = tensor.placement().clone();
    let view = tensor
        .backend_region_view_mut(vec![2, 2], vec![1, 4], 10)
        .unwrap();

    assert_eq!(view.shape(), &[2, 2]);
    assert_eq!(view.strides(), &[1, 4]);
    assert_eq!(view.offset(), 10);
    assert_eq!(view.placement(), &placement);
    let buffer = view.backend_buffer().expect("backend buffer");
    assert_eq!(buffer.len(), 16);
}

#[test]
fn backend_region_view_rejects_host_buffer() {
    let tensor = TypedTensor::<f64>::from_vec_col_major(vec![4], vec![0.0; 4]).unwrap();
    let err = tensor
        .backend_region_view(vec![2, 2], vec![1, 2], 0)
        .unwrap_err();
    assert!(err.to_string().contains("backend"));

    let mut tensor = tensor;
    let err = tensor
        .backend_region_view_mut(vec![2, 2], vec![1, 2], 0)
        .unwrap_err();
    assert!(err.to_string().contains("backend"));
}

#[test]
fn backend_region_view_rejects_out_of_bounds_span() {
    let tensor = backend_tensor_f64(92, 8);
    // Max reachable element offset 7 + 1*1 + 4*1 = 12 exceeds len 8.
    assert!(tensor
        .backend_region_view(vec![2, 2], vec![1, 4], 7)
        .is_err());
    let mut tensor = tensor;
    assert!(tensor
        .backend_region_view_mut(vec![2, 2], vec![1, 4], 7)
        .is_err());
}

#[test]
fn backend_region_view_mut_rejects_aliasing_layout() {
    let mut tensor = backend_tensor_f64(93, 8);
    // Zero stride on a non-singleton axis aliases physical elements.
    assert!(tensor
        .backend_region_view_mut(vec![2, 2], vec![0, 1], 0)
        .is_err());
    // The read-only variant accepts the same broadcast-style layout.
    assert!(tensor
        .backend_region_view(vec![2, 2], vec![0, 1], 0)
        .is_ok());
}
#[test]
fn as_view_paths_do_not_allocate_or_clone_storage() {
    let tensor = TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![1.0; 4]).unwrap();
    let view = tensor.as_view();
    assert_eq!(view.shape(), &[2, 2]);

    let mut tensor = tensor.duplicate().unwrap();
    let view_mut = tensor.as_view_mut();
    assert_eq!(view_mut.shape(), &[2, 2]);
}
