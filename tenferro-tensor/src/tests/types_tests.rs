use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use num_complex::{Complex32, Complex64};

use crate::types::{
    col_major_strides, flat_to_multi, Buffer, BufferHandle, ConjElem, DType, DeviceId, DeviceKind,
    GpuBackendKind, MemoryKind, Placement, Rank, StridedSliceSpec, StridedTensorView,
    StridedTensorViewMut, Tensor, TensorLayout, TensorRank, TensorRead, TensorScalar, TensorView,
    TypedStridedTensorView, TypedStridedTensorViewMut, TypedTensor, TypedTensorView,
};
use crate::Error;

mod strided_dynamic;

#[derive(Debug)]
struct NonCloneElement;

fn tensor_scalar_roundtrip<T>(shape: Vec<usize>, data: Vec<T>)
where
    T: TensorScalar + PartialEq + std::fmt::Debug,
{
    let tensor = T::into_tensor(shape.clone(), data.clone());

    assert_eq!(tensor.shape(), shape.as_slice());
    assert_eq!(T::try_as_slice(&tensor), Some(data.as_slice()));
    assert_eq!(tensor.as_slice::<T>(), Some(data.as_slice()));

    let mut mutable = tensor.clone();
    assert_eq!(
        T::try_as_slice_mut(&mut mutable).map(|slice| &*slice),
        Some(data.as_slice())
    );

    let typed = T::try_into_typed(tensor).unwrap();
    assert_eq!(typed.as_slice(), data.as_slice());
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
fn col_major_helpers_cover_scalar_and_higher_rank_shapes() {
    assert_eq!(col_major_strides(&[]), Vec::<isize>::new());
    assert_eq!(col_major_strides(&[2, 3, 4]), vec![1, 2, 6]);

    let mut scalar_idx = [];
    flat_to_multi(0, &[], &mut scalar_idx);

    let mut idx = [0usize; 3];
    flat_to_multi(13, &[2, 3, 4], &mut idx);
    assert_eq!(idx, [1, 0, 2]);
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
fn default_placement_is_unpinned_host_without_device() {
    let tensor = TypedTensor::<f64>::zeros(vec![2]);

    assert_eq!(tensor.placement.memory_kind, MemoryKind::UnpinnedHost);
    assert_eq!(tensor.placement.device, None);
}

#[test]
fn gpu_placement_is_metadata_only() {
    let placement = Placement {
        memory_kind: MemoryKind::Device,
        device: Some(DeviceId {
            kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
            ordinal: 0,
        }),
    };

    assert_eq!(placement.memory_kind, MemoryKind::Device);
    assert_eq!(
        placement.device.as_ref().map(|device| &device.kind),
        Some(&DeviceKind::Gpu(GpuBackendKind::Cuda))
    );
}

#[test]
fn typed_tensor_rank_zero_access_and_mutation_work() {
    let mut tensor = TypedTensor::<f64>::from_vec_col_major(vec![], vec![7.0_f64]);

    assert_eq!(tensor.n_elements(), 1);
    assert_eq!(tensor.linear_offset(&[]), 0);
    assert_eq!(*tensor.get(&[]), 7.0);

    *tensor.get_mut(&[]) = 9.5;
    assert_eq!(tensor.host_data(), &[9.5]);
}

#[test]
fn typed_tensor_static_rank_constructs_compact_layout() {
    let tensor = TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.layout().strides(), &[1, 2]);
    assert!(tensor.layout().is_compact_col_major());
}

#[test]
fn typed_tensor_owned_layout_is_always_compact() {
    let tensor = TypedTensor::<i32>::from_vec_col_major(vec![3], vec![1, 2, 3]);
    assert_eq!(
        tensor.layout(),
        &TensorLayout::compact(vec![3].into()).unwrap()
    );
}

#[test]
fn typed_tensor_backend_buffer_layout_length_mismatch_panics() {
    let mismatched = catch_unwind(AssertUnwindSafe(|| {
        TypedTensor::<f64>::from_buffer_col_major(
            vec![1],
            Buffer::Backend(Arc::new(BufferHandle::<f64>::new(9))),
            Placement {
                memory_kind: MemoryKind::Device,
                device: Some(DeviceId {
                    kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                    ordinal: 0,
                }),
            },
        );
    }));

    assert!(mismatched.is_err());
}

#[test]
fn typed_tensor_static_rank_shape_conversion_reports_rank_mismatch() {
    let err = <Rank<2> as TensorRank>::shape_from_vec(vec![2, 3, 4].into()).unwrap_err();

    assert!(matches!(
        err,
        tenferro_tensor_core::Error::RankMismatch {
            expected: 2,
            actual: 3
        }
    ));
}

#[test]
fn tensor_owned_export_returns_column_major_buffer() {
    let data = vec![1.0_f64, 2.0, 3.0, 4.0];
    let ptr = data.as_ptr();
    let tensor = Tensor::from_vec_col_major(vec![2, 2], data);

    let (shape, out) = tensor.try_into_vec_col_major::<f64>().unwrap();

    assert_eq!(shape, vec![2, 2]);
    assert_eq!(out.as_ptr(), ptr);
}

#[test]
fn tensor_owned_export_reports_dtype_mismatch() {
    let err = Tensor::from_vec_col_major(vec![1], vec![1.0_f64])
        .try_into_vec_col_major::<f32>()
        .unwrap_err();

    assert!(matches!(err, Error::DTypeMismatch { .. }));
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
        },
    );
    let col_err = tensor.clone().try_into_vec_col_major().unwrap_err();
    let row_err = tensor.try_into_vec_row_major().unwrap_err();

    assert!(matches!(col_err, Error::BackendFailure { .. }));
    assert!(col_err
        .to_string()
        .contains("backend buffers cannot be exported"));
    assert!(matches!(row_err, Error::BackendFailure { .. }));
    assert!(row_err
        .to_string()
        .contains("backend buffers cannot be exported"));
}

#[test]
fn typed_tensor_explicit_memory_order_constructors_match_logical_matrix() {
    let row =
        TypedTensor::<f64>::from_vec_row_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let col =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    assert_eq!(row.shape(), &[2, 3]);
    assert_eq!(row.as_slice(), col.as_slice());
    assert_eq!(row.get(&[0, 0]), &1.0);
    assert_eq!(row.get(&[1, 0]), &4.0);
    assert_eq!(row.get(&[0, 2]), &3.0);
    assert_eq!(row.get(&[1, 2]), &6.0);
}

#[test]
fn typed_tensor_explicit_memory_order_exports_requested_order() {
    let tensor =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    let (shape, col) = tensor.clone().try_into_vec_col_major().unwrap();
    assert_eq!(shape, vec![2, 3]);
    assert_eq!(col, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    let (shape, row) = tensor.try_into_vec_row_major().unwrap();
    assert_eq!(shape, vec![2, 3]);
    assert_eq!(row, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn tensor_explicit_memory_order_roundtrips_dynamic_dtype() {
    let tensor = Tensor::from_vec_row_major(vec![2, 2], vec![1_i64, 2, 3, 4]);

    assert_eq!(tensor.as_slice::<i64>().unwrap(), &[1, 3, 2, 4]);
    assert_eq!(
        tensor.clone().try_into_vec_col_major::<i64>().unwrap(),
        (vec![2, 2], vec![1, 3, 2, 4]),
    );
    assert_eq!(
        tensor.try_into_vec_row_major::<i64>().unwrap(),
        (vec![2, 2], vec![1, 2, 3, 4]),
    );
}

#[test]
fn col_major_typed_tensor_uses_logical_indices() {
    let tensor =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]);

    assert_eq!(*tensor.get(&[0, 2]), 3.0);
    assert_eq!(*tensor.get(&[1, 0]), 4.0);
}

#[test]
fn typed_tensor_linear_offsets_cover_higher_rank_shapes() {
    let tensor = TypedTensor::<f64>::zeros(vec![2, 3, 5]);

    assert_eq!(tensor.linear_offset(&[1, 2, 4]), 1 + 2 * 2 + 4 * 2 * 3);
}

#[test]
fn typed_tensor_iterators_follow_physical_buffer_order() {
    let tensor =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(
        tensor.iter().copied().collect::<Vec<_>>(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );

    let mut tensor = TypedTensor::<i64>::from_vec_col_major(vec![3], vec![1_i64, 2, 3]);
    for value in tensor.iter_mut() {
        *value *= 2;
    }
    assert_eq!(tensor.as_slice(), &[2, 4, 6]);
}

#[test]
fn tensor_iterators_are_typed_by_requested_scalar() {
    let mut tensor = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);

    assert_eq!(
        tensor.iter::<f64>().unwrap().copied().collect::<Vec<_>>(),
        vec![1.0, 2.0, 3.0]
    );
    assert!(tensor.iter::<f32>().is_none());

    for value in tensor.iter_mut::<f64>().unwrap() {
        *value += 1.0;
    }

    assert_eq!(tensor.as_slice::<f64>().unwrap(), &[2.0, 3.0, 4.0]);
    assert!(tensor.as_slice_mut::<f32>().is_none());
}

#[test]
fn typed_tensor_checked_and_unchecked_accessors_work() {
    let mut tensor =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]);

    assert_eq!(tensor.as_physical_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(tensor.try_get(&[1, 2]), Some(&6.0));
    assert_eq!(tensor.try_get(&[2, 0]), None);
    assert_eq!(tensor.try_get(&[0]), None);

    *tensor.try_get_mut(&[0, 1]).unwrap() = 20.0;
    assert_eq!(tensor.as_physical_slice()[2], 20.0);

    tensor.as_physical_slice_mut()[0] = 10.0;
    assert_eq!(unsafe { *tensor.get_unchecked(&[0, 0]) }, 10.0);

    unsafe {
        *tensor.get_unchecked_mut(&[1, 0]) = 40.0;
    }
    assert_eq!(tensor.try_get(&[1, 0]), Some(&40.0));
}

#[test]
fn typed_tensor_rank_fixed_accessors_use_column_major_offsets() {
    let mut col2 = TypedTensor::<i64>::from_vec_col_major(vec![2, 3], vec![1_i64, 2, 3, 4, 5, 6]);

    assert_eq!(col2.linear_offset2(1, 2), 5);
    assert_eq!(*col2.get2(1, 2), 6);
    *col2.get_mut2(0, 1) = 30;
    assert_eq!(col2.as_physical_slice()[2], 30);

    let mut col3 = TypedTensor::<i64>::from_vec_col_major(vec![2, 3, 2], (0_i64..12).collect());

    assert_eq!(col3.linear_offset3(1, 2, 1), 11);
    assert_eq!(*col3.get3(1, 2, 1), 11);
    *col3.get_mut3(0, 1, 1) = 60;
    assert_eq!(col3.as_physical_slice()[8], 60);
}

#[test]
fn tensor_checked_accessors_are_typed_and_use_physical_order() {
    let mut tensor = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]);

    assert_eq!(
        tensor.as_physical_slice::<f64>().unwrap(),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
    assert_eq!(tensor.linear_offset(&[1, 0]), 1);
    assert_eq!(tensor.linear_offset2(1, 0), 1);
    assert_eq!(tensor.try_get::<f64>(&[1, 2]), Some(&6.0));
    assert_eq!(tensor.try_get::<f32>(&[1, 2]), None);
    assert_eq!(tensor.try_get::<f64>(&[2, 0]), None);

    tensor.as_physical_slice_mut::<f64>().unwrap()[0] = 10.0;
    *tensor.try_get_mut::<f64>(&[0, 1]).unwrap() = 20.0;

    assert_eq!(
        unsafe { *tensor.get_unchecked::<f64>(&[0, 0]).unwrap() },
        10.0
    );
    unsafe {
        *tensor.get_unchecked_mut::<f64>(&[1, 0]).unwrap() = 40.0;
    }

    assert_eq!(
        tensor.as_physical_slice::<f64>().unwrap(),
        &[10.0, 40.0, 20.0, 5.0, 3.0, 6.0]
    );
}

#[test]
fn typed_tensor_panics_cover_length_and_indexing_errors() {
    let mismatched =
        catch_unwind(|| TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0]));
    assert!(mismatched.is_err());

    let tensor = TypedTensor::<f64>::zeros(vec![2, 3]);
    let rank_mismatch = catch_unwind(AssertUnwindSafe(|| tensor.linear_offset(&[0])));
    assert!(rank_mismatch.is_err());

    let oob = catch_unwind(AssertUnwindSafe(|| tensor.linear_offset(&[2, 0])));
    assert!(oob.is_err());
}

#[test]
fn backend_buffers_panic_when_host_access_is_requested() {
    let placement = Placement {
        memory_kind: MemoryKind::Other("backend".to_string()),
        device: Some(DeviceId {
            kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
            ordinal: 0,
        }),
    };
    let tensor = TypedTensor::<f64>::from_buffer_col_major(
        vec![1],
        Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(7, 1))),
        placement.clone(),
    );
    assert_eq!(tensor.placement.device.as_ref().unwrap().ordinal, 0);

    let host_data = catch_unwind(AssertUnwindSafe(|| tensor.host_data()));
    assert!(host_data.is_err());

    let mut mutable_tensor = TypedTensor::<f64>::from_buffer_col_major(
        vec![1],
        Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(8, 1))),
        placement,
    );
    let host_data_mut = catch_unwind(AssertUnwindSafe(|| {
        let _ = mutable_tensor.host_data_mut();
    }));
    assert!(host_data_mut.is_err());
}

#[test]
fn typed_tensor_metadata_accessors_accept_non_clone_elements() {
    let placement = Placement {
        memory_kind: MemoryKind::Other("backend".to_string()),
        device: None,
    };
    let tensor = TypedTensor::<NonCloneElement>::from_buffer_col_major(
        vec![2, 3],
        Buffer::Backend(Arc::new(BufferHandle::<NonCloneElement>::new_with_len(
            9, 6,
        ))),
        placement,
    );

    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.rank(), 2);
    assert_eq!(tensor.n_elements(), 6);
    assert_eq!(tensor.layout().strides(), &[1, 2]);
    assert!(tensor.into_layout().is_compact_col_major());
}

#[test]
fn tensor_shape_and_dtype_cover_all_variants() {
    let f32_tensor = Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]));
    let f64_tensor = Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![3.0_f64]));
    let i32_tensor = Tensor::I32(TypedTensor::from_vec_col_major(vec![2], vec![1_i32, -2]));
    let c32_tensor = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![1],
        vec![Complex32::new(1.0, -2.0)],
    ));
    let c64_tensor = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![1, 1],
        vec![Complex64::new(-3.0, 4.0)],
    ));
    let bool_tensor = Tensor::Bool(TypedTensor::from_vec_col_major(vec![2], vec![true, false]));

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
    let view = TypedTensorView::new(&shape, &data).unwrap();

    assert_eq!(view.shape, &[2, 2]);
    assert_eq!(view.as_slice(), &[1.0, 2.0, 3.0, 4.0]);

    let err = TypedTensorView::new(&shape, &data[..3]).unwrap_err();
    assert!(matches!(err, Error::InvalidConfig { .. }));
}

#[test]
fn tensor_view_covers_dtype_shape_and_materialization() {
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
        let tensor = view.to_tensor();
        assert_eq!(tensor.dtype(), dtype);
        assert_eq!(tensor.shape(), &[2]);
    }
}

#[test]
fn strided_tensor_view_materializes_sliced_host_layouts() {
    let row_major = [1_i32, 2, 3, 4, 5, 6];
    let view = TypedStridedTensorView::new(&[2, 3], &[3, 1], 0, &row_major).unwrap();

    assert_eq!(view.shape(), &[2, 3]);
    assert_eq!(view.strides(), &[3, 1]);
    assert_eq!(view.get(&[1, 2]), Some(&6));
    assert_eq!(
        view.materialize_col_major().unwrap().as_slice(),
        &[1, 4, 2, 5, 3, 6]
    );

    let transposed = view.try_permute_axes(&[1, 0]).unwrap();
    assert_eq!(transposed.shape(), &[3, 2]);
    assert_eq!(
        transposed.materialize_col_major().unwrap().as_slice(),
        &[1, 2, 3, 4, 5, 6]
    );

    let reversed_cols = view.try_slice_axis(1, StridedSliceSpec::reverse()).unwrap();
    assert_eq!(reversed_cols.strides(), &[3, -1]);
    assert_eq!(
        reversed_cols.materialize_col_major().unwrap().as_slice(),
        &[3, 6, 2, 5, 1, 4]
    );

    let every_other_col = view
        .try_slice(&[StridedSliceSpec::all(), StridedSliceSpec::new(0, None, 2)])
        .unwrap();
    assert_eq!(
        every_other_col.materialize_col_major().unwrap().as_slice(),
        &[1, 4, 3, 6]
    );
    assert!(view.try_reshape(&[6]).is_err());

    let col_major = [1_i32, 4, 2, 5, 3, 6];
    let contiguous = TypedStridedTensorView::from_col_major(&[2, 3], &col_major).unwrap();
    assert_eq!(contiguous.try_reshape(&[6]).unwrap().strides(), &[1]);
}

#[test]
fn dynamic_strided_tensor_view_covers_i32_and_bool() {
    let i32_data = [1_i32, 2, 3, 4];
    let i32_view = StridedTensorView::i32(&[2, 2], &[2, 1], 0, &i32_data).unwrap();
    assert_eq!(i32_view.dtype(), DType::I32);
    assert_eq!(
        i32_view.to_tensor().unwrap().as_slice::<i32>().unwrap(),
        &[1, 3, 2, 4]
    );

    let bool_data = [false, true, true];
    let bool_view = StridedTensorView::bool(&[3], &[-1], 2, &bool_data).unwrap();
    assert_eq!(bool_view.dtype(), DType::Bool);
    assert_eq!(
        bool_view.to_tensor().unwrap().as_slice::<bool>().unwrap(),
        &[true, true, false]
    );
}

#[test]
fn strided_tensor_view_mut_updates_sliced_host_layouts() {
    let mut row_major = [1_i32, 2, 3, 4, 5, 6];
    let mut view = TypedStridedTensorViewMut::new(&[2, 3], &[3, 1], 0, &mut row_major).unwrap();

    assert_eq!(view.shape(), &[2, 3]);
    assert_eq!(view.strides(), &[3, 1]);
    assert_eq!(view.get(&[1, 2]), Some(&6));

    *view.get_mut(&[1, 2]).unwrap() = 60;
    assert_eq!(view.get(&[1, 2]), Some(&60));

    {
        let mut transposed = view.try_permute_axes(&[1, 0]).unwrap();
        *transposed.get_mut(&[2, 1]).unwrap() = 600;
    }
    assert_eq!(view.get(&[1, 2]), Some(&600));

    {
        let mut reversed_cols = view.try_slice_axis(1, StridedSliceSpec::reverse()).unwrap();
        assert_eq!(reversed_cols.strides(), &[3, -1]);
        *reversed_cols.get_mut(&[0, 0]).unwrap() = 30;
    }
    assert_eq!(view.get(&[0, 2]), Some(&30));

    let materialized = view.materialize_col_major().unwrap();
    assert_eq!(materialized.as_slice(), &[1, 4, 2, 5, 30, 600]);
}

#[test]
fn strided_tensor_view_mut_rejects_aliasing_layouts() {
    let data = [1_i32, 2, 3, 4];
    assert!(TypedStridedTensorView::new(&[2, 2], &[1, 1], 0, &data).is_ok());

    let mut data = [1_i32, 2, 3, 4];
    let err = TypedStridedTensorViewMut::new(&[2, 2], &[1, 1], 0, &mut data).unwrap_err();
    assert!(matches!(err, Error::InvalidConfig { .. }));

    let mut data = [1_i32, 2];
    assert!(TypedStridedTensorViewMut::new(&[2], &[0], 0, &mut data).is_err());

    let mut data = [1_i32, 2, 3];
    let mut reversed = TypedStridedTensorViewMut::new(&[3], &[-1], 2, &mut data).unwrap();
    *reversed.get_mut(&[2]).unwrap() = 10;
    assert_eq!(reversed.as_physical_slice(), &[10, 2, 3]);

    let mut data = [1_i32, 2];
    let singleton_zero_stride =
        TypedStridedTensorViewMut::new(&[1, 2], &[0, 1], 0, &mut data).unwrap();
    assert_eq!(singleton_zero_stride.shape(), &[1, 2]);
}

#[test]
fn strided_tensor_view_mut_multi_slice_returns_option() {
    let mut data = [1_i32, 2, 3, 4, 5, 6];
    let mut view = TypedStridedTensorViewMut::new(&[6], &[1], 0, &mut data).unwrap();

    {
        let (mut left, mut right) = view
            .try_multi_slice_mut(
                &[StridedSliceSpec::new(0, Some(3), 1)],
                &[StridedSliceSpec::new(3, Some(6), 1)],
            )
            .unwrap();
        *left.get_mut(&[2]).unwrap() = 30;
        *right.get_mut(&[0]).unwrap() = 40;
    }
    assert_eq!(view.as_physical_slice(), &[1, 2, 30, 40, 5, 6]);

    assert!(view
        .try_multi_slice_mut(
            &[StridedSliceSpec::new(0, Some(4), 1)],
            &[StridedSliceSpec::new(3, Some(6), 1)],
        )
        .is_none());

    assert!(view
        .try_multi_slice_mut(
            &[StridedSliceSpec::new(0, Some(2), 0)],
            &[StridedSliceSpec::new(3, Some(6), 1)],
        )
        .is_none());
}

#[test]
fn dynamic_strided_tensor_view_mut_covers_i32_and_bool() {
    let mut i32_data = [1_i32, 2, 3, 4];
    let mut i32_view = StridedTensorViewMut::i32(&[2, 2], &[2, 1], 0, &mut i32_data).unwrap();
    assert_eq!(i32_view.dtype(), DType::I32);
    match &mut i32_view {
        StridedTensorViewMut::I32(view) => *view.get_mut(&[1, 1]).unwrap() = 40,
        _ => unreachable!(),
    }
    assert_eq!(
        i32_view.to_tensor().unwrap().as_slice::<i32>().unwrap(),
        &[1, 3, 2, 40]
    );

    let mut bool_data = [false, true, true];
    let mut bool_view = StridedTensorViewMut::bool(&[3], &[-1], 2, &mut bool_data).unwrap();
    assert_eq!(bool_view.dtype(), DType::Bool);
    match &mut bool_view {
        StridedTensorViewMut::Bool(view) => *view.get_mut(&[2]).unwrap() = true,
        _ => unreachable!(),
    }
    assert_eq!(
        bool_view.to_tensor().unwrap().as_slice::<bool>().unwrap(),
        &[true, true, true]
    );
}

#[test]
fn dynamic_strided_tensor_view_mut_multi_slice_returns_option() {
    let mut data = [1_i32, 2, 3, 4];
    let mut view = StridedTensorViewMut::i32(&[4], &[1], 0, &mut data).unwrap();
    {
        let (left, right) = view
            .try_multi_slice_mut(
                &[StridedSliceSpec::new(0, Some(2), 1)],
                &[StridedSliceSpec::new(2, Some(4), 1)],
            )
            .unwrap();
        assert_eq!(left.shape(), &[2]);
        assert_eq!(right.shape(), &[2]);
    }

    assert!(view
        .try_multi_slice_mut(
            &[StridedSliceSpec::new(0, Some(3), 1)],
            &[StridedSliceSpec::new(2, Some(4), 1)],
        )
        .is_none());
}

#[test]
fn strided_tensor_view_validation_covers_error_edges() {
    let data = [1_i32, 2, 3];

    let empty = TypedStridedTensorView::new(&[0, 3], &[1, 0], 3, &data).unwrap();
    assert_eq!(empty.n_elements(), 0);
    assert_eq!(
        empty.materialize_col_major().unwrap().as_slice(),
        &[] as &[i32]
    );
    assert_eq!(empty.get(&[0, 0]), None);
    assert!(matches!(
        TypedStridedTensorView::<i32>::new(&[0], &[1], 4, &data),
        Err(Error::InvalidConfig { .. })
    ));

    assert!(matches!(
        TypedStridedTensorView::<i32>::new(&[2], &[1, 1], 0, &data),
        Err(Error::RankMismatch { .. })
    ));
    assert!(matches!(
        TypedStridedTensorView::<i32>::new(&[2], &[-1], 0, &data[..1]),
        Err(Error::InvalidConfig { .. })
    ));
    assert!(matches!(
        TypedStridedTensorView::<i32>::new(&[2], &[2], 0, &data[..1]),
        Err(Error::InvalidConfig { .. })
    ));

    let view = TypedStridedTensorView::new(&[3], &[1], 0, &data).unwrap();
    assert_eq!(view.try_get(&[1]), Some(&2));
    assert_eq!(view.try_linear_offset(&[0, 0]), None);
    assert_eq!(view.try_linear_offset(&[3]), None);

    assert!(matches!(
        TypedStridedTensorView::<i32>::new(&[usize::MAX, 2], &[1, 1], 0, &[]),
        Err(Error::InvalidConfig { .. })
    ));
    assert!(matches!(
        TypedStridedTensorView::<i32>::new(&[3], &[isize::MAX], 0, &[]),
        Err(Error::InvalidConfig { .. })
    ));
    assert!(matches!(
        TypedStridedTensorView::<i32>::new(&[2, 2], &[isize::MAX, 1], 0, &[]),
        Err(Error::InvalidConfig { .. })
    ));
}

#[test]
fn strided_tensor_view_slice_permute_and_reshape_cover_boundaries() {
    let data = [1_i32, 2, 3, 4, 5, 6];
    let view = TypedStridedTensorView::new(&[2, 3], &[3, 1], 0, &data).unwrap();

    assert!(matches!(
        view.try_permute_axes(&[0]),
        Err(Error::RankMismatch { .. })
    ));
    assert!(matches!(
        view.try_permute_axes(&[2, 0]),
        Err(Error::AxisOutOfBounds { .. })
    ));
    assert!(matches!(
        view.try_permute_axes(&[0, 0]),
        Err(Error::DuplicateAxis { .. })
    ));

    assert!(matches!(
        view.try_slice(&[StridedSliceSpec::all()]),
        Err(Error::RankMismatch { .. })
    ));
    assert!(matches!(
        view.try_slice_axis(2, StridedSliceSpec::all()),
        Err(Error::AxisOutOfBounds { .. })
    ));
    assert!(matches!(
        view.try_slice(&[StridedSliceSpec::all(), StridedSliceSpec::new(0, None, 0)]),
        Err(Error::InvalidConfig { .. })
    ));
    assert!(matches!(
        view.try_slice(&[StridedSliceSpec::all(), StridedSliceSpec::new(-4, None, 1)]),
        Err(Error::InvalidConfig { .. })
    ));
    assert!(matches!(
        view.try_slice(&[
            StridedSliceSpec::all(),
            StridedSliceSpec::new(0, Some(4), 1)
        ]),
        Err(Error::InvalidConfig { .. })
    ));

    let empty = view
        .try_slice_axis(1, StridedSliceSpec::new(2, Some(1), 1))
        .unwrap();
    assert_eq!(empty.shape(), &[2, 0]);
    assert_eq!(
        empty.materialize_col_major().unwrap().as_slice(),
        &[] as &[i32]
    );

    assert!(matches!(
        view.try_reshape(&[5]),
        Err(Error::InvalidConfig { .. })
    ));
    assert!(matches!(
        TypedStridedTensorView::<i32>::from_col_major(&[isize::MAX as usize, 2, 2], &[]),
        Err(Error::InvalidConfig { .. })
    ));

    let scalar = TypedStridedTensorView::from_col_major(&[], &data[..1]).unwrap();
    assert_eq!(scalar.shape(), &[] as &[usize]);
    assert_eq!(scalar.strides(), &[] as &[isize]);
    assert_eq!(scalar.get(&[]), Some(&1));

    let singleton_axis = TypedStridedTensorView::new(&[1, 3], &[99, 1], 0, &data).unwrap();
    assert_eq!(singleton_axis.try_reshape(&[3]).unwrap().strides(), &[1]);
}

#[test]
fn strided_tensor_view_mut_multi_slice_covers_empty_reverse_and_conservative_cases() {
    let mut data = [0_i32, 1, 2, 3, 4, 5];
    let mut view = TypedStridedTensorViewMut::new(&[6], &[1], 0, &mut data).unwrap();
    {
        let (mut high, mut low) = view
            .try_multi_slice_mut(
                &[StridedSliceSpec::new(4, Some(6), 1)],
                &[StridedSliceSpec::new(1, Some(3), 1)],
            )
            .unwrap();
        *high.get_mut(&[0]).unwrap() = 40;
        *low.get_mut(&[1]).unwrap() = 20;
    }
    assert_eq!(view.as_physical_slice(), &[0, 1, 20, 3, 40, 5]);

    let mut data = [0_i32, 1, 2, 3];
    let mut view = TypedStridedTensorViewMut::new(&[4], &[1], 0, &mut data).unwrap();
    {
        let (empty, mut right) = view
            .try_multi_slice_mut(
                &[StridedSliceSpec::new(0, Some(0), 1)],
                &[StridedSliceSpec::new(2, Some(4), 1)],
            )
            .unwrap();
        assert_eq!(empty.n_elements(), 0);
        *right.get_mut(&[0]).unwrap() = 20;
    }
    assert_eq!(view.as_physical_slice(), &[0, 1, 20, 3]);

    let mut data = [0_i32, 1, 2, 3];
    let mut view = TypedStridedTensorViewMut::new(&[4], &[1], 0, &mut data).unwrap();
    {
        let (mut left, empty) = view
            .try_multi_slice_mut(
                &[StridedSliceSpec::new(0, Some(2), 1)],
                &[StridedSliceSpec::new(4, Some(4), 1)],
            )
            .unwrap();
        assert_eq!(empty.n_elements(), 0);
        *left.get_mut(&[1]).unwrap() = 10;
    }
    assert_eq!(view.as_physical_slice(), &[0, 10, 2, 3]);

    let mut data = [0_i32, 1, 2, 3];
    let mut view = TypedStridedTensorViewMut::new(&[4], &[1], 0, &mut data).unwrap();
    let (empty_left, empty_right) = view
        .try_multi_slice_mut(
            &[StridedSliceSpec::new(0, Some(0), 1)],
            &[StridedSliceSpec::new(4, Some(4), 1)],
        )
        .unwrap();
    assert_eq!(empty_left.n_elements(), 0);
    assert_eq!(empty_right.n_elements(), 0);

    let mut data = [0_i32, 1, 2, 3, 4, 5];
    let mut view = TypedStridedTensorViewMut::new(&[6], &[1], 0, &mut data).unwrap();
    {
        let (mut reversed_high, mut low) = view
            .try_multi_slice_mut(
                &[StridedSliceSpec::new(3, Some(6), -1)],
                &[StridedSliceSpec::new(0, Some(3), 1)],
            )
            .unwrap();
        assert_eq!(reversed_high.get(&[0]), Some(&5));
        *reversed_high.get_mut(&[2]).unwrap() = 30;
        *low.get_mut(&[2]).unwrap() = 20;
    }
    assert_eq!(view.as_physical_slice(), &[0, 1, 20, 30, 4, 5]);

    let mut data = [0_i32, 1, 2, 3, 4, 5];
    let mut view = TypedStridedTensorViewMut::new(&[6], &[1], 0, &mut data).unwrap();
    assert!(view
        .try_multi_slice_mut(
            &[StridedSliceSpec::new(0, Some(6), 2)],
            &[StridedSliceSpec::new(1, Some(6), 2)],
        )
        .is_none());
}

#[test]
fn tensor_read_wraps_owned_tensor_or_borrowed_view() {
    let tensor = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]);
    let read_tensor = TensorRead::from_tensor(&tensor);

    assert_eq!(read_tensor.dtype(), DType::F64);
    assert_eq!(read_tensor.shape(), &[2]);
    assert!(read_tensor.as_tensor().is_some());
    assert_eq!(
        read_tensor.to_tensor().as_slice::<f64>().unwrap(),
        &[3.0, 4.0]
    );

    let shape = [2usize];
    let data = [5.0_f64, 6.0];
    let read_view = TensorRead::from_view(TensorView::f64(&shape, &data).unwrap());

    assert_eq!(read_view.dtype(), DType::F64);
    assert_eq!(read_view.shape(), &[2]);
    assert!(read_view.as_tensor().is_none());
    assert_eq!(
        read_view.to_tensor().as_slice::<f64>().unwrap(),
        &[5.0, 6.0]
    );
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
    let tensor = <f64 as TensorScalar>::into_tensor(vec![2], vec![1.0, 2.0]);

    assert_eq!(tensor.as_slice::<f32>(), None);
}

#[test]
fn conj_elem_covers_real_and_complex_scalars() {
    assert_eq!(1.5_f32.conj_elem(), 1.5_f32);
    assert_eq!(2.5_f64.conj_elem(), 2.5_f64);
    assert_eq!(
        Complex32::new(1.0, 2.0).conj_elem(),
        Complex32::new(1.0, -2.0)
    );
    assert_eq!(
        Complex64::new(-3.0, 4.5).conj_elem(),
        Complex64::new(-3.0, -4.5)
    );
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
    let err = Error::AxisOutOfBounds {
        op: "dot_general",
        axis: 2,
        rank: 1,
    };

    assert!(err.to_string().contains("dot_general"));
}
