use std::panic::{catch_unwind, AssertUnwindSafe};

use num_complex::{Complex32, Complex64};

use crate::types::{
    col_major_strides, flat_to_multi, Buffer, BufferHandle, ComputeDevice, ConjElem, DType,
    MemoryKind, Placement, Tensor, TensorScalar, TypedTensor,
};
use crate::Error;

fn tensor_scalar_roundtrip<T>(shape: Vec<usize>, data: Vec<T>)
where
    T: TensorScalar + PartialEq + std::fmt::Debug,
{
    let tensor = T::into_tensor(shape.clone(), data.clone());

    assert_eq!(tensor.shape(), shape.as_slice());
    assert_eq!(T::try_as_slice(&tensor), Some(data.as_slice()));
    assert_eq!(tensor.as_slice::<T>(), Some(data.as_slice()));
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
fn typed_tensor_rank_zero_access_and_mutation_work() {
    let mut tensor = TypedTensor::from_vec(vec![], vec![7.0_f64]);

    assert_eq!(tensor.n_elements(), 1);
    assert_eq!(tensor.linear_offset(&[]), 0);
    assert_eq!(*tensor.get(&[]), 7.0);

    *tensor.get_mut(&[]) = 9.5;
    assert_eq!(tensor.host_data(), &[9.5]);
}

#[test]
fn tensor_owned_export_returns_column_major_buffer() {
    let data = vec![1.0_f64, 2.0, 3.0, 4.0];
    let ptr = data.as_ptr();
    let tensor = Tensor::from_vec(vec![2, 2], data);

    let (shape, out) = tensor.try_into_vec::<f64>().unwrap();

    assert_eq!(shape, vec![2, 2]);
    assert_eq!(out.as_ptr(), ptr);
}

#[test]
fn tensor_owned_export_reports_dtype_mismatch() {
    let err = Tensor::from_vec(vec![1], vec![1.0_f64])
        .try_into_vec::<f32>()
        .unwrap_err();

    assert!(matches!(err, Error::DTypeMismatch { .. }));
}

#[test]
fn col_major_typed_tensor_uses_logical_indices() {
    let tensor = TypedTensor::from_vec(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]);

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
    let tensor = TypedTensor::from_vec(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(
        tensor.iter().copied().collect::<Vec<_>>(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );

    let mut tensor = TypedTensor::from_vec(vec![3], vec![1_i64, 2, 3]);
    for value in tensor.iter_mut() {
        *value *= 2;
    }
    assert_eq!(tensor.as_slice(), &[2, 4, 6]);
}

#[test]
fn tensor_iterators_are_typed_by_requested_scalar() {
    let mut tensor = Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]);

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
    let mut tensor = TypedTensor::from_vec(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]);

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
    let mut col2 = TypedTensor::from_vec(vec![2, 3], vec![1_i64, 2, 3, 4, 5, 6]);

    assert_eq!(col2.linear_offset2(1, 2), 5);
    assert_eq!(*col2.get2(1, 2), 6);
    *col2.get_mut2(0, 1) = 30;
    assert_eq!(col2.as_physical_slice()[2], 30);

    let mut col3 = TypedTensor::from_vec(vec![2, 3, 2], (0_i64..12).collect());

    assert_eq!(col3.linear_offset3(1, 2, 1), 11);
    assert_eq!(*col3.get3(1, 2, 1), 11);
    *col3.get_mut3(0, 1, 1) = 60;
    assert_eq!(col3.as_physical_slice()[8], 60);
}

#[test]
fn tensor_checked_accessors_are_typed_and_use_physical_order() {
    let mut tensor = Tensor::from_vec(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]);

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
    let mismatched = catch_unwind(|| TypedTensor::<f64>::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0]));
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
        resident_device: Some(ComputeDevice {
            kind: "cuda".to_string(),
            ordinal: 0,
        }),
    };
    let tensor = TypedTensor {
        buffer: Buffer::Backend(BufferHandle::<f64>::new(7)),
        shape: vec![1],
        placement: placement.clone(),
    };
    assert_eq!(
        tensor.placement.resident_device.as_ref().unwrap().ordinal,
        0
    );

    let host_data = catch_unwind(AssertUnwindSafe(|| tensor.host_data()));
    assert!(host_data.is_err());

    let mut mutable_tensor = TypedTensor {
        buffer: Buffer::Backend(BufferHandle::<f64>::new(8)),
        shape: vec![1],
        placement,
    };
    let host_data_mut = catch_unwind(AssertUnwindSafe(|| {
        let _ = mutable_tensor.host_data_mut();
    }));
    assert!(host_data_mut.is_err());
}

#[test]
fn tensor_shape_and_dtype_cover_all_variants() {
    let f32_tensor = Tensor::F32(TypedTensor::from_vec(vec![2], vec![1.0_f32, 2.0]));
    let f64_tensor = Tensor::F64(TypedTensor::from_vec(vec![], vec![3.0_f64]));
    let c32_tensor = Tensor::C32(TypedTensor::from_vec(
        vec![1],
        vec![Complex32::new(1.0, -2.0)],
    ));
    let c64_tensor = Tensor::C64(TypedTensor::from_vec(
        vec![1, 1],
        vec![Complex64::new(-3.0, 4.0)],
    ));

    assert_eq!(f32_tensor.shape(), &[2]);
    assert_eq!(f32_tensor.dtype(), DType::F32);
    assert_eq!(f64_tensor.shape(), &[] as &[usize]);
    assert_eq!(f64_tensor.dtype(), DType::F64);
    assert_eq!(c32_tensor.shape(), &[1]);
    assert_eq!(c32_tensor.dtype(), DType::C32);
    assert_eq!(c64_tensor.shape(), &[1, 1]);
    assert_eq!(c64_tensor.dtype(), DType::C64);
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
