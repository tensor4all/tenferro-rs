use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use num_complex::{Complex32, Complex64};

use crate::types::{
    col_major_strides, flat_to_multi, Buffer, BufferHandle, ComputeDevice, ConjElem, DType,
    LayoutOrder, MemoryKind, Placement, Tensor, TypedTensor,
};
use crate::Error;

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
fn cloned_host_tensors_use_copy_on_write_for_mutation() {
    let original = TypedTensor::from_vec(vec![2], vec![1.0_f64, 2.0]);
    let mut cloned = original.clone();

    cloned.host_data_mut()[0] = 9.0;

    assert_eq!(original.host_data(), &[1.0, 2.0]);
    assert_eq!(cloned.host_data(), &[9.0, 2.0]);
}

#[test]
fn typed_tensor_constructors_populate_stride_metadata() {
    let scalar = TypedTensor::from_vec(vec![], vec![1.25_f64]);
    assert_eq!(scalar.strides(), &[] as &[isize]);
    assert_eq!(scalar.offset(), 0);
    assert!(scalar.is_contiguous_col_major());
    assert!(scalar.is_contiguous_row_major());

    let matrix = TypedTensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(matrix.strides(), &[1, 2]);
    assert_eq!(matrix.offset(), 0);
    assert!(matrix.is_contiguous_col_major());
    assert!(!matrix.is_contiguous_row_major());
}

#[test]
fn typed_tensor_contiguity_predicates_distinguish_layouts() {
    let col_major = TypedTensor::from_vec(vec![2, 3], vec![1.0_f64; 6]);
    assert!(col_major.is_contiguous_col_major());
    assert!(!col_major.is_contiguous_row_major());

    let row_major = TypedTensor {
        buffer: Buffer::Host(Arc::new(vec![1.0_f64; 6])),
        shape: vec![2, 3],
        strides: vec![3, 1],
        offset: 0,
        placement: Placement {
            memory_kind: MemoryKind::UnpinnedHost,
            resident_device: None,
        },
    };
    assert!(!row_major.is_contiguous_col_major());
    assert!(row_major.is_contiguous_row_major());

    let non_contiguous = TypedTensor {
        buffer: Buffer::Host(Arc::new(vec![1.0_f64; 12])),
        shape: vec![2, 3],
        strides: vec![2, 4],
        offset: 1,
        placement: Placement {
            memory_kind: MemoryKind::UnpinnedHost,
            resident_device: None,
        },
    };
    assert!(!non_contiguous.is_contiguous_col_major());
    assert!(!non_contiguous.is_contiguous_row_major());
}

#[test]
fn typed_tensor_to_contiguous_supports_row_and_column_major() {
    let source = TypedTensor {
        buffer: Buffer::Host(Arc::new(vec![
            1.0_f64, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0,
        ])),
        shape: vec![2, 3],
        strides: vec![2, 4],
        offset: 0,
        placement: Placement {
            memory_kind: MemoryKind::UnpinnedHost,
            resident_device: None,
        },
    };

    let col_major = source.to_contiguous(LayoutOrder::ColumnMajor).unwrap();
    assert!(col_major.is_contiguous_col_major());
    assert_eq!(col_major.strides(), &[1, 2]);
    assert_eq!(*col_major.get(&[0, 0]), 1.0);
    assert_eq!(*col_major.get(&[1, 0]), 2.0);
    assert_eq!(*col_major.get(&[0, 1]), 3.0);
    assert_eq!(*col_major.get(&[1, 2]), 6.0);

    let row_major = source.to_contiguous(LayoutOrder::RowMajor).unwrap();
    assert!(row_major.is_contiguous_row_major());
    assert_eq!(row_major.strides(), &[3, 1]);
    assert_eq!(*row_major.get(&[0, 0]), 1.0);
    assert_eq!(*row_major.get(&[1, 0]), 2.0);
    assert_eq!(*row_major.get(&[0, 1]), 3.0);
    assert_eq!(*row_major.get(&[1, 2]), 6.0);
}

#[test]
fn typed_tensor_panics_cover_length_and_indexing_errors() {
    let mismatched = catch_unwind(|| TypedTensor::<f64>::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0]));
    assert!(mismatched.is_err());

    let tensor = TypedTensor::<f64>::zeros(vec![2, 3]);
    let rank_mismatch = catch_unwind(|| tensor.linear_offset(&[0]));
    assert!(rank_mismatch.is_err());

    let oob = catch_unwind(|| tensor.linear_offset(&[2, 0]));
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
        strides: vec![1],
        offset: 0,
        placement: placement.clone(),
    };
    assert_eq!(
        tensor.placement.resident_device.as_ref().unwrap().ordinal,
        0
    );

    let host_data = catch_unwind(|| tensor.host_data());
    assert!(host_data.is_err());

    let mut mutable_tensor = TypedTensor {
        buffer: Buffer::Backend(BufferHandle::<f64>::new(8)),
        shape: vec![1],
        strides: vec![1],
        offset: 0,
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
    assert_eq!(f64_tensor.shape(), &[]);
    assert_eq!(f64_tensor.dtype(), DType::F64);
    assert_eq!(c32_tensor.shape(), &[1]);
    assert_eq!(c32_tensor.dtype(), DType::C32);
    assert_eq!(c64_tensor.shape(), &[1, 1]);
    assert_eq!(c64_tensor.dtype(), DType::C64);
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
