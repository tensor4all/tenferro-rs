use std::panic::{catch_unwind, AssertUnwindSafe};

use num_complex::{Complex32, Complex64};

use crate::types::{
    col_major_strides, flat_to_multi, Buffer, BufferHandle, ComputeDevice, ConjElem, DType,
    MemoryKind, Placement, Tensor, TypedTensor,
};

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
