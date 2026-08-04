use num_complex::{Complex32, Complex64};
use tenferro_tensor::{
    MemoryKind, Placement, StorageBuffer, TensorView, TypedTensor, TypedTensorView,
    TypedTensorViewMut,
};

#[test]
fn complex32_read_only_view_maps_interleaved_components() {
    let data = [Complex32::new(1.0, -2.0), Complex32::new(3.0, -4.0)];
    let view = TypedTensorView::from_slice([2], [1], 0, &data)
        .unwrap()
        .as_real_view()
        .unwrap();

    assert_eq!(view.shape(), &[2, 2]);
    assert_eq!(view.strides(), &[1, 2]);
    assert_eq!(view.get(&[0, 0]), Some(&1.0));
    assert_eq!(view.get(&[1, 0]), Some(&-2.0));
    assert_eq!(view.get(&[0, 1]), Some(&3.0));
    assert_eq!(view.get(&[1, 1]), Some(&-4.0));
}

#[test]
fn reinterpretation_keeps_the_same_host_pointer() {
    let data = [Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
    let source = TypedTensorView::from_slice([2], [1], 0, &data).unwrap();
    let real = source.as_real_view().unwrap();

    assert_eq!(
        source.as_slice().unwrap().as_ptr().cast::<f32>(),
        real.as_slice().unwrap().as_ptr()
    );
}

#[test]
fn reverse_complex_view_preserves_signed_mapping() {
    let data = [Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
    let view = TypedTensorView::from_slice([2], [-1], 1, &data)
        .unwrap()
        .as_real_view()
        .unwrap();

    assert_eq!(view.shape(), &[2, 2]);
    assert_eq!(view.strides(), &[1, -2]);
    assert_eq!(view.offset(), 2);
    assert_eq!(view.get(&[0, 0]), Some(&3.0));
    assert_eq!(view.get(&[1, 0]), Some(&4.0));
    assert_eq!(view.get(&[0, 1]), Some(&1.0));
    assert_eq!(view.get(&[1, 1]), Some(&2.0));
}

#[test]
fn mutable_reinterpretation_writes_the_same_bytes() {
    let mut data = [Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
    let mut source = TypedTensorViewMut::from_slice([2], [1], 0, &mut data).unwrap();
    let mut view = source.as_real_view_mut().unwrap();
    *view.get_mut(&[1, 0]).unwrap() = 20.0;
    drop(view);

    assert_eq!(data[0], Complex32::new(1.0, 20.0));
}

#[test]
fn mutable_owned_reinterpretation_uses_exclusive_root_access() {
    let mut tensor =
        TypedTensor::<Complex32>::from_vec_col_major(vec![1], vec![Complex32::new(1.0, 2.0)])
            .unwrap();
    let mut view = tensor.as_real_view_mut().unwrap();
    *view.get_mut(&[0, 0]).unwrap() = 7.0;
    drop(view);

    assert_eq!(tensor.host_data().unwrap(), &[Complex32::new(7.0, 2.0)]);
}

#[test]
fn real_view_maps_back_to_complex() {
    let data = [1.0_f32, 2.0, 3.0, 4.0];
    let view = TypedTensorView::from_slice([2, 2], [1, 2], 0, &data)
        .unwrap()
        .as_complex_view()
        .unwrap();

    assert_eq!(view.shape(), &[2]);
    assert_eq!(view.strides(), &[1]);
    assert_eq!(view.get(&[0]), Some(&Complex32::new(1.0, 2.0)));
    assert_eq!(view.get(&[1]), Some(&Complex32::new(3.0, 4.0)));
}

#[test]
fn consuming_owner_keeps_host_allocation_and_maps_data() {
    let tensor = TypedTensor::<Complex64>::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
    )
    .unwrap();
    let real = tensor.into_real().unwrap();

    assert_eq!(real.shape(), &[2, 2]);
    assert_eq!(real.layout().strides(), &[1, 2]);
    assert_eq!(real.host_data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn consuming_owner_failure_recovers_original() {
    let tensor = TypedTensor::<f32>::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let failure = tensor.into_complex().unwrap_err();
    let original = failure.into_owner();

    assert_eq!(original.shape(), &[3]);
    assert_eq!(original.host_data().unwrap(), &[1.0, 2.0, 3.0]);
}

#[test]
fn consuming_buffer_host_owner_reinterprets_without_retagging() {
    let tensor = TypedTensor::<Complex32>::from_buffer_col_major(
        vec![1],
        StorageBuffer::Host(vec![Complex32::new(1.0, 2.0)]),
        Placement {
            memory_kind: MemoryKind::UnpinnedHost,
            device: None,
            cpu_affinity: None,
        },
    )
    .unwrap();
    let real = tensor.into_real().unwrap();

    assert_eq!(real.shape(), &[2, 1]);
    assert_eq!(real.host_data().unwrap(), &[1.0, 2.0]);
}

#[test]
fn consuming_real_owner_maps_back_to_complex() {
    let tensor =
        TypedTensor::<f32>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let complex = tensor.into_complex().unwrap();

    assert_eq!(complex.shape(), &[2]);
    assert_eq!(
        complex.host_data().unwrap(),
        &[Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)]
    );
}

#[test]
fn dynamic_view_dispatches_only_sealed_pairs() {
    let data = [Complex32::new(1.0, 2.0)];
    let view = TensorView::C32(TypedTensorView::from_col_major(&[1], &data).unwrap());
    let real = view.as_real_view().unwrap();
    assert_eq!(real.shape(), &[2, 1]);
}
