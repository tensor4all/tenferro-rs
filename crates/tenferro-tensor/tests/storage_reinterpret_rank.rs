use num_complex::Complex32;
use tenferro_tensor::{Rank, TypedTensor, TypedTensorView};

#[test]
fn reinterpretation_publishes_dynamic_rank_after_component_axis_change() {
    let tensor = TypedTensor::<Complex32, Rank<2>>::from_vec_col_major(
        [2, 1],
        vec![Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)],
    )
    .unwrap();
    let real = tensor.into_real().unwrap();

    let _: &[usize] = real.shape();
    assert_eq!(real.rank(), 3);
    assert_eq!(real.shape(), &[2, 2, 1]);
}

#[test]
fn scalar_and_empty_shapes_keep_checked_layouts() {
    let scalar_owner = TypedTensor::<Complex32>::from_vec_col_major(
        Vec::<usize>::new(),
        vec![Complex32::new(5.0, 6.0)],
    )
    .unwrap();
    let scalar = scalar_owner.as_real_view().unwrap();
    assert_eq!(scalar.shape(), &[2]);
    assert_eq!(scalar.get(&[0]), Some(&5.0));
    assert_eq!(scalar.get(&[1]), Some(&6.0));

    let empty = TypedTensorView::from_slice([0], [1], 0, &[] as &[Complex32])
        .unwrap()
        .as_real_view()
        .unwrap();
    assert_eq!(empty.shape(), &[2, 0]);
    assert_eq!(empty.n_elements(), 0);
}

#[test]
fn invalid_real_to_complex_layout_is_rejected() {
    let data = [1.0_f32, 2.0, 3.0];
    let error = TypedTensorView::from_slice([3], [1], 0, &data)
        .unwrap()
        .as_complex_view()
        .unwrap_err();
    assert!(error.to_string().contains("leading extent"));
}
