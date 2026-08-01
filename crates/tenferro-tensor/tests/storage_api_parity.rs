use tenferro_tensor::{
    Tensor, TensorView, TensorViewMut, TypedTensor, TypedTensorView, TypedTensorViewMut,
};

// This macro is the one canonical direct read-only method intersection shared
// by every storage owner/view family in this parity probe. The assertions are
// deliberately meaningful for the same compact 2x2 layout: shape, linear
// offset, compactness, the textual layout contract, and the checked compactness
// assertion all agree.
// `get`, dtype, placement, host reads, consuming methods, and mutation methods
// remain outside this list because their current signatures/surfaces are not
// identical across all six families. #1559 owns their final convergence.
macro_rules! assert_common_read_only_methods {
    ($value:expr) => {{
        let value = $value;
        assert_eq!(value.shape(), &[2, 2]);
        assert_eq!(value.layout_linear_offset(&[1, 1])?, 3);
        assert!(value.is_col_major_contiguous()?);
        let summary = value.layout_summary();
        assert!(summary.contains("shape=[2, 2]"));
        assert!(summary.contains("strides=[1, 2]"));
        value.assert_col_major_contiguous()?;
    }};
}

fn assert_i32_view_value(view: &TensorView<'_>) {
    match view {
        TensorView::I32(view) => assert_eq!(view.get(&[1, 1]), Some(&4)),
        _ => panic!("parity fixture must construct an i32 view"),
    }
}

#[test]
fn common_read_only_methods_are_present_on_all_storage_faces() -> tenferro_tensor::Result<()> {
    let typed = TypedTensor::<i32>::from_vec_col_major(vec![2, 2], vec![1, 2, 3, 4])?;
    assert_common_read_only_methods!(&typed);
    assert_eq!(typed.get(&[1, 1])?, &4);

    let typed_data = [1, 2, 3, 4];
    let typed_view = TypedTensorView::from_slice([2, 2], [1, 2], 0, &typed_data)?;
    assert_common_read_only_methods!(&typed_view);
    assert_eq!(typed_view.get(&[1, 1]), Some(&4));

    let mut typed_mut_data = [1, 2, 3, 4];
    let typed_view_mut = TypedTensorViewMut::from_slice([2, 2], [1, 2], 0, &mut typed_mut_data)?;
    assert_common_read_only_methods!(&typed_view_mut);
    assert_eq!(typed_view_mut.get(&[1, 1]), Some(&4));

    let tensor = Tensor::from_vec_col_major(vec![2, 2], vec![1_i32, 2, 3, 4])?;
    assert_common_read_only_methods!(&tensor);
    assert_eq!(tensor.get::<i32>(&[1, 1])?, &4);

    let tensor_data = [1, 2, 3, 4];
    let tensor_view = TensorView::i32(&[2, 2], &tensor_data)?;
    assert_common_read_only_methods!(&tensor_view);
    assert_i32_view_value(&tensor_view);

    let mut tensor_mut_data = [1, 2, 3, 4];
    let tensor_view_mut = TensorViewMut::i32(&[2, 2], &mut tensor_mut_data)?;
    assert_common_read_only_methods!(&tensor_view_mut);
    assert_i32_view_value(&tensor_view_mut.as_read_only());

    Ok(())
}
