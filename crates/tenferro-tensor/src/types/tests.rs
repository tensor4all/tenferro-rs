use super::*;

#[test]
fn inline_metadata_collection_keeps_small_shapes_and_strides_inline() {
    let shape = shape_vec(&[2, 3]);
    let strides = stride_vec(&[1, 2]);

    assert_eq!(shape.as_slice(), &[2, 3]);
    assert_eq!(strides.as_slice(), &[1, 2]);
    assert!(!shape.is_heap());
    assert!(!strides.is_heap());
}

#[test]
fn erased_tensor_size_stays_within_the_documented_bound() {
    // The inline representation is deliberate: an erased `Tensor` must not
    // allocate for its metadata, so its size is the price paid on every move.
    // #1823 recorded 1464 B at its baseline and sliced the duplicated metadata
    // down to 776 B; keep the bound here so a future field addition has to
    // justify the bytes instead of silently regressing them.
    const ERASED_TENSOR_SIZE_BOUND: usize = 776;
    let size = size_of::<Tensor>();
    assert!(
        size <= ERASED_TENSOR_SIZE_BOUND,
        "erased Tensor grew to {size} B, above the {ERASED_TENSOR_SIZE_BOUND} B bound"
    );
    assert_eq!(align_of::<Tensor>(), 8);
    // The payload tag lives in a niche, so `Option<Tensor>` costs no extra word.
    assert_eq!(size_of::<Option<Tensor>>(), size);
}

// #1946 F7: view transforms keep the representation marker, so a `Host`
// view stays statically host-backed (the explicit annotations are the test).
#[test]
fn view_transforms_preserve_host_representation_marker() -> crate::Result<()> {
    let mut tensor =
        TypedTensor::<f64, Rank<2>, Host>::from_host_vec_col_major([2, 2], vec![1., 2., 3., 4.])?;

    let view: TypedTensorView<'_, f64, Rank<2>, Host> = tensor.as_view();
    let reshaped: TypedTensorView<'_, f64, DynRank, Host> = view.reshape_view(&[4])?;
    assert_eq!(reshaped.as_host_slice(), &[1., 2., 3., 4.]);
    assert_eq!(view.host_col_major_view()?.get([1, 0]), Some(&2.));
    assert_eq!(tensor.host_col_major_view()?.get([0, 1]), Some(&3.));

    let mut view_mut: TypedTensorViewMut<'_, f64, Rank<2>, Host> = tensor.as_view_mut();
    {
        let reshaped_mut: TypedTensorViewMut<'_, f64, DynRank, Host> =
            view_mut.reshape_view(&[4])?;
        let _ = reshaped_mut;
    }
    let read_only: TypedTensorView<'_, f64, Rank<2>, Host> = view_mut.as_read_only();
    assert_eq!(read_only.as_host_slice(), &[1., 2., 3., 4.]);
    let into_read_only: TypedTensorView<'_, f64, Rank<2>, Host> = view_mut.into_read_only();
    assert_eq!(into_read_only.as_host_slice().len(), 4);

    if let Some(value) = tensor.host_col_major_view_mut()?.get_mut([1, 1]) {
        *value = 9.;
    }
    assert_eq!(tensor.as_slice(), &[1., 2., 3., 9.]);
    Ok(())
}
