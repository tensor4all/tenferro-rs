use crate::cubecl::dispatch::cubecl_shape_and_strides;

#[test]
fn cubecl_metadata_uses_dense_column_major_strides() {
    assert_eq!(cubecl_shape_and_strides(&[]), (vec![], vec![]));
    assert_eq!(
        cubecl_shape_and_strides(&[2, 3, 4]),
        (vec![2, 3, 4], vec![1, 2, 6])
    );
}
