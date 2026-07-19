use super::*;

#[test]
fn inline_metadata_collection_keeps_small_shapes_and_strides_inline() {
    let shape = shape_vec(&[2, 3]);
    let strides = stride_vec(&[1, 2]);

    assert_eq!(shape.as_slice(), &[2, 3]);
    assert_eq!(strides.as_slice(), &[1, 2]);
    assert!(!shape.spilled());
    assert!(!strides.spilled());
}
