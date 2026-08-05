use super::physical_output_descriptor;

#[test]
fn physical_output_descriptor_matches_transposed_destination() {
    let (extents, strides, modes) = physical_output_descriptor("test", &[4, 3], &[3, 1]).unwrap();
    assert_eq!(extents, [3, 4]);
    assert_eq!(strides, [1, 3]);
    assert_eq!(modes, [1, 0]);
}

#[test]
fn physical_output_descriptor_preserves_identity_layout() {
    let (extents, strides, modes) = physical_output_descriptor("test", &[4, 3], &[1, 4]).unwrap();
    assert_eq!(extents, [4, 3]);
    assert_eq!(strides, [1, 4]);
    assert_eq!(modes, [0, 1]);
}

#[test]
fn physical_output_descriptor_rejects_negative_strides() {
    let err = physical_output_descriptor("test", &[4], &[-1]).unwrap_err();
    assert!(err.to_string().contains("negative strides"));
}
