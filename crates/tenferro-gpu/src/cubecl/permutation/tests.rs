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

#[test]
fn device_address_alignment_reports_the_shifted_pointer_alignment() {
    // A 256-byte aligned allocation base keeps the allocation alignment.
    assert_eq!(super::device_address_alignment(0x7f00_0000_0000_0000), 256);
    // An f64 region three elements into that allocation guarantees 8 bytes.
    assert_eq!(super::device_address_alignment(0x7f00_0000_0000_0018), 8);
    // An f32 region one element in guarantees 4 bytes.
    assert_eq!(super::device_address_alignment(0x7f00_0000_0000_0004), 4);
    // A complex64 region one element in guarantees 16 bytes.
    assert_eq!(super::device_address_alignment(0x7f00_0000_0000_0010), 16);
    // A null pointer never reaches a launch; report the allocation alignment.
    assert_eq!(super::device_address_alignment(0), 256);
}
