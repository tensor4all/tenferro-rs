use std::collections::TryReserveError;
use std::error::Error as _;

use super::reserve_download_destination;

#[test]
fn host_reservation_failure_is_typed_and_preserves_the_buffer() {
    let mut buffer = vec![1.0_f64, 2.0, 3.0];
    let pointer = buffer.as_ptr();
    let capacity = buffer.capacity();
    // Capacity overflow deterministically refuses without attempting a huge
    // allocation, on both debug and release builds and without CUDA hardware.
    let error = reserve_download_destination(&mut buffer, usize::MAX, "download").unwrap_err();
    assert!(matches!(
        error,
        crate::Error::BackendSource { op: "download", .. }
    ));
    assert!(error
        .source()
        .unwrap()
        .downcast_ref::<TryReserveError>()
        .is_some());
    assert_eq!(buffer, [1.0, 2.0, 3.0]);
    assert_eq!(buffer.as_ptr(), pointer);
    assert_eq!(buffer.capacity(), capacity);
}

#[test]
fn host_reservation_reuses_capacity_and_grows_without_initializing() {
    let mut buffer = Vec::with_capacity(16);
    buffer.extend_from_slice(&[1_u32, 2, 3]);
    let pointer = buffer.as_ptr();
    for len in [0, 2, 8, 16] {
        reserve_download_destination(&mut buffer, len, "download").unwrap();
        assert_eq!(buffer.as_ptr(), pointer);
        assert_eq!(buffer, [1, 2, 3]);
    }
    reserve_download_destination(&mut buffer, 32, "download").unwrap();
    assert!(buffer.capacity() >= 32);
    assert_eq!(buffer, [1, 2, 3]);
}
