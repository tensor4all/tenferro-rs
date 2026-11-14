//! `bool` payload handling.
//!
//! `bool` is the one wire scalar whose received bytes need validation: a `bool` must
//! hold 0 or 1, so a payload may only become `bool`s after every byte has been
//! checked. Sending needs no check, because every `bool` already is 0 or 1.

use super::{view_as_bytes, view_as_bytes_mut};

/// View `&[bool]` as bytes.
///
/// # Safety
///
/// `bool` is one byte with size and alignment equal to `u8`, and every `bool` is
/// already a valid `u8` value, so the view is always valid.
pub(super) fn as_bytes(slice: &[bool]) -> &[u8] {
    view_as_bytes(slice)
}

/// View `&mut [bool]` as mutable bytes.
///
/// Sound because every `bool` in the slice is already 0 or 1. A *receiver* must not use
/// this: writing a byte that is not 0 or 1 would leave an invalid `bool` behind, so
/// [`super::WireScalar::VALIDATES_PAYLOAD`] routes payloads through [`assign`] instead.
pub(super) fn as_bytes_mut(slice: &mut [bool]) -> &mut [u8] {
    view_as_bytes_mut(slice)
}

/// Fill `out` from a received byte buffer, and report whether every byte was 0 or 1.
///
/// On `false`, `out` is untouched: the destination never holds a byte that is not a
/// `bool`, which is what makes this sound where a byte view of the destination is not.
pub(super) fn assign(out: &mut [bool], bytes: &[u8]) -> bool {
    if bytes.len() != out.len() || !bytes.iter().all(|byte| *byte <= 1) {
        return false;
    }
    for (slot, byte) in out.iter_mut().zip(bytes) {
        *slot = *byte == 1;
    }
    true
}
