//! Helpers shared by the integration tests.

/// Distance in units in the last place between two finite `f32` values of
/// the same sign, counted in representable numbers.
pub fn ulps_f32(a: u32, b: u32) -> u32 {
    let (a, b) = (f32::from_bits(a), f32::from_bits(b));
    assert!(a.is_finite() && b.is_finite(), "{a} or {b} is not finite");
    assert!(
        a == 0.0 || b == 0.0 || a.is_sign_negative() == b.is_sign_negative(),
        "{a} and {b} differ in sign"
    );
    (a.abs().to_bits()).abs_diff(b.abs().to_bits())
}

/// The `f64` version of [`ulps_f32`].
pub fn ulps_f64(a: u64, b: u64) -> u64 {
    let (a, b) = (f64::from_bits(a), f64::from_bits(b));
    assert!(a.is_finite() && b.is_finite(), "{a} or {b} is not finite");
    assert!(
        a == 0.0 || b == 0.0 || a.is_sign_negative() == b.is_sign_negative(),
        "{a} and {b} differ in sign"
    );
    (a.abs().to_bits()).abs_diff(b.abs().to_bits())
}
