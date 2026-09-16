//! Directed conversions between bfloat16 and f32.
//!
//! #1785 asks for explicit conversions in both directions with stated rounding and range
//! behaviour. Widening is exact, narrowing rounds to nearest with ties to even, and neither
//! direction implies a promotion rule.

use tenferro_bf16_proof::conversion::{narrow, widen};
use tenferro_bf16_proof::Bf16;
use tenferro_tensor_core::HostTensor;

fn bf16(values: &[f32]) -> HostTensor<Bf16> {
    HostTensor::from_vec_col_major(
        vec![values.len()],
        values.iter().copied().map(Bf16::from_f32).collect(),
    )
    .expect("shape matches data")
}

fn scalar(value: f32) -> HostTensor<f32> {
    HostTensor::from_vec_col_major(vec![1], vec![value]).expect("shape matches data")
}

#[test]
fn widening_is_exact_and_narrowing_rounds_to_nearest() {
    // Every bfloat16 is an f32, so widening a stored value returns exactly what was stored.
    // Bfloat16 keeps eight bits of significand, so its spacing on [1, 2) is 2^-7, and
    // 1.00390625 = 1 + 2^-8 is exactly the midpoint between 1.0 and 1.0 + 2^-7. Ties go to the
    // even significand, so it stores as 1.0.
    let stored = bf16(&[1.0, 1.0 + 2f32.powi(-8), -2.5]);
    let widened = widen(&stored).expect("widen");
    assert_eq!(widened.as_slice(), &[1.0_f32, 1.0, -2.5]);

    // 1.0078125 = 1 + 2^-7 is the next representable value, and 1.002 is below the midpoint
    // between the two, so it rounds down; the representable value itself is kept exactly.
    let rounded = narrow(&scalar(1.002)).expect("narrow");
    assert_eq!(rounded.as_slice()[0].to_f32(), 1.0);
    let exact = narrow(&scalar(1.0 + 2f32.powi(-7))).expect("narrow");
    assert_eq!(exact.as_slice()[0].to_f32(), 1.0 + 2f32.powi(-7));
}

#[test]
fn rounding_boundaries_follow_ties_to_even() {
    // The spacing of bfloat16 on [1, 2) is 2^-7, so the midpoints are exact ties.
    let tie_low = narrow(&scalar(1.0 + 2f32.powi(-8))).expect("narrow");
    assert_eq!(
        tie_low.as_slice()[0].to_f32(),
        1.0,
        "a tie between 1.0 and 1.0 + 2^-7 goes to the even significand"
    );

    // The next midpoint is between 1.0 + 2^-7 and 1.0 + 2^-6, whose significands are odd and
    // even respectively, so this tie goes up.
    let tie_high = narrow(&scalar(1.0 + 3.0 * 2f32.powi(-8))).expect("narrow");
    assert_eq!(
        tie_high.as_slice()[0].to_f32(),
        1.0 + 2f32.powi(-6),
        "the tie above 1.0 + 2^-7 goes to the even significand"
    );
}

#[test]
fn a_round_trip_through_f32_is_stable() {
    let stored = bf16(&[0.0, 1.0, -1.5, 256.0, 1.0 + 2f32.powi(-8)]);
    let round_tripped = narrow(&widen(&stored).expect("widen")).expect("narrow");
    assert_eq!(
        round_tripped.as_slice(),
        stored.as_slice(),
        "narrowing a widened bfloat16 returns the same value"
    );
}

#[test]
fn special_values_and_range_survive_both_directions() {
    // The bottom of the range is preserved exactly, because bfloat16 shares the exponent range of
    // f32 and 2^-126 is a power of two.
    let widened = widen(&bf16(&[f32::MIN_POSITIVE])).expect("widen");
    assert_eq!(widened.as_slice()[0], f32::MIN_POSITIVE);

    // The top of the range is not: narrowing rounds to nearest with ties to even, and the
    // significand of `f32::MAX` is all ones, so the rounding carries past the largest finite
    // bfloat16 and the conversion becomes infinity. That is the documented rounding behaviour
    // rather than an unnoticed loss, and the test states it.
    assert_eq!(Bf16::from_f32(f32::MAX).to_f32(), f32::INFINITY);
    assert_eq!(
        half::bf16::MAX.to_f32(),
        Bf16::from_f32(half::bf16::MAX.to_f32()).to_f32()
    );

    let infinite = narrow(
        &HostTensor::from_vec_col_major(vec![2], vec![f32::INFINITY, f32::NEG_INFINITY])
            .expect("shape"),
    )
    .expect("narrow");
    assert!(infinite.as_slice()[0].to_f32().is_infinite());
    assert!(infinite.as_slice()[1].to_f32().is_infinite());

    let not_a_number = narrow(&scalar(f32::NAN)).expect("narrow");
    assert!(not_a_number.as_slice()[0].to_f32().is_nan());
}
