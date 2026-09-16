//! Reductions over bfloat16 storage.
//!
//! #1785 asks for sum reduction with a specified accumulation rule and for tests that distinguish
//! that rule from repeated rounding of the storage type. Two reductions do that here: one
//! accumulates in `f32` and rounds once, which is the contract this crate promises, and one
//! rounds at every step. The second is not offered as the contract; it exists so tests can measure
//! the difference instead of assuming it.

use crate::Bf16;

/// Sum a slice, accumulating in `f32` and rounding once at the end.
///
/// # Examples
///
/// ```rust
/// use tenferro_bf16_proof::{reduction::sum_in_f32_accumulation, Bf16};
///
/// let ones = vec![Bf16::from_f32(1.0); 300];
/// // Three hundred ones are exactly representable after rounding, because the accumulation kept
/// // the intermediate values in f32.
/// assert_eq!(sum_in_f32_accumulation(&ones).to_f32(), 300.0);
/// ```
#[must_use]
pub fn sum_in_f32_accumulation(values: &[Bf16]) -> Bf16 {
    let total: f32 = values.iter().map(|value| value.to_f32()).sum();
    Bf16::from_f32(total)
}

/// Sum a slice, rounding to bfloat16 after every step.
///
/// This is the weaker contract, and it is the one to avoid when the accumulation is promised in
/// `f32`: on a slice of three hundred ones it stalls at `256.0`, because from there on bfloat16
/// spacing above one is `2.0` and adding one rounds back down.
///
/// # Examples
///
/// ```rust
/// use tenferro_bf16_proof::{reduction::sum_with_per_step_rounding, Bf16};
///
/// let ones = vec![Bf16::from_f32(1.0); 300];
/// assert_eq!(sum_with_per_step_rounding(&ones).to_f32(), 256.0);
/// ```
#[must_use]
pub fn sum_with_per_step_rounding(values: &[Bf16]) -> Bf16 {
    values
        .iter()
        .fold(Bf16::zero(), |accumulator, value| accumulator + *value)
}

/// Product of a slice, accumulating in `f32` and rounding once.
///
/// # Examples
///
/// ```rust
/// use tenferro_bf16_proof::{reduction::product_in_f32_accumulation, Bf16};
///
/// let values = [Bf16::from_f32(3.0), Bf16::from_f32(4.0)];
/// assert_eq!(product_in_f32_accumulation(&values).to_f32(), 12.0);
/// ```
#[must_use]
pub fn product_in_f32_accumulation(values: &[Bf16]) -> Bf16 {
    let total: f32 = values.iter().map(|value| value.to_f32()).product();
    Bf16::from_f32(total)
}
