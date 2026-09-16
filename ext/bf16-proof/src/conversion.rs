//! Directed conversions between bfloat16 storage and `f32`.
//!
//! #1785 asks for explicit bf16/f32 conversions in both directions with rounding and range
//! behaviour that is stated rather than implied. Widening is exact, because every bfloat16 value
//! is an `f32`; narrowing rounds to the nearest bfloat16, with ties to even, which is what
//! [`half::bf16::from_f32`] does. Neither direction is a promotion rule, and neither is implicit.

use crate::Bf16;
use tenferro_tensor_core::{HostTensor, ValidationError};

/// Widen every stored value to `f32`, which is exact.
///
/// # Errors
///
/// Returns [`ValidationError`] when the output shape cannot be built, which cannot happen for a
/// shape that already exists.
///
/// # Examples
///
/// ```rust
/// use tenferro_bf16_proof::{conversion::widen, Bf16};
/// use tenferro_tensor_core::HostTensor;
///
/// let source = HostTensor::from_vec_col_major(vec![2], vec![Bf16::from_f32(1.0), Bf16::from_f32(2.5)])?;
/// let widened = widen(&source)?;
/// assert_eq!(widened.as_slice(), &[1.0_f32, 2.5]);
/// # Ok::<(), tenferro_tensor_core::ValidationError>(())
/// ```
pub fn widen(source: &HostTensor<Bf16>) -> Result<HostTensor<f32>, ValidationError> {
    HostTensor::from_vec_col_major(
        source.shape().to_vec(),
        source
            .as_slice()
            .iter()
            .map(|value| value.to_f32())
            .collect(),
    )
}

/// Round every value to the nearest bfloat16, with ties to even.
///
/// # Errors
///
/// Returns [`ValidationError`] when the output shape cannot be built, which cannot happen for a
/// shape that already exists.
///
/// # Examples
///
/// ```rust
/// use tenferro_bf16_proof::conversion::narrow;
/// use tenferro_tensor_core::HostTensor;
///
/// // 1.00390625 rounds down to 1.0: bfloat16 keeps eight bits of significand.
/// let source = HostTensor::from_vec_col_major(vec![1], vec![1.00390625_f32])?;
/// assert_eq!(narrow(&source)?.as_slice()[0].to_f32(), 1.0);
/// # Ok::<(), tenferro_tensor_core::ValidationError>(())
/// ```
pub fn narrow(source: &HostTensor<f32>) -> Result<HostTensor<Bf16>, ValidationError> {
    HostTensor::from_vec_col_major(
        source.shape().to_vec(),
        source
            .as_slice()
            .iter()
            .map(|value| Bf16::from_f32(*value))
            .collect(),
    )
}
