//! Tropical automatic differentiation via argmax routing.
//!
//! Tropical operations (max/min-based) are not smooth, so standard AD does not
//! apply. Instead, this module uses a subgradient-style route-through-winner
//! policy: the forward pass records which input element won each tropical
//! addition, and reverse/forward rules route derivatives through that winner.
//!
//! # Architecture
//!
//! - [`tropical_einsum_rrule`]: standalone reverse-mode rule
//! - [`tropical_einsum_frule`]: standalone forward-mode rule
//! - [`promote_to_tropical`] / [`extract_inner`]: conversions between standard
//!   and tropical scalar tensors

mod backward;
mod common;
mod convert;
mod forward;
mod rules;
mod scalar;

pub use convert::{extract_inner, promote_to_tropical};
pub use rules::{tropical_einsum_frule, tropical_einsum_rrule};
pub use scalar::TropicalScalar;

#[cfg(test)]
mod tests;
