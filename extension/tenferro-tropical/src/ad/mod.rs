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
//! - [`tracked_tropical_einsum`]: tape-aware tropical einsum
//! - [`promote_to_tropical`] / [`extract_inner`]: conversions between standard
//!   and tropical scalar tensors

mod backward;
mod common;
mod forward;
mod rules;
mod scalar;
mod tracked;

pub use rules::{tropical_einsum_frule, tropical_einsum_rrule};
pub use scalar::TropicalScalar;
pub use tracked::{
    extract_inner, promote_to_tropical, tracked_tropical_einsum, TropicalEinsumReverseRule,
};

#[cfg(test)]
mod tests;
