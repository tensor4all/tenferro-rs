//! Scalar AD helper rules for elementary operations.
//!
//! This crate provides stateless primal/frule/rrule helpers for scalar ops
//! used by wrapper-level AD APIs.
//!
//! Public helper families:
//!
//! - `add`, `sub`, `mul`, `div`
//! - `conj`
//! - `sqrt`
//! - `exp`, `log`
//! - `powf` (fixed real exponent)
//! - `powi` (fixed integer exponent)
//! - `atan2` (real scalars)
//!
//! The [`ScalarAd`] trait also provides the scalar method basis used by the
//! higher-level wrappers, including `expm1`, `log1p`, `sin`, `cos`, and
//! `tanh`.
//!
//! # Examples
//!
//! ```rust
//! use chainrules_scalarops::{powf_frule, powf_rrule};
//!
//! let (y, dy) = powf_frule(2.0_f64, 3.0, 1.0);
//! assert_eq!(y, 8.0);
//! assert_eq!(dy, 12.0);
//!
//! let dx = powf_rrule(2.0_f64, 3.0, 1.0);
//! assert_eq!(dx, 12.0);
//! ```

mod binary;
mod power;
mod real_ops;
mod scalar_ad;
mod unary;

#[doc(inline)]
pub use binary::{
    add, add_frule, add_rrule, div, div_frule, div_rrule, mul, mul_frule, mul_rrule, sub,
    sub_frule, sub_rrule,
};
#[doc(inline)]
pub use power::{powf, powf_frule, powf_rrule, powi, powi_frule, powi_rrule};
#[doc(inline)]
pub use real_ops::{atan2, atan2_frule, atan2_rrule};
#[doc(inline)]
pub use scalar_ad::{handle_r_to_c_f32, handle_r_to_c_f64, ScalarAd};
#[doc(inline)]
pub use unary::{
    conj, conj_frule, conj_rrule, exp, exp_frule, exp_rrule, log, log_frule, log_rrule, sqrt,
    sqrt_frule, sqrt_rrule,
};

#[cfg(test)]
mod tests;
