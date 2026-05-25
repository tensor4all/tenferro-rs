//! Traced tensor einsum operations.
//!
//! This module is the canonical traced tensor namespace for the einsum
//! extension crate. Root-level re-exports remain available for compatibility.

pub use crate::traced::{einsum, einsum_subscripts, einsum_subscripts_with, einsum_with};
