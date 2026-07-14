//! Tensor4All-maintained native TBLIS build and link directives.
//!
//! This crate is intended to be used as a source/link provider alongside an
//! FFI crate. Enabling `build_from_source` builds a pinned TBLIS revision;
//! enabling `static` requests static native link directives.
//!
//! # Examples
//!
//! Keep the provider crate linked in the final Rust crate graph:
//!
//! ```
//! use t4a_tblis_src as _;
//! ```
