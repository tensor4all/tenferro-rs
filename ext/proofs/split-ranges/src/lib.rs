//! Creusot proof harness for `tenferro_tensor`'s contiguous mutable split
//! (#2682).
//!
//! The verified code is not a copy: this crate compiles the shipped module
//! `crates/tenferro-tensor/src/types/split_ranges.rs` directly, so the proved
//! contracts are the ones the kernels run against. Proving the whole
//! `tenferro-tensor` crate is not an option today because its derived
//! `PartialEq`/`Debug` impls need `DeepModel` under `cfg(creusot)`, which is
//! outside this change's scope.
//!
//! Run `cargo creusot` from this directory (Creusot 0.15).

// The included module's specifications only name `creusot_std` under
// `cfg(creusot)`; this keeps the dependency used in a plain `cargo build`.
use creusot_std as _;

// The included module has no callers here: Creusot emits a verification
// condition per contract, so the functions need not be reachable.
#[allow(dead_code)]
#[path = "../../../../crates/tenferro-tensor/src/types/split_ranges.rs"]
mod split_ranges;
