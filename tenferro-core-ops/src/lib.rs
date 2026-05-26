//! Core primitive operation catalog for tenferro.
//!
//! This crate intentionally excludes standard extension families such as
//! linalg, FFT, and einsum.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_core_ops::{descriptor, PrimitiveOpKind};
//!
//! let add = descriptor(PrimitiveOpKind::Add);
//! assert_eq!(add.name, "add");
//! ```

mod catalog;

pub use catalog::{
    all_primitive_descriptors, descriptor, DTypePolicy, OpCategory, PrimitiveOpDescriptor,
    PrimitiveOpKind,
};
