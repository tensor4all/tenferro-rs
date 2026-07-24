//! Internal tensor operation vocabulary shared by tenferro graph crates.
//!
//! This crate owns symbolic dimension expressions, extension-op traits, shape
//! guards, and the standard tensor op enum used by traced execution and AD
//! lowering. End users normally interact through `tenferro-runtime` and
//! operation-family crates rather than importing this crate directly.
//!
//! # Examples
//!
//! ```
//! use tenferro_ops::{ShapeExtent, SymDim};
//! use tenferro_ops::std_tensor_op::StdTensorOp;
//!
//! let op = StdTensorOp::constant(2.0_f64);
//! let extent = ShapeExtent::exact(SymDim::from(3usize));
//! assert!(matches!(op, StdTensorOp::Constant { .. }));
//! assert!(extent.is_exact());
//! ```
//!
//! Collected shape constraints are inference-driver internals rather than a
//! crate-root extension-author API:
//!
//! ```compile_fail
//! use tenferro_ops::ExtensionShapeConstraint;
//! ```
//!
pub mod ad;
pub mod axis;
pub mod broadcast;
pub mod dim_expr;
pub mod ext_op;
pub mod input_key;
pub mod reduction;
#[doc(hidden)]
pub mod shape_constraint;
pub mod shape_extent;
pub mod std_tensor_op;
pub mod sym_dim;

pub use ad::context::{
    ShapeGuard, ShapeGuardContext, ShapeGuardError, ShapeGuardFailure, ShapeGuardResult, TensorMeta,
};
pub use ext_op::{ExtensionOp, HostReference};
pub use shape_constraint::{ExtensionShapeContext, ExtensionShapeError, ShapeRelation};
pub use shape_extent::ShapeExtent;
pub use sym_dim::{SymDim, SymDimConversionError};
pub use tenferro_extension_macros::ExtensionFamilyId;
pub use tenferro_tensor::config;

#[cfg(test)]
mod tests;
