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
pub mod ad;
pub mod axis;
pub mod broadcast;
pub mod dim_expr;
pub mod ext_op;
pub mod input_key;
pub mod reduction;
pub mod shape_extent;
pub mod std_tensor_op;
pub mod sym_dim;

pub use ad::context::{ShapeGuard, ShapeGuardContext, TensorMeta};
#[cfg(not(feature = "autodiff"))]
pub use ext_op::ExtensionOp;
#[cfg(feature = "autodiff")]
pub use ext_op::{
    is_extension_rule_registered, linearize_extension_rule, lookup_extension_rule,
    register_extension_rule, transpose_extension_rule, ExtensionAdRule, ExtensionOp,
    ExtensionRegistryError, ExtensionRuleSet,
};
pub use shape_extent::ShapeExtent;
pub use sym_dim::SymDim;
pub use tenferro_extension_macros::ExtensionFamilyId;
pub use tenferro_tensor::config;

#[cfg(test)]
mod tests;
