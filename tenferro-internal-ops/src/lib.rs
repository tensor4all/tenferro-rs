pub mod ad;
pub mod dim_expr;
pub mod ext_op;
pub mod input_key;
pub mod shape_extent;
pub mod std_tensor_op;
pub mod sym_dim;

pub use ad::context::{ShapeGuard, ShapeGuardContext, TensorMeta};
#[cfg(not(feature = "autodiff"))]
pub use ext_op::ExtensionOp;
#[cfg(feature = "autodiff")]
pub use ext_op::{
    is_extension_rule_registered, linearize_extension_rule, lookup_extension_rule,
    register_extension_chain_rule, register_extension_rule, transpose_extension_rule, AdValue,
    ExtensionAdRule, ExtensionChainRule, ExtensionOp, ExtensionRegistryError, FruleBuilder,
    RRuleBuilder,
};
pub use shape_extent::ShapeExtent;
pub use sym_dim::SymDim;
pub use tenferro_extension_macros::ExtensionFamilyId;
pub use tenferro_tensor::config;

#[cfg(test)]
mod tests;
