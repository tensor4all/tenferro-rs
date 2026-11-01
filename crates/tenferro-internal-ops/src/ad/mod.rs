//! Primitive-rule vocabulary shared by semantic AD transforms and extension rules.
//!
//! Core differentiation is implemented on semantic programs in `tenferro-ad`.
//! This facade preserves the shared metadata, builder and support paths; it does
//! not dispatch or implement core graph differentiation rules.

pub mod context;
#[cfg(feature = "autodiff")]
#[doc(hidden)]
pub mod support;
#[cfg(feature = "autodiff")]
mod zeros;

pub use crate::primitive_rules::ResidualSpec;
#[cfg(feature = "autodiff")]
pub use crate::primitive_rules::{
    primitive_residual_spec, ADRuleError, ADRuleKind, ADRuleResult, ExtensionAdDispatcher,
    PrimitiveRuleBuilder, PrimitiveTransposeInput,
};

#[cfg(test)]
mod tests;
