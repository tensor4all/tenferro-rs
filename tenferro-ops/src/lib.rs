pub mod ad;
pub mod dim_expr;
pub mod ext_op;
pub mod input_key;
pub mod std_tensor_op;
pub mod sym_dim;

pub use ad::context::{ShapeGuard, ShapeGuardContext, TensorMeta};
pub use ext_op::{
    is_extension_registered, lookup_extension_factory, register_extension, ExtensionFactory,
    ExtensionOp, ExtensionRegistryError,
};
pub use sym_dim::SymDim;
pub use tenferro_tensor::config;

#[cfg(test)]
mod tests;
