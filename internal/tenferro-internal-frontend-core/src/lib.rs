//! Internal implementation crate. Not a stable public API.
//!
//! # Examples
//!
//! ```text
//! // This crate is wired through tenferro surface crates and is not intended
//! // to be consumed directly.
//! ```

mod autodiff;
mod dyn_tensor;
mod scalar_type;
mod scalar_value;
pub mod snapshot;
mod structured_einsum;
mod structured_meta;
mod structured_tensor;
#[doc(hidden)]
pub mod tensor_ops;

pub use dyn_tensor::DynTensor;
#[doc(hidden)]
pub use dyn_tensor::DynTensorTyped;
#[doc(hidden)]
pub use scalar_type::AbsAsF64;
pub use scalar_type::ScalarType;
pub use scalar_value::ScalarValue;
pub use structured_einsum::{
    accumulate_tangent, compress_dense_to_layout_in_ctx, einsum_with_subscripts_in_ctx,
    reverse_subscripts, to_dense_in_ctx,
};
#[doc(hidden)]
pub use structured_einsum::{
    first_duplicate_pair, normalize_payload_for_roots, unique_ids_first_appearance,
    usize_vec_to_u32, StructuredDenseEinsumBackend, StructuredEinsumRuntimeValue,
};
pub use structured_meta::{
    plan_axis_classes_for_subscripts, AxisClassMergePlan, AxisClassPlanError, OperandAxisClassPlan,
    OperandAxisClasses,
};
pub use structured_tensor::StructuredTensor;

#[cfg(test)]
mod tests;
