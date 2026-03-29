mod einsum;
pub mod meta;

pub(crate) use einsum::{
    accumulate_tangent as accumulate_structured_tangent, compress_dense_to_layout_in_ctx,
    einsum_with_subscripts_in_ctx, reverse_subscripts, to_dense_in_ctx,
};
pub(crate) use tenferro_internal_frontend_core::StructuredTensor;
pub(crate) use tenferro_tensor::structured_tensor::canonicalize_axis_classes;
