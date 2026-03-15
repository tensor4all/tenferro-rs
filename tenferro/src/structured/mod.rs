mod autodiff;
mod einsum;
mod layout;
pub mod meta;
mod placement;

pub(crate) use einsum::{
    accumulate_tangent as accumulate_structured_tangent, compress_dense_to_layout_in_ctx,
    einsum_with_subscripts_in_ctx, reverse_subscripts, to_dense_in_ctx,
};
pub(crate) use layout::canonicalize_axis_classes;
pub(crate) use layout::StructuredTensor;
