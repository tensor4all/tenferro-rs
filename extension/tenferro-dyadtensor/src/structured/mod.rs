mod layout;
pub mod meta;

pub use layout::StructuredTensor;
pub(crate) use layout::{canonicalize_axis_classes, validate_layout};
