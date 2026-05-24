//! Concrete tensor operations.
//!
//! The core `tenferro` crate owns the tensor type and backend traits. Extension
//! crates such as `tenferro-einsum` expose operation-specific eager helpers.

pub use tenferro_tensor::Tensor;
