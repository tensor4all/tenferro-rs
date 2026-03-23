//! CUDA runtime substrate for `tenferro-device`.
//!
//! The implementation is split into smaller responsibility-focused modules.

mod kernels;
mod memory;
mod metadata;
mod pointwise;
mod rng;
mod shared;
mod state;
mod structural;

use kernels::*;
pub use shared::*;
pub(crate) use shared::{
    MetadataBinaryOp, MetadataBinarySpec, MetadataGenerateOp, MetadataGenerateSpec,
    MetadataReductionOp, MetadataReductionSpec, MetadataTensorMut, MetadataTensorRef,
    MetadataTernaryOp, MetadataTernarySpec,
};
pub use state::*;
