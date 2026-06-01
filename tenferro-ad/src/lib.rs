//! Automatic differentiation APIs for tenferro.
//!
//! This crate is the explicit opt-in boundary for traced and eager automatic
//! differentiation. Primal graph construction and execution live in
//! `tenferro-runtime`; tensor storage lives in `tenferro-tensor`, and CPU
//! execution lives in `tenferro-cpu`.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_ad::AdContext;
//! use tenferro_runtime::TracedTensor;
//!
//! let ad = AdContext::builder().with_core_rules().build().unwrap();
//! let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]);
//! let loss = &x * &x;
//! let dx = ad.grad(&loss, &x).unwrap();
//! assert_eq!(dx.rank, 0);
//! ```

mod context;
mod eager;
mod eager_backend;
mod eager_builder;
pub mod eager_exec;
pub(crate) mod eager_ops;
pub(crate) mod eager_ops_elementwise;
pub mod eager_tensor;
pub mod extension;
mod shape_packing;
pub mod traced;

pub use context::{AdContext, AdContextBuilder};
pub use eager::{EagerRuntime, EagerRuntimeCacheStats, EagerTensor};
pub use eager_backend::EagerBackend;
pub use tenferro_runtime::{extension_cache, extension_runtime, scalar_semantics, shape_infer};
pub use tenferro_runtime::{
    CompareDir, DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
    Tensor,
};
pub use traced::TracedTensorAdExt;

pub use tenferro_runtime::{ContextId, Error, Result};

pub mod error {
    pub use tenferro_runtime::{ContextId, Error, Result};
}

pub mod metadata {
    pub use tenferro_runtime::ad_support::{
        metadata_scopes_for_scope, metadata_scopes_with_new, metadata_scopes_with_scope,
        push_metadata_scope, register_scoped_graph_metadata, register_scoped_live_graph_metadata,
        register_scoped_metadata_batch, register_scoped_value_metadata, tensor_meta_from_tensor,
        MetadataScope,
    };
}
