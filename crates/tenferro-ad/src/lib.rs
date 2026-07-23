//! Automatic differentiation APIs for tenferro.
//!
//! This crate is the explicit opt-in boundary for traced and eager automatic
//! differentiation. Primal graph construction and execution live in
//! `tenferro-runtime`; tensor storage lives in `tenferro-tensor`, and CPU
//! execution lives in `tenferro-cpu`.
//!
//! Use [`EagerRuntime`] and [`EagerTensor`] for PyTorch-style immediate
//! execution where tracked variables accumulate gradients after `backward()`.
//! Use [`TracedTensorAdExt`] or [`AdContext`] for JAX-style graph transforms
//! such as `grad`, `vjp`, and `jvp` on [`tenferro_runtime::TracedTensor`]
//! values. `AdContext` is the explicit place to add extension AD rule sets for
//! operation-family crates such as `tenferro-linalg`.
//!
//! User-facing guides live at
//! <https://tensor4all.org/tenferro-rs/guides/autodiff.html> and
//! <https://tensor4all.org/tenferro-rs/guides/choosing-an-api.html>.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_ad::AdContext;
//! use tenferro_runtime::TracedTensor;
//!
//! let ad = AdContext::builder().build().unwrap();
//! let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
//! let loss = (&x * &x).unwrap();
//! let dx = ad.grad(&loss, &x).unwrap();
//! assert_eq!(dx.rank, 0);
//! ```

mod ad_rule_error;
mod context;
mod eager;
mod eager_backend;
mod eager_builder;
pub(crate) mod eager_exec;
pub(crate) mod eager_ops;
pub(crate) mod eager_ops_elementwise;
pub mod extension;
pub mod semantic_extension;
pub mod semantic_transform;
mod shape_packing;
pub mod traced;
mod transform_cache;

pub use context::{AdContext, AdContextBuilder, AdContextCacheStats};
pub use eager::{
    CpuPlacementBoundEager, EagerNoGradGuard, EagerRuntime, EagerRuntimeCacheStats, EagerTensor,
};
pub use eager_backend::EagerBackend;
pub use shape_packing::EagerSliceBuilder;
pub(crate) use tenferro_runtime::{extension_cache, extension_runtime, scalar_semantics};
pub use transform_cache::AdTransformCacheLimits;
pub(crate) mod shape_infer {
    pub use tenferro_runtime::extension::{
        promote_dtype, promote_dtype_for_binary_op, promote_dtypes,
    };
}
pub use tenferro_runtime::{
    CompareDir, DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
    Tensor,
};
pub use traced::TracedTensorAdExt;

pub use tenferro_runtime::{ContextId, Error, Result};

pub mod error {
    pub use tenferro_runtime::{ContextId, Error, Result};
}

pub(crate) mod metadata {
    pub use tenferro_runtime::ad_support::{
        metadata_scopes_for_scope, push_metadata_scope, register_scoped_live_graph_metadata,
        register_scoped_metadata_batch, register_scoped_value_metadata, tensor_meta_from_tensor,
        GlobalMetadataScope,
    };
}
