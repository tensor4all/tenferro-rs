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
//!
//! # Errors
//!
//! [`Error`] (the runtime error type, re-exported here) has public
//! constructors for the failures downstream code reports itself:
//! [`Error::invalid_argument`], [`Error::unsupported`],
//! [`Error::dtype_mismatch`], [`Error::validation`], [`Error::runtime_state`],
//! and [`Error::extension`] for a typed source error. Each takes the operation
//! name and an [`ErrorPhase`]. A `tenferro_tensor::Error` converts with
//! `From`, so `?` works on tensor-level results inside functions returning
//! [`Result`]. Match on [`Error::kind`] rather than on variant shapes.
//!
//! ```rust
//! use tenferro_ad::{Error, ErrorPhase};
//!
//! fn check_rank(rank: usize) -> tenferro_ad::Result<()> {
//!     if rank != 2 {
//!         return Err(Error::invalid_argument(
//!             "my_crate::attention",
//!             ErrorPhase::Execution,
//!             "query",
//!             format!("expected a rank-2 query, got rank {rank}"),
//!         ));
//!     }
//!     Ok(())
//! }
//!
//! let error = check_rank(3).unwrap_err();
//! assert_eq!(
//!     error.kind(),
//!     tenferro_tensor::ErrorKind::Validation(tenferro_tensor::ValidationKind::InvalidArgument)
//! );
//! let unsupported = Error::unsupported("my_crate::op", ErrorPhase::Execution, "no GPU path yet");
//! assert_eq!(unsupported.kind(), tenferro_tensor::ErrorKind::Unsupported);
//! let tensor_error: Error = tenferro_tensor::Error::invalid_argument("op", "arg", "bad").into();
//! assert!(matches!(tensor_error.kind(), tenferro_tensor::ErrorKind::Validation(_)));
//! ```

mod context;
mod eager;
mod eager_backend;
pub(crate) mod eager_exec;
pub(crate) mod eager_ops;
pub mod extension;
pub mod prelude;
// semantic_compat removed in Unification 7.
pub mod semantic_extension;
pub mod semantic_transform;
mod shape_packing;
pub mod traced;
mod transform_cache;

pub use context::{AdContext, AdContextBuilder, AdContextCacheStats};
pub use eager::{
    EagerNoGradGuard, EagerRuntime, EagerRuntimeCacheStats, EagerSession, EagerTensor,
    EagerTraceCaptureGuard, GradientValue, Gradients, IntoValueError, ValueGuard,
};
pub use shape_packing::EagerSliceBuilder;
pub(crate) use tenferro_runtime::scalar_semantics;
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

pub use tenferro_runtime::{ContextId, Error, ErrorPhase, Result};

pub mod error {
    pub use tenferro_runtime::{ContextId, Error, ErrorPhase, Result};
}

pub(crate) mod metadata {
    pub use tenferro_runtime::ad_support::tensor_meta_from_tensor;
}
