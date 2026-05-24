//! Runtime extension dispatch and cache infrastructure for tenferro.
//!
//! This crate intentionally contains operation-agnostic runtime machinery:
//! extension cache storage and backend-parametric extension runtime dispatch.
//! Standard operations live in `tenferro-ops`; tensor storage and backend
//! kernels live in `tenferro-tensor`.

pub mod extension_cache;
pub mod extension_runtime;

pub use extension_cache::{
    ExtensionCacheKey, ExtensionCacheLimits, ExtensionCacheSelector, ExtensionCacheStore,
};
pub use extension_runtime::{
    ExtensionExecutionContext, ExtensionExecutor, ExtensionRegistry, ExtensionRuntime,
    ExtensionRuntimeRegistryError,
};
