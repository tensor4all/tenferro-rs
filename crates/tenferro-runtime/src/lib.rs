//! Traced graph runtime and extension dispatch infrastructure for tenferro.
//!
//! This crate owns graph construction, lowering to execution IR, graph
//! execution, and backend-parametric extension runtime dispatch. Standard
//! operations live in `tenferro-internal-ops`; tensor storage and backend
//! kernels live in `tenferro-tensor`.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
//! use tenferro_cpu::CpuBackend;
//!
//! let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
//! let y = &x + &x;
//! let mut compiler = GraphCompiler::new();
//! let program = compiler.compile(&y).unwrap();
//! let out = GraphExecutor::new(CpuBackend::default()).run(&program).unwrap();
//! assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
//! ```

#[doc(hidden)]
pub mod ad_support;
mod checkpoint;
pub mod compiler;
pub mod error;
pub mod exec;
pub mod extension;
pub mod extension_cache;
pub mod extension_runtime;
pub mod graph;
mod metadata;
#[doc(hidden)]
pub mod scalar_semantics;
pub mod segment;
pub mod shape_infer;
mod shape_packing;
pub mod sym_dim;
pub mod tensor;
pub mod traced;
pub mod traced_tensor;
pub mod typed_tensor;

pub use error::{ContextId, Error, Result};
pub use extension_cache::{
    ExtensionCacheKey, ExtensionCacheLimits, ExtensionCacheSelector, ExtensionCacheStore,
};
pub use extension_runtime::{
    ExtensionExecutionContext, ExtensionExecutor, ExtensionRegistry, ExtensionRuntime,
    ExtensionRuntimeRegistryError,
};
pub use graph::{
    GraphCompiler, GraphCompilerCacheStats, GraphExecutor, GraphExecutorCacheStats, GraphProgram,
    GraphProgramInput,
};
pub use sym_dim::SymDim;
pub use tenferro_tensor::{
    CacheStats, CompareDir, DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig,
    SliceConfig, Tensor, TensorBackend, TensorRead, TensorScalar, TensorValue, TensorView,
    TypedTensor, TypedTensorView,
};
pub use traced::TracedTensor;
