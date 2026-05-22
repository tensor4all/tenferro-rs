#![allow(clippy::multiple_bound_locations)]

//! `tenferro`: traced tensor computation with StableHLO-style IR.
//!
//! This crate provides a tracing-based tensor computation framework where
//! operations are recorded into a StableHLO-compatible intermediate
//! representation, then compiled and executed on a backend (e.g., CPU).
//!
//! # Examples
//!
//! ```
//! use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};
//!
//! let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
//! let y = &x + &x;
//! let mut compiler = GraphCompiler::new();
//! let program = compiler.compile(&y).unwrap();
//! let out = GraphExecutor::new(CpuBackend::default()).run(&program).unwrap();
//! assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
//! ```

pub use tenferro_tensor::{DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig};

mod checkpoint;
pub mod compiler;
mod eager;
mod eager_backend;
mod eager_einsum;
mod eager_emitter;
pub mod eager_exec;
pub(crate) mod eager_ops;
pub(crate) mod eager_ops_elementwise;
pub(crate) mod eager_ops_linalg;
pub mod eager_tensor;
mod einsum;
mod einsum_subscripts;
pub mod error;
pub mod exec;
pub mod extension;
pub mod graph;
mod linalg_api;
mod metadata;
mod scalar_semantics;
pub mod segment;
pub mod shape_infer;
mod shape_packing;
pub mod sym_dim;
pub mod tensor;
pub mod traced;
pub mod traced_tensor;
pub mod typed_tensor;

pub use eager::{EagerRuntime, EagerTensor};
pub use einsum_subscripts::parse_einsum_subscripts;
pub use error::ContextId;
pub use graph::{
    CpuGraphExecutorCacheStats, GraphCompiler, GraphCompilerCacheStats, GraphExecutor,
    GraphExecutorCacheStats, GraphProgram, GraphProgramInput,
};
pub use sym_dim::SymDim;
pub use tenferro_ops::std_tensor_op::EinsumSubscripts;
pub use tenferro_tensor::cpu::CpuBackend;
pub use tenferro_tensor::{
    CacheStats, DType, Tensor, TensorBackend, TensorRead, TensorScalar, TensorView, TypedTensor,
    TypedTensorView,
};
pub use traced::TracedTensor;

#[cfg(feature = "cubecl")]
/// CUDA GPU backend facade.
///
/// This module exposes the public CUDA names for tenferro's GPU backend.
/// See `tenferro/examples/cuda_quickstart.rs` for a checked end-to-end example.
pub mod cuda {
    pub use tenferro_tensor::cubecl::{
        download_tensor, gpu_available, upload_tensor, CubeclBackend as CudaBackend,
    };
}
