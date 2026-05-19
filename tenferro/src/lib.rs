#![allow(clippy::multiple_bound_locations)]

//! `tenferro`: traced tensor computation with StableHLO-style IR.
//!
//! This crate provides a tracing-based tensor computation framework where
//! operations are recorded into a StableHLO-compatible intermediate
//! representation, then compiled and executed on a backend (e.g., CPU).
//!
//! # Examples
//!
//! ```rust,ignore
//! use tenferro::{CpuBackend, Engine, TracedTensor};
//!
//! let mut engine = Engine::new(CpuBackend::default());
//! // ... build and execute traced computations
//! ```

pub use tenferro_tensor::{DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig};

mod checkpoint;
pub mod compiler;
mod eager;
mod eager_einsum;
mod eager_emitter;
pub mod eager_exec;
pub(crate) mod eager_ops;
pub(crate) mod eager_ops_elementwise;
pub(crate) mod eager_ops_linalg;
pub mod eager_tensor;
mod einsum;
pub mod engine;
pub mod error;
pub mod exec;
pub mod extension;
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

pub use eager::{EagerContext, EagerTensor};
pub use engine::Engine;
pub use error::ContextId;
pub use sym_dim::SymDim;
pub use tenferro_tensor::cpu::CpuBackend;
pub use tenferro_tensor::{DType, Tensor, TensorBackend, TensorScalar, TypedTensor};
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
