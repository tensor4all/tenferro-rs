#![allow(clippy::multiple_bound_locations)]

//! Compatibility facade over the direct tenferro crates.
//!
//! New code should prefer importing direct crates such as `tenferro_runtime`,
//! `tenferro_tensor`, `tenferro_gpu`, and the operation extension crates.
//!
//! # Examples
//!
//! ```rust
//! use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};
//!
//! let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
//! let y = &x + &x;
//! let mut compiler = GraphCompiler::new();
//! let program = compiler.compile(&y).unwrap();
//! let out = GraphExecutor::new(CpuBackend::default()).run(&program).unwrap();
//! assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
//! ```

#[cfg(feature = "autodiff")]
pub use tenferro_ad::{
    EagerBackend, EagerRuntime, EagerRuntimeCacheStats, EagerTensor, TracedTensorAdExt,
};
pub use tenferro_runtime::*;

#[cfg(feature = "autodiff")]
pub mod ad {
    pub use tenferro_ad::*;
}

#[cfg(feature = "cuda")]
/// CUDA GPU backend facade.
///
/// This module exposes the public CUDA names for tenferro's GPU backend.
/// See `tenferro/examples/cuda_quickstart.rs` for a checked end-to-end example.
pub mod cuda {
    pub use tenferro_gpu::cubecl::{
        download_tensor, gpu_available, upload_tensor, CubeclBackend as CudaBackend,
    };
}
