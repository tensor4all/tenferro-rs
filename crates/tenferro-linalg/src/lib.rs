//! Linear algebra extension operations for tenferro.
//!
//! This crate owns the graph-facing linalg op payloads and runtime
//! registration. Traced helpers live under `tenferro_linalg::traced_tensor`.
//! CPU backend kernels live in this crate behind the linalg backend trait.
//!
//! # Examples
//!
//! ```
//! use tenferro_linalg::traced_tensor::cholesky;
//! use tenferro_cpu::CpuBackend;
//! use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
//!
//! let a = TracedTensor::from_vec_col_major(
//!     vec![2, 2],
//!     vec![4.0_f64, 2.0, 2.0, 3.0],
//! )
//! .unwrap();
//! let l = cholesky(&a).unwrap();
//!
//! let mut compiler = GraphCompiler::new();
//! let program = compiler.compile(&l).unwrap();
//! let mut executor = GraphExecutor::new(CpuBackend::new());
//! executor.register_extension(tenferro_linalg::register_runtime).unwrap();
//! let out = executor.run(&program).unwrap();
//! assert_eq!(out.shape(), &[2, 2]);
//! ```

#[cfg(feature = "autodiff")]
mod ad;
pub mod backend;
mod cpu;
#[cfg(feature = "autodiff")]
mod eager_backend;
#[cfg(feature = "autodiff")]
pub mod eager_tensor;
mod extension;
#[cfg(feature = "cuda")]
mod gpu;
mod traced;
pub mod traced_tensor;

#[cfg(feature = "autodiff")]
pub use ad::ad_rules;
#[cfg(feature = "autodiff")]
pub use ad::support::{
    all_linalg_ad_support, linalg_ad_support, LinalgAdOpKind, LinalgAdOutputSupport,
    LinalgAdRuleSupport, LinalgAdSupport,
};
pub use backend::LinalgBackend;
pub use extension::{register_runtime, LINALG_EXTENSION_FAMILY_ID};
