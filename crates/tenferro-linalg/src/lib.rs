//! Linear algebra extension operations for tenferro.
//!
//! This crate owns the graph-facing linalg op payloads and runtime
//! registration. Tensor-facing operations are exposed through extension traits.
//! CPU backend kernels live in this crate behind the linalg backend trait.
//!
//! # Examples
//!
//! ```
//! use tenferro_linalg::TracedTensorLinalgExt;
//! use tenferro_cpu::CpuBackend;
//! use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
//!
//! let a = TracedTensor::from_vec_col_major(
//!     vec![2, 2],
//!     vec![4.0_f64, 2.0, 2.0, 3.0],
//! )
//! .unwrap();
//! let l = a.cholesky().unwrap();
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
mod eager_ext;
mod extension;
#[cfg(feature = "cuda")]
mod gpu;
mod prepared_factorization;
mod prepared_svd;
mod traced;

#[cfg(feature = "autodiff")]
pub use ad::ad_rules;
#[cfg(feature = "autodiff")]
pub use ad::support::{
    all_linalg_ad_support, linalg_ad_support, LinalgAdModeSupport, LinalgAdOpKind,
    LinalgAdOutputSupport, LinalgAdRoute, LinalgAdRuleSupport, LinalgAdSupport,
};
pub use backend::LinalgBackend;
#[cfg(feature = "autodiff")]
pub use eager_ext::EagerTensorLinalgExt;
pub use extension::{
    register_runtime, EighGauge, EighOptions, QrGauge, QrOptions, SvdGauge, SvdOptions,
    DEFAULT_DECOMPOSITION_DERIVATIVE_EPS, LINALG_EXTENSION_FAMILY_ID,
};
pub use prepared_factorization::{PreparedFactorizationBackendExt, PreparedFactorizationSession};
pub use prepared_svd::{
    PreparedSvd, PreparedSvdBackendExt, SvdOutputSpec, SvdOutputSpecs, SvdOutputWrites,
    SvdWorkspace,
};
pub use traced::TracedTensorLinalgExt;
