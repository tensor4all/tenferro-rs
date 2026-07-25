//! Linear algebra extension operations for tenferro.
//!
//! This crate owns the graph-facing linalg op payloads and runtime
//! registration. Tensor-facing operations are exposed through extension traits.
//! CPU backend kernels live in this crate behind the linalg backend trait.
//! A CPU backend paired by `tenferro_gpu::AppleContext` additionally supports
//! guarded rank-2 Cholesky on matching Apple managed `F32`, `F64`, `C32`, and
//! `C64` tensors. This is an explicit CPU selection and is not a general
//! managed-memory fallback for other linalg operations.
//!
//! # Examples
//!
//! ```
//! use tenferro_linalg::TracedTensorLinalgExt;
//! use tenferro_cpu::CpuBackend;
//! use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
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
//! let backend = CpuBackend::new();
//! let engine_id = tenferro_cpu::runtime_engine_id().unwrap();
//! let mut builder = Runtime::builder();
//! builder
//!     .register_engine(tenferro_cpu::runtime_engine_registration(&backend).unwrap())
//!     .unwrap();
//! builder
//!     .install_extension_module(tenferro_linalg::extension_module::<CpuBackend>(engine_id).unwrap())
//!     .unwrap();
//! let runtime = builder.build().unwrap();
//! let out = runtime.run_compiled(&program, &[]).unwrap().pop().unwrap();
//! assert_eq!(out.shape(), &[2, 2]);
//! ```

#[cfg(feature = "autodiff")]
mod ad;
pub mod backend;
mod cpu;
#[cfg(feature = "autodiff")]
mod eager_backend;
#[cfg(feature = "autodiff")]
mod eager_composites;
#[cfg(feature = "autodiff")]
mod eager_ext;
pub mod error;
mod extension;
#[cfg(feature = "cuda")]
mod gpu;
mod tensor_ext;
mod traced;

#[cfg(feature = "autodiff")]
pub use ad::semantic_ad_rules;
#[cfg(feature = "autodiff")]
pub use ad::support::{
    all_linalg_ad_support, linalg_ad_support, LinalgAdModeSupport, LinalgAdOpKind,
    LinalgAdOutputSupport, LinalgAdRoute, LinalgAdRuleSupport, LinalgAdSupport,
};
pub use backend::LinalgBackend;
#[cfg(feature = "autodiff")]
pub use eager_ext::EagerTensorLinalgExt;
pub use error::{Error, Result};
pub use extension::{
    extension_module, EighGauge, EighOptions, QrGauge, QrOptions, SvdGauge, SvdOptions,
    DEFAULT_DECOMPOSITION_DERIVATIVE_EPS, LINALG_EXTENSION_FAMILY_ID,
};
pub use tensor_ext::{
    LinalgScalar, TensorLinalgExt, TensorReadLinalgExt, TypedEig, TypedFullPivLu, TypedLu,
    TypedSvd, TypedTensorLinalgExt,
};
pub use traced::TracedTensorLinalgExt;
