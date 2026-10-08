//! Linear algebra extension operations for tenferro.
//!
//! This crate owns the graph-facing linalg op payloads and runtime
//! registration. Tensor-facing operations are exposed through extension traits.
//! CPU backend kernels live in this crate behind the linalg backend trait.
//! A CPU backend paired by `tenferro_gpu::apple::AppleContext` additionally supports
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
//!
//! # Cargo features
//!
//! | Feature | Enables |
//! |---|---|
//! | `native` (default) and the other CPU provider features | Forwarded to `tenferro-cpu`; see its documentation. |
//! | `autodiff` | The eager surface (`EagerSessionLinalgExt`, `EagerTensorLinalgExt`) and AD rules. Adds the `tenferro-ad` dependency. |
//! | `cuda` | CUDA execution through `tenferro-gpu`. |
//! | `webgpu` | WebGPU/Metal execution through `tenferro-gpu` (a subset of operations). |
//! | `rocm` | Placeholder; HIP/ROCm is not implemented. |
//!
//! `autodiff` is not a heavy dependency switch: `tenferro-ad` is the crate that
//! owns `EagerSession`/`EagerTensor`, so an eager linalg surface without it
//! would save nothing and would only produce eager tensors whose linalg ops
//! fail on backward (#1972). For inference without AD, call the same
//! operations on concrete tensors inside a backend session through
//! [`TensorLinalgExt`] / [`TypedTensorLinalgExt`], which need no `autodiff`:
//!
//! ```
//! use tenferro_cpu::CpuBackend;
//! use tenferro_linalg::TypedTensorLinalgExt;
//! use tenferro_tensor::{BackendSessionHost, TypedTensor};
//!
//! let mut backend = CpuBackend::new();
//! // Lower-triangular [[2, 0], [1, 1]] (column-major) and right-hand side.
//! let l = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0, 1.0, 0.0, 1.0])?;
//! let b = TypedTensor::<f64>::from_vec_col_major(vec![2, 1], vec![4.0, 5.0])?;
//! let x = backend.with_backend_session(|session| {
//!     // left_side, lower, transpose_a, unit_diagonal
//!     l.triangular_solve(&b, true, true, false, false, session)
//! })??;
//! assert_eq!(x.host_data()?, &[2.0, 3.0]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
// A misaligned pointer handed to a CUDA library is undefined behaviour and
// fails only on some library versions: `cuDoubleComplex` is `double2`, which
// CUDA declares `__align__(16)`, while `num_complex::Complex64` is 8-aligned,
// so casting `&Complex64` to `*const cuDoubleComplex` produced a pointer
// cuBLAS >= 12.9 faults on (issue #1870). Deny the whole cast class at this
// FFI boundary rather than re-auditing it by hand.
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(clippy::cast_ptr_alignment)]

#[cfg(feature = "autodiff")]
mod ad;
pub mod backend;
mod cpu;
#[cfg(feature = "autodiff")]
mod eager_composites;
#[cfg(feature = "autodiff")]
mod eager_ext;
pub mod error;
mod extension;
#[cfg(feature = "cuda")]
mod gpu;
mod householder;
pub mod prelude;
mod rank_revealing_qr;
mod tensor_ext;
mod traced;
mod validation;

#[cfg(feature = "autodiff")]
pub use ad::semantic_ad_rules;
#[cfg(feature = "autodiff")]
pub use ad::support::{
    all_linalg_ad_support, linalg_ad_support, LinalgAdModeSupport, LinalgAdOpKind,
    LinalgAdOutputSupport, LinalgAdRoute, LinalgAdRuleSupport, LinalgAdSupport,
};
pub use backend::LinalgBackend;
#[cfg(feature = "autodiff")]
#[cfg_attr(docsrs, doc(cfg(feature = "autodiff")))]
pub use eager_ext::{EagerSessionLinalgExt, EagerTensorLinalgExt};
pub use error::{Error, Result};
pub use extension::{
    extension_module, EighDriver, EighGauge, EighOptions, QrGauge, QrOptions, SvdDriver, SvdGauge,
    SvdOptions, DEFAULT_DECOMPOSITION_DERIVATIVE_EPS, LINALG_EXTENSION_FAMILY_ID,
};
pub use householder::HouseholderQr;
pub use rank_revealing_qr::{RankRevealingQrOptions, RankRevealingQrResult};
pub use tensor_ext::{
    LinalgScalar, TensorLinalgExt, TensorReadLinalgExt, TypedEig, TypedFullPivLu, TypedLu,
    TypedRankRevealingQrResult, TypedSvd, TypedTensorLinalgExt,
};
pub use traced::TracedTensorLinalgExt;
