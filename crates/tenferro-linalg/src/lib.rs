//! Linear algebra extension operations for tenferro.
//!
//! This crate owns the graph-facing linalg op payloads and runtime
//! registration. Tensor-facing operations are exposed through extension traits.
//! CPU backend kernels live in this crate behind the linalg backend trait.
//!
//! Every operation has the same name on every layer and comes in at most three forms
//! (#2552):
//!
//! | layer | forms |
//! |---|---|
//! | concrete ([`TensorLinalgExt`]) | `x.op(exec)`, `x.op_with(&options, exec)`, `x.op_into(.., &options, exec, destination)` |
//! | traced ([`TracedTensorLinalgExt`]) | `t.op()`, `t.op_with(&options)` |
//! | eager (`EagerSessionLinalgExt`, feature `autodiff`) | `session.op(&x)`, `session.op_with(&x, &options)` |
//!
//! The short form runs the operation with `XxxOptions::default()`. Every operation has an
//! options type (`SvdOptions`, `QrOptions`, `TriangularSolveOptions`, ...): it is
//! `#[non_exhaustive]`, built with `Default` and `with_*` setters, and two-way choices are
//! enums (`SvdShape`, `Side`, `Uplo`, `Transpose`, `Diagonal`, `KeepDims`), never `bool`.
//! Axes are `isize`. Multi-output operations return named structs generic over the tensor
//! type ([`Svd`], [`Qr`], [`Lu`], [`FullPivLu`], [`Eig`], [`Eigh`], [`Slogdet`],
//! [`RankRevealingQrResult`]).
//!
//! On the concrete layer each operation is implemented once, on the borrowed
//! [`TensorRead`](tenferro_tensor::TensorRead) view. [`Tensor`](tenferro_tensor::Tensor),
//! `TensorRead`, [`TypedTensor`](tenferro_tensor::TypedTensor) and
//! [`TypedTensorView`](tenferro_tensor::TypedTensorView) reach it through [`LinalgInput`], a
//! metadata-only conversion whose associated output types keep typed input typed. The engine
//! is the last argument before the destination; it owns its execution resources, so no
//! `Exec` or buffer pool appears in an operation signature. Within execution the view may be
//! adapted to the layout the provider consumes (for example a negative-stride operand is
//! gathered); the placement and dtype never change implicitly.
//!
//! # Examples
//!
//! ```
//! use tenferro_linalg::TracedTensorLinalgExt;
//! use tenferro_cpu::CpuEngine;
//! use tenferro_runtime::{TraceContext, ExecutionEngine, GraphCompiler, Runtime, TracedTensor};
//!
//! let trace = TraceContext::new();
//! let a = trace.from_vec_col_major(
//!     vec![2, 2],
//!     vec![4.0_f64, 2.0, 2.0, 3.0],
//! )
//! .unwrap();
//! let l = a.cholesky().unwrap();
//!
//! let mut compiler = GraphCompiler::new();
//! let program = compiler.compile(&l).unwrap();
//! let mut backend = tenferro_cpu::CpuEngine::serial();
//! let engine_id = tenferro_cpu::runtime_engine_id().unwrap();
//! let mut builder = Runtime::builder();
//! builder
//!     .register_engine(tenferro_cpu::runtime_engine_registration(&backend).unwrap())
//!     .unwrap();
//! builder
//!     .install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(engine_id).unwrap())
//!     .unwrap();
//! let runtime = builder.build().unwrap();
//! let prepared = runtime.prepare(&program, &[], &[backend.target()]).unwrap();
//! let out = {
//!     let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut backend];
//!     runtime.run(&mut engines, &prepared, &[]).unwrap().pop().unwrap()
//! }
//! .into_tensor(&mut backend)
//! .unwrap();
//! assert_eq!(out.shape(), &[2, 2]);
//! ```
//!
//! # Cargo features
//!
//! | Feature | Enables |
//! |---|---|
//! | `native` (default) and the other CPU provider features | Forwarded to `tenferro-cpu`; see its documentation. |
//! | `autodiff` | The eager surface (`EagerSessionLinalgExt`) and AD rules. Adds the `tenferro-ad` dependency. |
//! | `cuda` | CUDA execution through `tenferro-gpu`. |
//!
//! `autodiff` is not a heavy dependency switch: `tenferro-ad` is the crate that
//! owns `EagerSession`/`EagerTensor`, so an eager linalg surface without it
//! would save nothing and would only produce eager tensors whose linalg ops
//! fail on backward (#1972). For inference without AD, call the same
//! operations on concrete tensors inside a backend session through
//! [`TensorLinalgExt`], which needs no `autodiff`:
//!
//! ```
//! use tenferro_cpu::CpuEngine;
//! use tenferro_linalg::{Side, TensorLinalgExt, TriangularSolveOptions, Uplo};
//! use tenferro_tensor::TypedTensor;
//!
//! let mut backend = tenferro_cpu::CpuEngine::serial();
//! // Lower-triangular [[2, 0], [1, 1]] (column-major) and right-hand side.
//! let l = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0, 1.0, 0.0, 1.0])?;
//! let b = TypedTensor::<f64>::from_vec_col_major(vec![2, 1], vec![4.0, 5.0])?;
//! let options = TriangularSolveOptions::default()
//!     .with_side(Side::Left)
//!     .with_uplo(Uplo::Lower);
//! let x = l.triangular_solve_with(&b, &options, &mut backend)?;
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
mod eig_normalize;
pub mod error;
mod extension;
#[cfg(feature = "cuda")]
mod gpu;
mod householder;
mod input;
mod options;
pub mod prelude;
mod rank_revealing_qr;
mod results;
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
pub use eager_ext::EagerSessionLinalgExt;
// The CPU linalg route is compiled per backend; a build with neither is a
// configuration error, named as such rather than as a missing internal type.
#[cfg(not(any(feature = "native", feature = "blas")))]
compile_error!(
    "tenferro-linalg needs exactly one CPU backend: enable the `native` (default) \
     or `blas` feature"
);

pub use eig_normalize::normalize_eig_vectors;
pub use error::Error;
pub use extension::{
    extension_module, EighDriver, EighGauge, EighOptions, QrGauge, QrOptions, SvdDriver, SvdGauge,
    SvdOptions, DEFAULT_DECOMPOSITION_DERIVATIVE_EPS, LINALG_EXTENSION_FAMILY_ID,
};
pub use householder::HouseholderQr;
pub use input::LinalgInput;
pub use options::{
    CholeskyOptions, DetOptions, Diagonal, EigOptions, EigvalsOptions, EigvalshOptions,
    FullPivLuOptions, FullPivLuSolveOptions, HouseholderQrOptions, InvOptions, KeepDims,
    LstsqOptions, LuOptions, NormOptions, PinvOptions, Side, SlogdetOptions, SolveOptions,
    SvdShape, SvdvalsOptions, Transpose, TriangularSolveOptions, Uplo,
};
pub use rank_revealing_qr::{RankRevealingQrOptions, RankRevealingQrResult};
pub use results::{Eig, Eigh, FullPivLu, Lu, Qr, Slogdet, Svd};
pub use tensor_ext::{LinalgScalar, TensorLinalgExt};
pub use traced::TracedTensorLinalgExt;
