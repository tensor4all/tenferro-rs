//! Linear algebra operations for tenferro.
//!
//! This crate owns the graph-facing linalg op payloads, their runtime
//! registration, the CPU linalg kernels and the public linalg grammar
//! (#2552). Every operation is declared once in the operation registry and
//! offered on three layers with the same three spellings:
//!
//! | Layer | Trait | Receiver | Spellings |
//! |---|---|---|---|
//! | concrete | [`LinalgOps`] | anything that implements [`AsView`](tenferro_tensor::AsView): `Tensor`, `TensorRead`, `TypedTensor<T>`, `TypedTensorView<T>` | `a.svd(exec)`, `a.svd_with(&opts, exec)`, `a.solve_into(&b, exec, dest)` |
//! | traced | [`TracedLinalgOps`] | [`TracedTensor`](tenferro_runtime::TracedTensor) | `t.svd()`, `t.svd_with(&opts)` |
//! | eager (`autodiff`) | `EagerLinalgOps` | `EagerSession` | `sess.svd(&x)`, `sess.svd_with(&x, &opts)` |
//!
//! An operation has an options type (`SvdOptions`, `PinvOptions`, ...) only
//! when it has semantic options; `solve`, `lstsq`, `lu`, `eig`, `det`, `inv`
//! and the other operations without settings have neither an options type nor
//! a `_with` form;
//! factorizations return named structs ([`Svd`], [`Qr`], [`Lu`], ...) that are
//! destructured by field. The generated registry page
//! (`docs/spec/generated/api-ops.md`) lists the operations, their layout
//! handling, errors and omissions.
//!
//! An operation without options has no `_with` form and no options type:
//!
//! ```compile_fail,E0599
//! use tenferro_cpu::CpuEngine;
//! use tenferro_linalg::LinalgOps;
//! use tenferro_tensor::Tensor;
//!
//! let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0]).unwrap();
//! let _ = a.solve_with(&a, &mut CpuEngine::serial());
//! ```
//!
//! ```compile_fail,E0432
//! use tenferro_linalg::SolveOptions;
//! ```
//!
//! # Examples
//!
//! Traced: record, compile, run.
//!
//! ```
//! use tenferro_cpu::CpuEngine;
//! use tenferro_linalg::TracedLinalgOps;
//! use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};
//!
//! let trace = TraceContext::new();
//! let a = trace.from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0])?;
//! let l = a.cholesky()?;
//!
//! let program = GraphCompiler::new().compile(&l)?;
//! let mut backend = CpuEngine::serial();
//! let engine_id = tenferro_cpu::runtime_engine_id()?;
//! let mut builder = Runtime::builder();
//! builder.register_engine(tenferro_cpu::runtime_engine_registration(&backend)?)?;
//! builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(engine_id)?)?;
//! let runtime = builder.build()?;
//! let prepared = runtime.prepare(&program, &[], &[backend.target()])?;
//! let out = {
//!     let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut backend];
//!     runtime.run(&mut engines, &prepared, &[])?.pop().ok_or("no output")?
//! }
//! .into_tensor(&mut backend)?;
//! assert_eq!(out.shape(), &[2, 2]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Concrete, on a typed tensor; the result is typed too:
//!
//! ```
//! use tenferro_cpu::CpuEngine;
//! use tenferro_linalg::{LinalgOps, Svd, TriangularSolveOptions};
//! use tenferro_tensor::TypedTensor;
//!
//! let mut engine = CpuEngine::serial();
//! // Lower-triangular [[2, 0], [1, 1]] (column-major) and right-hand side.
//! let l = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0, 1.0, 0.0, 1.0])?;
//! let b = TypedTensor::<f64>::from_vec_col_major(vec![2, 1], vec![4.0, 5.0])?;
//! let x = l.triangular_solve_with(&b, &TriangularSolveOptions::default(), &mut engine)?;
//! assert_eq!(x.host_data()?, &[2.0, 3.0]);
//!
//! let Svd { s, .. } = l.svd(&mut engine)?;
//! assert_eq!(s.shape(), &[2]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Cargo features
//!
//! | Feature | Enables |
//! |---|---|
//! | `autodiff` | The eager layer (`EagerLinalgOps`) and AD rules. Adds the `tenferro-ad` dependency. |
//! | `cuda` | CUDA execution through `tenferro-gpu`. |
//!
//! `autodiff` is not a heavy dependency switch: `tenferro-ad` is the crate that
//! owns `EagerSession`/`EagerTensor`, so an eager linalg surface without it
//! would save nothing and would only produce eager tensors whose linalg ops
//! fail on backward (#1972). For inference without AD, call the same
//! operations on concrete tensors through [`LinalgOps`], which needs no
//! `autodiff`.
//!
//! # Lanes
//!
//! A caller that splits one engine's threads into lanes and runs a different
//! factorization in each uses `CpuEngine::with_lanes`: every lane receives a
//! `&mut dyn Engine` that the ordinary grammar accepts, so the lane call is
//! `view.svd_with(&opts, lane)`. Each lane runs on its share of the engine's
//! thread budget and draws every output, workspace and gather buffer from the
//! caller's pool for that lane, so a lane that keeps its pool reaches zero
//! steady-state allocation for inputs without negative strides (a
//! negative-stride input is gathered once into the lane's pool). The dims and
//! strides of an operand of rank 9 or more are heap metadata, so such an
//! operand makes a few small allocations outside the pool per call.
//!
//! Only svd, svdvals, qr, eigh, eigvalsh and eigvals run in a lane; every other
//! operation returns `Unsupported` there (see the "Lanes" section of
//! `docs/guides/cpu-execution.md`).
//!
//! ```rust
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use tenferro_cpu::{linalg_interop::BufferPool, CpuEngine, Pool};
//! use tenferro_linalg::LinalgOps;
//! use tenferro_tensor::TypedTensor;
//!
//! let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![4.0, 0.0, 0.0, 9.0])?;
//! let b = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![3.0, 0.0, 0.0, 2.0])?;
//! let mut engine = CpuEngine::new(Pool::builder().threads(2).build()?);
//! let mut pools = [BufferPool::new(), BufferPool::new()];
//! let out = engine.with_lanes(&[1, 1], &mut pools, |lane_index, lane| {
//!     let input = if lane_index == 0 { &a } else { &b };
//!     input.eigvalsh(lane).and_then(|w| Ok(w.host_data()?.to_vec()))
//! })?;
//! assert_eq!(out[0].as_ref().ok(), Some(&vec![4.0, 9.0]));
//! assert_eq!(out[1].as_ref().ok(), Some(&vec![2.0, 3.0]));
//! # Ok(())
//! # }
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
mod api;
pub mod backend;
mod cpu;
#[cfg(feature = "autodiff")]
mod eager_composites;
#[cfg(feature = "autodiff")]
mod eager_ext;
mod eig_normalize;
pub mod error;
mod extension;
mod flags;
#[cfg(feature = "cuda")]
mod gpu;
mod householder;
mod lane;
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
#[cfg(feature = "autodiff")]
#[cfg_attr(docsrs, doc(cfg(feature = "autodiff")))]
pub use api::EagerLinalgOps;
pub use api::{
    CholeskyOptions, EigvalshOptions, LinalgOps, LinalgRepr, NormOptions, NormOrd, PinvOptions,
    SvdvalsOptions, TracedLinalgOps, TriangularSolveOptions,
};
pub use backend::LinalgBackend;
pub use eig_normalize::normalize_eig_vectors;
pub use error::{Error, Result};
pub use extension::{
    extension_module, EighDriver, EighGauge, EighOptions, QrGauge, QrOptions, SvdDriver, SvdGauge,
    SvdOptions, DEFAULT_DECOMPOSITION_DERIVATIVE_EPS, LINALG_EXTENSION_FAMILY_ID,
};
pub use flags::{Diag, KeepDims, MatOp, Side, SvdShape, Uplo};
pub use householder::HouseholderQr;
pub use rank_revealing_qr::{RankRevealingQrOptions, RankRevealingQrResult};
pub use results::{Eig, Eigh, FullPivLu, Lu, Qr, Slogdet, Svd};
pub use tensor_ext::LinalgScalar;
pub(crate) use tensor_ext::{OwnedLinalg, ReadLinalg};
