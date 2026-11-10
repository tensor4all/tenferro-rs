//! Linear algebra extension operations for tenferro.
//!
//! This crate owns the graph-facing linalg op payloads and runtime
//! registration. Tensor-facing operations are exposed through extension traits.
//! CPU backend kernels live in this crate behind the linalg backend trait.
//!
//! Traced graph construction uses [`TraceContextLinalgExt`] on a
//! [`TraceContext`](tenferro_runtime::TraceContext), or
//! [`TracedTensorLinalgExt`] for method chaining on a
//! [`TracedTensor`](tenferro_runtime::TracedTensor); both record the same
//! operations.
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
//! | `autodiff` | The eager surface (`EagerSessionLinalgExt`, `EagerTensorLinalgExt`) and AD rules. Adds the `tenferro-ad` dependency. |
//! | `cuda` | CUDA execution through `tenferro-gpu`. |
//!
//! `autodiff` is not a heavy dependency switch: `tenferro-ad` is the crate that
//! owns `EagerSession`/`EagerTensor`, so an eager linalg surface without it
//! would save nothing and would only produce eager tensors whose linalg ops
//! fail on backward (#1972). For inference without AD, call the same
//! operations on concrete tensors inside a backend session through
//! [`TensorLinalgExt`] / [`TypedTensorLinalgExt`], which need no `autodiff`:
//!
//! ```
//! use tenferro_cpu::CpuEngine;
//! use tenferro_linalg::TypedTensorLinalgExt;
//! use tenferro_tensor::TypedTensor;
//!
//! let mut backend = tenferro_cpu::CpuEngine::serial();
//! // Lower-triangular [[2, 0], [1, 1]] (column-major) and right-hand side.
//! let l = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0, 1.0, 0.0, 1.0])?;
//! let b = TypedTensor::<f64>::from_vec_col_major(vec![2, 1], vec![4.0, 5.0])?;
//! let x = {
//!     // left_side, lower, transpose_a, unit_diagonal
//!     l.triangular_solve(&b, true, true, false, false, &mut backend)
//! }?;
//! assert_eq!(x.host_data()?, &[2.0, 3.0]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Borrowed-`Exec` factorization entries
//!
//! A caller that runs its own lanes calls these on each lane's bundle and pool instead of going
//! through an engine. The engine's own entries run the same typed bodies and only erase the
//! results, so the same operand gives the same outputs, value bits and placement included, on
//! both paths.
//!
//! The entries exist on the `native` provider only; the BLAS provider has no per-context entry
//! (see the "Lanes and the vendor provider" section of `docs/guides/cpu-execution.md`).
//!
//! ## Contract
//!
//! - **Execution.** A call runs on the `exec` it is handed: the bundle's token, its width and its
//!   batch policy. It enters no other pool and does not re-enter an engine.
//! - **Storage.** Every output payload, the provider workspace and the transient spectrum of a
//!   complex input are leased from `buffers`; tenferro allocates no payload or workspace outside
//!   it. The guarantee is about those buffers, not about every byte a call touches: the dims and
//!   strides of a view and of an output are shape metadata, held inline up to `INLINE_RANK` (8)
//!   axes and heap-allocated past that, so an operand of rank 9 or more makes a few small
//!   metadata allocations outside the pool per call. `tests/cpu_linalg_allocation.rs` carries a
//!   rank-9 row beside its rank-2 and rank-8 rows, so this wording and the measured counts move
//!   together. (faer's own GEMM still allocates packing scratch per call off x86_64; that is
//!   inside the provider, and the same test records it.) An output returns its storage to the
//!   pool when it drops, so a lane that keeps one pool reaches a steady state. Results may outlive
//!   the pool; their storage is then freed instead of recycled.
//! - **View rule.** The operand is a matrix `[m, n]` or a batch `[m, n, b..]`; the eigensolvers
//!   require `m == n`. Its dims, strides and offset are read in place as given. A negative stride
//!   is refused with an `Unsupported` error before anything is computed. This is **not** the
//!   rule of the `*_read` engine entries, which read rank 2 in place, gather negative strides and
//!   materialize everything else; that stays on the owned path.
//! - **Empty input.** An operand with a zero dimension returns correctly shaped outputs without
//!   reading it; the square factors of a full SVD are then identities. Its strides are still
//!   validated first, so an empty operand with a negative stride is refused with the error a
//!   non-empty one gets instead of taking this shortcut.
//! - **Placement.** Every output carries the operand's placement.
//! - **Errors.** Rank and shape failures use the shared validation kinds; a provider failure is
//!   [`map_error`](crate::error::map_error) of the `tlinalg` error.
//!
//! ## Example
//!
//! One pool per lane, each lane on its own share of the engine's bundle:
//!
//! ```rust
//! # #[cfg(feature = "native")]
//! # fn main() -> Result<(), tenferro_tensor::Error> {
//! use tenferro_cpu::{linalg_interop::BufferPool, CpuEngine};
//! use tenferro_tensor::TypedTensor;
//!
//! let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![4.0, 0.0, 0.0, 9.0])?;
//! let mut engine = CpuEngine::serial();
//! let mut lane_pool = BufferPool::new();
//! let values = engine.with_exec(|exec| {
//!     tenferro_linalg::eigh_values_with_exec(exec, &mut lane_pool, &a.as_view())
//! })??;
//! assert_eq!(values.host_data()?, &[4.0, 9.0]);
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "native"))]
//! # fn main() {}
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
mod flags;
#[cfg(feature = "cuda")]
mod gpu;
mod householder;
pub mod prelude;
mod rank_revealing_qr;
mod results;
mod tensor_ext;
mod traced;
mod validation;
#[cfg(feature = "native")]
mod with_exec;

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
// The CPU linalg route is compiled per backend; a build with neither is a
// configuration error, named as such rather than as a missing internal type.
#[cfg(not(any(feature = "native", feature = "blas")))]
compile_error!(
    "tenferro-linalg needs exactly one CPU backend: enable the `native` (default) \
     or `blas` feature"
);

pub use eig_normalize::normalize_eig_vectors;
pub use error::{Error, Result};
pub use extension::{
    extension_module, EighDriver, EighGauge, EighOptions, QrGauge, QrOptions, SvdDriver, SvdGauge,
    SvdOptions, DEFAULT_DECOMPOSITION_DERIVATIVE_EPS, LINALG_EXTENSION_FAMILY_ID,
};
pub use flags::{Diag, KeepDims, MatOp, Side, Uplo};
pub use householder::HouseholderQr;
pub use rank_revealing_qr::{RankRevealingQrOptions, RankRevealingQrResult};
pub use results::{Eig, Eigh, FullPivLu, Lu, Qr, Slogdet, Svd};
pub use tensor_ext::{
    LinalgScalar, TensorLinalgExt, TensorReadLinalgExt, TypedEig, TypedFullPivLu, TypedLu,
    TypedRankRevealingQrResult, TypedSvd, TypedTensorLinalgExt,
};
pub use traced::{TraceContextLinalgExt, TracedTensorLinalgExt};
#[cfg(feature = "native")]
#[cfg_attr(docsrs, doc(cfg(feature = "native")))]
pub use with_exec::{
    eig_values_with_exec, eigh_values_with_exec, eigh_with_exec, qr_with_exec,
    svd_values_with_exec, svd_with_exec,
};
