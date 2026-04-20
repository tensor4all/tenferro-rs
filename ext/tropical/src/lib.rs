//! # tenferro-ext-tropical
//!
//! External (out-of-tree) tropical semiring extensions for `tenferro`.
//!
//! This crate ships:
//!
//! - Scalar newtypes [`newtype::MaxPlus`], [`newtype::MinPlus`],
//!   [`newtype::MaxMul`] for eager `TypedTensor<T>` T-generic kernels
//!   (eager-path integration deferred — see `newtype` module docs).
//! - Traced composition wrappers (Stages 4a / 4b) in the [`traced`]
//!   module: [`traced::tropical_dot_general`],
//!   [`traced::min_plus_dot_general`], [`traced::tropical_reduce_sum`].
//! - The Stage 7 fused [`fused::FusedTropicalDotGeneralOp`] `ExtensionOp`
//!   plus its user-facing wrappers
//!   [`traced::tropical_dot_general_fused`] and
//!   [`traced::min_plus_dot_general_fused`].
//!
//! # Design
//!
//! Stage 4a lowers tropical ops to `BroadcastInDim + Add + ReduceMax`
//! (or `ReduceMin`) compositions over the core op vocabulary. Automatic
//! differentiation therefore works for free via the existing core AD
//! rules — no new AD math is introduced there.
//!
//! Stage 7 packages a fused primal (conceptually a tropical GEMM kernel)
//! as a single `StdTensorOp::Extension(Arc<dyn ExtensionOp>)` node,
//! routed through the Stage 6 extension registry. The AD rule remains
//! expressed over core ops (indicator-based VJP/JVP), preserving the
//! Stage 5 AD closure contract.
//!
//! See:
//!
//! - `docs/design/design_v3/30-algebra-and-tropical.md` "Fused Tropical Support"
//! - `docs/design/design_v3/40-extension-boundary.md` Recipe B
//! - `docs/spec/extension-op.md` for the normative `ExtensionOp` contract
//!
//! # Examples
//!
//! Composition path (Stage 4):
//!
//! ```
//! use tenferro::{CpuBackend, Engine, TracedTensor};
//! use tenferro_ext_tropical::traced::tropical_dot_general;
//!
//! let a = TracedTensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
//! let b = TracedTensor::from_vec(vec![2, 2], vec![10.0_f64, 20.0, 30.0, 40.0]);
//!
//! let mut c = tropical_dot_general(&a, &b);
//! let mut engine = Engine::new(CpuBackend::new());
//! let out = c.eval(&mut engine).unwrap();
//! assert_eq!(out.shape(), &[2, 2]);
//! ```
//!
//! Fused `ExtensionOp` path (Stage 7):
//!
//! ```
//! use tenferro::{CpuBackend, Engine, TracedTensor};
//! use tenferro_ext_tropical::traced::tropical_dot_general_fused;
//!
//! let a = TracedTensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
//! let b = TracedTensor::from_vec(vec![2, 2], vec![10.0_f64, 20.0, 30.0, 40.0]);
//!
//! let mut c = tropical_dot_general_fused(&a, &b);
//! let mut engine = Engine::new(CpuBackend::new());
//! let out = c.eval(&mut engine).unwrap();
//! assert_eq!(out.shape(), &[2, 2]);
//! ```

pub mod fused;
pub mod newtype;
pub mod traced;

pub use fused::{
    register_fused_tropical, FusedTropicalDotGeneralFactory, FusedTropicalDotGeneralOp,
    TropicalKind, FUSED_TROPICAL_DOT_GENERAL_FAMILY_ID,
};
pub use newtype::{MaxMul, MaxPlus, MinPlus};
pub use traced::{
    min_plus_dot_general, min_plus_dot_general_fused, tropical_dot_general,
    tropical_dot_general_fused, tropical_reduce_sum,
};
