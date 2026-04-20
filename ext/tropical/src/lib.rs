//! # tenferro-ext-tropical
//!
//! External (out-of-tree) tropical semiring extensions for `tenferro`, built
//! only against the public `tenferro` facade.
//!
//! This crate ships:
//!
//! - Scalar newtypes [`newtype::MaxPlus`], [`newtype::MinPlus`],
//!   [`newtype::MaxMul`] for eager `TypedTensor<T>` T-generic kernels
//!   (eager-path integration deferred — see `newtype` module docs).
//! - Traced composition wrappers (concrete-shape, Stage 4a) in the
//!   [`traced`] module: [`traced::tropical_dot_general`],
//!   [`traced::min_plus_dot_general`], [`traced::tropical_reduce_sum`].
//!
//! # Design
//!
//! Stage 4a lowers tropical ops to `BroadcastInDim + Add + ReduceMax`
//! (or `ReduceMin`) compositions over the core op vocabulary. Automatic
//! differentiation therefore works for free via the existing core AD
//! rules — no new AD math is introduced here.
//!
//! Stage 7 will add `FusedTropicalDotGeneral` as the canonical
//! `ExtensionOp`, registered via the Stage 6 extension mechanism. Until
//! then, this crate is a **contract test** for the public facade:
//! if tropical composition can live outside the workspace, the public
//! surface is sufficient for realistic composition-based extensions.
//!
//! See `docs/design/design_v3/30-algebra-and-tropical.md` and
//! `docs/design/design_v3/40-extension-boundary.md` Recipe A.
//!
//! # Examples
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

pub mod newtype;
pub mod traced;

pub use newtype::{MaxMul, MaxPlus, MinPlus};
pub use traced::{min_plus_dot_general, tropical_dot_general, tropical_reduce_sum};
