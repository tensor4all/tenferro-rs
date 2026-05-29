#![deny(missing_docs)]
#![forbid(unsafe_code)]

//! Tropical semiring helpers for tenferro.
//!
//! This standalone crate restores the public skeleton for tropical extension
//! work against tenferro's current split crates. It currently provides scalar
//! newtypes, traced composition wrappers over core tenferro graph ops, and a
//! generic CPU value-plus-argmax GEMM fallback. Fused extension runtime
//! registration and tropical AD rules are intentionally deferred to later
//! implementation tasks.
//!
//! # Examples
//!
//! Scalar newtypes expose the intended semiring arithmetic:
//!
//! ```
//! use tenferro_ext_tropical::MaxPlus;
//!
//! assert_eq!(MaxPlus(2.0_f64) + MaxPlus(5.0_f64), MaxPlus(5.0_f64));
//! assert_eq!(MaxPlus(2.0_f64) * MaxPlus(5.0_f64), MaxPlus(7.0_f64));
//! ```
//!
//! The traced composition path lowers max-plus matrix multiplication to core
//! graph operations:
//!
//! ```
//! use tenferro_cpu::CpuBackend;
//! use tenferro_ext_tropical::tropical_dot_general;
//! use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
//!
//! let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
//! let b = TracedTensor::from_vec_col_major(vec![2, 2], vec![10.0_f64, 20.0, 30.0, 40.0]);
//! let out = tropical_dot_general(&a, &b);
//!
//! let mut compiler = GraphCompiler::new();
//! let program = compiler.compile(&out).unwrap();
//! let mut executor = GraphExecutor::new(CpuBackend::new());
//! let value = executor.run(&program).unwrap();
//! assert_eq!(value.as_slice::<f64>().unwrap(), &[23.0, 24.0, 43.0, 44.0]);
//! ```

pub mod cpu;
pub mod newtype;
pub mod traced;

pub use newtype::{MaxMul, MaxPlus, MinPlus};
pub use traced::{min_plus_dot_general, tropical_dot_general, tropical_reduce_sum};

/// Tropical semiring flavor used by traced and future fused tropical ops.
///
/// # Examples
///
/// ```
/// use tenferro_ext_tropical::TropicalKind;
///
/// assert_eq!(TropicalKind::MaxPlus, TropicalKind::MaxPlus);
/// assert_ne!(TropicalKind::MaxPlus, TropicalKind::MinPlus);
/// ```
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TropicalKind {
    /// Max-plus semiring: `out[i, j] = max_k(a[i, k] + b[k, j])`.
    MaxPlus,
    /// Min-plus semiring: `out[i, j] = min_k(a[i, k] + b[k, j])`.
    MinPlus,
}
