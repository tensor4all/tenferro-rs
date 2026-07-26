#![deny(missing_docs)]
#![forbid(unsafe_code)]

//! Tropical semiring helpers for tenferro.
//!
//! This standalone crate restores the public skeleton for tropical extension
//! work against tenferro's current split crates. It provides scalar newtypes,
//! traced composition wrappers over core tenferro graph ops, fused traced
//! tropical einsum extension helpers, and a generic CPU value-plus-argmax GEMM
//! fallback.
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
//! use tenferro_ext_tropical::traced::tropical_dot_general;
//! use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
//!
//! let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
//! let b = TracedTensor::from_vec_col_major(vec![2, 2], vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap();
//! let out = tropical_dot_general(&a, &b).unwrap();
//!
//! let mut compiler = GraphCompiler::new();
//! let program = compiler.compile(&out).unwrap();
//! let backend = CpuBackend::new();
//! let mut builder = Runtime::builder();
//! builder
//!     .register_engine(tenferro_cpu::runtime_engine_registration(&backend).unwrap())
//!     .unwrap();
//! let runtime = builder.build().unwrap();
//! let value = runtime.run_compiled(&program, &[]).unwrap().pop().unwrap();
//! assert_eq!(value.as_slice::<f64>().unwrap(), &[23.0, 24.0, 43.0, 44.0]);
//! ```

pub mod cpu;
pub mod einsum;
mod error;
mod extension;
pub mod newtype;
pub mod traced;

pub use extension::extension_modules;
#[cfg(feature = "autodiff")]
pub use extension::tropical_semantic_ad_rules;
pub use newtype::{MaxMul, MaxPlus, MinPlus};

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
