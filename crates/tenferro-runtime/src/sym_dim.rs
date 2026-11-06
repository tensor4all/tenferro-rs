//! Symbolic dimensions of traced values.
//!
//! A [`SymDim`] tensor axis names a traced handle by an id that only the handle
//! hands out, through [`TracedTensor::sym_size`](crate::TracedTensor::sym_size),
//! [`TracedTensor::axis_sym_dim`](crate::TracedTensor::axis_sym_dim) and
//! [`TracedTensor::sym_shape`](crate::TracedTensor::sym_shape). A program's
//! `DimExpr::InputDim` names a program input by position instead, and that
//! position is never turned into a `SymDim`: program input #1 and the handle whose
//! id is 1 are different things, so no public path produces one from the other.
//!
//! A conversion accepts the axes of the handles it was given and rejects any
//! other handle's axis:
//!
//! ```
//! use tenferro_runtime::{DType, Error, TraceContext};
//!
//! let trace = TraceContext::new();
//! let a = trace.input_symbolic_shape(DType::F64, 1)?; // program input #0
//! let b = trace.input_symbolic_shape(DType::F64, 1)?; // program input #1
//!
//! // `a`'s own extent converts against `a`.
//! assert!(a.reshape_sym(&[a.sym_size(0)?]).is_ok());
//! // `b`'s extent does not: `reshape_sym` only knows `a`.
//! let err = a.reshape_sym(&[b.sym_size(0)?]).unwrap_err();
//! assert!(matches!(err, Error::SymbolicShapeConversion { .. }));
//! # Ok::<(), tenferro_runtime::Error>(())
//! ```
//!
//! Program coordinates cannot be passed where a handle axis is expected, because
//! there is no way to turn one into a `SymDim` (#2443):
//!
//! ```compile_fail,E0425
//! use tenferro_ops::dim_expr::DimExpr;
//! use tenferro_runtime::{DType, TraceContext};
//!
//! let trace = TraceContext::new();
//! let a = trace.input_symbolic_shape(DType::F64, 1)?; // program input #0
//! let _b = trace.input_symbolic_shape(DType::F64, 1)?; // program input #1
//! // "Program input #1, axis 0" would carry the same id as a handle's axis.
//! let b_extent = tenferro_runtime::dim_expr_to_sym_dim_public(&DimExpr::InputDim {
//!     input_idx: 1,
//!     axis: 0,
//! })
//! .unwrap();
//! let _ = a.reshape_sym(&[b_extent]);
//! # Ok::<(), tenferro_runtime::Error>(())
//! ```

pub use tenferro_ops::sym_dim::*;
