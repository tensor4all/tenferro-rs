//! Symbolic dimensions of traced values.
//!
//! A [`SymDim`] tensor axis names a traced handle by an id that the handle hands
//! out, through [`TracedTensor::sym_size`](crate::TracedTensor::sym_size),
//! [`TracedTensor::axis_sym_dim`](crate::TracedTensor::axis_sym_dim) and
//! [`TracedTensor::sym_shape`](crate::TracedTensor::sym_shape). A program's
//! `DimExpr::InputDim` names a program input by position instead. Program input
//! #1 and the handle whose id is 1 are different things.
//!
//! The runtime and `tenferro-ad` conversion paths no longer fabricate handle
//! identity from a program position: a handle's cached extents are derived from
//! its value metadata as constants or as that handle's own axes, extension shape
//! inference gives each extension input a call-local id counted down from
//! `u64::MAX`, and the `tenferro-ad` traced path asks the handle for its axis
//! (#2443).
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
//! The removed `dim_expr_to_sym_dim_public` helper, which turned a program
//! coordinate into a `SymDim`, stays removed (#2443). This pins that one public
//! path; it does not show that no other path exists (see Known limits):
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
//!
//! # Known limits
//!
//! The guarantee above is a property of those paths, not of the type. The raw
//! constructor `SymDim::tensor_axis(tensor_id, axis)` (defined in
//! `tenferro-internal-ops`, `src/sym_dim.rs`) is `#[doc(hidden)]` but `pub`, and
//! reaches every downstream crate as `tenferro_runtime::SymDim::tensor_axis`
//! through the re-export below. It builds an axis carrying any id, including one
//! equal to a live handle's id or to a program input position, and a conversion
//! that knows a handle with that id accepts it. Its in-tree callers outside the
//! runtime are `tenferro-einsum`'s extension AD, which uses ids counted down from
//! `u64::MAX`, and `tenferro-linalg`'s semantic AD (`synthetic_input_shapes`),
//! which uses program input positions offset by one. Keeping those ids apart from
//! handle ids is the callers' responsibility, not the constructor's.

pub use tenferro_ops::sym_dim::*;
