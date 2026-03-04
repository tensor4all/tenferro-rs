//! PartialDiagonal tensor metadata and tensor wrappers.
//!
//! # Overview
//!
//! `PartialDiagTensor<T>` stores a compressed payload plus logical axis-class
//! metadata. Dense and diagonal tensors are represented as special cases of the
//! same type, so there is no separate Dense/Diag type hierarchy in this module.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_dyadtensor::partial_diag::PartialDiagTensor;
//! use tenferro_tensor::{MemoryOrder, Tensor};
//!
//! let payload = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();
//! let x = PartialDiagTensor::new(vec![3, 3], vec![0, 0], payload).unwrap();
//! assert_eq!(x.class_count(), 1);
//! ```

mod dyn_tensor;
pub mod meta;
mod typed;

pub use dyn_tensor::DynPartialDiagTensor;
pub use meta::{
    plan_axis_classes_for_subscripts, AxisClassMergePlan, AxisClassPlanError, OperandAxisClassPlan,
    OperandAxisClasses,
};
pub use typed::PartialDiagTensor;
