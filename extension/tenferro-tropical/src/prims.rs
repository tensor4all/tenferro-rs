//! Semiring-family implementations for tropical algebras on
//! [`tenferro_prims::CpuBackend`].
//!
//! Each tropical algebra gets its own `impl TensorSemiringCore<XxxAlgebra> for CpuBackend`
//! plus a `TensorSemiringFastPath` impl that reports no optional fast paths.
//! The orphan rule is satisfied because `XxxAlgebra` is defined in this crate.
//!
//! Optional fast paths (`Contract`, elementwise semiring binary ops) are not
//! supported for tropical algebras.
//!
//! Because tropical scalar types redefine `Add` (= max/min) and `Mul` (= +/×),
//! the standard ScalarBase-based helper functions work correctly: the expression
//! `sum = sum + a * b` becomes `sum = max(sum, a_val + b_val)` for MaxPlus,
//! which is exactly tropical GEMM.

#[path = "prims_execute.rs"]
mod prims_execute;
#[path = "prims_impls.rs"]
mod prims_impls;
#[path = "prims_plan.rs"]
mod prims_plan;
#[path = "prims_view.rs"]
mod prims_view;

pub use self::prims_plan::TropicalPlan;

#[allow(unused_imports)]
pub(crate) use self::prims_execute::{
    execute_anti_diag, execute_anti_trace, execute_batched_gemm_fallback, execute_make_contiguous,
    execute_trace, tropical_execute,
};
#[allow(unused_imports)]
pub(crate) use self::prims_plan::tropical_plan;
#[allow(unused_imports)]
pub(crate) use self::prims_view::{
    for_each_index, scale_output, tensor_to_view, tensor_to_view_mut, unflatten_index,
};

#[cfg(test)]
#[allow(unused_imports)]
use crate::algebra::MaxPlusAlgebra;
#[cfg(test)]
#[allow(unused_imports)]
use crate::scalar::MaxPlus;
#[cfg(test)]
#[allow(unused_imports)]
use std::marker::PhantomData;
#[cfg(test)]
#[allow(unused_imports)]
use tenferro_device::Error;
#[cfg(test)]
#[allow(unused_imports)]
use tenferro_prims::{
    CpuBackend, CpuContext, SemiringCoreDescriptor, SemiringFastPathDescriptor, TensorSemiringCore,
    TensorSemiringFastPath,
};

#[cfg(test)]
mod tests;
