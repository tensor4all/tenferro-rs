//! Eager AD entry points without `_ad` suffix.
//!
//! These functions are thin wrappers around the existing builder APIs
//! (`*_ad(...).run()`) and are intended for integration code paths that prefer
//! explicit eager execution.
//!
//! # Examples
//!
//! ```text
//! use tenferro::{Tensor, set_default_runtime, RuntimeContext};
//! use tenferro_prims::CpuContext;
//! use tenferro_tensor::{MemoryOrder, Tensor};
//!
//! let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
//! let a = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
//!     .unwrap();
//! let ad_a = Tensor::new_primal(a);
//! let out = ad_a.qr().unwrap();
//! assert_eq!(out.q.dims(), &[2, 2]);
//! ```

use tenferro_algebra::Scalar;
use tenferro_internal_ad_core::{AdTensor, DynAdTensor, DynAdTensorRef};
use tidu::Value;

use crate::structured::StructuredTensor;
use crate::{Error, Result};

use super::{einsum_ad, sum_ad, EinsumRuntimeValue, ScalarRuntimeValue};

mod layout;
mod pullback;
mod scalar_eager;

#[doc(hidden)]
pub use layout::normalize_cotangent_payload;
pub use pullback::{
    einsum_frule, einsum_hvp, einsum_rrule, pullback, pullback_wrt, solve_triangular_rrule,
};
pub use scalar_eager::{
    acos, acosh, add, asin, asinh, atan, atan2, atanh, cos, cosh, exp, expm1, hypot, log, log1p,
    mean, pow, sin, sinh, sqrt, std, tanh, var,
};

pub(crate) fn wrap_reverse_edge_output<T>(output: Value<StructuredTensor<T>>) -> Result<AdTensor<T>>
where
    T: Scalar + tenferro_internal_frontend_core::DynTensorTyped + 'static,
{
    AdTensor::from_reverse_edge_value(output)
}

macro_rules! match_dyn_unary_all {
    ($tensor:expr, $typed_fn:path) => {{
        match $tensor {
            DynAdTensorRef::F32(value) => Ok($typed_fn(value)?.into()),
            DynAdTensorRef::F64(value) => Ok($typed_fn(value)?.into()),
            DynAdTensorRef::C32(value) => Ok($typed_fn(value)?.into()),
            DynAdTensorRef::C64(value) => Ok($typed_fn(value)?.into()),
        }
    }};
}

macro_rules! match_dyn_binary_same_dtype {
    ($fn_name:ident, $lhs:expr, $rhs:expr, $typed_fn:path) => {{
        match ($lhs, $rhs) {
            (DynAdTensorRef::F32(lhs), DynAdTensorRef::F32(rhs)) => Ok($typed_fn(lhs, rhs)?.into()),
            (DynAdTensorRef::F64(lhs), DynAdTensorRef::F64(rhs)) => Ok($typed_fn(lhs, rhs)?.into()),
            (DynAdTensorRef::C32(lhs), DynAdTensorRef::C32(rhs)) => Ok($typed_fn(lhs, rhs)?.into()),
            (DynAdTensorRef::C64(lhs), DynAdTensorRef::C64(rhs)) => Ok($typed_fn(lhs, rhs)?.into()),
            (lhs, rhs) => Err(Error::InvalidAdTensor {
                message: format!(
                    "{} requires matching DynAdTensor inputs, got lhs={:?}, rhs={:?}",
                    stringify!($fn_name),
                    lhs.scalar_type(),
                    rhs.scalar_type()
                ),
            }),
        }
    }};
}

macro_rules! match_dyn_binary_real_only {
    ($fn_name:ident, $lhs:expr, $rhs:expr, $typed_fn:path) => {{
        let lhs = $lhs;
        let rhs = $rhs;
        match (lhs, rhs) {
            (DynAdTensorRef::F32(lhs), DynAdTensorRef::F32(rhs)) => Ok($typed_fn(lhs, rhs)?.into()),
            (DynAdTensorRef::F64(lhs), DynAdTensorRef::F64(rhs)) => Ok($typed_fn(lhs, rhs)?.into()),
            (lhs @ DynAdTensorRef::F32(_), rhs @ DynAdTensorRef::F64(_))
            | (lhs @ DynAdTensorRef::F64(_), rhs @ DynAdTensorRef::F32(_)) => {
                Err(Error::InvalidAdTensor {
                    message: format!(
                        "{} requires matching DynAdTensor inputs, got lhs={:?}, rhs={:?}",
                        stringify!($fn_name),
                        lhs.scalar_type(),
                        rhs.scalar_type()
                    ),
                })
            }
            (lhs, rhs) => Err(Error::InvalidAdTensor {
                message: format!(
                    "{} requires real-valued operands, got lhs={:?}, rhs={:?}",
                    stringify!($fn_name),
                    lhs.scalar_type(),
                    rhs.scalar_type()
                ),
            }),
        }
    }};
}

macro_rules! match_dyn_reduction_real_only {
    ($fn_name:ident, $tensor:expr, $typed_fn:path) => {{
        match $tensor {
            DynAdTensorRef::F32(value) => Ok($typed_fn(value)?.into()),
            DynAdTensorRef::F64(value) => Ok($typed_fn(value)?.into()),
            DynAdTensorRef::C32(_) | DynAdTensorRef::C64(_) => Err(Error::InvalidAdTensor {
                message: format!("{} requires real-valued input", stringify!($fn_name)),
            }),
        }
    }};
}

/// Eager AD einsum.
///
/// Equivalent to `crate::einsum_ad(...).run()`.
///
/// # Examples
///
/// ```text
/// let out = tenferro::ad::einsum("ij,jk->ik", &[&a, &b])?;
/// ```
pub fn einsum<'a, T>(subscripts: &'a str, operands: &'a [&'a AdTensor<T>]) -> Result<AdTensor<T>>
where
    T: EinsumRuntimeValue,
{
    einsum_ad(subscripts, operands).run()
}

/// Eager AD full reduction / sum.
///
/// Equivalent to `crate::sum_ad(...).run()`.
///
/// # Examples
///
/// ```text
/// let out = tenferro::ad::sum(&x)?;
/// ```
pub fn sum<T>(tensor: &AdTensor<T>) -> Result<AdTensor<T>>
where
    T: ScalarRuntimeValue,
{
    sum_ad(tensor).run()
}

/// Eager AD `exp` for erased dynamic AD tensors.
pub fn exp_dyn(tensor: DynAdTensorRef<'_>) -> Result<DynAdTensor> {
    match_dyn_unary_all!(tensor, exp)
}

macro_rules! define_dyn_unary_entrypoint {
    ($fn_name:ident, $typed_fn:path) => {
        /// Eager AD entry point for erased dynamic AD tensors.
        pub fn $fn_name(tensor: DynAdTensorRef<'_>) -> Result<DynAdTensor> {
            match tensor {
                DynAdTensorRef::F32(value) => Ok($typed_fn(value)?.into()),
                DynAdTensorRef::F64(value) => Ok($typed_fn(value)?.into()),
                DynAdTensorRef::C32(value) => Ok($typed_fn(value)?.into()),
                DynAdTensorRef::C64(value) => Ok($typed_fn(value)?.into()),
            }
        }
    };
}

macro_rules! define_dyn_binary_entrypoint {
    ($fn_name:ident, $typed_fn:path) => {
        /// Eager AD entry point for erased dynamic AD tensors.
        pub fn $fn_name(lhs: DynAdTensorRef<'_>, rhs: DynAdTensorRef<'_>) -> Result<DynAdTensor> {
            match_dyn_binary_same_dtype!($fn_name, lhs, rhs, $typed_fn)
        }
    };
}

macro_rules! define_dyn_binary_real_entrypoint {
    ($fn_name:ident, $typed_fn:path) => {
        /// Eager AD entry point for erased dynamic AD tensors.
        pub fn $fn_name(lhs: DynAdTensorRef<'_>, rhs: DynAdTensorRef<'_>) -> Result<DynAdTensor> {
            match_dyn_binary_real_only!($fn_name, lhs, rhs, $typed_fn)
        }
    };
}

macro_rules! define_dyn_reduction_real_entrypoint {
    ($fn_name:ident, $typed_fn:path) => {
        /// Eager AD full reduction for erased dynamic AD tensors.
        pub fn $fn_name(tensor: DynAdTensorRef<'_>) -> Result<DynAdTensor> {
            match_dyn_reduction_real_only!($fn_name, tensor, $typed_fn)
        }
    };
}

/// Eager AD `add` for erased dynamic AD tensors.
///
/// Unlike the typed variant, this wrapper requires both operands to already
/// share a dtype. Dynamic promotion remains the caller's responsibility.
pub fn add_dyn(lhs: DynAdTensorRef<'_>, rhs: DynAdTensorRef<'_>) -> Result<DynAdTensor> {
    match_dyn_binary_same_dtype!(add_dyn, lhs, rhs, add)
}

/// Eager AD `mean` for erased dynamic AD tensors.
pub fn mean_dyn(tensor: DynAdTensorRef<'_>) -> Result<DynAdTensor> {
    match_dyn_unary_all!(tensor, mean)
}

define_dyn_unary_entrypoint!(sqrt_dyn, sqrt);
define_dyn_unary_entrypoint!(expm1_dyn, expm1);
define_dyn_unary_entrypoint!(log_dyn, log);
define_dyn_unary_entrypoint!(log1p_dyn, log1p);
define_dyn_unary_entrypoint!(sin_dyn, sin);
define_dyn_unary_entrypoint!(cos_dyn, cos);
define_dyn_unary_entrypoint!(tanh_dyn, tanh);
define_dyn_unary_entrypoint!(asin_dyn, asin);
define_dyn_unary_entrypoint!(acos_dyn, acos);
define_dyn_unary_entrypoint!(atan_dyn, atan);
define_dyn_unary_entrypoint!(sinh_dyn, sinh);
define_dyn_unary_entrypoint!(cosh_dyn, cosh);
define_dyn_unary_entrypoint!(asinh_dyn, asinh);
define_dyn_unary_entrypoint!(acosh_dyn, acosh);
define_dyn_unary_entrypoint!(atanh_dyn, atanh);

define_dyn_reduction_real_entrypoint!(std_dyn, std);
define_dyn_reduction_real_entrypoint!(var_dyn, var);

define_dyn_binary_real_entrypoint!(atan2_dyn, atan2);
define_dyn_binary_real_entrypoint!(hypot_dyn, hypot);
define_dyn_binary_entrypoint!(pow_dyn, pow);

/// Eager AD full reduction / sum for erased dynamic AD tensors.
pub fn sum_dyn(tensor: DynAdTensorRef<'_>) -> Result<DynAdTensor> {
    macro_rules! sum_branch {
        ($value:ident) => {{
            if crate::ops::reduction::ad::can_use_edge_sum_reverse($value) {
                return crate::ops::reduction::ad::edge_sum($value).map(Into::into);
            }
            Ok(sum($value)?.into())
        }};
    }
    match tensor {
        DynAdTensorRef::F32(value) => sum_branch!(value),
        DynAdTensorRef::F64(value) => sum_branch!(value),
        DynAdTensorRef::C32(value) => sum_branch!(value),
        DynAdTensorRef::C64(value) => sum_branch!(value),
    }
}

/// Eager AD einsum for erased dynamic AD tensors.
///
/// Unlike the typed variant, this wrapper requires all operands to already
/// share a dtype. Dynamic promotion remains the caller's responsibility.
pub fn einsum_dyn(subscripts: &str, operands: &[DynAdTensorRef<'_>]) -> Result<DynAdTensor> {
    let Some(first) = operands.first().copied() else {
        return Err(Error::InvalidAdTensor {
            message: "einsum_dyn requires at least one operand".to_string(),
        });
    };

    macro_rules! dispatch_dyn_einsum {
        ($variant:ident) => {{
            let mut refs = Vec::with_capacity(operands.len());
            for operand in operands {
                match operand {
                    DynAdTensorRef::$variant(value) => refs.push(*value),
                    _ => {
                        return Err(Error::InvalidAdTensor {
                            message: format!(
                                "einsum_dyn requires matching DynAdTensor inputs, got first={:?}, operand={:?}",
                                first.scalar_type(),
                                operand.scalar_type()
                            ),
                        });
                    }
                }
            }
            if crate::ops::einsum::ad::can_use_edge_einsum_reverse(&refs) {
                return crate::ops::einsum::ad::edge_einsum(subscripts, &refs).map(Into::into);
            }
            Ok(einsum(subscripts, &refs)?.into())
        }};
    }

    match first {
        DynAdTensorRef::F32(_) => dispatch_dyn_einsum!(F32),
        DynAdTensorRef::F64(_) => dispatch_dyn_einsum!(F64),
        DynAdTensorRef::C32(_) => dispatch_dyn_einsum!(C32),
        DynAdTensorRef::C64(_) => dispatch_dyn_einsum!(C64),
    }
}
