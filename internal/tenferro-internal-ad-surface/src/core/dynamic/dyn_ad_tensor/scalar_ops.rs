mod mixed;

use tenferro_internal_ad_core::{AdTensor, DynAdTensor, DynAdTensorRef};

use super::super::tensor_ops::tensor_max_abs_diff_typed;
use super::basics::ensure_common_reverse_tape_impl;
use super::merge::merge_add_ad_tensors;
use super::promotion::{promote_many_to_common, promote_pair_to_common};
use super::Tensor;
use crate::{DynTensor, Error, Result};

use mixed::{div_ad_tensor_typed, scale_ad_tensor_typed};

#[cfg(test)]
mod tests;

macro_rules! match_dyn_ad_tensor_ref_pair {
    ($op_name:literal, $lhs:expr, $rhs:expr, |$lhs_var:ident, $rhs_var:ident| $body:block) => {{
        let lhs_ref = $lhs;
        let rhs_ref = $rhs;
        match (lhs_ref, rhs_ref) {
            (DynAdTensorRef::F32($lhs_var), DynAdTensorRef::F32($rhs_var)) => $body,
            (DynAdTensorRef::F64($lhs_var), DynAdTensorRef::F64($rhs_var)) => $body,
            (DynAdTensorRef::C32($lhs_var), DynAdTensorRef::C32($rhs_var)) => $body,
            (DynAdTensorRef::C64($lhs_var), DynAdTensorRef::C64($rhs_var)) => $body,
            _ => Err(Error::InvalidAdTensor {
                message: format!(
                    "dtype mismatch in {}: lhs={:?}, rhs={:?}",
                    $op_name,
                    lhs_ref.scalar_type(),
                    rhs_ref.scalar_type()
                ),
            }),
        }
    }};
}

macro_rules! match_dyn_ad_tensor_ref_quad {
    ($op_name:literal, $lhs:expr, $a:expr, $rhs:expr, $b:expr, |$lhs_var:ident, $a_var:ident, $rhs_var:ident, $b_var:ident| $body:block) => {{
        let lhs_ref = $lhs;
        let a_ref = $a;
        let rhs_ref = $rhs;
        let b_ref = $b;
        match (lhs_ref, a_ref, rhs_ref, b_ref) {
            (
                DynAdTensorRef::F32($lhs_var),
                DynAdTensorRef::F32($a_var),
                DynAdTensorRef::F32($rhs_var),
                DynAdTensorRef::F32($b_var),
            ) => $body,
            (
                DynAdTensorRef::F64($lhs_var),
                DynAdTensorRef::F64($a_var),
                DynAdTensorRef::F64($rhs_var),
                DynAdTensorRef::F64($b_var),
            ) => $body,
            (
                DynAdTensorRef::C32($lhs_var),
                DynAdTensorRef::C32($a_var),
                DynAdTensorRef::C32($rhs_var),
                DynAdTensorRef::C32($b_var),
            ) => $body,
            (
                DynAdTensorRef::C64($lhs_var),
                DynAdTensorRef::C64($a_var),
                DynAdTensorRef::C64($rhs_var),
                DynAdTensorRef::C64($b_var),
            ) => $body,
            _ => Err(Error::InvalidAdTensor {
                message: format!(
                    "dtype mismatch in {}: lhs={:?}, a={:?}, rhs={:?}, b={:?}",
                    $op_name,
                    lhs_ref.scalar_type(),
                    a_ref.scalar_type(),
                    rhs_ref.scalar_type(),
                    b_ref.scalar_type()
                ),
            }),
        }
    }};
}

fn scale_dyn(tensor: DynAdTensorRef<'_>, scalar: DynAdTensorRef<'_>) -> Result<DynAdTensor> {
    match_dyn_ad_tensor_ref_pair!("scale", tensor, scalar, |tensor, scalar| {
        Ok(scale_ad_tensor_typed(tensor, scalar)?.into())
    })
}

fn div_scalar_dyn(tensor: DynAdTensorRef<'_>, scalar: DynAdTensorRef<'_>) -> Result<DynAdTensor> {
    match_dyn_ad_tensor_ref_pair!("div_scalar", tensor, scalar, |tensor, scalar| {
        Ok(div_ad_tensor_typed(tensor, scalar)?.into())
    })
}

fn axpby_dyn(
    lhs: DynAdTensorRef<'_>,
    a: DynAdTensorRef<'_>,
    rhs: DynAdTensorRef<'_>,
    b: DynAdTensorRef<'_>,
) -> Result<DynAdTensor> {
    match_dyn_ad_tensor_ref_quad!("axpby", lhs, a, rhs, b, |lhs, a, rhs, b| {
        let lhs = scale_ad_tensor_typed(lhs, a)?;
        let rhs = scale_ad_tensor_typed(rhs, b)?;
        Ok(AdTensor::try_from(merge_add_ad_tensors(lhs.snapshot()?, rhs.snapshot()?)?)?.into())
    })
}

mod api;
