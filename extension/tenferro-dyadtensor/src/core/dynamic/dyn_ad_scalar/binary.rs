use chainrules_scalarops::ScalarAd;
use num_complex::{Complex32, Complex64};

use super::super::ScalarType;
use super::DynAdScalar;
use crate::core::value::map_ad_value_mixed_linear;
use crate::{AdScalar, AdValue, Error, Result};

#[derive(Clone, Copy)]
pub(super) enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinaryOp {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
        }
    }
}

pub(crate) fn promote_f32_to_c32(
    value: AdValue<f32>,
    _op_name: &'static str,
) -> AdValue<Complex32> {
    map_ad_value_mixed_linear(value, |x| Complex32::new(x, 0.0), |z| z.re)
}

pub(crate) fn promote_f64_to_c64(
    value: AdValue<f64>,
    _op_name: &'static str,
) -> AdValue<Complex64> {
    map_ad_value_mixed_linear(value, |x| Complex64::new(x, 0.0), |z| z.re)
}

pub(super) fn embed_f32_to_c32_imag(
    value: AdValue<f32>,
    _op_name: &'static str,
) -> AdValue<Complex32> {
    map_ad_value_mixed_linear(value, |y| Complex32::new(0.0, y), |z| z.im)
}

pub(super) fn embed_f64_to_c64_imag(
    value: AdValue<f64>,
    _op_name: &'static str,
) -> AdValue<Complex64> {
    map_ad_value_mixed_linear(value, |y| Complex64::new(0.0, y), |z| z.im)
}

fn apply_binary_ad<T: ScalarAd + 'static>(
    lhs: AdValue<T>,
    rhs: AdValue<T>,
    op: BinaryOp,
) -> Result<AdValue<T>> {
    let lhs = AdScalar::from(lhs);
    let rhs = AdScalar::from(rhs);
    match op {
        BinaryOp::Add => (lhs + rhs).map(AdScalar::into_value),
        BinaryOp::Sub => (lhs - rhs).map(AdScalar::into_value),
        BinaryOp::Mul => (lhs * rhs).map(AdScalar::into_value),
        BinaryOp::Div => (lhs / rhs).map(AdScalar::into_value),
    }
}

fn check_reverse_tape_compatibility<T>(
    lhs: &AdValue<T>,
    rhs: &AdValue<T>,
    op: BinaryOp,
) -> Result<()> {
    match (lhs.tape_id(), rhs.tape_id()) {
        (Some(lhs_tape), Some(rhs_tape)) if lhs_tape != rhs_tape => Err(Error::InvalidAdScalar {
            message: format!(
                "{}: reverse-mode tape mismatch (lhs={}, rhs={})",
                op.name(),
                lhs_tape.0,
                rhs_tape.0
            ),
        }),
        _ => Ok(()),
    }
}

pub(crate) fn checked_apply_binary_ad<T: ScalarAd + 'static>(
    lhs: AdValue<T>,
    rhs: AdValue<T>,
    op: BinaryOp,
) -> Result<AdValue<T>> {
    check_reverse_tape_compatibility(&lhs, &rhs, op)?;
    apply_binary_ad(lhs, rhs, op)
}

fn unsupported_binary_pair(op: BinaryOp, lhs: ScalarType, rhs: ScalarType) -> Error {
    Error::InvalidAdScalar {
        message: format!(
            "unsupported dtype pair for `{}`: lhs={lhs:?}, rhs={rhs:?}",
            op.name()
        ),
    }
}

pub(super) fn try_binary_dyn(
    lhs: DynAdScalar,
    rhs: DynAdScalar,
    op: BinaryOp,
) -> Result<DynAdScalar> {
    let lhs_ty = lhs.scalar_type();
    let rhs_ty = rhs.scalar_type();
    match (lhs, rhs) {
        (DynAdScalar::F32(a), DynAdScalar::F32(b)) => {
            Ok(DynAdScalar::F32(checked_apply_binary_ad(a, b, op)?))
        }
        (DynAdScalar::F64(a), DynAdScalar::F64(b)) => {
            Ok(DynAdScalar::F64(checked_apply_binary_ad(a, b, op)?))
        }
        (DynAdScalar::C32(a), DynAdScalar::C32(b)) => {
            Ok(DynAdScalar::C32(checked_apply_binary_ad(a, b, op)?))
        }
        (DynAdScalar::C64(a), DynAdScalar::C64(b)) => {
            Ok(DynAdScalar::C64(checked_apply_binary_ad(a, b, op)?))
        }
        (DynAdScalar::F32(a), DynAdScalar::C32(b)) => Ok(DynAdScalar::C32(
            checked_apply_binary_ad(promote_f32_to_c32(a, op.name()), b, op)?,
        )),
        (DynAdScalar::C32(a), DynAdScalar::F32(b)) => Ok(DynAdScalar::C32(
            checked_apply_binary_ad(a, promote_f32_to_c32(b, op.name()), op)?,
        )),
        (DynAdScalar::F64(a), DynAdScalar::C64(b)) => Ok(DynAdScalar::C64(
            checked_apply_binary_ad(promote_f64_to_c64(a, op.name()), b, op)?,
        )),
        (DynAdScalar::C64(a), DynAdScalar::F64(b)) => Ok(DynAdScalar::C64(
            checked_apply_binary_ad(a, promote_f64_to_c64(b, op.name()), op)?,
        )),
        _ => Err(unsupported_binary_pair(op, lhs_ty, rhs_ty)),
    }
}
