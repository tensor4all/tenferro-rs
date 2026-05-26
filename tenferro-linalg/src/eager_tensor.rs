use std::sync::Arc;

use tenferro_ad::error::{Error, Result};
use tenferro_ad::extension::apply_eager;
use tenferro_ad::EagerTensor;

use crate::extension::{ensure_linalg_extension_rule_registered, LinalgExtensionOp, LinalgOp};

pub fn svd(a: &EagerTensor) -> Result<(EagerTensor, EagerTensor, EagerTensor)> {
    ensure_ad_rule_registered()?;
    let mut outputs = apply_eager(
        Arc::new(LinalgExtensionOp::new(LinalgOp::Svd { eps: 0.0 })),
        &[a],
    )?
    .into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(u), Some(s), Some(vt), None) => Ok((u, s, vt)),
        _ => Err(Error::Internal(
            "svd eager op returned an unexpected number of outputs".to_string(),
        )),
    }
}

pub fn qr(a: &EagerTensor) -> Result<(EagerTensor, EagerTensor)> {
    ensure_ad_rule_registered()?;
    two_outputs(
        apply_eager(Arc::new(LinalgExtensionOp::new(LinalgOp::Qr)), &[a])?,
        "qr",
    )
}

pub fn lu(a: &EagerTensor) -> Result<(EagerTensor, EagerTensor, EagerTensor, EagerTensor)> {
    ensure_ad_rule_registered()?;
    let mut outputs =
        apply_eager(Arc::new(LinalgExtensionOp::new(LinalgOp::Lu)), &[a])?.into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(p), Some(l), Some(u), Some(parity), None) => Ok((p, l, u, parity)),
        _ => Err(Error::Internal(
            "lu eager op returned an unexpected number of outputs".to_string(),
        )),
    }
}

pub fn full_piv_lu(
    a: &EagerTensor,
) -> Result<(
    EagerTensor,
    EagerTensor,
    EagerTensor,
    EagerTensor,
    EagerTensor,
)> {
    ensure_ad_rule_registered()?;
    let mut outputs =
        apply_eager(Arc::new(LinalgExtensionOp::new(LinalgOp::FullPivLu)), &[a])?.into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(p), Some(l), Some(u), Some(q), Some(parity), None) => Ok((p, l, u, q, parity)),
        _ => Err(Error::Internal(
            "full_piv_lu eager op returned an unexpected number of outputs".to_string(),
        )),
    }
}

pub fn full_piv_lu_solve(a: &EagerTensor, b: &EagerTensor) -> Result<EagerTensor> {
    ensure_ad_rule_registered()?;
    one_output(
        apply_eager(
            Arc::new(LinalgExtensionOp::new(LinalgOp::FullPivLuSolve {
                transpose_a: false,
            })),
            &[a, b],
        )?,
        "full_piv_lu_solve",
    )
}

pub fn solve(a: &EagerTensor, b: &EagerTensor) -> Result<EagerTensor> {
    ensure_ad_rule_registered()?;
    one_output(
        apply_eager(
            Arc::new(LinalgExtensionOp::new(LinalgOp::Solve {
                transpose_a: false,
            })),
            &[a, b],
        )?,
        "solve",
    )
}

pub fn cholesky(a: &EagerTensor) -> Result<EagerTensor> {
    ensure_ad_rule_registered()?;
    one_output(
        apply_eager(Arc::new(LinalgExtensionOp::new(LinalgOp::Cholesky)), &[a])?,
        "cholesky",
    )
}

pub fn eigh(a: &EagerTensor) -> Result<(EagerTensor, EagerTensor)> {
    ensure_ad_rule_registered()?;
    two_outputs(
        apply_eager(
            Arc::new(LinalgExtensionOp::new(LinalgOp::Eigh { eps: 0.0 })),
            &[a],
        )?,
        "eigh",
    )
}

pub fn eig(a: &EagerTensor) -> Result<(EagerTensor, EagerTensor)> {
    ensure_ad_rule_registered()?;
    two_outputs(
        apply_eager(
            Arc::new(LinalgExtensionOp::new(LinalgOp::Eig {
                input_dtype: a.data().dtype(),
            })),
            &[a],
        )?,
        "eig",
    )
}

pub fn triangular_solve(
    a: &EagerTensor,
    b: &EagerTensor,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> Result<EagerTensor> {
    ensure_ad_rule_registered()?;
    one_output(
        apply_eager(
            Arc::new(LinalgExtensionOp::new(LinalgOp::TriangularSolve {
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            })),
            &[a, b],
        )?,
        "triangular_solve",
    )
}

fn one_output(outputs: Vec<EagerTensor>, name: &str) -> Result<EagerTensor> {
    let mut outputs = outputs.into_iter();
    match (outputs.next(), outputs.next()) {
        (Some(output), None) => Ok(output),
        _ => Err(Error::Internal(format!(
            "{name} eager op returned an unexpected number of outputs"
        ))),
    }
}

fn two_outputs(outputs: Vec<EagerTensor>, name: &str) -> Result<(EagerTensor, EagerTensor)> {
    let mut outputs = outputs.into_iter();
    match (outputs.next(), outputs.next(), outputs.next()) {
        (Some(lhs), Some(rhs), None) => Ok((lhs, rhs)),
        _ => Err(Error::Internal(format!(
            "{name} eager op returned an unexpected number of outputs"
        ))),
    }
}

fn ensure_ad_rule_registered() -> Result<()> {
    ensure_linalg_extension_rule_registered().map_err(|err| Error::Internal(err.to_string()))
}
