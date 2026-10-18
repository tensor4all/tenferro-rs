//! Fused square-QR JVP/VJP kernels used by [`LinalgOp::QrJvp`] and
//! [`LinalgOp::QrVjp`](crate::extension::LinalgOp::QrVjp).
//!
//! The generic QR AD rule lowers a square JVP/VJP to a graph of roughly forty
//! primitive operations. On CUDA each primitive enters its own backend session,
//! so host-side scope entry dominates the small-kernel work. The fused ops run
//! the same formula through the backend session helpers in one extension call.
//!
//! The formulas below are the square (`m == n`, full-rank) case of
//! `linearize_qr_with_factors` in [`crate::ad::rules`]:
//!
//! ```text
//! JVP:  X   = R^{-H} dA^H
//!       M   = Q^H X
//!       phi = triu(M + M^H, 1) + 1/2 diag(M + M^H)
//!       dQ  = X - Q phi
//!       dR  = phi R
//!
//! VJP:  dPhi = gR R^H - Q^H gQ
//!       dPsi = triu(dPhi, 1) + 1/2 diag(dPhi)
//!       dX   = gQ + Q (dPsi + dPsi^H)
//!       dA   = (R^{-1} dX^H)^H
//! ```
//!
//! The VJP is the mechanical transpose of the JVP graph, including the
//! conjugate-transpose conventions the primitive transpose rules use.

use num_complex::{Complex32, Complex64};
use tenferro_tensor::{
    BackendSession, ContractionScalar, DType, DotGeneralConfig, Error, Tensor, TensorRead,
    TensorWrite,
};

use crate::backend::LinalgBackend;

/// Square-matrix batch contraction config matching the generic AD graph.
fn matrix_config(rank: usize) -> DotGeneralConfig {
    assert!(rank >= 2, "matrix contraction expects rank >= 2");
    let batch_dims: Vec<usize> = (2..rank).collect();
    DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: batch_dims.iter().copied().collect(),
        rhs_batch_dims: batch_dims.iter().copied().collect(),
    }
}

fn is_complex(dtype: DType) -> bool {
    matches!(dtype, DType::C32 | DType::C64)
}

fn transpose_perm(rank: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.swap(0, 1);
    perm
}

fn validate_square(op: &'static str, q: &Tensor, r: &Tensor, other: &Tensor) -> Result<(), Error> {
    let rank = q.shape().len();
    if rank < 2 {
        return Err(Error::rank_mismatch(op, 2, rank));
    }
    if q.shape()[0] != q.shape()[1] {
        return Err(Error::shape_mismatch(op, [q.shape()[0]], [q.shape()[1]]));
    }
    if r.shape() != q.shape() {
        return Err(Error::shape_mismatch(op, q.shape(), r.shape()));
    }
    if other.shape() != q.shape() {
        return Err(Error::shape_mismatch(op, q.shape(), other.shape()));
    }
    if !matches!(q.dtype(), DType::F32 | DType::F64 | DType::C32 | DType::C64) {
        return Err(Error::unsupported(
            op,
            format!("dtype {:?} has no fused QR derivative kernel", q.dtype()),
        ));
    }
    if r.dtype() != q.dtype() || other.dtype() != q.dtype() {
        return Err(Error::dtype_mismatch(op, q.dtype(), r.dtype()));
    }
    Ok(())
}

fn transpose_read<B: BackendSession + ?Sized>(
    session: &mut B,
    input: &Tensor,
    perm: &[usize],
) -> Result<Tensor, Error> {
    session.transpose_read(TensorRead::from_tensor(input), perm)
}

/// Conjugate transpose of a matrix operand, skipping the conjugate for real
/// dtypes exactly like the generic graph's `adjoint_matrix_*` helpers.
fn adjoint<B: BackendSession + ?Sized>(session: &mut B, input: &Tensor) -> Result<Tensor, Error> {
    let perm = transpose_perm(input.shape().len());
    if is_complex(input.dtype()) {
        let conjugated = session.conj_read(TensorRead::from_tensor(input))?;
        transpose_read(session, &conjugated, &perm)
    } else {
        transpose_read(session, input, &perm)
    }
}

fn matmul<B: BackendSession + ?Sized>(
    session: &mut B,
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Tensor, Error> {
    let config = matrix_config(lhs.shape().len());
    session.dot_general_read_cached(
        None,
        TensorRead::from_tensor(lhs),
        TensorRead::from_tensor(rhs),
        &config,
    )
}

fn add<B: BackendSession + ?Sized>(
    session: &mut B,
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Tensor, Error> {
    session.add_read(TensorRead::from_tensor(lhs), TensorRead::from_tensor(rhs))
}

fn sub<B: BackendSession + ?Sized>(
    session: &mut B,
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Tensor, Error> {
    session.sub_read(TensorRead::from_tensor(lhs), TensorRead::from_tensor(rhs))
}

fn half_scalar(dtype: DType) -> Result<ContractionScalar, Error> {
    match dtype {
        DType::F32 => Ok(ContractionScalar::F32(0.5)),
        DType::F64 => Ok(ContractionScalar::F64(0.5)),
        DType::C32 => Ok(ContractionScalar::C32(Complex32::new(0.5, 0.0))),
        DType::C64 => Ok(ContractionScalar::C64(Complex64::new(0.5, 0.0))),
        DType::I32 | DType::I64 | DType::Bool | DType::External(_) => Err(Error::unsupported(
            "qr_backward",
            format!("dtype {dtype:?} has no fused QR derivative kernel"),
        )),
    }
}

/// Scale a tensor by the real scalar `0.5`.
///
/// `axpby` writes into caller-owned storage, so the destination is first
/// materialized as a fresh unary output. That keeps the source immutable and
/// the two buffers provably disjoint for the backend aliasing check.
fn scale_half<B: BackendSession + ?Sized>(
    session: &mut B,
    input: &Tensor,
) -> Result<Tensor, Error> {
    let dtype = input.dtype();
    let alpha = half_scalar(dtype)?;
    let beta = ContractionScalar::zero(dtype)?;
    let mut output = session.neg_read(TensorRead::from_tensor(input))?;
    session.axpby_read_into_accum(
        alpha,
        TensorRead::from_tensor(input),
        beta,
        TensorWrite::from_tensor(&mut output),
    )?;
    Ok(output)
}

/// `phi = triu(sym, 1) + 1/2 diag(sym)`.
fn upper_with_half_diagonal<B: BackendSession + ?Sized>(
    session: &mut B,
    symmetric: &Tensor,
) -> Result<Tensor, Error> {
    let upper = session.triu(symmetric, 1)?;
    let diagonal = session.extract_diagonal(symmetric, 0, 1)?;
    let half_diagonal = scale_half(session, &diagonal)?;
    let embedded = session.embed_diagonal(&half_diagonal, 0, 1)?;
    add(session, &upper, &embedded)
}

/// Fused square-QR JVP.
///
/// Inputs are the primal factors `q`, `r` and the tangent `da`. Outputs follow
/// `q_active`/`r_active`: `dq` first when requested, then `dr`.
pub(crate) fn qr_jvp<B: LinalgBackend + ?Sized>(
    session: &mut B,
    q: &Tensor,
    r: &Tensor,
    da: &Tensor,
    q_active: bool,
    r_active: bool,
) -> Result<Vec<Tensor>, Error> {
    validate_square("qr_jvp", q, r, da)?;
    let root = |session: &mut B| -> Result<(Tensor, Tensor), Error> {
        let r_h = adjoint(session, r)?;
        let da_h = adjoint(session, da)?;
        let x_rinv_h = session.triangular_solve(&r_h, &da_h, true, true, false, false)?;
        let x_rinv = adjoint(session, &x_rinv_h)?;
        let q_h = adjoint(session, q)?;
        let m = matmul(session, &q_h, &x_rinv)?;
        let m_h = adjoint(session, &m)?;
        let symmetric = add(session, &m, &m_h)?;
        let phi = upper_with_half_diagonal(session, &symmetric)?;
        Ok((x_rinv, phi))
    };
    let (x_rinv, phi) = root(session)?;

    let mut outputs = Vec::with_capacity(usize::from(q_active) + usize::from(r_active));
    if q_active {
        let q_phi = matmul(session, q, &phi)?;
        outputs.push(sub(session, &x_rinv, &q_phi)?);
    }
    if r_active {
        outputs.push(matmul(session, &phi, r)?);
    }
    Ok(outputs)
}

/// Fused square-QR VJP.
///
/// Inputs are `q`, `r`, and the cotangents that `q_active`/`r_active` request.
/// The single output is the cotangent of the QR input.
pub(crate) fn qr_vjp<B: LinalgBackend + ?Sized>(
    session: &mut B,
    q: &Tensor,
    r: &Tensor,
    q_cotangent: Option<&Tensor>,
    r_cotangent: Option<&Tensor>,
) -> Result<Tensor, Error> {
    let reference = q_cotangent.or(r_cotangent).ok_or_else(|| {
        Error::invalid_argument(
            "qr_vjp",
            "cotangent",
            "at least one QR output cotangent must be present",
        )
    })?;
    validate_square("qr_vjp", q, r, reference)?;

    let r_h = adjoint(session, r)?;
    let mut d_phi: Option<Tensor> = match r_cotangent {
        Some(cotangent) => Some(matmul(session, cotangent, &r_h)?),
        None => None,
    };

    let d_x_direct = q_cotangent;
    if let Some(cotangent) = q_cotangent {
        let q_h = adjoint(session, q)?;
        let from_q = matmul(session, &q_h, cotangent)?;
        let negated = session.neg_read(TensorRead::from_tensor(&from_q))?;
        d_phi = Some(match d_phi {
            Some(existing) => add(session, &existing, &negated)?,
            None => negated,
        });
    }
    // INVARIANT: either cotangent branch above populated `d_phi`.
    let d_phi = d_phi.ok_or_else(|| {
        Error::invalid_argument(
            "qr_vjp",
            "cotangent",
            "at least one QR output cotangent must be present",
        )
    })?;

    let d_psi = upper_with_half_diagonal(session, &d_phi)?;
    let d_psi_h = adjoint(session, &d_psi)?;
    let symmetric = add(session, &d_psi, &d_psi_h)?;
    let q_symmetric = matmul(session, q, &symmetric)?;
    let d_x = match d_x_direct {
        Some(cotangent) => add(session, cotangent, &q_symmetric)?,
        None => q_symmetric,
    };
    let d_x_h = adjoint(session, &d_x)?;
    // `transpose(r) = conj(adjoint(r))`, the matrix operand the generic
    // transpose rule builds for the flipped triangular solve.
    let r_t = transpose_read(session, r, &transpose_perm(r.shape().len()))?;
    let d_a_h = session.triangular_solve(&r_t, &d_x_h, true, true, true, false)?;
    adjoint(session, &d_a_h)
}
