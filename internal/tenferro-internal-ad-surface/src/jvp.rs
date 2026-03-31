use std::cell::RefCell;
use std::collections::HashMap;

use tenferro_internal_ad_core::{LinearizableOp, LinearizedOp};
use tenferro_internal_ad_linalg::{
    CholeskyOp, DetOp, EigOp, EigenOp, InvOp, LstsqOp, LuOp, MatrixExpOp, NormOp, PInvOp, QrOp,
    SlogdetOp, SolveOp, SolveTriangularOp, SvdOp,
};
use tenferro_internal_ad_ops::{AddOp, EinsumOp, ExpOp, SumOp};
use tenferro_internal_frontend_core::DynTensor;
use tenferro_linalg::{LuPivot, NormKind, SvdOptions};

use crate::{Error, Result, Tensor};

#[derive(Debug)]
pub struct JvpResult {
    pub outputs: Vec<Tensor>,
    pub output_tangents: Vec<Option<Tensor>>,
}

#[derive(Default)]
struct ForwardJvpContext {
    tangents: HashMap<usize, DynTensor>,
}

impl ForwardJvpContext {
    fn tangent_for_id(&self, id: usize) -> Option<DynTensor> {
        self.tangents.get(&id).cloned()
    }

    fn set_tangent(&mut self, id: usize, tangent: DynTensor) {
        self.tangents.insert(id, tangent);
    }
}

thread_local! {
    static FORWARD_JVP_STACK: RefCell<Vec<ForwardJvpContext>> = const { RefCell::new(Vec::new()) };
}

struct ForwardJvpGuard;

impl Drop for ForwardJvpGuard {
    fn drop(&mut self) {
        FORWARD_JVP_STACK.with(|stack| {
            stack.borrow_mut().pop();
        });
    }
}

fn invalid_argument(message: impl Into<String>) -> Error {
    Error::InvalidTensorOperands {
        message: message.into(),
    }
}

fn validate_compatibility(index: usize, primal: &Tensor, tangent: &Tensor) -> Result<()> {
    if primal.scalar_type() != tangent.scalar_type() {
        return Err(invalid_argument(format!(
            "jvp tangent {index} dtype mismatch: primal={:?}, tangent={:?}",
            primal.scalar_type(),
            tangent.scalar_type()
        )));
    }
    if primal.dims() != tangent.dims() {
        return Err(invalid_argument(format!(
            "jvp tangent {index} shape mismatch: primal={:?}, tangent={:?}",
            primal.dims(),
            tangent.dims()
        )));
    }
    if primal.axis_classes() != tangent.axis_classes()
        || primal.is_dense() != tangent.is_dense()
        || primal.is_diag() != tangent.is_diag()
    {
        return Err(invalid_argument(format!(
            "jvp tangent {index} layout mismatch: primal dense={} diag={} classes={:?}, tangent dense={} diag={} classes={:?}",
            primal.is_dense(),
            primal.is_diag(),
            primal.axis_classes(),
            tangent.is_dense(),
            tangent.is_diag(),
            tangent.axis_classes()
        )));
    }
    Ok(())
}

fn push_context(ctx: ForwardJvpContext) -> ForwardJvpGuard {
    FORWARD_JVP_STACK.with(|stack| {
        stack.borrow_mut().push(ctx);
    });
    ForwardJvpGuard
}

fn with_current_context<R>(f: impl FnOnce(&ForwardJvpContext) -> R) -> Option<R> {
    FORWARD_JVP_STACK.with(|stack| stack.borrow().last().map(f))
}

fn with_current_context_mut<R>(f: impl FnOnce(&mut ForwardJvpContext) -> R) -> Option<R> {
    FORWARD_JVP_STACK.with(|stack| stack.borrow_mut().last_mut().map(f))
}

pub(crate) fn is_active() -> bool {
    with_current_context(|_| ()).is_some()
}

pub(crate) fn tangent_for(tensor: &Tensor) -> Option<DynTensor> {
    with_current_context(|ctx| ctx.tangent_for_id(tensor.forward_id())).flatten()
}

pub(crate) fn record_tangent(tensor: &Tensor, tangent: Option<DynTensor>) {
    if let Some(tangent) = tangent {
        let _ = with_current_context_mut(|ctx| {
            ctx.set_tangent(tensor.forward_id(), tangent);
        });
    }
}

pub(crate) fn add_tangent(lhs: &Tensor, rhs: &Tensor, output: &Tensor) -> Result<()> {
    if !is_active() {
        return Ok(());
    }
    let linearized = AddOp.linearize(&[lhs.primal(), rhs.primal()], &[output.primal().clone()])?;
    let mut outputs = linearized.jvp(&[tangent_for(lhs), tangent_for(rhs)])?;
    record_tangent(output, outputs.pop().unwrap_or(None));
    Ok(())
}

pub(crate) fn exp_tangent(input: &Tensor, output: &Tensor) -> Result<()> {
    if !is_active() {
        return Ok(());
    }
    let linearized = ExpOp.linearize(&[input.primal()], &[output.primal().clone()])?;
    let mut outputs = linearized.jvp(&[tangent_for(input)])?;
    record_tangent(output, outputs.pop().unwrap_or(None));
    Ok(())
}

pub(crate) fn sum_tangent(input: &Tensor, output: &Tensor) -> Result<()> {
    if !is_active() {
        return Ok(());
    }
    let linearized = SumOp.linearize(&[input.primal()], &[output.primal().clone()])?;
    let mut outputs = linearized.jvp(&[tangent_for(input)])?;
    record_tangent(output, outputs.pop().unwrap_or(None));
    Ok(())
}

pub(crate) fn qr_tangents(input: &Tensor, q: &Tensor, r: &Tensor) -> Result<()> {
    if !is_active() {
        return Ok(());
    }
    let linearized =
        QrOp.linearize(&[input.primal()], &[q.primal().clone(), r.primal().clone()])?;
    let outputs = linearized.jvp(&[tangent_for(input)])?;
    let mut outputs = outputs.into_iter();
    record_tangent(q, outputs.next().unwrap_or(None));
    record_tangent(r, outputs.next().unwrap_or(None));
    Ok(())
}

pub(crate) fn einsum_tangent(
    subscripts: &str,
    operands: &[&Tensor],
    output: &Tensor,
) -> Result<()> {
    if !is_active() {
        return Ok(());
    }
    let primals = operands
        .iter()
        .map(|tensor| tensor.primal())
        .collect::<Vec<_>>();
    let linearized = EinsumOp::new(subscripts).linearize(&primals, &[output.primal().clone()])?;
    let tangents = operands
        .iter()
        .map(|tensor| tangent_for(tensor))
        .collect::<Vec<_>>();
    let mut outputs = linearized.jvp(&tangents)?;
    record_tangent(output, outputs.pop().unwrap_or(None));
    Ok(())
}

pub(crate) fn solve_tangent(lhs: &Tensor, rhs: &Tensor, output: &Tensor) -> Result<()> {
    if !is_active() {
        return Ok(());
    }
    let linearized =
        SolveOp.linearize(&[lhs.primal(), rhs.primal()], &[output.primal().clone()])?;
    let mut outputs = linearized.jvp(&[tangent_for(lhs), tangent_for(rhs)])?;
    record_tangent(output, outputs.pop().unwrap_or(None));
    Ok(())
}

pub(crate) fn lstsq_tangents(
    lhs: &Tensor,
    rhs: &Tensor,
    x: &Tensor,
    residual: &Tensor,
) -> Result<()> {
    if !is_active() {
        return Ok(());
    }
    let linearized = LstsqOp.linearize(
        &[lhs.primal(), rhs.primal()],
        &[x.primal().clone(), residual.primal().clone()],
    )?;
    let mut outputs = linearized
        .jvp(&[tangent_for(lhs), tangent_for(rhs)])?
        .into_iter();
    record_tangent(x, outputs.next().unwrap_or(None));
    record_tangent(residual, outputs.next().unwrap_or(None));
    Ok(())
}

pub(crate) fn solve_triangular_tangent(
    lhs: &Tensor,
    rhs: &Tensor,
    output: &Tensor,
    upper: bool,
) -> Result<()> {
    if !is_active() {
        return Ok(());
    }
    let linearized = SolveTriangularOp::new(upper)
        .linearize(&[lhs.primal(), rhs.primal()], &[output.primal().clone()])?;
    let mut outputs = linearized.jvp(&[tangent_for(lhs), tangent_for(rhs)])?;
    record_tangent(output, outputs.pop().unwrap_or(None));
    Ok(())
}

pub(crate) fn det_tangent(input: &Tensor, output: &Tensor) -> Result<()> {
    if !is_active() {
        return Ok(());
    }
    let linearized = DetOp.linearize(&[input.primal()], &[output.primal().clone()])?;
    let mut outputs = linearized.jvp(&[tangent_for(input)])?;
    record_tangent(output, outputs.pop().unwrap_or(None));
    Ok(())
}

pub(crate) fn inv_tangent(input: &Tensor, output: &Tensor) -> Result<()> {
    if !is_active() {
        return Ok(());
    }
    let linearized = InvOp.linearize(&[input.primal()], &[output.primal().clone()])?;
    let mut outputs = linearized.jvp(&[tangent_for(input)])?;
    record_tangent(output, outputs.pop().unwrap_or(None));
    Ok(())
}

pub(crate) fn slogdet_tangents(input: &Tensor, sign: &Tensor, logabsdet: &Tensor) -> Result<()> {
    if !is_active() {
        return Ok(());
    }
    let linearized = SlogdetOp.linearize(
        &[input.primal()],
        &[sign.primal().clone(), logabsdet.primal().clone()],
    )?;
    let mut outputs = linearized.jvp(&[tangent_for(input)])?.into_iter();
    record_tangent(sign, outputs.next().unwrap_or(None));
    record_tangent(logabsdet, outputs.next().unwrap_or(None));
    Ok(())
}

pub(crate) fn cholesky_tangent(input: &Tensor, output: &Tensor) -> Result<()> {
    if !is_active() {
        return Ok(());
    }
    let linearized = CholeskyOp.linearize(&[input.primal()], &[output.primal().clone()])?;
    let mut outputs = linearized.jvp(&[tangent_for(input)])?;
    record_tangent(output, outputs.pop().unwrap_or(None));
    Ok(())
}

pub(crate) fn lu_tangents(
    input: &Tensor,
    p: &Tensor,
    l: &Tensor,
    u: &Tensor,
    pivot: LuPivot,
) -> Result<()> {
    if !is_active() {
        return Ok(());
    }
    let linearized = LuOp::new(pivot).linearize(
        &[input.primal()],
        &[p.primal().clone(), l.primal().clone(), u.primal().clone()],
    )?;
    let mut outputs = linearized.jvp(&[tangent_for(input)])?.into_iter();
    record_tangent(p, outputs.next().unwrap_or(None));
    record_tangent(l, outputs.next().unwrap_or(None));
    record_tangent(u, outputs.next().unwrap_or(None));
    Ok(())
}

pub(crate) fn norm_tangent(input: &Tensor, output: &Tensor, kind: NormKind) -> Result<()> {
    if !is_active() {
        return Ok(());
    }
    let linearized = NormOp::new(kind).linearize(&[input.primal()], &[output.primal().clone()])?;
    let mut outputs = linearized.jvp(&[tangent_for(input)])?;
    record_tangent(output, outputs.pop().unwrap_or(None));
    Ok(())
}

pub(crate) fn eig_tangents(input: &Tensor, values: &Tensor, vectors: &Tensor) -> Result<()> {
    if !is_active() {
        return Ok(());
    }
    let linearized = EigOp.linearize(
        &[input.primal()],
        &[values.primal().clone(), vectors.primal().clone()],
    )?;
    let mut outputs = linearized.jvp(&[tangent_for(input)])?.into_iter();
    record_tangent(values, outputs.next().unwrap_or(None));
    record_tangent(vectors, outputs.next().unwrap_or(None));
    Ok(())
}

pub(crate) fn eigen_tangents(input: &Tensor, values: &Tensor, vectors: &Tensor) -> Result<()> {
    if !is_active() {
        return Ok(());
    }
    let linearized = EigenOp.linearize(
        &[input.primal()],
        &[values.primal().clone(), vectors.primal().clone()],
    )?;
    let mut outputs = linearized.jvp(&[tangent_for(input)])?.into_iter();
    record_tangent(values, outputs.next().unwrap_or(None));
    record_tangent(vectors, outputs.next().unwrap_or(None));
    Ok(())
}

pub(crate) fn svd_tangents(
    input: &Tensor,
    u: &Tensor,
    s: &Tensor,
    vt: &Tensor,
    options: Option<SvdOptions>,
) -> Result<()> {
    if !is_active() {
        return Ok(());
    }
    let linearized = SvdOp::new(options).linearize(
        &[input.primal()],
        &[u.primal().clone(), s.primal().clone(), vt.primal().clone()],
    )?;
    let mut outputs = linearized.jvp(&[tangent_for(input)])?.into_iter();
    record_tangent(u, outputs.next().unwrap_or(None));
    record_tangent(s, outputs.next().unwrap_or(None));
    record_tangent(vt, outputs.next().unwrap_or(None));
    Ok(())
}

pub(crate) fn pinv_tangent(input: &Tensor, output: &Tensor, rcond: Option<f64>) -> Result<()> {
    if !is_active() {
        return Ok(());
    }
    let linearized = PInvOp::new(rcond).linearize(&[input.primal()], &[output.primal().clone()])?;
    let mut outputs = linearized.jvp(&[tangent_for(input)])?;
    record_tangent(output, outputs.pop().unwrap_or(None));
    Ok(())
}

pub(crate) fn matrix_exp_tangent(input: &Tensor, output: &Tensor) -> Result<()> {
    if !is_active() {
        return Ok(());
    }
    let linearized = MatrixExpOp.linearize(&[input.primal()], &[output.primal().clone()])?;
    let mut outputs = linearized.jvp(&[tangent_for(input)])?;
    record_tangent(output, outputs.pop().unwrap_or(None));
    Ok(())
}

pub fn jvp<F>(f: F, primals: &[Tensor], tangents: &[Option<Tensor>]) -> Result<JvpResult>
where
    F: FnOnce(&[Tensor]) -> Result<Vec<Tensor>>,
{
    if primals.len() != tangents.len() {
        return Err(invalid_argument(format!(
            "jvp expected {} tangents for {} primals",
            primals.len(),
            tangents.len()
        )));
    }
    for (index, (primal, tangent)) in primals.iter().zip(tangents.iter()).enumerate() {
        if let Some(tangent) = tangent {
            validate_compatibility(index, primal, tangent)?;
        }
    }

    let mut ctx = ForwardJvpContext::default();
    for (primal, tangent) in primals.iter().zip(tangents.iter()) {
        if let Some(tangent) = tangent {
            ctx.set_tangent(primal.forward_id(), tangent.primal().clone());
        }
    }

    let _guard = push_context(ctx);
    let outputs = f(primals)?;
    let output_tangents = outputs
        .iter()
        .map(|output| tangent_for(output).map(Tensor::new))
        .collect();

    Ok(JvpResult {
        outputs,
        output_tangents,
    })
}

pub(crate) fn forward_id(tensor: &Tensor) -> usize {
    tensor.primal() as *const DynTensor as usize
}

#[cfg(test)]
mod tests {
    use super::forward_id;
    use crate::Tensor;

    fn round_trip(tensor: Tensor) -> (Tensor, usize) {
        let id = forward_id(&tensor);
        (tensor, id)
    }

    #[test]
    fn forward_id_is_stable_across_moves() {
        let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
        let before = forward_id(&x);
        let (x, moved_id) = round_trip(x);
        assert_eq!(before, moved_id);
        assert_eq!(before, forward_id(&x));
    }
}
