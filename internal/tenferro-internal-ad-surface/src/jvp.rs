use std::cell::RefCell;
use std::collections::HashMap;

use tenferro_internal_ad_core::{LinearizableOp, LinearizedOp};
use tenferro_internal_ad_linalg::QrOp;
use tenferro_internal_ad_ops::{AddOp, ExpOp, SumOp};
use tenferro_internal_frontend_core::DynTensor;

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
    static FORWARD_JVP_STACK: RefCell<Vec<ForwardJvpContext>> = RefCell::new(Vec::new());
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

fn unsupported_jvp(op: &'static str) -> Error {
    Error::UnsupportedAdOp { op }
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

pub(crate) fn unsupported_if_active(op: &'static str) -> Result<()> {
    if is_active() {
        Err(unsupported_jvp(op))
    } else {
        Ok(())
    }
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
