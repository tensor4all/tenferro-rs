use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use chainrules::{autograd, AdResult, Differentiable, TrackedTensor, Variable};
use tenferro_algebra::{HasAlgebra, Scalar, Semiring};
use tenferro_tensor::Tensor;

use crate::api::einsum;
use crate::execution::backend::{BackendContext, EinsumBackend};
use crate::syntax::subscripts::Subscripts;

use super::reverse_rule::EinsumReverseRule;
use super::rules::einsum_frule_impl;

/// Tracked einsum (reverse-mode AD).
///
/// # Examples
///
/// ```ignore
/// use std::cell::RefCell;
/// use std::rc::Rc;
/// use chainrules::Tape;
/// use tenferro_algebra::Standard;
/// use tenferro_einsum::tracked_einsum;
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let ctx = Rc::new(RefCell::new(CpuContext::new(1)));
/// let tape = Tape::<tenferro_tensor::Tensor<f64>>::new();
/// let a = tape.leaf(a_tensor);
/// let b = tape.leaf(b_tensor);
/// let out = tracked_einsum::<Standard<f64>, CpuBackend>(ctx, "ij,jk->ik", &[&a, &b]).unwrap();
/// let _ = out.node_id();
/// ```
pub fn tracked_einsum<Alg: 'static, Backend>(
    ctx: Rc<RefCell<BackendContext<Alg, Backend>>>,
    subscripts: &str,
    operands: &[&TrackedTensor<Tensor<Alg::Scalar>>],
) -> AdResult<TrackedTensor<Tensor<Alg::Scalar>>>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg> + 'static,
    Tensor<Alg::Scalar>: Differentiable<Tangent = Tensor<Alg::Scalar>>,
{
    let subs = Subscripts::parse(subscripts)
        .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?;

    let primals: Vec<&Tensor<Alg::Scalar>> = operands.iter().map(|op| op.value()).collect();
    let output = einsum::<Alg, Backend>(&mut *ctx.borrow_mut(), subscripts, &primals, None)
        .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?;

    let tangents: Vec<Option<&Tensor<Alg::Scalar>>> =
        operands.iter().map(|op| op.tangent()).collect();
    let tangent_out = if tangents.iter().any(Option::is_some) {
        Some(
            einsum_frule_impl::<Alg, Backend>(
                &mut *ctx.borrow_mut(),
                &subs,
                None,
                &primals,
                &tangents,
            )
            .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?,
        )
    } else {
        None
    };

    let any_requires_grad = operands.iter().any(|op| op.requires_grad());
    if !any_requires_grad {
        return Ok(TrackedTensor::new(output));
    }

    let tape = operands
        .iter()
        .filter(|op| op.requires_grad())
        .find_map(|op| op.tape())
        .ok_or(chainrules::AutodiffError::MissingNode)?
        .clone();

    for op in operands.iter().filter(|op| op.requires_grad()) {
        if let Some(op_tape) = op.tape() {
            if !tape.same_tape(op_tape) {
                return Err(chainrules::AutodiffError::InvalidArgument(
                    "tracked_einsum: operands belong to different AD tapes".into(),
                ));
            }
        }
    }

    let rule = EinsumReverseRule::<Alg, Backend> {
        ctx: ctx.clone(),
        subscripts: subs,
        primals: primals.iter().map(|&t| t.clone()).collect(),
        input_tangents: operands.iter().map(|op| op.tangent().cloned()).collect(),
        input_node_ids: operands.iter().map(|op| op.node_id()).collect(),
        _phantom: PhantomData,
    };

    Ok(tape.record_op(output, Box::new(rule), tangent_out))
}

/// Variable einsum (monomorphic reverse/forward-mode via `Variable` API).
///
/// # Examples
///
/// ```ignore
/// use std::cell::RefCell;
/// use std::rc::Rc;
/// use chainrules::Variable;
/// use tenferro_algebra::Standard;
/// use tenferro_einsum::variable_einsum;
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let ctx = Rc::new(RefCell::new(CpuContext::new(1)));
/// let out = variable_einsum::<Standard<f64>, CpuBackend>(ctx, "ij,jk->ik", &[&a, &b]).unwrap();
/// let _ = out.value();
/// ```
pub fn variable_einsum<Alg: 'static, Backend>(
    ctx: Rc<RefCell<BackendContext<Alg, Backend>>>,
    subscripts: &str,
    operands: &[&Variable<Tensor<Alg::Scalar>>],
) -> AdResult<Variable<Tensor<Alg::Scalar>>>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg> + 'static,
    Tensor<Alg::Scalar>: Differentiable<Tangent = Tensor<Alg::Scalar>> + 'static,
{
    let subs = Subscripts::parse(subscripts)
        .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?;

    let primals: Vec<&Tensor<Alg::Scalar>> = operands.iter().map(|op| op.value()).collect();
    let output = einsum::<Alg, Backend>(&mut *ctx.borrow_mut(), subscripts, &primals, None)
        .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?;

    let tangents: Vec<Option<&Tensor<Alg::Scalar>>> =
        operands.iter().map(|op| op.tangent()).collect();
    let tangent_out = if tangents.iter().any(Option::is_some) {
        Some(
            einsum_frule_impl::<Alg, Backend>(
                &mut *ctx.borrow_mut(),
                &subs,
                None,
                &primals,
                &tangents,
            )
            .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?,
        )
    } else {
        None
    };

    let rule = EinsumReverseRule::<Alg, Backend> {
        ctx: ctx.clone(),
        subscripts: subs,
        primals: primals.iter().map(|&t| t.clone()).collect(),
        input_tangents: operands.iter().map(|op| op.tangent().cloned()).collect(),
        input_node_ids: operands.iter().map(|op| op.node_id()).collect(),
        _phantom: PhantomData,
    };

    autograd::record_op(output, operands, Box::new(rule), tangent_out)
}
