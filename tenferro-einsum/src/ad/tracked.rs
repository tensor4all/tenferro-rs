use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use tenferro_algebra::{Conjugate, HasAlgebra, Scalar, Semiring};
use tenferro_prims::TensorTempPoolContext;
use tenferro_tensor::Tensor;
use tidu::expert::TrackedValue;
use tidu::{AdResult, Differentiable};

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
/// use std::sync::{Arc, Mutex};
/// use tidu::expert::Tape;
/// use tenferro_algebra::Standard;
/// use tenferro_einsum::tracked_einsum;
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let ctx = Arc::new(Mutex::new(CpuContext::new(1)));
/// let tape = Tape::<tenferro_tensor::Tensor<f64>>::new();
/// let a = tape.leaf(a_tensor);
/// let b = tape.leaf(b_tensor);
/// let out = tracked_einsum::<Standard<f64>, CpuBackend>(ctx, "ij,jk->ik", &[&a, &b]).unwrap();
/// let _ = out.node_id();
/// ```
pub fn tracked_einsum<Alg: 'static, Backend>(
    ctx: Arc<Mutex<BackendContext<Alg, Backend>>>,
    subscripts: &str,
    operands: &[&TrackedValue<Tensor<Alg::Scalar>>],
) -> AdResult<TrackedValue<Tensor<Alg::Scalar>>>
where
    Alg: Semiring + Send + Sync,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg> + Send + Sync + 'static,
    Tensor<Alg::Scalar>: Differentiable<Tangent = Tensor<Alg::Scalar>>,
    BackendContext<Alg, Backend>: Send,
    BackendContext<Alg, Backend>: TensorTempPoolContext,
{
    let subs = Subscripts::parse(subscripts)
        .map_err(|e| tidu::AutodiffError::InvalidArgument(format!("{e}")))?;

    let primals: Vec<&Tensor<Alg::Scalar>> = operands.iter().map(|op| op.value()).collect();
    let output = einsum::<Alg, Backend>(
        &mut *ctx.lock().map_err(|_| {
            tidu::AutodiffError::InvalidArgument(
                "tracked_einsum backend context lock is poisoned".to_string(),
            )
        })?,
        subscripts,
        &primals,
        None,
    )
    .map_err(|e| tidu::AutodiffError::InvalidArgument(format!("{e}")))?;

    let tangents: Vec<Option<&Tensor<Alg::Scalar>>> =
        operands.iter().map(|op| op.tangent()).collect();
    let tangent_out = if tangents.iter().any(Option::is_some) {
        Some(
            einsum_frule_impl::<Alg, Backend>(
                &mut *ctx.lock().map_err(|_| {
                    tidu::AutodiffError::InvalidArgument(
                        "tracked_einsum backend context lock is poisoned".to_string(),
                    )
                })?,
                &subs,
                None,
                &primals,
                &tangents,
            )
            .map_err(|e| tidu::AutodiffError::InvalidArgument(format!("{e}")))?,
        )
    } else {
        None
    };

    let any_requires_grad = operands.iter().any(|op| op.requires_grad());
    if !any_requires_grad {
        return Ok(TrackedValue::new(output));
    }

    let tape = operands
        .iter()
        .filter(|op| op.requires_grad())
        .find_map(|op| op.tape())
        .ok_or(tidu::AutodiffError::MissingNode)?
        .clone();

    for op in operands.iter().filter(|op| op.requires_grad()) {
        if let Some(op_tape) = op.tape() {
            if !tape.same_tape(op_tape) {
                return Err(tidu::AutodiffError::InvalidArgument(
                    "tracked_einsum: operands belong to different AD tapes".into(),
                ));
            }
        }
    }

    let rule = EinsumReverseRule::<Alg, Backend> {
        ctx: ctx.clone(),
        subscripts: subs,
        primals: primals.iter().map(|&t| t.clone()).collect(),
        input_node_ids: operands.iter().map(|op| op.node_id()).collect(),
        _phantom: PhantomData,
    };

    Ok(tape.record_op(output, Box::new(rule), tangent_out))
}
