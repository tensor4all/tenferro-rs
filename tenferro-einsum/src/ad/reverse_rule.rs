use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use chainrules::{AdResult, Differentiable, NodeId, ReverseRule};
use tenferro_algebra::{HasAlgebra, Scalar, Semiring};
use tenferro_tensor::Tensor;

use crate::api::einsum_with_subscripts;
use crate::backend::{BackendContext, EinsumBackend};
use crate::subscripts::Subscripts;

use super::rules::einsum_frule_impl;

/// ReverseRule for einsum — stores subscripts, primal tensors, and shared
/// backend context for backend-optimized pullback.
pub(super) struct EinsumReverseRule<Alg, Backend>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
    Tensor<Alg::Scalar>: Differentiable<Tangent = Tensor<Alg::Scalar>>,
{
    pub(super) ctx: Rc<RefCell<BackendContext<Alg, Backend>>>,
    pub(super) subscripts: Subscripts,
    pub(super) primals: Vec<Tensor<Alg::Scalar>>,
    pub(super) input_tangents: Vec<Option<Tensor<Alg::Scalar>>>,
    pub(super) input_node_ids: Vec<Option<NodeId>>,
    pub(super) _phantom: PhantomData<Alg>,
}

impl<Alg, Backend> ReverseRule<Tensor<Alg::Scalar>> for EinsumReverseRule<Alg, Backend>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
    Tensor<Alg::Scalar>: Differentiable<Tangent = Tensor<Alg::Scalar>>,
{
    fn pullback(
        &self,
        cotangent: &Tensor<Alg::Scalar>,
    ) -> AdResult<Vec<(NodeId, Tensor<Alg::Scalar>)>> {
        let n = self.primals.len();
        let mut results = Vec::new();
        let mut ctx = self.ctx.borrow_mut();

        for k in 0..n {
            let node_id = match self.input_node_ids[k] {
                Some(id) => id,
                None => continue,
            };

            let mut rev_inputs_subs = vec![self.subscripts.output.clone()];
            let mut rev_operands: Vec<&Tensor<Alg::Scalar>> = vec![cotangent];

            for (i, primal) in self.primals.iter().enumerate() {
                if i != k {
                    rev_inputs_subs.push(self.subscripts.inputs[i].clone());
                    rev_operands.push(primal);
                }
            }

            let rev_subs = Subscripts {
                inputs: rev_inputs_subs,
                output: self.subscripts.inputs[k].clone(),
            };

            let mut grad =
                einsum_with_subscripts::<Alg, Backend>(&mut *ctx, &rev_subs, &rev_operands, None)
                    .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?;

            if rev_operands.iter().any(|t| t.has_fw_grad()) {
                let tangents: Vec<Option<&Tensor<Alg::Scalar>>> =
                    rev_operands.iter().map(|t| t.fw_grad()).collect();
                if let Ok(grad_tangent) = einsum_frule_impl::<Alg, Backend>(
                    &mut *ctx,
                    &rev_subs,
                    None,
                    &rev_operands,
                    &tangents,
                ) {
                    grad.set_fw_grad(grad_tangent);
                }
            }

            results.push((node_id, grad));
        }

        Ok(results)
    }

    fn inputs(&self) -> Vec<NodeId> {
        self.input_node_ids.iter().filter_map(|id| *id).collect()
    }

    fn pullback_with_tangents(
        &self,
        cotangent: &Tensor<Alg::Scalar>,
        cotangent_tangent: &Tensor<Alg::Scalar>,
    ) -> AdResult<Vec<(NodeId, Tensor<Alg::Scalar>, Tensor<Alg::Scalar>)>> {
        let n = self.primals.len();
        let mut results = Vec::new();
        let mut ctx = self.ctx.borrow_mut();

        for k in 0..n {
            let node_id = match self.input_node_ids[k] {
                Some(id) => id,
                None => continue,
            };

            let mut rev_inputs_subs = vec![self.subscripts.output.clone()];
            let mut rev_operands: Vec<&Tensor<Alg::Scalar>> = vec![cotangent];
            let mut rev_tangents: Vec<Option<&Tensor<Alg::Scalar>>> = vec![Some(cotangent_tangent)];

            for (i, primal) in self.primals.iter().enumerate() {
                if i != k {
                    rev_inputs_subs.push(self.subscripts.inputs[i].clone());
                    rev_operands.push(primal);
                    rev_tangents.push(self.input_tangents[i].as_ref());
                }
            }

            let rev_subs = Subscripts {
                inputs: rev_inputs_subs,
                output: self.subscripts.inputs[k].clone(),
            };

            let grad =
                einsum_with_subscripts::<Alg, Backend>(&mut *ctx, &rev_subs, &rev_operands, None)
                    .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?;
            let grad_tangent = einsum_frule_impl::<Alg, Backend>(
                &mut *ctx,
                &rev_subs,
                None,
                &rev_operands,
                &rev_tangents,
            )
            .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?;
            results.push((node_id, grad, grad_tangent));
        }

        Ok(results)
    }
}
