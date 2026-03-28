use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use tenferro_algebra::{Conjugate, HasAlgebra, Scalar, Semiring};
use tenferro_prims::TensorTempPoolContext;
use tenferro_tensor::Tensor;
use tidu::{AdResult, Differentiable, NodeId, ReverseRule};

use crate::api::einsum_with_subscripts;
use crate::execution::backend::{BackendContext, EinsumBackend};
use crate::syntax::subscripts::Subscripts;

use super::rules::einsum_frule_impl;

/// ReverseRule for einsum — stores subscripts, primal tensors, and shared
/// backend context for backend-optimized pullback.
pub(super) struct EinsumReverseRule<Alg, Backend>
where
    Alg: Semiring + Send + Sync,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg> + Send + Sync,
    Tensor<Alg::Scalar>: Differentiable<Tangent = Tensor<Alg::Scalar>>,
    BackendContext<Alg, Backend>: Send,
    BackendContext<Alg, Backend>: TensorTempPoolContext,
{
    pub(super) ctx: Arc<Mutex<BackendContext<Alg, Backend>>>,
    pub(super) subscripts: Subscripts,
    pub(super) primals: Vec<Tensor<Alg::Scalar>>,
    pub(super) input_node_ids: Vec<Option<NodeId>>,
    pub(super) _phantom: PhantomData<Alg>,
}

impl<Alg, Backend> ReverseRule<Tensor<Alg::Scalar>> for EinsumReverseRule<Alg, Backend>
where
    Alg: Semiring + Send + Sync,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg> + Send + Sync,
    Tensor<Alg::Scalar>: Differentiable<Tangent = Tensor<Alg::Scalar>>,
    BackendContext<Alg, Backend>: Send,
    BackendContext<Alg, Backend>: TensorTempPoolContext,
{
    fn pullback(
        &self,
        cotangent: &Tensor<Alg::Scalar>,
    ) -> AdResult<Vec<(NodeId, Tensor<Alg::Scalar>)>> {
        let n = self.primals.len();
        let mut results = Vec::new();
        let mut ctx = self.ctx.lock().map_err(|_| {
            tidu::AutodiffError::InvalidArgument(
                "einsum reverse-rule backend context lock is poisoned".to_string(),
            )
        })?;

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

            let grad =
                einsum_with_subscripts::<Alg, Backend>(&mut *ctx, &rev_subs, &rev_operands, None)
                    .map_err(|e| tidu::AutodiffError::InvalidArgument(format!("{e}")))?;

            results.push((node_id, grad));
        }

        Ok(results)
    }

    fn inputs(&self) -> Vec<NodeId> {
        self.input_node_ids.iter().filter_map(|id| *id).collect()
    }

    fn forward_tangents<'t>(
        &self,
        input_tangents: &dyn Fn(NodeId) -> Option<&'t Tensor<Alg::Scalar>>,
    ) -> AdResult<Option<Tensor<Alg::Scalar>>>
    where
        Tensor<Alg::Scalar>: 't,
    {
        let tangents: Vec<Option<&Tensor<Alg::Scalar>>> = self
            .input_node_ids
            .iter()
            .map(|maybe_id| maybe_id.and_then(|id| input_tangents(id)))
            .collect();
        if tangents.iter().all(|t| t.is_none()) {
            return Ok(None);
        }
        let primals: Vec<&Tensor<Alg::Scalar>> = self.primals.iter().collect();
        let mut ctx = self.ctx.lock().map_err(|_| {
            tidu::AutodiffError::InvalidArgument(
                "einsum reverse-rule backend context lock is poisoned".to_string(),
            )
        })?;
        Ok(Some(
            einsum_frule_impl::<Alg, Backend>(
                &mut *ctx,
                &self.subscripts,
                None,
                &primals,
                &tangents,
            )
            .map_err(|e| tidu::AutodiffError::InvalidArgument(format!("{e}")))?,
        ))
    }

    fn pullback_with_tangents<'t>(
        &self,
        cotangent: &Tensor<Alg::Scalar>,
        cotangent_tangent: &Tensor<Alg::Scalar>,
        input_tangents: &dyn Fn(NodeId) -> Option<&'t Tensor<Alg::Scalar>>,
    ) -> AdResult<Vec<(NodeId, Tensor<Alg::Scalar>, Tensor<Alg::Scalar>)>>
    where
        Tensor<Alg::Scalar>: 't,
    {
        let n = self.primals.len();
        let mut results = Vec::new();
        let mut ctx = self.ctx.lock().map_err(|_| {
            tidu::AutodiffError::InvalidArgument(
                "einsum reverse-rule backend context lock is poisoned".to_string(),
            )
        })?;

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
                    rev_tangents.push(self.input_node_ids[i].and_then(|id| input_tangents(id)));
                }
            }

            let rev_subs = Subscripts {
                inputs: rev_inputs_subs,
                output: self.subscripts.inputs[k].clone(),
            };

            let grad =
                einsum_with_subscripts::<Alg, Backend>(&mut *ctx, &rev_subs, &rev_operands, None)
                    .map_err(|e| tidu::AutodiffError::InvalidArgument(format!("{e}")))?;
            let grad_tangent = einsum_frule_impl::<Alg, Backend>(
                &mut *ctx,
                &rev_subs,
                None,
                &rev_operands,
                &rev_tangents,
            )
            .map_err(|e| tidu::AutodiffError::InvalidArgument(format!("{e}")))?;
            results.push((node_id, grad, grad_tangent));
        }

        Ok(results)
    }
}
