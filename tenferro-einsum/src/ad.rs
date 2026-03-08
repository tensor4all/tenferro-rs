use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use chainrules::{
    autograd, AdResult, Differentiable, DualTensor, NodeId, ReverseRule, TrackedTensor, Variable,
};
use tenferro_algebra::{Algebra, HasAlgebra, Scalar};
use tenferro_device::{Error, Result};
use tenferro_prims::TensorPrims;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::api::{einsum, einsum_with_subscripts};
use crate::execute::execute_nested;
use crate::nested::NestedEinsum;
use crate::subscripts::Subscripts;

/// ReverseRule for einsum — stores subscripts, primal tensors, and shared
/// backend context for backend-optimized pullback.
struct EinsumReverseRule<Alg, Backend>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
    Tensor<Alg::Scalar>: Differentiable<Tangent = Tensor<Alg::Scalar>>,
{
    ctx: Rc<RefCell<Backend::Context>>,
    subscripts: Subscripts,
    primals: Vec<Tensor<Alg::Scalar>>,
    input_tangents: Vec<Option<Tensor<Alg::Scalar>>>,
    input_node_ids: Vec<Option<NodeId>>,
    _phantom: PhantomData<Alg>,
}

impl<Alg, Backend> ReverseRule<Tensor<Alg::Scalar>> for EinsumReverseRule<Alg, Backend>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
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

            // Build reverse einsum subscripts
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

            // Use backend-optimized einsum
            let mut grad =
                einsum_with_subscripts::<Alg, Backend>(&mut *ctx, &rev_subs, &rev_operands, None)
                    .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?;

            // Propagate fw_grad from operands through the reverse einsum
            if rev_operands.iter().any(|t| t.has_fw_grad()) {
                let tangents: Vec<Option<&Tensor<Alg::Scalar>>> =
                    rev_operands.iter().map(|t| t.fw_grad()).collect();
                if let Ok(grad_tangent) = einsum_frule_impl::<Alg, Backend>(
                    &mut *ctx,
                    &rev_subs,
                    None, // reverse subscripts are flat by construction
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

/// Tracked einsum (reverse-mode AD).
///
/// This is the AD-aware counterpart of [`einsum`]. It records the operation
/// on the reverse-mode tape so that [`chainrules::Tape::pullback`] can
/// compute gradients through it.
///
/// The context is wrapped in `Rc<RefCell<>>` so the pullback rule can
/// reuse the same backend context for computing gradients.
///
/// # Examples
///
/// ```ignore
/// use std::cell::RefCell;
/// use std::rc::Rc;
/// use chainrules::Tape;
/// use tenferro_einsum::tracked_einsum;
/// use tenferro_tensor::{MemoryOrder, Tensor};
/// use tenferro_device::LogicalMemorySpace;
///
/// let ctx = Rc::new(RefCell::new(CpuContext::new(1)));
/// let tape = Tape::<Tensor<f64>>::new();
/// let a = tape.leaf(Tensor::ones(
///     &[2, 3],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// ));
/// let b = tape.leaf(Tensor::ones(
///     &[3, 4],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// ));
/// let c = tracked_einsum::<_, _, CpuBackend>(ctx.clone(), "ij,jk->ik", &[&a, &b]).unwrap();
/// let loss = tracked_einsum::<_, _, CpuBackend>(ctx.clone(), "ij,ij->", &[&c, &c]).unwrap();
/// let grads = tape.pullback(&loss).unwrap();
/// let _ga = grads.get(a.node_id().unwrap()).unwrap();
/// ```
///
pub fn tracked_einsum<Alg: 'static, Backend>(
    ctx: Rc<RefCell<Backend::Context>>,
    subscripts: &str,
    operands: &[&TrackedTensor<Tensor<Alg::Scalar>>],
) -> AdResult<TrackedTensor<Tensor<Alg::Scalar>>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg> + 'static,
    Tensor<Alg::Scalar>: Differentiable<Tangent = Tensor<Alg::Scalar>>,
{
    let subs = Subscripts::parse(subscripts)
        .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?;

    // Extract primals and run forward einsum
    let primals: Vec<&Tensor<Alg::Scalar>> = operands.iter().map(|op| op.value()).collect();
    let output = einsum::<Alg, Backend>(&mut *ctx.borrow_mut(), subscripts, &primals, None)
        .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?;

    // Check if any operand requires gradients
    let any_requires_grad = operands.iter().any(|op| op.requires_grad());

    if !any_requires_grad {
        return Ok(TrackedTensor::new(output));
    }

    // Find tape from the first tracked operand that has one
    let tape = operands
        .iter()
        .filter(|op| op.requires_grad())
        .find_map(|op| op.tape())
        .ok_or(chainrules::AutodiffError::MissingNode)?
        .clone();

    // Reject mixed-tape operands: all grad-tracked tensors must share the same tape
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
        input_tangents: operands
            .iter()
            .map(|op| op.value().fw_grad().cloned())
            .collect(),
        input_node_ids: operands.iter().map(|op| op.node_id()).collect(),
        _phantom: PhantomData,
    };

    // Record the operation on the tape so pullback can compute gradients
    let result = tape.record_op(output, Box::new(rule), None);

    Ok(result)
}

/// Variable einsum (monomorphic reverse/forward-mode via `Variable` API).
///
/// This is the `Variable`-based counterpart of [`tracked_einsum`]. It records
/// a custom reverse rule into the `AutogradContext` shared by tracked inputs.
///
/// # Examples
///
/// ```ignore
/// use std::cell::RefCell;
/// use std::rc::Rc;
/// use std::sync::Arc;
/// use chainrules::{autograd, AutogradContext, BackwardOptions, Variable};
/// use tenferro_einsum::variable_einsum;
/// use tenferro_prims::{CpuBackend, CpuContext};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let ctx = Rc::new(RefCell::new(CpuContext::new(1)));
/// let ad_ctx = AutogradContext::<Tensor<f64>>::new();
/// let a = Variable::new_in(
///     Tensor::ones(&[2, 3], tenferro_device::LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor),
///     Arc::clone(&ad_ctx),
/// ).requires_grad_(true).unwrap();
/// let b = Variable::new_in(
///     Tensor::ones(&[3, 4], tenferro_device::LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor),
///     Arc::clone(&ad_ctx),
/// ).requires_grad_(true).unwrap();
/// let c = variable_einsum::<_, CpuBackend>(ctx.clone(), "ij,jk->ik", &[&a, &b]).unwrap();
/// let loss = variable_einsum::<_, CpuBackend>(ctx.clone(), "ij,ij->", &[&c, &c]).unwrap();
/// loss.backward(BackwardOptions::default()).unwrap();
/// ```
pub fn variable_einsum<Alg: 'static, Backend>(
    ctx: Rc<RefCell<Backend::Context>>,
    subscripts: &str,
    operands: &[&Variable<Tensor<Alg::Scalar>>],
) -> AdResult<Variable<Tensor<Alg::Scalar>>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg> + 'static,
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

/// Dual einsum (forward-mode JVP propagation).
///
/// This is the AD-aware counterpart of [`einsum`] for forward-mode.
/// It propagates tangent vectors through the einsum operation.
///
/// # Examples
///
/// ```ignore
/// use chainrules::DualTensor;
/// use tenferro_einsum::dual_einsum;
/// use tenferro_tensor::{MemoryOrder, Tensor};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::ones(&[2, 3], mem, col);
/// let da = Tensor::<f64>::ones(&[2, 3], mem, col);
/// let b = Tensor::<f64>::ones(&[3, 4], mem, col);
///
/// let a_dual = DualTensor::with_tangent(a, da).unwrap();
/// let b_dual = DualTensor::new(b);
/// let c_dual = dual_einsum::<_, _, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a_dual, &b_dual]).unwrap();
/// let _tangent = c_dual.tangent();
/// ```
pub fn dual_einsum<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &str,
    operands: &[&DualTensor<Tensor<Alg::Scalar>>],
) -> AdResult<DualTensor<Tensor<Alg::Scalar>>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
    Tensor<Alg::Scalar>: Differentiable<Tangent = Tensor<Alg::Scalar>>,
{
    // Extract primals
    let primals: Vec<&Tensor<Alg::Scalar>> = operands.iter().map(|op| op.primal()).collect();

    // Compute primal output
    let output = einsum::<Alg, Backend>(ctx, subscripts, &primals, None)
        .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?;

    // Compute tangent: dC = sum_k einsum(subs, [A0, ..., dAk, ..., An])
    let tangents: Vec<Option<&Tensor<Alg::Scalar>>> =
        operands.iter().map(|op| op.tangent()).collect();

    // If no operand carries a tangent, skip frule and return primal only.
    if tangents.iter().all(|t| t.is_none()) {
        return Ok(DualTensor::new(output));
    }

    let tangent = einsum_frule::<Alg, Backend>(ctx, subscripts, &primals, &tangents)
        .map_err(|e| chainrules::AutodiffError::InvalidArgument(format!("{e}")))?;
    DualTensor::with_tangent(output, tangent)
}

/// Reverse-mode rule (rrule) for einsum without building a global tape.
///
/// Computes the pullback (vector-Jacobian product) for an einsum operation.
/// Returns one gradient tensor per input operand.
///
/// Named after Julia's ChainRules.jl convention.
/// This API is intended for language interop and manual AD.
///
/// # Examples
///
/// ```ignore
/// use tenferro_einsum::einsum_rrule;
/// use tenferro_tensor::{MemoryOrder, Tensor};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::ones(&[2, 3], mem, col);
/// let b = Tensor::<f64>::ones(&[3, 4], mem, col);
/// let grad_c = Tensor::<f64>::ones(&[2, 4], mem, col);
///
/// let grads = einsum_rrule::<_, _, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], &grad_c).unwrap();
/// assert_eq!(grads.len(), 2);
/// ```
pub fn einsum_rrule<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &str,
    operands: &[&Tensor<Alg::Scalar>],
    cotangent: &Tensor<Alg::Scalar>,
) -> Result<Vec<Tensor<Alg::Scalar>>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let subs = Subscripts::parse(subscripts)?;
    let n = operands.len();
    let mut grads = Vec::with_capacity(n);

    for k in 0..n {
        // Build reverse subscripts for operand k:
        // grad_Ak = einsum([cotangent, A_0, ..., A_{k-1}, A_{k+1}, ..., A_n])
        let mut rev_inputs_subs = vec![subs.output.clone()];
        let mut rev_operands: Vec<&Tensor<Alg::Scalar>> = vec![cotangent];

        for (i, &op) in operands.iter().enumerate() {
            if i != k {
                rev_inputs_subs.push(subs.inputs[i].clone());
                rev_operands.push(op);
            }
        }

        let rev_subs = Subscripts {
            inputs: rev_inputs_subs,
            output: subs.inputs[k].clone(),
        };

        let grad = einsum_with_subscripts::<Alg, Backend>(ctx, &rev_subs, &rev_operands, None)?;
        grads.push(grad);
    }

    Ok(grads)
}

/// Forward-mode rule (frule) for einsum without building a global tape.
///
/// Computes the pushforward (Jacobian-vector product) for an einsum operation.
/// Inputs without tangent should use `None`.
///
/// Named after Julia's ChainRules.jl convention.
/// This API is intended for language interop and manual AD.
///
/// # Examples
///
/// ```ignore
/// use tenferro_einsum::einsum_frule;
/// use tenferro_tensor::{MemoryOrder, Tensor};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::ones(&[2, 3], mem, col);
/// let b = Tensor::<f64>::ones(&[3, 4], mem, col);
/// let da = Tensor::<f64>::ones(&[2, 3], mem, col);
///
/// let dc = einsum_frule::<_, _, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], &[Some(&da), None]).unwrap();
/// ```
pub fn einsum_frule<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &str,
    primals: &[&Tensor<Alg::Scalar>],
    tangents: &[Option<&Tensor<Alg::Scalar>>],
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let subs = Subscripts::parse(subscripts)?;
    let nested = if subscripts.contains('(') {
        Some(NestedEinsum::parse(subscripts)?)
    } else {
        None
    };
    einsum_frule_impl::<Alg, Backend>(ctx, &subs, nested.as_ref(), primals, tangents)
}

/// Internal frule implementation with pre-parsed subscripts.
///
/// When `nested` is `Some`, each frule term is computed via the nested tree
/// (respecting parenthesized contraction order). Otherwise the flat
/// `Subscripts` path is used.
pub(crate) fn einsum_frule_impl<Alg, Backend>(
    ctx: &mut Backend::Context,
    subs: &Subscripts,
    nested: Option<&NestedEinsum>,
    primals: &[&Tensor<Alg::Scalar>],
    tangents: &[Option<&Tensor<Alg::Scalar>>],
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let n = primals.len();

    // dC = sum_k einsum(subs, [A0, ..., dAk, ..., An]) for each k with tangent
    let mut result: Option<Tensor<Alg::Scalar>> = None;

    for k in 0..n {
        if let Some(tangent_k) = tangents[k] {
            let mut ops: Vec<&Tensor<Alg::Scalar>> = primals.to_vec();
            ops[k] = tangent_k;

            let term = if let Some(nested) = nested {
                execute_nested::<Alg, Backend>(ctx, nested, &ops, None)?
            } else {
                einsum_with_subscripts::<Alg, Backend>(ctx, subs, &ops, None)?
            };

            result = Some(match result {
                None => term,
                Some(existing) => Tensor::<Alg::Scalar>::accumulate_tangent(existing, &term),
            });
        }
    }

    match result {
        Some(r) => Ok(r),
        None => {
            // No tangents provided — return a zero tensor with the correct output shape.
            let primal_out = if let Some(nested) = nested {
                execute_nested::<Alg, Backend>(ctx, nested, primals, None)?
            } else {
                einsum_with_subscripts::<Alg, Backend>(ctx, subs, primals, None)?
            };
            Ok(Tensor::<Alg::Scalar>::zeros(
                primal_out.dims(),
                primal_out.logical_memory_space(),
                MemoryOrder::ColumnMajor,
            ))
        }
    }
}

/// Local HVP rule for einsum without building a global tape.
///
/// Computes the forward-over-reverse Hessian-vector product for an einsum
/// operation. Given primals, their tangents (direction v), an output
/// cotangent g, and its tangent dg, returns `(gradient, hvp)` pairs
/// for each input operand.
///
/// For C = einsum(subscripts, [A, B]):
/// - gradient: standard pullback (e.g., g_A = einsum(g_C, B))
/// - hvp: tangent of pullback (e.g., dg_A = einsum(dg_C, B) + einsum(g_C, dB))
///
/// This API is intended for language interop and manual AD.
///
/// # Examples
///
/// ```ignore
/// use tenferro_einsum::einsum_hvp;
/// use tenferro_tensor::{MemoryOrder, Tensor};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let a = Tensor::<f64>::ones(&[2, 3], mem, col);
/// let b = Tensor::<f64>::ones(&[3, 4], mem, col);
/// let da = Tensor::<f64>::ones(&[2, 3], mem, col);
///
/// let grad_c = Tensor::<f64>::ones(&[2, 4], mem, col);
/// let dgrad_c = Tensor::<f64>::ones(&[2, 4], mem, col);
///
/// let results = einsum_hvp::<_, _, CpuBackend>(
///     &mut ctx,
///     "ij,jk->ik",
///     &[&a, &b],
///     &[Some(&da), None],
///     &grad_c,
///     &dgrad_c,
/// ).unwrap();
/// assert_eq!(results.len(), 2);
/// let (_grad_a, _hvp_a) = &results[0];
/// let (_grad_b, _hvp_b) = &results[1];
/// ```
pub fn einsum_hvp<Alg, Backend>(
    ctx: &mut Backend::Context,
    subscripts: &str,
    primals: &[&Tensor<Alg::Scalar>],
    tangents: &[Option<&Tensor<Alg::Scalar>>],
    cotangent: &Tensor<Alg::Scalar>,
    cotangent_tangent: &Tensor<Alg::Scalar>,
) -> Result<Vec<(Tensor<Alg::Scalar>, Tensor<Alg::Scalar>)>>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + HasAlgebra<Algebra = Alg>,
    Backend: TensorPrims<Alg>,
{
    let subs = Subscripts::parse(subscripts)?;
    let n = primals.len();
    let mut results = Vec::with_capacity(n);

    for k in 0..n {
        // gradient_k = einsum([cotangent, A_0, ..., A_{k-1}, A_{k+1}, ..., An])
        // hvp_k = d/dv (gradient_k) = sum over sources of tangent:
        //   - from cotangent_tangent: einsum([dg, A_others...])
        //   - from each tangent_j (j != k): einsum([g, A_0, ..., dA_j, ..., An])

        // Build reverse subscripts for operand k
        let mut rev_inputs_subs = vec![subs.output.clone()];
        for (i, _) in primals.iter().enumerate() {
            if i != k {
                rev_inputs_subs.push(subs.inputs[i].clone());
            }
        }
        let rev_subs = Subscripts {
            inputs: rev_inputs_subs,
            output: subs.inputs[k].clone(),
        };

        // Compute gradient_k
        let mut rev_operands: Vec<&Tensor<Alg::Scalar>> = vec![cotangent];
        for (i, &op) in primals.iter().enumerate() {
            if i != k {
                rev_operands.push(op);
            }
        }
        let grad_k = einsum_with_subscripts::<Alg, Backend>(ctx, &rev_subs, &rev_operands, None)?;

        // Compute hvp_k: differentiate the gradient w.r.t. v
        // hvp_k = einsum([dg, A_others...]) + sum_{j!=k} einsum([g, ..., dA_j, ...])
        let mut hvp_k: Option<Tensor<Alg::Scalar>>;

        // Term from cotangent_tangent
        let mut ops: Vec<&Tensor<Alg::Scalar>> = vec![cotangent_tangent];
        for (i, &op) in primals.iter().enumerate() {
            if i != k {
                ops.push(op);
            }
        }
        let term = einsum_with_subscripts::<Alg, Backend>(ctx, &rev_subs, &ops, None)?;
        hvp_k = Some(term);

        // Terms from tangents of other primals
        for (j, tangent_j_opt) in tangents.iter().enumerate().take(n) {
            if j == k {
                continue;
            }
            if let Some(tangent_j) = *tangent_j_opt {
                let mut ops: Vec<&Tensor<Alg::Scalar>> = vec![cotangent];
                for (i, &op) in primals.iter().enumerate() {
                    if i != k {
                        if i == j {
                            ops.push(tangent_j);
                        } else {
                            ops.push(op);
                        }
                    }
                }
                let term = einsum_with_subscripts::<Alg, Backend>(ctx, &rev_subs, &ops, None)?;
                hvp_k = Some(match hvp_k {
                    None => term,
                    Some(existing) => Tensor::<Alg::Scalar>::accumulate_tangent(existing, &term),
                });
            }
        }

        // When no tangent contributions exist, allocate the zero HVP tensor
        // on the same memory space as the corresponding primal.
        let hvp_k = match hvp_k {
            Some(t) => t,
            None => {
                let space = primals[k].logical_memory_space();
                Tensor::zeros(primals[k].dims(), space, MemoryOrder::ColumnMajor)
            }
        };

        results.push((grad_k, hvp_k));
    }

    Ok(results)
}
