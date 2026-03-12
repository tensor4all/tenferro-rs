use crate::{
    autograd_context::VariableNodeKind, variable::effective_retain_graph, AdResult, AutodiffError,
    AutogradContext, BackwardOptions, NodeId, ReverseRule, Variable,
};
use std::marker::PhantomData;
use std::ops::{Add, Mul};
use std::sync::{Arc, Mutex};

struct AddRule<V: crate::Differentiable<Tangent = V> + Clone> {
    lhs: Option<NodeId>,
    rhs: Option<NodeId>,
    _marker: PhantomData<V>,
}

impl<V> ReverseRule<V> for AddRule<V>
where
    V: crate::Differentiable<Tangent = V> + Clone,
{
    fn pullback(&self, cotangent: &V::Tangent) -> AdResult<Vec<(NodeId, V::Tangent)>> {
        let mut out = Vec::new();
        if let Some(lhs) = self.lhs {
            out.push((lhs, cotangent.clone()));
        }
        if let Some(rhs) = self.rhs {
            out.push((rhs, cotangent.clone()));
        }
        Ok(out)
    }

    fn inputs(&self) -> Vec<NodeId> {
        let mut out = Vec::new();
        if let Some(lhs) = self.lhs {
            out.push(lhs);
        }
        if let Some(rhs) = self.rhs {
            out.push(rhs);
        }
        out
    }

    fn pullback_with_tangents(
        &self,
        cotangent: &V::Tangent,
        cotangent_tangent: &V::Tangent,
    ) -> AdResult<Vec<(NodeId, V::Tangent, V::Tangent)>> {
        let mut out = Vec::new();
        if let Some(lhs) = self.lhs {
            out.push((lhs, cotangent.clone(), cotangent_tangent.clone()));
        }
        if let Some(rhs) = self.rhs {
            out.push((rhs, cotangent.clone(), cotangent_tangent.clone()));
        }
        Ok(out)
    }
}

struct SquareRule<V: crate::Differentiable<Tangent = V> + Clone + Add<Output = V> + Mul<Output = V>>
{
    input: NodeId,
    two_x: V,
    two_dx: Option<V>,
}

impl<V> ReverseRule<V> for SquareRule<V>
where
    V: crate::Differentiable<Tangent = V> + Clone + Add<Output = V> + Mul<Output = V>,
{
    fn pullback(&self, cotangent: &V::Tangent) -> AdResult<Vec<(NodeId, V::Tangent)>> {
        Ok(vec![(self.input, cotangent.clone() * self.two_x.clone())])
    }

    fn inputs(&self) -> Vec<NodeId> {
        vec![self.input]
    }

    fn pullback_with_tangents(
        &self,
        cotangent: &V::Tangent,
        cotangent_tangent: &V::Tangent,
    ) -> AdResult<Vec<(NodeId, V::Tangent, V::Tangent)>> {
        let grad = cotangent.clone() * self.two_x.clone();
        let mut grad_tangent = cotangent_tangent.clone() * self.two_x.clone();
        if let Some(two_dx) = self.two_dx.as_ref() {
            grad_tangent = grad_tangent + cotangent.clone() * two_dx.clone();
        }
        Ok(vec![(self.input, grad, grad_tangent)])
    }
}

fn context_id<V: crate::Differentiable>(ctx: &Arc<Mutex<AutogradContext<V>>>) -> AdResult<u64> {
    ctx.lock().map(|guard| guard.id()).map_err(|_| {
        AutodiffError::InvalidArgument("autograd context lock is poisoned".to_string())
    })
}

fn merge_context_for_multi_op<V: crate::Differentiable>(
    inputs: &[&Variable<V>],
) -> AdResult<Option<Arc<Mutex<AutogradContext<V>>>>> {
    if inputs.iter().all(|input| !input.requires_grad()) {
        return Ok(None);
    }

    let mut picked: Option<(u64, Arc<Mutex<AutogradContext<V>>>)> = None;
    for ctx in inputs
        .iter()
        .filter(|input| input.requires_grad())
        .filter_map(|input| input.context.as_ref())
    {
        let id = context_id(ctx)?;
        match &picked {
            None => picked = Some((id, Arc::clone(ctx))),
            Some((picked_id, _)) if *picked_id == id => {}
            Some(_) => {
                return Err(AutodiffError::InvalidArgument(
                    "mixed autograd contexts in one operation; use Variable::new_in(..., same_ctx)"
                        .to_string(),
                ))
            }
        }
    }

    let Some((picked_id, picked_ctx)) = picked else {
        return Ok(None);
    };

    let any_tracked_on_picked = inputs.iter().any(|input| {
        input.requires_grad() && input.context_id() == Some(picked_id) && input.node_id.is_some()
    });
    if any_tracked_on_picked {
        Ok(Some(picked_ctx))
    } else {
        Ok(None)
    }
}

fn merge_context_for_binary_op<V: crate::Differentiable>(
    lhs: &Variable<V>,
    rhs: &Variable<V>,
) -> AdResult<Option<Arc<Mutex<AutogradContext<V>>>>> {
    merge_context_for_multi_op(&[lhs, rhs])
}

/// Records a custom operation on the monomorphic `Variable` graph.
///
/// This helper is intended for operation crates (for example einsum) that
/// need to construct `Variable` outputs with a custom reverse rule.
///
/// # Errors
///
/// Returns [`AutodiffError::InvalidArgument`] when inputs span different
/// autograd contexts.
///
/// # Examples
///
/// ```text
/// // Intended for operation implementations:
/// // let out = autograd::record_op(value, &[&x, &y], Box::new(rule), tangent)?;
/// ```
pub fn record_op<V>(
    value: V,
    inputs: &[&Variable<V>],
    rule: Box<dyn ReverseRule<V>>,
    tangent: Option<V::Tangent>,
) -> AdResult<Variable<V>>
where
    V: crate::Differentiable + 'static,
    V::Tangent: Clone,
{
    let out_ctx = merge_context_for_multi_op(inputs)?;
    let mut out_node = None;

    if let Some(ctx) = out_ctx.as_ref() {
        let mut guard = ctx.lock().map_err(|_| {
            AutodiffError::InvalidArgument("autograd context lock is poisoned".to_string())
        })?;
        out_node = Some(guard.record_op(rule, tangent.clone(), VariableNodeKind::Custom));
    }

    Ok(Variable {
        value,
        node_id: out_node,
        context: out_ctx.clone(),
        requires_grad: out_ctx.is_some(),
        tangent,
        is_leaf: false,
    })
}

/// Adds two variables and applies Context Merge Rule to the output.
///
/// # Errors
///
/// Returns [`AutodiffError::InvalidArgument`] when operands belong to
/// different contexts.
///
/// # Examples
///
/// ```
/// use chainrules::{autograd, AutogradContext, Variable};
/// use std::sync::Arc;
///
/// let ctx = AutogradContext::<f64>::new();
/// let a = Variable::new_in(1.0_f64, Arc::clone(&ctx)).requires_grad_(true).unwrap();
/// let b = Variable::new_in(2.0_f64, Arc::clone(&ctx)).requires_grad_(true).unwrap();
/// let c = autograd::add(&a, &b).unwrap();
/// assert!(c.requires_grad());
/// ```
pub fn add<V>(lhs: &Variable<V>, rhs: &Variable<V>) -> AdResult<Variable<V>>
where
    V: crate::Differentiable<Tangent = V> + Clone + Add<Output = V> + 'static,
{
    let out_ctx = merge_context_for_binary_op(lhs, rhs)?;
    let out_value = lhs.value.clone() + rhs.value.clone();
    let out_tangent = match (lhs.tangent.as_ref(), rhs.tangent.as_ref()) {
        (Some(lt), Some(rt)) => Some(lt.clone() + rt.clone()),
        (Some(lt), None) => Some(lt.clone()),
        (None, Some(rt)) => Some(rt.clone()),
        (None, None) => None,
    };

    let mut out_node = None;
    if let Some(ctx) = out_ctx.as_ref() {
        let lhs_ctx = lhs.context_id();
        let rhs_ctx = rhs.context_id();
        let mut guard = ctx.lock().map_err(|_| {
            AutodiffError::InvalidArgument("autograd context lock is poisoned".to_string())
        })?;
        let ctx_id = guard.id();
        let lhs_dep = if lhs.requires_grad() && lhs_ctx == Some(ctx_id) {
            Some(lhs.node_id.ok_or(AutodiffError::MissingNode)?)
        } else {
            None
        };
        let rhs_dep = if rhs.requires_grad() && rhs_ctx == Some(ctx_id) {
            Some(rhs.node_id.ok_or(AutodiffError::MissingNode)?)
        } else {
            None
        };
        let rule = AddRule::<V> {
            lhs: lhs_dep,
            rhs: rhs_dep,
            _marker: PhantomData,
        };
        out_node =
            Some(guard.record_op(Box::new(rule), out_tangent.clone(), VariableNodeKind::Add));
    }

    Ok(Variable {
        value: out_value,
        node_id: out_node,
        context: out_ctx.clone(),
        requires_grad: out_ctx.is_some(),
        tangent: out_tangent,
        is_leaf: false,
    })
}

/// Squares one variable and preserves context when tracked.
///
/// # Examples
///
/// ```
/// use chainrules::{autograd, Variable};
///
/// let x = Variable::new(3.0_f64).requires_grad_(true).unwrap();
/// let y = autograd::square(&x).unwrap();
/// assert!(y.requires_grad());
/// ```
pub fn square<V>(input: &Variable<V>) -> AdResult<Variable<V>>
where
    V: crate::Differentiable<Tangent = V> + Clone + Add<Output = V> + Mul<Output = V> + 'static,
{
    let out_ctx = if input.requires_grad() {
        input.context.as_ref().map(Arc::clone)
    } else {
        None
    };

    let two_x = input.value.clone() + input.value.clone();
    let out_value = input.value.clone() * input.value.clone();
    let out_tangent = input.tangent.as_ref().map(|dx| two_x.clone() * dx.clone());
    let two_dx = input.tangent.as_ref().map(|dx| dx.clone() + dx.clone());

    let mut out_node = None;
    if let Some(ctx) = out_ctx.as_ref() {
        let input_ctx = input.context_id();
        let mut guard = ctx.lock().map_err(|_| {
            AutodiffError::InvalidArgument("autograd context lock is poisoned".to_string())
        })?;
        let ctx_id = guard.id();
        if input.requires_grad() && input_ctx == Some(ctx_id) {
            let input_node = input.node_id.ok_or(AutodiffError::MissingNode)?;
            let rule = SquareRule::<V> {
                input: input_node,
                two_x: two_x.clone(),
                two_dx: two_dx.clone(),
            };
            out_node = Some(guard.record_op(
                Box::new(rule),
                out_tangent.clone(),
                VariableNodeKind::Square { input: input_node },
            ));
        }
    }

    Ok(Variable {
        value: out_value,
        node_id: out_node,
        context: out_ctx.clone(),
        requires_grad: out_ctx.is_some(),
        tangent: out_tangent,
        is_leaf: false,
    })
}

/// Side-effect-free gradient query returning detached tangents.
///
/// # Errors
///
/// Returns `ModeNotSupported { mode: "create_graph_tangent", .. }` when
/// `create_graph=true`.
pub fn grad_tangent<V>(
    output: &Variable<V>,
    inputs: &[&Variable<V>],
    options: BackwardOptions<V>,
) -> AdResult<Vec<V::Tangent>>
where
    V: crate::Differentiable,
    V::Tangent: Clone,
{
    if options.create_graph {
        return Err(AutodiffError::ModeNotSupported {
            mode: "create_graph_tangent".to_string(),
            reason: "grad_tangent does not support create_graph".to_string(),
        });
    }

    let retain = effective_retain_graph(options.retain_graph, options.create_graph);
    if !output.requires_grad() {
        return Err(AutodiffError::InvalidArgument(
            "grad_tangent requires output with requires_grad=true".to_string(),
        ));
    }
    let Some(ctx) = output.context.as_ref() else {
        return Err(AutodiffError::InvalidArgument(
            "grad_tangent requires output connected to an autograd context".to_string(),
        ));
    };
    let Some(output_node) = output.node_id else {
        return Err(AutodiffError::InvalidArgument(
            "grad_tangent requires output connected to a graph node".to_string(),
        ));
    };
    if !output.is_scalar() && options.seed_grad.is_none() {
        return Err(AutodiffError::InvalidArgument(
            "grad_tangent requires seed_grad for non-scalar output".to_string(),
        ));
    }

    let mut guard = ctx.lock().map_err(|_| {
        AutodiffError::InvalidArgument("autograd context lock is poisoned".to_string())
    })?;
    guard.ensure_alive()?;
    for input in inputs {
        if let Some(input_ctx) = input.context.as_ref() {
            if !Arc::ptr_eq(input_ctx, ctx) {
                return Err(AutodiffError::InvalidArgument(
                    "mixed autograd contexts in grad query".to_string(),
                ));
            }
        }
    }

    let seed = options.seed_grad.unwrap_or_else(|| output.ones_like());
    let cotangents = guard.compute_cotangents(output_node, seed)?;
    let mut out = Vec::with_capacity(inputs.len());
    for input in inputs {
        let grad = match input.node_id {
            Some(node) => cotangents
                .get(node.index())
                .and_then(|v| v.as_ref().cloned())
                .unwrap_or_else(|| input.value.zero_tangent()),
            None => input.value.zero_tangent(),
        };
        out.push(grad);
    }

    if !retain {
        guard.free_graph();
    }
    Ok(out)
}

/// Side-effect-free gradient query returning monomorphic variables.
///
/// # Examples
///
/// ```
/// use chainrules::{autograd, BackwardOptions, Variable};
///
/// let x = Variable::new(2.0_f64).requires_grad_(true).unwrap();
/// let y = autograd::square(&x).unwrap();
/// let grads = autograd::grad_variable(&y, &[&x], BackwardOptions::default()).unwrap();
/// assert_eq!(grads.len(), 1);
/// ```
pub fn grad_variable<V>(
    output: &Variable<V>,
    inputs: &[&Variable<V>],
    options: BackwardOptions<V>,
) -> AdResult<Vec<Variable<V>>>
where
    V: crate::Differentiable<Tangent = V> + Clone + Add<Output = V> + Mul<Output = V> + 'static,
{
    let retain = effective_retain_graph(options.retain_graph, options.create_graph);
    if !output.requires_grad() {
        return Err(AutodiffError::InvalidArgument(
            "grad_variable requires output with requires_grad=true".to_string(),
        ));
    }
    let Some(ctx) = output.context.as_ref() else {
        return Err(AutodiffError::InvalidArgument(
            "grad_variable requires output connected to an autograd context".to_string(),
        ));
    };
    let Some(output_node) = output.node_id else {
        return Err(AutodiffError::InvalidArgument(
            "grad_variable requires output connected to a graph node".to_string(),
        ));
    };
    if !output.is_scalar() && options.seed_grad.is_none() {
        return Err(AutodiffError::InvalidArgument(
            "grad_variable requires seed_grad for non-scalar output".to_string(),
        ));
    }

    let guard = ctx.lock().map_err(|_| {
        AutodiffError::InvalidArgument("autograd context lock is poisoned".to_string())
    })?;
    guard.ensure_alive()?;
    for input in inputs {
        if let Some(input_ctx) = input.context.as_ref() {
            if !Arc::ptr_eq(input_ctx, ctx) {
                return Err(AutodiffError::InvalidArgument(
                    "mixed autograd contexts in grad query".to_string(),
                ));
            }
        }
    }

    let seed = options.seed_grad.unwrap_or_else(|| output.ones_like());
    let cotangents = guard.compute_cotangents(output_node, seed)?;
    let output_kind = guard.node_kind(output_node);
    drop(guard);

    let mut out = Vec::with_capacity(inputs.len());
    for input in inputs {
        let grad_value = match input.node_id {
            Some(node) => cotangents
                .get(node.index())
                .and_then(|v| v.as_ref().cloned())
                .unwrap_or_else(|| input.value.zero_tangent()),
            None => input.value.zero_tangent(),
        };

        if options.create_graph {
            let symbolic_square = matches!(
                (output_kind, input.node_id),
                (Some(VariableNodeKind::Square { input: src }), Some(n)) if src == n
            );
            if symbolic_square {
                out.push(add(input, input)?);
            } else {
                return Err(AutodiffError::ModeNotSupported {
                    mode: "create_graph_grad_variable".to_string(),
                    reason: "only direct square gradients are graph-connected in this phase"
                        .to_string(),
                });
            }
        } else {
            out.push(Variable::new(grad_value));
        }
    }

    if !retain {
        let mut guard = ctx.lock().map_err(|_| {
            AutodiffError::InvalidArgument("autograd context lock is poisoned".to_string())
        })?;
        guard.free_graph();
    }
    Ok(out)
}
