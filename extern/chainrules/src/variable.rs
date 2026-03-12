use std::sync::{Arc, Mutex};

use crate::autograd_context::AutogradContext;
use crate::{AdResult, AutodiffError, Differentiable, NodeId};

pub(crate) fn effective_retain_graph(retain_graph: Option<bool>, create_graph: bool) -> bool {
    retain_graph.unwrap_or(create_graph)
}

/// Options for monomorphic backward/grad APIs.
///
/// # Examples
///
/// ```
/// use chainrules::BackwardOptions;
///
/// let opts = BackwardOptions::<f64>::default();
/// assert_eq!(opts.retain_graph, None);
/// assert!(!opts.create_graph);
/// ```
pub struct BackwardOptions<V: Differentiable> {
    pub retain_graph: Option<bool>,
    pub create_graph: bool,
    pub seed_grad: Option<V::Tangent>,
}

impl<V: Differentiable> Default for BackwardOptions<V> {
    fn default() -> Self {
        Self {
            retain_graph: None,
            create_graph: false,
            seed_grad: None,
        }
    }
}

/// Monomorphic AD variable handle for next API.
///
/// # Examples
///
/// ```
/// use chainrules::Variable;
///
/// let v = Variable::new(3.0_f64);
/// assert!(!v.requires_grad());
/// assert!(v.node_id().is_none());
/// ```
pub struct Variable<V: Differentiable> {
    pub(crate) value: V,
    pub(crate) node_id: Option<NodeId>,
    pub(crate) context: Option<Arc<Mutex<AutogradContext<V>>>>,
    pub(crate) requires_grad: bool,
    pub(crate) tangent: Option<V::Tangent>,
    pub(crate) is_leaf: bool,
}

impl<V: Differentiable> Variable<V> {
    pub fn new(value: V) -> Self {
        Self {
            value,
            node_id: None,
            context: None,
            requires_grad: false,
            tangent: None,
            is_leaf: true,
        }
    }

    pub fn new_in(value: V, ctx: Arc<Mutex<AutogradContext<V>>>) -> Self {
        Self {
            value,
            node_id: None,
            context: Some(ctx),
            requires_grad: false,
            tangent: None,
            is_leaf: true,
        }
    }

    pub fn value(&self) -> &V {
        &self.value
    }

    pub fn ones_like(&self) -> V::Tangent {
        self.value.seed_cotangent()
    }

    pub fn is_scalar(&self) -> bool {
        self.value.num_elements() == 1
    }

    pub fn node_id(&self) -> Option<NodeId> {
        self.node_id
    }

    pub fn context_id(&self) -> Option<u64> {
        self.context
            .as_ref()
            .and_then(|ctx| ctx.lock().ok().map(|g| g.id()))
    }

    pub fn context(&self) -> Option<Arc<Mutex<AutogradContext<V>>>> {
        self.context.as_ref().map(Arc::clone)
    }

    pub fn is_leaf(&self) -> bool {
        self.is_leaf
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn requires_grad_(mut self, enabled: bool) -> AdResult<Self> {
        if enabled && self.context.is_none() {
            self.context = Some(AutogradContext::new());
        }
        if enabled && self.node_id.is_none() {
            if let Some(ctx) = self.context.as_ref() {
                let mut guard = ctx.lock().map_err(|_| {
                    AutodiffError::InvalidArgument("autograd context lock is poisoned".to_string())
                })?;
                self.node_id = Some(guard.record_leaf(None));
            }
        }
        self.requires_grad = enabled;
        Ok(self)
    }

    pub fn with_tangent_(mut self, tangent: V::Tangent) -> AdResult<Self>
    where
        V::Tangent: Clone,
    {
        if let (Some(ctx), Some(node)) = (self.context.as_ref(), self.node_id) {
            let mut guard = ctx.lock().map_err(|_| {
                AutodiffError::InvalidArgument("autograd context lock is poisoned".to_string())
            })?;
            guard.set_node_tangent(node, tangent.clone())?;
        }
        self.tangent = Some(tangent);
        Ok(self)
    }

    pub fn tangent(&self) -> Option<&V::Tangent> {
        self.tangent.as_ref()
    }

    pub fn detach(&self) -> Self
    where
        V: Clone,
    {
        Self {
            value: self.value.clone(),
            node_id: None,
            context: None,
            requires_grad: false,
            tangent: None,
            is_leaf: true,
        }
    }

    pub fn backward(&self, options: BackwardOptions<V>) -> AdResult<()> {
        let retain = effective_retain_graph(options.retain_graph, options.create_graph);
        if !self.requires_grad {
            return Err(AutodiffError::InvalidArgument(
                "backward requires output with requires_grad=true".to_string(),
            ));
        }
        let Some(ctx) = self.context.as_ref() else {
            return Err(AutodiffError::InvalidArgument(
                "backward requires output connected to an autograd context".to_string(),
            ));
        };
        let Some(output_node) = self.node_id else {
            return Err(AutodiffError::InvalidArgument(
                "backward requires output connected to a graph node".to_string(),
            ));
        };
        if !self.is_scalar() && options.seed_grad.is_none() {
            return Err(AutodiffError::InvalidArgument(
                "backward requires seed_grad for non-scalar output".to_string(),
            ));
        }

        let mut guard = ctx.lock().map_err(|_| {
            AutodiffError::InvalidArgument("autograd context lock is poisoned".to_string())
        })?;
        guard.ensure_alive()?;
        let seed = options.seed_grad.unwrap_or_else(|| self.ones_like());
        let mut cotangents = guard.compute_cotangents(output_node, seed)?;
        guard.accumulate_leaf_grads(&mut cotangents);
        if !retain {
            guard.free_graph();
        }
        Ok(())
    }

    pub fn backward_hvp(&self, options: BackwardOptions<V>) -> AdResult<()>
    where
        V::Tangent: Clone + Differentiable<Tangent = V::Tangent>,
    {
        if options.create_graph {
            return Err(AutodiffError::ModeNotSupported {
                mode: "create_graph_hvp".to_string(),
                reason: "backward_hvp with create_graph=true is not implemented yet".to_string(),
            });
        }

        let retain = effective_retain_graph(options.retain_graph, options.create_graph);
        if !self.requires_grad {
            return Err(AutodiffError::InvalidArgument(
                "backward_hvp requires output with requires_grad=true".to_string(),
            ));
        }
        let Some(ctx) = self.context.as_ref() else {
            return Err(AutodiffError::InvalidArgument(
                "backward_hvp requires output connected to an autograd context".to_string(),
            ));
        };
        let Some(output_node) = self.node_id else {
            return Err(AutodiffError::InvalidArgument(
                "backward_hvp requires output connected to a graph node".to_string(),
            ));
        };
        if !self.is_scalar() && options.seed_grad.is_none() {
            return Err(AutodiffError::InvalidArgument(
                "backward_hvp requires seed_grad for non-scalar output".to_string(),
            ));
        }

        let mut guard = ctx.lock().map_err(|_| {
            AutodiffError::InvalidArgument("autograd context lock is poisoned".to_string())
        })?;
        guard.ensure_alive()?;
        if !guard.has_any_leaf_tangent() {
            return Err(AutodiffError::InvalidArgument(
                "hvp requires tangent-seeded leaves".to_string(),
            ));
        }

        let seed = options.seed_grad.unwrap_or_else(|| self.ones_like());
        let seed_tangent = self.value.zero_tangent();
        let (mut cotangents, mut cot_tangents) = guard
            .compute_cotangents_with_tangents(output_node, seed, seed_tangent)
            .map_err(|err| match err {
                AutodiffError::HvpNotSupported => AutodiffError::ModeNotSupported {
                    mode: "hvp".to_string(),
                    reason: "reverse rule does not support pullback_with_tangents".to_string(),
                },
                other => other,
            })?;
        guard.accumulate_leaf_grads(&mut cotangents);
        guard.accumulate_leaf_hvps(&mut cot_tangents);
        if !retain {
            guard.free_graph();
        }
        Ok(())
    }

    pub fn grad(&self) -> Option<V::Tangent>
    where
        V::Tangent: Clone,
    {
        let (Some(ctx), Some(node)) = (self.context.as_ref(), self.node_id) else {
            return None;
        };
        ctx.lock().ok().and_then(|guard| guard.grad_at(node))
    }

    pub fn hvp(&self) -> Option<V::Tangent>
    where
        V::Tangent: Clone,
    {
        let (Some(ctx), Some(node)) = (self.context.as_ref(), self.node_id) else {
            return None;
        };
        ctx.lock().ok().and_then(|guard| guard.hvp_at(node))
    }

    pub fn zero_grad(&self) -> AdResult<()> {
        if !self.is_leaf {
            return Err(AutodiffError::InvalidArgument(
                "zero_grad is valid on leaf variables only".to_string(),
            ));
        }
        let (Some(ctx), Some(node)) = (self.context.as_ref(), self.node_id) else {
            return Ok(());
        };
        let mut guard = ctx.lock().map_err(|_| {
            AutodiffError::InvalidArgument("autograd context lock is poisoned".to_string())
        })?;
        guard.clear_leaf_buffers(node)
    }
}

impl<V: Differentiable + Clone> Clone for Variable<V> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            node_id: self.node_id,
            context: self.context.as_ref().map(Arc::clone),
            requires_grad: self.requires_grad,
            tangent: self.tangent.clone(),
            is_leaf: self.is_leaf,
        }
    }
}
