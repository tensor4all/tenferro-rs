use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::{AdResult, AutodiffError, Differentiable, NodeId, ReverseRule};

static NEXT_CONTEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Copy, Clone)]
pub(crate) enum VariableNodeKind {
    Leaf,
    Add,
    Square { input: NodeId },
    Custom,
}

struct VariableNode<V: Differentiable> {
    rule: Option<Box<dyn ReverseRule<V>>>,
    tangent: Option<V::Tangent>,
    kind: VariableNodeKind,
    is_leaf: bool,
}

/// Shared monomorphic autograd context.
///
/// # Examples
///
/// ```
/// use chainrules::AutogradContext;
///
/// let ctx = AutogradContext::<f64>::new();
/// let id = ctx.lock().unwrap().id();
/// assert!(id > 0);
/// ```
pub struct AutogradContext<V: Differentiable> {
    id: u64,
    graph_alive: bool,
    nodes: Vec<VariableNode<V>>,
    leaf_grads: Vec<Option<V::Tangent>>,
    leaf_hvps: Vec<Option<V::Tangent>>,
}

impl<V: Differentiable> AutogradContext<V> {
    /// Creates a new context.
    pub fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            id: NEXT_CONTEXT_ID.fetch_add(1, Ordering::Relaxed),
            graph_alive: true,
            nodes: Vec::new(),
            leaf_grads: Vec::new(),
            leaf_hvps: Vec::new(),
        }))
    }

    /// Returns context identifier.
    pub fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn ensure_alive(&self) -> AdResult<()> {
        if self.graph_alive {
            Ok(())
        } else {
            Err(AutodiffError::GraphFreed)
        }
    }

    pub(crate) fn free_graph(&mut self) {
        self.graph_alive = false;
    }

    pub(crate) fn record_leaf(&mut self, tangent: Option<V::Tangent>) -> NodeId {
        let id = NodeId::new(self.nodes.len());
        self.nodes.push(VariableNode {
            rule: None,
            tangent,
            kind: VariableNodeKind::Leaf,
            is_leaf: true,
        });
        self.leaf_grads.push(None);
        self.leaf_hvps.push(None);
        self.graph_alive = true;
        id
    }

    pub(crate) fn record_op(
        &mut self,
        rule: Box<dyn ReverseRule<V>>,
        tangent: Option<V::Tangent>,
        kind: VariableNodeKind,
    ) -> NodeId {
        let id = NodeId::new(self.nodes.len());
        self.nodes.push(VariableNode {
            rule: Some(rule),
            tangent,
            kind,
            is_leaf: false,
        });
        self.leaf_grads.push(None);
        self.leaf_hvps.push(None);
        self.graph_alive = true;
        id
    }

    pub(crate) fn set_node_tangent(&mut self, node: NodeId, tangent: V::Tangent) -> AdResult<()> {
        let idx = node.index();
        let Some(entry) = self.nodes.get_mut(idx) else {
            return Err(AutodiffError::MissingNode);
        };
        entry.tangent = Some(tangent);
        Ok(())
    }

    pub(crate) fn node_kind(&self, node: NodeId) -> Option<VariableNodeKind> {
        self.nodes.get(node.index()).map(|n| n.kind)
    }

    pub(crate) fn has_any_leaf_tangent(&self) -> bool {
        self.nodes
            .iter()
            .any(|node| node.is_leaf && node.tangent.is_some())
    }

    pub(crate) fn compute_cotangents(
        &self,
        output_node: NodeId,
        seed: V::Tangent,
    ) -> AdResult<Vec<Option<V::Tangent>>> {
        let n = self.nodes.len();
        if output_node.index() >= n {
            return Err(AutodiffError::MissingNode);
        }

        let mut cotangents = vec![None; n];
        cotangents[output_node.index()] = Some(seed);

        for i in (0..=output_node.index()).rev() {
            let Some(rule) = self.nodes[i].rule.as_ref() else {
                continue;
            };
            let Some(cot) = cotangents[i].take() else {
                continue;
            };
            let input_grads = rule.pullback(&cot)?;
            for (node_id, grad) in input_grads {
                let idx = node_id.index();
                match cotangents[idx].take() {
                    Some(existing) => {
                        cotangents[idx] = Some(V::accumulate_tangent(existing, &grad))
                    }
                    None => cotangents[idx] = Some(grad),
                }
            }
        }

        Ok(cotangents)
    }

    pub(crate) fn compute_cotangents_with_tangents(
        &self,
        output_node: NodeId,
        seed: V::Tangent,
        seed_tangent: V::Tangent,
    ) -> AdResult<(Vec<Option<V::Tangent>>, Vec<Option<V::Tangent>>)>
    where
        V::Tangent: Clone + Differentiable<Tangent = V::Tangent>,
    {
        let n = self.nodes.len();
        if output_node.index() >= n {
            return Err(AutodiffError::MissingNode);
        }

        let mut cotangents = vec![None; n];
        let mut cot_tangents = vec![None; n];
        cotangents[output_node.index()] = Some(seed);
        cot_tangents[output_node.index()] = Some(seed_tangent);

        for i in (0..=output_node.index()).rev() {
            let Some(rule) = self.nodes[i].rule.as_ref() else {
                continue;
            };
            let Some(cot) = cotangents[i].take() else {
                continue;
            };
            let cot_tan = cot_tangents[i].take().unwrap_or_else(|| cot.zero_tangent());
            let input_grads = rule.pullback_with_tangents(&cot, &cot_tan)?;
            for (node_id, grad, grad_tan) in input_grads {
                let idx = node_id.index();
                match cotangents[idx].take() {
                    Some(existing) => {
                        cotangents[idx] = Some(V::accumulate_tangent(existing, &grad))
                    }
                    None => cotangents[idx] = Some(grad),
                }
                match cot_tangents[idx].take() {
                    Some(existing) => {
                        cot_tangents[idx] = Some(V::accumulate_tangent(existing, &grad_tan))
                    }
                    None => cot_tangents[idx] = Some(grad_tan),
                }
            }
        }

        Ok((cotangents, cot_tangents))
    }

    pub(crate) fn accumulate_leaf_grads(&mut self, cotangents: &mut [Option<V::Tangent>]) {
        for (i, cot) in cotangents.iter_mut().enumerate() {
            if !self.nodes[i].is_leaf {
                continue;
            }
            let Some(value) = cot.take() else {
                continue;
            };
            match self.leaf_grads[i].take() {
                Some(existing) => {
                    self.leaf_grads[i] = Some(V::accumulate_tangent(existing, &value));
                }
                None => self.leaf_grads[i] = Some(value),
            }
        }
    }

    pub(crate) fn accumulate_leaf_hvps(&mut self, cot_tangents: &mut [Option<V::Tangent>]) {
        for (i, hv) in cot_tangents.iter_mut().enumerate() {
            if !self.nodes[i].is_leaf {
                continue;
            }
            let Some(value) = hv.take() else {
                continue;
            };
            match self.leaf_hvps[i].take() {
                Some(existing) => {
                    self.leaf_hvps[i] = Some(V::accumulate_tangent(existing, &value));
                }
                None => self.leaf_hvps[i] = Some(value),
            }
        }
    }

    pub(crate) fn grad_at(&self, node: NodeId) -> Option<V::Tangent>
    where
        V::Tangent: Clone,
    {
        self.leaf_grads
            .get(node.index())
            .and_then(|entry| entry.as_ref().cloned())
    }

    pub(crate) fn hvp_at(&self, node: NodeId) -> Option<V::Tangent>
    where
        V::Tangent: Clone,
    {
        self.leaf_hvps
            .get(node.index())
            .and_then(|entry| entry.as_ref().cloned())
    }

    pub(crate) fn clear_leaf_buffers(&mut self, node: NodeId) -> AdResult<()> {
        let idx = node.index();
        let Some(entry) = self.nodes.get(idx) else {
            return Err(AutodiffError::MissingNode);
        };
        if !entry.is_leaf {
            return Err(AutodiffError::InvalidArgument(
                "zero_grad is valid on leaf variables only".to_string(),
            ));
        }
        self.leaf_grads[idx] = None;
        self.leaf_hvps[idx] = None;
        Ok(())
    }
}
