use std::cell::RefCell;
use std::rc::Rc;

use crate::engine::{Gradients, TrackedTensor};
use crate::{AdResult, AutodiffError, Differentiable, HvpResult, NodeId, ReverseRule};

struct TapeNode<V: Differentiable> {
    rule: Option<Box<dyn ReverseRule<V>>>,
    #[allow(dead_code)]
    tangent: Option<V::Tangent>,
    is_leaf: bool,
}

struct TapeInner<V: Differentiable> {
    nodes: Vec<TapeNode<V>>,
}

/// Reverse-mode AD tape.
///
/// The tape records operations performed on [`TrackedTensor`] values and
/// enables gradient computation via [`Tape::pullback`] or HVP via
/// [`Tape::hvp`].
///
/// `Tape` is cheaply cloneable (internally reference-counted). Multiple
/// clones refer to the same underlying tape.
///
/// # Examples
///
/// ```ignore
/// use chainrules::Tape;
/// use std::cell::RefCell;
/// use std::rc::Rc;
/// use tenferro_algebra::Standard;
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_einsum::tracked_einsum;
/// use tenferro_prims::{CpuBackend, CpuContext};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let tape = Tape::<Tensor<f64>>::new();
/// let ctx = Rc::new(RefCell::new(CpuContext::new(1)));
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
/// let c =
///     tracked_einsum::<Standard<f64>, CpuBackend>(ctx.clone(), "ij,jk->ik", &[&a, &b]).unwrap();
/// let loss =
///     tracked_einsum::<Standard<f64>, CpuBackend>(ctx.clone(), "ij,ij->", &[&c, &c]).unwrap();
/// let grads = tape.pullback(&loss).unwrap();
/// let _ga = grads.get(a.node_id().unwrap()).unwrap();
/// ```
pub struct Tape<V: Differentiable> {
    inner: Rc<RefCell<TapeInner<V>>>,
}

impl<V: Differentiable> Tape<V> {
    /// Creates a new empty tape.
    pub fn new() -> Self {
        Self {
            inner: Rc::new(RefCell::new(TapeInner { nodes: Vec::new() })),
        }
    }

    /// Returns `true` if `self` and `other` are the same tape.
    pub fn same_tape(&self, other: &Tape<V>) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }

    /// Creates a leaf value requiring gradient on this tape.
    pub fn leaf(&self, value: V) -> TrackedTensor<V> {
        let mut inner = self.inner.borrow_mut();
        let node_id = NodeId::new(inner.nodes.len());
        inner.nodes.push(TapeNode {
            rule: None,
            tangent: None,
            is_leaf: true,
        });
        TrackedTensor {
            value,
            node_id: Some(node_id),
            tape: Some(self.clone()),
            requires_grad: true,
            tangent: None,
        }
    }

    /// Creates a leaf value with a tangent for HVP computation.
    pub fn leaf_with_tangent(&self, value: V, tangent: V::Tangent) -> AdResult<TrackedTensor<V>> {
        let mut inner = self.inner.borrow_mut();
        let node_id = NodeId::new(inner.nodes.len());
        inner.nodes.push(TapeNode {
            rule: None,
            tangent: Some(tangent.clone()),
            is_leaf: true,
        });
        Ok(TrackedTensor {
            value,
            node_id: Some(node_id),
            tape: Some(self.clone()),
            requires_grad: true,
            tangent: Some(tangent),
        })
    }

    /// Records an operation on the tape, returning a tracked output.
    pub fn record_op(
        &self,
        output_value: V,
        rule: Box<dyn ReverseRule<V>>,
        output_tangent: Option<V::Tangent>,
    ) -> TrackedTensor<V> {
        let mut inner = self.inner.borrow_mut();
        let node_id = NodeId::new(inner.nodes.len());
        inner.nodes.push(TapeNode {
            rule: Some(rule),
            tangent: output_tangent.clone(),
            is_leaf: false,
        });
        TrackedTensor {
            value: output_value,
            node_id: Some(node_id),
            tape: Some(self.clone()),
            requires_grad: true,
            tangent: output_tangent,
        }
    }

    /// Runs reverse-mode pullback from a scalar loss value.
    pub fn pullback(&self, loss: &TrackedTensor<V>) -> AdResult<Gradients<V>> {
        let loss_node = loss.node_id.ok_or(AutodiffError::MissingNode)?;
        let n = loss.value.num_elements();
        if n != 1 {
            return Err(AutodiffError::NonScalarLoss { num_elements: n });
        }

        let inner = self.inner.borrow();
        let mut cotangents = vec![None; inner.nodes.len()];
        cotangents[loss_node.index()] = Some(loss.value.seed_cotangent());

        for i in (0..=loss_node.index()).rev() {
            let Some(rule) = inner.nodes[i].rule.as_ref() else {
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
                        cotangents[idx] = Some(V::accumulate_tangent(existing, &grad));
                    }
                    None => cotangents[idx] = Some(grad),
                }
            }
        }

        let mut result = Gradients::new();
        for (i, cot) in cotangents.into_iter().enumerate() {
            if let Some(c) = cot {
                if inner.nodes[i].is_leaf {
                    result.push_entry(NodeId::new(i), c);
                }
            }
        }
        Ok(result)
    }

    /// Computes gradient and Hessian-vector product via forward-over-reverse.
    pub fn hvp(&self, loss: &TrackedTensor<V>) -> AdResult<HvpResult<V>>
    where
        V::Tangent: Differentiable<Tangent = V::Tangent>,
    {
        let loss_node = loss.node_id.ok_or(AutodiffError::MissingNode)?;
        let n = loss.value.num_elements();
        if n != 1 {
            return Err(AutodiffError::NonScalarLoss { num_elements: n });
        }

        let inner = self.inner.borrow();
        let mut cotangents = vec![None; inner.nodes.len()];
        let mut cot_tangents = vec![None; inner.nodes.len()];
        cotangents[loss_node.index()] = Some(loss.value.seed_cotangent());
        cot_tangents[loss_node.index()] = Some(loss.value.zero_tangent());

        for i in (0..=loss_node.index()).rev() {
            let Some(rule) = inner.nodes[i].rule.as_ref() else {
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
                        cotangents[idx] = Some(V::accumulate_tangent(existing, &grad));
                    }
                    None => cotangents[idx] = Some(grad),
                }
                match cot_tangents[idx].take() {
                    Some(existing) => {
                        cot_tangents[idx] = Some(V::accumulate_tangent(existing, &grad_tan));
                    }
                    None => cot_tangents[idx] = Some(grad_tan),
                }
            }
        }

        let mut gradients = Gradients::new();
        let mut hvp_grads = Gradients::new();
        for i in 0..inner.nodes.len() {
            if inner.nodes[i].is_leaf {
                if let Some(c) = cotangents[i].take() {
                    gradients.push_entry(NodeId::new(i), c);
                }
                if let Some(ct) = cot_tangents[i].take() {
                    hvp_grads.push_entry(NodeId::new(i), ct);
                }
            }
        }

        Ok(HvpResult {
            gradients,
            hvp: hvp_grads,
        })
    }
}

impl<V: Differentiable> Clone for Tape<V> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<V: Differentiable> Default for Tape<V> {
    fn default() -> Self {
        Self::new()
    }
}
