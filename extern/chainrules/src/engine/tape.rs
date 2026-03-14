use std::cell::RefCell;
use std::rc::Rc;

use crate::engine::{Gradients, TrackedValue};
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
/// The tape records operations performed on [`TrackedValue`] values and
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

    /// Returns a stable process-local identifier for this tape.
    ///
    /// This is derived from the shared allocation backing the tape and is
    /// intended for diagnostics and higher-level wrappers that need an opaque
    /// tape token.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::Tape;
    ///
    /// let tape = Tape::<f64>::new();
    /// assert_ne!(tape.id(), 0);
    /// ```
    pub fn id(&self) -> usize {
        Rc::as_ptr(&self.inner) as usize
    }

    /// Returns the current number of nodes recorded on this tape.
    ///
    /// # Examples
    ///
    /// ```
    /// use chainrules::Tape;
    ///
    /// let tape = Tape::<f64>::new();
    /// assert_eq!(tape.node_count(), 0);
    /// let _x = tape.leaf(1.0);
    /// assert_eq!(tape.node_count(), 1);
    /// ```
    pub fn node_count(&self) -> usize {
        self.inner.borrow().nodes.len()
    }

    /// Creates a leaf value requiring gradient on this tape.
    pub fn leaf(&self, value: V) -> TrackedValue<V> {
        let mut inner = self.inner.borrow_mut();
        let node_id = NodeId::new(inner.nodes.len());
        inner.nodes.push(TapeNode {
            rule: None,
            tangent: None,
            is_leaf: true,
        });
        TrackedValue {
            value,
            node_id: Some(node_id),
            tape: Some(self.clone()),
            requires_grad: true,
            tangent: None,
        }
    }

    /// Creates a leaf value with a tangent for HVP computation.
    pub fn leaf_with_tangent(&self, value: V, tangent: V::Tangent) -> AdResult<TrackedValue<V>> {
        let mut inner = self.inner.borrow_mut();
        let node_id = NodeId::new(inner.nodes.len());
        inner.nodes.push(TapeNode {
            rule: None,
            tangent: Some(tangent.clone()),
            is_leaf: true,
        });
        Ok(TrackedValue {
            value,
            node_id: Some(node_id),
            tape: Some(self.clone()),
            requires_grad: true,
            tangent: Some(tangent),
        })
    }

    /// Records an output value on the tape before attaching its reverse rule.
    ///
    /// This is useful for wrappers that need the output node handle before the
    /// pullback closure can be finalized.
    ///
    /// The resulting node is not considered a leaf and contributes gradients
    /// only after a rule is attached via [`Tape::attach_rule`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use chainrules::{Tape, ReverseRule};
    ///
    /// let tape = Tape::<f64>::new();
    /// let out = tape.placeholder(3.0, None);
    /// # let _ = out;
    /// ```
    pub fn placeholder(&self, value: V, tangent: Option<V::Tangent>) -> TrackedValue<V> {
        let mut inner = self.inner.borrow_mut();
        let node_id = NodeId::new(inner.nodes.len());
        inner.nodes.push(TapeNode {
            rule: None,
            tangent: tangent.clone(),
            is_leaf: false,
        });
        TrackedValue {
            value,
            node_id: Some(node_id),
            tape: Some(self.clone()),
            requires_grad: true,
            tangent,
        }
    }

    /// Reconstructs a tracked handle for an existing node already recorded on
    /// this tape.
    ///
    /// This is intended for higher-level wrappers that persist opaque
    /// `NodeId`/tape tokens and need to re-enter the generic tape API.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use chainrules::Tape;
    ///
    /// let tape = Tape::<f64>::new();
    /// let x = tape.leaf(1.0);
    /// let tracked = tape.tracked_existing(x.node_id().unwrap(), 1.0, None)?;
    /// # Ok::<(), chainrules::AutodiffError>(())
    /// ```
    pub fn tracked_existing(
        &self,
        node_id: NodeId,
        value: V,
        tangent: Option<V::Tangent>,
    ) -> AdResult<TrackedValue<V>> {
        let inner = self.inner.borrow();
        if node_id.index() >= inner.nodes.len() {
            return Err(AutodiffError::InvalidArgument(format!(
                "node {} is not present on this tape",
                node_id.index()
            )));
        }
        Ok(TrackedValue {
            value,
            node_id: Some(node_id),
            tape: Some(self.clone()),
            requires_grad: true,
            tangent,
        })
    }

    /// Records an operation on the tape, returning a tracked output.
    pub fn record_op(
        &self,
        output_value: V,
        rule: Box<dyn ReverseRule<V>>,
        output_tangent: Option<V::Tangent>,
    ) -> TrackedValue<V> {
        let mut inner = self.inner.borrow_mut();
        let node_id = NodeId::new(inner.nodes.len());
        inner.nodes.push(TapeNode {
            rule: Some(rule),
            tangent: output_tangent.clone(),
            is_leaf: false,
        });
        TrackedValue {
            value: output_value,
            node_id: Some(node_id),
            tape: Some(self.clone()),
            requires_grad: true,
            tangent: output_tangent,
        }
    }

    /// Attaches or replaces the reverse rule for an existing output node.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use chainrules::{NodeId, ReverseRule, Tape};
    ///
    /// let tape = Tape::<f64>::new();
    /// let out = tape.placeholder(1.0, None);
    /// # let _ = tape.attach_rule(out.node_id().unwrap(), todo!());
    /// ```
    pub fn attach_rule(&self, node_id: NodeId, rule: Box<dyn ReverseRule<V>>) -> AdResult<()> {
        let mut inner = self.inner.borrow_mut();
        let Some(node) = inner.nodes.get_mut(node_id.index()) else {
            return Err(AutodiffError::InvalidArgument(format!(
                "node {} is not present on this tape",
                node_id.index()
            )));
        };
        node.rule = Some(rule);
        node.is_leaf = false;
        Ok(())
    }

    /// Runs reverse-mode pullback from a scalar loss value.
    pub fn pullback(&self, loss: &TrackedValue<V>) -> AdResult<Gradients<V>> {
        let n = loss.value.num_elements();
        if n != 1 {
            return Err(AutodiffError::NonScalarLoss { num_elements: n });
        }
        self.pullback_with_seed(loss, loss.value.seed_cotangent())
    }

    /// Runs reverse-mode pullback from an arbitrary output cotangent seed.
    ///
    /// Unlike [`Tape::pullback`], this method does not require the output to be
    /// scalar. It is intended for tensor-valued reverse-mode wrappers that
    /// provide an explicit cotangent.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use chainrules::Tape;
    ///
    /// let tape = Tape::<f64>::new();
    /// let x = tape.leaf(1.0);
    /// let grads = tape.pullback_with_seed(&x, 3.0).unwrap();
    /// assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 3.0);
    /// ```
    pub fn pullback_with_seed(
        &self,
        output: &TrackedValue<V>,
        seed: V::Tangent,
    ) -> AdResult<Gradients<V>> {
        let loss_node = output.node_id.ok_or(AutodiffError::MissingNode)?;

        let inner = self.inner.borrow();
        let mut cotangents = vec![None; inner.nodes.len()];
        cotangents[loss_node.index()] = Some(seed);

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
    pub fn hvp(&self, loss: &TrackedValue<V>) -> AdResult<HvpResult<V>>
    where
        V::Tangent: Differentiable<Tangent = V::Tangent>,
    {
        let loss_node = loss.node_id.ok_or(AutodiffError::MissingNode)?;
        let n = loss.value.num_elements();
        if n != 1 {
            return Err(AutodiffError::NonScalarLoss { num_elements: n });
        }

        let inner = self.inner.borrow();
        let mut cotangents: Vec<Option<V::Tangent>> = vec![None; inner.nodes.len()];
        let mut cot_tangents: Vec<Option<V::Tangent>> = vec![None; inner.nodes.len()];
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
