use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use computegraph::fragment::Fragment;
use computegraph::{GlobalValKey, ValRef};
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::ShapeGuardContext;
use tenferro_tensor::cpu::CpuBackend;
use tenferro_tensor::{Tensor, TensorBackend, TypedTensor};
use tidu::{
    topo_sort_grad_dag, try_backward_dag, BackwardCallbacks, EagerOutput, EagerValue, GradNode,
    LinearFragment,
};

use crate::eager_emitter::EagerEmitter;
use crate::eager_exec::exec_op_on_tensors;
use crate::error::{ContextId, Error, Result};
use crate::metadata::{
    metadata_scopes_for_scope, push_metadata_scope, register_scoped_fragment_metadata,
    register_scoped_metadata_batch, register_scoped_value_metadata, tensor_meta_from_tensor,
    MetadataScope,
};
use crate::traced::next_input_key;

pub(crate) type GradSlot = Arc<Mutex<Option<Arc<Tensor>>>>;
pub(crate) type WeakGradSlot = Weak<Mutex<Option<Arc<Tensor>>>>;

/// Shared eager execution context for tensors on a backend.
///
/// Reusing one context lets eager tensors share backend state and gradient
/// storage across a computation.
///
/// # Examples
///
/// ```
/// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
///
/// let ctx = EagerContext::with_backend(CpuBackend::new());
/// let x = EagerTensor::from_tensor_in(Tensor::from_vec(vec![1], vec![1.0_f64]), ctx.clone());
/// let y = EagerTensor::from_tensor_in(Tensor::from_vec(vec![1], vec![2.0_f64]), ctx);
/// let z = &x + &y;
///
/// assert_eq!(z.data().as_slice::<f64>().unwrap(), &[3.0]);
/// ```
pub struct EagerContext<B: TensorBackend> {
    pub(crate) backend: Mutex<B>,
    grad_slots: Mutex<HashMap<GlobalValKey<StdTensorOp>, WeakGradSlot>>,
}

impl<B: TensorBackend> EagerContext<B> {
    fn new(backend: B) -> Self {
        Self {
            backend: Mutex::new(backend),
            grad_slots: Mutex::new(HashMap::new()),
        }
    }

    /// Create a shared eager execution context for the provided backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerContext};
    ///
    /// let ctx = EagerContext::with_backend(CpuBackend::new());
    /// assert_eq!(std::sync::Arc::strong_count(&ctx), 1);
    /// ```
    pub fn with_backend(backend: B) -> Arc<Self> {
        Arc::new(Self::new(backend))
    }

    /// Return an opaque identifier for this context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerContext};
    ///
    /// let ctx = EagerContext::with_backend(CpuBackend::new());
    /// assert_ne!(ctx.id(), EagerContext::with_backend(CpuBackend::new()).id());
    /// ```
    pub fn id(&self) -> ContextId {
        ContextId::from_ptr(self)
    }

    pub(crate) fn register_grad_slot(&self, key: &GlobalValKey<StdTensorOp>, slot: &GradSlot) {
        self.grad_slots
            .lock()
            .unwrap()
            .insert(key.clone(), Arc::downgrade(slot));
    }

    /// Clear all live gradient slots tracked by this context.
    ///
    /// This resets the stored gradients to `None` without unregistering the
    /// tensors, so future `backward()` calls can accumulate again.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_backend(CpuBackend::new());
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]), ctx.clone());
    /// let y = EagerTensor::requires_grad_in(Tensor::from_vec(vec![3], vec![4.0_f64, 5.0, 6.0]), ctx.clone());
    /// let loss = (&x * &y).reduce_sum(&[0]).unwrap();
    /// let _ = loss.backward().unwrap();
    ///
    /// ctx.clear_grads();
    ///
    /// assert!(x.grad().is_none());
    /// assert!(y.grad().is_none());
    /// ```
    pub fn clear_grads(&self) {
        self.grad_slots.lock().unwrap().retain(|_, slot| {
            if let Some(slot) = slot.upgrade() {
                *slot.lock().unwrap() = None;
                true
            } else {
                false
            }
        });
    }

    /// Import a concrete tensor into this context as an untracked constant.
    ///
    /// The returned tensor does not participate in gradient tracking.
    /// Use this for fixed masks, quadrature weights, physical constants,
    /// and other data that should not receive gradients.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_backend(CpuBackend::new());
    /// let c = ctx.constant_from(Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]));
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec(vec![2], vec![3.0_f64, 4.0]), ctx);
    /// let z = x.add(&c).unwrap();
    ///
    /// assert_eq!(z.data().as_slice::<f64>().unwrap(), &[4.0, 6.0]);
    /// ```
    pub fn constant_from(self: &Arc<Self>, tensor: Tensor) -> EagerTensor<B> {
        EagerTensor::new_leaf(Arc::clone(self), tensor, false)
    }

    /// Import a concrete tensor into this context as a trainable variable.
    ///
    /// The returned tensor participates in gradient tracking; its gradient
    /// slot is registered in this context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_backend(CpuBackend::new());
    /// let p = ctx.variable_from(Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]));
    /// let loss = p.exp().unwrap().reduce_sum(&[0]).unwrap();
    /// let _ = loss.backward().unwrap();
    ///
    /// let grad = p.grad().unwrap();
    /// assert_eq!(grad.shape(), &[2]);
    /// ```
    pub fn variable_from(self: &Arc<Self>, tensor: Tensor) -> EagerTensor<B> {
        EagerTensor::new_leaf(Arc::clone(self), tensor, true)
    }

    fn store_grads(
        &self,
        cotangents: &HashMap<GlobalValKey<StdTensorOp>, Arc<Tensor>>,
        backend: &mut B,
    ) -> Result<()> {
        let mut updates = Vec::new();
        let mut staged = Vec::new();

        {
            let mut slots = self.grad_slots.lock().unwrap();
            slots.retain(|key, slot| {
                let Some(slot) = slot.upgrade() else {
                    return false;
                };

                if let Some(incoming) = cotangents.get(key) {
                    updates.push((slot, Arc::clone(incoming)));
                }

                true
            });
        }

        for (slot, incoming) in updates {
            let next = {
                let current = slot.lock().unwrap();
                match current.as_ref() {
                    Some(existing) => Arc::new(existing.as_ref().add(incoming.as_ref(), backend)?),
                    None => incoming,
                }
            };
            staged.push((slot, next));
        }

        for (slot, next) in staged {
            *slot.lock().unwrap() = Some(next);
        }

        Ok(())
    }
}

/// Eager tensor with reverse-mode autodiff over concrete tensor values.
///
/// This executes each primitive immediately and records a lightweight reverse
/// DAG for `backward()`. Gradients accumulate across repeated `backward()`
/// calls until they are cleared explicitly.
///
/// # Examples
///
/// ```
/// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
///
/// let ctx = EagerContext::with_backend(CpuBackend::new());
/// let x = EagerTensor::requires_grad_in(Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]), ctx);
/// let loss = (&x * &x).reduce_sum(&[0]).unwrap();
/// let _cotangents = loss.backward().unwrap();
/// let loss = (&x * &x).reduce_sum(&[0]).unwrap();
/// let _cotangents = loss.backward().unwrap();
///
/// assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[4.0, 8.0, 12.0]);
/// x.clear_grad();
///
/// assert!(x.grad().is_none());
/// ```
#[derive(Clone)]
pub struct EagerTensor<B: TensorBackend = CpuBackend> {
    pub(crate) data: Arc<Tensor>,
    pub(crate) key: GlobalValKey<StdTensorOp>,
    pub(crate) grad_node: Option<Arc<GradNode<StdTensorOp>>>,
    pub(crate) requires_grad: bool,
    grad_slot: GradSlot,
    pub(crate) metadata_scopes: Vec<Arc<MetadataScope>>,
    pub(crate) ctx: Arc<EagerContext<B>>,
}

impl<B: TensorBackend> std::ops::Add for &EagerTensor<B> {
    type Output = EagerTensor<B>;

    fn add(self, rhs: &EagerTensor<B>) -> Self::Output {
        EagerTensor::add(self, rhs).unwrap_or_else(|err| panic!("eager add failed: {}", err))
    }
}

impl<B: TensorBackend> std::ops::Mul for &EagerTensor<B> {
    type Output = EagerTensor<B>;

    fn mul(self, rhs: &EagerTensor<B>) -> Self::Output {
        EagerTensor::mul(self, rhs).unwrap_or_else(|err| panic!("eager mul failed: {}", err))
    }
}

impl<B: TensorBackend> std::ops::Neg for &EagerTensor<B> {
    type Output = EagerTensor<B>;

    fn neg(self) -> Self::Output {
        EagerTensor::neg(self).unwrap_or_else(|err| panic!("eager neg failed: {}", err))
    }
}

impl<B: TensorBackend> EagerTensor<B> {
    /// Create an untracked eager tensor inside an existing eager context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]), ctx);
    ///
    /// assert_eq!(x.data().as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    /// ```
    pub fn from_tensor_in(tensor: Tensor, ctx: Arc<EagerContext<B>>) -> Self {
        Self::new_leaf(ctx, tensor, false)
    }

    /// Create a tracked eager leaf inside an existing eager context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_backend(CpuBackend::new());
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]), ctx);
    ///
    /// assert!(x.grad().is_none());
    /// ```
    pub fn requires_grad_in(tensor: Tensor, ctx: Arc<EagerContext<B>>) -> Self {
        Self::new_leaf(ctx, tensor, true)
    }

    pub(crate) fn new_leaf(ctx: Arc<EagerContext<B>>, tensor: Tensor, requires_grad: bool) -> Self {
        let key = eager_val_key();
        let metadata_scope =
            register_scoped_value_metadata(key.clone(), tensor_meta_from_tensor(&tensor));
        let grad_slot = Arc::new(Mutex::new(None));
        if requires_grad {
            ctx.register_grad_slot(&key, &grad_slot);
        }

        Self {
            data: Arc::new(tensor),
            key,
            grad_node: None,
            requires_grad,
            grad_slot,
            metadata_scopes: metadata_scopes_for_scope(metadata_scope),
            ctx,
        }
    }

    pub(crate) fn new_result(
        ctx: Arc<EagerContext<B>>,
        key: GlobalValKey<StdTensorOp>,
        tensor: Tensor,
        requires_grad: bool,
        grad_node: Option<Arc<GradNode<StdTensorOp>>>,
        metadata_scopes: Vec<Arc<MetadataScope>>,
    ) -> Self {
        let grad_slot = Arc::new(Mutex::new(None));
        if requires_grad {
            ctx.register_grad_slot(&key, &grad_slot);
        }

        Self {
            data: Arc::new(tensor),
            key,
            grad_node,
            requires_grad,
            grad_slot,
            metadata_scopes,
            ctx,
        }
    }

    /// Detach this tensor from the reverse graph.
    ///
    /// The returned tensor keeps the concrete value but no longer contributes
    /// gradients to the original graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_backend(CpuBackend::new());
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]), ctx);
    /// let y = x.detach();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    /// assert!(y.grad().is_none());
    /// ```
    pub fn detach(&self) -> Self {
        Self::new_leaf(self.ctx.clone(), self.data.as_ref().clone(), false)
    }

    /// Detach this tensor from its graph and re-register it in a different
    /// context as an untracked leaf.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx_a = EagerContext::with_backend(CpuBackend::new());
    /// let ctx_b = EagerContext::with_backend(CpuBackend::new());
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]), ctx_a);
    /// let d = x.detach_into(&ctx_b);
    ///
    /// assert!(!d.tracks_grad());
    /// assert_eq!(d.ctx_id(), ctx_b.id());
    /// ```
    pub fn detach_into(&self, ctx: &Arc<EagerContext<B>>) -> Self {
        Self::new_leaf(Arc::clone(ctx), self.data.as_ref().clone(), false)
    }

    /// Borrow the concrete tensor value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec(vec![1], vec![3.0_f64]), ctx);
    /// assert_eq!(x.data().as_slice::<f64>().unwrap(), &[3.0]);
    /// ```
    pub fn data(&self) -> &Tensor {
        self.data.as_ref()
    }

    /// Return the accumulated gradient currently stored for this tensor.
    ///
    /// The stored gradient accumulates across repeated `backward()` calls
    /// until it is cleared explicitly.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_backend(CpuBackend::new());
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]), ctx);
    /// let loss = x.exp().unwrap().reduce_sum(&[0]).unwrap();
    /// let _cotangents = loss.backward().unwrap();
    ///
    /// let grad = x.grad().unwrap();
    /// assert_eq!(grad.shape(), &[2]);
    /// ```
    pub fn grad(&self) -> Option<Arc<Tensor>> {
        self.grad_slot.lock().unwrap().clone()
    }

    /// Clear the accumulated gradient stored for this tensor.
    ///
    /// This only affects this tensor's gradient slot. Other tensors in the
    /// same context retain their gradients until they are cleared explicitly or
    /// overwritten by later accumulation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_backend(CpuBackend::new());
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]), ctx.clone());
    /// let y = EagerTensor::requires_grad_in(Tensor::from_vec(vec![3], vec![4.0_f64, 5.0, 6.0]), ctx);
    /// let loss = (&x * &y).reduce_sum(&[0]).unwrap();
    /// let _ = loss.backward().unwrap();
    ///
    /// x.clear_grad();
    ///
    /// assert!(x.grad().is_none());
    /// assert!(y.grad().is_some());
    /// ```
    pub fn clear_grad(&self) {
        *self.grad_slot.lock().unwrap() = None;
    }

    /// Report whether this tensor participates in gradient tracking.
    ///
    /// Tracked tensors keep a gradient slot in their eager context; untracked
    /// tensors and detached tensors do not.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_backend(CpuBackend::new());
    /// let plain = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]), ctx.clone());
    /// let tracked = EagerTensor::requires_grad_in(Tensor::from_vec(vec![2], vec![3.0_f64, 4.0]), ctx.clone());
    /// let detached = tracked.detach();
    ///
    /// assert!(!plain.tracks_grad());
    /// assert!(tracked.tracks_grad());
    /// assert!(!detached.tracks_grad());
    /// ```
    pub fn tracks_grad(&self) -> bool {
        self.requires_grad
    }

    /// Return the opaque identifier of the context this tensor belongs to.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec(vec![1], vec![1.0_f64]), ctx.clone());
    ///
    /// assert_eq!(x.ctx_id(), ctx.id());
    /// ```
    pub fn ctx_id(&self) -> ContextId {
        self.ctx.id()
    }

    /// Check whether two tensors belong to the same eager context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec(vec![1], vec![1.0_f64]), ctx.clone());
    /// let y = EagerTensor::from_tensor_in(Tensor::from_vec(vec![1], vec![2.0_f64]), ctx);
    ///
    /// assert!(x.same_context(&y));
    /// ```
    pub fn same_context(&self, other: &Self) -> bool {
        self.ctx_id() == other.ctx_id()
    }

    /// Run reverse-mode AD from this scalar output.
    ///
    /// Returns the full cotangent map produced by the reverse pass and also
    /// accumulates into `grad()` for tracked eager tensors reachable from this
    /// output.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_backend(CpuBackend::new());
    /// let x = EagerTensor::requires_grad_in(Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]), ctx);
    /// let loss = (&x + &x).reduce_sum(&[0]).unwrap();
    /// let _cotangents = loss.backward().unwrap();
    /// let loss = (&x + &x).reduce_sum(&[0]).unwrap();
    /// let _cotangents = loss.backward().unwrap();
    ///
    /// assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[4.0, 4.0, 4.0]);
    /// ```
    pub fn backward(&self) -> Result<HashMap<GlobalValKey<StdTensorOp>, Arc<Tensor>>> {
        if !self.data.shape().is_empty() {
            return Err(Error::NonScalarGrad {
                shape: self.data.shape().to_vec(),
            });
        }

        let sorted = topo_sort_grad_dag(&self.grad_node);
        let mut backend = self.ctx.backend.lock().unwrap();
        let seed = Arc::new(one_like_tensor(self.data.as_ref(), &mut *backend));
        let mut callbacks = TenferroBackwardCallbacks {
            backend: &mut *backend,
            metadata_scopes: self.metadata_scopes.clone(),
        };
        let mut ad_ctx = ShapeGuardContext::with_global_metadata();
        let cotangents = try_backward_dag(&sorted, &self.key, seed, &mut callbacks, &mut ad_ctx)?;
        self.ctx.store_grads(&cotangents, &mut *backend)?;
        Ok(cotangents)
    }
}

pub(crate) struct TenferroBackwardCallbacks<'a, B: TensorBackend> {
    backend: &'a mut B,
    metadata_scopes: Vec<Arc<MetadataScope>>,
}

impl<B: TensorBackend> BackwardCallbacks<StdTensorOp> for TenferroBackwardCallbacks<'_, B> {
    fn execute_forward(
        &mut self,
        fragment: &Fragment<StdTensorOp>,
        initial_data: &HashMap<GlobalValKey<StdTensorOp>, Arc<Tensor>>,
    ) -> HashMap<GlobalValKey<StdTensorOp>, Arc<Tensor>> {
        let mut all_values = initial_data.clone();

        for &input_id in fragment.inputs() {
            let key = fragment.vals()[input_id].key.clone();
            all_values.entry(key.clone()).or_insert_with(|| {
                let GlobalValKey::Input(tangent_key) = &key else {
                    panic!("expected input key for eager forward: {:?}", key);
                };
                let tenferro_ops::input_key::TensorInputKey::Tangent { of, .. } = tangent_key
                else {
                    panic!("missing concrete eager value for {:?}", key);
                };
                let base_key = GlobalValKey::Input((**of).clone());
                let base = initial_data
                    .get(&base_key)
                    .unwrap_or_else(|| panic!("missing base eager value for {:?}", base_key));
                Arc::new(zero_like_tensor(base.as_ref(), self.backend))
            });
        }

        for op_node in fragment.ops() {
            let resolved_inputs: Vec<&Tensor> = op_node
                .inputs
                .iter()
                .map(|input| match input {
                    ValRef::Local(local_id) => {
                        let key = &fragment.vals()[*local_id].key;
                        all_values
                            .get(key)
                            .unwrap_or_else(|| panic!("missing eager value for local {:?}", key))
                            .as_ref()
                    }
                    ValRef::External(key) => all_values
                        .get(key)
                        .unwrap_or_else(|| panic!("missing eager value for external {:?}", key))
                        .as_ref(),
                })
                .collect();
            let outputs = exec_op_on_tensors(&op_node.op, &resolved_inputs, self.backend)
                .unwrap_or_else(|err| {
                    panic!("eager forward exec failed for {:?}: {}", op_node.op, err)
                });

            for (output_id, output) in op_node.outputs.iter().zip(outputs.into_iter()) {
                let key = fragment.vals()[*output_id].key.clone();
                all_values.insert(key, Arc::new(output));
            }
        }

        let metadata_scope = register_scoped_fragment_metadata(
            fragment,
            fragment.inputs().iter().map(|input_id| {
                let key = fragment.vals()[*input_id].key.clone();
                let meta = tensor_meta_from_tensor(
                    all_values
                        .get(&key)
                        .unwrap_or_else(|| {
                            panic!("missing eager value for fragment input {:?}", key)
                        })
                        .as_ref(),
                );
                (key, meta)
            }),
        );
        push_metadata_scope(&mut self.metadata_scopes, Arc::new(metadata_scope));

        all_values
    }

    fn eager_transpose(
        &mut self,
        linear: &LinearFragment<StdTensorOp>,
        cotangent_out: &[Option<Arc<Tensor>>],
        external_data: &HashMap<GlobalValKey<StdTensorOp>, Arc<Tensor>>,
        ctx: &mut ShapeGuardContext,
    ) -> Vec<Option<Arc<Tensor>>> {
        let mut emitter = EagerEmitter::new(self.backend);
        emitter.external_data = external_data.clone();
        let cotangent_seed_ids = cotangent_out
            .iter()
            .map(|maybe_seed| {
                maybe_seed
                    .as_ref()
                    .map(|seed| emitter.push_tensor(Arc::clone(seed)))
            })
            .collect::<Vec<_>>();

        ctx.refresh_global_metadata();
        tidu::eager_transpose_fragment(linear, &mut emitter, &cotangent_seed_ids, ctx)
            .into_iter()
            .map(|maybe_id| maybe_id.map(|id| emitter.tensor(id)))
            .collect()
    }

    fn try_eager_transpose(
        &mut self,
        linear: &LinearFragment<StdTensorOp>,
        cotangent_out: &[Option<Arc<Tensor>>],
        external_data: &HashMap<GlobalValKey<StdTensorOp>, Arc<Tensor>>,
        ctx: &mut ShapeGuardContext,
    ) -> chainrules_core::ADRuleResult<Vec<Option<Arc<Tensor>>>> {
        let mut emitter = EagerEmitter::new(self.backend);
        emitter.external_data = external_data.clone();
        let cotangent_seed_ids = cotangent_out
            .iter()
            .map(|maybe_seed| {
                maybe_seed
                    .as_ref()
                    .map(|seed| emitter.push_tensor(Arc::clone(seed)))
            })
            .collect::<Vec<_>>();

        ctx.refresh_global_metadata();
        tidu::try_eager_transpose_fragment(linear, &mut emitter, &cotangent_seed_ids, ctx).map(
            |cotangent_ids| {
                cotangent_ids
                    .into_iter()
                    .map(|maybe_id| maybe_id.map(|id| emitter.tensor(id)))
                    .collect()
            },
        )
    }

    fn add_operands(&mut self, a: &Arc<Tensor>, b: &Arc<Tensor>) -> Arc<Tensor> {
        Arc::new(
            a.as_ref()
                .add(b.as_ref(), self.backend)
                .unwrap_or_else(|err| panic!("eager cotangent add failed: {}", err)),
        )
    }
}

pub(crate) fn eager_val_key() -> GlobalValKey<StdTensorOp> {
    GlobalValKey::Input(next_input_key())
}

pub(crate) struct EagerTensorKeySource;

impl tidu::EagerKeySource<StdTensorOp> for EagerTensorKeySource {
    fn fresh_input_key(&mut self) -> TensorInputKey {
        next_input_key()
    }
}

pub(crate) fn eager_value<B: TensorBackend>(tensor: &EagerTensor<B>) -> EagerValue<StdTensorOp> {
    EagerValue {
        key: tensor.key.clone(),
        node: tensor.grad_node.clone(),
        requires_grad: tensor.requires_grad,
        data: Arc::clone(&tensor.data),
    }
}

pub(crate) struct RecordedEagerOutputs {
    pub(crate) traces: Vec<EagerOutput<StdTensorOp>>,
    pub(crate) metadata_scope: Arc<MetadataScope>,
}

pub(crate) fn record_eager_outputs<B: TensorBackend>(
    op: &StdTensorOp,
    outputs: &[Arc<Tensor>],
    inputs: &[&EagerTensor<B>],
) -> RecordedEagerOutputs {
    let input_values: Vec<_> = inputs.iter().map(|tensor| eager_value(*tensor)).collect();
    let mut key_source = EagerTensorKeySource;
    let traces = tidu::record_eager_op(&mut key_source, op.clone(), &input_values, outputs);

    let mut registrations = Vec::new();
    for trace in &traces {
        if let Some(output) = outputs.get(trace.output_slot) {
            registrations.push((trace.key.clone(), tensor_meta_from_tensor(output.as_ref())));
        }
    }

    if let Some(node) = traces.iter().find_map(|trace| trace.node.as_ref()) {
        for (key, value) in node.saved_data() {
            registrations.push((key.clone(), tensor_meta_from_tensor(value.as_ref())));
        }
    }

    RecordedEagerOutputs {
        traces,
        metadata_scope: Arc::new(register_scoped_metadata_batch(registrations)),
    }
}

pub(crate) fn exec_single_output<B: TensorBackend>(
    op: &StdTensorOp,
    inputs: &[&Tensor],
    ctx: &EagerContext<B>,
) -> Result<Tensor> {
    let mut backend = ctx.backend.lock().unwrap();
    let mut outputs = exec_op_on_tensors(op, inputs, &mut *backend)?;
    if outputs.len() != 1 {
        return Err(Error::Internal(format!(
            "expected one eager output for {:?}, got {}",
            op,
            outputs.len()
        )));
    }
    Ok(outputs.remove(0))
}

pub(crate) fn zero_like_tensor<B: TensorBackend>(input: &Tensor, _backend: &mut B) -> Tensor {
    match input {
        Tensor::F32(tensor) => Tensor::F32(TypedTensor::zeros(tensor.shape.clone())),
        Tensor::F64(tensor) => Tensor::F64(TypedTensor::zeros(tensor.shape.clone())),
        Tensor::I64(tensor) => Tensor::I64(TypedTensor::zeros(tensor.shape.clone())),
        Tensor::C32(tensor) => Tensor::C32(TypedTensor::zeros(tensor.shape.clone())),
        Tensor::C64(tensor) => Tensor::C64(TypedTensor::zeros(tensor.shape.clone())),
    }
}

pub(crate) fn one_like_tensor<B: TensorBackend>(input: &Tensor, backend: &mut B) -> Tensor {
    let zero = zero_like_tensor(input, backend);
    backend
        .exp(&zero)
        .unwrap_or_else(|err| panic!("one_like exp failed: {}", err))
}
