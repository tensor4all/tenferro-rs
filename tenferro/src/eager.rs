use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use computegraph::fragment::Fragment;
use computegraph::{GlobalOpKey, GlobalValKey, OpMode, ValRef};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::ShapeGuardContext;
use tenferro_tensor::cpu::CpuBackend;
use tenferro_tensor::{DotGeneralConfig, Tensor, TensorBackend};
use tidu::{
    backward_dag, topo_sort_grad_dag, BackwardCallbacks, GradEdge, GradNode, LinearFragment,
};

use crate::eager_emitter::EagerEmitter;
use crate::eager_exec::exec_op_on_tensors;
use crate::error::{Error, Result};
use crate::traced::next_input_key;

type GradSlot = Rc<RefCell<Option<Arc<Tensor>>>>;
type WeakGradSlot = Weak<RefCell<Option<Arc<Tensor>>>>;

struct EagerContext<B: TensorBackend> {
    backend: RefCell<B>,
    grad_slots: RefCell<HashMap<GlobalValKey<StdTensorOp>, WeakGradSlot>>,
}

impl<B: TensorBackend> EagerContext<B> {
    fn new(backend: B) -> Self {
        Self {
            backend: RefCell::new(backend),
            grad_slots: RefCell::new(HashMap::new()),
        }
    }

    fn register_grad_slot(&self, key: &GlobalValKey<StdTensorOp>, slot: &GradSlot) {
        self.grad_slots
            .borrow_mut()
            .insert(key.clone(), Rc::downgrade(slot));
    }

    fn absorb_from(&self, other: &Self) {
        let other_slots = other.grad_slots.borrow();
        let mut slots = self.grad_slots.borrow_mut();
        for (key, slot) in other_slots.iter() {
            slots.entry(key.clone()).or_insert_with(|| slot.clone());
        }
    }

    fn clear_grads(&self) {
        self.grad_slots.borrow_mut().retain(|_, slot| {
            if let Some(slot) = slot.upgrade() {
                *slot.borrow_mut() = None;
                true
            } else {
                false
            }
        });
    }

    fn store_grads(&self, cotangents: &HashMap<GlobalValKey<StdTensorOp>, Arc<Tensor>>) {
        self.grad_slots.borrow_mut().retain(|key, slot| {
            let Some(slot) = slot.upgrade() else {
                return false;
            };
            let value = cotangents.get(key).cloned();
            *slot.borrow_mut() = value;
            true
        });
    }
}

/// Eager tensor with reverse-mode autodiff over concrete tensor values.
///
/// This executes each primitive immediately and records a lightweight reverse
/// DAG for `backward()`.
///
/// # Examples
///
/// ```
/// use tenferro::{EagerTensor, Tensor};
///
/// let x = EagerTensor::requires_grad(Tensor::new(vec![3], vec![1.0_f64, 2.0, 3.0]));
/// let loss = (&x * &x).reduce_sum(&[0]);
/// let _cotangents = loss.backward().unwrap();
///
/// assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0]);
/// ```
#[derive(Clone)]
pub struct EagerTensor<B: TensorBackend = CpuBackend> {
    data: Arc<Tensor>,
    key: GlobalValKey<StdTensorOp>,
    grad_node: Option<Arc<GradNode<StdTensorOp>>>,
    requires_grad: bool,
    grad_slot: GradSlot,
    ctx: Rc<EagerContext<B>>,
}

impl<B: TensorBackend> std::ops::Add for &EagerTensor<B> {
    type Output = EagerTensor<B>;

    fn add(self, rhs: &EagerTensor<B>) -> Self::Output {
        EagerTensor::add(self, rhs)
    }
}

impl<B: TensorBackend> std::ops::Mul for &EagerTensor<B> {
    type Output = EagerTensor<B>;

    fn mul(self, rhs: &EagerTensor<B>) -> Self::Output {
        EagerTensor::mul(self, rhs)
    }
}

impl<B: TensorBackend> std::ops::Neg for &EagerTensor<B> {
    type Output = EagerTensor<B>;

    fn neg(self) -> Self::Output {
        EagerTensor::neg(self)
    }
}

impl EagerTensor<CpuBackend> {
    /// Create an untracked eager tensor on the default CPU backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![2], vec![1.0_f64, 2.0]));
    /// assert_eq!(x.data().as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    /// assert!(x.grad().is_none());
    /// ```
    pub fn from_tensor(tensor: Tensor) -> Self {
        Self::new_leaf(Rc::new(EagerContext::new(CpuBackend::new())), tensor, false)
    }

    /// Create a tracked eager leaf on the default CPU backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::requires_grad(Tensor::new(vec![2], vec![1.0_f64, 2.0]));
    /// assert!(x.grad().is_none());
    /// ```
    pub fn requires_grad(tensor: Tensor) -> Self {
        Self::new_leaf(Rc::new(EagerContext::new(CpuBackend::new())), tensor, true)
    }
}

impl<B: TensorBackend> EagerTensor<B> {
    fn new_leaf(ctx: Rc<EagerContext<B>>, tensor: Tensor, requires_grad: bool) -> Self {
        let key = eager_val_key();
        let grad_slot = Rc::new(RefCell::new(None));
        if requires_grad {
            ctx.register_grad_slot(&key, &grad_slot);
        }

        Self {
            data: Arc::new(tensor),
            key,
            grad_node: None,
            requires_grad,
            grad_slot,
            ctx,
        }
    }

    fn new_result(
        ctx: Rc<EagerContext<B>>,
        key: GlobalValKey<StdTensorOp>,
        tensor: Tensor,
        requires_grad: bool,
        grad_node: Option<Arc<GradNode<StdTensorOp>>>,
    ) -> Self {
        let grad_slot = Rc::new(RefCell::new(None));
        if requires_grad {
            ctx.register_grad_slot(&key, &grad_slot);
        }

        Self {
            data: Arc::new(tensor),
            key,
            grad_node,
            requires_grad,
            grad_slot,
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
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::requires_grad(Tensor::new(vec![2], vec![1.0_f64, 2.0]));
    /// let y = x.detach();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    /// assert!(y.grad().is_none());
    /// ```
    pub fn detach(&self) -> Self {
        Self::new_leaf(self.ctx.clone(), self.data.as_ref().clone(), false)
    }

    /// Borrow the concrete tensor value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![1], vec![3.0_f64]));
    /// assert_eq!(x.data().as_slice::<f64>().unwrap(), &[3.0]);
    /// ```
    pub fn data(&self) -> &Tensor {
        self.data.as_ref()
    }

    /// Return the accumulated gradient from the last `backward()` call.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::requires_grad(Tensor::new(vec![2], vec![1.0_f64, 2.0]));
    /// let loss = x.exp().reduce_sum(&[0]);
    /// let _cotangents = loss.backward().unwrap();
    ///
    /// let grad = x.grad().unwrap();
    /// assert_eq!(grad.shape(), &[2]);
    /// ```
    pub fn grad(&self) -> Option<Arc<Tensor>> {
        self.grad_slot.borrow().clone()
    }

    /// Run reverse-mode AD from this scalar output.
    ///
    /// Returns the full cotangent map produced by the reverse pass and also
    /// populates `grad()` for tracked eager tensors reachable from this output.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::requires_grad(Tensor::new(vec![3], vec![1.0_f64, 2.0, 3.0]));
    /// let loss = (&x + &x).reduce_sum(&[0]);
    /// let _cotangents = loss.backward().unwrap();
    ///
    /// assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[2.0, 2.0, 2.0]);
    /// ```
    pub fn backward(&self) -> Result<HashMap<GlobalValKey<StdTensorOp>, Arc<Tensor>>> {
        if !self.data.shape().is_empty() {
            return Err(Error::NonScalarGrad {
                shape: self.data.shape().to_vec(),
            });
        }

        self.ctx.clear_grads();

        let sorted = topo_sort_grad_dag(&self.grad_node);
        let mut backend = self.ctx.backend.borrow_mut();
        let seed = Arc::new(one_like_tensor(self.data.as_ref(), &mut *backend));
        let mut callbacks = TenferroBackwardCallbacks {
            backend: &mut *backend,
        };
        let mut ad_ctx = ShapeGuardContext::default();
        let cotangents = backward_dag(&sorted, &self.key, seed, &mut callbacks, &mut ad_ctx);
        self.ctx.store_grads(&cotangents);
        Ok(cotangents)
    }

    /// Elementwise addition.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![2], vec![1.0_f64, 2.0]));
    /// let y = EagerTensor::from_tensor(Tensor::new(vec![2], vec![3.0_f64, 4.0]));
    /// let z = x.add(&y);
    ///
    /// assert_eq!(z.data().as_slice::<f64>().unwrap(), &[4.0, 6.0]);
    /// ```
    pub fn add(&self, other: &Self) -> Self {
        self.binary_op(other, StdTensorOp::Add)
    }

    /// Elementwise multiplication.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![2], vec![1.0_f64, 2.0]));
    /// let y = EagerTensor::from_tensor(Tensor::new(vec![2], vec![3.0_f64, 4.0]));
    /// let z = x.mul(&y);
    ///
    /// assert_eq!(z.data().as_slice::<f64>().unwrap(), &[3.0, 8.0]);
    /// ```
    pub fn mul(&self, other: &Self) -> Self {
        self.binary_op(other, StdTensorOp::Mul)
    }

    /// Negate the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![2], vec![1.0_f64, -2.0]));
    /// let y = x.neg();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[-1.0, 2.0]);
    /// ```
    pub fn neg(&self) -> Self {
        self.unary_op(StdTensorOp::Neg)
    }

    /// Elementwise exponential.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![1], vec![0.0_f64]));
    /// let y = x.exp();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[1.0]);
    /// ```
    pub fn exp(&self) -> Self {
        self.unary_op(StdTensorOp::Exp)
    }

    /// Reduce sum over the requested axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]));
    /// let y = x.reduce_sum(&[0, 1]);
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[10.0]);
    /// ```
    pub fn reduce_sum(&self, axes: &[usize]) -> Self {
        self.unary_op(StdTensorOp::ReduceSum {
            axes: axes.to_vec(),
            input_shape: DimExpr::from_concrete(self.data.shape()),
        })
    }

    /// Execute a dot-general contraction eagerly.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{DotGeneralConfig, EagerTensor, Tensor};
    ///
    /// let a = EagerTensor::from_tensor(Tensor::new(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]));
    /// let b = EagerTensor::from_tensor(Tensor::new(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]));
    /// let c = a.dot_general(&b, DotGeneralConfig {
    ///     lhs_contracting_dims: vec![1],
    ///     rhs_contracting_dims: vec![0],
    ///     lhs_batch_dims: vec![],
    ///     rhs_batch_dims: vec![],
    ///     lhs_rank: 2,
    ///     rhs_rank: 2,
    /// });
    ///
    /// assert_eq!(c.data().shape(), &[2, 2]);
    /// ```
    pub fn dot_general(&self, other: &Self, config: DotGeneralConfig) -> Self {
        config
            .validate_ranks(self.data.shape().len(), other.data.shape().len())
            .expect("DotGeneral config rank validation failed");
        config
            .validate_dims()
            .expect("DotGeneral config dimension validation failed");
        self.binary_op(other, StdTensorOp::DotGeneral(config))
    }

    fn unary_op(&self, op: StdTensorOp) -> Self {
        let output = exec_single_output(&op, &[self.data.as_ref()], &self.ctx);
        let result_key = eager_val_key();
        let input_aliases = vec![eager_val_key()];
        let grad_node = self.requires_grad.then(|| {
            Arc::new(GradNode {
                op: op.clone(),
                primal_in_keys: input_aliases.clone(),
                primal_out_keys: vec![result_key.clone()],
                saved_data: saved_forward_values(
                    &op,
                    &input_aliases,
                    &[Arc::clone(&self.data)],
                    Arc::new(output.clone()),
                ),
                input_edges: vec![GradEdge {
                    node: self.grad_node.clone(),
                    key: self.key.clone(),
                    requires_grad: self.requires_grad,
                }],
                output_idx: 0,
            })
        });
        Self::new_result(
            self.ctx.clone(),
            result_key,
            output,
            self.requires_grad,
            grad_node,
        )
    }

    fn binary_op(&self, other: &Self, op: StdTensorOp) -> Self {
        if !Rc::ptr_eq(&self.ctx, &other.ctx) {
            self.ctx.absorb_from(&other.ctx);
        }

        let output = exec_single_output(&op, &[self.data.as_ref(), other.data.as_ref()], &self.ctx);
        let requires_grad = self.requires_grad || other.requires_grad;
        let result_key = eager_val_key();
        let input_aliases = vec![eager_val_key(), eager_val_key()];
        let grad_node = requires_grad.then(|| {
            Arc::new(GradNode {
                op: op.clone(),
                primal_in_keys: input_aliases.clone(),
                primal_out_keys: vec![result_key.clone()],
                saved_data: saved_forward_values(
                    &op,
                    &input_aliases,
                    &[Arc::clone(&self.data), Arc::clone(&other.data)],
                    Arc::new(output.clone()),
                ),
                input_edges: vec![
                    GradEdge {
                        node: self.grad_node.clone(),
                        key: self.key.clone(),
                        requires_grad: self.requires_grad,
                    },
                    GradEdge {
                        node: other.grad_node.clone(),
                        key: other.key.clone(),
                        requires_grad: other.requires_grad,
                    },
                ],
                output_idx: 0,
            })
        });
        Self::new_result(
            self.ctx.clone(),
            result_key,
            output,
            requires_grad,
            grad_node,
        )
    }
}

struct TenferroBackwardCallbacks<'a, B: TensorBackend> {
    backend: &'a mut B,
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

        tidu::eager_transpose_fragment(linear, &mut emitter, &cotangent_seed_ids, ctx)
            .into_iter()
            .map(|maybe_id| maybe_id.map(|id| emitter.tensor(id)))
            .collect()
    }

    fn add_operands(&mut self, a: &Arc<Tensor>, b: &Arc<Tensor>) -> Arc<Tensor> {
        Arc::new(
            a.as_ref()
                .add(b.as_ref(), self.backend)
                .unwrap_or_else(|err| panic!("eager cotangent add failed: {}", err)),
        )
    }
}

fn eager_val_key() -> GlobalValKey<StdTensorOp> {
    GlobalValKey::Input(next_input_key())
}

fn saved_forward_values(
    op: &StdTensorOp,
    input_keys: &[GlobalValKey<StdTensorOp>],
    inputs: &[Arc<Tensor>],
    output: Arc<Tensor>,
) -> HashMap<GlobalValKey<StdTensorOp>, Arc<Tensor>> {
    let mut saved = HashMap::with_capacity(input_keys.len() + 1);
    for (key, value) in input_keys.iter().zip(inputs.iter()) {
        saved.insert(key.clone(), Arc::clone(value));
    }
    saved.insert(derived_output_key(op, input_keys), output);
    saved
}

fn derived_output_key(
    op: &StdTensorOp,
    input_keys: &[GlobalValKey<StdTensorOp>],
) -> GlobalValKey<StdTensorOp> {
    GlobalValKey::Derived {
        op: GlobalOpKey {
            primitive: op.clone(),
            inputs: input_keys.to_vec(),
            mode: OpMode::Primal,
        },
        output_slot: 0,
    }
}

fn exec_single_output<B: TensorBackend>(
    op: &StdTensorOp,
    inputs: &[&Tensor],
    ctx: &EagerContext<B>,
) -> Tensor {
    let mut backend = ctx.backend.borrow_mut();
    let mut outputs = exec_op_on_tensors(op, inputs, &mut *backend)
        .unwrap_or_else(|err| panic!("eager exec failed for {:?}: {}", op, err));
    assert_eq!(
        outputs.len(),
        1,
        "expected one eager output for {:?}, got {}",
        op,
        outputs.len()
    );
    outputs.remove(0)
}

fn zero_like_tensor<B: TensorBackend>(input: &Tensor, backend: &mut B) -> Tensor {
    let neg = input
        .neg(backend)
        .unwrap_or_else(|err| panic!("zero_like neg failed: {}", err));
    input
        .add(&neg, backend)
        .unwrap_or_else(|err| panic!("zero_like add failed: {}", err))
}

fn one_like_tensor<B: TensorBackend>(input: &Tensor, backend: &mut B) -> Tensor {
    let zero = zero_like_tensor(input, backend);
    backend
        .exp(&zero)
        .unwrap_or_else(|err| panic!("one_like exp failed: {}", err))
}
