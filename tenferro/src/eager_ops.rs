use std::rc::Rc;
use std::sync::Arc;

use tidu::{GradEdge, GradNode};

use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{DotGeneralConfig, Tensor, TensorBackend};

use crate::eager::{
    eager_val_key, exec_single_output, saved_forward_values, saved_forward_values_multi,
    EagerTensor,
};
use crate::eager_exec::exec_op_on_tensors;
use crate::error::{Error, Result};

impl<B: TensorBackend> EagerTensor<B> {
    /// Elementwise addition.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![2], vec![1.0_f64, 2.0]));
    /// let y = EagerTensor::from_tensor(Tensor::new(vec![2], vec![3.0_f64, 4.0]));
    /// let z = x.add(&y).unwrap();
    ///
    /// assert_eq!(z.data().as_slice::<f64>().unwrap(), &[4.0, 6.0]);
    /// ```
    pub fn add(&self, other: &Self) -> Result<Self> {
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
    /// let z = x.mul(&y).unwrap();
    ///
    /// assert_eq!(z.data().as_slice::<f64>().unwrap(), &[3.0, 8.0]);
    /// ```
    pub fn mul(&self, other: &Self) -> Result<Self> {
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
    /// let y = x.neg().unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[-1.0, 2.0]);
    /// ```
    pub fn neg(&self) -> Result<Self> {
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
    /// let y = x.exp().unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[1.0]);
    /// ```
    pub fn exp(&self) -> Result<Self> {
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
    /// let y = x.reduce_sum(&[0, 1]).unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[10.0]);
    /// ```
    pub fn reduce_sum(&self, axes: &[usize]) -> Result<Self> {
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
    /// }).unwrap();
    ///
    /// assert_eq!(c.data().shape(), &[2, 2]);
    /// ```
    pub fn dot_general(&self, other: &Self, config: DotGeneralConfig) -> Result<Self> {
        self.binary_op(other, StdTensorOp::DotGeneral(config))
    }

    pub(crate) fn unary_op(&self, op: StdTensorOp) -> Result<Self> {
        let output = exec_single_output(&op, &[self.data.as_ref()], &self.ctx)?;
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
        Ok(Self::new_result(
            Rc::clone(&self.ctx),
            result_key,
            output,
            self.requires_grad,
            grad_node,
        ))
    }

    pub(crate) fn binary_op(&self, other: &Self, op: StdTensorOp) -> Result<Self> {
        Self::nary_op(&[self, other], op)
    }

    #[allow(dead_code)]
    pub(crate) fn multi_output_unary_op(
        &self,
        op: StdTensorOp,
        num_outputs: usize,
    ) -> Result<Vec<Self>> {
        let outputs = {
            let mut backend = self.ctx.backend.borrow_mut();
            exec_op_on_tensors(&op, &[self.data.as_ref()], &mut *backend)?
        };
        if outputs.len() != num_outputs {
            return Err(Error::Internal(format!(
                "expected {} eager outputs for {:?}, got {}",
                num_outputs,
                op,
                outputs.len()
            )));
        }

        let outputs: Vec<Arc<Tensor>> = outputs.into_iter().map(Arc::new).collect();
        let output_keys: Vec<_> = (0..num_outputs).map(|_| eager_val_key()).collect();
        let input_aliases = vec![eager_val_key()];
        let grad_node = self.requires_grad.then(|| {
            Arc::new(GradNode {
                op: op.clone(),
                primal_in_keys: input_aliases.clone(),
                primal_out_keys: output_keys.clone(),
                saved_data: saved_forward_values_multi(
                    &op,
                    &input_aliases,
                    &[Arc::clone(&self.data)],
                    &output_keys,
                    &outputs,
                ),
                input_edges: vec![GradEdge {
                    node: self.grad_node.clone(),
                    key: self.key.clone(),
                    requires_grad: self.requires_grad,
                }],
                output_idx: 0,
            })
        });

        Ok(output_keys
            .into_iter()
            .zip(outputs)
            .map(|(output_key, output)| {
                Self::new_result(
                    Rc::clone(&self.ctx),
                    output_key,
                    output.as_ref().clone(),
                    self.requires_grad,
                    grad_node.clone(),
                )
            })
            .collect())
    }

    #[allow(dead_code)]
    pub(crate) fn ternary_op(&self, b: &Self, c: &Self, op: StdTensorOp) -> Result<Self> {
        Self::nary_op(&[self, b, c], op)
    }

    pub(crate) fn nary_op(tensors: &[&Self], op: StdTensorOp) -> Result<Self> {
        let Some(first) = tensors.first() else {
            return Err(Error::Internal(
                "nary eager op requires at least one input tensor".to_string(),
            ));
        };

        let ctx = Rc::clone(&first.ctx);
        for tensor in tensors.iter().skip(1) {
            if !Rc::ptr_eq(&ctx, &tensor.ctx) {
                ctx.absorb_from(&tensor.ctx);
            }
        }

        let inputs: Vec<&Tensor> = tensors.iter().map(|tensor| tensor.data.as_ref()).collect();
        let output = exec_single_output(&op, &inputs, &ctx)?;
        let requires_grad = tensors.iter().any(|tensor| tensor.requires_grad);
        let result_key = eager_val_key();
        let input_aliases: Vec<_> = tensors.iter().map(|_| eager_val_key()).collect();
        let input_data: Vec<_> = tensors
            .iter()
            .map(|tensor| Arc::clone(&tensor.data))
            .collect();
        let grad_node = requires_grad.then(|| {
            Arc::new(GradNode {
                op: op.clone(),
                primal_in_keys: input_aliases.clone(),
                primal_out_keys: vec![result_key.clone()],
                saved_data: saved_forward_values(
                    &op,
                    &input_aliases,
                    &input_data,
                    Arc::new(output.clone()),
                ),
                input_edges: tensors
                    .iter()
                    .map(|tensor| GradEdge {
                        node: tensor.grad_node.clone(),
                        key: tensor.key.clone(),
                        requires_grad: tensor.requires_grad,
                    })
                    .collect(),
                output_idx: 0,
            })
        });

        Ok(Self::new_result(
            ctx,
            result_key,
            output,
            requires_grad,
            grad_node,
        ))
    }
}
