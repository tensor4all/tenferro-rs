use std::sync::Arc;

use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{
    DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig, Tensor,
    TensorBackend,
};

use crate::eager::{exec_single_output, record_eager_outputs, EagerTensor};
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
    /// let x = EagerTensor::from_tensor(Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]));
    /// let y = EagerTensor::from_tensor(Tensor::from_vec(vec![2], vec![3.0_f64, 4.0]));
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
    /// let x = EagerTensor::from_tensor(Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]));
    /// let y = EagerTensor::from_tensor(Tensor::from_vec(vec![2], vec![3.0_f64, 4.0]));
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
    /// let x = EagerTensor::from_tensor(Tensor::from_vec(vec![2], vec![1.0_f64, -2.0]));
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
    /// let x = EagerTensor::from_tensor(Tensor::from_vec(vec![1], vec![0.0_f64]));
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
    /// let x = EagerTensor::from_tensor(Tensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]));
    /// let y = x.reduce_sum(&[0, 1]).unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[10.0]);
    /// ```
    pub fn reduce_sum(&self, axes: &[usize]) -> Result<Self> {
        self.unary_op(StdTensorOp::ReduceSum {
            axes: axes.to_vec(),
        })
    }

    /// Execute a dot-general contraction eagerly.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{DotGeneralConfig, EagerTensor, Tensor};
    ///
    /// let a = EagerTensor::from_tensor(Tensor::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]));
    /// let b = EagerTensor::from_tensor(Tensor::from_vec(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]));
    /// let c = a.dot_general(&b, DotGeneralConfig {
    ///     lhs_contracting_dims: vec![1],
    ///     rhs_contracting_dims: vec![0],
    ///     lhs_batch_dims: vec![],
    ///     rhs_batch_dims: vec![],
    /// }).unwrap();
    ///
    /// assert_eq!(c.data().shape(), &[2, 2]);
    /// ```
    pub fn dot_general(&self, other: &Self, config: DotGeneralConfig) -> Result<Self> {
        self.binary_op(other, StdTensorOp::DotGeneral { config })
    }

    /// Permute tensor axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::from_vec(
    ///     vec![2, 3],
    ///     vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    /// ));
    /// let y = x.transpose(&[1, 0]).unwrap();
    ///
    /// assert_eq!(y.data().shape(), &[3, 2]);
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    /// ```
    pub fn transpose(&self, perm: &[usize]) -> Result<Self> {
        self.unary_op(StdTensorOp::Transpose {
            perm: perm.to_vec(),
        })
    }

    /// Reshape without changing element order.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::from_vec(
    ///     vec![2, 3],
    ///     vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    /// ));
    /// let y = x.reshape(&[6]).unwrap();
    ///
    /// assert_eq!(y.data().shape(), &[6]);
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// ```
    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        self.unary_op(StdTensorOp::Reshape {
            to_shape: DimExpr::from_concrete(shape),
        })
    }

    /// Slice with explicit start, limit, and stride per axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, SliceConfig, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::from_vec(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]));
    /// let y = x
    ///     .slice(SliceConfig {
    ///         starts: vec![1],
    ///         limits: vec![3],
    ///         strides: vec![1],
    ///     })
    ///     .unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[2.0, 3.0]);
    /// ```
    pub fn slice(&self, config: SliceConfig) -> Result<Self> {
        self.unary_op(StdTensorOp::Slice(config))
    }

    /// Broadcast into a larger shape with explicit dimension placement.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]));
    /// let y = x.broadcast_in_dim(&[3, 2], &[0]).unwrap();
    ///
    /// assert_eq!(y.data().shape(), &[3, 2]);
    /// ```
    pub fn broadcast_in_dim(&self, shape: &[usize], dims: &[usize]) -> Result<Self> {
        self.unary_op(StdTensorOp::BroadcastInDim {
            shape: DimExpr::from_concrete(shape),
            dims: dims.to_vec(),
        })
    }

    /// Convert the tensor to a different dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{DType, EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::from_vec(vec![2], vec![1.0_f64, -2.0]));
    /// let y = x.convert(DType::C64).unwrap();
    ///
    /// assert_eq!(y.data().dtype(), DType::C64);
    /// assert_eq!(y.data().shape(), &[2]);
    /// ```
    pub fn convert(&self, to: DType) -> Result<Self> {
        self.unary_op(StdTensorOp::Convert {
            from: self.data.dtype(),
            to,
        })
    }

    /// Pad with zeros using StableHLO-style edge and interior padding.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, PadConfig, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]));
    /// let y = x
    ///     .pad(PadConfig {
    ///         edge_padding_low: vec![1],
    ///         edge_padding_high: vec![1],
    ///         interior_padding: vec![1],
    ///     })
    ///     .unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[0.0, 1.0, 0.0, 2.0, 0.0]);
    /// ```
    pub fn pad(&self, config: PadConfig) -> Result<Self> {
        self.unary_op(StdTensorOp::Pad(config))
    }

    /// Reverse the order of elements along the requested axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::from_vec(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]));
    /// let y = x.reverse(&[0]).unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[4.0, 3.0, 2.0, 1.0]);
    /// ```
    pub fn reverse(&self, axes: &[usize]) -> Result<Self> {
        self.unary_op(StdTensorOp::Reverse {
            axes: axes.to_vec(),
        })
    }

    /// Gather slices from `self` using integer start indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, GatherConfig, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::from_vec(
    ///     vec![5],
    ///     vec![10.0_f64, 20.0, 30.0, 40.0, 50.0],
    /// ));
    /// let indices = EagerTensor::from_tensor(Tensor::from_vec(vec![3], vec![4_i64, 1, 0]));
    /// let y = x
    ///     .gather(
    ///         &indices,
    ///         GatherConfig {
    ///             offset_dims: vec![],
    ///             collapsed_slice_dims: vec![0],
    ///             start_index_map: vec![0],
    ///             index_vector_dim: 1,
    ///             slice_sizes: vec![1],
    ///         },
    ///     )
    ///     .unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[50.0, 20.0, 10.0]);
    /// ```
    pub fn gather(&self, indices: &Self, config: GatherConfig) -> Result<Self> {
        self.binary_op(indices, StdTensorOp::Gather(config))
    }

    /// Scatter updates into `self` using StableHLO scatter semantics.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, ScatterConfig, Tensor};
    ///
    /// let operand = EagerTensor::from_tensor(Tensor::from_vec(vec![4], vec![0.0_f64, 0.0, 0.0, 0.0]));
    /// let indices = EagerTensor::from_tensor(Tensor::from_vec(vec![2, 1], vec![1_i64, 3]));
    /// let updates = EagerTensor::from_tensor(Tensor::from_vec(vec![2], vec![5.0_f64, 7.0]));
    /// let result = operand
    ///     .scatter(
    ///         &indices,
    ///         &updates,
    ///         ScatterConfig {
    ///             update_window_dims: vec![],
    ///             inserted_window_dims: vec![0],
    ///             scatter_dims_to_operand_dims: vec![0],
    ///             index_vector_dim: 1,
    ///         },
    ///     )
    ///     .unwrap();
    ///
    /// assert_eq!(result.data().as_slice::<f64>().unwrap(), &[0.0, 5.0, 0.0, 7.0]);
    /// ```
    pub fn scatter(&self, indices: &Self, updates: &Self, config: ScatterConfig) -> Result<Self> {
        self.ternary_op(indices, updates, StdTensorOp::Scatter(config))
    }

    /// Slice using runtime start indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::from_vec(vec![5], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]));
    /// let starts = EagerTensor::from_tensor(Tensor::from_vec(vec![1], vec![2_i64]));
    /// let y = x.dynamic_slice(&starts, &[2]).unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[3.0, 4.0]);
    /// ```
    pub fn dynamic_slice(&self, starts: &Self, sizes: &[usize]) -> Result<Self> {
        self.binary_op(
            starts,
            StdTensorOp::DynamicSlice {
                slice_sizes: sizes.to_vec(),
            },
        )
    }

    /// Concatenate tensors along one axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]));
    /// let y = EagerTensor::from_tensor(Tensor::from_vec(vec![2], vec![3.0_f64, 4.0]));
    /// let z = EagerTensor::concatenate(&[&x, &y], 0).unwrap();
    ///
    /// assert_eq!(z.data().as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn concatenate(tensors: &[&Self], axis: usize) -> Result<Self> {
        Self::nary_op(
            tensors,
            StdTensorOp::Concatenate {
                axis,
                n_inputs: tensors.len(),
            },
        )
    }

    /// Extract the diagonal along two axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::from_vec(
    ///     vec![3, 3],
    ///     vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    /// ));
    /// let y = x.extract_diag(0, 1).unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[1.0, 5.0, 9.0]);
    /// ```
    pub fn extract_diag(&self, axis_a: usize, axis_b: usize) -> Result<Self> {
        self.unary_op(StdTensorOp::ExtractDiag { axis_a, axis_b })
    }

    /// Embed a vector or lower-rank tensor along a diagonal.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]));
    /// let y = x.embed_diag(0, 1).unwrap();
    ///
    /// assert_eq!(y.data().shape(), &[3, 3]);
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    /// ```
    pub fn embed_diag(&self, axis_a: usize, axis_b: usize) -> Result<Self> {
        self.unary_op(StdTensorOp::EmbedDiag { axis_a, axis_b })
    }

    /// Keep the lower triangle and zero the rest.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]));
    /// let y = x.tril(0).unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[1.0, 2.0, 0.0, 4.0]);
    /// ```
    pub fn tril(&self, k: i64) -> Result<Self> {
        self.unary_op(StdTensorOp::Tril { k })
    }

    /// Keep the upper triangle and zero the rest.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]));
    /// let y = x.triu(0).unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[1.0, 0.0, 3.0, 4.0]);
    /// ```
    pub fn triu(&self, k: i64) -> Result<Self> {
        self.unary_op(StdTensorOp::Triu { k })
    }

    /// Reduce product over the requested axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]));
    /// let y = x.reduce_prod(&[0, 1]).unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[24.0]);
    /// ```
    pub fn reduce_prod(&self, axes: &[usize]) -> Result<Self> {
        self.unary_op(StdTensorOp::ReduceProd {
            axes: axes.to_vec(),
        })
    }

    /// Reduce maximum over the requested axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]));
    /// let y = x.reduce_max(&[0, 1]).unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[4.0]);
    /// ```
    pub fn reduce_max(&self, axes: &[usize]) -> Result<Self> {
        self.unary_op(StdTensorOp::ReduceMax {
            axes: axes.to_vec(),
        })
    }

    /// Reduce minimum over the requested axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]));
    /// let y = x.reduce_min(&[0, 1]).unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[1.0]);
    /// ```
    pub fn reduce_min(&self, axes: &[usize]) -> Result<Self> {
        self.unary_op(StdTensorOp::ReduceMin {
            axes: axes.to_vec(),
        })
    }

    pub(crate) fn unary_op(&self, op: StdTensorOp) -> Result<Self> {
        Self::nary_op(&[self], op)
    }

    pub(crate) fn binary_op(&self, other: &Self, op: StdTensorOp) -> Result<Self> {
        Self::nary_op(&[self, other], op)
    }

    pub(crate) fn multi_output_unary_op(
        &self,
        op: StdTensorOp,
        num_outputs: usize,
    ) -> Result<Vec<Self>> {
        let outputs = {
            let mut backend = self.ctx.backend.lock().unwrap();
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
        let traces = record_eager_outputs(&op, &outputs, &[self]);
        if traces.len() != outputs.len() {
            return Err(Error::Internal(format!(
                "expected {} eager traces for {:?}, got {}",
                outputs.len(),
                op,
                traces.len()
            )));
        }

        Ok(traces
            .into_iter()
            .zip(outputs)
            .map(|(trace, output)| {
                Self::new_result(
                    Arc::clone(&self.ctx),
                    trace.key,
                    output.as_ref().clone(),
                    trace.requires_grad,
                    trace.node,
                )
            })
            .collect())
    }

    pub(crate) fn ternary_op(&self, b: &Self, c: &Self, op: StdTensorOp) -> Result<Self> {
        Self::nary_op(&[self, b, c], op)
    }

    pub(crate) fn nary_op(tensors: &[&Self], op: StdTensorOp) -> Result<Self> {
        let Some(first) = tensors.first() else {
            return Err(Error::Internal(
                "nary eager op requires at least one input tensor".to_string(),
            ));
        };

        let ctx = Arc::clone(&first.ctx);
        for tensor in tensors.iter().skip(1) {
            if !Arc::ptr_eq(&ctx, &tensor.ctx) {
                ctx.absorb_from(&tensor.ctx);
            }
        }

        let inputs: Vec<&Tensor> = tensors.iter().map(|tensor| tensor.data.as_ref()).collect();
        let output = Arc::new(exec_single_output(&op, &inputs, &ctx)?);
        let outputs = vec![Arc::clone(&output)];
        let mut traces = record_eager_outputs(&op, &outputs, tensors);
        let trace = traces.pop().ok_or_else(|| {
            Error::Internal(format!("expected one eager trace for {:?}, got 0", op))
        })?;

        Ok(Self::new_result(
            ctx,
            trace.key,
            output.as_ref().clone(),
            trace.requires_grad,
            trace.node,
        ))
    }
}
