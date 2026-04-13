use std::sync::Arc;

use tidu::{GradEdge, GradNode};

use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{
    DType, DotGeneralConfig, GatherConfig, PadConfig, SliceConfig, Tensor, TensorBackend,
};

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

    /// Permute tensor axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(
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
    /// let x = EagerTensor::from_tensor(Tensor::new(
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
            from_shape: DimExpr::from_concrete(self.data.shape()),
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
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]));
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
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![3], vec![1.0_f64, 2.0, 3.0]));
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

    /// Pad with zeros using StableHLO-style edge and interior padding.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, PadConfig, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![2], vec![1.0_f64, 2.0]));
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
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]));
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
    /// let x = EagerTensor::from_tensor(Tensor::new(
    ///     vec![5],
    ///     vec![10.0_f64, 20.0, 30.0, 40.0, 50.0],
    /// ));
    /// let indices = EagerTensor::from_tensor(Tensor::new(vec![3], vec![4.0_f64, 1.0, 0.0]));
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

    /// Slice using runtime start indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![5], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]));
    /// let starts = EagerTensor::from_tensor(Tensor::new(vec![1], vec![2.0_f64]));
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
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![2], vec![1.0_f64, 2.0]));
    /// let y = EagerTensor::from_tensor(Tensor::new(vec![2], vec![3.0_f64, 4.0]));
    /// let z = EagerTensor::concatenate(&[&x, &y], 0).unwrap();
    ///
    /// assert_eq!(z.data().as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn concatenate(tensors: &[&Self], axis: usize) -> Result<Self> {
        Self::nary_op(tensors, StdTensorOp::Concatenate { axis })
    }

    /// Elementwise absolute value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![2], vec![-1.0_f64, 2.0]));
    /// let y = x.abs().unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    /// ```
    pub fn abs(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Abs)
    }

    /// Elementwise complex conjugate.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![2], vec![1.0_f64, -2.0]));
    /// let y = x.conj().unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[1.0, -2.0]);
    /// ```
    pub fn conj(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Conj)
    }

    /// Elementwise sign.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![2], vec![-2.0_f64, 3.0]));
    /// let y = x.sign().unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[-1.0, 1.0]);
    /// ```
    pub fn sign(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Sign)
    }

    /// Elementwise natural logarithm.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![1], vec![1.0_f64]));
    /// let y = x.log().unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[0.0]);
    /// ```
    pub fn log(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Log)
    }

    /// Elementwise square root.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![1], vec![4.0_f64]));
    /// let y = x.sqrt().unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[2.0]);
    /// ```
    pub fn sqrt(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Sqrt)
    }

    /// Elementwise reciprocal square root.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![1], vec![4.0_f64]));
    /// let y = x.rsqrt().unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[0.5]);
    /// ```
    pub fn rsqrt(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Rsqrt)
    }

    /// Elementwise sine.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![1], vec![0.0_f64]));
    /// let y = x.sin().unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[0.0]);
    /// ```
    pub fn sin(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Sin)
    }

    /// Elementwise cosine.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![1], vec![0.0_f64]));
    /// let y = x.cos().unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[1.0]);
    /// ```
    pub fn cos(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Cos)
    }

    /// Elementwise hyperbolic tangent.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![1], vec![0.0_f64]));
    /// let y = x.tanh().unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[0.0]);
    /// ```
    pub fn tanh(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Tanh)
    }

    /// Elementwise `exp(x) - 1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![1], vec![0.0_f64]));
    /// let y = x.expm1().unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[0.0]);
    /// ```
    pub fn expm1(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Expm1)
    }

    /// Elementwise `log(1 + x)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![1], vec![0.0_f64]));
    /// let y = x.log1p().unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[0.0]);
    /// ```
    pub fn log1p(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Log1p)
    }

    /// Elementwise division.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![3], vec![8.0_f64, -6.0, 9.0]));
    /// let y = EagerTensor::from_tensor(Tensor::new(vec![3], vec![2.0_f64, 3.0, 3.0]));
    /// let z = x.div(&y).unwrap();
    ///
    /// assert_eq!(z.data().as_slice::<f64>().unwrap(), &[4.0, -2.0, 3.0]);
    /// ```
    pub fn div(&self, other: &Self) -> Result<Self> {
        self.binary_op(other, StdTensorOp::Div)
    }

    /// Elementwise power.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let base = EagerTensor::from_tensor(Tensor::new(vec![2], vec![2.0_f64, 3.0]));
    /// let exp = EagerTensor::from_tensor(Tensor::new(vec![2], vec![3.0_f64, 2.0]));
    /// let y = base.pow(&exp).unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[8.0, 9.0]);
    /// ```
    pub fn pow(&self, other: &Self) -> Result<Self> {
        self.binary_op(other, StdTensorOp::Pow)
    }

    /// Elementwise maximum.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![2], vec![1.0_f64, 5.0]));
    /// let y = EagerTensor::from_tensor(Tensor::new(vec![2], vec![3.0_f64, 4.0]));
    /// let z = x.maximum(&y).unwrap();
    ///
    /// assert_eq!(z.data().as_slice::<f64>().unwrap(), &[3.0, 5.0]);
    /// ```
    pub fn maximum(&self, other: &Self) -> Result<Self> {
        self.binary_op(other, StdTensorOp::Maximum)
    }

    /// Elementwise minimum.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![2], vec![1.0_f64, 5.0]));
    /// let y = EagerTensor::from_tensor(Tensor::new(vec![2], vec![3.0_f64, 4.0]));
    /// let z = x.minimum(&y).unwrap();
    ///
    /// assert_eq!(z.data().as_slice::<f64>().unwrap(), &[1.0, 4.0]);
    /// ```
    pub fn minimum(&self, other: &Self) -> Result<Self> {
        self.binary_op(other, StdTensorOp::Minimum)
    }

    /// Select values from `on_true` or `on_false` using `condition`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let condition = EagerTensor::from_tensor(Tensor::new(vec![2], vec![0.0_f64, 1.0]));
    /// let on_true = EagerTensor::from_tensor(Tensor::new(vec![2], vec![10.0_f64, 20.0]));
    /// let on_false = EagerTensor::from_tensor(Tensor::new(vec![2], vec![1.0_f64, 2.0]));
    /// let y = EagerTensor::select(&condition, &on_true, &on_false).unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[1.0, 20.0]);
    /// ```
    pub fn select(condition: &Self, on_true: &Self, on_false: &Self) -> Result<Self> {
        condition.ternary_op(on_true, on_false, StdTensorOp::Select)
    }

    /// Extract the diagonal along two axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(
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
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![3], vec![1.0_f64, 2.0, 3.0]));
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
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]));
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
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]));
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
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]));
    /// let y = x.reduce_prod(&[0, 1]).unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[24.0]);
    /// ```
    pub fn reduce_prod(&self, axes: &[usize]) -> Result<Self> {
        self.unary_op(StdTensorOp::ReduceProd {
            axes: axes.to_vec(),
            input_shape: DimExpr::from_concrete(self.data.shape()),
        })
    }

    /// Reduce maximum over the requested axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]));
    /// let y = x.reduce_max(&[0, 1]).unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[4.0]);
    /// ```
    pub fn reduce_max(&self, axes: &[usize]) -> Result<Self> {
        self.unary_op(StdTensorOp::ReduceMax {
            axes: axes.to_vec(),
            input_shape: DimExpr::from_concrete(self.data.shape()),
        })
    }

    /// Reduce minimum over the requested axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let x = EagerTensor::from_tensor(Tensor::new(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]));
    /// let y = x.reduce_min(&[0, 1]).unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[1.0]);
    /// ```
    pub fn reduce_min(&self, axes: &[usize]) -> Result<Self> {
        self.unary_op(StdTensorOp::ReduceMin {
            axes: axes.to_vec(),
            input_shape: DimExpr::from_concrete(self.data.shape()),
        })
    }

    /// Singular value decomposition: `A = U diag(S) Vt`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let a = EagerTensor::from_tensor(Tensor::new(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]));
    /// let (u, s, vt) = a.svd().unwrap();
    ///
    /// assert_eq!(u.data().shape(), &[2, 2]);
    /// assert_eq!(s.data().shape(), &[2]);
    /// assert_eq!(vt.data().shape(), &[2, 2]);
    /// ```
    pub fn svd(&self) -> Result<(Self, Self, Self)> {
        let mut outputs = self
            .multi_output_unary_op(
                StdTensorOp::Svd {
                    eps: 1.0e-12,
                    input_shape: DimExpr::from_concrete(self.data.shape()),
                },
                3,
            )?
            .into_iter();
        match (
            outputs.next(),
            outputs.next(),
            outputs.next(),
            outputs.next(),
        ) {
            (Some(u), Some(s), Some(vt), None) => Ok((u, s, vt)),
            _ => Err(Error::Internal(
                "svd eager op returned an unexpected number of outputs".to_string(),
            )),
        }
    }

    /// QR decomposition: `A = Q R`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let a = EagerTensor::from_tensor(Tensor::new(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]));
    /// let (q, r) = a.qr().unwrap();
    ///
    /// assert_eq!(q.data().shape(), &[2, 2]);
    /// assert_eq!(r.data().shape(), &[2, 2]);
    /// ```
    pub fn qr(&self) -> Result<(Self, Self)> {
        let mut outputs = self
            .multi_output_unary_op(
                StdTensorOp::Qr {
                    input_shape: DimExpr::from_concrete(self.data.shape()),
                },
                2,
            )?
            .into_iter();
        match (outputs.next(), outputs.next(), outputs.next()) {
            (Some(q), Some(r), None) => Ok((q, r)),
            _ => Err(Error::Internal(
                "qr eager op returned an unexpected number of outputs".to_string(),
            )),
        }
    }

    /// LU decomposition with partial pivoting: `P A = L U`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let a = EagerTensor::from_tensor(Tensor::new(vec![2, 2], vec![0.0_f64, 1.0, 1.0, 0.0]));
    /// let (p, l, u, parity) = a.lu().unwrap();
    ///
    /// assert_eq!(p.data().shape(), &[2, 2]);
    /// assert_eq!(l.data().shape(), &[2, 2]);
    /// assert_eq!(u.data().shape(), &[2, 2]);
    /// assert_eq!(parity.data().shape(), &[] as &[usize]);
    /// ```
    pub fn lu(&self) -> Result<(Self, Self, Self, Self)> {
        let mut outputs = self
            .multi_output_unary_op(
                StdTensorOp::Lu {
                    input_shape: DimExpr::from_concrete(self.data.shape()),
                },
                4,
            )?
            .into_iter();
        match (
            outputs.next(),
            outputs.next(),
            outputs.next(),
            outputs.next(),
            outputs.next(),
        ) {
            (Some(p), Some(l), Some(u), Some(parity), None) => Ok((p, l, u, parity)),
            _ => Err(Error::Internal(
                "lu eager op returned an unexpected number of outputs".to_string(),
            )),
        }
    }

    /// Cholesky factorization: `A = L L^T` for real inputs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let a = EagerTensor::from_tensor(Tensor::new(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]));
    /// let l = a.cholesky().unwrap();
    ///
    /// assert_eq!(l.data().shape(), &[2, 2]);
    /// assert_eq!(l.data().as_slice::<f64>().unwrap(), &[1.0, 0.0, 0.0, 1.0]);
    /// ```
    pub fn cholesky(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Cholesky {
            input_shape: DimExpr::from_concrete(self.data.shape()),
        })
    }

    /// Symmetric or Hermitian eigendecomposition: `A = V diag(W) V^T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let a = EagerTensor::from_tensor(Tensor::new(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0]));
    /// let (values, vectors) = a.eigh().unwrap();
    ///
    /// assert_eq!(values.data().shape(), &[2]);
    /// assert_eq!(vectors.data().shape(), &[2, 2]);
    /// ```
    pub fn eigh(&self) -> Result<(Self, Self)> {
        let mut outputs = self
            .multi_output_unary_op(
                StdTensorOp::Eigh {
                    eps: 1.0e-12,
                    input_shape: DimExpr::from_concrete(self.data.shape()),
                },
                2,
            )?
            .into_iter();
        match (outputs.next(), outputs.next(), outputs.next()) {
            (Some(values), Some(vectors), None) => Ok((values, vectors)),
            _ => Err(Error::Internal(
                "eigh eager op returned an unexpected number of outputs".to_string(),
            )),
        }
    }

    /// General eigendecomposition.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let a = EagerTensor::from_tensor(Tensor::new(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0]));
    /// let (values, vectors) = a.eig().unwrap();
    ///
    /// assert_eq!(values.data().shape(), &[2]);
    /// assert_eq!(vectors.data().shape(), &[2, 2]);
    /// ```
    pub fn eig(&self) -> Result<(Self, Self)> {
        let input_dtype: DType = self.data.dtype();
        let mut outputs = self
            .multi_output_unary_op(
                StdTensorOp::Eig {
                    input_dtype,
                    input_shape: DimExpr::from_concrete(self.data.shape()),
                },
                2,
            )?
            .into_iter();
        match (outputs.next(), outputs.next(), outputs.next()) {
            (Some(values), Some(vectors), None) => Ok((values, vectors)),
            _ => Err(Error::Internal(
                "eig eager op returned an unexpected number of outputs".to_string(),
            )),
        }
    }

    /// Solve a triangular linear system.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let a = EagerTensor::from_tensor(Tensor::new(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]));
    /// let b = EagerTensor::from_tensor(Tensor::new(vec![2, 1], vec![4.0_f64, 8.0]));
    /// let x = a
    ///     .triangular_solve(&b, true, true, false, false)
    ///     .unwrap();
    ///
    /// assert_eq!(x.data().shape(), &[2, 1]);
    /// assert_eq!(x.data().as_slice::<f64>().unwrap(), &[2.0, 2.0]);
    /// ```
    pub fn triangular_solve(
        &self,
        b: &Self,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Result<Self> {
        self.binary_op(
            b,
            StdTensorOp::TriangularSolve {
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
                lhs_shape: DimExpr::from_concrete(self.data.shape()),
                rhs_shape: DimExpr::from_concrete(b.data.shape()),
            },
        )
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
            Arc::clone(&self.ctx),
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
                    num_outputs,
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
                    Arc::clone(&self.ctx),
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

        let ctx = Arc::clone(&first.ctx);
        for tensor in tensors.iter().skip(1) {
            if !Arc::ptr_eq(&ctx, &tensor.ctx) {
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
