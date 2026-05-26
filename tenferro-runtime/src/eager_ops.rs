use std::sync::Arc;

use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{
    DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig, Tensor,
};

use crate::eager::{
    exec_single_output, maybe_print_eager_op_profile, profile_eager_op_section,
    record_eager_op_profile, record_eager_outputs, EagerTensor,
};
use crate::eager_exec::exec_dot_general_with_conj_on_tensors;
use crate::error::{Error, Result};
use crate::metadata::push_metadata_scope;

impl EagerTensor {
    /// Elementwise addition.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]), ctx.clone());
    /// let y = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]), ctx.clone());
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
    /// use tenferro_runtime::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]), ctx.clone());
    /// let y = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]), ctx.clone());
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
    /// use tenferro_runtime::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0]), ctx.clone());
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
    /// use tenferro_runtime::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![0.0_f64]), ctx.clone());
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
    /// use tenferro_runtime::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]), ctx.clone());
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
    /// use tenferro_runtime::{CpuBackend, DotGeneralConfig, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let a = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]), ctx.clone());
    /// let b = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]), ctx.clone());
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

    /// Execute a dot-general contraction, optionally conjugating either operand.
    ///
    /// Untracked tensors route the conjugation flags directly to the backend so
    /// the conjugated operand does not need to be materialized. Tracked tensors
    /// fall back to explicit `Conj` plus `DotGeneral` so reverse-mode AD keeps
    /// the same graph semantics as the standard eager ops.
    pub fn dot_general_with_conj(
        &self,
        other: &Self,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> Result<Self> {
        if !self.same_context(other) {
            return Err(Error::ContextMismatch {
                lhs: self.ctx_id(),
                rhs: other.ctx_id(),
            });
        }

        if !self.requires_grad && !other.requires_grad {
            let ctx = Arc::clone(&self.ctx);
            let output = {
                let mut backend = ctx.backend.lock().unwrap();
                exec_dot_general_with_conj_on_tensors(
                    self.data.as_ref(),
                    other.data.as_ref(),
                    config,
                    lhs_conj,
                    rhs_conj,
                    &mut *backend,
                )?
            };
            return Ok(Self::new_untracked_result(ctx, output));
        }

        match (lhs_conj, rhs_conj) {
            (false, false) => self.dot_general(other, config.clone()),
            (true, false) => self.conj()?.dot_general(other, config.clone()),
            (false, true) => {
                let rhs = other.conj()?;
                self.dot_general(&rhs, config.clone())
            }
            (true, true) => {
                let lhs = self.conj()?;
                let rhs = other.conj()?;
                lhs.dot_general(&rhs, config.clone())
            }
        }
    }

    /// Matrix multiplication for rank-2 tensors.
    ///
    /// This is a convenience wrapper over [`Self::dot_general`] that
    /// contracts the left matrix's column axis with the right matrix's row
    /// axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let a = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]),
    ///     ctx.clone(),
    /// );
    /// let b = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f64, 6.0]),
    ///     ctx,
    /// );
    /// let c = a.matmul(&b).unwrap();
    ///
    /// assert_eq!(c.data().shape(), &[2, 1]);
    /// assert_eq!(c.data().as_slice::<f64>().unwrap(), &[23.0, 34.0]);
    /// ```
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        let lhs_shape = self.data().shape();
        let rhs_shape = other.data().shape();
        if lhs_shape.len() != 2 {
            return Err(tenferro_tensor::Error::RankMismatch {
                op: "matmul",
                expected: 2,
                actual: lhs_shape.len(),
            }
            .into());
        }
        if rhs_shape.len() != 2 {
            return Err(tenferro_tensor::Error::RankMismatch {
                op: "matmul",
                expected: 2,
                actual: rhs_shape.len(),
            }
            .into());
        }
        if lhs_shape[1] != rhs_shape[0] {
            return Err(tenferro_tensor::Error::ShapeMismatch {
                op: "matmul",
                lhs: lhs_shape.to_vec(),
                rhs: rhs_shape.to_vec(),
            }
            .into());
        }
        self.dot_general(
            other,
            DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        )
    }

    /// Permute tensor axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(
    ///     vec![2, 3],
    ///     vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    /// ), ctx.clone());
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
    /// use tenferro_runtime::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(
    ///     vec![2, 3],
    ///     vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    /// ), ctx.clone());
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
    /// use tenferro_runtime::{CpuBackend, EagerRuntime, EagerTensor, SliceConfig, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]), ctx.clone());
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
    /// use tenferro_runtime::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]), ctx.clone());
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
    /// use tenferro_runtime::{CpuBackend, DType, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0]), ctx.clone());
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
    /// use tenferro_runtime::{CpuBackend, EagerRuntime, EagerTensor, PadConfig, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]), ctx.clone());
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
    /// use tenferro_runtime::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]), ctx.clone());
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
    /// use tenferro_runtime::{CpuBackend, EagerRuntime, EagerTensor, GatherConfig, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(
    ///     vec![5],
    ///     vec![10.0_f64, 20.0, 30.0, 40.0, 50.0],
    /// ), ctx.clone());
    /// let indices = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![3], vec![4_i64, 1, 0]), ctx.clone());
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
    /// use tenferro_runtime::{CpuBackend, EagerRuntime, EagerTensor, ScatterConfig, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let operand = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![4], vec![0.0_f64, 0.0, 0.0, 0.0]), ctx.clone());
    /// let indices = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2, 1], vec![1_i64, 3]), ctx.clone());
    /// let updates = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![5.0_f64, 7.0]), ctx.clone());
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
    /// use tenferro_runtime::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![5], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]), ctx.clone());
    /// let starts = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![2_i64]), ctx.clone());
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
    /// use tenferro_runtime::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]), ctx.clone());
    /// let y = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]), ctx.clone());
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
    /// use tenferro_runtime::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(
    ///     vec![3, 3],
    ///     vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    /// ), ctx.clone());
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
    /// use tenferro_runtime::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]), ctx.clone());
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
    /// use tenferro_runtime::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]), ctx.clone());
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
    /// use tenferro_runtime::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]), ctx.clone());
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
    /// use tenferro_runtime::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]), ctx.clone());
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
    /// use tenferro_runtime::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]), ctx.clone());
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
    /// use tenferro_runtime::{CpuBackend, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]), ctx.clone());
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

    pub(crate) fn ternary_op(&self, b: &Self, c: &Self, op: StdTensorOp) -> Result<Self> {
        Self::nary_op(&[self, b, c], op)
    }

    pub(crate) fn nary_op(tensors: &[&Self], op: StdTensorOp) -> Result<Self> {
        let total_started = std::time::Instant::now();
        let Some(first) = tensors.first() else {
            return Err(Error::Internal(
                "nary eager op requires at least one input tensor".to_string(),
            ));
        };

        let ctx = Arc::clone(&first.ctx);
        profile_eager_op_section("nary_op.context_check", || -> Result<()> {
            for tensor in tensors.iter().skip(1) {
                if !first.same_context(tensor) {
                    return Err(Error::ContextMismatch {
                        lhs: first.ctx_id(),
                        rhs: tensor.ctx_id(),
                    });
                }
            }
            Ok(())
        })?;

        let inputs: Vec<&Tensor> = profile_eager_op_section("nary_op.collect_inputs", || {
            tensors.iter().map(|tensor| tensor.data.as_ref()).collect()
        });
        let output = profile_eager_op_section("nary_op.exec_single_output", || {
            exec_single_output(&op, &inputs, &ctx)
        })?;
        let any_requires_grad = profile_eager_op_section("nary_op.requires_grad_scan", || {
            tensors.iter().any(|tensor| tensor.requires_grad)
        });
        if !any_requires_grad {
            let result = profile_eager_op_section("nary_op.new_untracked_result", || {
                Self::new_untracked_result(ctx, output)
            });
            record_eager_op_profile("nary_op.total", total_started.elapsed());
            maybe_print_eager_op_profile();
            return Ok(result);
        }

        let output = Arc::new(output);
        let outputs = vec![Arc::clone(&output)];
        let mut recorded = profile_eager_op_section("nary_op.record_outputs", || {
            record_eager_outputs(&op, &outputs, tensors)
        });
        let trace = recorded.traces.pop().ok_or_else(|| {
            Error::Internal(format!("expected one eager trace for {:?}, got 0", op))
        })?;
        let mut metadata_scopes = vec![Arc::clone(&recorded.metadata_scope)];
        for tensor in tensors {
            for scope in &tensor.metadata_scopes {
                push_metadata_scope(&mut metadata_scopes, Arc::clone(scope));
            }
        }

        let result = profile_eager_op_section("nary_op.new_tracked_result", || {
            Self::new_result_arc(
                ctx,
                trace.key,
                output,
                trace.requires_grad,
                trace.node,
                metadata_scopes,
            )
        });
        record_eager_op_profile("nary_op.total", total_started.elapsed());
        maybe_print_eager_op_profile();
        Ok(result)
    }
}
