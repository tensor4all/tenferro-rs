use std::sync::Arc;

use computegraph::GraphOperation;
use num_complex::{Complex32, Complex64};
use tenferro_ops::broadcast::{
    broadcast_error_to_validation, broadcast_in_dim_extent_error, broadcast_input_plan,
    broadcast_shape, broadcast_shapes,
};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{
    DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig, Tensor,
    TensorValue,
};

use crate::eager::{
    eager_grad_recording_enabled, eager_op_profile_start, exec_single_output,
    exec_single_output_read, maybe_print_eager_op_profile, profile_eager_op_section,
    record_eager_op_profile, record_eager_outputs, record_eager_value_outputs, EagerTensor,
};
use crate::eager_exec::exec_dot_general_with_conj_on_tensor_reads;
use crate::error::{Error, Result};
use crate::metadata::push_metadata_scope;

pub(crate) fn broadcast_binary(
    op: &'static str,
    lhs: &EagerTensor,
    rhs: &EagerTensor,
) -> Result<(EagerTensor, EagerTensor)> {
    ensure_same_context(lhs, rhs)?;
    let shape =
        broadcast_shape(lhs.shape(), rhs.shape()).map_err(|err| broadcast_error(op, err))?;
    Ok((
        broadcast_to(op, lhs, &shape)?,
        broadcast_to(op, rhs, &shape)?,
    ))
}

pub(crate) fn broadcast_ternary(
    op: &'static str,
    first: &EagerTensor,
    second: &EagerTensor,
    third: &EagerTensor,
) -> Result<(EagerTensor, EagerTensor, EagerTensor)> {
    ensure_same_context(first, second)?;
    ensure_same_context(first, third)?;
    let shape = broadcast_shapes([first.shape(), second.shape(), third.shape()])
        .map_err(|err| broadcast_error(op, err))?;
    Ok((
        broadcast_to(op, first, &shape)?,
        broadcast_to(op, second, &shape)?,
        broadcast_to(op, third, &shape)?,
    ))
}

fn broadcast_to(
    op: &'static str,
    input: &EagerTensor,
    target_shape: &[usize],
) -> Result<EagerTensor> {
    let input_shape = input.shape();
    if input_shape == target_shape {
        return Ok(input.clone());
    }

    let plan =
        broadcast_input_plan(input_shape, target_shape).map_err(|err| broadcast_error(op, err))?;
    let source = if plan.source_shape == input_shape {
        input.clone()
    } else {
        input.reshape(&plan.source_shape)?
    };
    source.broadcast_in_dim(target_shape, &plan.dims)
}

fn broadcast_error(op: &'static str, err: tenferro_ops::broadcast::BroadcastError) -> Error {
    tenferro_tensor::Error::validation(op, broadcast_error_to_validation(err)).into()
}

fn ensure_same_context(lhs: &EagerTensor, rhs: &EagerTensor) -> Result<()> {
    if !lhs.same_context(rhs) {
        return Err(Error::ContextMismatch {
            lhs: lhs.ctx_id(),
            rhs: rhs.ctx_id(),
        });
    }
    Ok(())
}

impl std::ops::Add for &EagerTensor {
    type Output = Result<EagerTensor>;

    fn add(self, rhs: &EagerTensor) -> Result<EagerTensor> {
        EagerTensor::add(self, rhs)
    }
}

impl std::ops::Sub for &EagerTensor {
    type Output = Result<EagerTensor>;

    fn sub(self, rhs: &EagerTensor) -> Result<EagerTensor> {
        EagerTensor::sub(self, rhs)
    }
}

impl std::ops::Mul for &EagerTensor {
    type Output = Result<EagerTensor>;

    fn mul(self, rhs: &EagerTensor) -> Result<EagerTensor> {
        EagerTensor::mul(self, rhs)
    }
}

impl std::ops::Div for &EagerTensor {
    type Output = Result<EagerTensor>;

    fn div(self, rhs: &EagerTensor) -> Result<EagerTensor> {
        EagerTensor::div(self, rhs)
    }
}

impl std::ops::Rem for &EagerTensor {
    type Output = Result<EagerTensor>;

    fn rem(self, rhs: &EagerTensor) -> Result<EagerTensor> {
        EagerTensor::rem(self, rhs)
    }
}

impl std::ops::Neg for &EagerTensor {
    type Output = Result<EagerTensor>;

    fn neg(self) -> Result<EagerTensor> {
        EagerTensor::neg(self)
    }
}

impl EagerTensor {
    /// Elementwise addition.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(), ctx.clone()).unwrap();
    /// let z = x.add(&y).unwrap();
    ///
    /// assert_eq!(z.materialized().unwrap().as_slice::<f64>().unwrap(), &[4.0, 6.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::ContextMismatch`] for tensors from different eager
    /// runtimes, [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch`/`DTypeMismatch` for incompatible operands, or a typed
    /// backend/runtime-state error during execution.
    pub fn add(&self, other: &Self) -> Result<Self> {
        let (lhs, rhs) = broadcast_binary("add", self, other)?;
        lhs.binary_op(&rhs, StdTensorOp::Add)
    }

    /// Elementwise subtraction.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ContextMismatch`] for tensors from different eager
    /// runtimes, [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch`/`DTypeMismatch` for incompatible operands, or a typed
    /// backend/runtime-state error during execution.
    pub fn sub(&self, other: &Self) -> Result<Self> {
        let (lhs, rhs) = broadcast_binary("sub", self, other)?;
        lhs.binary_op(&rhs, StdTensorOp::Sub)
    }

    /// Elementwise multiplication.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(), ctx.clone()).unwrap();
    /// let z = x.mul(&y).unwrap();
    ///
    /// assert_eq!(z.materialized().unwrap().as_slice::<f64>().unwrap(), &[3.0, 8.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::ContextMismatch`] for tensors from different eager
    /// runtimes, [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch`/`DTypeMismatch` for incompatible operands, or a typed
    /// backend/runtime-state error during execution.
    pub fn mul(&self, other: &Self) -> Result<Self> {
        let (lhs, rhs) = broadcast_binary("mul", self, other)?;
        lhs.binary_op(&rhs, StdTensorOp::Mul)
    }

    /// Negate the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.neg().unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[-1.0, 2.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] when the backend does
    /// not implement negation for the dtype, or a typed backend/runtime-state
    /// error during execution.
    pub fn neg(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Neg)
    }

    /// Elementwise exponential.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![0.0_f64]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.exp().unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[1.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] when the backend does
    /// not implement exponentiation for the dtype, or a typed backend/
    /// runtime-state error during execution.
    pub fn exp(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Exp)
    }

    /// Reduce sum over the requested axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.reduce_sum(None).unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[10.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `AxisOutOfBounds` or
    /// `DuplicateAxis` for an invalid reduction axis, or a typed
    /// unsupported/backend/runtime-state error for the selected dtype.
    pub fn reduce_sum(&self, axes: Option<&[usize]>) -> Result<Self> {
        let axes = axes.map_or_else(|| (0..self.shape().len()).collect(), <[usize]>::to_vec);
        validate_eager_axes("EagerTensor::reduce_sum", self.shape().len(), &axes)?;
        self.unary_op(StdTensorOp::ReduceSum { axes })
    }

    /// Sum elementwise squares over the requested axes.
    ///
    /// Each value is squared in its input dtype before reduction. The initial
    /// supported dtypes are `f32` and `f64`; other dtypes return a typed
    /// unsupported error. Passing an empty axis slice returns the elementwise
    /// square without reducing rank.
    ///
    /// This operation is useful when the squared sum is needed directly. Use
    /// the linalg norm APIs when a square root or complex magnitude semantics
    /// are required.
    ///
    /// # Errors
    ///
    /// Returns a typed validation error for invalid axes, a typed unsupported
    /// error for other dtypes, or a typed backend or runtime-state error during
    /// execution.
    pub fn reduce_sum_squares(&self, axes: &[usize]) -> Result<Self> {
        validate_eager_axes("EagerTensor::reduce_sum_squares", self.shape().len(), axes)?;
        self.unary_op(StdTensorOp::ReduceSumSquares {
            axes: axes.to_vec(),
        })
    }

    /// Execute a dot-general contraction eagerly.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{DotGeneralConfig, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let a = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(), ctx.clone()).unwrap();
    /// let b = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(), ctx.clone()).unwrap();
    /// let c = a.dot_general(&b, DotGeneralConfig {
    ///     lhs_contracting_dims: vec![1],
    ///     rhs_contracting_dims: vec![0],
    ///     lhs_batch_dims: vec![],
    ///     rhs_batch_dims: vec![],
    /// }).unwrap();
    ///
    /// assert_eq!(c.shape(), &[2, 2]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `RankMismatch`,
    /// `AxisOutOfBounds`, `DuplicateAxis`, `ShapeMismatch`, or `DTypeMismatch`
    /// when `config` or the operands are invalid; backend and runtime-state
    /// failures retain their typed sources.
    pub fn dot_general(&self, other: &Self, config: DotGeneralConfig) -> Result<Self> {
        validate_eager_dot_general_config(
            "EagerTensor::dot_general",
            &config,
            self.shape().len(),
            other.shape().len(),
        )?;
        self.binary_op(other, StdTensorOp::DotGeneral { config })
    }

    /// Execute a dot-general contraction, optionally conjugating either operand.
    ///
    /// Untracked tensors route the conjugation flags directly to the backend so
    /// the conjugated operand does not need to be materialized. Tracked tensors
    /// fall back to explicit `Conj` plus `DotGeneral` so reverse-mode AD keeps
    /// the same graph semantics as the standard eager ops.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ContextMismatch`] for operands from different eager
    /// runtimes, [`tenferro_tensor::Error::Validation`] for rank/axis/shape or
    /// dtype mismatches in `config`, or a typed backend/runtime-state error.
    pub fn dot_general_with_conj(
        &self,
        other: &Self,
        config: DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> Result<Self> {
        if !self.same_context(other) {
            return Err(Error::ContextMismatch {
                lhs: self.ctx_id(),
                rhs: other.ctx_id(),
            });
        }
        validate_eager_dot_general_config(
            "EagerTensor::dot_general_with_conj",
            &config,
            self.shape().len(),
            other.shape().len(),
        )?;

        if !self.requires_grad && !other.requires_grad {
            let ctx = Arc::clone(&self.ctx);
            let mut backend = ctx.lock_backend()?;
            let output = exec_dot_general_with_conj_on_tensor_reads(
                self.tensor_read(),
                other.tensor_read(),
                &config,
                lhs_conj,
                rhs_conj,
                &mut *backend,
            )?;
            drop(backend);
            return Self::new_untracked_result(ctx, output);
        }

        match (lhs_conj, rhs_conj) {
            (false, false) => self.dot_general(other, config),
            (true, false) => self.conj()?.dot_general(other, config),
            (false, true) => {
                let rhs = other.conj()?;
                self.dot_general(&rhs, config)
            }
            (true, true) => {
                let lhs = self.conj()?;
                let rhs = other.conj()?;
                lhs.dot_general(&rhs, config)
            }
        }
    }

    /// Scale by a real scalar: `y = factor * x`.
    ///
    /// Integer factors are rounded to the nearest integer before multiplication,
    /// boolean factors map finite zero to `false` and other finite values to
    /// `true`, and complex tensors receive a zero-imaginary scalar.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TensorRuntime`] with
    /// [`tenferro_tensor::ValidationError::InvalidArgument`] when an integer or
    /// boolean factor is non-finite or outside the input dtype's range. Backend
    /// and runtime execution failures retain their typed source variants.
    pub fn scale_real(&self, factor: f64) -> Result<Self> {
        let scalar = match self.dtype() {
            DType::F64 => Tensor::from_vec_col_major(vec![], vec![factor])?,
            DType::F32 => Tensor::from_vec_col_major(vec![], vec![factor as f32])?,
            DType::I32 => Tensor::from_vec_col_major(vec![], vec![round_real_to_i32(factor)?])?,
            DType::I64 => Tensor::from_vec_col_major(vec![], vec![round_real_to_i64(factor)?])?,
            DType::Bool => Tensor::from_vec_col_major(vec![], vec![bool_from_real(factor)?])?,
            DType::C64 => Tensor::from_vec_col_major(vec![], vec![Complex64::new(factor, 0.0)])?,
            DType::C32 => {
                Tensor::from_vec_col_major(vec![], vec![Complex32::new(factor as f32, 0.0)])?
            }
        };
        let scalar = EagerTensor::from_tensor_in(scalar, Arc::clone(&self.ctx))?;
        self.mul(&scalar)
    }

    /// Scale a complex tensor by a complex scalar: `y = factor * x`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TensorRuntime`] with
    /// [`tenferro_tensor::ValidationError::InvalidArgument`] for a non-complex
    /// input dtype. Backend and runtime execution failures retain their typed
    /// source variants.
    pub fn scale_complex(&self, factor: Complex64) -> Result<Self> {
        let scalar = match self.dtype() {
            DType::C64 => Tensor::from_vec_col_major(vec![], vec![factor])?,
            DType::C32 => Tensor::from_vec_col_major(
                vec![],
                vec![Complex32::new(factor.re as f32, factor.im as f32)],
            )?,
            dtype => {
                return Err(Error::TensorRuntime(
                    tenferro_tensor::Error::invalid_argument(
                        "scale_complex",
                        "dtype",
                        format!("requires complex tensor dtype, got {dtype:?}"),
                    ),
                ));
            }
        };
        let scalar = EagerTensor::from_tensor_in(scalar, Arc::clone(&self.ctx))?;
        self.mul(&scalar)
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
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let a = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
    ///     ctx.clone(),
    /// ).unwrap();
    /// let b = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f64, 6.0]).unwrap(),
    ///     ctx,
    /// ).unwrap();
    /// let c = a.matmul(&b).unwrap();
    ///
    /// assert_eq!(c.shape(), &[2, 1]);
    /// assert_eq!(c.materialized().unwrap().as_slice::<f64>().unwrap(), &[23.0, 34.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::ValidationError::RankMismatch`] when either operand is
    /// not rank 2, `ShapeMismatch` when the inner dimensions differ, or a typed
    /// dtype/backend/runtime-state error during the contraction.
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        let lhs_shape = self.shape();
        let rhs_shape = other.shape();
        if lhs_shape.len() != 2 {
            return Err(tenferro_tensor::Error::rank_mismatch("matmul", 2, lhs_shape.len()).into());
        }
        if rhs_shape.len() != 2 {
            return Err(tenferro_tensor::Error::rank_mismatch("matmul", 2, rhs_shape.len()).into());
        }
        if lhs_shape[1] != rhs_shape[0] {
            return Err(
                tenferro_tensor::Error::shape_mismatch("matmul", lhs_shape, rhs_shape).into(),
            );
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
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(
    ///     vec![2, 3],
    ///     vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    /// ).unwrap(), ctx.clone()).unwrap();
    /// let y = x.transpose(&[1, 0]).unwrap();
    ///
    /// assert_eq!(y.shape(), &[3, 2]);
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `AxisOutOfBounds`
    /// or `DuplicateAxis` when `perm` is not a permutation, or a typed
    /// backend/runtime-state error while creating the view.
    pub fn transpose(&self, perm: &[usize]) -> Result<Self> {
        let op = StdTensorOp::Transpose {
            perm: perm.to_vec(),
        };
        let value = self
            .value
            .transpose_view(perm)
            .map_err(Error::TensorRuntime)?;
        Self::nary_value_op(&[self], op, value)
    }

    /// Reshape without changing element order.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(
    ///     vec![2, 3],
    ///     vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    /// ).unwrap(), ctx.clone()).unwrap();
    /// let y = x.reshape(&[6]).unwrap();
    ///
    /// assert_eq!(y.shape(), &[6]);
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::ValidationError::ShapeMismatch`] when the element count
    /// changes, `InvalidArgument` when the target shape product overflows, or a
    /// typed backend/runtime-state error.
    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        let op = StdTensorOp::Reshape {
            to_shape: DimExpr::from_concrete(shape),
        };
        if let Ok(value) = self.value.reshape_view(shape) {
            return Self::nary_value_op(&[self], op, value);
        }
        self.unary_op(op)
    }

    /// Slice with explicit start, limit, and stride per axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, SliceConfig, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = x
    ///     .slice(SliceConfig {
    ///         starts: vec![1],
    ///         limits: vec![3],
    ///         strides: vec![1],
    ///     })
    ///     .unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[2.0, 3.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `AxisOutOfBounds`/`InvalidArgument` when starts, limits, or strides are
    /// invalid, or a typed backend/runtime-state error while creating the view.
    pub fn slice(&self, config: SliceConfig) -> Result<Self> {
        let value = self
            .value
            .slice_view(&config)
            .map_err(Error::TensorRuntime)?;
        Self::nary_value_op(&[self], StdTensorOp::Slice(config), value)
    }

    /// Broadcast into a larger shape with explicit dimension placement.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.broadcast_in_dim(&[3, 2], &[0]).unwrap();
    ///
    /// assert_eq!(y.shape(), &[3, 2]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `AxisOutOfBounds`,
    /// `DuplicateAxis`, or `ShapeMismatch` when `shape`/`dims` cannot broadcast
    /// the input, or a typed backend/runtime-state error.
    pub fn broadcast_in_dim(&self, shape: &[usize], dims: &[usize]) -> Result<Self> {
        if let Some(error) = broadcast_in_dim_extent_error(self.shape(), shape, dims) {
            return Err(broadcast_error("EagerTensor::broadcast_in_dim", error));
        }
        let op = StdTensorOp::BroadcastInDim {
            shape: DimExpr::from_concrete(shape),
            dims: dims.to_vec(),
        };
        let value = self
            .value
            .broadcast_in_dim_view(shape, dims)
            .map_err(Error::TensorRuntime)?;
        Self::nary_value_op(&[self], op, value)
    }

    /// Convert the tensor to a different dtype using checked conversion.
    ///
    /// Use [`cast`](Self::cast) when a lossy dtype projection is intended.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{DType, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.convert(DType::C64).unwrap();
    ///
    /// assert_eq!(y.dtype(), DType::C64);
    /// assert_eq!(y.shape(), &[2]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDTypeConversion`] when the
    /// requested pair is outside tenferro's checked dtype-promotion lattice.
    /// Use [`cast`](Self::cast) for explicit lossy projection; backend
    /// execution can additionally return a typed runtime-state error.
    pub fn convert(&self, to: DType) -> Result<Self> {
        tenferro_tensor::validate::validate_convert_dtype("EagerTensor::convert", self.dtype(), to)
            .map_err(Error::TensorRuntime)?;
        self.cast(to)
    }

    /// Cast the tensor to a different dtype using explicit dtype projection.
    ///
    /// `cast` may truncate, narrow precision, project complex values to their
    /// real component, or use boolean truthiness where the backend supports the
    /// requested projection.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{DType, EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.2_f64, -2.8]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.cast(DType::I32).unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<i32>().unwrap(), &[1, -2]);
    /// ```
    /// # Errors
    ///
    /// Returns a typed [`tenferro_tensor::Error::Unsupported`] when the eager
    /// backend cannot project the requested dtype, or a backend/runtime-state
    /// error during execution.
    pub fn cast(&self, to: DType) -> Result<Self> {
        self.unary_op(StdTensorOp::Convert {
            from: self.dtype(),
            to,
        })
    }

    /// Pad with zeros using StableHLO-style edge and interior padding.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, PadConfig, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = x
    ///     .pad(PadConfig {
    ///         edge_padding_low: vec![1],
    ///         edge_padding_high: vec![1],
    ///         interior_padding: vec![1],
    ///     })
    ///     .unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[0.0, 1.0, 0.0, 2.0, 0.0]);
    /// ```
    /// # Errors
    ///
    /// Returns [`tenferro_runtime::Error::TensorRuntime`] containing
    /// [`tenferro_tensor::ValidationError::InvalidArgument`] when a
    /// padding vector has a length different from the input rank, interior
    /// padding is negative, or edge/interior padding produces a negative
    /// dimension or checked output-size arithmetic overflows.
    /// Backend execution and unavailable runtime state are propagated as their
    /// typed [`tenferro_runtime::Error::TensorRuntime`] or
    /// [`tenferro_runtime::Error::RuntimeState`] variants.
    pub fn pad(&self, config: PadConfig) -> Result<Self> {
        self.unary_op(StdTensorOp::Pad(config))
    }

    /// Reverse the order of elements along the requested axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.reverse(&[0]).unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[4.0, 3.0, 2.0, 1.0]);
    /// ```
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `AxisOutOfBounds` or
    /// `DuplicateAxis` for an invalid axis list, or a typed backend/
    /// runtime-state error during execution.
    pub fn reverse(&self, axes: &[usize]) -> Result<Self> {
        validate_eager_axes("EagerTensor::reverse", self.shape().len(), axes)?;
        self.unary_op(StdTensorOp::Reverse {
            axes: axes.to_vec(),
        })
    }

    /// Gather slices from `self` using integer start indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, GatherConfig, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(
    ///     vec![5],
    ///     vec![10.0_f64, 20.0, 30.0, 40.0, 50.0],
    /// ).unwrap(), ctx.clone()).unwrap();
    /// let indices = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![3], vec![4_i64, 1, 0]).unwrap(), ctx.clone()).unwrap();
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
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[50.0, 20.0, 10.0]);
    /// ```
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] when the gather
    /// configuration has an invalid rank, axis, shape, or index dtype, or a
    /// typed backend/runtime-state error.
    pub fn gather(&self, indices: &Self, config: GatherConfig) -> Result<Self> {
        self.binary_op(indices, StdTensorOp::Gather(config))
    }

    /// Scatter updates into `self` using StableHLO scatter semantics.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, ScatterConfig, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let operand = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![4], vec![0.0_f64, 0.0, 0.0, 0.0]).unwrap(), ctx.clone()).unwrap();
    /// let indices = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2, 1], vec![1_i64, 3]).unwrap(), ctx.clone()).unwrap();
    /// let updates = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![5.0_f64, 7.0]).unwrap(), ctx.clone()).unwrap();
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
    /// assert_eq!(result.materialized().unwrap().as_slice::<f64>().unwrap(), &[0.0, 5.0, 0.0, 7.0]);
    /// ```
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] when the scatter
    /// configuration, index/update shapes, or index dtype is invalid, or a
    /// typed backend/runtime-state error.
    pub fn scatter(&self, indices: &Self, updates: &Self, config: ScatterConfig) -> Result<Self> {
        self.ternary_op(indices, updates, StdTensorOp::Scatter(config))
    }

    /// Slice using runtime start indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![5], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]).unwrap(), ctx.clone()).unwrap();
    /// let starts = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![2_i64]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.dynamic_slice(&starts, &[2]).unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[3.0, 4.0]);
    /// ```
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] when `starts` has the
    /// wrong dtype/shape or `sizes` exceeds the operand rank, including an
    /// `AxisOutOfBounds` or `ShapeMismatch`, or a typed backend/runtime-state
    /// error.
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
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(), ctx.clone()).unwrap();
    /// let z = EagerTensor::concatenate(&[&x, &y], 0).unwrap();
    ///
    /// assert_eq!(z.materialized().unwrap().as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    /// ```
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::ValidationError::InvalidArgument`] when `tensors` is
    /// empty or `axis` is outside the rank, `ShapeMismatch`/`DTypeMismatch`
    /// when inputs cannot be concatenated, or a typed backend/runtime-state
    /// error.
    pub fn concatenate(tensors: &[&Self], axis: usize) -> Result<Self> {
        Self::nary_op(
            tensors,
            StdTensorOp::Concatenate {
                axis,
                input_count: tensors.len(),
            },
        )
    }

    /// Extract the diagonal along two axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(
    ///     vec![3, 3],
    ///     vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    /// ).unwrap(), ctx.clone()).unwrap();
    /// let y = x.extract_diag(0, 1).unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[1.0, 5.0, 9.0]);
    /// ```
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `RankMismatch`,
    /// `AxisOutOfBounds`, or `DuplicateAxis` when the selected axes cannot form
    /// a diagonal, or a typed backend/runtime-state error.
    pub fn extract_diag(&self, axis_a: usize, axis_b: usize) -> Result<Self> {
        self.unary_op(StdTensorOp::ExtractDiag { axis_a, axis_b })
    }

    /// Embed a vector or lower-rank tensor along a diagonal.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.embed_diag(0, 1).unwrap();
    ///
    /// assert_eq!(y.shape(), &[3, 3]);
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    /// ```
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `RankMismatch`,
    /// `AxisOutOfBounds`, or `DuplicateAxis` when the diagonal axes are not
    /// valid for embedding, or a typed backend/runtime-state error.
    pub fn embed_diag(&self, axis_a: usize, axis_b: usize) -> Result<Self> {
        self.unary_op(StdTensorOp::EmbedDiag { axis_a, axis_b })
    }

    /// Keep the lower triangle and zero the rest.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.tril(0).unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[1.0, 2.0, 0.0, 4.0]);
    /// ```
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::ValidationError::RankMismatch`] when the operand is not
    /// a matrix, or a typed unsupported/backend/runtime-state error.
    pub fn tril(&self, k: i64) -> Result<Self> {
        self.unary_op(StdTensorOp::Tril { k })
    }

    /// Keep the upper triangle and zero the rest.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.triu(0).unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[1.0, 0.0, 3.0, 4.0]);
    /// ```
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::ValidationError::RankMismatch`] when the operand is not
    /// a matrix, or a typed unsupported/backend/runtime-state error.
    pub fn triu(&self, k: i64) -> Result<Self> {
        self.unary_op(StdTensorOp::Triu { k })
    }

    /// Reduce product over the requested axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.reduce_prod(None).unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[24.0]);
    /// ```
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `AxisOutOfBounds` or
    /// `DuplicateAxis` for an invalid reduction axis, or a typed
    /// unsupported/backend/runtime-state error for the selected dtype.
    pub fn reduce_prod(&self, axes: Option<&[usize]>) -> Result<Self> {
        let axes = axes.map_or_else(|| (0..self.shape().len()).collect(), <[usize]>::to_vec);
        validate_eager_axes("EagerTensor::reduce_prod", self.shape().len(), &axes)?;
        self.unary_op(StdTensorOp::ReduceProd { axes })
    }

    /// Reduce maximum over the requested axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.reduce_max(None).unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[4.0]);
    /// ```
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `AxisOutOfBounds` or
    /// `DuplicateAxis` for an invalid reduction axis, or a typed
    /// unsupported/backend/runtime-state error for the selected dtype.
    pub fn reduce_max(&self, axes: Option<&[usize]>) -> Result<Self> {
        let axes = axes.map_or_else(|| (0..self.shape().len()).collect(), <[usize]>::to_vec);
        validate_eager_axes("EagerTensor::reduce_max", self.shape().len(), &axes)?;
        self.unary_op(StdTensorOp::ReduceMax { axes })
    }

    /// Reduce minimum over the requested axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    ///
    /// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(), ctx.clone()).unwrap();
    /// let y = x.reduce_min(None).unwrap();
    ///
    /// assert_eq!(y.materialized().unwrap().as_slice::<f64>().unwrap(), &[1.0]);
    /// ```
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `AxisOutOfBounds` or
    /// `DuplicateAxis` for an invalid reduction axis, or a typed
    /// unsupported/backend/runtime-state error for the selected dtype.
    pub fn reduce_min(&self, axes: Option<&[usize]>) -> Result<Self> {
        let axes = axes.map_or_else(|| (0..self.shape().len()).collect(), <[usize]>::to_vec);
        validate_eager_axes("EagerTensor::reduce_min", self.shape().len(), &axes)?;
        self.unary_op(StdTensorOp::ReduceMin { axes })
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

    pub(crate) fn nary_value_op(
        tensors: &[&Self],
        op: StdTensorOp,
        value: TensorValue,
    ) -> Result<Self> {
        let Some(first) = tensors.first() else {
            return Err(empty_nary_input_error(&op));
        };

        let ctx = Arc::clone(&first.ctx);
        for tensor in tensors.iter().skip(1) {
            if !first.same_context(tensor) {
                return Err(Error::ContextMismatch {
                    lhs: first.ctx_id(),
                    rhs: tensor.ctx_id(),
                });
            }
        }

        if !eager_grad_recording_enabled() {
            return Ok(Self::new_untracked_value_result(ctx, value));
        }

        let output_ref = &value;
        let mut recorded = record_eager_value_outputs(&op, &[output_ref], tensors)?;
        let trace = recorded.traces.pop().ok_or_else(|| {
            Error::Internal(format!("expected one eager trace for {:?}, got 0", op))
        })?;
        let semantic_trace = recorded.semantic_traces.pop().flatten();
        let mut metadata_scopes = vec![Arc::clone(&recorded.metadata_scope)];
        for tensor in tensors {
            for scope in &tensor.metadata_scopes {
                push_metadata_scope(&mut metadata_scopes, Arc::clone(scope));
            }
        }

        Self::new_result_value(
            ctx,
            trace.key,
            value,
            trace.requires_grad,
            trace.trace,
            semantic_trace,
            metadata_scopes,
        )
    }

    pub(crate) fn nary_op(tensors: &[&Self], op: StdTensorOp) -> Result<Self> {
        let total_started = eager_op_profile_start();
        let Some(first) = tensors.first() else {
            return Err(empty_nary_input_error(&op));
        };
        let expected = op.input_count();
        if tensors.len() != expected {
            return Err(wrong_nary_input_count_error(&op, expected, tensors.len()));
        }

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

        let any_requires_grad = profile_eager_op_section("nary_op.requires_grad_scan", || {
            eager_grad_recording_enabled() && tensors.iter().any(|tensor| tensor.requires_grad)
        });
        if !eager_grad_recording_enabled() {
            let input_reads = profile_eager_op_section("nary_op.collect_input_reads", || {
                tensors
                    .iter()
                    .map(|tensor| tensor.tensor_read())
                    .collect::<Vec<_>>()
            });
            let output = profile_eager_op_section("nary_op.exec_single_output_read", || {
                exec_single_output_read(&op, &input_reads, &ctx)
            })?;
            let result = profile_eager_op_section("nary_op.new_untracked_result", || {
                Self::new_untracked_result(ctx, output)
            });
            if let Some(total_started) = total_started {
                record_eager_op_profile("nary_op.total", total_started.elapsed());
                maybe_print_eager_op_profile();
            }
            return result;
        }

        if !any_requires_grad {
            let input_reads = profile_eager_op_section("nary_op.collect_input_reads", || {
                tensors
                    .iter()
                    .map(|tensor| tensor.tensor_read())
                    .collect::<Vec<_>>()
            });
            let output = profile_eager_op_section("nary_op.exec_single_output_read", || {
                exec_single_output_read(&op, &input_reads, &ctx)
            })?;
            let output = Arc::new(output);
            let outputs = vec![Arc::clone(&output)];
            let mut recorded =
                profile_eager_op_section("nary_op.record_untracked_outputs", || {
                    record_eager_outputs(&op, &outputs, tensors)
                })?;
            let trace = recorded.traces.pop().ok_or_else(|| {
                Error::Internal(format!("expected one eager trace for {:?}, got 0", op))
            })?;
            let semantic_trace = recorded.semantic_traces.pop().flatten();
            let mut metadata_scopes = vec![Arc::clone(&recorded.metadata_scope)];
            for tensor in tensors {
                for scope in &tensor.metadata_scopes {
                    push_metadata_scope(&mut metadata_scopes, Arc::clone(scope));
                }
            }
            let result = profile_eager_op_section("nary_op.new_untracked_semantic_result", || {
                Self::new_unregistered_result_arc_with_semantic_trace(
                    ctx,
                    trace.key,
                    output,
                    trace.requires_grad,
                    trace.trace,
                    semantic_trace,
                    metadata_scopes,
                )
            });
            if let Some(total_started) = total_started {
                record_eager_op_profile("nary_op.total", total_started.elapsed());
                maybe_print_eager_op_profile();
            }
            return result;
        }

        let input_arcs = profile_eager_op_section("nary_op.materialize_inputs", || {
            tensors
                .iter()
                .map(|tensor| tensor.materialized_arc())
                .collect::<Result<Vec<_>>>()
        })?;
        let inputs: Vec<&Tensor> = profile_eager_op_section("nary_op.collect_inputs", || {
            input_arcs.iter().map(|tensor| tensor.as_ref()).collect()
        });
        let output = profile_eager_op_section("nary_op.exec_single_output", || {
            exec_single_output(&op, &inputs, &ctx)
        })?;

        let output = Arc::new(output);
        let outputs = vec![Arc::clone(&output)];
        let mut recorded = profile_eager_op_section("nary_op.record_outputs", || {
            record_eager_outputs(&op, &outputs, tensors)
        })?;
        let trace = recorded.traces.pop().ok_or_else(|| {
            Error::Internal(format!("expected one eager trace for {:?}, got 0", op))
        })?;
        let semantic_trace = recorded.semantic_traces.pop().flatten();
        let mut metadata_scopes = vec![Arc::clone(&recorded.metadata_scope)];
        for tensor in tensors {
            for scope in &tensor.metadata_scopes {
                push_metadata_scope(&mut metadata_scopes, Arc::clone(scope));
            }
        }

        let result = profile_eager_op_section("nary_op.new_tracked_result", || {
            Self::new_result_arc_with_semantic_trace(
                ctx,
                trace.key,
                output,
                trace.requires_grad,
                trace.trace,
                semantic_trace,
                metadata_scopes,
            )
        });
        if let Some(total_started) = total_started {
            record_eager_op_profile("nary_op.total", total_started.elapsed());
            maybe_print_eager_op_profile();
        }
        result
    }
}

fn validate_eager_axes(op: &'static str, rank: usize, axes: &[usize]) -> Result<()> {
    tenferro_tensor::validate::validate_unique_axes(op, "axis", rank, axes)
        .map_err(Error::TensorRuntime)
}

fn validate_eager_dot_general_config(
    _op: &'static str,
    config: &DotGeneralConfig,
    lhs_rank: usize,
    rhs_rank: usize,
) -> Result<()> {
    config
        .validate_dims_with_ranks(lhs_rank, rhs_rank)
        .map_err(Error::TensorRuntime)
}

fn empty_nary_input_error(op: &StdTensorOp) -> Error {
    Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
        eager_validation_op_name(op),
        "inputs",
        "operation requires at least one input tensor",
    ))
}

fn wrong_nary_input_count_error(op: &StdTensorOp, expected: usize, actual: usize) -> Error {
    Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
        eager_validation_op_name(op),
        "inputs",
        format!("operation expects {expected} inputs, got {actual}"),
    ))
}

fn eager_validation_op_name(op: &StdTensorOp) -> &'static str {
    match op {
        StdTensorOp::Concatenate { .. } => "concatenate",
        _ => "eager_nary_op",
    }
}

fn finite_real_factor(value: f64) -> Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(Error::TensorRuntime(
            tenferro_tensor::Error::invalid_argument(
                "scale_real",
                "factor",
                format!("real scalar must be finite, got {value}"),
            ),
        ))
    }
}

fn round_real_to_i64(value: f64) -> Result<i64> {
    let rounded = finite_real_factor(value)?.round();
    if rounded < i64::MIN as f64 || rounded >= -(i64::MIN as f64) {
        return Err(Error::TensorRuntime(
            tenferro_tensor::Error::invalid_argument(
                "scale_real",
                "factor",
                format!("rounded real scalar {rounded} is out of i64 range"),
            ),
        ));
    }
    Ok(rounded as i64)
}

fn round_real_to_i32(value: f64) -> Result<i32> {
    let rounded = round_real_to_i64(value)?;
    i32::try_from(rounded).map_err(|_| {
        Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
            "scale_real",
            "factor",
            format!("rounded real scalar {rounded} is out of i32 range"),
        ))
    })
}

fn bool_from_real(value: f64) -> Result<bool> {
    Ok(finite_real_factor(value)? != 0.0)
}
