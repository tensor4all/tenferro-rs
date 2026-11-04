//! Traced composite operations (activations, normalizations, softmax,
//! `take_along_axis`), built from primitives by [`crate::composite`].

use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{CompareDir, DType, GatherConfig};

use super::{apply_nullary, TracedTensor};
use crate::composite::{
    self, scalar_bytes, zero_pad_config, CompositeBinary, CompositeOps, CompositeReduce,
    CompositeUnary,
};
use crate::error::{Error, Result};

/// [`CompositeOps`] over traced graph construction.
pub(crate) struct TracedComposite;

impl CompositeOps for TracedComposite {
    type Value = TracedTensor;
    type Error = Error;

    fn dtype(&self, value: &TracedTensor) -> DType {
        value.dtype()
    }

    fn shape(&self, value: &TracedTensor) -> Result<Vec<usize>> {
        value.concrete_shape()
    }

    fn scalar(&mut self, dtype: DType, value: f64) -> Result<TracedTensor> {
        let bytes = scalar_bytes(dtype, value)?;
        apply_nullary(
            StdTensorOp::Constant { dtype, bytes },
            0,
            dtype,
            Some(vec![]),
        )
    }

    fn unary(&mut self, op: CompositeUnary, value: &TracedTensor) -> Result<TracedTensor> {
        match op {
            CompositeUnary::Neg => value.neg(),
            CompositeUnary::Exp => value.exp(),
            CompositeUnary::Log => value.log(),
            CompositeUnary::Log1p => value.log1p(),
            CompositeUnary::Tanh => value.tanh(),
            CompositeUnary::Erf => value.erf(),
            CompositeUnary::Rsqrt => value.rsqrt(),
        }
    }

    fn binary(
        &mut self,
        op: CompositeBinary,
        lhs: &TracedTensor,
        rhs: &TracedTensor,
    ) -> Result<TracedTensor> {
        match op {
            CompositeBinary::Add => lhs.add(rhs),
            CompositeBinary::Sub => lhs.sub(rhs),
            CompositeBinary::Mul => lhs.mul(rhs),
            CompositeBinary::Div => lhs.div(rhs),
            CompositeBinary::Maximum => lhs.maximum(rhs),
        }
    }

    fn compare(
        &mut self,
        lhs: &TracedTensor,
        rhs: &TracedTensor,
        dir: CompareDir,
    ) -> Result<TracedTensor> {
        lhs.compare(rhs, dir)
    }

    fn select(
        &mut self,
        condition: &TracedTensor,
        on_true: &TracedTensor,
        on_false: &TracedTensor,
    ) -> Result<TracedTensor> {
        TracedTensor::where_select(condition, on_true, on_false)
    }

    fn reduce(
        &mut self,
        op: CompositeReduce,
        value: &TracedTensor,
        axes: &[usize],
    ) -> Result<TracedTensor> {
        match op {
            CompositeReduce::Sum => value.reduce_sum(Some(axes)),
            CompositeReduce::Max => value.reduce_max(Some(axes)),
            CompositeReduce::SumSquares => value.reduce_sum_squares(Some(axes)),
        }
    }

    fn broadcast_in_dim(
        &mut self,
        value: &TracedTensor,
        shape: &[usize],
        dims: &[usize],
    ) -> Result<TracedTensor> {
        value.broadcast_in_dim(shape, dims)
    }

    fn reshape(&mut self, value: &TracedTensor, shape: &[usize]) -> Result<TracedTensor> {
        value.reshape(shape.to_vec())
    }

    fn concatenate(&mut self, values: &[&TracedTensor], axis: usize) -> Result<TracedTensor> {
        TracedTensor::concatenate(values, axis)
    }

    fn pad(&mut self, value: &TracedTensor, low: &[usize], high: &[usize]) -> Result<TracedTensor> {
        value.pad(zero_pad_config(low, high))
    }

    fn gather(
        &mut self,
        operand: &TracedTensor,
        indices: &TracedTensor,
        config: GatherConfig,
    ) -> Result<TracedTensor> {
        operand.gather(indices, config)
    }
}

impl TracedTensor {
    /// Logistic sigmoid `1 / (1 + exp(-x))`, for real `F32`/`F64` tensors.
    ///
    /// Evaluated as `1 / (1 + e)` for `x > 0` and `e / (1 + e)` otherwise,
    /// with `e = exp(-|x|)`, so no intermediate overflows and the derivative
    /// is finite everywhere (`sigmoid'(0) = 1/4`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// let x = TracedTensor::from_vec_col_major(vec![3], vec![-1.0_f64, 0.0, 1.0])?;
    /// let y = x.sigmoid()?;
    /// assert_eq!(y.try_concrete_shape(), Some(vec![3]));
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::TensorRuntime`] wrapping `UnsupportedDType` for
    /// complex, integer, or `Bool` input, [`Error::Validation`] with
    /// `InvalidArgument` when the input shape is symbolic, or
    /// [`Error::RuntimeStateSource`] when graph metadata registration fails.
    pub fn sigmoid(&self) -> Result<TracedTensor> {
        composite::sigmoid(&mut TracedComposite, self)
    }

    /// SiLU (swish) `x * sigmoid(x)`, for real `F32`/`F64` tensors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// let x = TracedTensor::from_vec_col_major(vec![2], vec![-1.0_f64, 1.0])?;
    /// let y = x.silu()?;
    /// assert_eq!(y.dtype(), tenferro_runtime::DType::F64);
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::TensorRuntime`] wrapping `UnsupportedDType` for
    /// complex, integer, or `Bool` input, [`Error::Validation`] with
    /// `InvalidArgument` when the input shape is symbolic, or
    /// [`Error::RuntimeStateSource`] when graph metadata registration fails.
    pub fn silu(&self) -> Result<TracedTensor> {
        composite::silu(&mut TracedComposite, self)
    }

    /// Softplus `log(1 + exp(x))` in the stable form `max(x, 0) + log1p(exp(-|x|))`.
    ///
    /// Never overflows; `softplus'(0) = 1/2` and `softplus''(0) = 1/4`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// let x = TracedTensor::from_vec_col_major(vec![2], vec![-1000.0_f64, 1000.0])?;
    /// let y = x.softplus()?;
    /// assert_eq!(y.try_concrete_shape(), Some(vec![2]));
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::TensorRuntime`] wrapping `UnsupportedDType` for
    /// complex, integer, or `Bool` input, [`Error::Validation`] with
    /// `InvalidArgument` when the input shape is symbolic, or
    /// [`Error::RuntimeStateSource`] when graph metadata registration fails.
    pub fn softplus(&self) -> Result<TracedTensor> {
        composite::softplus(&mut TracedComposite, self)
    }

    /// Exact GELU `x/2 * (1 + erf(x / sqrt(2)))` (PyTorch `approximate="none"`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// let x = TracedTensor::from_vec_col_major(vec![2], vec![-1.0_f64, 1.0])?;
    /// let y = x.gelu()?;
    /// assert_eq!(y.try_concrete_shape(), Some(vec![2]));
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::TensorRuntime`] wrapping `UnsupportedDType` for
    /// complex, integer, or `Bool` input, [`Error::Validation`] with
    /// `InvalidArgument` when the input shape is symbolic, or
    /// [`Error::RuntimeStateSource`] when graph metadata registration fails.
    pub fn gelu(&self) -> Result<TracedTensor> {
        composite::gelu(&mut TracedComposite, self)
    }

    /// GELU tanh approximation (PyTorch `approximate="tanh"`).
    ///
    /// `x/2 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// let x = TracedTensor::from_vec_col_major(vec![2], vec![-1.0_f64, 1.0])?;
    /// let y = x.gelu_tanh()?;
    /// assert_eq!(y.try_concrete_shape(), Some(vec![2]));
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::TensorRuntime`] wrapping `UnsupportedDType` for
    /// complex, integer, or `Bool` input, [`Error::Validation`] with
    /// `InvalidArgument` when the input shape is symbolic, or
    /// [`Error::RuntimeStateSource`] when graph metadata registration fails.
    pub fn gelu_tanh(&self) -> Result<TracedTensor> {
        composite::gelu_tanh(&mut TracedComposite, self)
    }

    /// Arithmetic mean over `axes` (`None` reduces every axis).
    ///
    /// Defined for float and complex tensors. A mean over zero elements is
    /// `NaN`; `Some(&[])` is the identity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    /// let y = x.reduce_mean(Some(&[0]))?;
    /// assert_eq!(y.try_concrete_shape(), Some(vec![2]));
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::TensorRuntime`] wrapping `UnsupportedDType` for
    /// integer or `Bool` input or `AxisOutOfBounds` / `DuplicateAxis` for
    /// invalid axes, [`Error::Validation`] with `InvalidArgument` when the
    /// input shape is symbolic, or [`Error::RuntimeStateSource`] when graph
    /// metadata registration fails.
    pub fn reduce_mean(&self, axes: Option<&[usize]>) -> Result<TracedTensor> {
        composite::reduce_mean(&mut TracedComposite, self, axes)
    }

    /// Max-subtracted softmax along `axis`, for real `F32`/`F64` tensors.
    ///
    /// A slice that is entirely `-inf` returns zeros (with a finite gradient)
    /// instead of `NaN`; a participating `NaN` or `+inf` makes its slice
    /// `NaN`; a zero-length `axis` returns an empty result.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// let x = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    /// let y = x.softmax(1)?;
    /// assert_eq!(y.try_concrete_shape(), Some(vec![2, 3]));
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::TensorRuntime`] wrapping `UnsupportedDType` for
    /// non-real input or `AxisOutOfBounds` for an invalid axis,
    /// [`Error::Validation`] with `InvalidArgument` when the input shape is
    /// symbolic, or [`Error::RuntimeStateSource`] when graph metadata
    /// registration fails.
    pub fn softmax(&self, axis: usize) -> Result<TracedTensor> {
        composite::softmax(&mut TracedComposite, self, axis)
    }

    /// Max-subtracted log-softmax along `axis`, for real `F32`/`F64` tensors.
    ///
    /// A slice that is entirely `-inf` returns `-inf` (with a finite
    /// gradient) instead of `NaN`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// let x = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])?;
    /// let y = x.log_softmax(0)?;
    /// assert_eq!(y.try_concrete_shape(), Some(vec![3]));
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::TensorRuntime`] wrapping `UnsupportedDType` for
    /// non-real input or `AxisOutOfBounds` for an invalid axis,
    /// [`Error::Validation`] with `InvalidArgument` when the input shape is
    /// symbolic, or [`Error::RuntimeStateSource`] when graph metadata
    /// registration fails.
    pub fn log_softmax(&self, axis: usize) -> Result<TracedTensor> {
        composite::log_softmax(&mut TracedComposite, self, axis)
    }

    /// Softmax along `axis` over the entries where the `Bool` `mask` is true.
    ///
    /// `mask` broadcasts to this tensor's shape. Masked-out entries get
    /// probability `0` and a zero gradient, whatever their value; a slice
    /// with no unmasked entry returns zeros with a zero gradient.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// let x = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])?;
    /// let mask = TracedTensor::from_vec_col_major(vec![3], vec![true, true, false])?;
    /// let y = x.masked_softmax(&mask, 0)?;
    /// assert_eq!(y.try_concrete_shape(), Some(vec![3]));
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::TensorRuntime`] wrapping `UnsupportedDType` for
    /// non-real input, `DTypeMismatch` for a non-`Bool` mask, `ShapeMismatch`
    /// for a mask that does not broadcast to the input, or `AxisOutOfBounds`;
    /// [`Error::Validation`] with `InvalidArgument` for symbolic shapes; or
    /// [`Error::RuntimeStateSource`] when graph metadata registration fails.
    pub fn masked_softmax(&self, mask: &TracedTensor, axis: usize) -> Result<TracedTensor> {
        composite::masked_softmax(&mut TracedComposite, self, mask, axis)
    }

    /// Log-softmax along `axis` over the entries where the `Bool` `mask` is true.
    ///
    /// Masked-out entries are `-inf` with a zero gradient; a slice with no
    /// unmasked entry is all `-inf` with a zero gradient.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// let x = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])?;
    /// let mask = TracedTensor::from_vec_col_major(vec![3], vec![true, false, true])?;
    /// let y = x.masked_log_softmax(&mask, 0)?;
    /// assert_eq!(y.try_concrete_shape(), Some(vec![3]));
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::TensorRuntime`] wrapping `UnsupportedDType` for
    /// non-real input, `DTypeMismatch` for a non-`Bool` mask, `ShapeMismatch`
    /// for a mask that does not broadcast to the input, or `AxisOutOfBounds`;
    /// [`Error::Validation`] with `InvalidArgument` for symbolic shapes; or
    /// [`Error::RuntimeStateSource`] when graph metadata registration fails.
    pub fn masked_log_softmax(&self, mask: &TracedTensor, axis: usize) -> Result<TracedTensor> {
        composite::masked_log_softmax(&mut TracedComposite, self, mask, axis)
    }

    /// Layer normalization along `axis` with optional affine `weight` / `bias`.
    ///
    /// `(x - mean) / sqrt(var + eps) * weight + bias`, with the biased
    /// variance of the centered values. `weight` and `bias` are rank-1 of
    /// length `shape[axis]`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// let x = TracedTensor::from_vec_col_major(vec![4, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 0.0, 0.0, 1.0, 1.0])?;
    /// let w = TracedTensor::from_vec_col_major(vec![4], vec![1.0_f64, 1.0, 2.0, 2.0])?;
    /// let y = x.layer_norm(0, Some(&w), None, 1e-5)?;
    /// assert_eq!(y.try_concrete_shape(), Some(vec![4, 2]));
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::TensorRuntime`] wrapping `UnsupportedDType` for
    /// non-real input, `AxisOutOfBounds`, `InvalidArgument` for a negative or
    /// non-finite `eps`, or `DTypeMismatch` / `ShapeMismatch` for a bad
    /// weight or bias; [`Error::Validation`] with `InvalidArgument` for
    /// symbolic shapes; or [`Error::RuntimeStateSource`] when graph metadata
    /// registration fails.
    pub fn layer_norm(
        &self,
        axis: usize,
        weight: Option<&TracedTensor>,
        bias: Option<&TracedTensor>,
        eps: f64,
    ) -> Result<TracedTensor> {
        composite::layer_norm(&mut TracedComposite, self, axis, weight, bias, eps)
    }

    /// RMS normalization along `axis` with optional affine `weight` / `bias`.
    ///
    /// `x / sqrt(mean(x^2) + eps) * weight + bias`; `weight` and `bias` are
    /// rank-1 of length `shape[axis]`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    /// let y = x.rms_norm(0, None, None, 1e-6)?;
    /// assert_eq!(y.try_concrete_shape(), Some(vec![2, 2]));
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::TensorRuntime`] wrapping `UnsupportedDType` for
    /// non-real input, `AxisOutOfBounds`, `InvalidArgument` for a negative or
    /// non-finite `eps`, or `DTypeMismatch` / `ShapeMismatch` for a bad
    /// weight or bias; [`Error::Validation`] with `InvalidArgument` for
    /// symbolic shapes; or [`Error::RuntimeStateSource`] when graph metadata
    /// registration fails.
    pub fn rms_norm(
        &self,
        axis: usize,
        weight: Option<&TracedTensor>,
        bias: Option<&TracedTensor>,
        eps: f64,
    ) -> Result<TracedTensor> {
        composite::rms_norm(&mut TracedComposite, self, axis, weight, bias, eps)
    }

    /// NumPy-style `take_along_axis` over `gather`.
    ///
    /// `out[.., i, ..] = self[.., indices[.., i, ..], ..]` along `axis`.
    /// `indices` (I32/I64) has this tensor's rank; each other dimension is
    /// either this tensor's extent (batch-varying indices) or `1` (the whole
    /// extent is taken). Indices must be in bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_runtime::TracedTensor;
    /// // Gather whole rows of each matrix in a [2, 2, batch=2] stack.
    /// let x = TracedTensor::from_vec_col_major(vec![2, 2, 2], (0..8).map(f64::from).collect::<Vec<_>>())?;
    /// let rows = TracedTensor::from_vec_col_major(vec![2, 1, 2], vec![1_i64, 0, 0, 0])?;
    /// let y = x.take_along_axis(&rows, 0)?;
    /// assert_eq!(y.try_concrete_shape(), Some(vec![2, 2, 2]));
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::TensorRuntime`] wrapping `RankMismatch` /
    /// `ShapeMismatch` for incompatible index shapes, `AxisOutOfBounds`,
    /// `UnsupportedDType` for a non-integer index dtype, or `InvalidArgument`
    /// when taking from a zero-length axis; [`Error::Validation`] with
    /// `InvalidArgument` for symbolic shapes; or
    /// [`Error::RuntimeStateSource`] when graph metadata registration fails.
    pub fn take_along_axis(&self, indices: &TracedTensor, axis: usize) -> Result<TracedTensor> {
        composite::take_along_axis(&mut TracedComposite, self, indices, axis)
    }
}
