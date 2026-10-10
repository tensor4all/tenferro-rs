//! Eager composite operations (activations, normalizations, softmax,
//! `take_along_axis`) on the borrowed [`EagerSession`].
//!
//! The formulation and edge-case policy are shared with the traced and
//! concrete-session surfaces through `tenferro_runtime::composite`; AD follows
//! from the recorded primitives.

use tenferro_runtime::composite::{
    self, scalar_tensor, zero_pad_config, CompositeBinary, CompositeOps, CompositeReduce,
    CompositeUnary,
};
use tenferro_tensor::{ActivationOp, CompareDir, DType, GatherConfig};

use super::{EagerSession, EagerTensor};
use crate::{Error, Result};

/// [`CompositeOps`] over a borrowed eager session.
struct EagerComposite<'s, 'a> {
    session: &'s mut EagerSession<'a>,
}

impl CompositeOps for EagerComposite<'_, '_> {
    type Value = EagerTensor;
    type Error = Error;

    fn dtype(&self, value: &EagerTensor) -> DType {
        value.dtype()
    }

    fn shape(&self, value: &EagerTensor) -> Result<Vec<usize>> {
        Ok(value.shape().to_vec())
    }

    fn scalar(&mut self, dtype: DType, value: f64) -> Result<EagerTensor> {
        self.session
            .constant_from_host(scalar_tensor(dtype, value)?)
    }

    fn unary(&mut self, op: CompositeUnary, value: &EagerTensor) -> Result<EagerTensor> {
        let session = &mut *self.session;
        match op {
            CompositeUnary::Neg => session.neg(value),
            CompositeUnary::Exp => session.exp(value),
            CompositeUnary::Log => session.log(value),
            CompositeUnary::Log1p => session.log1p(value),
            CompositeUnary::Tanh => session.tanh(value),
            CompositeUnary::Erf => session.erf(value),
            CompositeUnary::Rsqrt => session.rsqrt(value),
        }
    }

    fn binary(
        &mut self,
        op: CompositeBinary,
        lhs: &EagerTensor,
        rhs: &EagerTensor,
    ) -> Result<EagerTensor> {
        let session = &mut *self.session;
        match op {
            CompositeBinary::Add => session.add(lhs, rhs),
            CompositeBinary::Sub => session.sub(lhs, rhs),
            CompositeBinary::Mul => session.mul(lhs, rhs),
            CompositeBinary::Div => session.div(lhs, rhs),
            CompositeBinary::Maximum => session.maximum(lhs, rhs),
        }
    }

    fn compare(
        &mut self,
        lhs: &EagerTensor,
        rhs: &EagerTensor,
        dir: CompareDir,
    ) -> Result<EagerTensor> {
        self.session.compare(lhs, rhs, dir)
    }

    fn select(
        &mut self,
        condition: &EagerTensor,
        on_true: &EagerTensor,
        on_false: &EagerTensor,
    ) -> Result<EagerTensor> {
        self.session.where_select(condition, on_true, on_false)
    }

    fn reduce(
        &mut self,
        op: CompositeReduce,
        value: &EagerTensor,
        axes: &[usize],
    ) -> Result<EagerTensor> {
        match op {
            CompositeReduce::Sum => self.session.reduce_sum(value, Some(axes)),
            CompositeReduce::Max => self.session.reduce_max(value, Some(axes)),
            CompositeReduce::SumSquares => self.session.reduce_sum_squares(value, Some(axes)),
        }
    }

    fn broadcast_in_dim(
        &mut self,
        value: &EagerTensor,
        shape: &[usize],
        dims: &[usize],
    ) -> Result<EagerTensor> {
        self.session.broadcast_in_dim(value, shape, dims)
    }

    fn reshape(&mut self, value: &EagerTensor, shape: &[usize]) -> Result<EagerTensor> {
        self.session.reshape(value, shape.to_vec())
    }

    fn concatenate(&mut self, values: &[&EagerTensor], axis: usize) -> Result<EagerTensor> {
        self.session.concatenate(values, axis)
    }

    fn pad(&mut self, value: &EagerTensor, low: &[usize], high: &[usize]) -> Result<EagerTensor> {
        self.session.pad(value, zero_pad_config(low, high))
    }

    fn gather(
        &mut self,
        operand: &EagerTensor,
        indices: &EagerTensor,
        config: GatherConfig,
    ) -> Result<EagerTensor> {
        self.session.gather(operand, indices, config)
    }
}

impl<'a> EagerSession<'a> {
    fn composite(&mut self) -> EagerComposite<'_, 'a> {
        EagerComposite { session: self }
    }

    /// Run a fused activation through the backend when no AD record is needed.
    ///
    /// Returns `Ok(None)` when the operands are tracked, semantic capture is
    /// active, or the backend declines the fused form; the caller then records
    /// the shared composite formulation instead.
    fn fused_activation(
        &mut self,
        op: ActivationOp,
        input: &EagerTensor,
    ) -> Result<Option<EagerTensor>> {
        if !crate::eager_ops::untracked_fast_path_allowed(&[input]) {
            return Ok(None);
        }
        let read = input.tensor_read();
        let Some(output) = self.backend.fused_activation_read(op, read)? else {
            return Ok(None);
        };
        Ok(Some(EagerTensor::new_untracked_result(
            std::sync::Arc::clone(&input.ctx),
            output,
        )?))
    }

    /// Logistic sigmoid `1 / (1 + exp(-x))`, overflow-free.
    ///
    /// Evaluated as `1 / (1 + e)` for `x > 0` and `e / (1 + e)` otherwise, with
    /// `e = exp(-|x|)`; the derivative is finite everywhere (`sigmoid'(0) = 1/4`).
    /// Real `F32`/`F64` only.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// let ctx = EagerRuntime::new()?;
    /// let y = ctx.with_eager_session(|s| {
    ///     let x = s.constant_from_host(Tensor::from_vec_col_major(vec![3], vec![-700.0_f64, 0.0, 1000.0])?)?;
    ///     s.sigmoid(&x)
    /// })?;
    /// let y = y.value()?;
    /// let y = y.as_slice::<f64>()?;
    /// assert_eq!(y[1], 0.5);
    /// assert!(y[0] > 0.0 && y[0] < 1e-300);
    /// assert_eq!(y[2], 1.0);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a typed `UnsupportedDType` error for complex, integer, or `Bool`
    /// input, [`Error::ContextMismatch`] for a tensor from another runtime, or a
    /// backend error.
    pub fn sigmoid(&mut self, input: &EagerTensor) -> Result<EagerTensor> {
        if let Some(out) = self.fused_activation(ActivationOp::Sigmoid, input)? {
            return Ok(out);
        }
        composite::sigmoid(&mut self.composite(), input)
    }

    /// SiLU (swish) `x * sigmoid(x)`.
    ///
    /// Real `F32`/`F64` only.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// let ctx = EagerRuntime::new()?;
    /// let y = ctx.with_eager_session(|s| {
    ///     let x = s.constant_from_host(Tensor::from_vec_col_major(vec![3], vec![-1.0_f64, 0.0, 1.0])?)?;
    ///     s.silu(&x)
    /// })?;
    /// let y = y.value()?;
    /// let y = y.as_slice::<f64>()?;
    /// assert_eq!(y[1], 0.0);
    /// assert!((y[2] - 1.0 / (1.0 + (-1.0_f64).exp())).abs() < 1e-15);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a typed `UnsupportedDType` error for complex, integer, or `Bool`
    /// input, [`Error::ContextMismatch`] for a tensor from another runtime, or a
    /// backend error.
    pub fn silu(&mut self, input: &EagerTensor) -> Result<EagerTensor> {
        if let Some(out) = self.fused_activation(ActivationOp::Silu, input)? {
            return Ok(out);
        }
        composite::silu(&mut self.composite(), input)
    }

    /// Softplus `log(1 + exp(x))` in the stable form `max(x, 0) + log1p(exp(-|x|))`.
    ///
    /// Never overflows; `softplus'(0) = 1/2` and `softplus''(0) = 1/4`. Real `F32`/`F64` only.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// let ctx = EagerRuntime::new()?;
    /// let y = ctx.with_eager_session(|s| {
    ///     let x = s.constant_from_host(Tensor::from_vec_col_major(vec![3], vec![-1000.0_f64, 0.0, 1000.0])?)?;
    ///     s.softplus(&x)
    /// })?;
    /// let y = y.value()?;
    /// let y = y.as_slice::<f64>()?;
    /// assert_eq!(y[0], 0.0);
    /// assert!((y[1] - 2.0_f64.ln()).abs() < 1e-15);
    /// assert_eq!(y[2], 1000.0);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a typed `UnsupportedDType` error for complex, integer, or `Bool`
    /// input, [`Error::ContextMismatch`] for a tensor from another runtime, or a
    /// backend error.
    pub fn softplus(&mut self, input: &EagerTensor) -> Result<EagerTensor> {
        if let Some(out) = self.fused_activation(ActivationOp::Softplus, input)? {
            return Ok(out);
        }
        composite::softplus(&mut self.composite(), input)
    }

    /// Exact GELU `x/2 * (1 + erf(x / sqrt(2)))` (PyTorch `approximate="none"`).
    ///
    /// Real `F32`/`F64` only.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// let ctx = EagerRuntime::new()?;
    /// let y = ctx.with_eager_session(|s| {
    ///     let x = s.constant_from_host(Tensor::from_vec_col_major(vec![3], vec![-1.0_f64, 0.0, 1.0])?)?;
    ///     s.gelu(&x)
    /// })?;
    /// let y = y.value()?;
    /// let y = y.as_slice::<f64>()?;
    /// assert_eq!(y[1], 0.0);
    /// assert!((y[2] - 0.841_344_746_068_542_9).abs() < 1e-15);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a typed `UnsupportedDType` error for complex, integer, or `Bool`
    /// input, [`Error::ContextMismatch`] for a tensor from another runtime, or a
    /// backend error.
    pub fn gelu(&mut self, input: &EagerTensor) -> Result<EagerTensor> {
        if let Some(out) = self.fused_activation(ActivationOp::Gelu, input)? {
            return Ok(out);
        }
        composite::gelu(&mut self.composite(), input)
    }

    /// GELU tanh approximation (PyTorch `approximate="tanh"`).
    ///
    /// `x/2 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))`; real `F32`/`F64` only.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// let ctx = EagerRuntime::new()?;
    /// let y = ctx.with_eager_session(|s| {
    ///     let x = s.constant_from_host(Tensor::from_vec_col_major(vec![3], vec![-1.0_f64, 0.0, 1.0])?)?;
    ///     s.gelu_tanh(&x)
    /// })?;
    /// let y = y.value()?;
    /// let y = y.as_slice::<f64>()?;
    /// assert_eq!(y[1], 0.0);
    /// assert!((y[2] - 0.841_191_990_608_276_8).abs() < 1e-12);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a typed `UnsupportedDType` error for complex, integer, or `Bool`
    /// input, [`Error::ContextMismatch`] for a tensor from another runtime, or a
    /// backend error.
    pub fn gelu_tanh(&mut self, input: &EagerTensor) -> Result<EagerTensor> {
        if let Some(out) = self.fused_activation(ActivationOp::GeluTanh, input)? {
            return Ok(out);
        }
        composite::gelu_tanh(&mut self.composite(), input)
    }

    /// Arithmetic mean over `axes` (`None` reduces every axis).
    ///
    /// Float and complex dtypes. The sum is divided by the element count; a mean
    /// over zero elements is `NaN`, and `Some(&[])` is the identity.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// let ctx = EagerRuntime::new()?;
    /// let y = ctx.with_eager_session(|s| {
    ///     let x = s.constant_from_host(Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?)?;
    ///     s.reduce_mean(&x, Some(&[1]))
    /// })?;
    /// assert_eq!(y.value()?.as_slice::<f64>()?, &[2.0, 3.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a typed `UnsupportedDType` error for integer or `Bool` input, an
    /// `AxisOutOfBounds` / `DuplicateAxis` validation error for invalid axes,
    /// [`Error::ContextMismatch`] for a tensor from another runtime, or a backend error.
    pub fn reduce_mean(
        &mut self,
        input: &EagerTensor,
        axes: Option<&[usize]>,
    ) -> Result<EagerTensor> {
        composite::reduce_mean(&mut self.composite(), input, axes)
    }

    /// Max-subtracted softmax along `axis`.
    ///
    /// A slice that is entirely `-inf` returns zeros with a finite gradient instead
    /// of `NaN`; a participating `NaN` or `+inf` makes its slice `NaN`; a
    /// zero-length `axis` returns an empty result. Real `F32`/`F64` only.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// let ctx = EagerRuntime::new()?;
    /// let y = ctx.with_eager_session(|s| {
    ///     let x = s.constant_from_host(Tensor::from_vec_col_major(vec![2], vec![0.0_f64, f64::NEG_INFINITY])?)?;
    ///     s.softmax(&x, 0)
    /// })?;
    /// assert_eq!(y.value()?.as_slice::<f64>()?, &[1.0, 0.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a typed `UnsupportedDType` error for non-real input, an
    /// `AxisOutOfBounds` validation error for an invalid axis,
    /// [`Error::ContextMismatch`] for a tensor from another runtime, or a backend error.
    pub fn softmax(&mut self, input: &EagerTensor, axis: usize) -> Result<EagerTensor> {
        composite::softmax(&mut self.composite(), input, axis)
    }

    /// Max-subtracted log-softmax along `axis`.
    ///
    /// A slice that is entirely `-inf` returns `-inf` with a finite gradient
    /// instead of `NaN`. Real `F32`/`F64` only.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// let ctx = EagerRuntime::new()?;
    /// let y = ctx.with_eager_session(|s| {
    ///     let x = s.constant_from_host(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 1.0])?)?;
    ///     s.log_softmax(&x, 0)
    /// })?;
    /// assert_eq!(y.value()?.as_slice::<f64>()?, &[-std::f64::consts::LN_2; 2]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a typed `UnsupportedDType` error for non-real input, an
    /// `AxisOutOfBounds` validation error for an invalid axis,
    /// [`Error::ContextMismatch`] for a tensor from another runtime, or a backend error.
    pub fn log_softmax(&mut self, input: &EagerTensor, axis: usize) -> Result<EagerTensor> {
        composite::log_softmax(&mut self.composite(), input, axis)
    }

    /// Softmax along `axis` over the entries where the `Bool` `mask` is true.
    ///
    /// `mask` broadcasts to the input shape. Masked-out entries get probability `0`
    /// and a zero gradient whatever their value; a slice with no unmasked entry
    /// returns zeros with a zero gradient.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// let ctx = EagerRuntime::new()?;
    /// let y = ctx.with_eager_session(|s| {
    ///     let x = s.constant_from_host(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 1.0, f64::NAN])?)?;
    ///     let mask = s.constant_from_host(Tensor::from_vec_col_major(vec![3], vec![true, true, false])?)?;
    ///     s.masked_softmax(&x, &mask, 0)
    /// })?;
    /// assert_eq!(y.value()?.as_slice::<f64>()?, &[0.5, 0.5, 0.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a typed `UnsupportedDType` error for non-real input, a
    /// `DTypeMismatch` validation error for a non-`Bool` mask, `ShapeMismatch` for
    /// a mask that does not broadcast to the input, `AxisOutOfBounds` for an invalid
    /// axis, [`Error::ContextMismatch`] for a tensor from another runtime, or a
    /// backend error.
    pub fn masked_softmax(
        &mut self,
        input: &EagerTensor,
        mask: &EagerTensor,
        axis: usize,
    ) -> Result<EagerTensor> {
        composite::masked_softmax(&mut self.composite(), input, mask, axis)
    }

    /// Log-softmax along `axis` over the entries where the `Bool` `mask` is true.
    ///
    /// Masked-out entries are `-inf` with a zero gradient; a slice with no unmasked
    /// entry is all `-inf` with a zero gradient.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// let ctx = EagerRuntime::new()?;
    /// let y = ctx.with_eager_session(|s| {
    ///     let x = s.constant_from_host(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0])?)?;
    ///     let mask = s.constant_from_host(Tensor::from_vec_col_major(vec![2], vec![true, false])?)?;
    ///     s.masked_log_softmax(&x, &mask, 0)
    /// })?;
    /// assert_eq!(y.value()?.as_slice::<f64>()?, &[0.0, f64::NEG_INFINITY]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a typed `UnsupportedDType` error for non-real input, a
    /// `DTypeMismatch` validation error for a non-`Bool` mask, `ShapeMismatch` for
    /// a mask that does not broadcast to the input, `AxisOutOfBounds` for an invalid
    /// axis, [`Error::ContextMismatch`] for a tensor from another runtime, or a
    /// backend error.
    pub fn masked_log_softmax(
        &mut self,
        input: &EagerTensor,
        mask: &EagerTensor,
        axis: usize,
    ) -> Result<EagerTensor> {
        composite::masked_log_softmax(&mut self.composite(), input, mask, axis)
    }

    /// Layer normalization along `axis` with optional affine `weight` / `bias`.
    ///
    /// `(x - mean) / sqrt(var + eps) * weight + bias` with the biased variance of the
    /// centered values; `weight` and `bias` are rank-1 of length `shape[axis]`. A
    /// zero-variance slice normalizes to `0` (then `bias`) with a finite gradient
    /// when `eps > 0`. Real `F32`/`F64` only.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// let ctx = EagerRuntime::new()?;
    /// let y = ctx.with_eager_session(|s| {
    ///     let x = s.constant_from_host(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 3.0])?)?;
    ///     s.layer_norm(&x, 0, None, None, 0.0)
    /// })?;
    /// assert_eq!(y.value()?.as_slice::<f64>()?, &[-1.0, 1.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a typed `UnsupportedDType` error for non-real input, an
    /// `AxisOutOfBounds` validation error for an invalid axis, `InvalidArgument` for
    /// a negative or non-finite `eps`, `DTypeMismatch` / `ShapeMismatch` for a weight
    /// or bias that is not a same-dtype vector of the axis length,
    /// [`Error::ContextMismatch`] for a tensor from another runtime, or a backend error.
    pub fn layer_norm(
        &mut self,
        input: &EagerTensor,
        axis: usize,
        weight: Option<&EagerTensor>,
        bias: Option<&EagerTensor>,
        eps: f64,
    ) -> Result<EagerTensor> {
        composite::layer_norm(&mut self.composite(), input, axis, weight, bias, eps)
    }

    /// RMS normalization along `axis` with optional affine `weight` / `bias`.
    ///
    /// `x / sqrt(mean(x^2) + eps) * weight + bias`; `weight` and `bias` are rank-1 of
    /// length `shape[axis]`. An all-zero slice normalizes to `0` (then `bias`) with
    /// a finite gradient when `eps > 0`. Real `F32`/`F64` only.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// let ctx = EagerRuntime::new()?;
    /// let y = ctx.with_eager_session(|s| {
    ///     let x = s.constant_from_host(Tensor::from_vec_col_major(vec![2], vec![0.0_f64, 0.0])?)?;
    ///     s.rms_norm(&x, 0, None, None, 1e-6)
    /// })?;
    /// assert_eq!(y.value()?.as_slice::<f64>()?, &[0.0, 0.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a typed `UnsupportedDType` error for non-real input, an
    /// `AxisOutOfBounds` validation error for an invalid axis, `InvalidArgument` for
    /// a negative or non-finite `eps`, `DTypeMismatch` / `ShapeMismatch` for a weight
    /// or bias that is not a same-dtype vector of the axis length,
    /// [`Error::ContextMismatch`] for a tensor from another runtime, or a backend error.
    pub fn rms_norm(
        &mut self,
        input: &EagerTensor,
        axis: usize,
        weight: Option<&EagerTensor>,
        bias: Option<&EagerTensor>,
        eps: f64,
    ) -> Result<EagerTensor> {
        composite::rms_norm(&mut self.composite(), input, axis, weight, bias, eps)
    }

    /// NumPy-style `take_along_axis` over `gather`.
    ///
    /// `out[.., i, ..] = input[.., indices[.., i, ..], ..]` along `axis`. `indices`
    /// (I32/I64) has the input's rank; every other dimension is either the input's
    /// extent (batch-varying indices) or `1` (the whole extent is taken). Indices
    /// must be in bounds. The gradient flows to `input` only.
    ///
    /// # Examples
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// let ctx = EagerRuntime::new()?;
    /// // Per-batch row gather: out[i, j, b] = x[idx[i, b], j, b].
    /// let y = ctx.with_eager_session(|s| {
    ///     let x = s.constant_from_host(Tensor::from_vec_col_major(vec![2, 2, 2], (0..8).map(f64::from).collect::<Vec<_>>())?)?;
    ///     let idx = s.constant_from_host(Tensor::from_vec_col_major(vec![2, 1, 2], vec![1_i64, 0, 0, 0])?)?;
    ///     s.take_along_axis(&x, &idx, 0)
    /// })?;
    /// assert_eq!(y.value()?.as_slice::<f64>()?, &[1.0, 0.0, 3.0, 2.0, 4.0, 4.0, 6.0, 6.0]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a `RankMismatch` / `ShapeMismatch` validation error for incompatible
    /// index shapes, `AxisOutOfBounds` for an invalid axis, `InvalidArgument` when
    /// taking from a zero-length axis, a typed `UnsupportedDType` error for a
    /// non-integer index dtype, [`Error::ContextMismatch`] for a tensor from another
    /// runtime, or a backend error.
    pub fn take_along_axis(
        &mut self,
        input: &EagerTensor,
        indices: &EagerTensor,
        axis: usize,
    ) -> Result<EagerTensor> {
        composite::take_along_axis(&mut self.composite(), input, indices, axis)
    }
}
