//! Composite operations shared by the traced, eager and concrete-session surfaces.
//!
//! `sigmoid`, `silu`, `softplus`, `gelu`, `gelu_tanh`, `reduce_mean`,
//! `softmax`, `log_softmax`, their masked forms, `layer_norm`, `rms_norm` and
//! `take_along_axis` are compositions of existing primitives. Each is written
//! once here over [`CompositeOps`], which every public surface implements, so
//! the numerical formulation and the edge-case policy (documented in
//! `docs/spec/tensor-semantics.md`) are identical on every surface and backend.
//! AD of the traced and eager forms follows from the primitives' rules.
//!
//! This module is public only so `tenferro-ad` can implement [`CompositeOps`]
//! for its eager session; it is not a user-facing API.

use num_complex::{Complex32, Complex64};
use tenferro_tensor::{CompareDir, DType, Error, GatherConfig, PadConfig, Tensor};

pub(crate) mod session;

/// Which broadcast route a composite took (#2143 item 1).
///
/// Hidden observation surface for test and benchmark harnesses: it is how a
/// caller checks that the session softmax does not enter the materializing
/// `broadcast_in_dim_read` route. Recording is off by default.
#[doc(hidden)]
pub use session::BroadcastRoute;

/// Start or stop recording composite broadcast routes on every thread.
#[doc(hidden)]
pub fn set_broadcast_route_observation(enabled: bool) {
    session::set_broadcast_route_observation(enabled);
}

/// Take the composite broadcast routes recorded on this thread.
#[doc(hidden)]
pub fn take_broadcast_routes() -> Vec<BroadcastRoute> {
    session::take_broadcast_routes()
}

/// Elementwise unary primitives a composite may emit.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::composite::CompositeUnary;
/// assert_ne!(CompositeUnary::Exp, CompositeUnary::Log1p);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompositeUnary {
    Neg,
    Exp,
    Log,
    Log1p,
    Tanh,
    Erf,
    Rsqrt,
}

/// Elementwise binary primitives a composite may emit (NumPy broadcasting).
///
/// # Examples
///
/// ```
/// use tenferro_runtime::composite::CompositeBinary;
/// assert_ne!(CompositeBinary::Add, CompositeBinary::Div);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompositeBinary {
    Add,
    Sub,
    Mul,
    Div,
    /// NaN-propagating elementwise maximum.
    Maximum,
}

/// Reductions a composite may emit.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::composite::CompositeReduce;
/// assert_ne!(CompositeReduce::Sum, CompositeReduce::Max);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompositeReduce {
    Sum,
    Max,
    /// Sum of squares of a real tensor.
    SumSquares,
}

/// The primitive vocabulary a public surface provides to the shared composites.
///
/// Its methods are hidden like this module: they are the per-surface wiring,
/// exercised through the composite functions' examples.
///
/// Binary, compare and select operations follow the surface's NumPy-style
/// broadcasting; reductions take explicit, validated axes.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::composite::CompositeOps;
///
/// fn uses_composites<O: CompositeOps>(_ops: &mut O) {}
/// ```
pub trait CompositeOps {
    /// Tensor handle of the surface.
    type Value;
    /// Error type of the surface.
    type Error: From<Error>;

    /// Dtype of `value`.
    #[doc(hidden)]
    fn dtype(&self, value: &Self::Value) -> DType;

    /// Concrete shape of `value`.
    ///
    /// # Errors
    ///
    /// Returns a `ValidationError::InvalidArgument` when the shape is symbolic.
    #[doc(hidden)]
    fn shape(&self, value: &Self::Value) -> Result<Vec<usize>, Self::Error>;

    /// A rank-0 constant of a float or complex `dtype` (non-finite values allowed).
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedDType`] for a dtype without a real embedding,
    /// or the surface's constant-registration or upload error.
    #[doc(hidden)]
    fn scalar(&mut self, dtype: DType, value: f64) -> Result<Self::Value, Self::Error>;

    /// Apply an elementwise unary primitive.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedDType`] for a dtype the primitive does not
    /// support, or [`Error::BackendSource`] for a backend failure.
    #[doc(hidden)]
    fn unary(
        &mut self,
        op: CompositeUnary,
        value: &Self::Value,
    ) -> Result<Self::Value, Self::Error>;

    /// Apply an elementwise binary primitive with broadcasting.
    ///
    /// # Errors
    ///
    /// Returns `ValidationError::ShapeMismatch` for incompatible shapes,
    /// [`Error::UnsupportedDType`] for an unsupported dtype, or
    /// [`Error::BackendSource`] for a backend failure.
    #[doc(hidden)]
    fn binary(
        &mut self,
        op: CompositeBinary,
        lhs: &Self::Value,
        rhs: &Self::Value,
    ) -> Result<Self::Value, Self::Error>;

    /// Compare with broadcasting, producing `Bool`.
    ///
    /// # Errors
    ///
    /// Returns `ValidationError::ShapeMismatch` for incompatible shapes,
    /// [`Error::UnsupportedDType`] for an unsupported dtype, or
    /// [`Error::BackendSource`] for a backend failure.
    #[doc(hidden)]
    fn compare(
        &mut self,
        lhs: &Self::Value,
        rhs: &Self::Value,
        dir: CompareDir,
    ) -> Result<Self::Value, Self::Error>;

    /// Select `on_true` where `condition` holds, else `on_false`, with broadcasting.
    ///
    /// # Errors
    ///
    /// Returns `ValidationError::ShapeMismatch` for incompatible shapes,
    /// [`Error::UnsupportedDType`] for an unsupported dtype, or
    /// [`Error::BackendSource`] for a backend failure.
    #[doc(hidden)]
    fn select(
        &mut self,
        condition: &Self::Value,
        on_true: &Self::Value,
        on_false: &Self::Value,
    ) -> Result<Self::Value, Self::Error>;

    /// Reduce over validated, non-empty `axes`.
    ///
    /// # Errors
    ///
    /// Returns `ValidationError::AxisOutOfBounds` for an invalid axis,
    /// [`Error::UnsupportedDType`] for an unsupported dtype, or
    /// [`Error::BackendSource`] for a backend failure.
    #[doc(hidden)]
    fn reduce(
        &mut self,
        op: CompositeReduce,
        value: &Self::Value,
        axes: &[usize],
    ) -> Result<Self::Value, Self::Error>;

    /// StableHLO `broadcast_in_dim`.
    ///
    /// # Errors
    ///
    /// Returns `ValidationError::ShapeMismatch` for incompatible shapes or
    /// [`Error::BackendSource`] for a backend failure.
    #[doc(hidden)]
    fn broadcast_in_dim(
        &mut self,
        value: &Self::Value,
        shape: &[usize],
        dims: &[usize],
    ) -> Result<Self::Value, Self::Error>;

    /// Broadcast `value` into `shape`, consuming `value`.
    ///
    /// A surface whose value owns its tensor and can carry a metadata-only view
    /// layout overrides this to return a view; the default materializes through
    /// [`Self::broadcast_in_dim`]. Callers that still need `value` afterwards, or
    /// that only hold a borrow of it, keep using `broadcast_in_dim`.
    ///
    /// # Errors
    ///
    /// Returns `ValidationError::ShapeMismatch` for incompatible shapes or
    /// [`Error::BackendSource`] for a backend failure.
    #[doc(hidden)]
    fn broadcast_in_dim_owned(
        &mut self,
        value: Self::Value,
        shape: &[usize],
        dims: &[usize],
    ) -> Result<Self::Value, Self::Error> {
        self.broadcast_in_dim(&value, shape, dims)
    }

    /// Reshape to a shape with the same element count.
    ///
    /// # Errors
    ///
    /// Returns `ValidationError::ShapeMismatch` for incompatible shapes or
    /// [`Error::BackendSource`] for a backend failure.
    #[doc(hidden)]
    fn reshape(&mut self, value: &Self::Value, shape: &[usize])
        -> Result<Self::Value, Self::Error>;

    /// Concatenate along `axis`.
    ///
    /// # Errors
    ///
    /// Returns `ValidationError::ShapeMismatch` for incompatible shapes,
    /// [`Error::UnsupportedDType`] for an unsupported dtype, or
    /// [`Error::BackendSource`] for a backend failure.
    #[doc(hidden)]
    fn concatenate(
        &mut self,
        values: &[&Self::Value],
        axis: usize,
    ) -> Result<Self::Value, Self::Error>;

    /// Zero-pad with `low` / `high` elements per axis (no interior padding).
    ///
    /// # Errors
    ///
    /// Returns `ValidationError::InvalidArgument` for an invalid configuration
    /// or [`Error::BackendSource`] for a backend failure.
    #[doc(hidden)]
    fn pad(
        &mut self,
        value: &Self::Value,
        low: &[usize],
        high: &[usize],
    ) -> Result<Self::Value, Self::Error>;

    /// StableHLO `gather`.
    ///
    /// # Errors
    ///
    /// Returns `ValidationError::InvalidArgument` for an invalid configuration
    /// or index, or [`Error::BackendSource`] for a backend failure.
    #[doc(hidden)]
    fn gather(
        &mut self,
        operand: &Self::Value,
        indices: &Self::Value,
        config: GatherConfig,
    ) -> Result<Self::Value, Self::Error>;
}

/// The [`PadConfig`] for zero edge padding of `low` / `high` elements per axis.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::composite::zero_pad_config;
/// let config = zero_pad_config(&[1], &[2]);
/// assert_eq!(config.edge_padding_low, vec![1]);
/// assert_eq!(config.interior_padding, vec![0]);
/// ```
#[must_use]
pub fn zero_pad_config(low: &[usize], high: &[usize]) -> PadConfig {
    // INVARIANT: composites pad by at most the index-tuple length (rank + 1).
    let edges = |values: &[usize]| values.iter().map(|&v| v as i64).collect();
    PadConfig {
        edge_padding_low: edges(low),
        edge_padding_high: edges(high),
        interior_padding: vec![0; low.len()],
    }
}

/// Little-endian bytes of the scalar `value` in a float or complex `dtype`.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::composite::scalar_bytes;
/// use tenferro_tensor::DType;
/// assert_eq!(scalar_bytes(DType::F64, 1.5)?, 1.5_f64.to_le_bytes().to_vec());
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::UnsupportedDType`] for a `Bool` or external dtype, and
/// `InvalidArgument` for an integer dtype when `value` is not an integer in
/// its range.
pub fn scalar_bytes(dtype: DType, value: f64) -> Result<Vec<u8>, Error> {
    Ok(match dtype {
        DType::F32 => (value as f32).to_le_bytes().to_vec(),
        DType::F64 => value.to_le_bytes().to_vec(),
        DType::I32 => integer_scalar::<i32>(dtype, value)?.to_le_bytes().to_vec(),
        DType::I64 => integer_scalar::<i64>(dtype, value)?.to_le_bytes().to_vec(),
        DType::C32 => {
            let mut bytes = (value as f32).to_le_bytes().to_vec();
            bytes.extend_from_slice(&0.0_f32.to_le_bytes());
            bytes
        }
        DType::C64 => {
            let mut bytes = value.to_le_bytes().to_vec();
            bytes.extend_from_slice(&0.0_f64.to_le_bytes());
            bytes
        }
        other => return Err(no_scalar_embedding(other)),
    })
}

/// A rank-0 host tensor holding `value` in a float or complex `dtype`.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::composite::scalar_tensor;
/// use tenferro_tensor::DType;
/// let t = scalar_tensor(DType::F32, f64::NEG_INFINITY)?;
/// assert_eq!(t.as_slice::<f32>()?, &[f32::NEG_INFINITY]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::UnsupportedDType`] for a `Bool` or external dtype, and
/// `InvalidArgument` for an integer dtype when `value` is not an integer in
/// its range.
pub fn scalar_tensor(dtype: DType, value: f64) -> Result<Tensor, Error> {
    match dtype {
        DType::F32 => Tensor::from_vec_col_major(vec![], vec![value as f32]),
        DType::F64 => Tensor::from_vec_col_major(vec![], vec![value]),
        DType::I32 => {
            Tensor::from_vec_col_major(vec![], vec![integer_scalar::<i32>(dtype, value)?])
        }
        DType::I64 => {
            Tensor::from_vec_col_major(vec![], vec![integer_scalar::<i64>(dtype, value)?])
        }
        DType::C32 => Tensor::from_vec_col_major(vec![], vec![Complex32::new(value as f32, 0.0)]),
        DType::C64 => Tensor::from_vec_col_major(vec![], vec![Complex64::new(value, 0.0)]),
        other => Err(no_scalar_embedding(other)),
    }
}

/// An exactly representable integer constant (composites use small counts
/// and coordinates bounded by a tensor extent).
fn integer_scalar<T: TryFrom<i64>>(dtype: DType, value: f64) -> Result<T, Error> {
    let exact = value.is_finite() && value.fract() == 0.0 && value.abs() < 9.0e15;
    exact
        .then(|| T::try_from(value as i64).ok())
        .flatten()
        .ok_or_else(|| {
            Error::invalid_argument(
                "composite_scalar",
                "value",
                format!("{value} is not representable in {dtype:?}"),
            )
        })
}

fn no_scalar_embedding(dtype: DType) -> Error {
    Error::unsupported_dtype(
        "composite_scalar",
        dtype,
        "composite constants exist only for numeric dtypes",
    )
}

fn require_real_float(op: &'static str, dtype: DType) -> Result<(), Error> {
    if matches!(dtype, DType::F32 | DType::F64) {
        Ok(())
    } else {
        Err(Error::unsupported_dtype(
            op,
            dtype,
            format!("{op} is defined for real F32/F64 tensors"),
        ))
    }
}

fn validate_axis(op: &'static str, axis: usize, rank: usize) -> Result<(), Error> {
    if axis < rank {
        Ok(())
    } else {
        Err(Error::axis_out_of_bounds(op, axis, rank))
    }
}

fn validate_axes(op: &'static str, axes: &[usize], rank: usize) -> Result<(), Error> {
    for (position, &axis) in axes.iter().enumerate() {
        validate_axis(op, axis, rank)?;
        if axes[..position].contains(&axis) {
            return Err(Error::duplicate_axis(op, axis, "axes"));
        }
    }
    Ok(())
}

fn other_axes(rank: usize, axis: usize) -> Vec<usize> {
    (0..rank).filter(|&dim| dim != axis).collect()
}

fn reduced_shape(shape: &[usize], axes: &[usize]) -> Vec<usize> {
    shape
        .iter()
        .enumerate()
        .filter(|(dim, _)| !axes.contains(dim))
        .map(|(_, &extent)| extent)
        .collect()
}

/// The constant `value` of `dtype` broadcast to `shape`: one constant and one
/// `broadcast_in_dim`, so the operands of the following binary ops already
/// match and no per-op reshape/broadcast is emitted.
fn splat<O: CompositeOps>(
    ops: &mut O,
    dtype: DType,
    value: f64,
    shape: &[usize],
) -> Result<O::Value, O::Error> {
    let scalar = ops.scalar(dtype, value)?;
    if shape.is_empty() {
        return Ok(scalar);
    }
    ops.broadcast_in_dim_owned(scalar, shape, &[])
}

/// `value * factor` with the factor splatted to `shape` (the value's shape).
///
/// The surfaces' `scale_real` multiplies by an unbroadcast rank-0 constant,
/// which the traced and eager CUDA paths do not accept; a splatted factor is
/// one constant and one broadcast on every surface.
fn scale_by<O: CompositeOps>(
    ops: &mut O,
    value: &O::Value,
    factor: f64,
    shape: &[usize],
) -> Result<O::Value, O::Error> {
    let dtype = ops.dtype(value);
    let factor = splat(ops, dtype, factor, shape)?;
    ops.binary(CompositeBinary::Mul, value, &factor)
}

/// `(x > 0, m)` with `m = -|x|` written as `select(x > 0, -x, x)`.
///
/// Unlike `-abs(x)`, the derivative of `m` at `x = 0` is `1` (the `x <= 0`
/// branch), which gives `softplus'(0) = sigmoid(0) = 1/2` and
/// `sigmoid'(0) = 1/4` through AD. `exp(m)` never overflows. `x > 0` is
/// computed as `x > -x` (equal for every value, `NaN` and `+-0` included), so
/// no zero constant is materialized.
fn positive_and_neg_abs<O: CompositeOps>(
    ops: &mut O,
    x: &O::Value,
) -> Result<(O::Value, O::Value), O::Error> {
    let negated = ops.unary(CompositeUnary::Neg, x)?;
    let positive = ops.compare(x, &negated, CompareDir::Gt)?;
    let neg_abs = ops.select(&positive, &negated, x)?;
    Ok((positive, neg_abs))
}

/// Logistic sigmoid `1 / (1 + exp(-x))` in an overflow-free form.
///
/// With `e = exp(-|x|)`, the result is `1 / (1 + e)` for `x > 0` and
/// `e / (1 + e)` otherwise, so every intermediate (and every derivative) is
/// finite for finite and infinite `x`.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::TracedTensor;
/// let x = TracedTensor::from_vec_col_major(vec![1], vec![0.0_f64])?;
/// let _y = x.sigmoid()?;
/// # Ok::<(), tenferro_runtime::Error>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::UnsupportedDType`] unless the input is `F32`/`F64`, or
/// the surface's primitive errors.
pub fn sigmoid<O: CompositeOps>(ops: &mut O, x: &O::Value) -> Result<O::Value, O::Error> {
    let dtype = ops.dtype(x);
    require_real_float("sigmoid", dtype)?;
    let shape = ops.shape(x)?;
    let (positive, neg_abs) = positive_and_neg_abs(ops, x)?;
    let e = ops.unary(CompositeUnary::Exp, &neg_abs)?;
    drop(neg_abs);
    let one = splat(ops, dtype, 1.0, &shape)?;
    let numerator = ops.select(&positive, &one, &e)?;
    drop(positive);
    let denominator = ops.binary(CompositeBinary::Add, &one, &e)?;
    drop((one, e));
    ops.binary(CompositeBinary::Div, &numerator, &denominator)
}

/// SiLU / swish `x * sigmoid(x)`.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::TracedTensor;
/// let x = TracedTensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
/// let _y = x.silu()?;
/// # Ok::<(), tenferro_runtime::Error>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::UnsupportedDType`] unless the input is `F32`/`F64`, or
/// the surface's primitive errors.
pub fn silu<O: CompositeOps>(ops: &mut O, x: &O::Value) -> Result<O::Value, O::Error> {
    let gate = sigmoid(ops, x)?;
    ops.binary(CompositeBinary::Mul, x, &gate)
}

/// Softplus `log(1 + exp(x))` in the stable form `max(x, 0) + log1p(exp(-|x|))`.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::TracedTensor;
/// let x = TracedTensor::from_vec_col_major(vec![1], vec![0.0_f64])?;
/// let _y = x.softplus()?;
/// # Ok::<(), tenferro_runtime::Error>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::UnsupportedDType`] unless the input is `F32`/`F64`, or
/// the surface's primitive errors.
pub fn softplus<O: CompositeOps>(ops: &mut O, x: &O::Value) -> Result<O::Value, O::Error> {
    let dtype = ops.dtype(x);
    require_real_float("softplus", dtype)?;
    let (positive, neg_abs) = positive_and_neg_abs(ops, x)?;
    let e = ops.unary(CompositeUnary::Exp, &neg_abs)?;
    drop(neg_abs);
    let tail = ops.unary(CompositeUnary::Log1p, &e)?;
    drop(e);
    // `x + tail` for x > 0, `tail` otherwise: `max(x, 0) + tail` without a
    // zero constant.
    let shifted = ops.binary(CompositeBinary::Add, x, &tail)?;
    ops.select(&positive, &shifted, &tail)
}

/// Exact GELU `x/2 * (1 + erf(x / sqrt(2)))`.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::TracedTensor;
/// let x = TracedTensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
/// let _y = x.gelu()?;
/// # Ok::<(), tenferro_runtime::Error>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::UnsupportedDType`] unless the input is `F32`/`F64`, or
/// the surface's primitive errors.
pub fn gelu<O: CompositeOps>(ops: &mut O, x: &O::Value) -> Result<O::Value, O::Error> {
    let dtype = ops.dtype(x);
    require_real_float("gelu", dtype)?;
    let shape = ops.shape(x)?;
    let scaled = scale_by(ops, x, std::f64::consts::FRAC_1_SQRT_2, &shape)?;
    let erf = ops.unary(CompositeUnary::Erf, &scaled)?;
    drop(scaled);
    let one = splat(ops, dtype, 1.0, &shape)?;
    let gate = ops.binary(CompositeBinary::Add, &one, &erf)?;
    drop((one, erf));
    let gated = ops.binary(CompositeBinary::Mul, x, &gate)?;
    drop(gate);
    scale_by(ops, &gated, 0.5, &shape)
}

/// `sqrt(2 / pi)`, the GELU tanh-approximation scale.
const SQRT_2_OVER_PI: f64 = std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2;
/// Cubic coefficient of the GELU tanh approximation.
const GELU_TANH_CUBIC: f64 = 0.044_715;

/// GELU tanh approximation `x/2 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))`.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::TracedTensor;
/// let x = TracedTensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
/// let _y = x.gelu_tanh()?;
/// # Ok::<(), tenferro_runtime::Error>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::UnsupportedDType`] unless the input is `F32`/`F64`, or
/// the surface's primitive errors.
pub fn gelu_tanh<O: CompositeOps>(ops: &mut O, x: &O::Value) -> Result<O::Value, O::Error> {
    let dtype = ops.dtype(x);
    require_real_float("gelu_tanh", dtype)?;
    let shape = ops.shape(x)?;
    let square = ops.binary(CompositeBinary::Mul, x, x)?;
    let cube = ops.binary(CompositeBinary::Mul, &square, x)?;
    drop(square);
    let cube_term = scale_by(ops, &cube, GELU_TANH_CUBIC, &shape)?;
    drop(cube);
    let polynomial = ops.binary(CompositeBinary::Add, x, &cube_term)?;
    drop(cube_term);
    let inner = scale_by(ops, &polynomial, SQRT_2_OVER_PI, &shape)?;
    drop(polynomial);
    let tanh = ops.unary(CompositeUnary::Tanh, &inner)?;
    drop(inner);
    let one = splat(ops, dtype, 1.0, &shape)?;
    let gate = ops.binary(CompositeBinary::Add, &one, &tanh)?;
    drop((one, tanh));
    let gated = ops.binary(CompositeBinary::Mul, x, &gate)?;
    drop(gate);
    scale_by(ops, &gated, 0.5, &shape)
}

/// Arithmetic mean over `axes` (`None` = every axis), dividing the sum by the count.
///
/// A mean over zero elements is `NaN` (0/0), like NumPy and PyTorch; an empty
/// axis list is the identity.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::TracedTensor;
/// let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 3.0])?;
/// let _y = x.reduce_mean(None)?;
/// # Ok::<(), tenferro_runtime::Error>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::UnsupportedDType`] for integer or `Bool` input,
/// `AxisOutOfBounds` / `DuplicateAxis` validation errors for invalid axes, or
/// the surface's primitive errors.
pub fn reduce_mean<O: CompositeOps>(
    ops: &mut O,
    x: &O::Value,
    axes: Option<&[usize]>,
) -> Result<O::Value, O::Error> {
    let dtype = ops.dtype(x);
    if !matches!(dtype, DType::F32 | DType::F64 | DType::C32 | DType::C64) {
        return Err(Error::unsupported_dtype(
            "reduce_mean",
            dtype,
            "reduce_mean is defined for float and complex tensors",
        )
        .into());
    }
    let shape = ops.shape(x)?;
    let axes: Vec<usize> = axes.map_or_else(|| (0..shape.len()).collect(), <[usize]>::to_vec);
    validate_axes("reduce_mean", &axes, shape.len())?;
    mean_over(ops, x, &shape, &axes)
}

/// Mean over validated `axes` of a float or complex `x` with concrete `shape`.
fn mean_over<O: CompositeOps>(
    ops: &mut O,
    x: &O::Value,
    shape: &[usize],
    axes: &[usize],
) -> Result<O::Value, O::Error> {
    let dtype = ops.dtype(x);
    let count: usize = axes.iter().map(|&axis| shape[axis]).product();
    if count == 0 {
        // Reductions reject zero-length axes; the mean of nothing is 0/0.
        let nan = ops.scalar(dtype, f64::NAN)?;
        return ops.broadcast_in_dim_owned(nan, &reduced_shape(shape, axes), &[]);
    }
    if axes.is_empty() {
        let count = ops.scalar(dtype, 1.0)?;
        return ops.binary(CompositeBinary::Div, x, &count);
    }
    let sum = ops.reduce(CompositeReduce::Sum, x, axes)?;
    let count = splat(ops, dtype, count as f64, &reduced_shape(shape, axes))?;
    ops.binary(CompositeBinary::Div, &sum, &count)
}

/// An empty result shaped like `x` (`x` has a zero extent).
fn empty_like<O: CompositeOps>(ops: &mut O, x: &O::Value) -> Result<O::Value, O::Error> {
    ops.unary(CompositeUnary::Neg, x)
}

fn validate_mask(
    op: &'static str,
    mask_dtype: DType,
    mask: &[usize],
    x: &[usize],
) -> Result<(), Error> {
    if mask_dtype != DType::Bool {
        return Err(Error::dtype_mismatch(op, DType::Bool, mask_dtype));
    }
    let broadcastable = mask.len() <= x.len()
        && mask
            .iter()
            .rev()
            .zip(x.iter().rev())
            .all(|(&m, &extent)| m == extent || m == 1);
    if broadcastable {
        Ok(())
    } else {
        Err(Error::shape_mismatch(op, x.to_vec(), mask.to_vec()))
    }
}

/// Max-subtracted softmax or log-softmax along `axis`, optionally masked.
///
/// Policy (shared by every surface and backend):
/// - masked-out entries (`mask == false`) are treated as `-inf`: softmax 0,
///   log-softmax `-inf`, and their gradient is exactly zero, even when the
///   masked value itself is `NaN` or infinite;
/// - a slice whose entries are all masked out (or all `-inf`) yields softmax
///   `0` and log-softmax `-inf` with a finite gradient instead of the `NaN`
///   of the naive `-inf - (-inf)`; in the masked form that gradient is zero;
/// - a participating `NaN` or `+inf` makes the whole slice `NaN`;
/// - a zero-length `axis` returns an empty result.
fn softmax_impl<O: CompositeOps>(
    ops: &mut O,
    op: &'static str,
    x: &O::Value,
    mask: Option<&O::Value>,
    axis: usize,
    log: bool,
) -> Result<O::Value, O::Error> {
    let dtype = ops.dtype(x);
    require_real_float(op, dtype)?;
    let shape = ops.shape(x)?;
    validate_axis(op, axis, shape.len())?;
    if let Some(mask) = mask {
        let mask_shape = ops.shape(mask)?;
        validate_mask(op, ops.dtype(mask), &mask_shape, &shape)?;
    }
    if shape[axis] == 0 {
        return empty_like(ops, x);
    }
    let masked = match mask {
        Some(mask) => Some({
            let neg_inf = splat(ops, dtype, f64::NEG_INFINITY, &shape)?;
            // Broadcast a smaller mask explicitly, right-aligned and without a
            // reshape: the implicit broadcast of `select` reshapes away the
            // unit dimensions first, which is several times slower on the eager
            // CPU path for a large `Bool` mask.
            let mask_shape = ops.shape(mask)?;
            let full_mask;
            let mask = if mask_shape == shape {
                mask
            } else {
                let dims: Vec<usize> = (shape.len() - mask_shape.len()..shape.len()).collect();
                full_mask = ops.broadcast_in_dim(mask, &shape, &dims)?;
                &full_mask
            };
            ops.select(mask, x, &neg_inf)?
        }),
        None => None,
    };
    let x = masked.as_ref().unwrap_or(x);
    let keep = other_axes(shape.len(), axis);
    let reduced = reduced_shape(&shape, &[axis]);
    let (lowest, smallest_positive) = match dtype {
        DType::F32 => (f64::from(f32::MIN), f64::from(f32::MIN_POSITIVE)),
        _ => (f64::MIN, f64::MIN_POSITIVE),
    };

    let max = ops.reduce(CompositeReduce::Max, x, &[axis])?;
    // An all-masked slice has max -inf; clamping it to the lowest finite value
    // keeps `x - max` at -inf there instead of NaN, while a NaN max still
    // propagates (NaN-propagating maximum) and every other max is unchanged.
    let floor = splat(ops, dtype, lowest, &reduced)?;
    let safe_max = ops.binary(CompositeBinary::Maximum, &max, &floor)?;
    drop((max, floor));
    let safe_max = ops.broadcast_in_dim_owned(safe_max, &shape, &keep)?;
    let shifted = ops.binary(CompositeBinary::Sub, x, &safe_max)?;
    drop((safe_max, masked));
    let exp = ops.unary(CompositeUnary::Exp, &shifted)?;
    let sum = ops.reduce(CompositeReduce::Sum, &exp, &[axis])?;
    // Any participating slice sums to at least 1, an all-masked slice to 0;
    // clamping to the smallest positive normal keeps the value 0 / -inf and
    // the backward pass finite (the clamp, not `sum`, receives the gradient).
    let tiny = splat(ops, dtype, smallest_positive, &reduced)?;
    let safe_sum = ops.binary(CompositeBinary::Maximum, &sum, &tiny)?;
    drop((sum, tiny));
    if log {
        drop(exp);
        let log_sum = ops.unary(CompositeUnary::Log, &safe_sum)?;
        let log_sum = ops.broadcast_in_dim_owned(log_sum, &shape, &keep)?;
        ops.binary(CompositeBinary::Sub, &shifted, &log_sum)
    } else {
        drop(shifted);
        let safe_sum = ops.broadcast_in_dim_owned(safe_sum, &shape, &keep)?;
        ops.binary(CompositeBinary::Div, &exp, &safe_sum)
    }
}

/// Softmax along `axis` (max-subtracted).
///
/// # Examples
///
/// ```
/// use tenferro_runtime::TracedTensor;
/// let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
/// let _y = x.softmax(0)?;
/// # Ok::<(), tenferro_runtime::Error>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::UnsupportedDType`] unless the input is `F32`/`F64`,
/// `AxisOutOfBounds` for an invalid axis, or the surface's primitive errors
/// (including a symbolic-shape error on the traced surface).
pub fn softmax<O: CompositeOps>(
    ops: &mut O,
    x: &O::Value,
    axis: usize,
) -> Result<O::Value, O::Error> {
    softmax_impl(ops, "softmax", x, None, axis, false)
}

/// Log-softmax along `axis` (max-subtracted).
///
/// # Examples
///
/// ```
/// use tenferro_runtime::TracedTensor;
/// let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
/// let _y = x.log_softmax(0)?;
/// # Ok::<(), tenferro_runtime::Error>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::UnsupportedDType`] unless the input is `F32`/`F64`,
/// `AxisOutOfBounds` for an invalid axis, or the surface's primitive errors.
pub fn log_softmax<O: CompositeOps>(
    ops: &mut O,
    x: &O::Value,
    axis: usize,
) -> Result<O::Value, O::Error> {
    softmax_impl(ops, "log_softmax", x, None, axis, true)
}

/// Softmax along `axis` over the entries where the `Bool` `mask` is true.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::TracedTensor;
/// let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
/// let mask = TracedTensor::from_vec_col_major(vec![2], vec![true, false])?;
/// let _y = x.masked_softmax(&mask, 0)?;
/// # Ok::<(), tenferro_runtime::Error>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::UnsupportedDType`] unless the input is `F32`/`F64`,
/// `DTypeMismatch` for a non-`Bool` mask, `ShapeMismatch` when the mask does
/// not broadcast to the input shape, `AxisOutOfBounds` for an invalid axis,
/// or the surface's primitive errors.
pub fn masked_softmax<O: CompositeOps>(
    ops: &mut O,
    x: &O::Value,
    mask: &O::Value,
    axis: usize,
) -> Result<O::Value, O::Error> {
    softmax_impl(ops, "masked_softmax", x, Some(mask), axis, false)
}

/// Log-softmax along `axis` over the entries where the `Bool` `mask` is true.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::TracedTensor;
/// let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
/// let mask = TracedTensor::from_vec_col_major(vec![2], vec![true, false])?;
/// let _y = x.masked_log_softmax(&mask, 0)?;
/// # Ok::<(), tenferro_runtime::Error>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::UnsupportedDType`] unless the input is `F32`/`F64`,
/// `DTypeMismatch` for a non-`Bool` mask, `ShapeMismatch` when the mask does
/// not broadcast to the input shape, `AxisOutOfBounds` for an invalid axis,
/// or the surface's primitive errors.
pub fn masked_log_softmax<O: CompositeOps>(
    ops: &mut O,
    x: &O::Value,
    mask: &O::Value,
    axis: usize,
) -> Result<O::Value, O::Error> {
    softmax_impl(ops, "masked_log_softmax", x, Some(mask), axis, true)
}

fn validate_eps(op: &'static str, eps: f64) -> Result<(), Error> {
    if eps.is_finite() && eps >= 0.0 {
        Ok(())
    } else {
        Err(Error::invalid_argument(
            op,
            "eps",
            format!("eps must be finite and non-negative, got {eps}"),
        ))
    }
}

fn validate_affine<O: CompositeOps>(
    ops: &O,
    op: &'static str,
    dtype: DType,
    extent: usize,
    param: Option<&O::Value>,
) -> Result<(), O::Error> {
    if let Some(param) = param {
        let param_dtype = ops.dtype(param);
        if param_dtype != dtype {
            return Err(Error::dtype_mismatch(op, dtype, param_dtype).into());
        }
        let param_shape = ops.shape(param)?;
        if param_shape != [extent] {
            return Err(Error::shape_mismatch(op, vec![extent], param_shape).into());
        }
    }
    Ok(())
}

/// Shared tail of the normalizations: `normalized * weight + bias` along `axis`.
fn apply_affine<O: CompositeOps>(
    ops: &mut O,
    normalized: O::Value,
    shape: &[usize],
    axis: usize,
    weight: Option<&O::Value>,
    bias: Option<&O::Value>,
) -> Result<O::Value, O::Error> {
    let mut out = normalized;
    if let Some(weight) = weight {
        let weight = ops.broadcast_in_dim(weight, shape, &[axis])?;
        out = ops.binary(CompositeBinary::Mul, &out, &weight)?;
    }
    if let Some(bias) = bias {
        let bias = ops.broadcast_in_dim(bias, shape, &[axis])?;
        out = ops.binary(CompositeBinary::Add, &out, &bias)?;
    }
    Ok(out)
}

/// Shared validation of `layer_norm` / `rms_norm`; returns the concrete shape.
fn validate_norm<O: CompositeOps>(
    ops: &O,
    op: &'static str,
    x: &O::Value,
    axis: usize,
    weight: Option<&O::Value>,
    bias: Option<&O::Value>,
    eps: f64,
) -> Result<Vec<usize>, O::Error> {
    let dtype = ops.dtype(x);
    require_real_float(op, dtype)?;
    let shape = ops.shape(x)?;
    validate_axis(op, axis, shape.len())?;
    validate_eps(op, eps)?;
    validate_affine(ops, op, dtype, shape[axis], weight)?;
    validate_affine(ops, op, dtype, shape[axis], bias)?;
    Ok(shape)
}

/// Layer normalization along `axis`: `(x - mean) / sqrt(var + eps) * weight + bias`.
///
/// `var` is the biased (population) variance computed from the centered
/// values (two passes). `weight` and `bias` are optional rank-1 tensors of
/// length `x.shape[axis]`. A zero-variance slice normalizes to `0` (then
/// `bias`) with a finite gradient when `eps > 0`.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::TracedTensor;
/// let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 3.0])?;
/// let _y = x.layer_norm(0, None, None, 1e-5)?;
/// # Ok::<(), tenferro_runtime::Error>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::UnsupportedDType`] unless the input is `F32`/`F64`,
/// `AxisOutOfBounds` for an invalid axis, `InvalidArgument` for a negative or
/// non-finite `eps`, `DTypeMismatch` / `ShapeMismatch` for a weight or bias
/// that is not a same-dtype vector of the axis length, or the surface's
/// primitive errors.
pub fn layer_norm<O: CompositeOps>(
    ops: &mut O,
    x: &O::Value,
    axis: usize,
    weight: Option<&O::Value>,
    bias: Option<&O::Value>,
    eps: f64,
) -> Result<O::Value, O::Error> {
    let shape = validate_norm(ops, "layer_norm", x, axis, weight, bias, eps)?;
    if shape[axis] == 0 {
        return empty_like(ops, x);
    }
    let dtype = ops.dtype(x);
    let keep = other_axes(shape.len(), axis);
    let inv_count = 1.0 / shape[axis] as f64;
    // The same primitive sequence as the hand composition: sum, scale,
    // broadcast, subtract, fused sum of squares, scale, add eps, rsqrt.
    let sum = ops.reduce(CompositeReduce::Sum, x, &[axis])?;
    let reduced = reduced_shape(&shape, &[axis]);
    let mean = scale_by(ops, &sum, inv_count, &reduced)?;
    drop(sum);
    let mean = ops.broadcast_in_dim_owned(mean, &shape, &keep)?;
    let centered = ops.binary(CompositeBinary::Sub, x, &mean)?;
    drop(mean);
    let squares = ops.reduce(CompositeReduce::SumSquares, &centered, &[axis])?;
    let variance = scale_by(ops, &squares, inv_count, &reduced)?;
    drop(squares);
    let eps = splat(ops, dtype, eps, &reduced)?;
    let shifted = ops.binary(CompositeBinary::Add, &variance, &eps)?;
    drop((variance, eps));
    let inv_std = ops.unary(CompositeUnary::Rsqrt, &shifted)?;
    drop(shifted);
    let inv_std = ops.broadcast_in_dim_owned(inv_std, &shape, &keep)?;
    let normalized = ops.binary(CompositeBinary::Mul, &centered, &inv_std)?;
    drop((centered, inv_std));
    apply_affine(ops, normalized, &shape, axis, weight, bias)
}

/// RMS normalization along `axis`: `x / sqrt(mean(x^2) + eps) * weight + bias`.
///
/// `weight` and `bias` are optional rank-1 tensors of length
/// `x.shape[axis]`. An all-zero slice normalizes to `0` (then `bias`) with a
/// finite gradient when `eps > 0`.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::TracedTensor;
/// let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 3.0])?;
/// let _y = x.rms_norm(0, None, None, 1e-6)?;
/// # Ok::<(), tenferro_runtime::Error>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::UnsupportedDType`] unless the input is `F32`/`F64`,
/// `AxisOutOfBounds` for an invalid axis, `InvalidArgument` for a negative or
/// non-finite `eps`, `DTypeMismatch` / `ShapeMismatch` for a weight or bias
/// that is not a same-dtype vector of the axis length, or the surface's
/// primitive errors.
pub fn rms_norm<O: CompositeOps>(
    ops: &mut O,
    x: &O::Value,
    axis: usize,
    weight: Option<&O::Value>,
    bias: Option<&O::Value>,
    eps: f64,
) -> Result<O::Value, O::Error> {
    let shape = validate_norm(ops, "rms_norm", x, axis, weight, bias, eps)?;
    if shape[axis] == 0 {
        return empty_like(ops, x);
    }
    let dtype = ops.dtype(x);
    let keep = other_axes(shape.len(), axis);
    let squares = ops.reduce(CompositeReduce::SumSquares, x, &[axis])?;
    let reduced = reduced_shape(&shape, &[axis]);
    let mean_square = scale_by(ops, &squares, 1.0 / shape[axis] as f64, &reduced)?;
    drop(squares);
    let eps = splat(ops, dtype, eps, &reduced)?;
    let shifted = ops.binary(CompositeBinary::Add, &mean_square, &eps)?;
    drop((mean_square, eps));
    let inv_rms = ops.unary(CompositeUnary::Rsqrt, &shifted)?;
    drop(shifted);
    let inv_rms = ops.broadcast_in_dim_owned(inv_rms, &shape, &keep)?;
    let normalized = ops.binary(CompositeBinary::Mul, x, &inv_rms)?;
    drop(inv_rms);
    apply_affine(ops, normalized, &shape, axis, weight, bias)
}

/// `[0, 1, ..., n - 1]` (`n >= 1`) in an integer `dtype`, built on the
/// surface from rank-0 constants by doubling, so no host index data or
/// transfer is involved (`O(log n)` operations).
fn iota<O: CompositeOps>(ops: &mut O, dtype: DType, n: usize) -> Result<O::Value, O::Error> {
    if n == 1 {
        let zero = ops.scalar(dtype, 0.0)?;
        return ops.reshape(&zero, &[1]);
    }
    if n % 2 == 1 {
        let head = iota(ops, dtype, n - 1)?;
        let last = ops.scalar(dtype, coordinate(dtype, n - 1)?)?;
        let last = ops.reshape(&last, &[1])?;
        return ops.concatenate(&[&head, &last], 0);
    }
    let half = iota(ops, dtype, n / 2)?;
    let offset = ops.scalar(dtype, coordinate(dtype, n / 2)?)?;
    let upper = ops.binary(CompositeBinary::Add, &half, &offset)?;
    ops.concatenate(&[&half, &upper], 0)
}

/// `value` as an exact coordinate of `dtype`.
fn coordinate(dtype: DType, value: usize) -> Result<f64, Error> {
    let max = if dtype == DType::I32 {
        i32::MAX as usize
    } else {
        1 << 53
    };
    if value <= max {
        Ok(value as f64)
    } else {
        Err(coordinate_overflow(dtype))
    }
}

fn coordinate_overflow(dtype: DType) -> Error {
    Error::invalid_argument(
        "take_along_axis",
        "indices",
        format!("an operand extent does not fit the {dtype:?} index dtype"),
    )
}

/// NumPy `take_along_axis`: `out[.., i, ..] = x[.., indices[.., i, ..], ..]` along `axis`.
///
/// `indices` (I32 or I64) has the rank of `x`; every other dimension of
/// `indices` equals the matching extent of `x` (a batch dimension, indexed
/// per element) or is `1` (broadcast: the whole extent is taken). The output
/// has `x`'s shape with `indices.shape[axis]` along `axis`. Indices must be in
/// `[0, x.shape[axis])`; out-of-range indices follow `gather`. Implemented as
/// one `gather` whose index tuples pair each index with its batch coordinates.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::TracedTensor;
/// let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
/// let idx = TracedTensor::from_vec_col_major(vec![1, 2], vec![1_i64, 0])?;
/// let _y = x.take_along_axis(&idx, 0)?;
/// # Ok::<(), tenferro_runtime::Error>(())
/// ```
///
/// # Errors
///
/// Returns `RankMismatch` / `ShapeMismatch` when `indices` does not have the
/// rank of `x` or a non-axis extent that is neither `1` nor `x`'s,
/// `AxisOutOfBounds` for an invalid axis, [`Error::UnsupportedDType`] for a
/// non-integer index dtype, `InvalidArgument` when taking from a zero-length
/// axis, or the surface's primitive errors.
pub fn take_along_axis<O: CompositeOps>(
    ops: &mut O,
    x: &O::Value,
    indices: &O::Value,
    axis: usize,
) -> Result<O::Value, O::Error> {
    const OP: &str = "take_along_axis";
    let shape = ops.shape(x)?;
    let index_shape = ops.shape(indices)?;
    let index_dtype = ops.dtype(indices);
    let rank = shape.len();
    validate_axis(OP, axis, rank)?;
    if index_shape.len() != rank {
        return Err(Error::rank_mismatch(OP, rank, index_shape.len()).into());
    }
    if !matches!(index_dtype, DType::I32 | DType::I64) {
        return Err(Error::unsupported_dtype(OP, index_dtype, "indices must be I32 or I64").into());
    }
    for dim in other_axes(rank, axis) {
        if index_shape[dim] != shape[dim] && index_shape[dim] != 1 {
            return Err(Error::shape_mismatch(OP, shape.clone(), index_shape.clone()).into());
        }
    }
    if shape[axis] == 0 && index_shape[axis] > 0 {
        return Err(Error::invalid_argument(
            OP,
            "indices",
            format!("cannot take indices from zero-length axis {axis}"),
        )
        .into());
    }

    // Batch dimensions get an explicit coordinate in each index tuple; the
    // remaining (broadcast) dimensions are taken whole as gather windows.
    let batch_dims: Vec<usize> = (0..rank)
        .filter(|&dim| dim == axis || (index_shape[dim] == shape[dim] && shape[dim] != 1))
        .collect();
    let window_dims: Vec<usize> = (0..rank).filter(|dim| !batch_dims.contains(dim)).collect();
    let batch_shape: Vec<usize> = batch_dims.iter().map(|&dim| index_shape[dim]).collect();
    let mut tuple_shape = batch_shape.clone();
    tuple_shape.push(1);

    // Window dimensions of `indices` all have extent 1, so this drops them.
    let axis_component = ops.reshape(indices, &tuple_shape)?;
    let positions: Vec<usize> = (0..batch_dims.len())
        .filter(|&position| batch_dims[position] != axis)
        .collect();
    let mut start_index_map = vec![axis];
    start_index_map.extend(positions.iter().map(|&position| batch_dims[position]));
    // Index tuple `(indices, batch coordinates...)` = indices * e_0 + coordinates,
    // built with broadcasting arithmetic rather than a concatenation: traced
    // `concatenate` of distinct inputs fails to compile without input specs
    // (#2018), so it only ever joins constants here (in `iota`).
    let start_indices = if positions.is_empty() {
        axis_component
    } else {
        let components = positions.len() + 1;
        let one = ops.scalar(index_dtype, 1.0)?;
        let one = ops.reshape(&one, &[1])?;
        // e_c is the one-hot vector of length `components` with a 1 at c.
        let unit = |ops: &mut O, c: usize| ops.pad(&one, &[c], &[components - 1 - c]);
        let e0 = unit(ops, 0)?;
        let mut tuples = ops.binary(CompositeBinary::Mul, &axis_component, &e0)?;
        // An empty batch has no tuples to fill in.
        let positions = if batch_shape.contains(&0) {
            &[][..]
        } else {
            &positions[..]
        };
        for (offset, &position) in positions.iter().enumerate() {
            let values = iota(ops, index_dtype, batch_shape[position])?;
            let values = ops.broadcast_in_dim_owned(values, &tuple_shape, &[position])?;
            let ec = unit(ops, offset + 1)?;
            let placed = ops.binary(CompositeBinary::Mul, &values, &ec)?;
            tuples = ops.binary(CompositeBinary::Add, &tuples, &placed)?;
        }
        tuples
    };
    let slice_sizes = (0..rank)
        .map(|dim| {
            if batch_dims.contains(&dim) {
                1
            } else {
                shape[dim]
            }
        })
        .collect();
    let config = GatherConfig {
        offset_dims: window_dims,
        collapsed_slice_dims: batch_dims,
        start_index_map,
        index_vector_dim: batch_shape.len(),
        slice_sizes,
    };
    ops.gather(x, &start_indices, config)
}
