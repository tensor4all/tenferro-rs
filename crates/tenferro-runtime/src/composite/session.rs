//! [`CompositeOps`] over a concrete [`Engine`].
//!
//! Values stay borrowed (`TensorRead`) until an operation produces an owned
//! result, so a composite never copies its caller's input.
//!
//! A composite constant (an activation coefficient, a normalization epsilon, a
//! mean's count) stays a host value ([`SessionValue::Splat`]) until an
//! operation consumes it. A binary or select whose other operands already have
//! the result's shape hands the constant to the engine's
//! [`Engine::binary_scalar_read`] / [`Engine::select_scalar_read`], so a device
//! engine that keeps its scalars resident performs no host-to-device transfer
//! per call (#2701). Any other use uploads it as before.

use std::marker::PhantomData;

use num_complex::{Complex32, Complex64};
use tenferro_ops::broadcast::{broadcast_shape, broadcast_shapes};
use tenferro_tensor::{
    CompareDir, ContractionScalar, DType, Engine, Error, GatherConfig, Result, ScalarBinaryOp,
    ScalarOperandSide, Tensor, TensorRead, TensorValue,
};

use super::{zero_pad_config, CompositeBinary, CompositeOps, CompositeReduce, CompositeUnary};
use crate::typed_tensor::{broadcast_error, broadcast_to_in_read, ReadInput};

/// Run one composite in `session` and return its owned result.
pub(crate) fn run_session_composite<'a>(
    session: &mut dyn Engine,
    composite: impl FnOnce(&mut SessionComposite<'_, 'a>) -> Result<SessionValue<'a>>,
) -> Result<Tensor> {
    let mut ops = SessionComposite::new(session);
    let out = composite(&mut ops)?;
    ops.finish(out)
}

/// A value of the concrete-session composite surface.
// INVARIANT: a value lives on one composite call's stack and is consumed within it; boxing the
// tensor variant would add a heap allocation per primitive for no space that matters here.
#[allow(clippy::large_enum_variant)]
pub(crate) enum SessionValue<'a> {
    /// A tensor, borrowed or owned.
    Read(ReadInput<'a>),
    /// The host constant `value` of `dtype` broadcast to `shape`, not yet uploaded.
    Splat {
        dtype: DType,
        value: f64,
        shape: Vec<usize>,
    },
}

impl<'a> SessionValue<'a> {
    /// A borrowed composite input.
    pub(crate) fn borrowed(read: TensorRead<'a>) -> Self {
        Self::Read(ReadInput::Borrowed(read))
    }

    fn dtype(&self) -> DType {
        match self {
            Self::Read(read) => read.tensor_read().dtype(),
            Self::Splat { dtype, .. } => *dtype,
        }
    }

    fn shape(&self) -> Vec<usize> {
        match self {
            Self::Read(read) => read.tensor_read().shape().to_vec(),
            Self::Splat { shape, .. } => shape.clone(),
        }
    }
}

/// The host scalar a splat of `dtype` holds, converted as [`Tensor::full`] converts it.
///
/// `None` for a dtype the scalar hooks do not carry (integers, `Bool`, external
/// payloads); such a constant is uploaded instead.
fn host_scalar(dtype: DType, value: f64) -> Option<ContractionScalar> {
    match dtype {
        DType::F32 => Some(ContractionScalar::F32(value as f32)),
        DType::F64 => Some(ContractionScalar::F64(value)),
        DType::C32 => Some(ContractionScalar::C32(Complex32::new(value as f32, 0.0))),
        DType::C64 => Some(ContractionScalar::C64(Complex64::new(value, 0.0))),
        _ => None,
    }
}

/// Whether a splat of `splat_shape` broadcasts into `shape` without changing it.
fn splat_fits(splat_shape: &[usize], shape: &[usize]) -> bool {
    matches!(broadcast_shape(shape, splat_shape), Ok(out) if out == shape)
}

/// Concrete-session implementation of the composite primitive vocabulary.
///
/// `'a` is the lifetime of the borrowed composite inputs.
pub(crate) struct SessionComposite<'s, 'a> {
    session: &'s mut dyn Engine,
    inputs: PhantomData<ReadInput<'a>>,
}

/// Operands with every splat uploaded, borrowing the rest.
struct Uploaded<'v> {
    owned: Vec<Option<ReadInput<'static>>>,
    values: Vec<&'v SessionValue<'v>>,
}

impl Uploaded<'_> {
    fn read(&self, index: usize) -> TensorRead<'_> {
        match (&self.owned[index], self.values[index]) {
            (Some(owned), _) => owned.tensor_read(),
            (None, SessionValue::Read(read)) => read.tensor_read(),
            (None, SessionValue::Splat { .. }) => {
                unreachable!("every splat operand is uploaded before it is read")
            }
        }
    }
}

impl<'s, 'a> SessionComposite<'s, 'a> {
    fn new(session: &'s mut dyn Engine) -> Self {
        Self {
            session,
            inputs: PhantomData,
        }
    }

    /// Turn a composite result into an owned tensor.
    fn finish(&mut self, value: SessionValue<'_>) -> Result<Tensor> {
        let value = match value {
            SessionValue::Read(read) => read,
            splat @ SessionValue::Splat { .. } => self.upload_splat(&splat)?,
        };
        match value {
            ReadInput::Owned(tensor) => Ok(tensor),
            ReadInput::Value(value) => {
                if value.as_tensor().is_some() {
                    Ok(value.into_tensor()?)
                } else {
                    self.session
                        .to_contiguous_read(ReadInput::Value(value).tensor_read())
                }
            }
            ReadInput::Borrowed(read) => self.session.to_contiguous_read(read),
        }
    }

    /// Upload a splat: one rank-0 constant read as a zero-stride view of its shape.
    fn upload_splat(&mut self, value: &SessionValue<'_>) -> Result<ReadInput<'static>> {
        let SessionValue::Splat {
            dtype,
            value,
            shape,
        } = value
        else {
            unreachable!("upload_splat is called on splats only")
        };
        let scalar = self
            .session
            .upload_host_tensor(TensorRead::from_tensor(&Tensor::full(
                *dtype,
                [0_usize; 0],
                *value,
            )?))?;
        if shape.is_empty() {
            return Ok(ReadInput::Owned(scalar));
        }
        Ok(ReadInput::Value(
            TensorValue::from_tensor(scalar).broadcast_in_dim_view(shape, [])?,
        ))
    }

    /// Upload every splat among `values`, so the operation can read them all.
    fn uploaded<'v>(&mut self, values: &[&'v SessionValue<'v>]) -> Result<Uploaded<'v>> {
        let owned = values
            .iter()
            .map(|value| match value {
                SessionValue::Read(_) => Ok(None),
                splat => self.upload_splat(splat).map(Some),
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Uploaded {
            owned,
            values: values.to_vec(),
        })
    }

    fn read_binary(
        &mut self,
        op: CompositeBinary,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
    ) -> Result<Tensor> {
        let shape = broadcast_shape(lhs.shape(), rhs.shape()).map_err(broadcast_error)?;
        let lhs = broadcast_to_in_read(lhs, &shape, self.session)?;
        let rhs = broadcast_to_in_read(rhs, &shape, self.session)?;
        let (lhs, rhs) = (lhs.tensor_read(), rhs.tensor_read());
        let session = &mut *self.session;
        match op {
            CompositeBinary::Add => session.add_read(lhs, rhs),
            CompositeBinary::Sub => session.sub_read(lhs, rhs),
            CompositeBinary::Mul => session.mul_read(lhs, rhs),
            CompositeBinary::Div => session.div_read(lhs, rhs),
            CompositeBinary::Maximum => session.maximum_read(lhs, rhs),
        }
    }

    fn read_select(
        &mut self,
        condition: TensorRead<'_>,
        on_true: TensorRead<'_>,
        on_false: TensorRead<'_>,
    ) -> Result<Tensor> {
        let shape = broadcast_shapes([condition.shape(), on_true.shape(), on_false.shape()])
            .map_err(broadcast_error)?;
        let condition = broadcast_to_in_read(condition, &shape, self.session)?;
        let on_true = broadcast_to_in_read(on_true, &shape, self.session)?;
        let on_false = broadcast_to_in_read(on_false, &shape, self.session)?;
        self.session.select_read(
            condition.tensor_read(),
            on_true.tensor_read(),
            on_false.tensor_read(),
        )
    }
}

fn scalar_binary_op(op: CompositeBinary) -> ScalarBinaryOp {
    match op {
        CompositeBinary::Add => ScalarBinaryOp::Add,
        CompositeBinary::Sub => ScalarBinaryOp::Subtract,
        CompositeBinary::Mul => ScalarBinaryOp::Multiply,
        CompositeBinary::Div => ScalarBinaryOp::Divide,
        CompositeBinary::Maximum => ScalarBinaryOp::Maximum,
    }
}

impl<'a> CompositeOps for SessionComposite<'_, 'a> {
    type Value = SessionValue<'a>;
    type Error = Error;

    fn dtype(&self, value: &SessionValue<'a>) -> DType {
        value.dtype()
    }

    fn shape(&self, value: &SessionValue<'a>) -> Result<Vec<usize>> {
        Ok(value.shape())
    }

    fn scalar(&mut self, dtype: DType, value: f64) -> Result<SessionValue<'a>> {
        // The conversion `Tensor::full` applies is checked here, where the constant is
        // created, not where it is first consumed.
        Tensor::full(dtype, [0_usize; 0], value)?;
        Ok(SessionValue::Splat {
            dtype,
            value,
            shape: Vec::new(),
        })
    }

    fn unary(&mut self, op: CompositeUnary, value: &SessionValue<'a>) -> Result<SessionValue<'a>> {
        let operands = self.uploaded(&[value])?;
        let input = operands.read(0);
        let session = &mut *self.session;
        let out = match op {
            CompositeUnary::Neg => session.neg_read(input),
            CompositeUnary::Exp => session.exp_read(input),
            CompositeUnary::Log => session.log_read(input),
            CompositeUnary::Log1p => session.log1p_read(input),
            CompositeUnary::Tanh => session.tanh_read(input),
            CompositeUnary::Erf => session.erf_read(input),
            CompositeUnary::Rsqrt => session.rsqrt_read(input),
        }?;
        Ok(SessionValue::Read(ReadInput::Owned(out)))
    }

    fn binary(
        &mut self,
        op: CompositeBinary,
        lhs: &SessionValue<'a>,
        rhs: &SessionValue<'a>,
    ) -> Result<SessionValue<'a>> {
        // A constant against a tensor that already has the result's shape: the engine
        // takes the constant as a host scalar.
        let scalar_operand = match (lhs, rhs) {
            (
                SessionValue::Splat {
                    dtype,
                    value,
                    shape,
                },
                SessionValue::Read(tensor),
            ) => Some((tensor, *dtype, *value, shape, ScalarOperandSide::Left)),
            (
                SessionValue::Read(tensor),
                SessionValue::Splat {
                    dtype,
                    value,
                    shape,
                },
            ) => Some((tensor, *dtype, *value, shape, ScalarOperandSide::Right)),
            _ => None,
        };
        if let Some((tensor, dtype, value, shape, side)) = scalar_operand {
            let read = tensor.tensor_read();
            if let Some(scalar) = host_scalar(dtype, value)
                && dtype == read.dtype()
                && splat_fits(shape, read.shape())
            {
                let out =
                    self.session
                        .binary_scalar_read(scalar_binary_op(op), read, scalar, side)?;
                return Ok(SessionValue::Read(ReadInput::Owned(out)));
            }
        }
        let operands = self.uploaded(&[lhs, rhs])?;
        let out = self.read_binary(op, operands.read(0), operands.read(1))?;
        Ok(SessionValue::Read(ReadInput::Owned(out)))
    }

    fn compare(
        &mut self,
        lhs: &SessionValue<'a>,
        rhs: &SessionValue<'a>,
        dir: CompareDir,
    ) -> Result<SessionValue<'a>> {
        let operands = self.uploaded(&[lhs, rhs])?;
        let (lhs, rhs) = (operands.read(0), operands.read(1));
        let shape = broadcast_shape(lhs.shape(), rhs.shape()).map_err(broadcast_error)?;
        let lhs = broadcast_to_in_read(lhs, &shape, self.session)?;
        let rhs = broadcast_to_in_read(rhs, &shape, self.session)?;
        let out = self
            .session
            .compare_read(lhs.tensor_read(), rhs.tensor_read(), &dir)?;
        Ok(SessionValue::Read(ReadInput::Owned(out)))
    }

    fn select(
        &mut self,
        condition: &SessionValue<'a>,
        on_true: &SessionValue<'a>,
        on_false: &SessionValue<'a>,
    ) -> Result<SessionValue<'a>> {
        // One constant branch against a tensor branch and a predicate that both have
        // the result's shape: the engine takes the constant as a host scalar.
        if let SessionValue::Read(predicate) = condition {
            let scalar_branch = match (on_true, on_false) {
                (
                    SessionValue::Splat {
                        dtype,
                        value,
                        shape,
                    },
                    SessionValue::Read(tensor),
                ) => Some((tensor, *dtype, *value, shape, ScalarOperandSide::Left)),
                (
                    SessionValue::Read(tensor),
                    SessionValue::Splat {
                        dtype,
                        value,
                        shape,
                    },
                ) => Some((tensor, *dtype, *value, shape, ScalarOperandSide::Right)),
                _ => None,
            };
            if let Some((tensor, dtype, value, shape, side)) = scalar_branch {
                let (predicate, read) = (predicate.tensor_read(), tensor.tensor_read());
                if let Some(scalar) = host_scalar(dtype, value)
                    && dtype == read.dtype()
                    && predicate.shape() == read.shape()
                    && splat_fits(shape, read.shape())
                {
                    let out = self
                        .session
                        .select_scalar_read(predicate, read, scalar, side)?;
                    return Ok(SessionValue::Read(ReadInput::Owned(out)));
                }
            }
        }
        let operands = self.uploaded(&[condition, on_true, on_false])?;
        let out = self.read_select(operands.read(0), operands.read(1), operands.read(2))?;
        Ok(SessionValue::Read(ReadInput::Owned(out)))
    }

    fn reduce(
        &mut self,
        op: CompositeReduce,
        value: &SessionValue<'a>,
        axes: &[usize],
    ) -> Result<SessionValue<'a>> {
        let operands = self.uploaded(&[value])?;
        let input = operands.read(0);
        let out = match op {
            CompositeReduce::Sum => self.session.reduce_sum_read(input, axes),
            CompositeReduce::Max => self.session.reduce_max_read(input, axes),
            CompositeReduce::SumSquares => self.session.reduce_sum_squares_read(input, axes),
        }?;
        Ok(SessionValue::Read(ReadInput::Owned(out)))
    }

    fn broadcast_in_dim(
        &mut self,
        value: &SessionValue<'a>,
        shape: &[usize],
        dims: &[usize],
    ) -> Result<SessionValue<'a>> {
        if let SessionValue::Splat {
            dtype,
            value: constant,
            shape: from,
        } = value
        {
            // A broadcast constant is the same constant over the new shape; validate the
            // broadcast as a view would.
            check_broadcast(from, shape, dims)?;
            return Ok(SessionValue::Splat {
                dtype: *dtype,
                value: *constant,
                shape: shape.to_vec(),
            });
        }
        record_broadcast_route(BroadcastRoute::Materializing);
        let operands = self.uploaded(&[value])?;
        let out = self
            .session
            .broadcast_in_dim_read(operands.read(0), shape, dims)?;
        Ok(SessionValue::Read(ReadInput::Owned(out)))
    }

    fn broadcast_in_dim_owned(
        &mut self,
        value: SessionValue<'a>,
        shape: &[usize],
        dims: &[usize],
    ) -> Result<SessionValue<'a>> {
        let value = match value {
            splat @ SessionValue::Splat { .. } => {
                return self.broadcast_in_dim(&splat, shape, dims);
            }
            SessionValue::Read(read) => read,
        };
        record_broadcast_route(BroadcastRoute::View);
        // INVARIANT: the composite surface consumes these operands read-only,
        // so a broadcast is a layout change over storage this value already
        // owns; the materializing `broadcast_in_dim` stays for the callers that
        // only hold a borrow (#2143 item 1).
        Ok(SessionValue::Read(match value {
            ReadInput::Borrowed(read) => {
                ReadInput::Borrowed(read.broadcast_in_dim_view(shape, dims)?)
            }
            // A view of a caller-owned payload has no `TensorRead` form (#2486),
            // and `ReadInput::tensor_read` relies on `Value` never holding one.
            ReadInput::Owned(tensor) if matches!(tensor.dtype(), DType::External(_)) => {
                return Err(Error::unsupported_dtype(
                    "broadcast_in_dim",
                    tensor.dtype(),
                    "a caller-owned payload has no metadata-only broadcast view",
                ));
            }
            ReadInput::Owned(tensor) => ReadInput::Value(
                TensorValue::from_tensor(tensor).broadcast_in_dim_view(shape, dims)?,
            ),
            ReadInput::Value(value) => ReadInput::Value(value.broadcast_in_dim_view(shape, dims)?),
        }))
    }

    fn reshape(&mut self, value: &SessionValue<'a>, shape: &[usize]) -> Result<SessionValue<'a>> {
        let operands = self.uploaded(&[value])?;
        let out = self.session.reshape_read(operands.read(0), shape)?;
        Ok(SessionValue::Read(ReadInput::Owned(out)))
    }

    fn concatenate(
        &mut self,
        values: &[&SessionValue<'a>],
        axis: usize,
    ) -> Result<SessionValue<'a>> {
        let operands = self.uploaded(values)?;
        let reads: Vec<TensorRead<'_>> = (0..values.len()).map(|i| operands.read(i)).collect();
        Ok(SessionValue::Read(ReadInput::Owned(
            self.session.concatenate_read(&reads, axis)?,
        )))
    }

    fn pad(
        &mut self,
        value: &SessionValue<'a>,
        low: &[usize],
        high: &[usize],
    ) -> Result<SessionValue<'a>> {
        let operands = self.uploaded(&[value])?;
        let out = self
            .session
            .pad_read(operands.read(0), &zero_pad_config(low, high))?;
        Ok(SessionValue::Read(ReadInput::Owned(out)))
    }

    fn gather(
        &mut self,
        operand: &SessionValue<'a>,
        indices: &SessionValue<'a>,
        config: GatherConfig,
    ) -> Result<SessionValue<'a>> {
        let operands = self.uploaded(&[operand, indices])?;
        Ok(SessionValue::Read(ReadInput::Owned(
            self.session
                .gather_read(operands.read(0), operands.read(1), &config)?,
        )))
    }
}

/// The `broadcast_in_dim` validation a metadata view applies, for a constant that
/// has no tensor to build the view on.
fn check_broadcast(from: &[usize], shape: &[usize], dims: &[usize]) -> Result<()> {
    let valid = dims.len() == from.len()
        && dims
            .iter()
            .zip(from)
            .all(|(&dim, &extent)| dim < shape.len() && (extent == 1 || extent == shape[dim]))
        && dims.windows(2).all(|pair| pair[0] < pair[1]);
    if valid {
        Ok(())
    } else {
        Err(Error::shape_mismatch(
            "broadcast_in_dim",
            from.to_vec(),
            shape.to_vec(),
        ))
    }
}

/// Which broadcast route the session composite took (#2143 item 1).
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BroadcastRoute {
    /// [`SessionComposite::broadcast_in_dim_owned`] over an owned value: the
    /// broadcast is a metadata view over storage the caller already owns.
    View,
    /// [`SessionComposite::broadcast_in_dim`]: the materializing route, kept for
    /// callers that only hold a borrow of the operand.
    Materializing,
}

/// Observation is off by default, so a release build pays one relaxed atomic load
/// per composite broadcast and nothing else.
static OBSERVE_BROADCAST_ROUTES: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

thread_local! {
    /// Broadcast routes taken by `SessionComposite` on this thread.
    static BROADCAST_ROUTES: std::cell::RefCell<Vec<BroadcastRoute>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Start or stop recording composite broadcast routes.
pub(crate) fn set_broadcast_route_observation(enabled: bool) {
    if !enabled {
        let _ = take_broadcast_routes();
    }
    OBSERVE_BROADCAST_ROUTES.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

fn record_broadcast_route(route: BroadcastRoute) {
    if OBSERVE_BROADCAST_ROUTES.load(std::sync::atomic::Ordering::Relaxed) {
        BROADCAST_ROUTES.with(|routes| routes.borrow_mut().push(route));
    }
}

/// Take the composite broadcast routes recorded on this thread.
pub(crate) fn take_broadcast_routes() -> Vec<BroadcastRoute> {
    BROADCAST_ROUTES.with(|routes| std::mem::take(&mut *routes.borrow_mut()))
}
