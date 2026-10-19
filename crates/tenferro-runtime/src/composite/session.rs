//! [`CompositeOps`] over a concrete [`Engine`].
//!
//! Values stay borrowed (`TensorRead`) until an operation produces an owned
//! result, so a composite never copies its caller's input.

use std::marker::PhantomData;

use tenferro_ops::broadcast::{broadcast_shape, broadcast_shapes};
use tenferro_tensor::{
    Engine, CompareDir, DType, Error, GatherConfig, Result, Tensor, TensorRead,
    TensorScalar, TensorValue, TypedTensor,
};

use super::{
    scalar_tensor, zero_pad_config, CompositeBinary, CompositeOps, CompositeReduce, CompositeUnary,
};
use crate::typed_tensor::{broadcast_error, broadcast_to_in_read, ReadInput};

/// Borrow an erased tensor as a composite input.
pub(crate) fn borrowed(tensor: &Tensor) -> ReadInput<'_> {
    ReadInput::Borrowed(TensorRead::from_tensor(tensor))
}

/// Borrow a typed tensor as a composite input.
pub(crate) fn typed_borrowed<T: TensorScalar>(tensor: &TypedTensor<T>) -> ReadInput<'_> {
    ReadInput::Borrowed(T::tensor_read(tensor))
}

/// Run one composite in `session` and return its owned result.
pub(crate) fn run_session_composite<'a>(
    session: &mut dyn Engine,
    composite: impl FnOnce(&mut SessionComposite<'_, 'a>) -> Result<ReadInput<'a>>,
) -> Result<Tensor> {
    let mut ops = SessionComposite::new(session);
    let out = composite(&mut ops)?;
    ops.finish(out)
}

/// Concrete-session implementation of the composite primitive vocabulary.
///
/// `'a` is the lifetime of the borrowed composite inputs.
pub(crate) struct SessionComposite<'s, 'a> {
    session: &'s mut dyn Engine,
    inputs: PhantomData<ReadInput<'a>>,
}

/// A tensor operand that is either borrowed or materialized for this call.
enum Operand<'v> {
    Borrowed(&'v Tensor),
    Owned(Box<Tensor>),
}

impl Operand<'_> {
    fn tensor(&self) -> &Tensor {
        match self {
            Self::Borrowed(tensor) => tensor,
            Self::Owned(tensor) => tensor,
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
    fn finish(&mut self, value: ReadInput<'_>) -> Result<Tensor> {
        match value {
            ReadInput::Owned(tensor) => Ok(tensor),
            ReadInput::Value(value) => {
                if value.as_tensor().is_some() {
                    Ok(value.into_tensor()?)
                } else {
                    self.session.to_contiguous_read(value.tensor_read())
                }
            }
            ReadInput::Borrowed(read) => self.session.to_contiguous_read(read),
        }
    }

    /// Gather and concatenate take owned tensors; a borrowed view is copied once.
    fn materialized<'v>(&mut self, value: &'v ReadInput<'_>) -> Result<Operand<'v>> {
        match value {
            ReadInput::Owned(tensor) => Ok(Operand::Borrowed(tensor)),
            ReadInput::Value(value) => match value.as_tensor() {
                Some(tensor) => Ok(Operand::Borrowed(tensor)),
                None => Ok(Operand::Owned(Box::new(
                    self.session.to_contiguous_read(value.tensor_read())?,
                ))),
            },
            ReadInput::Borrowed(read) => match read.as_tensor() {
                Some(tensor) => Ok(Operand::Borrowed(tensor)),
                None => Ok(Operand::Owned(Box::new(
                    self.session.to_contiguous_read(read.clone())?,
                ))),
            },
        }
    }

    fn upload(&mut self, tensor: &Tensor) -> Result<ReadInput<'static>> {
        Ok(ReadInput::Owned(
            self.session
                .upload_host_tensor(TensorRead::from_tensor(tensor))?,
        ))
    }
}

impl<'a> CompositeOps for SessionComposite<'_, 'a> {
    type Value = ReadInput<'a>;
    type Error = Error;

    fn dtype(&self, value: &ReadInput<'a>) -> DType {
        value.tensor_read().dtype()
    }

    fn shape(&self, value: &ReadInput<'a>) -> Result<Vec<usize>> {
        Ok(value.tensor_read().shape().to_vec())
    }

    fn scalar(&mut self, dtype: DType, value: f64) -> Result<ReadInput<'a>> {
        self.upload(&scalar_tensor(dtype, value)?)
    }

    fn unary(&mut self, op: CompositeUnary, value: &ReadInput<'a>) -> Result<ReadInput<'a>> {
        let input = value.tensor_read();
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
        Ok(ReadInput::Owned(out))
    }

    fn binary(
        &mut self,
        op: CompositeBinary,
        lhs: &ReadInput<'a>,
        rhs: &ReadInput<'a>,
    ) -> Result<ReadInput<'a>> {
        let (lhs, rhs) = (lhs.tensor_read(), rhs.tensor_read());
        let shape = broadcast_shape(lhs.shape(), rhs.shape()).map_err(broadcast_error)?;
        let lhs = broadcast_to_in_read(lhs, &shape, self.session)?;
        let rhs = broadcast_to_in_read(rhs, &shape, self.session)?;
        let (lhs, rhs) = (lhs.tensor_read(), rhs.tensor_read());
        let session = &mut *self.session;
        let out = match op {
            CompositeBinary::Add => session.add_read(lhs, rhs),
            CompositeBinary::Sub => session.sub_read(lhs, rhs),
            CompositeBinary::Mul => session.mul_read(lhs, rhs),
            CompositeBinary::Div => session.div_read(lhs, rhs),
            CompositeBinary::Maximum => session.maximum_read(lhs, rhs),
        }?;
        Ok(ReadInput::Owned(out))
    }

    fn compare(
        &mut self,
        lhs: &ReadInput<'a>,
        rhs: &ReadInput<'a>,
        dir: CompareDir,
    ) -> Result<ReadInput<'a>> {
        let (lhs, rhs) = (lhs.tensor_read(), rhs.tensor_read());
        let shape = broadcast_shape(lhs.shape(), rhs.shape()).map_err(broadcast_error)?;
        let lhs = broadcast_to_in_read(lhs, &shape, self.session)?;
        let rhs = broadcast_to_in_read(rhs, &shape, self.session)?;
        let out = self
            .session
            .compare_read(lhs.tensor_read(), rhs.tensor_read(), &dir)?;
        Ok(ReadInput::Owned(out))
    }

    fn select(
        &mut self,
        condition: &ReadInput<'a>,
        on_true: &ReadInput<'a>,
        on_false: &ReadInput<'a>,
    ) -> Result<ReadInput<'a>> {
        let (condition, on_true, on_false) = (
            condition.tensor_read(),
            on_true.tensor_read(),
            on_false.tensor_read(),
        );
        let shape = broadcast_shapes([condition.shape(), on_true.shape(), on_false.shape()])
            .map_err(broadcast_error)?;
        let condition = broadcast_to_in_read(condition, &shape, self.session)?;
        let on_true = broadcast_to_in_read(on_true, &shape, self.session)?;
        let on_false = broadcast_to_in_read(on_false, &shape, self.session)?;
        let out = self.session.select_read(
            condition.tensor_read(),
            on_true.tensor_read(),
            on_false.tensor_read(),
        )?;
        Ok(ReadInput::Owned(out))
    }

    fn reduce(
        &mut self,
        op: CompositeReduce,
        value: &ReadInput<'a>,
        axes: &[usize],
    ) -> Result<ReadInput<'a>> {
        let input = value.tensor_read();
        let out = match op {
            CompositeReduce::Sum => self.session.reduce_sum_read(input, axes),
            CompositeReduce::Max => self.session.reduce_max_read(input, axes),
            CompositeReduce::SumSquares => self.session.reduce_sum_squares_read(input, axes),
        }?;
        Ok(ReadInput::Owned(out))
    }

    fn broadcast_in_dim(
        &mut self,
        value: &ReadInput<'a>,
        shape: &[usize],
        dims: &[usize],
    ) -> Result<ReadInput<'a>> {
        record_broadcast_route(BroadcastRoute::Materializing);
        let out = self
            .session
            .broadcast_in_dim_read(value.tensor_read(), shape, dims)?;
        Ok(ReadInput::Owned(out))
    }

    fn broadcast_in_dim_owned(
        &mut self,
        value: ReadInput<'a>,
        shape: &[usize],
        dims: &[usize],
    ) -> Result<ReadInput<'a>> {
        record_broadcast_route(BroadcastRoute::View);
        // INVARIANT: the composite surface consumes these operands read-only,
        // so a broadcast is a layout change over storage this value already
        // owns; the materializing `broadcast_in_dim` stays for the callers that
        // only hold a borrow (#2143 item 1).
        Ok(match value {
            ReadInput::Borrowed(read) => {
                ReadInput::Borrowed(read.broadcast_in_dim_view(shape, dims)?)
            }
            ReadInput::Owned(tensor) => ReadInput::Value(
                TensorValue::from_tensor(tensor).broadcast_in_dim_view(shape, dims)?,
            ),
            ReadInput::Value(value) => ReadInput::Value(value.broadcast_in_dim_view(shape, dims)?),
        })
    }

    fn reshape(&mut self, value: &ReadInput<'a>, shape: &[usize]) -> Result<ReadInput<'a>> {
        let out = self.session.reshape_read(value.tensor_read(), shape)?;
        Ok(ReadInput::Owned(out))
    }

    fn concatenate(&mut self, values: &[&ReadInput<'a>], axis: usize) -> Result<ReadInput<'a>> {
        let tensors = values
            .iter()
            .map(|value| self.materialized(value))
            .collect::<Result<Vec<_>>>()?;
        let refs: Vec<&Tensor> = tensors.iter().map(Operand::tensor).collect();
        Ok(ReadInput::Owned(self.session.concatenate(&refs, axis)?))
    }

    fn pad(
        &mut self,
        value: &ReadInput<'a>,
        low: &[usize],
        high: &[usize],
    ) -> Result<ReadInput<'a>> {
        let input = self.materialized(value)?;
        let out = self
            .session
            .pad(input.tensor(), &zero_pad_config(low, high))?;
        Ok(ReadInput::Owned(out))
    }

    fn gather(
        &mut self,
        operand: &ReadInput<'a>,
        indices: &ReadInput<'a>,
        config: GatherConfig,
    ) -> Result<ReadInput<'a>> {
        let operand = self.materialized(operand)?;
        let indices = self.materialized(indices)?;
        Ok(ReadInput::Owned(self.session.gather(
            operand.tensor(),
            indices.tensor(),
            &config,
        )?))
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
