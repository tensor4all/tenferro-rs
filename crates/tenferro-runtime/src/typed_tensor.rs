//! Borrowed-read adapters shared by the core grammar bodies and composites.

use tenferro_ops::broadcast::BroadcastError;
use tenferro_tensor::{DType, Engine, Error, Result, Tensor, TensorRead, TensorValue};
use tinyvec::TinyVec;

// INVARIANT: this private adapter keeps borrowed reads borrowed, owns the
// explicit fallback tensor, and owns an intermediate whose layout is a
// metadata-only view of that tensor; it is never exposed or cloned. `Value`
// always holds a preset scalar: its only constructor,
// `SessionComposite::broadcast_in_dim_owned`, refuses a caller-owned payload.
#[allow(clippy::large_enum_variant)]
pub(crate) enum ReadInput<'a> {
    Borrowed(TensorRead<'a>),
    Owned(Tensor),
    Value(TensorValue),
}

impl ReadInput<'_> {
    pub(crate) fn tensor_read(&self) -> TensorRead<'_> {
        match self {
            Self::Borrowed(read) => read.clone(),
            Self::Owned(tensor) => TensorRead::from_tensor(tensor),
            Self::Value(value) => value.tensor_read().expect(
                "ReadInput::Value is built only by SessionComposite::broadcast_in_dim_owned, \
                 which refuses a caller-owned payload before building the view, and only a \
                 view of a caller-owned payload has no read",
            ),
        }
    }
}

pub(crate) fn broadcast_to_in_read<'a>(
    input: TensorRead<'a>,
    target_shape: &[usize],
    session: &mut dyn Engine,
) -> Result<ReadInput<'a>> {
    if input.shape() == target_shape {
        return Ok(ReadInput::Borrowed(input));
    }

    // The eager surface aligns the operand's axes with the trailing output axes
    // and reads expanding axes through zero strides, so an implicit broadcast is
    // metadata. The traced/AD lowering instead removes expanding singleton axes
    // before broadcasting; that plan needs a materializing reshape and is not
    // used here (#2143 item 1).
    let dims = broadcast_aligned_dims(input.shape(), target_shape).map_err(broadcast_error)?;

    // A caller-owned payload has no borrowed view, so it keeps the session's
    // typed refusal path rather than a view.
    if let DType::External(_) = input.dtype() {
        let out = session.broadcast_in_dim_read(input, target_shape, &dims)?;
        return Ok(ReadInput::Owned(out));
    }

    Ok(ReadInput::Borrowed(
        input.broadcast_in_dim_view(target_shape, &dims)?,
    ))
}

/// The trailing-axis mapping of `input` onto `output`, or the same broadcast
/// failure the traced/AD plan reports.
fn broadcast_aligned_dims(
    input: &[usize],
    output: &[usize],
) -> std::result::Result<TinyVec<[usize; 8]>, BroadcastError> {
    if input.len() > output.len() {
        return Err(BroadcastError::RankTooLarge {
            input: input.to_vec(),
            output: output.to_vec(),
        });
    }
    let rank_diff = output.len() - input.len();
    for (src_axis, &src_dim) in input.iter().enumerate() {
        let dst_dim = output[src_axis + rank_diff];
        if src_dim != dst_dim && src_dim != 1 {
            return Err(BroadcastError::IncompatibleInput {
                input: input.to_vec(),
                output: output.to_vec(),
            });
        }
    }
    Ok((0..input.len()).map(|axis| axis + rank_diff).collect())
}

pub(crate) fn broadcast_error(err: BroadcastError) -> Error {
    match err {
        BroadcastError::IncompatibleBinary { lhs, rhs } => {
            Error::shape_mismatch("broadcast", lhs, rhs)
        }
        BroadcastError::IncompatibleInput { input, output } => {
            Error::shape_mismatch("broadcast", input, output)
        }
        BroadcastError::RankTooLarge { input, output } => {
            Error::rank_mismatch("broadcast", output.len(), input.len())
        }
    }
}
