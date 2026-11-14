//! Wire frames, the dtype code table and the payload views.
//!
//! Frames are byte-oriented on purpose: the header, the ack and the payload all go
//! through the same probe-then-receive path, so a frame is never received into a
//! buffer shorter than its probed length and MPI truncation is never reached.
//! [`docs/design/mpi-transport.md`] specifies the format.

use tenferro_tensor::{Complex32, Complex64, DType, TensorScalar};

mod bool_wire;

#[cfg(test)]
mod tests;

/// A point-to-point tag, matching the MPI and rsmpi `Tag` type.
pub type Tag = i32;

/// Protocol version carried in every frame.
pub const PROTOCOL_VERSION: u16 = 1;

/// Largest rank a header can carry.
pub const MAX_WIRE_RANK: usize = 64;

/// Largest element count one message may carry.
///
/// The protocol rejects anything larger instead of chunking or truncating.
pub const MAX_WIRE_ELEMENTS: u64 = i32::MAX as u64;

pub(crate) const HEADER_PROLOGUE: usize = 24;
pub(crate) const SHAPE_STRIDE: usize = 8;
pub(crate) const MAX_HEADER_LEN: usize = HEADER_PROLOGUE + SHAPE_STRIDE * MAX_WIRE_RANK;
pub(crate) const ACK_LEN: usize = 8;
/// Largest frame this receiver will allocate for to classify a malformed one.
///
/// A frame above this is consumed without a verdict: it is not a header this protocol
/// could ever have produced, so it belongs to a different tag user.
pub(crate) const MAX_MALFORMED_FRAME: i64 = 1 << 20;
pub(crate) const MAGIC: [u8; 4] = *b"TFMP";
pub(crate) const KIND_HEADER: u8 = 1;
pub(crate) const KIND_ACK: u8 = 2;
pub(crate) const STATUS_OK: u8 = 0;

/// Byte order marker for a little-endian peer.
pub(crate) const BYTE_ORDER_LE: u8 = 1;
/// Byte order marker for a big-endian peer.
pub(crate) const BYTE_ORDER_BE: u8 = 2;

/// This host's byte-order marker.
pub(crate) const HOST_BYTE_ORDER: u8 = if cfg!(target_endian = "little") {
    BYTE_ORDER_LE
} else {
    BYTE_ORDER_BE
};

/// A header frame, decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Header {
    pub(crate) version: u16,
    pub(crate) dtype_code: u8,
    pub(crate) status: u8,
    pub(crate) byte_order: u8,
    pub(crate) elements: u64,
    shape: [usize; MAX_WIRE_RANK],
    rank: usize,
}

impl Header {
    pub(crate) fn new(dtype_code: u8, byte_order: u8, elements: u64, shape: &[usize]) -> Self {
        let mut extents = [0usize; MAX_WIRE_RANK];
        for (slot, extent) in extents.iter_mut().zip(shape) {
            *slot = *extent;
        }
        Self {
            version: PROTOCOL_VERSION,
            dtype_code,
            status: STATUS_OK,
            byte_order,
            elements,
            shape: extents,
            rank: shape.len(),
        }
    }

    /// A header that carries only an abort status: no dtype, rank or payload.
    pub(crate) fn abort(reason: u8) -> Self {
        Self {
            version: PROTOCOL_VERSION,
            dtype_code: 0,
            status: reason,
            byte_order: 0,
            elements: 0,
            shape: [0usize; MAX_WIRE_RANK],
            rank: 0,
        }
    }

    pub(crate) fn rank(&self) -> usize {
        self.rank
    }

    pub(crate) fn shape(&self) -> &[usize] {
        &self.shape[..self.rank]
    }

    /// Encoded length in bytes.
    pub(crate) fn encoded_len(&self) -> usize {
        HEADER_PROLOGUE + SHAPE_STRIDE * self.rank
    }

    /// Encode into `buffer`, which must hold at least [`Self::encoded_len`] bytes.
    pub(crate) fn encode(&self, buffer: &mut [u8]) -> usize {
        let len = self.encoded_len();
        let out = &mut buffer[..len];
        out[..4].copy_from_slice(&MAGIC);
        out[4..6].copy_from_slice(&self.version.to_le_bytes());
        out[6] = KIND_HEADER;
        out[7] = self.dtype_code;
        out[8] = self.status;
        out[9] = self.byte_order;
        out[10..12].fill(0);
        out[12..16].copy_from_slice(&(self.rank as u32).to_le_bytes());
        out[16..24].copy_from_slice(&self.elements.to_le_bytes());
        for (index, extent) in self.shape().iter().enumerate() {
            let start = HEADER_PROLOGUE + SHAPE_STRIDE * index;
            out[start..start + SHAPE_STRIDE].copy_from_slice(&(*extent as u64).to_le_bytes());
        }
        len
    }

    /// Decode a header frame.
    ///
    /// Structural checks only: the magic, the declared rank bound and the frame
    /// length. The version, dtype code, element count and byte order are checked by
    /// the caller so that each has its own typed rejection.
    pub(crate) fn decode(bytes: &[u8]) -> std::result::Result<Self, DecodeError> {
        if bytes.len() < HEADER_PROLOGUE {
            return Err(DecodeError::Length("shorter than the fixed header"));
        }
        if bytes[..4] != MAGIC {
            return Err(DecodeError::Magic);
        }
        if bytes[6] != KIND_HEADER {
            return Err(DecodeError::Kind(bytes[6]));
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        let rank = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
        if rank > MAX_WIRE_RANK {
            return Err(DecodeError::Length("declared rank exceeds MAX_WIRE_RANK"));
        }
        if bytes.len() != HEADER_PROLOGUE + SHAPE_STRIDE * rank {
            return Err(DecodeError::Length(
                "length does not match the declared rank",
            ));
        }
        let mut shape = [0usize; MAX_WIRE_RANK];
        for (index, extent) in shape.iter_mut().enumerate().take(rank) {
            let start = HEADER_PROLOGUE + SHAPE_STRIDE * index;
            let raw = u64::from_le_bytes(
                bytes[start..start + SHAPE_STRIDE]
                    .try_into()
                    .map_err(|_| DecodeError::Length("truncated shape extent"))?,
            );
            *extent = usize::try_from(raw).map_err(|_| DecodeError::Extent)?;
        }
        Ok(Self {
            version,
            dtype_code: bytes[7],
            status: bytes[8],
            byte_order: bytes[9],
            elements: u64::from_le_bytes(
                bytes[16..24]
                    .try_into()
                    .map_err(|_| DecodeError::Length("truncated element count"))?,
            ),
            shape,
            rank,
        })
    }
}

/// An ack frame, decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Ack {
    pub(crate) version: u16,
    pub(crate) verdict: u8,
}

impl Ack {
    pub(crate) fn new(verdict: u8) -> Self {
        Self {
            version: PROTOCOL_VERSION,
            verdict,
        }
    }

    pub(crate) fn encode(&self) -> [u8; ACK_LEN] {
        let mut out = [0u8; ACK_LEN];
        out[..4].copy_from_slice(&MAGIC);
        out[4..6].copy_from_slice(&self.version.to_le_bytes());
        out[6] = KIND_ACK;
        out[7] = self.verdict;
        out
    }

    pub(crate) fn decode(bytes: &[u8]) -> std::result::Result<Self, DecodeError> {
        if bytes.len() < ACK_LEN {
            return Err(DecodeError::Length("ack frame is shorter than 8 bytes"));
        }
        if bytes[..4] != MAGIC {
            return Err(DecodeError::Magic);
        }
        if bytes[6] != KIND_ACK {
            return Err(DecodeError::Kind(bytes[6]));
        }
        Ok(Self {
            version: u16::from_le_bytes([bytes[4], bytes[5]]),
            verdict: bytes[7],
        })
    }
}

/// Why a frame could not be decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DecodeError {
    /// The frame does not carry this protocol's magic.
    Magic,
    /// The frame has an unexpected length or declared rank.
    Length(&'static str),
    /// The frame's kind byte is not the expected one.
    Kind(u8),
    /// A shape extent does not fit this platform's `usize`.
    Extent,
}

/// The wire code for a dtype, or `None` when the dtype has no encoding.
pub(crate) fn dtype_code(dtype: DType) -> Option<u8> {
    match dtype {
        DType::Bool => Some(1),
        DType::I32 => Some(2),
        DType::I64 => Some(3),
        DType::F32 => Some(4),
        DType::F64 => Some(5),
        DType::C32 => Some(6),
        DType::C64 => Some(7),
        DType::External(_) => None,
    }
}

/// The dtype a wire code stands for, or `None` for an unknown or reserved code.
pub(crate) fn dtype_from_code(code: u8) -> Option<DType> {
    match code {
        1 => Some(DType::Bool),
        2 => Some(DType::I32),
        3 => Some(DType::I64),
        4 => Some(DType::F32),
        5 => Some(DType::F64),
        6 => Some(DType::C32),
        7 => Some(DType::C64),
        _ => None,
    }
}

/// Byte length of one element of a wire code.
pub(crate) fn element_bytes(code: u8) -> Option<usize> {
    dtype_from_code(code).map(|dtype| match dtype {
        DType::Bool => std::mem::size_of::<bool>(),
        DType::I32 => std::mem::size_of::<i32>(),
        DType::I64 => std::mem::size_of::<i64>(),
        DType::F32 => std::mem::size_of::<f32>(),
        DType::F64 => std::mem::size_of::<f64>(),
        DType::C32 => std::mem::size_of::<Complex32>(),
        DType::C64 => std::mem::size_of::<Complex64>(),
        DType::External(_) => 0,
    })
}

/// Checked shape product.
pub(crate) fn checked_elements(shape: &[usize]) -> Option<u64> {
    let mut total: u64 = 1;
    for extent in shape {
        total = total.checked_mul(*extent as u64)?;
    }
    Some(total)
}

/// Checked payload byte length for `elements` items of `element_bytes`.
pub(crate) fn checked_payload_bytes(elements: u64, element_bytes: usize) -> Option<i64> {
    let bytes = elements.checked_mul(element_bytes as u64)?;
    i64::try_from(bytes).ok()
}

/// Scalars this crate can put on the wire.
///
/// Sealed: the seven tenferro preset scalars and nothing else. A caller-owned
/// external scalar needs an explicit codec, because a process-local Rust
/// `TypeId` cannot be a wire identifier.
pub(crate) trait WireScalar: TensorScalar + Default + WireScalarSealed {
    /// Whether a received byte may hold a value of `Self` that does not exist.
    ///
    /// Only `bool` does: one byte of payload can be `2`. A receiver must then fill the
    /// destination from validated values instead of writing into it, because a `bool`
    /// holding `2` is undefined behaviour the moment anything reads it.
    const VALIDATES_PAYLOAD: bool;

    /// View the slice as wire bytes.
    fn bytes(slice: &[Self]) -> &[u8];
    /// View the slice as mutable wire bytes.
    fn bytes_mut(slice: &mut [Self]) -> &mut [u8];
    /// Fill `out` from received `bytes`, and report whether every byte was a value of
    /// `Self`. On `false`, `out` is left untouched.
    fn assign_from_bytes(out: &mut [Self], bytes: &[u8]) -> bool;
}

mod sealed {
    pub(crate) trait WireScalarSealed {}
}

use sealed::WireScalarSealed;

/// The six scalars that need no validation: no padding, and every bit pattern is a
/// valid value.
macro_rules! plain_wire_scalar {
    ($($ty:ty),* $(,)?) => {$(
        impl WireScalarSealed for $ty {}

        impl WireScalar for $ty {
            const VALIDATES_PAYLOAD: bool = false;

            fn bytes(slice: &[Self]) -> &[u8] {
                view_as_bytes(slice)
            }

            fn bytes_mut(slice: &mut [Self]) -> &mut [u8] {
                view_as_bytes_mut(slice)
            }

            fn assign_from_bytes(_out: &mut [Self], _bytes: &[u8]) -> bool {
                // Unused: with `VALIDATES_PAYLOAD = false` the receiver copies straight
                // into the destination, so no byte buffer is ever assigned from.
                unreachable!("a plain scalar payload is received in place")
            }
        }
    )*};
}

plain_wire_scalar!(f32, f64, i32, i64, Complex32, Complex64);

impl WireScalarSealed for bool {}

impl WireScalar for bool {
    const VALIDATES_PAYLOAD: bool = true;

    fn bytes(slice: &[Self]) -> &[u8] {
        bool_wire::as_bytes(slice)
    }

    fn bytes_mut(slice: &mut [Self]) -> &mut [u8] {
        // Sound in itself: every `bool` in the slice is already 0 or 1. A receiver must
        // not use it, which is why `VALIDATES_PAYLOAD` sends a payload through a byte
        // scratch buffer instead.
        bool_wire::as_bytes_mut(slice)
    }

    fn assign_from_bytes(out: &mut [Self], bytes: &[u8]) -> bool {
        bool_wire::assign(out, bytes)
    }
}

/// View a slice of a padding-free, all-bit-patterns-valid scalar as bytes.
///
/// # Safety
///
/// The caller must be an instantiation of [`plain_wire_scalar`], whose types are
/// `f32`, `f64`, `i32`, `i64`, `Complex32` and `Complex64`: each has no padding
/// bytes and every bit pattern is a valid value, so the byte view is always a valid
/// view of the same memory.
fn view_as_bytes<T>(slice: &[T]) -> &[u8] {
    // SAFETY: see the function contract; `size_of_val` covers exactly the slice.
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), std::mem::size_of_val(slice)) }
}

/// Mutable counterpart of [`view_as_bytes`], with the same contract.
fn view_as_bytes_mut<T>(slice: &mut [T]) -> &mut [u8] {
    // SAFETY: see `view_as_bytes`; the exclusive borrow keeps the view unique.
    unsafe {
        std::slice::from_raw_parts_mut(
            slice.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(slice),
        )
    }
}
