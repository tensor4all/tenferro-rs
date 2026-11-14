//! The transport protocol: header → ack → payload.
//!
//! One implementation serves every backend; a backend crate only implements
//! [`MpiBackend`] and re-exports these entry points with its own
//! peer type.

use tenferro_tensor::{
    Complex32, Complex64, DType, DynRank, Dynamic, Host, ShapeVec, Tensor, TensorRank, TypedTensor,
};

use crate::backend::MpiBackend;
use crate::error::{AbortReason, Error, RejectReason, Result};
use crate::wire::{
    checked_elements, checked_payload_bytes, dtype_code, dtype_from_code, element_bytes, Ack,
    DecodeError, Header, Tag, WireScalar, ACK_LEN, HEADER_PROLOGUE, HOST_BYTE_ORDER, KIND_ACK,
    KIND_HEADER, MAGIC, MAX_HEADER_LEN, MAX_MALFORMED_FRAME, MAX_WIRE_ELEMENTS, MAX_WIRE_RANK,
    PROTOCOL_VERSION, SHAPE_STRIDE, STATUS_OK,
};

/// A tensor value the adapter can put on the wire.
///
/// Implemented for [`TypedTensor<T, R, Host>`](TypedTensor), host-resident
/// [`TypedTensor<T, R, Dynamic>`](TypedTensor) and the dtype-erased [`Tensor`].
/// A [`Gpu`](tenferro_tensor::Gpu)-representation tensor does not implement it, so
/// an accidental device transfer is a compile error.
///
/// # Examples
///
/// ```
/// use tenferro_mpi::mpi::traits::*;
/// use tenferro_mpi::{recv, send};
/// use tenferro_tensor::{Host, TypedTensor};
///
/// let universe = tenferro_mpi::mpi::initialize().unwrap();
/// let world = universe.world();
/// let value = TypedTensor::<f64, tenferro_tensor::DynRank, Host>::from_host_vec_col_major(
///     vec![2],
///     vec![1.0, 2.0],
/// )?;
/// // A blocking send to the calling rank could never be received, so it is refused
/// // before any message.
/// assert!(matches!(
///     send(world.this_process(), 7, &value),
///     Err(tenferro_mpi::Error::SelfMessage)
/// ));
/// // A blocking send needs a posted receive, so one rank sends while the other takes.
/// if world.size() >= 2 {
///     if world.rank() == 0 {
///         send(world.process_at_rank(1), 7, &value)?;
///     } else if world.rank() == 1 {
///         let received: TypedTensor<f64, tenferro_tensor::DynRank, Host> =
///             recv(world.process_at_rank(0), 7)?;
///         assert_eq!(received.as_slice(), &[1.0, 2.0]);
///     }
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait Transport: sealed::Sealed + Sized {
    /// Send `value` to `to` on `tag`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidTag`] for a negative tag and [`Error::SelfMessage`]
    /// for a send to the calling rank; neither sends a message. Returns
    /// [`Error::NotHostResident`], [`Error::NonCompact`], [`Error::UnsupportedDType`],
    /// [`Error::RankTooLarge`] or [`Error::TooManyElements`] when the value cannot be
    /// described on the wire (the peer is told with an abort header), and
    /// [`Error::PeerRejected`] when the receiver refuses it.
    fn send_to<B: MpiBackend>(backend: &B, to: B::Peer<'_>, tag: Tag, value: &Self) -> Result<()>;

    /// Receive one value, allocating it once at the received shape.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidTag`] for a negative tag, [`Error::DTypeMismatch`] or
    /// [`Error::RankMismatch`] when the incoming header does not match this receiver
    /// (the sender is told with a reject ack), [`Error::PeerAborted`] when the sender
    /// aborted, and [`Error::PayloadCountMismatch`] / [`Error::InvalidBoolPayload`]
    /// for a payload that disagrees with its header.
    fn recv_from<B: MpiBackend>(backend: &B, from: B::Peer<'_>, tag: Tag) -> Result<Self>;

    /// Receive one value into an existing destination of identical dtype and shape.
    ///
    /// No payload allocation is performed. The destination is left unchanged by
    /// every failure detected before the payload is accepted.
    ///
    /// # Errors
    ///
    /// As [`Transport::recv_from`], plus [`Error::ShapeMismatch`] or
    /// [`Error::RankMismatch`] for a destination of a different shape, and
    /// [`Error::NotHostResident`] / [`Error::NonCompact`] for a destination that
    /// cannot be written as a compact host slice.
    fn recv_from_into<B: MpiBackend>(
        backend: &B,
        from: B::Peer<'_>,
        tag: Tag,
        out: &mut Self,
    ) -> Result<()>;
}

/// Send one tensor value to `to` on `tag`.
///
/// # Examples
///
/// ```
/// use tenferro_mpi::mpi::traits::*;
/// use tenferro_mpi::send;
/// use tenferro_tensor::{Host, TypedTensor};
///
/// let universe = tenferro_mpi::mpi::initialize().unwrap();
/// let world = universe.world();
/// let value = TypedTensor::<f64, tenferro_tensor::DynRank, Host>::from_host_vec_col_major(
///     vec![1],
///     vec![3.0],
/// )?;
/// assert!(matches!(send(world.this_process(), 3, &value), Err(tenferro_mpi::Error::SelfMessage)));
/// if world.size() >= 2 && world.rank() == 0 {
///     send(world.process_at_rank(1), 3, &value)?;
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::InvalidTag`] for a negative tag and [`Error::SelfMessage`] for a
/// send to the calling rank; neither sends a message. Returns
/// [`Error::NotHostResident`], [`Error::NonCompact`], [`Error::UnsupportedDType`],
/// [`Error::RankTooLarge`] or [`Error::TooManyElements`] when the value cannot be
/// described on the wire, and [`Error::PeerRejected`] when the receiver refuses it.
pub fn send<B: MpiBackend, X: Transport>(
    backend: &B,
    to: B::Peer<'_>,
    tag: Tag,
    value: &X,
) -> Result<()> {
    X::send_to(backend, to, tag, value)
}

/// Receive one tensor value from `from` on `tag`.
///
/// # Examples
///
/// ```
/// use tenferro_mpi::mpi::traits::*;
/// use tenferro_mpi::recv;
/// use tenferro_tensor::{DynRank, Host, TypedTensor};
///
/// let universe = tenferro_mpi::mpi::initialize().unwrap();
/// let world = universe.world();
/// if world.size() >= 2 {
///     let sent = TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1.0, 2.0])?;
///     if world.rank() == 0 {
///         tenferro_mpi::send(world.process_at_rank(1), 11, &sent)?;
///     } else if world.rank() == 1 {
///         let received: TypedTensor<f64, DynRank, Host> = recv(world.process_at_rank(0), 11)?;
///         assert_eq!(received.shape(), &[2]);
///     }
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::InvalidTag`] for a negative tag, [`Error::DTypeMismatch`] or
/// [`Error::RankMismatch`] when the incoming header does not match this receiver,
/// [`Error::PeerAborted`] when the sender aborted before offering a payload, and
/// [`Error::PayloadCountMismatch`] or [`Error::InvalidBoolPayload`] for a payload that
/// disagrees with its header.
pub fn recv<B: MpiBackend, X: Transport>(backend: &B, from: B::Peer<'_>, tag: Tag) -> Result<X> {
    X::recv_from(backend, from, tag)
}

/// Receive one tensor value into `out` from `from` on `tag`.
///
/// # Examples
///
/// ```
/// use tenferro_mpi::mpi::traits::*;
/// use tenferro_mpi::recv_into;
/// use tenferro_tensor::{DynRank, Host, TypedTensor};
///
/// let universe = tenferro_mpi::mpi::initialize().unwrap();
/// let world = universe.world();
/// let mut out = TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![2], vec![0.0, 0.0])?;
/// if world.size() >= 2 {
///     let value = TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![2], vec![5.0, 6.0])?;
///     if world.rank() == 0 {
///         tenferro_mpi::send(world.process_at_rank(1), 12, &value)?;
///     } else if world.rank() == 1 {
///         recv_into(world.process_at_rank(0), 12, &mut out)?;
///         assert_eq!(out.as_slice(), &[5.0, 6.0]);
///     }
/// }
/// assert_eq!(out.shape(), &[2]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::InvalidTag`] for a negative tag, [`Error::DTypeMismatch`],
/// [`Error::RankMismatch`] or [`Error::ShapeMismatch`] for a destination of a different
/// dtype or shape, [`Error::NotHostResident`] or [`Error::NonCompact`] for a destination
/// that cannot be written as a compact host slice, [`Error::PeerAborted`] when the
/// sender aborted, and [`Error::PayloadCountMismatch`] or [`Error::InvalidBoolPayload`]
/// for a payload that disagrees with its header.
pub fn recv_into<B: MpiBackend, X: Transport>(
    backend: &B,
    from: B::Peer<'_>,
    tag: Tag,
    out: &mut X,
) -> Result<()> {
    X::recv_from_into(backend, from, tag, out)
}

mod sealed {
    use super::{Dynamic, Host, Tensor, TensorRank, TypedTensor};
    use crate::wire::WireScalar;

    pub trait Sealed {}

    impl<T: WireScalar, R: TensorRank> Sealed for TypedTensor<T, R, Host> {}
    impl<T: WireScalar, R: TensorRank> Sealed for TypedTensor<T, R, Dynamic> {}
    impl Sealed for Tensor {}
}

/// A descriptor that passed every local pre-message check.
struct Prepared<'a> {
    dtype_code: u8,
    elements: u64,
    shape: &'a [usize],
    payload_bytes: i64,
}

fn check_peer<B: MpiBackend>(to: B::Peer<'_>, tag: Tag) -> Result<()> {
    if tag < 0 {
        return Err(Error::InvalidTag { tag });
    }
    if B::is_self(to) {
        return Err(Error::SelfMessage);
    }
    Ok(())
}

fn describe<'a>(dtype_code: u8, shape: &'a [usize], payload_len: usize) -> Result<Prepared<'a>> {
    if shape.len() > MAX_WIRE_RANK {
        return Err(Error::RankTooLarge {
            rank: shape.len(),
            max: MAX_WIRE_RANK,
        });
    }
    let elements = checked_elements(shape).ok_or(Error::MalformedHeader {
        reason: "shape product overflows",
    })?;
    if elements != payload_len as u64 {
        return Err(Error::MalformedHeader {
            reason: "shape product disagrees with the host payload length",
        });
    }
    let width = element_bytes(dtype_code).ok_or(Error::MalformedHeader {
        reason: "dtype code has no element width",
    })?;
    let payload_bytes = payload_bytes_of(elements, width)?;
    Ok(Prepared {
        dtype_code,
        elements,
        shape,
        payload_bytes,
    })
}

fn payload_bytes_of(elements: u64, width: usize) -> Result<i64> {
    if elements > MAX_WIRE_ELEMENTS {
        return Err(Error::TooManyElements {
            elements,
            max: MAX_WIRE_ELEMENTS,
        });
    }
    let bytes = checked_payload_bytes(elements, width).ok_or(Error::TooManyElements {
        elements,
        max: MAX_WIRE_ELEMENTS,
    })?;
    // One message carries the payload in bytes, so the MPI count bound is on bytes.
    if bytes > MAX_WIRE_ELEMENTS as i64 {
        return Err(Error::TooManyElements {
            elements,
            max: MAX_WIRE_ELEMENTS / width.max(1) as u64,
        });
    }
    Ok(bytes)
}

fn send_frames<B: MpiBackend>(
    backend: &B,
    to: B::Peer<'_>,
    tag: Tag,
    prepared: &Prepared<'_>,
    payload: &[u8],
) -> Result<()> {
    let header = Header::new(
        prepared.dtype_code,
        HOST_BYTE_ORDER,
        prepared.elements,
        prepared.shape,
    );
    let mut buffer = [0u8; MAX_HEADER_LEN];
    let len = header.encode(&mut buffer);
    backend.send_bytes(to, tag, &buffer[..len]);

    let verdict = receive_ack(backend, to, tag)?;
    if verdict != STATUS_OK {
        return Err(Error::PeerRejected {
            reason: RejectReason::from_code(verdict),
        });
    }
    if prepared.payload_bytes > 0 {
        backend.send_bytes(to, tag, payload);
    }
    Ok(())
}

fn send_abort<B: MpiBackend>(backend: &B, to: B::Peer<'_>, tag: Tag, error: &Error) {
    let header = Header::abort(error.abort_reason().code());
    let mut buffer = [0u8; MAX_HEADER_LEN];
    let len = header.encode(&mut buffer);
    backend.send_bytes(to, tag, &buffer[..len]);
}

fn abort_and_error<B: MpiBackend>(backend: &B, to: B::Peer<'_>, tag: Tag, error: Error) -> Error {
    send_abort(backend, to, tag, &error);
    error
}

/// Receive a probed message whose length is already known and discard it.
fn consume<B: MpiBackend>(backend: &B, probed: B::Probed, length: i64) {
    let len = usize::try_from(length).unwrap_or(0);
    let mut scratch = vec![0u8; len];
    let _ = backend.receive_probed(probed, &mut scratch);
}

/// Receive a probed message only to read its frame kind, then discard it.
fn consume_kind<B: MpiBackend>(backend: &B, probed: B::Probed, length: i64) -> u8 {
    let len = usize::try_from(length).unwrap_or(0);
    if (HEADER_PROLOGUE..=MAX_HEADER_LEN).contains(&len) {
        let mut buffer = [0u8; MAX_HEADER_LEN];
        let _ = backend.receive_probed(probed, &mut buffer[..len]);
        return buffer[6];
    }
    let mut scratch = vec![0u8; len];
    let _ = backend.receive_probed(probed, &mut scratch);
    scratch.get(6).copied().unwrap_or(0)
}

fn receive_ack<B: MpiBackend>(backend: &B, from: B::Peer<'_>, tag: Tag) -> Result<u8> {
    let (probed, length) = backend.probe(from, tag);
    if length != ACK_LEN as i64 {
        // An ack is exactly eight bytes. Anything else is another frame and must still
        // be consumed: two ranks that send to each other on one tag see the other's
        // header here instead of hanging.
        return Err(Error::UnexpectedFrame {
            kind: consume_kind(backend, probed, length),
        });
    }
    let mut buffer = [0u8; ACK_LEN];
    let received = backend.receive_probed(probed, &mut buffer);
    if received != ACK_LEN as i64 {
        return Err(Error::MalformedHeader {
            reason: "ack frame was truncated",
        });
    }
    let ack = Ack::decode(&buffer).map_err(decode_error)?;
    if ack.version != PROTOCOL_VERSION {
        return Err(Error::UnsupportedVersion {
            received: ack.version,
            supported: PROTOCOL_VERSION,
        });
    }
    Ok(ack.verdict)
}

fn decode_error(error: DecodeError) -> Error {
    match error {
        DecodeError::Magic => Error::MalformedHeader {
            reason: "frame does not carry the TFMP magic",
        },
        DecodeError::Length(reason) => Error::MalformedHeader { reason },
        DecodeError::Kind(kind) => Error::UnexpectedFrame { kind },
        DecodeError::Extent => Error::MalformedHeader {
            reason: "shape extent does not fit this platform's usize",
        },
    }
}

/// Map a local validation error to the ack code that tells the peer about it.
fn reject_reason(error: &Error) -> RejectReason {
    match error {
        Error::DTypeMismatch { .. } => RejectReason::DTypeMismatch,
        Error::RankMismatch { .. } => RejectReason::RankMismatch,
        Error::ShapeMismatch { .. } => RejectReason::ShapeMismatch,
        Error::NotHostResident | Error::NonCompact => RejectReason::DestinationNotWritable,
        Error::TooManyElements { .. } => RejectReason::TooManyElements,
        Error::ByteOrderMismatch { .. } => RejectReason::ByteOrderMismatch,
        Error::UnsupportedVersion { .. } => RejectReason::UnsupportedVersion,
        _ => RejectReason::MalformedHeader,
    }
}

fn send_ack<B: MpiBackend>(backend: &B, to: B::Peer<'_>, tag: Tag, verdict: u8) {
    let ack = Ack::new(verdict);
    backend.send_bytes(to, tag, &ack.encode());
}

fn reject_ack<B: MpiBackend>(backend: &B, to: B::Peer<'_>, tag: Tag, reason: RejectReason) {
    send_ack(backend, to, tag, reason.code());
}

/// Receive the header frame and settle its verdict with the sender.
///
/// `accept` performs the receiver-side checks that need the decoded dtype and
/// shape; its error, or a malformed header, becomes the ack the sender receives.
/// Consume a frame that cannot be a version 1 header and report what it was.
///
/// A peer that sent a TFMP header with a length its own rank field contradicts is still
/// waiting for a verdict, so it gets one; a frame that is not this protocol at all gets
/// nothing, because an ack would be meaningless to its sender.
fn reject_malformed_frame<B: MpiBackend>(
    backend: &B,
    from: B::Peer<'_>,
    tag: Tag,
    probed: B::Probed,
    length: i64,
) -> Error {
    let error = Error::MalformedHeader {
        reason: "header byte count is not a version 1 header length",
    };
    if length <= 0 || length > MAX_MALFORMED_FRAME {
        consume(backend, probed, length);
        return error;
    }
    // Not a successful-call path, so a scratch buffer is fine here.
    let mut scratch = vec![0u8; length as usize];
    let received = backend.receive_probed(probed, &mut scratch);
    let frame = &scratch[..received.clamp(0, length) as usize];
    match frame_kind(frame) {
        FrameKind::Header => reject_ack(backend, from, tag, reject_reason(&error)),
        FrameKind::Ack => return Error::UnexpectedFrame { kind: KIND_ACK },
        FrameKind::Other(kind) => return Error::UnexpectedFrame { kind },
        FrameKind::NotOurs => return error,
    }
    error
}

/// What a consumed frame announces itself as, if anything.
enum FrameKind {
    Header,
    Ack,
    Other(u8),
    NotOurs,
}

fn frame_kind(frame: &[u8]) -> FrameKind {
    if frame.len() < ACK_LEN || frame[..4] != MAGIC {
        return FrameKind::NotOurs;
    }
    match frame[6] {
        KIND_HEADER => FrameKind::Header,
        KIND_ACK => FrameKind::Ack,
        kind => FrameKind::Other(kind),
    }
}

fn negotiate<B: MpiBackend, F>(
    backend: &B,
    from: B::Peer<'_>,
    tag: Tag,
    accept: F,
) -> Result<Header>
where
    F: FnOnce(&Header, DType) -> Result<()>,
{
    let (probed, length) = backend.probe(from, tag);
    if length < HEADER_PROLOGUE as i64
        || length > MAX_HEADER_LEN as i64
        || (length - HEADER_PROLOGUE as i64) % SHAPE_STRIDE as i64 != 0
    {
        return Err(reject_malformed_frame(backend, from, tag, probed, length));
    }
    let mut buffer = [0u8; MAX_HEADER_LEN];
    let received = backend.receive_probed(probed, &mut buffer[..length as usize]);
    if received != length {
        return Err(Error::MalformedHeader {
            reason: "header receive was truncated",
        });
    }
    let header = match Header::decode(&buffer[..length as usize]) {
        Ok(header) => header,
        Err(DecodeError::Magic) => {
            // Not this protocol: an ack would be meaningless.
            return Err(Error::MalformedHeader {
                reason: "frame does not carry the TFMP magic",
            });
        }
        Err(DecodeError::Kind(kind)) => return Err(Error::UnexpectedFrame { kind }),
        Err(other) => {
            let error = decode_error(other);
            reject_ack(backend, from, tag, reject_reason(&error));
            return Err(error);
        }
    };

    if header.status != STATUS_OK {
        // An abort frame carries no payload and expects no ack.
        return Err(Error::PeerAborted {
            reason: AbortReason::from_code(header.status),
        });
    }
    if header.version != PROTOCOL_VERSION {
        reject_ack(backend, from, tag, RejectReason::UnsupportedVersion);
        return Err(Error::UnsupportedVersion {
            received: header.version,
            supported: PROTOCOL_VERSION,
        });
    }
    if header.byte_order != HOST_BYTE_ORDER {
        reject_ack(backend, from, tag, RejectReason::ByteOrderMismatch);
        return Err(Error::ByteOrderMismatch {
            received: header.byte_order,
        });
    }
    let dtype = match dtype_from_code(header.dtype_code) {
        Some(dtype) => dtype,
        None => {
            let error = Error::MalformedHeader {
                reason: "unknown dtype code",
            };
            reject_ack(backend, from, tag, reject_reason(&error));
            return Err(error);
        }
    };
    if checked_elements(header.shape()) != Some(header.elements) {
        let error = Error::MalformedHeader {
            reason: "element count disagrees with the shape",
        };
        reject_ack(backend, from, tag, reject_reason(&error));
        return Err(error);
    }
    if let Err(error) = payload_bytes_of(
        header.elements,
        element_bytes(header.dtype_code).unwrap_or(1),
    ) {
        reject_ack(backend, from, tag, reject_reason(&error));
        return Err(error);
    }
    if let Err(error) = accept(&header, dtype) {
        reject_ack(backend, from, tag, reject_reason(&error));
        return Err(error);
    }
    send_ack(backend, from, tag, STATUS_OK);
    Ok(header)
}

/// Receive the payload frame into `out`.
fn receive_payload<B: MpiBackend, T: WireScalar>(
    backend: &B,
    from: B::Peer<'_>,
    tag: Tag,
    expected_bytes: i64,
    out: &mut [T],
) -> Result<()> {
    if expected_bytes == 0 {
        return Ok(());
    }
    if T::VALIDATES_PAYLOAD {
        // The payload may hold a byte the destination type cannot represent, so it is
        // received into bytes and assigned only once it has been checked. Writing it
        // into `out` first would leave an invalid value there, and unwinding would
        // leave it behind.
        let mut scratch = vec![0u8; expected_bytes as usize];
        receive_frame(backend, from, tag, expected_bytes, &mut scratch)?;
        return if T::assign_from_bytes(out, &scratch) {
            Ok(())
        } else {
            Err(Error::InvalidBoolPayload)
        };
    }
    // No invalid bit pattern exists for `T`, so the payload can go straight in.
    receive_frame(backend, from, tag, expected_bytes, T::bytes_mut(out))
}

/// Receive exactly `expected_bytes` into `buffer`.
///
/// A frame of the wrong length is consumed, so a failed receive leaves no payload in
/// flight for the next call on the tag to trip over.
fn receive_frame<B: MpiBackend>(
    backend: &B,
    from: B::Peer<'_>,
    tag: Tag,
    expected_bytes: i64,
    buffer: &mut [u8],
) -> Result<()> {
    let (probed, length) = backend.probe(from, tag);
    if length != expected_bytes {
        if length > expected_bytes {
            consume(backend, probed, length);
        } else {
            let _ = backend.receive_probed(probed, buffer);
        }
        return Err(Error::PayloadCountMismatch {
            expected: expected_bytes,
            received: length,
        });
    }
    let received = backend.receive_probed(probed, buffer);
    if received != expected_bytes {
        return Err(Error::PayloadCountMismatch {
            expected: expected_bytes,
            received,
        });
    }
    Ok(())
}

fn payload_bytes_for(header: &Header) -> Result<i64> {
    payload_bytes_of(
        header.elements,
        element_bytes(header.dtype_code).ok_or(Error::MalformedHeader {
            reason: "dtype code has no element width",
        })?,
    )
}

fn recv_typed<B: MpiBackend, T: WireScalar, R: TensorRank>(
    backend: &B,
    from: B::Peer<'_>,
    tag: Tag,
) -> Result<TypedTensor<T, R, Host>> {
    let expected = T::dtype();
    let header = negotiate(backend, from, tag, |header, dtype| {
        if dtype != expected {
            return Err(Error::DTypeMismatch {
                expected,
                received: dtype,
            });
        }
        if let Some(rank) = R::RANK {
            if rank != header.rank() {
                return Err(Error::RankMismatch {
                    expected: rank,
                    received: header.rank(),
                });
            }
        }
        Ok(())
    })?;
    let payload_bytes = payload_bytes_for(&header)?;
    let mut data = vec![T::default(); header.elements as usize];
    receive_payload::<B, T>(backend, from, tag, payload_bytes, &mut data)?;
    let value = TypedTensor::<T, R, Host>::from_host_vec_col_major(shape_of::<R>(&header)?, data)?;
    Ok(value)
}

fn recv_host_into<B: MpiBackend, T: WireScalar, R: TensorRank>(
    backend: &B,
    from: B::Peer<'_>,
    tag: Tag,
    out: &mut TypedTensor<T, R, Host>,
) -> Result<()> {
    let expected = T::dtype();
    let header = negotiate(backend, from, tag, |header, dtype| {
        check_destination(header, dtype, expected, out.shape())
    })?;
    let payload_bytes = payload_bytes_for(&header)?;
    let slice = out.host_data_mut();
    if slice.len() as u64 != header.elements {
        return Err(Error::MalformedHeader {
            reason: "destination length disagrees with the received header",
        });
    }
    receive_payload::<B, T>(backend, from, tag, payload_bytes, slice)
}

fn check_destination(
    header: &Header,
    dtype: DType,
    expected: DType,
    shape: &[usize],
) -> Result<()> {
    if dtype != expected {
        return Err(Error::DTypeMismatch {
            expected,
            received: dtype,
        });
    }
    if header.rank() != shape.len() {
        return Err(Error::RankMismatch {
            expected: shape.len(),
            received: header.rank(),
        });
    }
    if !header.shape().iter().copied().eq(shape.iter().copied()) {
        return Err(Error::ShapeMismatch {
            expected: shape.to_vec(),
            received: shape_usize(header),
        });
    }
    Ok(())
}

/// The header shape as a dynamic shape, inline for ordinary ranks.
fn shape_usize(header: &Header) -> Vec<usize> {
    header.shape().to_vec()
}

/// The header shape in `R`'s representation.
///
/// Collected straight into `ShapeVec`, which is inline for an ordinary rank, so a
/// receive does not allocate a shape vector.
fn shape_of<R: TensorRank>(header: &Header) -> Result<R::Shape> {
    let shape: ShapeVec = header.shape().iter().copied().collect();
    R::shape_from_vec(shape).map_err(|_| Error::RankMismatch {
        expected: R::RANK.unwrap_or(header.rank()),
        received: header.rank(),
    })
}

fn dynamic_host_error(is_host: bool) -> Error {
    if is_host {
        Error::NonCompact
    } else {
        Error::NotHostResident
    }
}

fn send_dynamic<B: MpiBackend, T: WireScalar, R: TensorRank>(
    backend: &B,
    to: B::Peer<'_>,
    tag: Tag,
    value: &TypedTensor<T, R, Dynamic>,
) -> Result<()> {
    let is_host = value.is_host_memory();
    let slice = match value.host_data() {
        Ok(slice) => slice,
        Err(_) => {
            let error = dynamic_host_error(is_host);
            return Err(abort_and_error(backend, to, tag, error));
        }
    };
    let dtype_code = dtype_code(T::dtype()).ok_or(Error::UnsupportedDType { dtype: T::dtype() })?;
    match describe(dtype_code, value.shape(), slice.len()) {
        Ok(prepared) => send_frames(backend, to, tag, &prepared, T::bytes(slice)),
        Err(error) => Err(abort_and_error(backend, to, tag, error)),
    }
}

fn send_host<B: MpiBackend, T: WireScalar, R: TensorRank>(
    backend: &B,
    to: B::Peer<'_>,
    tag: Tag,
    value: &TypedTensor<T, R, Host>,
) -> Result<()> {
    let slice = value.as_slice();
    let dtype_code = dtype_code(T::dtype()).ok_or(Error::UnsupportedDType { dtype: T::dtype() })?;
    match describe(dtype_code, value.shape(), slice.len()) {
        Ok(prepared) => send_frames(backend, to, tag, &prepared, T::bytes(slice)),
        Err(error) => Err(abort_and_error(backend, to, tag, error)),
    }
}

fn recv_dynamic_into<B: MpiBackend, T: WireScalar, R: TensorRank>(
    backend: &B,
    from: B::Peer<'_>,
    tag: Tag,
    out: &mut TypedTensor<T, R, Dynamic>,
) -> Result<()> {
    let is_host = out.is_host_memory();
    let expected = T::dtype();
    let header = negotiate(backend, from, tag, |header, dtype| {
        if !is_host {
            return Err(Error::NotHostResident);
        }
        let written = out.host_data().map_err(|_| dynamic_host_error(is_host))?;
        if written.len() as u64 != header.elements {
            return Err(Error::ShapeMismatch {
                expected: out.shape().to_vec(),
                received: shape_usize(header),
            });
        }
        check_destination(header, dtype, expected, out.shape())
    })?;
    let payload_bytes = payload_bytes_for(&header)?;
    match out.host_data_mut() {
        Ok(slice) if slice.len() as u64 == header.elements => {
            receive_payload::<B, T>(backend, from, tag, payload_bytes, slice)
        }
        Ok(_) => Err(Error::MalformedHeader {
            reason: "destination length disagrees with the received header",
        }),
        Err(_) => Err(dynamic_host_error(is_host)),
    }
}

/// Dispatch the dtype-erased [`Tensor`] onto one typed payload.
macro_rules! dispatch_erased {
    ($tensor:expr, $typed:ident => $body:block) => {{
        let dtype = $tensor.dtype();
        match dtype {
            DType::Bool => {
                let Some($typed) = $tensor.as_typed::<bool>() else {
                    return Err(Error::UnsupportedDType { dtype });
                };
                $body
            }
            DType::I32 => {
                let Some($typed) = $tensor.as_typed::<i32>() else {
                    return Err(Error::UnsupportedDType { dtype });
                };
                $body
            }
            DType::I64 => {
                let Some($typed) = $tensor.as_typed::<i64>() else {
                    return Err(Error::UnsupportedDType { dtype });
                };
                $body
            }
            DType::F32 => {
                let Some($typed) = $tensor.as_typed::<f32>() else {
                    return Err(Error::UnsupportedDType { dtype });
                };
                $body
            }
            DType::F64 => {
                let Some($typed) = $tensor.as_typed::<f64>() else {
                    return Err(Error::UnsupportedDType { dtype });
                };
                $body
            }
            DType::C32 => {
                let Some($typed) = $tensor.as_typed::<Complex32>() else {
                    return Err(Error::UnsupportedDType { dtype });
                };
                $body
            }
            DType::C64 => {
                let Some($typed) = $tensor.as_typed::<Complex64>() else {
                    return Err(Error::UnsupportedDType { dtype });
                };
                $body
            }
            _ => Err(Error::UnsupportedDType { dtype }),
        }
    }};
}

fn recv_erased<B: MpiBackend>(backend: &B, from: B::Peer<'_>, tag: Tag) -> Result<Tensor> {
    let header = negotiate(backend, from, tag, |_header, _dtype| Ok(()))?;
    let payload_bytes = payload_bytes_for(&header)?;
    macro_rules! typed {
        ($ty:ty) => {{
            let mut data = vec![<$ty>::default(); header.elements as usize];
            receive_payload::<B, $ty>(backend, from, tag, payload_bytes, &mut data)?;
            let shape = shape_of::<DynRank>(&header)?;
            let tensor = TypedTensor::<$ty, DynRank, Host>::from_host_vec_col_major(shape, data)?;
            Ok(Tensor::from_typed(tensor.into_dynamic()))
        }};
    }
    match dtype_from_code(header.dtype_code) {
        Some(DType::Bool) => typed!(bool),
        Some(DType::I32) => typed!(i32),
        Some(DType::I64) => typed!(i64),
        Some(DType::F32) => typed!(f32),
        Some(DType::F64) => typed!(f64),
        Some(DType::C32) => typed!(Complex32),
        Some(DType::C64) => typed!(Complex64),
        _ => Err(Error::MalformedHeader {
            reason: "unknown dtype code",
        }),
    }
}

fn recv_erased_into<B: MpiBackend>(
    backend: &B,
    from: B::Peer<'_>,
    tag: Tag,
    out: &mut Tensor,
) -> Result<()> {
    let expected = out.dtype();
    // Writability is a property of the destination, not of the header, so it is settled
    // before the ack: a device-backed or non-compact `Tensor` must not learn "accept"
    // and then have nowhere to put the payload the sender would immediately send.
    let is_host = out.is_host_memory();
    let writable = match expected {
        DType::Bool => out.as_slice::<bool>().is_ok(),
        DType::I32 => out.as_slice::<i32>().is_ok(),
        DType::I64 => out.as_slice::<i64>().is_ok(),
        DType::F32 => out.as_slice::<f32>().is_ok(),
        DType::F64 => out.as_slice::<f64>().is_ok(),
        DType::C32 => out.as_slice::<Complex32>().is_ok(),
        DType::C64 => out.as_slice::<Complex64>().is_ok(),
        DType::External(_) => false,
    };
    let header = negotiate(backend, from, tag, |header, dtype| {
        if !writable {
            return Err(dynamic_host_error(is_host));
        }
        check_destination(header, dtype, expected, out.shape())
    })?;
    let payload_bytes = payload_bytes_for(&header)?;
    macro_rules! typed {
        ($ty:ty) => {{
            let slice = out.as_slice_mut::<$ty>()?;
            if slice.len() as u64 != header.elements {
                return Err(Error::MalformedHeader {
                    reason: "destination length disagrees with the received header",
                });
            }
            receive_payload::<B, $ty>(backend, from, tag, payload_bytes, slice)
        }};
    }
    match dtype_from_code(header.dtype_code) {
        Some(DType::Bool) => typed!(bool),
        Some(DType::I32) => typed!(i32),
        Some(DType::I64) => typed!(i64),
        Some(DType::F32) => typed!(f32),
        Some(DType::F64) => typed!(f64),
        Some(DType::C32) => typed!(Complex32),
        Some(DType::C64) => typed!(Complex64),
        _ => Err(Error::MalformedHeader {
            reason: "unknown dtype code",
        }),
    }
}

impl<T: WireScalar, R: TensorRank> Transport for TypedTensor<T, R, Host> {
    fn send_to<B: MpiBackend>(backend: &B, to: B::Peer<'_>, tag: Tag, value: &Self) -> Result<()> {
        check_peer::<B>(to, tag)?;
        send_host(backend, to, tag, value)
    }

    fn recv_from<B: MpiBackend>(backend: &B, from: B::Peer<'_>, tag: Tag) -> Result<Self> {
        if tag < 0 {
            return Err(Error::InvalidTag { tag });
        }
        recv_typed::<B, T, R>(backend, from, tag)
    }

    fn recv_from_into<B: MpiBackend>(
        backend: &B,
        from: B::Peer<'_>,
        tag: Tag,
        out: &mut Self,
    ) -> Result<()> {
        if tag < 0 {
            return Err(Error::InvalidTag { tag });
        }
        recv_host_into::<B, T, R>(backend, from, tag, out)
    }
}

impl<T: WireScalar, R: TensorRank> Transport for TypedTensor<T, R, Dynamic> {
    fn send_to<B: MpiBackend>(backend: &B, to: B::Peer<'_>, tag: Tag, value: &Self) -> Result<()> {
        check_peer::<B>(to, tag)?;
        send_dynamic(backend, to, tag, value)
    }

    fn recv_from<B: MpiBackend>(backend: &B, from: B::Peer<'_>, tag: Tag) -> Result<Self> {
        if tag < 0 {
            return Err(Error::InvalidTag { tag });
        }
        recv_typed::<B, T, R>(backend, from, tag).map(|value| value.into_dynamic())
    }

    fn recv_from_into<B: MpiBackend>(
        backend: &B,
        from: B::Peer<'_>,
        tag: Tag,
        out: &mut Self,
    ) -> Result<()> {
        if tag < 0 {
            return Err(Error::InvalidTag { tag });
        }
        recv_dynamic_into::<B, T, R>(backend, from, tag, out)
    }
}

impl Transport for Tensor {
    fn send_to<B: MpiBackend>(backend: &B, to: B::Peer<'_>, tag: Tag, value: &Self) -> Result<()> {
        check_peer::<B>(to, tag)?;
        // A dtype with no wire code must still tell the peer it is not coming, or the
        // peer waits forever for a payload behind a header it will never see.
        let dtype = value.dtype();
        if dtype_code(dtype).is_none() {
            return Err(abort_and_error(
                backend,
                to,
                tag,
                Error::UnsupportedDType { dtype },
            ));
        }
        dispatch_erased!(value, typed => { send_dynamic(backend, to, tag, typed) })
    }

    fn recv_from<B: MpiBackend>(backend: &B, from: B::Peer<'_>, tag: Tag) -> Result<Self> {
        if tag < 0 {
            return Err(Error::InvalidTag { tag });
        }
        recv_erased(backend, from, tag)
    }

    fn recv_from_into<B: MpiBackend>(
        backend: &B,
        from: B::Peer<'_>,
        tag: Tag,
        out: &mut Self,
    ) -> Result<()> {
        if tag < 0 {
            return Err(Error::InvalidTag { tag });
        }
        recv_erased_into(backend, from, tag, out)
    }
}
