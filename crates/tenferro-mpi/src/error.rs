//! Errors the transport adapter reports.

use std::fmt;

use tenferro_tensor::DType;

/// Result alias for this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Why a receiver refused a tensor before its payload was sent.
///
/// The code travels in the ack frame, so the sender learns the reason instead of
/// only observing that its send failed.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    /// The receiver's element type differs from the header's dtype.
    DTypeMismatch,
    /// The receiver's rank differs from the header's rank.
    RankMismatch,
    /// The receiver's shape differs from the header's shape.
    ShapeMismatch,
    /// The receiver does not implement the header's protocol version.
    UnsupportedVersion,
    /// The header frame could not be decoded.
    MalformedHeader,
    /// The declared element count exceeds one MPI message.
    TooManyElements,
    /// The destination buffer refuses host writes (device-only or non-compact).
    DestinationNotWritable,
    /// The peer's element byte order differs from the receiver's.
    ByteOrderMismatch,
    /// A reject code this version does not know.
    Unknown(u8),
}

impl RejectReason {
    pub(crate) fn code(self) -> u8 {
        match self {
            Self::DTypeMismatch => 1,
            Self::RankMismatch => 2,
            Self::ShapeMismatch => 3,
            Self::UnsupportedVersion => 4,
            Self::MalformedHeader => 5,
            Self::TooManyElements => 6,
            Self::DestinationNotWritable => 7,
            Self::ByteOrderMismatch => 8,
            Self::Unknown(code) => code,
        }
    }

    pub(crate) fn from_code(code: u8) -> Self {
        match code {
            0 => Self::MalformedHeader,
            1 => Self::DTypeMismatch,
            2 => Self::RankMismatch,
            3 => Self::ShapeMismatch,
            4 => Self::UnsupportedVersion,
            5 => Self::MalformedHeader,
            6 => Self::TooManyElements,
            7 => Self::DestinationNotWritable,
            8 => Self::ByteOrderMismatch,
            other => Self::Unknown(other),
        }
    }
}

/// Why a sender aborted before offering a payload.
///
/// An abort is sent as a header frame with a non-zero status and no payload, so a
/// receiver that is already blocking learns the outcome instead of hanging.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbortReason {
    /// The tensor's storage is not host-addressed memory.
    NotHostResident,
    /// The payload is not one compact column-major slice.
    NonCompact,
    /// The dtype has no wire encoding (a caller-owned external payload).
    UnsupportedDType,
    /// The element count exceeds one MPI message.
    TooManyElements,
    /// The rank exceeds [`crate::MAX_WIRE_RANK`].
    RankTooLarge,
    /// The host payload could not be borrowed.
    HostAccess,
    /// An abort code this version does not know.
    Unknown(u8),
}

impl AbortReason {
    pub(crate) fn code(self) -> u8 {
        match self {
            Self::NotHostResident => 1,
            Self::NonCompact => 2,
            Self::UnsupportedDType => 3,
            Self::TooManyElements => 4,
            Self::RankTooLarge => 5,
            Self::HostAccess => 6,
            Self::Unknown(code) => code,
        }
    }

    pub(crate) fn from_code(code: u8) -> Self {
        match code {
            1 => Self::NotHostResident,
            2 => Self::NonCompact,
            3 => Self::UnsupportedDType,
            4 => Self::TooManyElements,
            5 => Self::RankTooLarge,
            6 => Self::HostAccess,
            other => Self::Unknown(other),
        }
    }
}

/// Failure of one transport call.
///
/// Variants split into three groups: local argument and payload checks (nothing is
/// sent), receiver-side validation the sender is told about through the ack frame
/// ([`Self::PeerRejected`]), and sender-side aborts the receiver is told about
/// through an abort header ([`Self::PeerAborted`]).
///
/// # Examples
///
/// ```
/// use tenferro_mpi::Error;
///
/// let error = Error::InvalidTag { tag: -1 };
/// assert_eq!(error.to_string(), "tag -1 is negative");
/// ```
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The tensor's storage is not host-addressed memory; nothing was sent.
    ///
    /// The adapter never transfers between host and device: download the tensor
    /// explicitly (for a `Dynamic` owner, `TypedTensor::into_host`) first.
    #[error("tensor is not host-resident; download it explicitly before sending")]
    NotHostResident,
    /// The payload is not one compact column-major host slice; nothing was sent.
    #[error("tensor payload is not a compact column-major host slice")]
    NonCompact,
    /// The dtype has no wire encoding; nothing was sent.
    #[error("dtype {dtype:?} has no MPI wire encoding")]
    UnsupportedDType {
        /// The dtype that could not be encoded.
        dtype: DType,
    },
    /// More elements than one MPI message can carry; nothing was sent.
    #[error("{elements} elements exceed the per-message limit {max}")]
    TooManyElements {
        /// Declared element count.
        elements: u64,
        /// Largest element count the wire protocol accepts.
        max: u64,
    },
    /// The rank exceeds [`crate::MAX_WIRE_RANK`]; nothing was sent.
    #[error("tensor rank {rank} exceeds the wire limit {max}")]
    RankTooLarge {
        /// Declared rank.
        rank: usize,
        /// Largest rank the header can carry.
        max: usize,
    },
    /// The tag is negative; nothing was sent.
    #[error("tag {tag} is negative")]
    InvalidTag {
        /// The rejected tag.
        tag: i32,
    },
    /// A blocking send to the calling rank would deadlock; nothing was sent.
    #[error("blocking self-send would deadlock; send to a different process")]
    SelfMessage,
    /// The receiver holds a different element type; the payload was not sent.
    #[error("dtype mismatch: expected {expected:?}, received {received:?}")]
    DTypeMismatch {
        /// The receiver's element type.
        expected: DType,
        /// The dtype the header declared.
        received: DType,
    },
    /// The receiver holds a different rank; the payload was not sent.
    #[error("rank mismatch: expected {expected}, received {received}")]
    RankMismatch {
        /// The receiver's rank.
        expected: usize,
        /// The rank the header declared.
        received: usize,
    },
    /// The receiver holds a different shape; the payload was not sent.
    #[error("shape mismatch: expected {expected:?}, received {received:?}")]
    ShapeMismatch {
        /// The receiver's shape.
        expected: Vec<usize>,
        /// The shape the header declared.
        received: Vec<usize>,
    },
    /// The frame's protocol version is not this crate's.
    #[error("unsupported protocol version {received} (supported {supported})")]
    UnsupportedVersion {
        /// Version found in the frame.
        received: u16,
        /// Version this crate implements.
        supported: u16,
    },
    /// A frame could not be decoded.
    #[error("malformed frame: {reason}")]
    MalformedHeader {
        /// Which part of the frame failed.
        reason: &'static str,
    },
    /// A frame arrived where the protocol expects a different one.
    ///
    /// Two ranks that send to each other on one tag at the same time each see the
    /// other's header here, instead of hanging.
    #[error("unexpected frame kind {kind}")]
    UnexpectedFrame {
        /// The frame kind that arrived.
        kind: u8,
    },
    /// The peer's element byte order differs from the receiver's.
    #[error("peer element byte order {received} differs from this host")]
    ByteOrderMismatch {
        /// The byte-order marker the header declared.
        received: u8,
    },
    /// The receiver refused the tensor; it sent this reason in the ack frame.
    #[error("peer rejected the tensor: {reason:?}")]
    PeerRejected {
        /// The receiver's reason.
        reason: RejectReason,
    },
    /// The sender aborted before sending a payload.
    #[error("peer aborted the send: {reason:?}")]
    PeerAborted {
        /// The sender's reason.
        reason: AbortReason,
    },
    /// The received payload length disagrees with the header.
    #[error("payload byte count mismatch: expected {expected}, received {received}")]
    PayloadCountMismatch {
        /// Byte count the header promised.
        expected: i64,
        /// Byte count the frame actually carried.
        received: i64,
    },
    /// A `bool` payload carried a byte other than 0 or 1.
    #[error("bool payload contained a byte other than 0 or 1")]
    InvalidBoolPayload,
    /// A tenferro tensor accessor failed.
    #[error(transparent)]
    Tensor(#[from] tenferro_tensor::Error),
}

impl Error {
    pub(crate) fn abort_reason(&self) -> AbortReason {
        match self {
            Self::NotHostResident => AbortReason::NotHostResident,
            Self::NonCompact => AbortReason::NonCompact,
            Self::UnsupportedDType { .. } => AbortReason::UnsupportedDType,
            Self::TooManyElements { .. } => AbortReason::TooManyElements,
            Self::RankTooLarge { .. } => AbortReason::RankTooLarge,
            Self::Tensor(_) => AbortReason::HostAccess,
            _ => AbortReason::HostAccess,
        }
    }
}

impl fmt::Display for AbortReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}
