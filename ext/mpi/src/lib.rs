//! Blocking point-to-point MPI transport for tenferro host tensors.
//!
//! This crate moves **one tensor value** between two MPI ranks. It borrows the
//! caller's communicator and owns nothing else: it never calls `MPI_Init` or
//! `MPI_Finalize`, never creates `MPI_COMM_WORLD`, never duplicates a communicator,
//! and never transfers between host and device. Distributed scheduling stays with
//! the caller (for example hataori); this is transport only.
//!
//! # What can be sent
//!
//! [`Transport`] covers the three values the adapter accepts, with dtype and shape
//! preserved:
//!
//! - [`TypedTensor<T, R, Host>`](tenferro_tensor::TypedTensor), the statically
//!   host-owned tensor;
//! - a host-resident [`TypedTensor<T, R, Dynamic>`](tenferro_tensor::TypedTensor);
//! - the dtype-erased [`Tensor`](tenferro_tensor::Tensor).
//!
//! A `Gpu`-representation tensor does not implement [`Transport`], so an accidental
//! device transfer is a compile error. Payloads must be compact column-major host
//! slices; the adapter never materializes a view.
//!
//! # Wire format
//!
//! Each call sends a header, waits for the receiver's ack, and only then sends the
//! payload, so a receiver can refuse a tensor without ever receiving its payload
//! and a sender always learns the outcome. Every frame uses **the caller's single
//! tag**, which therefore belongs to that pair for the duration of the call. The
//! format is versioned (v1) and is specified in `docs/design/mpi-transport.md` of
//! the tenferro-rs repository; the pieces a caller can observe are:
//!
//! - supported dtypes are `bool`, `i32`, `i64`, `f32`, `f64`, `Complex32` and
//!   `Complex64`, plus nothing else: a caller-owned external scalar needs an
//!   explicit codec, because a process-local `TypeId` cannot be a wire identifier;
//! - `bool` travels as one byte per element, 0 or 1, validated on receipt;
//! - the payload is the compact column-major element representation in the host's
//!   byte order, and the header carries a byte-order marker, so a peer with a
//!   different byte order is rejected explicitly rather than mis-decoded;
//! - one message carries at most [`MAX_WIRE_ELEMENTS`] elements, and the payload
//!   also has to fit one MPI byte count; larger tensors are rejected, never
//!   chunked or truncated.
//!
//! # Backend features
//!
//! `upstream` (default) builds against crates.io [`mpi`](https://crates.io/crates/mpi)
//! and therefore needs system MPI headers and libclang at build time. The
//! runtime-loading rsmpi-rt backend lives in the unpublished `tenferro-mpi-rt`
//! crate, which depends on this crate with `features = ["engine"]`; `engine` alone
//! builds the protocol without any MPI dependency.
//!
//! # Examples
//!
//! ```
//! use tenferro_mpi::mpi::traits::*;
//! use tenferro_mpi::{recv, send, Error};
//! use tenferro_tensor::{DynRank, Host, Tensor, TypedTensor};
//!
//! let universe = tenferro_mpi::mpi::initialize().unwrap();
//! let world = universe.world();
//!
//! let typed =
//!     TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
//! let erased = Tensor::from_typed(
//!     TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?
//!         .into_dynamic(),
//! );
//!
//! // A blocking send to the calling rank could never be received, so it is refused
//! // before any message.
//! assert!(matches!(send(world.this_process(), 5, &typed), Err(Error::SelfMessage)));
//!
//! // One rank sends while the other receives: a blocking send needs a posted
//! // receive, so both ranks may not send at each other.
//! if world.size() >= 2 {
//!     if world.rank() == 0 {
//!         send(world.process_at_rank(1), 5, &typed)?;
//!         send(world.process_at_rank(1), 6, &erased)?;
//!     } else if world.rank() == 1 {
//!         let typed_back: TypedTensor<f64, DynRank, Host> = recv(world.process_at_rank(0), 5)?;
//!         let erased_back: Tensor = recv(world.process_at_rank(0), 6)?;
//!         assert_eq!(typed_back.get(&[1, 1])?, &4.0);
//!         assert_eq!(erased_back.shape(), &[2, 2]);
//!     }
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![deny(missing_docs)]

#[cfg(not(any(feature = "upstream", feature = "engine")))]
compile_error!(
    "tenferro-mpi: enable the `upstream` MPI backend feature, or `engine` when a \
     separate backend crate (such as tenferro-mpi-rt) supplies the MPI bindings"
);

mod backend;
pub mod engine;
mod error;
mod wire;

pub use backend::MpiBackend;
#[cfg(feature = "upstream")]
pub use backend::Upstream;
pub use engine::Transport;
pub use error::{AbortReason, Error, RejectReason, Result};
pub use wire::{Tag, MAX_WIRE_ELEMENTS, MAX_WIRE_RANK, PROTOCOL_VERSION};

/// The MPI binding this crate was built against.
///
/// A caller must take its communicator, `Process` handles and `initialize` call from
/// this re-export, so the adapter and the caller always share one `mpi` identity.
#[cfg(feature = "upstream")]
pub use mpi;

/// Send one tensor value to `to` on `tag`.
///
/// # Examples
///
/// ```
/// use tenferro_mpi::mpi::traits::*;
/// use tenferro_mpi::send;
/// use tenferro_tensor::{DynRank, Host, TypedTensor};
///
/// let universe = tenferro_mpi::mpi::initialize().unwrap();
/// let world = universe.world();
/// let value = TypedTensor::<i64, DynRank, Host>::from_host_vec_col_major(vec![3], vec![1, 2, 3])?;
/// assert!(matches!(send(world.this_process(), 9, &value), Err(tenferro_mpi::Error::SelfMessage)));
/// if world.size() >= 2 && world.rank() == 0 {
///     send(world.process_at_rank(1), 9, &value)?;
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
#[cfg(feature = "upstream")]
pub fn send<X: Transport>(to: mpi::topology::Process<'_>, tag: Tag, value: &X) -> Result<()> {
    engine::send(&Upstream, to, tag, value)
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
///     let value = TypedTensor::<i64, DynRank, Host>::from_host_vec_col_major(vec![2], vec![7, 8])?;
///     if world.rank() == 0 {
///         tenferro_mpi::send(world.process_at_rank(1), 10, &value)?;
///     } else if world.rank() == 1 {
///         let back: TypedTensor<i64, DynRank, Host> = recv(world.process_at_rank(0), 10)?;
///         assert_eq!(back.as_slice(), &[7, 8]);
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
#[cfg(feature = "upstream")]
pub fn recv<X: Transport>(from: mpi::topology::Process<'_>, tag: Tag) -> Result<X> {
    engine::recv(&Upstream, from, tag)
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
/// let mut out = TypedTensor::<i64, DynRank, Host>::from_host_vec_col_major(vec![2], vec![0, 0])?;
/// if world.size() >= 2 {
///     let value = TypedTensor::<i64, DynRank, Host>::from_host_vec_col_major(vec![2], vec![4, 5])?;
///     if world.rank() == 0 {
///         tenferro_mpi::send(world.process_at_rank(1), 13, &value)?;
///     } else if world.rank() == 1 {
///         recv_into(world.process_at_rank(0), 13, &mut out)?;
///         assert_eq!(out.as_slice(), &[4, 5]);
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
#[cfg(feature = "upstream")]
pub fn recv_into<X: Transport>(
    from: mpi::topology::Process<'_>,
    tag: Tag,
    out: &mut X,
) -> Result<()> {
    engine::recv_into(&Upstream, from, tag, out)
}
