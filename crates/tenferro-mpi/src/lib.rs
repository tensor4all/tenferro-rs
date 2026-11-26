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
//! This is a workspace member with `default = []`: a default build compiles the
//! wire format, the error taxonomy and the [`MpiBackend`] trait but selects no MPI
//! binding, so the root `cargo test --workspace` graph needs neither MPI headers
//! nor libclang.
//!
//! Exactly one backend feature is selected when a caller wants transport. Both
//! build the published [`rsmpi-rt`](https://crates.io/crates/rsmpi-rt) 0.8.2
//! package, whose library target is still `mpi`:
//!
//! - `system` enables rsmpi-rt's bindgen `mpi-sys-backend` (crates.io
//!   [`rsmpi-sys`](https://crates.io/crates/rsmpi-sys) 0.2.4) and therefore needs
//!   system MPI headers and libclang at build time.
//! - `runtime` enables its MPIABI `mpi-rt-sys-backend` (crates.io
//!   [`mpi-rt-sys`](https://crates.io/crates/mpi-rt-sys) 0.1.0), which loads MPI at
//!   run time and needs no headers or libclang.
//!
//! The two backends are mutually exclusive; rsmpi-rt itself rejects enabling both
//! backend features at once. `engine` compiles the protocol with no binding at all.
//!
//! # Examples
//!
//! The transport entry points exist only with a backend, so the example below is
//! compiled in the `system` and `runtime` lanes:
//!
//! ```
//! # #[cfg(any(feature = "system", feature = "runtime"))]
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
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
//! # Ok(())
//! # }
//! # #[cfg(any(feature = "system", feature = "runtime"))]
//! # fn main() { example().unwrap(); }
//! # #[cfg(not(any(feature = "system", feature = "runtime")))]
//! # fn main() {}
//! ```

#![deny(missing_docs)]
// The default (no-backend) build is the wire/error shell: the codec helpers and
// error codes are exercised only by the engine and its backend tests, so they are
// intentionally unused there. Keep the root `-D warnings` lane green without
// weakening the backend builds.
#![cfg_attr(
    not(any(feature = "system", feature = "runtime", feature = "engine")),
    allow(dead_code)
)]

mod backend;
#[cfg(any(feature = "system", feature = "runtime", feature = "engine"))]
pub mod engine;
mod error;
mod wire;

pub use backend::MpiBackend;
#[cfg(feature = "runtime")]
pub use backend::{Rt, RtMessage};
#[cfg(feature = "system")]
pub use backend::System;
#[cfg(any(feature = "system", feature = "runtime", feature = "engine"))]
pub use engine::Transport;
pub use error::{AbortReason, Error, RejectReason, Result};
pub use wire::{Tag, MAX_WIRE_ELEMENTS, MAX_WIRE_RANK, PROTOCOL_VERSION};

/// The MPI binding this crate was built against.
///
/// A caller must take its communicator, `Process` handles and `initialize` call from
/// this re-export, so the adapter and the caller always share one `mpi` identity.
#[cfg(any(feature = "system", feature = "runtime"))]
pub use mpi;

/// The backend selected by the one enabled binding feature.
#[cfg(feature = "system")]
type ActiveBackend = System;
#[cfg(feature = "runtime")]
type ActiveBackend = Rt;

#[cfg(feature = "system")]
const ACTIVE_BACKEND: ActiveBackend = System;
#[cfg(feature = "runtime")]
const ACTIVE_BACKEND: ActiveBackend = Rt;

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
#[cfg(any(feature = "system", feature = "runtime"))]
pub fn send<X: Transport>(to: mpi::topology::Process<'_>, tag: Tag, value: &X) -> Result<()> {
    engine::send(&ACTIVE_BACKEND, to, tag, value)
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
#[cfg(any(feature = "system", feature = "runtime"))]
pub fn recv<X: Transport>(from: mpi::topology::Process<'_>, tag: Tag) -> Result<X> {
    engine::recv(&ACTIVE_BACKEND, from, tag)
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
#[cfg(any(feature = "system", feature = "runtime"))]
pub fn recv_into<X: Transport>(
    from: mpi::topology::Process<'_>,
    tag: Tag,
    out: &mut X,
) -> Result<()> {
    engine::recv_into(&ACTIVE_BACKEND, from, tag, out)
}
