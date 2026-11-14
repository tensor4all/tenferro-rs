//! Blocking point-to-point MPI transport for tenferro host tensors, loaded at run time.
//!
//! This is the same transport as `tenferro-mpi`, built against
//! [rsmpi-rt](https://github.com/tensor4all/rsmpi-rt) instead of the crates.io `mpi`
//! bindings. The difference is where MPI comes from: rsmpi-rt talks to the MPIABI of
//! [MPIwrapper](https://github.com/eschnett/MPIwrapper), so neither this crate nor its
//! dependencies need MPI headers, a C compiler or libclang at build time, and the MPI
//! implementation is chosen at run time by `MPI_RT_LIB`. That is what lets a Rust
//! library share one communicator with MPI.jl or mpi4py.
//!
//! The protocol, the wire format, the error type and the [`Transport`] set are the
//! ones documented by [`tenferro_mpi`]; only the four MPI operations are supplied
//! here. The adapter still owns no part of the MPI lifecycle: the caller initializes
//! and finalizes MPI and lends the communicator.
//!
//! ```sh
//! # build and install MPIwrapper first (see README.md)
//! export MPI_RT_LIB=$HOME/.local/lib/libmpiwrapper.so
//! scripts/run-mpi.sh rt
//! ```
//!
//! # Examples
//!
//! ```
//! use tenferro_mpi_rt::mpi::traits::*;
//! use tenferro_mpi_rt::{recv, send};
//! use tenferro_tensor::{DynRank, Host, TypedTensor};
//!
//! let universe = tenferro_mpi_rt::mpi::initialize().unwrap();
//! let world = universe.world();
//! let value = TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1.0, 2.0])?;
//! // A blocking self-send would deadlock, so it is refused before any message.
//! assert!(matches!(
//!     send(world.this_process(), 3, &value),
//!     Err(tenferro_mpi_rt::Error::SelfMessage)
//! ));
//! // A blocking send needs a posted receive: one rank sends, the other takes.
//! if world.size() >= 2 {
//!     if world.rank() == 0 {
//!         send(world.process_at_rank(1), 3, &value)?;
//!     } else if world.rank() == 1 {
//!         let back: TypedTensor<f64, DynRank, Host> = recv(world.process_at_rank(0), 3)?;
//!         assert_eq!(back.as_slice(), &[1.0, 2.0]);
//!     }
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![deny(missing_docs)]

use mpi::point_to_point::Message;
use mpi::topology::Process;
use mpi::traits::{Destination, Equivalence, Source};

pub use tenferro_mpi::{engine, AbortReason, Error, MpiBackend, RejectReason, Result, Transport};

/// The rsmpi-rt binding this crate was built against.
///
/// An `mpi` identity is per-crate: a caller must take its communicator, process
/// handles and `initialize` call from this re-export, so the adapter and the caller
/// share one set of MPIABI handles.
pub use mpi;

/// The runtime-loading rsmpi-rt backend.
///
/// # Examples
///
/// ```
/// use tenferro_mpi_rt::Rt;
///
/// let backend = Rt;
/// let copy = backend;
/// assert_eq!(backend, copy);
/// ```
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Rt;

impl MpiBackend for Rt {
    type Peer<'a> = Process<'a>;
    type Probed = Message;

    fn is_self(peer: Self::Peer<'_>) -> bool {
        peer.is_self()
    }

    fn send_bytes(&self, to: Self::Peer<'_>, tag: tenferro_mpi::Tag, buffer: &[u8]) {
        to.send_with_tag(buffer, tag);
    }

    fn probe(&self, from: Self::Peer<'_>, tag: tenferro_mpi::Tag) -> (Self::Probed, i64) {
        let (message, status) = from.matched_probe_with_tag(tag);
        (message, i64::from(status.count(u8::equivalent_datatype())))
    }

    fn receive_probed(&self, probed: Self::Probed, buffer: &mut [u8]) -> i64 {
        let status = probed.matched_receive_into(buffer);
        i64::from(status.count(u8::equivalent_datatype()))
    }
}

/// Send one tensor value to `to` on `tag`.
///
/// # Examples
///
/// ```
/// use tenferro_mpi_rt::mpi::traits::*;
/// use tenferro_mpi_rt::send;
/// use tenferro_tensor::{DynRank, Host, TypedTensor};
///
/// let universe = tenferro_mpi_rt::mpi::initialize().unwrap();
/// let world = universe.world();
/// let value = TypedTensor::<i64, DynRank, Host>::from_host_vec_col_major(vec![2], vec![4, 5])?;
/// assert!(matches!(send(world.this_process(), 1, &value), Err(tenferro_mpi_rt::Error::SelfMessage)));
/// if world.size() >= 2 && world.rank() == 0 {
///     send(world.process_at_rank(1), 1, &value)?;
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
pub fn send<X: Transport>(to: Process<'_>, tag: tenferro_mpi::Tag, value: &X) -> Result<()> {
    engine::send(&Rt, to, tag, value)
}

/// Receive one tensor value from `from` on `tag`.
///
/// # Examples
///
/// ```
/// use tenferro_mpi_rt::mpi::traits::*;
/// use tenferro_mpi_rt::recv;
/// use tenferro_tensor::{DynRank, Host, TypedTensor};
///
/// let universe = tenferro_mpi_rt::mpi::initialize().unwrap();
/// let world = universe.world();
/// if world.size() >= 2 {
///     let value = TypedTensor::<i64, DynRank, Host>::from_host_vec_col_major(vec![1], vec![6])?;
///     if world.rank() == 0 {
///         tenferro_mpi_rt::send(world.process_at_rank(1), 2, &value)?;
///     } else if world.rank() == 1 {
///         let back: TypedTensor<i64, DynRank, Host> = recv(world.process_at_rank(0), 2)?;
///         assert_eq!(back.as_slice(), &[6]);
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
pub fn recv<X: Transport>(from: Process<'_>, tag: tenferro_mpi::Tag) -> Result<X> {
    engine::recv(&Rt, from, tag)
}

/// Receive one tensor value into `out` from `from` on `tag`.
///
/// # Examples
///
/// ```
/// use tenferro_mpi_rt::mpi::traits::*;
/// use tenferro_mpi_rt::recv_into;
/// use tenferro_tensor::{DynRank, Host, TypedTensor};
///
/// let universe = tenferro_mpi_rt::mpi::initialize().unwrap();
/// let world = universe.world();
/// let mut out = TypedTensor::<i64, DynRank, Host>::from_host_vec_col_major(vec![2], vec![0, 0])?;
/// if world.size() >= 2 {
///     let value = TypedTensor::<i64, DynRank, Host>::from_host_vec_col_major(vec![2], vec![8, 9])?;
///     if world.rank() == 0 {
///         tenferro_mpi_rt::send(world.process_at_rank(1), 4, &value)?;
///     } else if world.rank() == 1 {
///         recv_into(world.process_at_rank(0), 4, &mut out)?;
///         assert_eq!(out.as_slice(), &[8, 9]);
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
pub fn recv_into<X: Transport>(
    from: Process<'_>,
    tag: tenferro_mpi::Tag,
    out: &mut X,
) -> Result<()> {
    engine::recv_into(&Rt, from, tag, out)
}
