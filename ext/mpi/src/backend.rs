//! The MPI primitives the transport protocol needs, and the `upstream` backend.

use crate::wire::Tag;

/// The MPI operations one transport call performs.
///
/// The protocol itself lives in [`crate::engine`] and is written once against this
/// trait, so a second MPI binding (for example the runtime-loading
/// `tenferro-mpi-rt`) only supplies these four operations.
///
/// Implementations borrow the caller's communicator through `Peer`; nothing here
/// initializes, finalizes or duplicates a communicator.
///
/// # Examples
///
/// ```
/// use tenferro_mpi::mpi::traits::*;
/// use tenferro_mpi::MpiBackend;
///
/// fn is_self<B: MpiBackend>(backend: &B, peer: B::Peer<'_>) -> bool {
///     let _ = backend;
///     B::is_self(peer)
/// }
/// # let _ = is_self::<tenferro_mpi::Upstream>;
/// ```
pub trait MpiBackend {
    /// A destination or source rank handle, borrowed from the caller's communicator.
    type Peer<'a>: Copy;

    /// A message that has been probed and must be received exactly once.
    type Probed;

    /// Whether `peer` is the calling rank.
    fn is_self(peer: Self::Peer<'_>) -> bool;

    /// Blocking send of `buffer` as bytes.
    fn send_bytes(&self, to: Self::Peer<'_>, tag: Tag, buffer: &[u8]);

    /// Blocking probe: consume the next message from `from` on `tag` and report its
    /// byte length.
    ///
    /// Probing before receiving is what keeps a non-conforming frame a typed error
    /// instead of a fatal MPI truncation.
    fn probe(&self, from: Self::Peer<'_>, tag: Tag) -> (Self::Probed, i64);

    /// Receive a probed message, returning the byte count it carried.
    fn receive_probed(&self, probed: Self::Probed, buffer: &mut [u8]) -> i64;
}

#[cfg(feature = "upstream")]
mod upstream {
    use mpi::point_to_point::Message;
    use mpi::topology::Process;
    use mpi::traits::{Destination, Equivalence, Source};

    use super::MpiBackend;
    use crate::wire::Tag;

    /// The crates.io `mpi` (rsmpi) backend, with its system-MPI `mpi-sys` bindings.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_mpi::mpi::traits::*;
    /// use tenferro_mpi::Upstream;
    ///
    /// let backend = Upstream;
    /// assert_eq!(format!("{backend:?}"), "Upstream");
    /// ```
    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub struct Upstream;

    impl MpiBackend for Upstream {
        type Peer<'a> = Process<'a>;
        type Probed = Message;

        fn is_self(peer: Self::Peer<'_>) -> bool {
            peer.is_self()
        }

        fn send_bytes(&self, to: Self::Peer<'_>, tag: Tag, buffer: &[u8]) {
            to.send_with_tag(buffer, tag);
        }

        fn probe(&self, from: Self::Peer<'_>, tag: Tag) -> (Self::Probed, i64) {
            let (message, status) = from.matched_probe_with_tag(tag);
            (message, i64::from(status.count(u8::equivalent_datatype())))
        }

        fn receive_probed(&self, probed: Self::Probed, buffer: &mut [u8]) -> i64 {
            let status = probed.matched_receive_into(buffer);
            i64::from(status.count(u8::equivalent_datatype()))
        }
    }
}

#[cfg(feature = "upstream")]
pub use upstream::Upstream;
