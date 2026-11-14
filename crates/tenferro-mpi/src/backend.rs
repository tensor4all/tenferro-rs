//! The MPI primitives the transport protocol needs, and the two backend bindings.

use crate::wire::Tag;

/// The MPI operations one transport call performs.
///
/// The protocol itself lives in [`crate::engine`] and is written once against this
/// trait, so a second MPI binding only supplies these four operations.
///
/// Implementations borrow the caller's communicator through `Peer`; nothing here
/// initializes, finalizes or duplicates a communicator.
///
/// # Examples
///
/// ```
/// use tenferro_mpi::MpiBackend;
///
/// fn is_self<B: MpiBackend>(peer: B::Peer<'_>) -> bool {
///     B::is_self(peer)
/// }
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

#[cfg(feature = "system")]
mod system {
    use mpi::point_to_point::Message;
    use mpi::topology::Process;
    use mpi::traits::{Destination, Equivalence, Source};

    use super::MpiBackend;
    use crate::wire::Tag;

    /// The `system` backend: rsmpi-rt's bindgen `mpi-sys-backend` (crates.io
    /// rsmpi-sys 0.2.4) against system MPI headers and libclang.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_mpi::System;
    ///
    /// let backend = System;
    /// assert_eq!(format!("{backend:?}"), "System");
    /// ```
    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub struct System;

    impl MpiBackend for System {
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

#[cfg(feature = "runtime")]
mod runtime {
    use mpi::point_to_point::Message;
    use mpi::topology::Process;
    use mpi::traits::{Destination, Equivalence, Source};

    use super::MpiBackend;
    use crate::wire::Tag;

    /// The `runtime` backend: rsmpi-rt's MPIABI `mpi-rt-sys-backend` (crates.io
    /// mpi-rt-sys 0.1.0), which loads MPI at run time.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_mpi::Rt;
    ///
    /// let backend = Rt;
    /// assert_eq!(format!("{backend:?}"), "Rt");
    /// ```
    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub struct Rt;

    impl MpiBackend for Rt {
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

#[cfg(feature = "runtime")]
pub use runtime::Rt;
#[cfg(feature = "system")]
pub use system::System;
