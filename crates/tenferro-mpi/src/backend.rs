//! The MPI primitives the transport protocol needs, and the two backend bindings.

use crate::error::Result;
use crate::wire::Tag;

/// The MPI operations one transport call performs.
///
/// The protocol itself lives in [`crate::engine`] and is written once against this
/// trait, so a second MPI binding only supplies these four operations.
///
/// Every operation returns a `Result`: under `MPI_ERRORS_RETURN` (the default in MPI.jl
/// and mpi4py) a failing MPI call comes back as [`crate::Error::Mpi`] instead of being
/// ignored. Under `MPI_ERRORS_ARE_FATAL` the library aborts before returning.
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
    ///
    /// # Errors
    ///
    /// [`crate::Error::Mpi`] when the MPI call returns an error code (only under
    /// `MPI_ERRORS_RETURN`).
    fn send_bytes(&self, to: Self::Peer<'_>, tag: Tag, buffer: &[u8]) -> Result<()>;

    /// Blocking probe: consume the next message from `from` on `tag` and report its
    /// byte length.
    ///
    /// Probing before receiving is what keeps a non-conforming frame a typed error
    /// instead of a fatal MPI truncation.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Mpi`] when `MPI_Mprobe` or `MPI_Get_count` returns an error code.
    fn probe(&self, from: Self::Peer<'_>, tag: Tag) -> Result<(Self::Probed, i64)>;

    /// Receive a probed message, returning the byte count it carried.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Mpi`] when `MPI_Mrecv` or `MPI_Get_count` returns an error code.
    fn receive_probed(&self, probed: Self::Probed, buffer: &mut [u8]) -> Result<i64>;
}

#[cfg(feature = "system")]
mod system {
    use mpi::point_to_point::Message;
    use mpi::topology::Process;
    use mpi::traits::{Destination, Equivalence, Source};

    use super::MpiBackend;
    use crate::error::Result;
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

        fn send_bytes(&self, to: Self::Peer<'_>, tag: Tag, buffer: &[u8]) -> Result<()> {
            to.send_with_tag(buffer, tag);
            Ok(())
        }

        fn probe(&self, from: Self::Peer<'_>, tag: Tag) -> Result<(Self::Probed, i64)> {
            let (message, status) = from.matched_probe_with_tag(tag);
            Ok((message, i64::from(status.count(u8::equivalent_datatype()))))
        }

        fn receive_probed(&self, probed: Self::Probed, buffer: &mut [u8]) -> Result<i64> {
            let status = probed.matched_receive_into(buffer);
            Ok(i64::from(status.count(u8::equivalent_datatype())))
        }
    }
}

#[cfg(feature = "runtime")]
mod runtime {
    use std::os::raw::{c_char, c_int};

    use mpi::ffi;
    use mpi::raw::AsRaw;
    use mpi::topology::{AsCommunicator, Process};
    use mpi::traits::{Destination, Source};

    use super::MpiBackend;
    use crate::error::{Error, Result};
    use crate::wire::Tag;

    /// The `runtime` backend: rsmpi-rt's `mpi-native-sys-backend`, which finds the host's
    /// MPI library at run time and passes that library's own handles.
    ///
    /// It calls the MPI functions directly and checks every return code. rsmpi-rt's own
    /// `send_with_tag`/`matched_probe_with_tag` ignore return codes, which under
    /// `MPI_ERRORS_RETURN` would hand back an uninitialized message handle.
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

    /// A message matched by `MPI_Mprobe`, to be received exactly once.
    #[derive(Debug)]
    pub struct RtMessage(ffi::MPI_Message);

    /// `MPI_Error_string`, called at the failure site while MPI is known to be initialized.
    fn error_string(code: c_int) -> String {
        let mut buffer = [0 as c_char; 1024];
        let mut length: c_int = 0;
        // SAFETY: the buffer is larger than every family's MPI_MAX_ERROR_STRING.
        let rc = unsafe { ffi::MPI_Error_string(code, buffer.as_mut_ptr(), &mut length) };
        if rc != 0 {
            return String::from("(no error string)");
        }
        let length = usize::try_from(length).unwrap_or(0).min(buffer.len());
        let bytes: Vec<u8> = buffer[..length].iter().map(|&c| c as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// `MPI_SUCCESS` is 0 in every family.
    fn check(code: c_int, call: &'static str) -> Result<()> {
        if code == 0 {
            return Ok(());
        }
        Err(Error::Mpi {
            code,
            message: format!("{call} failed: {}", error_string(code)),
        })
    }

    fn count_of(len: usize) -> Result<c_int> {
        c_int::try_from(len).map_err(|_| Error::Mpi {
            code: -1,
            message: format!("a frame of {len} bytes exceeds one MPI message"),
        })
    }

    impl MpiBackend for Rt {
        type Peer<'a> = Process<'a>;
        type Probed = RtMessage;

        fn is_self(peer: Self::Peer<'_>) -> bool {
            peer.is_self()
        }

        fn send_bytes(&self, to: Self::Peer<'_>, tag: Tag, buffer: &[u8]) -> Result<()> {
            let count = count_of(buffer.len())?;
            // SAFETY: `buffer` is valid for `count` bytes; the communicator is borrowed.
            let rc = unsafe {
                ffi::MPI_Send(
                    buffer.as_ptr().cast(),
                    count,
                    ffi::RSMPI_UINT8_T_fn(),
                    to.destination_rank(),
                    tag,
                    to.as_communicator().as_raw(),
                )
            };
            check(rc, "MPI_Send")
        }

        fn probe(&self, from: Self::Peer<'_>, tag: Tag) -> Result<(Self::Probed, i64)> {
            let mut message = ffi::RSMPI_MESSAGE_NULL_fn();
            let mut status = std::mem::MaybeUninit::<ffi::MPI_Status>::zeroed();
            // SAFETY: both out-pointers are valid; the status is zeroed storage.
            let rc = unsafe {
                ffi::MPI_Mprobe(
                    from.source_rank(),
                    tag,
                    from.as_communicator().as_raw(),
                    &mut message,
                    status.as_mut_ptr(),
                )
            };
            check(rc, "MPI_Mprobe")?;
            let mut count: c_int = 0;
            // SAFETY: the status was filled by the successful probe.
            let rc =
                unsafe { ffi::MPI_Get_count(status.as_ptr(), ffi::RSMPI_UINT8_T_fn(), &mut count) };
            check(rc, "MPI_Get_count")?;
            Ok((RtMessage(message), i64::from(count)))
        }

        fn receive_probed(&self, probed: Self::Probed, buffer: &mut [u8]) -> Result<i64> {
            let capacity = count_of(buffer.len())?;
            let mut message = probed.0;
            let mut status = std::mem::MaybeUninit::<ffi::MPI_Status>::zeroed();
            // SAFETY: `buffer` is valid for `capacity` bytes; the message was matched by
            // `probe` and is received exactly once (`probed` is consumed).
            let rc = unsafe {
                ffi::MPI_Mrecv(
                    buffer.as_mut_ptr().cast(),
                    capacity,
                    ffi::RSMPI_UINT8_T_fn(),
                    &mut message,
                    status.as_mut_ptr(),
                )
            };
            check(rc, "MPI_Mrecv")?;
            let mut count: c_int = 0;
            // SAFETY: the status was filled by the successful receive.
            let rc =
                unsafe { ffi::MPI_Get_count(status.as_ptr(), ffi::RSMPI_UINT8_T_fn(), &mut count) };
            check(rc, "MPI_Get_count")?;
            Ok(i64::from(count))
        }
    }
}

#[cfg(feature = "runtime")]
pub use runtime::{Rt, RtMessage};
#[cfg(feature = "system")]
pub use system::System;
