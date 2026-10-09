//! Typed admission failures for backend session entry.
//!
//! [`BackendSessionHost::with_backend_session`](crate::BackendSessionHost::with_backend_session)
//! returns [`SessionEntryError`] when a backend refuses to open a session, or when
//! entering it fails. Except for one case, every variant is reported *before* the
//! session callback runs, so a caller that sees this error knows the callback was
//! not executed.
//!
//! The exception is restoration, not admission. A CPU session narrows the calling
//! thread's CPU affinity for the callback and restores it afterwards; when that
//! restoration fails, entry reports [`SessionEntryError::Executor`] **after** the
//! callback has run. That failure is therefore not proof that no work happened, and
//! it replaces the callback's own value in the `Err` arm. Affinity restoration is
//! retried by the guard's destructor, so the failure is about the report, not about
//! a process left half-confined. A callback's own result is otherwise returned
//! inside the `Ok` value and is never folded into this type.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_tensor::{ErrorKind, SessionEntryError};
//!
//! let error = SessionEntryError::Reentered { backend: "CpuBackend" };
//! assert_eq!(error.kind(), ErrorKind::RuntimeState);
//! assert_eq!(error.backend(), "CpuBackend");
//! ```

use tenferro_tensor_core::ErrorKind;

use crate::BoxError;

/// Why a backend refused to open an execution session, or why entering it failed.
///
/// The session callback has not run when this error is returned, with one
/// exception: the CPU backend reports [`SessionEntryError::Executor`] when it cannot
/// restore the calling thread's CPU affinity *after* the callback has run, and that
/// error replaces the callback's value. Contention that the backend can wait out
/// (another thread holding an overlapping CPU resource permit) is waited for, not
/// reported; this type covers only states that waiting cannot resolve.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{Error, ErrorKind, SessionEntryError};
///
/// let entry = SessionEntryError::IncompatibleContext {
///     backend: "CpuBackend",
///     message: "operation backend does not match the active execution scope".into(),
/// };
/// let error = Error::from(entry);
/// assert_eq!(error.kind(), ErrorKind::RuntimeState);
/// assert!(std::error::Error::source(&error).is_some());
/// ```
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SessionEntryError {
    /// An execution for this backend is already active on the calling thread
    /// (or its managed worker scope), so opening another session would nest
    /// provider or resource exclusion.
    #[error(
        "{backend}: session entry rejected because an execution is already active on this \
         thread; pass the entered session to nested operations instead of entering the \
         backend again"
    )]
    Reentered {
        /// Backend that rejected the entry.
        backend: &'static str,
    },
    /// A resource that admission cannot wait for is held by another user, such
    /// as a caller-managed CPU domain already executing elsewhere.
    #[error("{backend}: session entry rejected because the execution resource is busy: {message}")]
    Contended {
        /// Backend that rejected the entry.
        backend: &'static str,
        /// Which resource was busy.
        message: String,
    },
    /// The session cannot be opened in the declared or active execution
    /// context, for example an execution scope entered for a different backend.
    #[error("{backend}: session entry rejected: {message}")]
    IncompatibleContext {
        /// Backend that rejected the entry.
        backend: &'static str,
        /// Which context requirement failed.
        message: String,
    },
    /// Admission state was poisoned by an earlier panic and cannot be trusted.
    #[error("{backend}: session entry rejected because {resource} is poisoned")]
    ResourcePoisoned {
        /// Backend that rejected the entry.
        backend: &'static str,
        /// Poisoned admission resource.
        resource: &'static str,
    },
    /// The backend's executor could not be entered.
    #[error("{backend}: executor entry failed: {source}")]
    Executor {
        /// Backend that rejected the entry.
        backend: &'static str,
        /// Typed executor failure.
        #[source]
        source: BoxError,
    },
}

impl SessionEntryError {
    /// Return the backend that rejected the session entry.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::SessionEntryError;
    ///
    /// let error = SessionEntryError::ResourcePoisoned {
    ///     backend: "CpuBackend",
    ///     resource: "the CPU resource arbiter",
    /// };
    /// assert_eq!(error.backend(), "CpuBackend");
    /// ```
    #[must_use]
    pub fn backend(&self) -> &'static str {
        match self {
            Self::Reentered { backend }
            | Self::Contended { backend, .. }
            | Self::IncompatibleContext { backend, .. }
            | Self::ResourcePoisoned { backend, .. }
            | Self::Executor { backend, .. } => backend,
        }
    }

    /// Return the coarse classification of this failure.
    ///
    /// Every session-entry failure is invalid or unavailable execution state.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{ErrorKind, SessionEntryError};
    ///
    /// let error = SessionEntryError::Contended {
    ///     backend: "CpuBackend",
    ///     message: "caller-managed CPU domain".into(),
    /// };
    /// assert_eq!(error.kind(), ErrorKind::RuntimeState);
    /// ```
    #[must_use]
    pub fn kind(&self) -> ErrorKind {
        ErrorKind::RuntimeState
    }
}

#[cfg(test)]
#[path = "session_entry/tests.rs"]
mod tests;
