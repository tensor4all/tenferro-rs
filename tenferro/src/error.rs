/// Crate-wide error type for dynamic `tenferro` APIs.
///
/// # Examples
///
/// ```rust
/// use tenferro::{Error, Result};
///
/// fn maybe_fail(flag: bool) -> Result<()> {
///     if flag {
///         Ok(())
///     } else {
///         Err(Error::UnsupportedAdOp { op: "demo" })
///     }
/// }
///
/// assert!(maybe_fail(true).is_ok());
/// assert!(maybe_fail(false).is_err());
/// ```
pub use tenferro_internal_error::Error;

/// Convenience result alias for dynamic `tenferro` APIs.
///
/// # Examples
///
/// ```rust
/// use tenferro::{Error, Result};
///
/// let ok: Result<i32> = Ok(1);
/// let err: Result<i32> = Err(Error::UnsupportedAdOp { op: "sample" });
///
/// assert_eq!(ok.unwrap(), 1);
/// assert!(err.is_err());
/// ```
pub use tenferro_internal_error::Result;
