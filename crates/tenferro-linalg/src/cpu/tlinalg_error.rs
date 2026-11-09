//! Translation of a provider failure into tenferro's error vocabulary.
//!
//! Each extracted provider (`tlinalg` on the faer route, `tlinalg-blas` on the LAPACK route) owns
//! its own error enum; the interface tenferro requires lives here, so the mapping does too. Both
//! enums carry the same shapes, and the mapping is one-to-one on kind, role and typed source,
//! because callers downcast the source and classify by kind.

#![cfg(any(feature = "native", feature = "blas"))]

/// Rebuild tenferro's error from one provider's error enum.
///
/// The variants shared by both providers are mapped identically; every provider-specific variant
/// (`Internal` exists only in `tlinalg-blas`, `InsufficientWorkspace` only in `tlinalg`) is passed
/// as an extra arm.
macro_rules! map_provider_error {
    ($op:expr, $error:expr, $provider:ident $(, $extra_pat:pat => $extra_arm:expr)*) => {{
        use $provider::Error as ProviderError;
        let op_str = $op.as_str();
        match $error {
            ProviderError::NonConvergence { .. } => crate::error::into_tensor_error(
                op_str,
                crate::Error::NonConvergence { op: op_str },
            ),
            ProviderError::NonFinite { role, .. } => crate::error::into_tensor_error(
                op_str,
                crate::Error::NonFinite {
                    op: op_str,
                    role: role.as_str(),
                },
            ),
            ProviderError::Singular { .. } => {
                crate::error::into_tensor_error(op_str, crate::Error::Singular { op: op_str })
            }
            ProviderError::InvalidArgument { role, detail, .. } => {
                tenferro_tensor::Error::invalid_argument(op_str, role, detail)
            }
            ProviderError::InvalidWorkspace {
                library,
                routine,
                detail,
                ..
            } => crate::error::invalid_workspace(op_str, library, routine, detail),
            ProviderError::Inconsistent { detail, .. } => {
                tenferro_tensor::Error::Internal(format!("{op_str}: {detail}"))
            }
            $($extra_pat => $extra_arm,)*
            // The provider enums are non-exhaustive so a new variant is not a breaking change for
            // the crate that reports it. The host cannot reproduce a payload it does not know, so
            // it fails loudly instead of guessing a kind and silently misclassifying the failure.
            other => tenferro_tensor::Error::Internal(format!(
                "{op_str}: unmapped linalg provider error {other:?}"
            )),
        }
    }};
}

/// Rebuild tenferro's error from a `tlinalg` (faer route) failure.
///
/// This is the translation tenferro's own linalg entries apply to every `tlinalg` failure. A
/// caller that runs `tlinalg` factorizations itself, for example on the bundle
/// `tenferro_cpu::CpuEngine::with_exec` lends, calls it to report
/// exactly the error tenferro would have reported for the same failure. The types are those of the
/// `tlinalg` build this crate links, so a caller names them through a `tlinalg` dependency of the
/// same semver-compatible version.
///
/// # Contract
///
/// The mapping is total and never panics. It is decided in two steps:
///
/// 1. **Operation name.** Every error it builds names `op.as_str()` from the `op` argument. The
///    `op` field stored inside `error` is not consulted, so when the two disagree the argument
///    wins. The only place the inner `op` survives is the typed source of
///    `InsufficientWorkspace`, which is the provider error itself.
/// 2. **Variant.** One `match` on the `tlinalg::Error` variant. The arms are disjoint, so their
///    order does not change the result; the catch-all arm is last and fires only for a variant no
///    arm names.
///
/// | `tlinalg::Error` | tenferro error | [`ErrorKind`](tenferro_tensor::ErrorKind) | Downcastable source |
/// | --- | --- | --- | --- |
/// | `NonConvergence` | `Extension` (linalg family) | `NumericalFailure` | [`Error::NonConvergence`](crate::Error::NonConvergence) `{ op }` |
/// | `NonFinite { role }` | `Extension` (linalg family) | `NumericalFailure` | [`Error::NonFinite`](crate::Error::NonFinite) `{ op, role: role.as_str() }` |
/// | `Singular` | `Extension` (linalg family) | `NumericalFailure` | [`Error::Singular`](crate::Error::Singular) `{ op }` |
/// | `InvalidArgument { role, detail }` | `Validation` | `Validation(InvalidArgument)` | `ValidationError::InvalidArgument { argument: role, message: detail }` |
/// | `InvalidWorkspace { library, routine, detail }` | `BackendSource` | `BackendFailure` | crate-private; `library`, `routine` and `detail` are in the message |
/// | `Inconsistent { detail }` | `Internal("{op}: {detail}")` | `Internal` | none |
/// | `InsufficientWorkspace` | `RuntimeStateSource` | `RuntimeState` | the `tlinalg::Error` itself, `op`, `needed` and `have` included |
/// | any other variant | `Internal("{op}: unmapped linalg provider error {error:?}")` | `Internal` | none |
///
/// The last row is the case that cannot be represented. `tlinalg::Error` is
/// `#[non_exhaustive]`, so a later `tlinalg` may add a variant this crate does not know. The
/// mapping does not guess a kind for it: it returns the generic `Internal` variant with the
/// provider error's `Debug` text, so the failure stays visible and is never misclassified as a
/// numerical or caller error. Against the `tlinalg` this crate links, every variant has its own
/// row and the last row is unreachable.
///
/// # Examples
///
/// ```rust
/// use std::error::Error as _;
/// use tenferro_tensor::ErrorKind;
///
/// let error = tenferro_linalg::error::map_error(
///     tlinalg::Op::Svd,
///     tlinalg::Error::NonConvergence { op: tlinalg::Op::Svd },
/// );
/// assert_eq!(error.kind(), ErrorKind::NumericalFailure);
/// assert!(matches!(
///     error.source().and_then(|source| source.downcast_ref()),
///     Some(tenferro_linalg::Error::NonConvergence { op: "svd" })
/// ));
/// ```
pub fn map_error(op: tlinalg::Op, error: tlinalg::Error) -> tenferro_tensor::Error {
    map_provider_error!(op, error, tlinalg,
    // The host sizes every call with the matching `*_workspace` query at the
    // same `Exec`, so a short region is a broken host contract rather than a
    // caller error. Keep the provider error as the typed source: it names the
    // bytes the resolved route wanted against the bytes it was given.
    ProviderError::InsufficientWorkspace { .. } => {
        tenferro_tensor::Error::runtime_state_source(op.as_str(), error)
    })
}

/// Rebuild tenferro's error from a `tlinalg-blas` (LAPACK route) failure.
///
/// `Internal` carries the complete message the host reported before the move (an impossible
/// provider pivot), so it is passed through unchanged.
#[cfg(feature = "blas")]
pub(crate) fn map_blas_error(
    op: tlinalg_blas::Op,
    error: tlinalg_blas::Error,
) -> tenferro_tensor::Error {
    map_provider_error!(op, error, tlinalg_blas,
        ProviderError::Internal { detail, .. } => tenferro_tensor::Error::Internal(detail))
}

#[cfg(test)]
#[path = "tlinalg_error/tests.rs"]
mod tests;
