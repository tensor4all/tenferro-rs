//! Translation of a provider failure into tenferro's error vocabulary.
//!
//! Each extracted provider (`tlinalg` on the faer route, `tlinalg-blas` on the LAPACK route) owns
//! its own error enum; the interface tenferro requires lives here, so the mapping does too. Both
//! enums carry the same shapes, and the mapping is one-to-one on kind, role and typed source,
//! because callers downcast the source and classify by kind.

#![cfg(any(feature = "native", feature = "blas"))]

/// Rebuild tenferro's error from one provider's error enum.
///
/// The variants shared by both providers are mapped identically; `Internal` exists only in
/// `tlinalg-blas` and is passed as an extra arm.
macro_rules! map_provider_error {
    ($op:expr, $error:expr, $provider:ident $(, $extra_pat:pat => $extra_arm:expr)?) => {{
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
            $($extra_pat => $extra_arm,)?
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
#[cfg(feature = "native")]
pub(crate) fn map_error(op: tlinalg::Op, error: tlinalg::Error) -> tenferro_tensor::Error {
    map_provider_error!(op, error, tlinalg)
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
