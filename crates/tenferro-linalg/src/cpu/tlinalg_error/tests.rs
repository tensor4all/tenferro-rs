use std::error::Error as _;

use tenferro_tensor::{Error as TensorError, ErrorKind};

fn payload(error: &TensorError) -> Option<&crate::Error> {
    error.source().and_then(|source| source.downcast_ref())
}

#[cfg(feature = "native")]
#[test]
fn every_variant_rebuilds_tenferros_kind_role_and_source() {
    use super::map_error;
    use tlinalg::{Error as TlError, NonFiniteRole, Op};

    // The linalg-owned kinds carry the same downcastable payload a caller classifies on.
    let error = map_error(Op::Svd, TlError::NonConvergence { op: Op::Svd });
    assert_eq!(error.kind(), ErrorKind::NumericalFailure);
    assert!(matches!(
        payload(&error),
        Some(crate::Error::NonConvergence { op: "svd" })
    ));

    let error = map_error(
        Op::Eigh,
        TlError::NonFinite {
            op: Op::Eigh,
            role: NonFiniteRole::RDiagonal,
        },
    );
    assert_eq!(error.kind(), ErrorKind::NumericalFailure);
    assert!(matches!(
        payload(&error),
        Some(crate::Error::NonFinite {
            op: "eigh",
            role: "R diagonal"
        })
    ));

    let error = map_error(Op::Solve, TlError::Singular { op: Op::Solve });
    assert_eq!(error.kind(), ErrorKind::NumericalFailure);
    assert!(matches!(
        payload(&error),
        Some(crate::Error::Singular { op: "solve" })
    ));

    // The provider shapes keep their role and their text, because callers match on both.
    let error = map_error(
        Op::LuFactor,
        TlError::InvalidArgument {
            op: Op::LuFactor,
            role: "configuration",
            detail: "input describes [2], expected a matrix".to_owned(),
        },
    );
    let rendered = error.to_string();
    assert!(rendered.contains("configuration"), "{rendered}");
    assert!(rendered.contains("expected a matrix"), "{rendered}");

    let error = map_error(
        Op::Svd,
        TlError::InvalidWorkspace {
            op: Op::Svd,
            library: "faer",
            routine: "svd",
            detail: "query was zero".to_owned(),
        },
    );
    assert_eq!(error.kind(), ErrorKind::BackendFailure);
    assert!(error.to_string().contains("query was zero"));

    let error = map_error(
        Op::LuSolvePrepared,
        TlError::Inconsistent {
            op: Op::LuSolvePrepared,
            detail: "different batches",
        },
    );
    assert!(matches!(error, TensorError::Internal(_)));
    assert!(error.to_string().contains("different batches"));
}

#[cfg(feature = "blas")]
#[test]
fn every_blas_variant_rebuilds_tenferros_kind_role_and_source() {
    use super::map_blas_error;
    use tlinalg_blas::{Error as BlasError, NonFiniteRole, Op};

    let error = map_blas_error(Op::Cholesky, BlasError::NonConvergence { op: Op::Cholesky });
    assert_eq!(error.kind(), ErrorKind::NumericalFailure);
    assert!(matches!(
        payload(&error),
        Some(crate::Error::NonConvergence { op: "cholesky" })
    ));

    let error = map_blas_error(
        Op::RankRevealingQr,
        BlasError::NonFinite {
            op: Op::RankRevealingQr,
            role: NonFiniteRole::Input,
        },
    );
    assert!(matches!(
        payload(&error),
        Some(crate::Error::NonFinite {
            op: "rank_revealing_qr",
            role: "input"
        })
    ));

    let error = map_blas_error(Op::Solve, BlasError::Singular { op: Op::Solve });
    assert!(matches!(
        payload(&error),
        Some(crate::Error::Singular { op: "solve" })
    ));

    let error = map_blas_error(
        Op::LuFactor,
        BlasError::InvalidArgument {
            op: Op::LuFactor,
            role: "lapack_argument",
            detail: "LAPACK getrf argument 3 had an illegal value".to_owned(),
        },
    );
    let rendered = error.to_string();
    assert!(rendered.contains("lapack_argument"), "{rendered}");
    assert!(rendered.contains("getrf argument 3"), "{rendered}");

    let error = map_blas_error(
        Op::Svd,
        BlasError::InvalidWorkspace {
            op: Op::Svd,
            library: "LAPACK",
            routine: "dgesdd",
            detail: "query was zero".to_owned(),
        },
    );
    assert_eq!(error.kind(), ErrorKind::BackendFailure);
    assert!(error.to_string().contains("dgesdd"));

    let error = map_blas_error(
        Op::LuSolvePrepared,
        BlasError::Inconsistent {
            op: Op::LuSolvePrepared,
            detail: "different batches",
        },
    );
    assert!(matches!(error, TensorError::Internal(_)));

    // `Internal` is the provider's complete message, passed through unchanged.
    let error = map_blas_error(
        Op::Lu,
        BlasError::Internal {
            op: Op::Lu,
            detail: "LAPACK getrf returned an invalid pivot index".to_owned(),
        },
    );
    assert!(matches!(
        &error,
        TensorError::Internal(message) if message == "LAPACK getrf returned an invalid pivot index"
    ));
}
