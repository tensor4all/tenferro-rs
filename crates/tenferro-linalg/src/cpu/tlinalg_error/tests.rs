use std::error::Error as _;

use tenferro_tensor::{Error as TensorError, ErrorKind};

fn payload(error: &TensorError) -> Option<&crate::Error> {
    error.source().and_then(|source| source.downcast_ref())
}

/// One row of the public `map_error` contract: a provider failure and the tenferro error it must
/// become. `check` asserts the variant and payload; `kind` is asserted separately so a drift in
/// either is reported on its own.
#[cfg(feature = "native")]
struct Case {
    name: &'static str,
    op: tlinalg::Op,
    error: tlinalg::Error,
    kind: ErrorKind,
    check: fn(&TensorError) -> bool,
}

/// Every `tlinalg::Error` variant 0.9 defines, one row each, in the order of the documented table.
///
/// `tlinalg::Error` is `#[non_exhaustive]`, so the compiler cannot tell this list that a variant
/// was added; a row per known variant instead fails when an arm is removed, reordered into the
/// catch-all, or rebuilt as a different tenferro variant or kind.
#[cfg(feature = "native")]
fn contract_cases() -> Vec<Case> {
    use tenferro_tensor::{ValidationError, ValidationKind};
    use tlinalg::{Error as TlError, NonFiniteRole, Op};

    vec![
        Case {
            name: "NonConvergence",
            op: Op::Svd,
            error: TlError::NonConvergence { op: Op::Svd },
            kind: ErrorKind::NumericalFailure,
            check: |error| {
                matches!(
                    payload(error),
                    Some(crate::Error::NonConvergence { op: "svd" })
                )
            },
        },
        Case {
            name: "NonFinite",
            op: Op::Eigh,
            error: TlError::NonFinite {
                op: Op::Eigh,
                role: NonFiniteRole::RDiagonal,
            },
            kind: ErrorKind::NumericalFailure,
            check: |error| {
                matches!(
                    payload(error),
                    Some(crate::Error::NonFinite {
                        op: "eigh",
                        role: "R diagonal"
                    })
                )
            },
        },
        Case {
            name: "Singular",
            op: Op::Solve,
            error: TlError::Singular { op: Op::Solve },
            kind: ErrorKind::NumericalFailure,
            check: |error| matches!(payload(error), Some(crate::Error::Singular { op: "solve" })),
        },
        Case {
            name: "InvalidArgument",
            op: Op::LuFactor,
            error: TlError::InvalidArgument {
                op: Op::LuFactor,
                role: "configuration",
                detail: "input describes [2], expected a matrix".to_owned(),
            },
            kind: ErrorKind::Validation(ValidationKind::InvalidArgument),
            check: |error| {
                matches!(
                    error,
                    TensorError::Validation {
                        op: "lu_factor",
                        source: ValidationError::InvalidArgument { argument: "configuration", message },
                    } if message == "input describes [2], expected a matrix"
                )
            },
        },
        Case {
            name: "InvalidWorkspace",
            op: Op::Svd,
            error: TlError::InvalidWorkspace {
                op: Op::Svd,
                library: "faer",
                routine: "svd",
                detail: "query was zero".to_owned(),
            },
            kind: ErrorKind::BackendFailure,
            check: |error| {
                matches!(error, TensorError::BackendSource { op: "svd", .. })
                    && error.to_string().contains("faer routine svd")
                    && error.to_string().contains("query was zero")
            },
        },
        Case {
            name: "Inconsistent",
            op: Op::LuSolvePrepared,
            error: TlError::Inconsistent {
                op: Op::LuSolvePrepared,
                detail: "different batches",
            },
            kind: ErrorKind::Internal,
            check: |error| {
                matches!(
                    error,
                    TensorError::Internal(message) if message == "lu_solve_prepared: different batches"
                )
            },
        },
        // A workspace shorter than the host's own query is a broken host contract, not a caller
        // error, so it reports as a runtime-state failure that keeps the provider refusal
        // downcastable, bytes and operation included.
        Case {
            name: "InsufficientWorkspace",
            op: Op::Qr,
            error: TlError::InsufficientWorkspace {
                op: Op::Qr,
                needed: 128,
                have: 64,
            },
            kind: ErrorKind::RuntimeState,
            check: |error| {
                matches!(error, TensorError::RuntimeStateSource { op: "qr", .. })
                    && matches!(
                        error.source().and_then(|source| source.downcast_ref()),
                        Some(TlError::InsufficientWorkspace {
                            op: Op::Qr,
                            needed: 128,
                            have: 64,
                        })
                    )
            },
        },
    ]
}

#[cfg(feature = "native")]
#[test]
fn every_known_variant_maps_to_its_documented_tenferro_error() {
    for case in contract_cases() {
        let error = super::map_error(case.op, case.error);
        assert_eq!(error.kind(), case.kind, "{}: {error:?}", case.name);
        assert!((case.check)(&error), "{}: {error:?}", case.name);
        // No known variant may fall through to the catch-all arm.
        assert!(
            !error.to_string().contains("unmapped linalg provider error"),
            "{} fell through to the catch-all: {error}",
            case.name
        );
    }
}

#[cfg(feature = "native")]
#[test]
fn public_path_is_the_mapping_tenferro_uses() {
    // The re-export is the same function, not a second copy: both paths agree on every row.
    for (via_crate, via_public) in contract_cases().into_iter().zip(contract_cases()) {
        let internal = super::map_error(via_crate.op, via_crate.error);
        let public = crate::error::map_error(via_public.op, via_public.error);
        assert_eq!(
            internal.to_string(),
            public.to_string(),
            "{}",
            via_crate.name
        );
        assert_eq!(internal.kind(), public.kind(), "{}", via_crate.name);
    }
}

#[cfg(feature = "native")]
#[test]
fn op_argument_wins_over_the_op_stored_in_the_error() {
    use tlinalg::{Error as TlError, NonFiniteRole, Op};

    // Two names could fire for every variant: the `op` argument and the `op` inside the error.
    // The argument names the tenferro error in every row.
    let error = super::map_error(Op::Svd, TlError::NonConvergence { op: Op::Eigh });
    assert!(matches!(
        payload(&error),
        Some(crate::Error::NonConvergence { op: "svd" })
    ));

    let error = super::map_error(
        Op::Qr,
        TlError::NonFinite {
            op: Op::Cholesky,
            role: NonFiniteRole::Input,
        },
    );
    assert!(matches!(
        payload(&error),
        Some(crate::Error::NonFinite {
            op: "qr",
            role: "input"
        })
    ));

    let error = super::map_error(
        Op::LuFactorSolve,
        TlError::Inconsistent {
            op: Op::LuSolvePrepared,
            detail: "different batches",
        },
    );
    assert!(matches!(
        &error,
        TensorError::Internal(message) if message == "lu_factor_solve: different batches"
    ));

    // `InsufficientWorkspace` is the one row where both survive: the tenferro error names the
    // argument, and the typed source is the provider error, which keeps its own `op`.
    let error = super::map_error(
        Op::LuFactorSolve,
        TlError::InsufficientWorkspace {
            op: Op::LuFactor,
            needed: 32,
            have: 16,
        },
    );
    assert!(matches!(
        error,
        TensorError::RuntimeStateSource {
            op: "lu_factor_solve",
            ..
        }
    ));
    assert!(matches!(
        error.source().and_then(|source| source.downcast_ref()),
        Some(TlError::InsufficientWorkspace {
            op: Op::LuFactor,
            ..
        })
    ));
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
