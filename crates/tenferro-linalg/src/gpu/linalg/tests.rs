use super::{
    select_eigh_driver, select_svd_driver, CusolverEighRoutine, CusolverSvdRoutine,
    JAX_COMPATIBLE_GESVDJ_MAX_DIM,
};
use crate::extension::{EighBatchPolicy, EighDriver, SvdDriver};
use tenferro_tensor::ErrorKind;

/// The capability the batched divide-and-conquer eigh needs on a real batch.
const BATCHED_EIGH: (i32, bool) = (12_000, true);
/// A cuSOLVER older than 11.7.1: both symbols load but the version gate fails.
const OLD_CUSOLVER: (i32, bool) = (11_700, true);
/// A newer cuSOLVER that does not export the two batched symbols.
const MISSING_SYMBOLS: (i32, bool) = (12_000, false);

mod xgesvdp;

#[test]
fn auto_driver_keeps_jax_compatible_threshold() {
    let max = JAX_COMPATIBLE_GESVDJ_MAX_DIM;
    assert_eq!(max, 1024);
    for (m, n, expected) in [
        (1, 1, CusolverSvdRoutine::Gesvdj),
        (max, max, CusolverSvdRoutine::Gesvdj),
        (max + 1, 4, CusolverSvdRoutine::Gesvd),
        (4, max + 1, CusolverSvdRoutine::Gesvd),
        (max + 1, max + 1, CusolverSvdRoutine::Gesvd),
    ] {
        assert_eq!(
            select_svd_driver(SvdDriver::Auto, m, n),
            expected,
            "Auto policy for {m}x{n}"
        );
    }
}

#[test]
fn explicit_driver_overrides_dimension_policy() {
    let max = JAX_COMPATIBLE_GESVDJ_MAX_DIM;
    for (m, n) in [(64, 64), (max, max), (max + 1, 4), (4, max + 1)] {
        assert_eq!(
            select_svd_driver(SvdDriver::Polar, m, n),
            CusolverSvdRoutine::Xgesvdp
        );
        assert_ne!(
            select_svd_driver(SvdDriver::Auto, m, n),
            CusolverSvdRoutine::Xgesvdp
        );
    }
    // Below the threshold Auto would pick gesvdj; Gesvd must still win.
    assert_eq!(
        select_svd_driver(SvdDriver::QrBased, 64, 64),
        CusolverSvdRoutine::Gesvd
    );
    // Above the threshold Auto would pick gesvd; Gesvdj must still win.
    assert_eq!(
        select_svd_driver(SvdDriver::Jacobi, max + 1, 4),
        CusolverSvdRoutine::Gesvdj
    );
    assert_eq!(
        select_svd_driver(SvdDriver::Jacobi, 4, max + 1),
        CusolverSvdRoutine::Gesvdj
    );
}

#[test]
fn eigh_batches_take_a_batched_routine_at_every_order() {
    // There is no size threshold: a real batch always takes a batched
    // cuSOLVER routine, and the driver alone decides which one. #1852
    // measured both batched routines beating their per-matrix loops at every
    // order from 8 to 512, so no order falls back.
    for n in [1, 8, 32, 33, 64, 128, 512, 4096] {
        let _ = n; // the policy is order-independent; the range documents that.
        for batch in [2, 1024] {
            for policy in [
                EighBatchPolicy::AllowPerMatrixFallback,
                EighBatchPolicy::RequireBatched,
            ] {
                assert_eq!(
                    select(EighDriver::Auto, batch, policy, BATCHED_EIGH),
                    Ok(CusolverEighRoutine::XsyevBatched),
                    "Auto batch={batch} {policy:?}"
                );
                assert_eq!(
                    select(EighDriver::DivideAndConquer, batch, policy, BATCHED_EIGH),
                    Ok(CusolverEighRoutine::XsyevBatched),
                    "Syevd batch={batch} {policy:?}"
                );
                assert_eq!(
                    select(EighDriver::Jacobi, batch, policy, BATCHED_EIGH),
                    Ok(CusolverEighRoutine::SyevjBatched),
                    "Syevj batch={batch} {policy:?}"
                );
            }
        }
    }
}

#[test]
fn eigh_batches_fall_back_to_the_syevd_loop_without_the_batched_entry_point() {
    // A cuSOLVER older than 11.7.1 exports no `XsyevBatched`, so the batch
    // takes the per-matrix `syevd` loop instead of failing (#2413).
    for batch in [2, 1024] {
        for capability in [OLD_CUSOLVER, MISSING_SYMBOLS] {
            assert_eq!(
                select(
                    EighDriver::Auto,
                    batch,
                    EighBatchPolicy::AllowPerMatrixFallback,
                    capability,
                ),
                Ok(CusolverEighRoutine::Syevd),
                "Auto batch={batch} {capability:?}"
            );
            assert_eq!(
                select(
                    EighDriver::DivideAndConquer,
                    batch,
                    EighBatchPolicy::AllowPerMatrixFallback,
                    capability,
                ),
                Ok(CusolverEighRoutine::Syevd),
                "Syevd batch={batch} {capability:?}"
            );
        }
        // The Jacobi batched entry point predates the version boundary, so its
        // capability does not depend on `XsyevBatched`.
        assert_eq!(
            select(
                EighDriver::Jacobi,
                batch,
                EighBatchPolicy::AllowPerMatrixFallback,
                MISSING_SYMBOLS,
            ),
            Ok(CusolverEighRoutine::SyevjBatched),
            "Syevj batch={batch}"
        );
    }
    assert_eq!(
        select(
            EighDriver::Auto,
            1,
            EighBatchPolicy::AllowPerMatrixFallback,
            MISSING_SYMBOLS,
        ),
        Ok(CusolverEighRoutine::Syevd)
    );
}

#[test]
fn eigh_route_is_recorded_in_the_solver_diagnostics() {
    assert_eq!(
        CusolverEighRoutine::XsyevBatched.call_name(),
        "cusolverDnXsyevBatched"
    );
    assert_eq!(CusolverEighRoutine::Syevd.call_name(), "cusolverDn*syevd");
}

#[test]
fn eigh_single_matrices_keep_the_per_matrix_entry_points() {
    // One matrix has no launch overhead to amortize, so the batched entry
    // points have nothing to win and the per-matrix routines stay.
    for policy in [
        EighBatchPolicy::AllowPerMatrixFallback,
        EighBatchPolicy::RequireBatched,
    ] {
        assert_eq!(
            select(EighDriver::Auto, 1, policy, BATCHED_EIGH),
            Ok(CusolverEighRoutine::Syevd),
            "{policy:?}"
        );
        assert_eq!(
            select(EighDriver::DivideAndConquer, 1, policy, BATCHED_EIGH),
            Ok(CusolverEighRoutine::Syevd),
            "{policy:?}"
        );
        assert_eq!(
            select(EighDriver::Jacobi, 1, policy, BATCHED_EIGH),
            Ok(CusolverEighRoutine::Syevj),
            "{policy:?}"
        );
    }
    // A batch of at most one matrix is one call, so `RequireBatched` has nothing
    // to refuse even on a library without the batched entry point.
    assert_eq!(
        select(
            EighDriver::Auto,
            0,
            EighBatchPolicy::RequireBatched,
            MISSING_SYMBOLS,
        ),
        Ok(CusolverEighRoutine::Syevd)
    );
    assert_eq!(
        select(
            EighDriver::Auto,
            1,
            EighBatchPolicy::RequireBatched,
            MISSING_SYMBOLS,
        ),
        Ok(CusolverEighRoutine::Syevd)
    );
}

/// Call the selector and normalise the two failure arms to their kinds, so the
/// table-style assertions above stay readable.
fn select(
    driver: EighDriver,
    batch_total: usize,
    policy: EighBatchPolicy,
    (version, symbols_resolved): (i32, bool),
) -> Result<CusolverEighRoutine, ErrorKind> {
    select_eigh_driver(driver, batch_total, policy, version, symbols_resolved)
        .map_err(|error| error.kind())
}

#[test]
fn require_batched_refuses_the_per_matrix_loop_and_names_the_capability() {
    // The reported case: a batch of 17 with a cuSOLVER below the 11.7.1 floor.
    for driver in [EighDriver::Auto, EighDriver::DivideAndConquer] {
        let error = select_eigh_driver(driver, 17, EighBatchPolicy::RequireBatched, 11_700, true)
            .expect_err("a batch of 17 without XsyevBatched must not be launched per matrix");
        assert_eq!(error.kind(), ErrorKind::Unsupported, "{driver:?}");
        let message = error.to_string();
        for needle in [
            "EighBatchPolicy::RequireBatched",
            "one launch",
            "cusolverDnXsyevBatched",
            "below 11.7.1",
            "cusolverDn<t>syevd launch per matrix",
            "EighBatchPolicy::AllowPerMatrixFallback",
        ] {
            assert!(
                message.contains(needle),
                "the refusal should name {needle}: {message}"
            );
        }
    }

    // The same version with the two symbols missing: the message names those,
    // not the version.
    let error = select_eigh_driver(
        EighDriver::DivideAndConquer,
        17,
        EighBatchPolicy::RequireBatched,
        12_000,
        false,
    )
    .expect_err("unresolved batched eigh symbols must not be launched per matrix");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let message = error.to_string();
    assert!(
        message.contains("unresolved"),
        "the refusal should name the unresolved symbols: {message}"
    );
    assert!(
        !message.contains("below 11.7.1"),
        "a new enough library must not be reported as too old: {message}"
    );

    // An old library that also lacks the symbols names both facts.
    let error = select_eigh_driver(
        EighDriver::Auto,
        17,
        EighBatchPolicy::RequireBatched,
        0,
        false,
    )
    .expect_err("the refusal covers both missing facts");
    let message = error.to_string();
    assert!(message.contains("unresolved"), "{message}");
    assert!(message.contains("below 11.7.1"), "{message}");
}

#[test]
fn allow_per_matrix_fallback_keeps_todays_routing_under_the_same_capabilities() {
    // The default policy is exactly the pre-#2616 behaviour: the refusal never
    // fires for it, whatever the capability.
    for (version, symbols_resolved) in [BATCHED_EIGH, OLD_CUSOLVER, MISSING_SYMBOLS, (0, false)] {
        for batch in [0, 1, 2, 17, 1024] {
            for driver in [
                EighDriver::Auto,
                EighDriver::DivideAndConquer,
                EighDriver::Jacobi,
            ] {
                assert!(
                    select_eigh_driver(
                        driver,
                        batch,
                        EighBatchPolicy::AllowPerMatrixFallback,
                        version,
                        symbols_resolved,
                    )
                    .is_ok(),
                    "the default policy never refuses {driver:?} batch={batch} \
                     version={version} symbols={symbols_resolved}"
                );
            }
        }
    }
}

#[test]
fn require_batched_is_satisfied_by_the_jacobi_batch_route() {
    // `syevjBatched` is a required symbol and predates the `XsyevBatched`
    // version boundary, so a Jacobi batch always has its single launch.
    for capability in [BATCHED_EIGH, OLD_CUSOLVER, MISSING_SYMBOLS] {
        assert_eq!(
            select(
                EighDriver::Jacobi,
                17,
                EighBatchPolicy::RequireBatched,
                capability,
            ),
            Ok(CusolverEighRoutine::SyevjBatched),
            "{capability:?}"
        );
    }
}
