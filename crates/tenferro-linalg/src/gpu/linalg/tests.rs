use super::{
    select_eigh_driver, select_svd_driver, CusolverEighRoutine, CusolverSvdRoutine,
    JAX_COMPATIBLE_GESVDJ_MAX_DIM,
};
use crate::extension::{EighDriver, SvdDriver};

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
            assert_eq!(
                select_eigh_driver(EighDriver::Auto, batch, true),
                CusolverEighRoutine::XsyevBatched,
                "Auto batch={batch}"
            );
            assert_eq!(
                select_eigh_driver(EighDriver::DivideAndConquer, batch, true),
                CusolverEighRoutine::XsyevBatched,
                "Syevd batch={batch}"
            );
            assert_eq!(
                select_eigh_driver(EighDriver::Jacobi, batch, true),
                CusolverEighRoutine::SyevjBatched,
                "Syevj batch={batch}"
            );
        }
    }
}

#[test]
fn eigh_batches_fall_back_to_the_syevd_loop_without_the_batched_entry_point() {
    // A cuSOLVER older than 11.7.1 exports no `XsyevBatched`, so the batch
    // takes the per-matrix `syevd` loop instead of failing (#2413).
    for batch in [2, 1024] {
        assert_eq!(
            select_eigh_driver(EighDriver::Auto, batch, false),
            CusolverEighRoutine::Syevd,
            "Auto batch={batch}"
        );
        assert_eq!(
            select_eigh_driver(EighDriver::DivideAndConquer, batch, false),
            CusolverEighRoutine::Syevd,
            "Syevd batch={batch}"
        );
        // The Jacobi batched entry point predates the version boundary, so its
        // capability does not depend on `XsyevBatched`.
        assert_eq!(
            select_eigh_driver(EighDriver::Jacobi, batch, false),
            CusolverEighRoutine::SyevjBatched,
            "Syevj batch={batch}"
        );
    }
    assert_eq!(
        select_eigh_driver(EighDriver::Auto, 1, false),
        CusolverEighRoutine::Syevd
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
    assert_eq!(
        select_eigh_driver(EighDriver::Auto, 1, true),
        CusolverEighRoutine::Syevd
    );
    assert_eq!(
        select_eigh_driver(EighDriver::DivideAndConquer, 1, true),
        CusolverEighRoutine::Syevd
    );
    assert_eq!(
        select_eigh_driver(EighDriver::Jacobi, 1, true),
        CusolverEighRoutine::Syevj
    );
}
