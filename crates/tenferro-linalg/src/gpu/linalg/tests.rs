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
            select_svd_driver(SvdDriver::Xgesvdp, m, n),
            CusolverSvdRoutine::Xgesvdp
        );
        assert_ne!(
            select_svd_driver(SvdDriver::Auto, m, n),
            CusolverSvdRoutine::Xgesvdp
        );
    }
    // Below the threshold Auto would pick gesvdj; Gesvd must still win.
    assert_eq!(
        select_svd_driver(SvdDriver::Gesvd, 64, 64),
        CusolverSvdRoutine::Gesvd
    );
    // Above the threshold Auto would pick gesvd; Gesvdj must still win.
    assert_eq!(
        select_svd_driver(SvdDriver::Gesvdj, max + 1, 4),
        CusolverSvdRoutine::Gesvdj
    );
    assert_eq!(
        select_svd_driver(SvdDriver::Gesvdj, 4, max + 1),
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
                select_eigh_driver(EighDriver::Auto, batch),
                CusolverEighRoutine::XsyevBatched,
                "Auto batch={batch}"
            );
            assert_eq!(
                select_eigh_driver(EighDriver::Syevd, batch),
                CusolverEighRoutine::XsyevBatched,
                "Syevd batch={batch}"
            );
            assert_eq!(
                select_eigh_driver(EighDriver::Syevj, batch),
                CusolverEighRoutine::SyevjBatched,
                "Syevj batch={batch}"
            );
        }
    }
}

#[test]
fn eigh_single_matrices_keep_the_per_matrix_entry_points() {
    // One matrix has no launch overhead to amortize, so the batched entry
    // points have nothing to win and the per-matrix routines stay.
    assert_eq!(
        select_eigh_driver(EighDriver::Auto, 1),
        CusolverEighRoutine::Syevd
    );
    assert_eq!(
        select_eigh_driver(EighDriver::Syevd, 1),
        CusolverEighRoutine::Syevd
    );
    assert_eq!(
        select_eigh_driver(EighDriver::Syevj, 1),
        CusolverEighRoutine::Syevj
    );
}
