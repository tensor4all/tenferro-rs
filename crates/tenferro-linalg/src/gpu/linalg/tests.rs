use super::{
    select_eigh_driver, select_svd_driver, CusolverEighRoutine, CusolverSvdRoutine,
    CUSOLVER_SYEVJ_BATCHED_MAX_DIM, JAX_COMPATIBLE_GESVDJ_MAX_DIM,
};
use crate::extension::{EighDriver, SvdDriver};

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
fn eigh_auto_driver_keeps_the_pre_driver_routine() {
    // `Auto` must reproduce the behavior that existed before the driver: the
    // divide-and-conquer routine at every size and batch count.
    for (n, batch) in [(1, 1), (8, 1), (8, 1024), (4096, 1), (4096, 8)] {
        assert_eq!(
            select_eigh_driver(EighDriver::Auto, n, batch),
            CusolverEighRoutine::Syevd,
            "Auto policy for n={n} batch={batch}"
        );
        assert_eq!(
            select_eigh_driver(EighDriver::Syevd, n, batch),
            CusolverEighRoutine::Syevd,
            "forced Syevd for n={n} batch={batch}"
        );
    }
}

#[test]
fn eigh_jacobi_driver_takes_the_batched_entry_point_only_where_cusolver_allows_it() {
    let max = CUSOLVER_SYEVJ_BATCHED_MAX_DIM;
    assert_eq!(max, 32);
    // A real batch within the size limit is the only case with a batched
    // cuSOLVER entry point to reach.
    for n in [1, 8, max] {
        assert_eq!(
            select_eigh_driver(EighDriver::Syevj, n, 2),
            CusolverEighRoutine::SyevjBatched,
            "batched Jacobi for n={n}"
        );
    }
    // Single matrices have no batch to amortize, and cuSOLVER rejects the
    // batched entry point above the limit.
    assert_eq!(
        select_eigh_driver(EighDriver::Syevj, 8, 1),
        CusolverEighRoutine::Syevj
    );
    assert_eq!(
        select_eigh_driver(EighDriver::Syevj, max + 1, 256),
        CusolverEighRoutine::Syevj
    );
}
