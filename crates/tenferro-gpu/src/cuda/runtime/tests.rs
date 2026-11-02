//! The `TENFERRO_CUDA_ARTIFACT_CACHE` contract.

use super::artifact_cache_dir;

#[test]
fn the_environment_names_the_persistent_artifact_directory() {
    assert_eq!(artifact_cache_dir(None), None);
    assert_eq!(artifact_cache_dir(Some("")), None);
    assert_eq!(artifact_cache_dir(Some("  ")), None);
    // An explicit memory-only request reads the same as unset.
    for disabled in ["none", "off", "memory"] {
        assert_eq!(artifact_cache_dir(Some(disabled)), None, "{disabled}");
    }
    assert_eq!(
        artifact_cache_dir(Some("/var/tmp/t4a-artifacts")),
        Some(std::path::PathBuf::from("/var/tmp/t4a-artifacts"))
    );
    // Surrounding whitespace is not a directory name.
    assert_eq!(
        artifact_cache_dir(Some(" /var/tmp/t4a-artifacts\n")),
        Some(std::path::PathBuf::from("/var/tmp/t4a-artifacts"))
    );
}

// The allocation pool's retention budget (#2381).
use super::{pool_release_threshold_from, MAX_POOL_RELEASE_BYTES, MIN_POOL_RELEASE_BYTES};

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;

#[test]
fn the_budget_grows_with_the_device_and_stays_bounded() {
    // An 80 GiB A100 keeps a working set of several hundred MiB mapped instead of re-mapping it.
    assert_eq!(
        pool_release_threshold_from(None, 80 * GIB),
        MAX_POOL_RELEASE_BYTES
    );
    // A 16 GiB card gets a quarter of it, not the whole device.
    assert_eq!(
        pool_release_threshold_from(None, 16 * GIB),
        (2 * GIB) as usize
    );
    // The floor keeps small devices from releasing on every call.
    assert_eq!(
        pool_release_threshold_from(None, GIB),
        MIN_POOL_RELEASE_BYTES
    );
    // An unknown device size still yields the floor rather than zero.
    assert_eq!(pool_release_threshold_from(None, 0), MIN_POOL_RELEASE_BYTES);
}

#[test]
fn the_environment_overrides_the_budget_in_mib() {
    assert_eq!(
        pool_release_threshold_from(Some(1024), 80 * GIB),
        (1024 * MIB) as usize
    );
    // An override below the floor or above the ceiling is clamped, not honoured literally.
    assert_eq!(
        pool_release_threshold_from(Some(1), 80 * GIB),
        MIN_POOL_RELEASE_BYTES
    );
    assert_eq!(
        pool_release_threshold_from(Some(1_000_000), 80 * GIB),
        MAX_POOL_RELEASE_BYTES
    );
}
