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
