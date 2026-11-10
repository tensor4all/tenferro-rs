//! The `TENFERRO_CUDA_ARTIFACT_CACHE` contract.

use std::ffi::{OsStr, OsString};
use std::path::PathBuf;

use super::{artifact_cache_dir, cache_root, usable_artifact_cache_dir};

fn no_variables(_: &str) -> Option<OsString> {
    None
}

/// The environment a test supplies, so no test reads or writes the real one.
fn environment(
    pairs: &'static [(&'static str, &'static str)],
) -> impl Fn(&str) -> Option<OsString> + Copy {
    move |asked| {
        pairs
            .iter()
            .find(|(name, _)| *name == asked)
            .map(|(_, value)| OsString::from(*value))
    }
}

#[test]
fn the_environment_names_the_persistent_artifact_directory() {
    let home = environment(&[("HOME", "/home/tester")]);
    // An explicit memory-only request, and an empty one, select the in-memory cache.
    for disabled in ["", "  ", "none", "off", "memory"] {
        assert_eq!(
            artifact_cache_dir(Some(OsStr::new(disabled)), home),
            None,
            "{disabled}"
        );
    }
    // Any other value is the directory, and it wins over the per-user default.
    assert_eq!(
        artifact_cache_dir(Some(OsStr::new("/var/tmp/t4a-artifacts")), home),
        Some(PathBuf::from("/var/tmp/t4a-artifacts"))
    );
    // Surrounding whitespace is not a directory name.
    assert_eq!(
        artifact_cache_dir(Some(OsStr::new(" /var/tmp/t4a-artifacts\n")), home),
        Some(PathBuf::from("/var/tmp/t4a-artifacts"))
    );
}

#[test]
#[cfg(unix)]
fn a_non_utf8_override_names_a_directory_rather_than_the_default() {
    use std::os::unix::ffi::OsStrExt;
    // Paths need not be UTF-8 on Unix, so such a value is not one of the tokens: the caller asked
    // for that directory, not for the per-user default.
    let value = OsStr::from_bytes(b"/var/tmp/\xff-artifacts");
    assert_eq!(
        artifact_cache_dir(Some(value), environment(&[("HOME", "/home/tester")])),
        Some(PathBuf::from(value))
    );
}

#[test]
#[cfg(not(windows))]
fn unset_selects_the_per_user_cache_directory() {
    let both = environment(&[("XDG_CACHE_HOME", "/xdg/cache"), ("HOME", "/home/tester")]);
    assert_eq!(
        artifact_cache_dir(None, both),
        Some(PathBuf::from("/xdg/cache/tenferro/t4a-cuda-artifacts"))
    );
    assert_eq!(
        artifact_cache_dir(None, environment(&[("HOME", "/home/tester")])),
        Some(PathBuf::from(
            "/home/tester/.cache/tenferro/t4a-cuda-artifacts"
        ))
    );
    // A relative XDG_CACHE_HOME is not a cache root: it would make the location depend on the
    // working directory.
    let relative = environment(&[("XDG_CACHE_HOME", "cache"), ("HOME", "/home/tester")]);
    assert_eq!(
        cache_root(relative),
        Some(PathBuf::from("/home/tester/.cache"))
    );
    // Neither is a relative fallback root.
    let relative_home = environment(&[("XDG_CACHE_HOME", ""), ("HOME", "home/tester")]);
    assert_eq!(artifact_cache_dir(None, relative_home), None);
    // No per-user root, as on a host that reads only $HOME or in a container without one.
    assert_eq!(artifact_cache_dir(None, no_variables), None);
    let empty = environment(&[("XDG_CACHE_HOME", ""), ("HOME", "")]);
    assert_eq!(artifact_cache_dir(None, empty), None);
}

#[test]
fn a_directory_that_cannot_be_created_falls_back_to_memory() {
    let base = std::env::temp_dir().join(format!("t4a-artifact-cache-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).expect("scratch directory");
    // A regular file where a parent directory has to be fails on every platform, root included.
    let blocker = base.join("blocker");
    std::fs::write(&blocker, b"").expect("blocking file");
    assert_eq!(usable_artifact_cache_dir(blocker.join("deeper")), None);

    let created = base.join("made");
    assert_eq!(
        usable_artifact_cache_dir(created.clone()),
        Some(created.clone())
    );
    assert!(created.is_dir(), "the directory exists after the call");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&created)
            .expect("metadata")
            .permissions()
            .mode();
        assert_eq!(
            mode & 0o777,
            0o700,
            "owner-only, never shared between users"
        );
    }
    let _ = std::fs::remove_dir_all(&base);
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

// The persistent artifact directory's bound (#2475).
use super::{artifact_cache_max_bytes, write_best_effort_warning};

#[test]
fn the_persistent_artifact_directory_is_bounded_by_default() {
    // Unset keeps the runtime's finite default; persistence is never unbounded.
    assert_eq!(
        artifact_cache_max_bytes(None),
        t4a_cuda_runtime::DEFAULT_ARTIFACT_DIR_BYTES
    );
    assert_eq!(artifact_cache_max_bytes(None), 256 * MIB);
    // A value that is not a size is not "no limit": it falls back to the default too.
    for unusable in ["", " ", "unlimited", "-1", "1.5", "none"] {
        assert_eq!(
            artifact_cache_max_bytes(Some(OsStr::new(unusable))),
            t4a_cuda_runtime::DEFAULT_ARTIFACT_DIR_BYTES,
            "{unusable:?}"
        );
    }
}

#[test]
fn the_environment_sets_the_artifact_directory_bound_in_mib() {
    assert_eq!(artifact_cache_max_bytes(Some(OsStr::new("1024"))), GIB);
    assert_eq!(
        artifact_cache_max_bytes(Some(OsStr::new(" 64\n"))),
        64 * MIB
    );
    // Zero keeps reading what is there and publishes nothing new.
    assert_eq!(artifact_cache_max_bytes(Some(OsStr::new("0"))), 0);
    // An absurd value saturates rather than wrapping to a small bound.
    assert_eq!(
        artifact_cache_max_bytes(Some(OsStr::new("18446744073709551615"))),
        u64::MAX
    );
}

#[test]
fn a_warning_to_a_closed_stderr_is_dropped_not_a_panic() {
    /// Stands in for a closed stderr: every write fails.
    struct Closed;
    impl std::io::Write for Closed {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::from(std::io::ErrorKind::BrokenPipe))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Err(std::io::Error::from(std::io::ErrorKind::BrokenPipe))
        }
    }
    write_best_effort_warning(&mut Closed, format_args!("tenferro-gpu: {}", "dropped"));

    let mut captured = Vec::new();
    write_best_effort_warning(&mut captured, format_args!("tenferro-gpu: {}", "kept"));
    assert_eq!(captured, b"tenferro-gpu: kept\n");
}
