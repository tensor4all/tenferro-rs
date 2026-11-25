#![cfg(all(feature = "fftw", target_os = "linux"))]
//! An FFTW installation without its threading library exports no
//! `fftw_make_planner_thread_safe`, so FFTW's own lock is never installed and
//! tenferro's private planner lock alone serializes tenferro's planning. This
//! binary copies only the core FFTW libraries into a directory of their own,
//! loads a single-threaded provider from it, and plans from several threads at
//! once. The copies are separate files, so the process loads a separate FFTW
//! instance in which nothing has installed FFTW's lock.

#[path = "fftw_support/mod.rs"]
mod support;

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use tenferro_fft::{Fftw, FftwError, FftwPlanning};

/// The directory of the FFTW installation under test.
fn installation() -> Option<PathBuf> {
    if let Some(directory) = support::test_directory() {
        return Some(directory);
    }
    let candidates = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
        "/usr/local/lib",
    ];
    candidates
        .iter()
        .map(Path::new)
        .find(|directory| {
            ["fftw3", "fftw3f"]
                .iter()
                .all(|stem| support::library_path(Some(directory), stem).exists())
        })
        .map(Path::to_path_buf)
}

#[test]
fn tenferro_lock_alone_serializes_planning_without_the_thread_api() {
    let Some(source) = installation() else {
        assert!(!support::fftw_required(), "no FFTW installation found");
        eprintln!("FFTW integration requires a runtime");
        return;
    };
    let core_only =
        std::env::temp_dir().join(format!("tenferro-fftw-core-only-{}", std::process::id()));
    std::fs::create_dir_all(&core_only).unwrap();
    for stem in ["fftw3", "fftw3f"] {
        // `copy` follows the soname symlink and writes a regular file.
        std::fs::copy(
            support::library_path(Some(&source), stem),
            support::library_path(Some(&core_only), stem),
        )
        .unwrap();
    }

    // Without the threading library a threaded provider cannot load, which
    // shows that the installation exports no thread API, unless its core
    // library carries the thread API itself.
    match Fftw::load_from(
        &core_only,
        FftwPlanning::Estimate,
        NonZeroUsize::new(2).unwrap(),
    ) {
        Err(FftwError::Load { library, .. }) => {
            assert!(
                library.to_string_lossy().contains("fftw3_threads"),
                "{library:?}"
            );
        }
        Ok(_) => {
            eprintln!("{source:?} builds threads into the core library; nothing to test");
            std::fs::remove_dir_all(&core_only).unwrap();
            return;
        }
        Err(error) => panic!("{error}"),
    }
    let provider = support::load(Some(&core_only), 1).expect("the core libraries load");
    support::plan_concurrently(&provider, 4, support::rounds());
    drop(provider);
    std::fs::remove_dir_all(&core_only).unwrap();
}
