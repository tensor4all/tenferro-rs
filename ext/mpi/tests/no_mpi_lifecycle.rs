//! The adapter must never own the MPI lifecycle or a communicator.
//!
//! #2612 scope: the crate borrows a caller's communicator and owns nothing else, so a
//! call to `MPI_Init`, `MPI_Finalize`, `MPI_Comm_dup`, `MPI_Comm_create` or
//! `MPI_Comm_free` in the adapter's own sources is a scope violation, not a style
//! preference. This is a source scan because the property has to hold for code paths
//! no test executes.

use std::path::{Path, PathBuf};

const FORBIDDEN: &[&str] = &[
    "MPI_Init",
    "MPI_Init_thread",
    "MPI_Finalize",
    "MPI_Comm_dup",
    "MPI_Comm_create",
    "MPI_Comm_free",
    "MPI_Comm_split",
    "mpi::initialize",
    "initialize_with_threading",
    "SimpleCommunicator::world",
    "Communicator::world",
    "world()",
];

fn sources(dir: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            sources(&path, found);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            found.push(path);
        }
    }
}

#[test]
fn the_adapter_never_initializes_or_owns_mpi() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    sources(&root, &mut files);
    assert!(!files.is_empty(), "no sources under {}", root.display());

    for file in files {
        let text = std::fs::read_to_string(&file).expect("readable source");
        // Test modules may name the forbidden calls to assert their absence, and a
        // doc comment may mention them; only code lines are checked.
        for (number, line) in text.lines().enumerate() {
            let code = line.trim_start();
            if code.starts_with("//") || code.starts_with("*") {
                continue;
            }
            for forbidden in FORBIDDEN {
                assert!(
                    !code.contains(forbidden),
                    "{}:{}: the transport adapter must not call `{forbidden}`",
                    file.display(),
                    number + 1
                );
            }
        }
    }
}
