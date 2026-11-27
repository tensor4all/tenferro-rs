//! The adapter must never own the MPI lifecycle or a communicator, and the deleted
//! MPItrampoline backend must stay deleted.
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
    // #2735 (docs/design/mpi-native-abi.md §3 "May not"): the host owns the error handler,
    // the attributes and the process; tenferro never aborts, disconnects or sets either.
    "MPI_Abort",
    "MPI_Comm_disconnect",
    "MPI_Comm_set_errhandler",
    "MPI_Comm_set_attr",
    "MPI_Comm_create_keyval",
    "MPI_Comm_free_keyval",
    "MPI_Comm_set_name",
    // Wildcard receives: the API names its source and tag, so a stray message cannot match.
    "any_process",
    "MPI_ANY_SOURCE",
];

/// Whether `code` names `forbidden` as a whole identifier: `MPI_Finalized` (allowed, and called on
/// every C entry) must not trip the `MPI_Finalize` entry, nor `MPI_Initialized` the `MPI_Init` one.
fn names(code: &str, forbidden: &str) -> bool {
    let is_ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    code.match_indices(forbidden).any(|(at, _)| {
        let before = code[..at].chars().next_back();
        let after = code[at + forbidden.len()..].chars().next();
        // `forbidden` may itself end in punctuation (`world()`); only word-like ends need a boundary.
        let starts_word = forbidden.chars().next().is_some_and(is_ident);
        let ends_word = forbidden.chars().next_back().is_some_and(is_ident);
        !(starts_word && before.is_some_and(is_ident))
            && !(ends_word && after.is_some_and(is_ident))
    })
}

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
    let mut files = Vec::new();
    for crate_dir in ["", "../tenferro-mpi-capi"] {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(crate_dir)
            .join("src");
        let before = files.len();
        sources(&root, &mut files);
        assert!(files.len() > before, "no sources under {}", root.display());
    }

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
                    !names(code, forbidden),
                    "{}:{}: the transport adapter must not call `{forbidden}`",
                    file.display(),
                    number + 1
                );
            }
        }
    }
}

/// Every file under `dir` that is text we own (not build output).
fn text_files(dir: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        if name == "target" || name == ".git" {
            continue;
        }
        if path.is_dir() {
            text_files(&path, found);
        } else if path.extension().is_some_and(|e| {
            ["rs", "sh", "py", "md", "toml", "yml", "yaml", "txt"]
                .contains(&e.to_str().unwrap_or(""))
        }) {
            found.push(path);
        }
    }
}

/// #2735 / D9: the MPItrampoline path is deleted, not kept as a fallback. Its absence is
/// asserted, not hoped for (docs/design/mpi-native-abi.md, "What depends on the absence").
/// The tokens are assembled at run time so this file does not match itself. Worklogs and the
/// design notes under `docs/` are dated history and are not scanned.
#[test]
fn the_mpitrampoline_backend_stays_deleted() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repo = manifest.join("../..");
    let banned = [
        ["MPI_RT", "_LIB"].concat(),
        ["libmpi", "wrapper"].concat(),
        ["MPIABI", "_"].concat(),
    ];
    // The crate name `mpi-rt-sys` is asserted absent from every dependency graph by
    // scripts/check-mpi-features.sh, which has to name it, so it is not banned as text.
    let mut files = Vec::new();
    for dir in ["crates", "scripts"] {
        text_files(&repo.join(dir), &mut files);
    }
    files.push(repo.join("Cargo.toml"));
    assert!(
        files.len() > 10,
        "the scan found almost nothing under {}",
        repo.display()
    );
    for file in files {
        let text = std::fs::read_to_string(&file).unwrap_or_default();
        for token in &banned {
            assert!(
                !text.contains(token.as_str()),
                "{} mentions `{token}`: the MPItrampoline backend is deleted (#2735)",
                file.display()
            );
        }
    }
}
