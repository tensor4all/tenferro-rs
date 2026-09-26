//! Compile-time contract for the surviving session surface (issue #1926).
//!
//! The route/API unification deletes the one-shot operation spelling
//! (`backend.add(&a, &b)`) and keeps the borrowed-session surface. These
//! fixtures pin the surface that must keep compiling **on both sides** of the
//! unification, so the refactor cannot quietly change what a caller is allowed
//! to write:
//!
//! * `_read`/`_into` operations on a borrowed `&mut dyn BackendSession`;
//! * `with_backend_session` as the boundary;
//! * `with_execution_scope` wrapping a session entry;
//! * the `Tensor`-side session extension (`a.add(&b, session)`, which is the
//!   only spelling that broadcasts);
//! * a generic helper over `&mut dyn BackendSession`, which is the shape the
//!   scheduler and extension code use;
//! * the cache-aware session route (`with_backend_session_cached` plus a
//!   session `_cached` contraction), which survives while the owner-level
//!   `BackendCachedDot` spelling does not.
//!
//! These fixtures deliberately do **not** import the operation traits
//! (`TensorElementwise`, `TensorDot`, `TensorReduction`). A method of a
//! supertrait is reachable through a type that is bounded by the subtrait
//! without importing the supertrait, and `BackendSession: TensorBackendOps`, so
//! `session.add_read(..)` resolves without those imports. That is the property
//! being pinned: the operations are reachable *through* `BackendSession`, which
//! is what must survive. Adding the trait imports would produce unused-import
//! warnings and would pin nothing extra.
//!
//! Sequencing note. The deletion has landed, and the *fail* side is pinned by
//! rustdoc `compile_fail` examples rather than by trybuild `.stderr` fixtures:
//! `BackendSession::add` in this crate's dependency, the owner one-shot
//! spellings and the owner `BackendCachedDot` bound in `tenferro-cpu`, and the
//! deleted `default_backend_session` factory in `tenferro-tensor`. A
//! `compile_fail` example only requires compilation to fail, so it does not
//! depend on the compiler's span rendering or on a local build wrapper
//! rewriting paths, which is what makes trybuild `.stderr` comparisons fragile
//! here. The `pass` fixtures in this file remain the counterweight proving the
//! surviving surface did not change, which is why
//! `session_surface_pass_contract` drives only the `pass` directory.

use std::fs;
use std::path::{Path, PathBuf};

/// Collect every `.rs` fixture under `root`, or an empty list when absent.
fn collect_rs_files(root: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if !root.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_rs_files(&path, files)?;
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
    Ok(())
}

fn surface_ui_files(kind: &str) -> Vec<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/ui/session_surface")
        .join(kind);
    let mut files = Vec::new();
    collect_rs_files(&root, &mut files).unwrap_or_else(|error| {
        panic!(
            "failed to discover session-surface {kind} UI fixtures under {}: {error}",
            root.display()
        )
    });
    files.sort();
    assert!(
        !files.is_empty(),
        "session-surface {kind} UI fixture set is empty under {}",
        root.display()
    );
    files
}

#[test]
fn session_surface_pass_contract() {
    // The contract drives `trybuild`, which compiles each fixture as an external
    // crate against the built library. That works under the nextest runner: the
    // CI workspace profile runs `cargo nextest run --workspace` plus
    // `cargo test --doc --workspace`, and the `fail` side of this contract is
    // rustdoc `compile_fail` examples, so both sides execute in CI. Measured here
    // under `cargo nextest run -p tenferro-runtime --test session_surface_contract`:
    // 1 passed, about 60 s cold (compiling the five fixtures) and about 3 s warm.
    let tests = trybuild::TestCases::new();
    for path in surface_ui_files("pass") {
        tests.pass(path);
    }
}
