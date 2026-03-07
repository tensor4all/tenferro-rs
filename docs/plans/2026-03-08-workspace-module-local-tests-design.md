# Workspace Design: Move Inline Unit Tests into Module-Local Test Directories

## Summary

The workspace currently mixes production code and inline `#[cfg(test)]` blocks
across multiple crates. This wastes reading context, especially in large Rust
modules, and makes test organization inconsistent across the repository.

The workspace rule is now:

- keep production files focused on production code
- move inline unit tests into module-local test directories
- leave only `#[cfg(test)] mod tests;` in source files
- reserve crate-root `tests/` for integration tests

This rollout applies to the whole workspace, including root crates,
`extension/*`, and `extern/*`.

## Problem

Several crates still keep unit tests inline inside `src/**`. This creates three
practical issues:

- production files are harder to scan
- AI agents waste context tokens reading large test blocks
- the repository has no uniform unit-vs-integration layout

The problem is structural rather than behavioral. Existing tests already cover
the expected behavior; the issue is that they live in the wrong place.

## Decision

Adopt module-local test directories everywhere in the workspace where inline
unit tests still exist.

For a file module:

- `foo.rs` keeps `#[cfg(test)] mod tests;`
- test bodies move to `foo/tests/mod.rs`

For root or directory modules:

- `lib.rs` keeps `#[cfg(test)] mod tests;`
- test bodies move to `src/tests/mod.rs`

Crate-root `tests/` remains dedicated to integration tests.

## Scope

The rollout audits the entire workspace:

- root crates
- `extension/*`
- `extern/*`

At the time of design, inline unit tests requiring migration exist in:

- `tenferro-tensor`
- `tenferro-prims`
- `tenferro-einsum`
- `tenferro-linalg`
- `extension/tenferro-tropical`

`extern/*` is in scope for the audit and final verification, but there are no
inline `mod tests { ... }` blocks there at the time of this design.

## Implementation Strategy

Use crate-level parallelization to minimize edit conflicts:

- one lane for `tenferro-tensor`
- one lane for `tenferro-prims` and `extension/tenferro-tropical`
- one lane for `tenferro-einsum`
- one lane for `tenferro-linalg`
- one audit lane for `extern/*`

Within each crate:

- preserve test contents exactly on first move
- extract to `tests/mod.rs` beside the owning module
- split only obviously huge suites after the directory layout is in place

This keeps the refactor mechanical and low-risk.

## Behavior Changes

No runtime behavior change is intended.

Expected visible effects are limited to:

- shorter production files
- consistent workspace test layout
- unchanged unit and integration test behavior

## Risks

- path resolution can be subtle when moving from inline modules to sibling test
  directories
- root modules like `lib.rs` require a different destination than normal file
  modules
- large mechanical moves can drop helper imports if not verified carefully

## Testing

Each affected crate should pass its own tests after migration:

- `cargo fmt --all --check`
- `cargo test -p <crate>`

After integrating all crate-level changes, run:

- `cargo fmt --all --check`
- `cargo test --workspace`

Also re-scan the workspace, including `extern/*`, to confirm that no inline
`mod tests { ... }` blocks remain in tracked source files.

## Rationale

This keeps unit tests close to the code they exercise without polluting the
production file itself. It also turns the new `AGENTS.md` rule into an actual
workspace convention rather than a crate-specific cleanup.
