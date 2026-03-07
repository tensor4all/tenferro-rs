# Issue 356 Design: Move DyAdTensor Unit Tests into Module-Local Test Directories

## Summary

`tenferro-dyadtensor` currently mixes production code and large inline
`#[cfg(test)]` blocks across many `src/**` modules. That makes the production
source harder to read and wastes both human and AI context when inspecting the
implementation.

For issue #356, inline unit tests in normal modules will move into
module-local test directories. Crate-root `tests/` remains reserved for
integration tests.

## Problem

The crate currently uses three different test layouts:

- inline `#[cfg(test)] mod tests { ... }` inside `src/**`
- crate-root integration tests under `extension/tenferro-dyadtensor/tests/`
- small module-local test blocks in a few files

The main problem is the large inline unit-test blocks. Files such as
`src/api/ad.rs`, `src/api/mod.rs`, and `src/dyn_types.rs` are already large,
and the appended test blocks make them significantly harder to navigate.

This is not just a style issue. In practice:

- humans have to scroll through test code to understand the implementation
- AI agents consume unnecessary context tokens when opening production files
- the crate has inconsistent test organization rules, so new tests tend to
  follow the existing local style rather than a clear project convention

## Decision

The crate will adopt a stronger separation rule for Rust unit tests:

- keep production source files focused on production code
- move inline unit tests out of normal modules into module-local test
  directories
- leave only `#[cfg(test)] mod tests;` in source files
- keep crate-root `tests/` for integration tests only

In `tenferro-dyadtensor`, this applies to all current inline test modules
except genuinely tiny leaf modules where extracting the tests adds more noise
than it removes. Because the explicit goal is to keep AI and human reading
context clean, this issue will move essentially all existing inline test blocks
in the crate, including small-to-medium ones such as `context.rs`.

## Scope

Move inline unit tests out of the following modules into module-local test
directories:

- `src/ad_value.rs`
- `src/context.rs`
- `src/dyn_types.rs`
- `src/reverse_tape.rs`
- `src/runtime.rs`
- `src/api/mod.rs`
- `src/api/ad.rs`
- `src/api/chainrules_api.rs`
- `src/structured/layout.rs`
- `src/structured/einsum.rs`
- `src/structured/meta.rs`

Do not move existing crate-root integration tests under
`extension/tenferro-dyadtensor/tests/`.

## Layout

Each affected source file will keep:

```rust
#[cfg(test)]
mod tests;
```

The test bodies will move into sibling module directories, for example:

- `src/dyn_types/tests/mod.rs`
- `src/api/tests/mod.rs`
- `src/api/ad/tests/mod.rs`
- `src/structured/meta/tests/mod.rs`

Where a moved test suite is already large, it should be split by concern inside
that test directory rather than keeping a single monolithic file. The first
step for this issue is to establish the directory-based layout everywhere and
split the largest suites where the grouping is obvious.

## AGENTS.md Update

`AGENTS.md` will gain a repository-wide rule for Rust test organization:

- avoid inline `#[cfg(test)]` blocks in normal modules
- use module-local test directories by default
- reserve crate-root `tests/` for integration tests
- optimize for clean reading context for both humans and AI agents

This change makes the issue-specific cleanup a codified project convention
instead of a one-off refactor.

## Behavior Changes

There is no intended runtime behavior change.

Expected externally visible effects are limited to:

- the same unit tests continue to compile and run
- module organization becomes more uniform
- production files become shorter and easier to inspect

## Risks

- moving tests may break imports if they relied on `super::*` in a way that
  changes across nested module boundaries
- module path resolution for file modules with sibling test directories can be
  subtle, especially for paths like `src/api/ad.rs` -> `src/api/ad/tests/mod.rs`
- large mechanical moves can accidentally drop helper functions or `use`
  statements if not verified carefully

## Testing

Verification should focus on structure-preserving safety:

- run `cargo fmt --all`
- run `cargo test -p tenferro-dyadtensor`
- if needed while iterating, run targeted crate tests for the moved modules

## Rationale

This is the narrowest refactor that directly addresses the real pain point:
production files should not be padded with unit-test bodies. Moving tests into
module-local directories keeps unit tests close to the code they exercise,
preserves the unit-vs-integration boundary, and reduces reading-context waste
without changing public behavior.
