# Nested Cargo Debug Profile Design

## Goal

Keep tenferro's independent guide and trybuild Cargo fixtures debug-free by default, matching the root workspace's local `dev` and `test` profiles, while preserving the documented one-command debugger override.

## Current behavior

The root workspace uses `debug = 0` for local `dev` and `test` builds. The guide dependency checker creates standalone Cargo workspaces and runs `cargo run`; trybuild creates standalone Cargo workspaces and runs nested Cargo checks. Cargo profile tables do not cross workspace roots, so these fixtures currently fall back to Cargo's full dev debuginfo and retain multi-gigabyte DWARF artifacts.

## Design

Add a repository-local `.cargo/config.toml` environment default:

```toml
[env]
CARGO_PROFILE_DEV_DEBUG = { value = "0", force = false }
```

Cargo passes repository-configured environment variables to test executables, so nested trybuild Cargo inherits the default. `force = false` preserves an explicit caller value such as `CARGO_PROFILE_DEV_DEBUG=1` or `2`.

The guide dependency checker can also be launched directly as Python, outside Cargo. Its subprocess environment therefore sets the same default with `setdefault` before invoking fixture Cargo. It must preserve an explicit caller override.

Do not force `CARGO_PROFILE_TEST_DEBUG`: the independent fixtures compile with Cargo's dev profile. Do not add a custom fixture profile, fork trybuild, or mutate process-global environment from Rust test threads.

## Measurement

Use fresh target directories, four Cargo jobs, empty `RUSTC_WRAPPER`, and the same toolchain and commit. Measure allocated bytes, wall time, nested subtree size, and largest files for:

1. `scripts/check-guide-dependency-snippets.py`;
2. `tenferro-tensor`'s storage trybuild contract.

Compare the unmodified baseline to the implementation. Confirm representative fixture artifacts contain DWARF before and no `.debug_*` sections after. A single cold sample per configuration establishes artifact-size effect; timings are reported as indicative rather than a benchmark.

## Verification

- Add Python contract tests proving the guide subprocess defaults dev debuginfo to zero and preserves an explicit override.
- Add a repository config contract test proving the nested-Cargo default and `force = false` override behavior.
- Run the guide checker and the focused trybuild contract.
- Run repository formatting, focused local PR gate, CI-parity clippy, repository-rule review, and required PR checks.

## Compatibility and non-goals

Compile diagnostics, feature selection, optimization level, debug assertions, overflow checks, and incremental behavior remain unchanged. This change does not introduce shared cross-worktree target directories, automatic target garbage collection, sccache, or new dependencies.
