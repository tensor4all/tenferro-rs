# Quality refactor and CI gate

**Date:** 2026-05-30
**Status:** Approved design, pending implementation plan

## Problem

The workspace currently passes formatting, `cargo check`, and tests, but
`cargo clippy --workspace --all-targets -- -D warnings` fails. That means CI can
merge changes that keep tests green while accumulating lint regressions and
avoidable dead code.

The quality review also found broader maintainability pressure:

- Some simple code paths trigger clippy warnings that are safe to fix without
  changing behavior.
- `ext/tropical` is a standalone crate outside the root workspace and needs an
  explicit gate if it should stay healthy.
- Several production modules use `#[allow(dead_code)]` around crate-private
  helpers. Some are feature-combination artifacts, but some are removable thin
  wrappers.
- Large public operation surfaces duplicate forwarding code across tensor API
  layers. That is real DRY debt, but it is too broad to fix safely in the same
  change as introducing a CI gate.

## Scope

In scope:

- Add a CI clippy gate for the root Rust workspace.
- Include `ext/tropical` in the clippy gate because it is not a workspace
  member.
- Fix the currently failing clippy lints.
- Remove low-risk unused crate-private wrappers where `rg` confirms there are
  no call sites.
- Leave any feature-conditional helper in place unless it is clearly unused
  across the crate.

Out of scope:

- Splitting large modules such as GPU CubeCL dispatch, tensor type surfaces, or
  runtime compiler planning.
- Reworking public tensor/eager/traced operation wrappers into a generated or
  macro-based abstraction.
- Changing coverage thresholds.
- Changing numerical behavior, backend dispatch, tensor layout, or public API.

## Design

### CI gate

Add a `clippy` job to `.github/workflows/ci.yml`:

```yaml
clippy:
  name: clippy
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
      with:
        components: clippy
    - uses: Swatinem/rust-cache@v2
    - name: Run workspace clippy
      run: cargo clippy --workspace --all-targets -- -D warnings
    - name: Run tropical extension clippy
      run: cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings
```

Update `.github/workflows/CI_gpu.yml` so the GPU workflow waits for `clippy`
alongside the existing non-GPU checks. This preserves the current pattern where
the expensive GPU lane starts only after cheaper required checks pass.

### Clippy fixes

Fix only mechanical warnings with no behavior change:

- Replace manual zero-extent scans such as
  `shape.iter().any(|&extent| extent == 0)` with `shape.contains(&0)`.
- Replace the primitive catalog count expression that expands to an unnecessary
  unit expression with an array-length expression.

These changes are intentionally local and do not alter shape semantics.

### Dead code cleanup

Use repository search before deletion:

```bash
rg 'typed_abs\(' tenferro-cpu
rg 'dot_general_faer\(' tenferro-cpu
```

Remove only crate-private wrappers that have no call sites and simply delegate
to still-retained `_with_pool` or cached variants. Keep helpers that are used in
non-default feature combinations, and prefer a short reason comment only when a
dead-code allow is still necessary after clippy passes.

### DRY boundary

Do not introduce a macro for the public operation wrapper families in this
change. The duplicated tensor/eager/traced wrappers are public-surface code, and
collapsing them safely requires API-level review. This change should make CI
enforce the baseline first; larger wrapper consolidation can follow once the
gate prevents regression.

## Testing strategy

Before implementation, verify the red condition:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

After implementation, run:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings
cargo test --workspace
cargo test --manifest-path ext/tropical/Cargo.toml
```

No new Rust unit tests are required because this change is intended to preserve
behavior and remove unused wrappers. If any refactor changes a callable helper
or public behavior, add a targeted test in the owning crate before production
code changes.

## Risks

- Feature-gated CPU helpers may appear dead in the default local build while
  being needed under a different feature set. Deletion must be limited to
  helpers confirmed unused by repository search and by the clippy gate.
- Adding clippy as `-D warnings` may expose future lint changes when stable Rust
  updates. That is intentional for CI, but fixes should remain mechanical and
  localized.
- The root CI workflow currently has a commented-out full workspace test job.
  This change does not alter that policy; it only adds the missing lint gate.

## Success criteria

- Root workspace clippy passes with `-D warnings`.
- Tropical extension clippy passes with `-D warnings`.
- Existing workspace tests still pass.
- CI lists `clippy` as a non-GPU prerequisite before CUDA tests run.
- The diff does not introduce new public APIs or numerical behavior changes.
