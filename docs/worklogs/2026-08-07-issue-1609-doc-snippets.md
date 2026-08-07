# Work log: issue #1609 compiled guide and tutorial snippets

## Summary

Migrated the remaining plain Rust fences under `docs/guides/` and
`docs/tutorials/` to named regions in four compiled tutorial binaries. The
existing snippet synchronizer now accepts `path#region` references and rejects
malformed source regions. This work is standalone for #1609.

## Context read

- `AGENTS.md`, shared common/docs-and-tests and Rust rules, and
  `REPOSITORY_RULES.md`.
- Accepted #1609 implementation plan and audit comments (including the
  current-main re-audit requirement, four binary families, and ignore rules).
- `scripts/check-doc-snippets.py`, the new focused checker tests, existing
  tutorial-code manifest/tests, and all guide/tutorial fence inventories.

## Decisions

- Kept plain `path` references as whole-file extraction and added line-based
  named-region extraction without a Markdown/Rust parser dependency.
- Used the four accepted binaries: `core_tensor_snippets`, `math_snippets`,
  `execution_snippets`, and `extension_snippets`.
- Added only four ignored fragments: the two abbreviated KdV autodiff bodies
  and the two AD continuation fragments in sparse/tropical tutorials. Each has
  an adjacent explanation. Hardware-only Apple snippets are feature-gated;
  CUDA snippets compile against the current `tenferro_gpu::cuda` API and exit
  cleanly when no device is present.
- Added root-workspace excludes for the two standalone extension workspaces so
  the tutorial crate can depend on them by path without flattening their
  independent workspaces.
- Re-audited and fixed the named remaining defects: the one-thread backend
  construction, both troubleshooting tensor constructors, and both KdV launch
  commands. The broader migration also updated stale examples to current
  fallible constructors and backend/session APIs where compilation exposed
  drift, including the Apple FFT capability wrappers and the eager FFT trait
  feature.
- Guide regions may use hidden setup lines in their compiled family function;
  this preserves the displayed continuation while compiling the complete
  example, as allowed by the accepted plan. Nested cache and einsum examples
  execute their assertions rather than leaving an uncalled helper function.
- CUDA guide snippets probe `gpu_available()` before driver discovery so the
  tutorial binary exits cleanly on non-CUDA hosts. The TBLIS workspace contract
  test now checks for the required exclusion within the root exclusion list,
  while allowing the additional standalone extension-workspace exclusions.

## Migration ledger

The final audit covers 91 Rust fences: 87 compiled references and four
explicitly ignored fragments. The 75 newly named regions are distributed as:

| Family binary | Guide/tutorial fences | Regions |
| --- | ---: | ---: |
| `core_tensor_snippets.rs` | 36 | 36 |
| `math_snippets.rs` | 24 | 24 |
| `execution_snippets.rs` | 10 | 10 |
| `extension_snippets.rs` | 5 | 5 |

The 12 remaining compiled references were already complete standalone source
files and retain whole-file `snippet-source` references: CPU provider choice
and multiple engines; complex AD; CUDA quickstart; traced FFT; XLA/PJRT
execution; and the six existing standalone tutorial binaries (dynamic-shape
SVD, eager AD, einsum gradients, traced AD, typed tensor, and XLA einsum).
They are included in the 87-fence compiled total and are not plain unmarked
fences.

Family coverage is:

- Core: eager operations, tensor operations, memory order, CPU execution,
  complex AD, device/GPU placement, views, and the two standalone CPU-provider
  guide examples.
- Math: linear algebra, einsum, autodiff, FFT, and the standalone traced-FFT
  guide example.
- Execution: parallelism/caching, troubleshooting, XLA, KdV context, and the
  standalone XLA/PJRT guide example.
- Extension: sparse and tropical extension entry points.
- Ignored: two intentionally abbreviated KdV autodiff fragments and the
  sparse/tropical AD continuation fragments whose setup is defined by the
  preceding example. Each has an adjacent explanation and uses
  `rust,ignore`.

## Verification

- `python3 scripts/test-check-doc-snippets.py` → `check-doc-snippets-tests-ok`.
- `python3 scripts/check-doc-snippets.py --check` → `doc-snippets-ok`.
- `cargo test --manifest-path docs/tutorial-code/Cargo.toml --release tutorial_binaries_run_successfully -- --exact` → passed.
- `cargo check` and Clippy for `docs/tutorial-code` with `--all-features
  --all-targets` → passed.
- The documented Apple feature test command with `doc-snippets` → passed on
  non-Apple hardware with Apple runtime binaries gated/skipped.
- `cargo test --manifest-path docs/tutorial-code/Cargo.toml --no-default-features
  --features cpu-blas --no-run` and the equivalent `cpu-faer` command → passed;
  doc-only binaries and their integration test are now gated by `doc-snippets`.
- `cargo test -p tenferro-gpu --features cuda --lib
  cubecl::runtime::identity_tests::cuda_backend_identity_tracks_the_exact_runtime_when_hardware_is_available
  -- --exact` → passed; `gpu_available()` now probes full runtime initialization
  instead of CubeCL's lazy client construction, and synchronizes the created
  stream, so no-driver hosts are skipped.
- `python3 scripts/ci/run_profile.py docs` and the coverage-reviewed fast PR
  checks → passed.
- The inventory reports zero unmarked plain Rust fences; 91 fences are
  accounted for as 87 compiled references and four ignored fragments.

## Residual risks

- CUDA and Apple examples are compile/run gated by hardware, target, and
  features; ordinary CI validates compilation and skips unavailable hardware
  execution. CUDA availability probing now includes driver/runtime/context
  initialization; hosted CUDA coverage remains dependent on runner setup.
- The tutorial crate now resolves the standalone extension crates through path
  dependencies; the root workspace excludes preserve those crates' own
  workspace behavior.
- Tutorial snippet binaries intentionally contain many independent examples,
  so narrow crate-level allowances for `dead_code`, `unused_imports`,
  `unused_variables`, and `unused_mut` (plus `clippy::needless_borrow` in
  `math_snippets.rs`) keep `-D warnings` from flagging composition noise
  without rewriting the preserved snippets.
