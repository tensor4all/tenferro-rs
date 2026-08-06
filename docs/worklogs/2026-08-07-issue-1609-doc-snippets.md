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
- Used the four accepted binaries: `core_snippets`, `math_snippets`,
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
  drift.

## Migration ledger

The final audit covers 91 Rust fences: 87 compiled references and four
explicitly ignored fragments. The 75 newly named regions are distributed as:

| Family binary | Guide/tutorial fences | Regions |
| --- | ---: | ---: |
| `core_snippets.rs` | 36 | 36 |
| `math_snippets.rs` | 24 | 24 |
| `execution_snippets.rs` | 10 | 10 |
| `extension_snippets.rs` | 5 | 5 |

The 12 remaining compiled references were already complete standalone source
files and retain whole-file `snippet-source` references: CPU provider choice
and multiple engines; CUDA quickstart; traced FFT; XLA/PJRT execution; and the
six existing standalone tutorial binaries (dynamic-shape SVD, eager AD,
einsum gradients, traced AD, typed tensor, and XLA einsum). They are included
in the 87-fence compiled total and are not plain unmarked fences.

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

- `python3 scripts/test-check-doc-snippets.py`
- `python3 scripts/check-doc-snippets.py --check` → `doc-snippets-ok`.
- `python3 scripts/test-check-doc-snippets.py` → `check-doc-snippets-tests-ok`.
- `cargo test --manifest-path docs/tutorial-code/Cargo.toml --release tutorial_binaries_run_successfully -- --exact` → passed.
- The inventory reports zero unmarked plain Rust fences; 91 fences are
  accounted for as 87 compiled references and four ignored fragments.

## Residual risks

- CUDA and Apple examples are compile/run gated by hardware/features; ordinary
  CI validates compilation and skips unavailable hardware execution.
- The tutorial crate now resolves the standalone extension crates through path
  dependencies; the root workspace excludes preserve those crates' own
  workspace behavior.
- Tutorial snippet binaries intentionally contain many independent examples,
  so compiler warnings for unused setup/items are expected and harmless.
