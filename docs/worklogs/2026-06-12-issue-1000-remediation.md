# Issue 1000 Remediation Slice

## Session Summary

This slice addressed the concrete Auto Fix items from GitHub issue #1000 that
could be verified against the current tree without changing public API, AD
semantics, GPU support claims, coverage policy, or performance design. The
work started as a local no-PR pass and was later promoted into the same draft
PR as the AI documentation issue batch.

- Fixed `CpuBackend::solve` so invalid dtype pairs are rejected before the
  zero-dimension fast path.
- Hardened extension direct execution boundaries so registered runtimes cannot
  silently return fewer or more outputs than the declaring `ExtensionOp`.
- Hardened `apply_expanded_graph` so graph-build output metadata count
  mismatches return a typed runtime error instead of silently truncating.
- Added checked tensor/view materialization helpers and used them at runtime
  `Result` boundaries that previously could panic on backend-backed views.
- Aligned `REPOSITORY_RULES.md` with the current CPU provider default:
  `CpuBackend::new()` selects BLAS/LAPACK when `cpu-blas` is compiled,
  otherwise faer.
- Updated active reference/design docs that still described retired
  `tenferro-prims`, `tenferro-linalg-prims`, and `tenferro-internal-device`
  crate boundaries as if those crates existed in the current workspace.
- Recorded a classification ledger for the broader #1000 findings so the
  umbrella issue is not treated as fully closed by this narrow fix.

## Context Read

- `AGENTS.md`
- `CONTRIBUTING.md`
- `REPOSITORY_RULES.md`
- `ai/contribution-workflows/repository-remediation.md`
- GitHub issue #1000 body
- `crates/tenferro-linalg/src/cpu/backend.rs`
- `crates/tenferro-linalg/tests/backend_errors.rs`
- `crates/tenferro-cpu/src/backend.rs`
- `crates/tenferro-runtime/src/extension.rs`
- `crates/tenferro-runtime/src/extension_runtime.rs`
- `crates/tenferro-runtime/src/exec.rs`
- `crates/tenferro-tensor/src/types.rs`
- `crates/tenferro-tensor/src/backend.rs`
- `docs/reference/jax-stablehlo-needed.md`
- `docs/reference/libtorch.md`
- `docs/reference/pytorch-dense-cpu-parity.md`
- `docs/design/linalg-prims.md`
- `docs/getting-started/index.md`

## Classification Ledger

| #1000 finding | Classification | Current action |
| --- | --- | --- |
| Active AD rules exceed oracle/replay coverage | Verify First / Deferred | Not changed here. #987 already tracks full-pivot LU oracle work; broader oracle/support alignment needs its own focused slice. |
| CUDA placement and backend-buffer diagnostics | Verify First / Deferred | Not changed here. Needs GPU/feature-aware behavior tests for host, wrong-device, zero-length, and unsupported-dtype ordering. |
| CPU/GPU performance and hidden materialization risks | Verify First / Deferred | Not changed here. Needs scale-sensitive tests or benchmarks before remediation. |
| Public API and extension-boundary panic or invariant-loss risks | Mixed / Partially fixed | Added reduced repro tests and fixes for extension output-count mismatches, expanded-graph output metadata mismatches, and backend-view materialization through `Result` boundaries. Other non-`Result` public panic APIs such as operator overloads and `dot_general` remain verify-first/deferred. |
| CPU `solve` empty-shape path bypasses dtype-pair validation | Auto Fix / Fixed | Added regression coverage and moved dtype-pair validation before the zero-dimension fast path. |
| CPU provider rule drift | Auto Fix / Fixed | Updated `REPOSITORY_RULES.md` to match current `CpuBackendKind::default_compiled()`. |
| Guide dependency drift for einsum/FFT/basic guides | Stale in current branch | Addressed earlier in the same local batch for #1011-#1014. |
| Broad docs/tooling drift around unmarked snippets and stale reference docs | Mixed / Partially fixed | Updated active reference/design docs that named retired crate boundaries. Snippet/tooling expansion is still deferred. |

## Decisions Made

- Kept `ensure_supported_linalg_pair` as the single validation helper for
  `solve`, matching `lu_solve_prepared` and the linalg dtype contract.
- Did not change the unsupported-dtype error shape. Same unsupported integer
  dtype pairs still return the existing backend failure message; mixed dtype
  pairs return `DTypeMismatch`.
- Kept existing panic-capable convenience APIs such as `TensorView::to_tensor`
  for compatibility, but added checked counterparts (`try_to_tensor`) and used
  those at `Result` boundaries.
- Did not add a new public `try_dot_general` or change operator-overload
  semantics; those would be broader public API decisions.
- Did not attempt to close #1000 as a whole. The issue remains an umbrella
  backlog with multiple verify-first or design-gated findings.

## Verification Performed

- RED: `cargo test -p tenferro-linalg --test backend_errors solve_rejects_invalid_dtype_pairs_before_zero_dim_fast_path`
  failed because `solve` returned `Ok(C64 empty)` for `F64`/`C64` empty inputs.
- GREEN: `cargo test -p tenferro-linalg --test backend_errors solve_rejects_invalid_dtype_pairs_before_zero_dim_fast_path`
- `cargo fmt --all`
- `cargo test -p tenferro-linalg --test backend_errors`
- RED: `cargo test -p tenferro-runtime --test extension_runtime output_count_mismatch`
  failed because direct runtime execution returned `Ok` with the wrong output count.
- RED: `cargo test -p tenferro-runtime extension::tests::apply_expanded_graph_rejects_output_metadata_count_mismatch`
  failed because `apply_expanded_graph` returned `Ok(vec![])`.
- RED: `cargo test -p tenferro-runtime --test extension_runtime backend_view_materialization`
  failed because backend-backed view materialization panicked before returning `Err`.
- GREEN: `cargo test -p tenferro-runtime --test extension_runtime output_count_mismatch`
- GREEN: `cargo test -p tenferro-runtime extension::tests::apply_expanded_graph_rejects_output_metadata_count_mismatch`
- GREEN: `cargo test -p tenferro-runtime --test extension_runtime backend_view_materialization`
- `cargo test -p tenferro-runtime --test extension_runtime`
- `cargo test -p tenferro-runtime --lib extension::tests`
- `cargo test -p tenferro-runtime --test runtime_public_api`
- `cargo test -p tenferro-runtime --test graph_default_input_placement`
- `cargo test -p tenferro-tensor types_tests`
- `cargo test -p tenferro-tensor --doc`
- `cargo test -p tenferro-cpu --test runtime_error_tests`
- `cargo fmt --all --check`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-doc-snippets.py --check`
- `python3 scripts/check-docs-site.py`
- `git diff --check`

## Remaining Risks

- Full #1000 remediation is incomplete by design. GPU, AD oracle,
  performance/materialization, non-`Result` public API panic, and broader
  docs/tooling findings need separate focused tests or design gates before
  implementation.
