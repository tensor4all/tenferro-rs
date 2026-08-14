# 2026-08-14 — #1680 Phase 3: extension surfaces + GPU guard (single API complete)

## Session summary

Completed the single-API end state for the remaining backend-taking concrete
surfaces and wired the GPU nested-entry guard, in one designed-and-gated PR
(39 files, net −11 lines: 1949+ / 1960−). After this phase, **no concrete op
takes a non-session `&mut B`** in the migrated areas; every concrete op runs
on a borrowed session (core, einsum, fft, linalg, shape-packing), and
CUDA/WebGPU enforce the nested-entry contract like CPU and the default
adapter.

## Gate record (frontier review, per AGENTS.md)

- **Pre-implementation design gate** (reviewer-gpt on
  `docs/design/session-oriented-concrete-apis.md` §"Phase 3"): 3 rounds →
  **approved**. Round 1: 4 blocking (custom-capability compatibility →
  chose option (b) built-in dispatch with documented breaking restriction;
  linalg dispatch completeness incl. solve_read_into + typed_output; GPU
  verification commands; missing FFT/linalg baselines) + 3 minor (tensordot
  not a plan path; FFT inventory; shape-packing grep gate). Rounds 2-3:
  baseline values + wording.
- **Post-implementation diff gate** (reviewer-gpt on the full ~7.8k-line
  diff): 2 rounds → **Correct-to-merge**. Round 1: 3 blocking (docs/guides
  still taught the retired `&mut impl LinalgBackend` contract;
  caller-downcasts remained in concrete tests; worklog + verification
  evidence missing) + 1 minor (direct solve_read_into path not provably
  taken). All fixed; Round 2 confirmed.

## Design decisions

- **Option (b) capability decision**: concrete ops take `&mut dyn
  BackendSession` with internal built-in dispatch (reusing the existing
  linalg/fft extension-session downcast patterns). Documented breaking
  restriction: the concrete op traits no longer accept third-party
  LinalgBackend/FftBackend impls; the SPI traits stay for backend
  implementers; test-only custom backends migrated (assert typed capability
  errors) or removed.
- **solve_read_into direct path preserved** (no allocate+copy); typed
  contract via `typed_output` (not into_typed_result).
- **GPU guard**: `with_session_entry_guard` (`#[doc(hidden)] pub`, thread-
  local flag + panic-safe Drop restore + debug assert) extracted and used by
  `default_backend_session` + both GPU `with_backend_session` overrides; CPU
  EXECUTION_OWNER release panic and Send soundness bounds untouched.
- tensordot calls `session.dot_general` directly (not a plan path); FFT plan
  cache stays executor-owned; einsum top-level internals keep routing
  through ConcreteEinsumPlan.

## Measurements (release, pinned core 40; pre = main, post = Phase-3)

| arm | pre | post |
|---|---|---|
| FFT 256×1 direct_one_shot | 15.0 µs | 14.5 µs |
| FFT 256×1 executor_warm | 11.3 µs | 10.9 µs |
| FFT 1024×16 direct_one_shot | 88.3 µs | 84.2 µs |
| FFT 1024×16 executor_warm | 73.1 µs | 71.5 µs |
| linalg solve 8×8 (session form) | 14.4 µs | **14.3 µs** (orchestrator-verified) |
| linalg svd 8×8 (session form) | 18.6 µs | **18.8 µs** (orchestrator-verified) |

All within the interleaved noise band — **no regression** (the internal
dispatch replaced the caller's manual `with_cpu_exec_session` wrapper with
no measurable cost; FFT arms slightly improved).

## Verification

- Workspace build; test suites: runtime 402+146+1+413, einsum 161+24+1+87,
  fft 18+6+20+1+18, linalg 124+134+1+126, ad 86+337+1+147, cpu
  512+1+46+2+185, tensor 260+…+326 — all pass (incl. doctests)
- Doc scripts: guide-dependency-snippets-ok, test-doc-consistency exit 0,
  doc-snippets-ok; fmt clean; clippy workspace 0 findings
- GPU: `cargo check -p tenferro-gpu --features cuda` and
  `--features webgpu` (both --all-targets) pass; the cfg-gated nested-entry
  tests for CUDA/WebGPU compile but **execute only on a GPU host** (CI gate
  item); the shared helper's panic-restore/nested tests run on CPU
- Source-contract tests: `cpu_solve_read_into_reaches_direct_write_for_eligible_outputs`
  and `cpu_solve_read_into_entered_writes_into_the_caller_buffer` prove the
  direct write path is taken (not allocate+copy)
- Grep: no `&mut B` in the 5 areas' public concrete methods; no
  `with_cpu_exec_session`/`with_cuda_exec_session` at migrated call sites
  (remaining uses are SPI-level: runtime() access, provider-contract tests);
  `with_session_entry_guard` used by default adapter + CUDA + WebGPU

## Residual risks

- GPU nested-entry tests are compile-verified locally; runtime execution
  requires a CUDA host (CI gate).
- The option-(b) breaking restriction (concrete op traits are
  built-in-session only) narrows third-party extensibility of the concrete
  path; the SPI traits remain for backend implementers. Documented in the
  design doc + PR body.
