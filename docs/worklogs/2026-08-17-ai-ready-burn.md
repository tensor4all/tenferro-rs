# AI-ready issue burn work log

This change batches the eight open `ai-ready` issues present on 2026-08-17 into
one reviewable PR. The worktree was created from `origin/main`; unrelated
uncommitted changes in the parent worktree were left untouched.

## Scope and decisions

| Issue | Delivered decision |
| --- | --- |
| #1694 | Gate the debug-assertion-only residual-mask panic test with `cfg(debug_assertions)`; release runtime behavior is unchanged. |
| #1693 | Remove the non-homogeneous `max_diagonal.max(1.0)` floor from Faer complete-pivot LU singularity checks and add small/large scale regression coverage for f32/f64/complex plus exact singular rejection. |
| #1646 | Add `svdvals`, `svdvals_read`, and typed `svdvals` surfaces; add borrowed values-only backend hooks. Faer consumes eligible host-strided views directly with no vector outputs; providers needing owned storage materialize explicitly; CUDA read hooks materialize on-device. Existing `eigvalsh`/`eigvalsh_read` route through the same values-only hook family. |
| #1687 | Add the CPU/faer-specific scoped `FaerParallelismExt::with_faer_parallelism` capability. It derives `faer::Par` from the existing execution context and exposes no executor handle or lease. |
| #1688 | Extend the existing release benchmark with checked-vs-unsafe slice and nested fixed-rank comparisons. Document checked/unchecked contracts and the supported compact-host expert path; add no new unsafe public API. |
| #1686 | Reorder the tutorial landing page around ordinary workflows, make the CPU quickstart use one session for matmul/solve/SVD values, and link GPU, ndarray/faer, and direct external-linalg paths. |
| #1661 | Add the focused API migration guide, index it through `docs/llms.txt`, update the bundled compute skill and mirrors, and add the approval-gated documentation-gap reporting procedure. No aliases or runtime compatibility shims were added. |
| #1046 | Add one small CUDA upload/session/download tutorial artifact. Both non-GPU archive lanes build/archive the exact artifact; RunPod and the supported legacy GPU lane execute it with `TENFERRO_REQUIRE_CUDA=1`. |

## Evidence

- Debug and release residual-mask tests: `cargo test -p tenferro-internal-ops residual_mask_detector_rejects_undeclared_input_access` and the matching `--release` command; debug passed and release ran zero tests as intended.
- Values-only and public linalg surface: `cargo test -p tenferro-linalg --test integration concrete_surface -- --nocapture` (13 passed).
- Scaled Faer LU regression: `cargo test -p tenferro-linalg --test integration full_piv_lu_solve_accepts_small -- --nocapture` (2 passed before the later surface additions; the full focused integration run is required before merge).
- Scoped faer capability: `cargo test -p tenferro-cpu faer_parallelism_capability_runs_inside_a_cpu_session -- --nocapture` (passed).
- Tutorial consumer checks: `cargo check --manifest-path docs/tutorial-code/Cargo.toml --bin direct_linalg_quickstart`, `--bin faer_interop`, and `--no-default-features --features cuda,cpu-faer --bin cuda_tutorial` (all passed).
- Documentation/skill consistency: `python3 scripts/check-doc-snippets.py --check` and `python3 scripts/check-agent-skills.py` passed after refreshing source-backed snippets.
- Release element-access benchmark (Rust 1.97.1, x86_64 Linux, one thread, 1 s warmup/2 s measurement/20 samples): checked vs unchecked direct slice was 3.839 µs vs 3.842 µs for 4096 random accesses; fixed-rank nested `get2` was 31.315 µs versus 3.775 µs for the expert unchecked slice loop. The benchmark also covered rank-3, iterator, and strided cases.
- Optimized codegen: `python3 scripts/check-storage-static-rank-codegen.py --report /tmp/ai-ready-static-rank.md --refresh` passed; both fixed-rank probes were present and their traversal loops contained no prohibited setup calls.

## Remaining verification before PR merge

Run the repository PR-fast gate, the deterministic repository-rules review,
full focused linalg/CPU/tutorial tests, docs-site checks, and the hosted GPU
lanes. Do not close or merge the issue batch until those exact-head checks and
PR status are green.
