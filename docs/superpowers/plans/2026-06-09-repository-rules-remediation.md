# Repository Rules Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or inline execution with TDD checkpoints to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix repository-rule violations found in the whole-code audit, excluding the full-pivot LU AD oracle-family gap that requires upstream `tensor-ad-oracles` work and should be tracked as a separate issue.

**Architecture:** Keep public APIs intentionally narrow, make extension/runtime ownership explicit, move tests to the owning modules, and replace hidden fallback/copy/cache behavior with explicit contracts and verification.

**Tech Stack:** Rust workspace, Cargo tests/doctests, Python repository contract scripts, Markdown docs.

---

### Task 1: Extension dispatch and public API cleanup

**Files:**
- Modify: `crates/tenferro-ad/src/eager_exec.rs`
- Modify: `crates/tenferro-ad/src/lib.rs`
- Modify: `crates/tenferro-runtime/src/compiler/mod.rs`
- Modify tests that currently reach through public internals.

- [x] Add failing regression coverage for missing registered extension runtime behavior.
- [x] Make eager extension dispatch fail explicitly when a runtime owner exists but the family is not registered.
- [x] Remove or privatize public AD/runtime internals that are test/cache/lowering details.
- [x] Move compiler-pass tests to the owning crate or through owner-scoped APIs.

### Task 2: CPU materialization, indexing, and threading contracts

**Files:**
- Modify: `crates/tenferro-cpu/src/context.rs`
- Modify: `crates/tenferro-cpu/src/lib.rs`
- Modify: `crates/tenferro-cpu/src/indexing.rs`
- Modify: `crates/tenferro-cpu/src/structural.rs`

- [x] Add source-contract or behavior tests for no hidden view materialization in generic CPU read paths.
- [x] Replace repeated tensor-loop index decomposition with incremental/validated layout traversal where practical.
- [x] Route faer work through `CpuContext::install(...)` and use `Par::rayon(0)` for multithread contexts.
- [x] Convert `tril`/`triu` structural loops to the accepted strided-kernel path or document/rename an explicit dense boundary.

### Task 3: Linalg API, AD validation, and faer allocation fixes

**Files:**
- Modify: `crates/tenferro-linalg/src/backend.rs`
- Modify: `crates/tenferro-linalg/src/cpu/linalg/faer_linalg.rs`
- Modify: `crates/tenferro-linalg/tests/traced_ad_explicit.rs`
- Modify linalg callers/tests affected by `_view` API naming.

- [x] Rename allocation/execution APIs that currently use `_view`, with no backward-compatibility shim.
- [x] Add finite-difference or residual checks to linalg AD tests that currently check only shape/finite values.
- [x] Reuse faer scratch/output buffers across batched loops where the operation boundary permits it.
- [x] Keep `faer_linalg.rs` unsplit after the changes remained concentrated around scratch-reuse helpers.

### Task 4: GPU cache and narrow interop API

**Files:**
- Modify: `crates/tenferro-gpu/src/cubecl/mod.rs`
- Modify: `crates/tenferro-gpu/src/cubecl/runtime.rs`
- Modify: `crates/tenferro-gpu/src/lib.rs`
- Modify: `crates/tenferro-linalg/src/gpu/linalg.rs`
- Modify: `docs/design/gpu-backend-design.md`

- [x] Add failing tests for CubeCL extension cache capacity, stats, clear, and retained-byte accounting.
- [x] Give the CubeCL extension cache bounded default capacity and user-facing controls/stats.
- [x] Replace raw CubeCL handle exposure used by linalg with the narrowest backend-owned interop API.
- [x] Update GPU backend design docs for any cache or interop contract change.

### Task 5: Docs, snippets, rustdoc examples, and test organization

**Files:**
- Modify: `README.md`
- Modify: `docs/guides/tenferro-fft.md`
- Modify: `docs/getting-started/core-concepts.md`
- Modify: `scripts/check-doc-snippets.py`
- Modify: `crates/tenferro-runtime/src/segment.rs`
- Modify: `crates/tenferro-cpu/src/elementwise.rs`
- Create module-local test files as needed.

- [x] Add executable snippet-source coverage for README/core concepts drift.
- [x] Fix FFT traced guide registration example.
- [x] Remove internal crate names from user-facing docs/rustdoc.
- [x] Add missing runnable rustdoc examples for public APIs touched by the audit.
- [x] Move large inline unit-test blocks to module-local `src/**/tests/*.rs` files.

### Task 6: Work log, verification, PR, and follow-up issue

**Files:**
- Create: `docs/worklogs/2026-06-09-repository-rules-remediation.md`

- [x] Record decisions, deferred oracle issue, and verification in a work log.
- [x] Run focused tests after each task and full workspace verification before commit.
- [x] Open a PR that links the work log.
- [x] Create or link a GitHub issue for the full-pivot LU AD oracle-family gap.
- [x] Fix the initial PR coverage failure in `tenferro-cpu/src/elementwise.rs`.
- [ ] Monitor PR checks, fix remaining failures, and merge when green.
