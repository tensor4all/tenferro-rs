# Eager semantic metadata-scope retention

## Session summary

Issue [#1700](https://github.com/tensor4all/tenferro-rs/issues/1700) was isolated while validating tensor4all-rs #623 against tenferro commit `8a8196a95363158f147b1feff2bc3b2d4bc4d267`. A mixed-dtype eager operation fails during deferred AD compilation when promotion casts a non-leaf raw semantic trace that depends on a temporary untracked constant.

## Context reviewed

- `AGENTS.md`, `CONTRIBUTING.md`, `REPOSITORY_RULES.md`
- shared tensor4all repository, performance, docs/tests, and Rust numerical rules
- `ai/contribution-workflows/bugfix-pr.md`
- `ai/contribution-workflows/repository-remediation.md`
- `crates/tenferro-ad/src/eager.rs`
- `crates/tenferro-ad/src/eager_exec.rs`
- `crates/tenferro-runtime/src/ad_support.rs`
- `crates/tenferro-runtime/src/metadata.rs`
- `crates/tenferro-runtime/src/traced.rs`
- eager promotion integration tests
- design records for #1692 and #1698

## Reproduction and classification

Classification: **Auto Fix**. Existing eager AD behavior requires graph-owned dependencies to survive temporary Rust values; no public API or AD semantics change is needed.

The issue reproducer combines a tracked `F64` value, a `pow` with a temporary untracked exponent, and a later mixed `F64 * C64`. Either operation alone succeeds. Their composition fails because the semantic promotion `Convert` owns metadata scopes that the following raw carrier discards.

## Decision

Retain the pointer-deduplicated metadata-scope histories of the final semantic inputs in each raw eager output. Runtime metadata ownership performs a persistent private-chain merge in a hidden carrier-construction helper extending the existing raw-append seam; the eager layer delegates after selecting its final semantic inputs. `RawAppend` and `TracedTensorParts` keep their existing public hidden layouts. Metadata materialization deduplicates both scopes and shared chain nodes.

Rejected downstream lifetime workarounds, primal materialization, duplicate raw cast assembly, eager-local scope deduplication, scope-vector materialization at the per-operation parts boundary, and a source-breaking public transfer field. See `docs/design/eager-semantic-metadata-scope-retention.md`.

## Review gates

- Pre-implementation design review round 1 (`reviewer-gpt`): **Findings**. It approved the root-cause and ownership direction but rejected per-operation vector materialization as potentially quadratic and required concatenate plus chain-scaling coverage.
- Pre-implementation design review round 2 (`reviewer-gpt`): **Findings**. It approved persistent complexity and expanded coverage, but identified replacing `TracedTensorParts::metadata_scopes` as a source-breaking change to a public hidden struct.
- Pre-implementation design review round 3 (`reviewer-gpt`): **Correct-to-implement**. It confirmed correctness, source compatibility, layering, persistent-chain complexity, and the expanded test plan.
- Post-implementation full-diff review (`reviewer-gpt`): **Correct-to-merge**. It reported no Critical, Important, or Minor findings after reviewing the complete 546-line patch, all new files, source context, and verification evidence.

## Implementation

- Added persistent `MetadataScopeChain::merge` with empty/single-parent fast paths.
- Added node-identity deduplication during metadata-scope materialization.
- Added `extension::append_raw_eager_outputs`, which keeps the private chain in `tenferro-runtime`, shares one merged chain across outputs, and leaves `RawAppend` / `TracedTensorParts` source layouts unchanged.
- Replaced eager-layer manual carrier assembly with the runtime helper after semantic promotion and exactification.
- Added runtime chain-scaling, scope/node deduplication, multi-output sharing, and RAII lifetime tests.
- Added mixed-promotion backward/VJP/JVP and concatenate exactification lifetime regressions.

## Verification

Passed:

- `cargo test -p tenferro-runtime metadata_scope -- --nocapture` (4 passed)
- `cargo test -p tenferro-runtime raw_eager_outputs_share_and_retain_helper_metadata_scopes -- --nocapture` (1 passed)
- `cargo test -p tenferro-ad temporary_constants_before_mixed_promotion_retain_metadata_for_derivatives -- --nocapture` (1 passed)
- `cargo test -p tenferro-ad temporary_exactified_concatenate_input_retains_metadata_for_vjp -- --nocapture` (1 passed)
- `cargo test -p tenferro-runtime` (968 passed)
- `cargo test -p tenferro-ad` (582 passed)
- `cargo fmt --all`
- `git diff --check`
- `bash scripts/check-pr-fast.sh --coverage-reviewed --test 'cargo test -p tenferro-ad temporary_constants_before_mixed_promotion_retain_metadata_for_derivatives'`
- `python3 scripts/repository-rules-review.py --base origin/main --worktree --dry-run --llm-skipped-reason 'local deterministic review' --output-json /tmp/tenferro-1700-rules-preview.json` (pass; expected LLM-skipped warning only)

The deterministic rules review was rerun against the committed head and passed with the expected LLM-skipped warning only.

## Coverage impact

Reviewed. The new tests directly exercise every added merge/materialization branch, shared-node and shared-scope deduplication, multi-output chain sharing, helper-scope RAII lifetime, promotion retention through all eager derivative entry paths, and concatenate exactification retention. No code or tests were removed, and no coverage threshold changed.

## Remaining risks

Hosted CI remains responsible for complete workspace, backend, feature, GPU, and coverage matrices. No local correctness or review blocker remains.
