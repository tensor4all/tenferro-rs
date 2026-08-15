# Issue #1692: eager concatenate and mixed stack verification

## Scope

This change implements the approved eager-only fix, integration coverage, and
this curated worklog. No dependency, tolerance, branch, commit, or push was
changed here.

## Source and design review

Reviewed:

- issue [#1692](https://github.com/tensor4all/tenferro-rs/issues/1692) and
  `docs/design/eager-concatenate-shape-recording.md`;
- eager recording in `crates/tenferro-ad/src/eager.rs`, including
  `record_semantic_eager_outputs`, the semantic VJP/JVP execution path, and
  prepared-derivative cache keys;
- `crates/tenferro-ad/src/shape_packing.rs` (`stack` lowers reshape plus
  concatenate);
- `crates/tenferro-ad/src/semantic_transform/core_structural.rs` (concatenate
  zero-tangent and VJP slice/offset rules);
- `crates/tenferro-ad/src/transform_cache.rs` (bound-shape cache metadata);
- the existing eager structural tests and
  `primitive_ops::traced_stack_rejects_empty_mismatched_invalid_axis_and_symbolic_shapes`.

The production path shape-specializes only deferred eager concatenate inputs:
non-concrete semantic carriers receive an exact-shape semantic reshape before
concatenate. Concrete eager values are unchanged, and the reshape adds no
runtime tensor copy or kernel. Stack consequently receives the same behavior
through its existing reshape/concatenate composition.

## Review gates and rejected scope

The reviewer-gpt pre-implementation gate ran in three rounds. Round 1 rejected
a broad core-constraint design because it omitted traced/import/staging owners.
Round 2 required accurate existing-materialization wording, cache-identity
coverage, direct JVP, and higher-order coverage. Round 3 returned
**Correct-to-merge**, authorizing implementation of the narrowed eager-only
design.

Post-implementation reviewer-gpt round 1 found the implementation and
numerical/AD behavior sound, but returned **Not Correct-to-merge** because this
worklog described only the test-file coverage impact and had not recorded the
post-implementation round. The coverage record below was corrected.
Post-implementation round 2 returned **Correct-to-merge** on the corrected full
diff.

Rejected or deferred alternatives were:

- broad core symbolic concatenate constraints, which would require changes to
  shape analysis, graph import/remapping, staging, and strict inference;
- concrete lazy constants only, which would not repair direct concatenate of
  distinct tracked leaves;
- downstream tensor4all workarounds using temporary tracked leaves or pad/add
  substitutes, which add copies/kernels at the wrong layer; and
- skipping concatenate equality validation, which would weaken reusable traced
  shape safety.

## Test coverage added

`crates/tenferro-ad/tests/integration/ad_structural_primitives.rs` now uses a
small generic exact-value helper to cover:

- direct concatenate of separately tracked `[2, 1]`, `[2, 2]`, and `[2, 1]`
  inputs with unequal concatenate-axis extents, exact weighted VJPs for every
  input, and an exact JVP oracle;
- the same direct concatenate VJP/JVP contract for `f64` and `Complex64`;
- mixed tracked/untracked direct concatenate in both operand orders, checking
  the tracked cotangent slice after the inactive operand's offset;
- mixed stack in both operand orders and insertion axes `0` and `-1`, with
  exact primal, tracked VJP, inactive-edge, and JVP assertions; one complex
  stack case also checks exact complex cotangent placement;
- one mixed-stack VJP-to-JVP higher-order composition (`2*x` then `2*tangent`);
- one shared-runtime cache-isolation case with compatible output shapes but
  swapped concatenate-axis extents, checking both exact second-shape values and
  two prepared derivative entries.

The existing traced symbolic rejection test was not rewritten.

## Verification

- `cargo fmt --all` — passed.
- `git diff --check` — passed.
- `cargo test --release -p tenferro-ad --test integration eager_concatenate_semantic_vjp_accepts_distinct_tracked_shapes` — passed (1 test, 342 filtered).
- `cargo test --release -p tenferro-ad --test integration eager_concatenate_semantic_vjp` — passed (2 tests, 341 filtered).
- `cargo test --release -p tenferro-ad --test integration eager_concatenate_mixed_activity_preserves_tracked_gradients_and_offsets` — passed (1 test, 342 filtered).
- `cargo test --release -p tenferro-ad --test integration eager_mixed_stack_vjp_and_jvp_cover_orders_and_axes` — passed (1 test, 342 filtered).
- `cargo test --release -p tenferro-ad --test integration eager_mixed_stack_vjp_then_jvp_remains_composable` — passed (1 test, 342 filtered).
- `cargo test --release -p tenferro-ad --test integration eager_concatenate_cache_isolated_for_compatible_shapes` — passed (1 test, 342 filtered).
- `cargo test --release -p tenferro-ad --test integration traced_stack_rejects_empty_mismatched_invalid_axis_and_symbolic_shapes` — passed (1 test, 342 filtered).
- `cargo test --release -p tenferro-ad --test integration` — passed (343 tests).
- `bash scripts/check-pr-fast.sh --coverage-reviewed --test 'cargo test -p tenferro-ad --test integration eager_concatenate_semantic_vjp_accepts_distinct_tracked_shapes'` — passed, including root and standalone-extension formatting/clippy gates.
- `cargo test --doc --release --workspace` — passed (1,677 tests).
- `cargo doc --workspace --no-deps` — passed with four pre-existing broken-link warnings outside this diff.
- `python3 scripts/repository-rules-review.py --base origin/main --worktree --dry-run` and `python3 scripts/test-repository-rules-review.py` — passed.
- `cargo nextest run --release --workspace -E 'not test(residual_mask_detector_rejects_undeclared_input_access)' --no-fail-fast` — passed (3,042 tests; 131 skipped).
- The unfiltered release workspace command is blocked on the unchanged `origin/main` test tracked as [#1694](https://github.com/tensor4all/tenferro-rs/issues/1694): a `#[should_panic]` release test expects a debug-only assertion. Its isolated release reproduction fails, while its debug build passes.

## Coverage impact and residual risk

Production coverage was reviewed for both changed Rust source files:

- `crates/tenferro-ad/src/eager.rs`: the new concatenate-only branch is covered
  by direct concatenate and stack, tracked-only and mixed-activity operands,
  both inactive operand orders, axes `0`/`-1`, f64/Complex64, VJP/JVP,
  higher-order composition, and cross-shape cache isolation. The full 343-test
  integration target also exercises existing non-concatenate recording.
- `crates/tenferro-runtime/src/ad_support.rs`: only explanatory comments changed;
  deferred semantic analysis behavior is unchanged.

No production coverage threshold, tolerance, or coverage configuration was
changed. The new tests use numerical whole-value assertions rather than
shape-only smoke checks, and existing eager and traced tests remain in the same
integration target.

The residual limitation is intentional: unresolved symbolic traced
`concatenate`/`stack` behavior remains rejected and is not expanded by #1692.
A future core symbolic-shape design must address that separately.
