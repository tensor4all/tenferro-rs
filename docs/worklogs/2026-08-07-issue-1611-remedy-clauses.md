# Work log: issue #1611 remedy clauses

## Context

Issue #1611 requires error messages to prescribe a reliable next action when
one exists, without changing error variants or adding machine-readable error
codes. The accepted plan limits this work to the structured-error rule, CPU
unsupported-dtype families, and the two tensor contiguity messages.

## References checked

- `REPOSITORY_RULES.md` Structured Error Classification
- `crates/tenferro-tensor-core/src/error.rs`
- `crates/tenferro-cpu/src/reduction.rs`
- `crates/tenferro-cpu/src/analytic.rs`
- `crates/tenferro-cpu/src/exec_session.rs` and `src/dot_runtime.rs`
- `crates/tenferro-internal-cpu-kernels/src/elementwise.rs`
- `docs/guides/troubleshooting.md`
- checked conversion rules in `crates/tenferro-tensor/src/validate/mod.rs`

## Implementation decisions

- Keep all existing error variants and classifications unchanged.
- List operation-specific supported dtypes instead of introducing a global
  dtype registry.
- Recommend `TensorOpsExt::convert` only for source/target pairs accepted by
  the existing checked promotion lattice. Bool and complex ordered/index
  cases list supported values without promising a semantic conversion.
- Keep the concrete tensor remedy callable for runtime tensors; eager and
  traced users retain their existing inherent `convert` methods.
- Align the existing CPU `Sign` capability descriptors with the already
  implemented C32/C64 dispatch so supported-dtype messages and capability
  queries agree.
- Use the existing named tutorial-snippet mechanism for troubleshooting
  examples; no second Markdown compiler or checker was added.
- Keep the high-level `TypedTensorView::as_slice` example keyed to its
  existing `view is not contiguous column-major` diagnostic, while the
  tensor-core `NonContiguousViewAsSlice` variant retains its separate
  `view is not slice-contiguous` remedy text and classification.

## Verification

- `cargo test -p tenferro-tensor-core --test core`
- `cargo test -p tenferro-cpu --lib`
- `cargo test -p tenferro-tensor error`
- `cargo test --workspace --release`
- `python3 scripts/check-doc-snippets.py --check`
- `python3 scripts/ci/run_profile.py fmt clippy`
- `python3 scripts/ci/run_profile.py docs`
- `cargo llvm-cov --workspace --exclude tenferro-tutorial-code --profile ci --json --output-path /tmp/coverage-1611-ci.json` plus `scripts/check-coverage.py` (191/191 files)
- `python3 scripts/repository-rules-review.py --base origin/main --head HEAD --output-json /tmp/repository-rules-review-1611.json` (pass, 0 findings)
- `cargo fmt --all --check`
- `git diff --check`

## Residual risks

Dtype remedies remain intentionally operation-specific. Messages do not claim
that conversions are lossless; callers must choose a target appropriate to the
operation and their numeric semantics.
