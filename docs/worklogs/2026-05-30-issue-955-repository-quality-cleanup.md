# Repository quality cleanup work log

Issue: <https://github.com/tensor4all/tenferro-rs/issues/955>

Child issues:

- <https://github.com/tensor4all/tenferro-rs/issues/951>
- <https://github.com/tensor4all/tenferro-rs/issues/952>
- <https://github.com/tensor4all/tenferro-rs/issues/953>
- <https://github.com/tensor4all/tenferro-rs/issues/954>

Date: 2026-05-30

## Session summary

This cleanup closed the remaining repository-quality stream by combining four
small, related changes:

- local lint/dead-code cleanup and reason comments
- focused argument objects for internal compiler helpers
- a real module split for the runtime dot-decomposition pass
- a narrow DRY refactor for same-shape traced unary wrappers
- a portability fix for the lightweight PR gate

The PR also added durable rules for reviewer-facing work logs. The goal is to
make future reviews read the implementer's recorded design context before
challenging abstraction, split, macro/codegen, or deferral choices.

## Context read

| Source | Why it was read | Decision impact |
| --- | --- | --- |
| `AGENTS.md` | Locate the agent/reviewer entry point and existing design-doc guidance. | Added review guidance there instead of burying it only in PR template text. |
| `REPOSITORY_RULES.md` | Find the durable tenferro-specific rules location. | Added wrapper DRY/codegen rules and work-log/design-record rules there. |
| `.github/pull_request_template.md` | Check where PR authors see required metadata. | Added a work-log link section and checklist item. |
| `docs/reference/repository-quality-cleanup.md` | Reuse the #955 quality cleanup policy from #950. | Added wrapper codegen criteria and cleanup work-log expectations. |
| `tenferro-runtime/src/compiler/mod.rs` | Inspect large-module boundaries and `too_many_arguments` sites. | Chose `DotDecomposer` as the split boundary and converted local helper state into named objects. |
| `tenferro-ad/tests/compiler_passes.rs` and `tenferro-ad/tests/compiler_passes/dot_decomposer_tests.rs` | Confirm existing coverage around compiler passes and dot decomposition. | Used existing dot-decomposer tests as behavior coverage for the split and argument refactor. |
| `tenferro-runtime/src/traced.rs` | Inspect duplicated public traced unary wrappers. | Chose a private same-shape unary helper while preserving explicit public methods and rustdoc. |
| `tenferro-runtime/src/tensor.rs`, `tenferro-runtime/src/typed_tensor.rs`, and `tenferro-runtime/src/traced_tensor.rs` | Check existing macro-based wrapper style in neighboring free-function surfaces. | Treated macros as valid prior art, but not the right first change for `TracedTensor` methods. |
| `tenferro-ad/tests/numpy_api.rs` | Locate public wrapper exposure tests. | Added a forwarding behavior test for traced unary methods. |
| `scripts/check-pr-fast.sh` | Run the local PR gate after committing the cleanup. | Replaced Bash 4-only `mapfile` usage with Bash 3.2-compatible read loops so the gate runs on macOS default Bash. |

## Decisions made

- **Work logs are separate from plans.** `docs/plans/` remains historical and
  may be stale. New PR/session rationale belongs in `docs/worklogs/`, while
  durable design intent belongs in `docs/design/`.
- **Reviews should use work logs as context.** `AGENTS.md` now tells reviewers
  to read linked work logs and design docs before challenging scope,
  abstraction, or design intent.
- **Dot decomposition is a real module boundary.** The dot-decomposition pass
  owns canonical dot layout analysis and instruction emission, and already had
  dedicated tests. Moving it to `compiler/dot_decomposer.rs` improves navigation
  without splitting by line count alone.
- **Compiler argument objects should be local and semantic.** `ConjSinkingState`,
  `DotDecomposeInput`, `OperandMeta`, `InstructionEmitter`, and
  `MergeReshapeSpec` replace long internal helper signatures without changing
  public APIs or backend behavior.
- **Wrapper DRY should keep public methods explicit.** `TracedTensor` unary
  methods now share `apply_same_shape_unary`, but public names and docs remain
  hand-written.
- **Local PR gates should run on the supported developer shell.**
  `check-pr-fast.sh` still uses Bash arrays, but no longer depends on Bash
  4-only `mapfile`, which is absent from macOS default Bash.

## Rejected or deferred alternatives

- **No broad workspace-wide `too_many_arguments` removal.** FFI, BLAS/LAPACK,
  backend trait, AD rule, and extension boundaries still keep explicit argument
  lists where that is the clearest contract.
- **No arbitrary large-file split.** GPU, linalg, tensor types, and registry
  files remain untouched because this PR did not expose a similarly low-risk
  responsibility boundary there.
- **No macro/codegen for `TracedTensor` methods in this PR.** Neighboring
  free-function modules use macros successfully, but `TracedTensor` methods
  carry public rustdoc and mixed operation-specific semantics. A private helper
  reduced real duplication without hiding public API behavior.
- **No wrapper-family rewrite across eager/traced/tensor/extension layers.**
  Those surfaces are not fully isomorphic. Future macro/codegen remains allowed
  only for a genuinely same-shaped family with tests and visible docs/features.
- **No deletion of feature-gated backend helpers.** Some `dead_code` allowances
  are dormant only under the currently compiled feature set. They were retained
  with reason comments instead of deleting feature-path diagnostics or provider
  entry points.

## Reference code

- Existing macro wrappers in `tenferro-runtime/src/tensor.rs`,
  `tenferro-runtime/src/typed_tensor.rs`, and
  `tenferro-runtime/src/traced_tensor.rs` show that macros are acceptable when
  the generated free-function family is same-shaped and remains visible at the
  invocation site.
- Existing compiler-pass tests in `tenferro-ad/tests/compiler_passes.rs` and
  `tenferro-ad/tests/compiler_passes/dot_decomposer_tests.rs` were the primary
  behavior reference for the dot-decomposition split.
- Existing `#[allow]` reason comments around ABI/backend boundaries were used
  as the style reference for retained production allowances.

## Verification performed

- `cargo fmt --all --check`
- `cargo test -p tenferro-ad --test numpy_api traced_unary_methods_forward_to_distinct_primal_ops`
- `cargo test -p tenferro-ad --test compiler_passes dot_decomposer`
- `cargo clippy -p tenferro-runtime --all-targets -- -D warnings`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- `cargo test --workspace --no-fail-fast`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`
- `python3 scripts/check-crate-boundaries.py`
- `python3 scripts/check-no-facade-crate.py`
- `python3 scripts/check-ad-boundaries.py`
- `python3 scripts/check-linalg-ad-boundaries.py`
- `cargo test --manifest-path ext/tropical/Cargo.toml --no-fail-fast`
- `bash scripts/check-pr-fast.sh --coverage-reviewed --test 'cargo test -p tenferro-ad --test numpy_api traced_unary_methods_forward_to_distinct_primal_ops' --test 'cargo test -p tenferro-ad --test compiler_passes dot_decomposer'`

## Remaining risk

- The compiler split is behavior-preserving and covered by existing pass tests,
  but it moved a large block of code. Review should check module visibility and
  that only `dot_decomposer` moved out of `compiler/mod.rs`.
- Retained lint allowances still exist at backend, FFI, AD, and test boundaries.
  They are intentional after this cleanup, but future work can continue reducing
  them one boundary at a time.
