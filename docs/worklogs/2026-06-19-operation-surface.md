# Operation Surface Cleanup

## Scope

Issue #1049 was resolved by making the core AD operation surface Rust-first:
`EagerTensor` and `TracedTensor` expose core operations as methods or associated
functions, while the old core module-function shims were removed.

## Decisions

- Core AD operations are canonical on tensor types. Single-output operations are
  methods, and operations without a natural receiver use associated functions.
- Compatibility shims were deliberately not kept. `tenferro_ad::eager_tensor`
  and `tenferro_runtime::traced_tensor` were deleted for the core operation
  surface.
- Non-AD concrete operations remain backend-explicit module functions under
  `tenferro_runtime::tensor` and selected `typed_tensor` wrappers.
- Extension operation families use crate-root extension traits instead of
  public `traced_tensor` / `eager_tensor` module namespaces, because Rust
  extension crates cannot add inherent methods to external tensor types.
- Eager broadcast planning now preserves user-facing tensor errors for shape
  mismatches instead of collapsing them into `Internal` errors.

## Rule And Checker Updates

- `REPOSITORY_RULES.md` records the method/associated-function rule for the core
  AD operation surface and rejects compatibility escape shims when API breakage
  is allowed.
- `docs/spec/operation-categories.md` is now the v0.1 operation-surface
  contract.
- `scripts/check-operation-categories.py` enforces Eager/Traced parity, removal
  of the old module surfaces, and stale live-doc references.
- CI runs the operation-surface checker with `--fail-on-findings`.

## Verification

- `cargo test --workspace`
- `python3 scripts/check-operation-categories.py --fail-on-findings`
- `python3 scripts/check-doc-snippets.py --check`
- `/opt/homebrew/bin/python3.12 scripts/check-guide-dependency-snippets.py`
- `/opt/homebrew/bin/python3.12 scripts/check-api-consistency.py --fail-on-findings`
- `python3 scripts/test-doc-consistency.py`

## Remaining Risk

GPU execution paths were not exercised on local hardware. GPU-gated tests remain
CI or hardware-runner work.
