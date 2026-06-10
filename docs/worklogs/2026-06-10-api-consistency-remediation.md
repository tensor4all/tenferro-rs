# API Consistency Remediation Work Log

## Summary

This cleanup made the release-freeze API convention audit enforceable for the
first detected naming findings.

## Context Read

- `AGENTS.md`
- `REPOSITORY_RULES.md`
- shared tensor4all agent rules
- `docs/spec/api-conventions.md`
- `docs/design/api-and-convention-freeze.md`
- `scripts/check-api-consistency.py`
- `/tmp/tenferro-api-consistency.md`
- `crates/tenferro-runtime/src/ad_support.rs`
- `crates/tenferro-ad/src/traced.rs`
- `crates/tenferro-internal-ops/src/std_tensor_op.rs`
- `crates/tenferro-internal-ops/src/tests/std_tensor_op_tests.rs`

## Decisions

- Kept `traced_tensor` as a documented namespace paired with `eager_tensor`.
- Treated `traced_` prefixes on public function names as findings.
- Renamed `ad_support` bridge helpers because the module name already supplies
  the traced AD bridge context.
- Replaced dtype-specific `StdTensorOp::constant_*` constructors with one
  sealed generic constructor.
- Added `docs/spec/api-conventions.md` as the fixed source of truth for the
  checker rules so future changes do not rely on script behavior alone.

## Verification

- `python3 -m py_compile scripts/check-api-consistency.py`
- `python3 scripts/check-api-consistency.py --output /tmp/tenferro-api-consistency.md`
- `python3 scripts/check-api-consistency.py --fail-on-findings`
- `python3 scripts/check-doc-snippets.py --root-dir . --check`
- `cargo fmt --all --check`
- `cargo test -p tenferro-internal-ops`
- `cargo test -p tenferro-runtime traced_tensor`
- `cargo test -p tenferro-ad traced`
- `cargo test -p tenferro-ad compiler_wiring`
- `cargo test -p tenferro-ad --test compiler_wiring`
- `cargo test -p tenferro-linalg`

## Residual Risks

The concept-family matrices still need human triage for behavior-level
differences such as eager vs traced error surfaces, CPU vs GPU unsupported
paths, and view vs materializing operation boundaries.
