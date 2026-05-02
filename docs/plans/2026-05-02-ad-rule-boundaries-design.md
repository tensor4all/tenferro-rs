# AD Rule Boundaries Dispatch Design

**Date:** 2026-05-02

**Status:** Proposed dispatch spec

## Issues

Primary:

- #772: BUG: TriangularSolve/FullPivLuSolve transpose rule drops A-cotangent
- #773: BUG: Pad missing transpose_rule, panics on VJP
- #777: BUG: 7 linalg ops missing transpose_rule, panics on VJP
- #783: AD: Slice/DynamicSlice/Concatenate/Reverse/Select/Clamp/Maximum/Minimum have no AD rules
- #787: BUG: transpose_scatter panics on symbolic shapes

## Goal

Make AD behavior explicit and correct at transpose-rule boundaries. Add rules
only when the repository oracle and finite-difference requirements can be met.

## Scope

This dispatch covers graph-level AD rules in `tenferro-ops/src/ad/` and any
minimal facade tests needed to exercise them.

It does not cover:

- primal CPU/GPU indexing implementation,
- linalg numerical kernel changes,
- dtype promotion policy,
- broad AD support for every missing op in one patch.

## Repository Contract

Before touching these files, reread `REPOSITORY_RULES.md`.

The source of truth is:

- `PrimitiveOp::linearize`,
- `PrimitiveOp::transpose_rule`.

Every new or changed rule must have a corresponding finite-difference
integration test. For linalg rules, prefer oracle families with both Torch
reference data and finite-difference checks when available.

Do not add or keep a mainline AD rule without the required oracle coverage.

## Acceptance Specification

### TriangularSolve and FullPivLuSolve

If `active_mask[0]` is true, the transpose rule must produce the A-cotangent
instead of silently returning `None`.

Expected mathematical direction:

- for `A @ X = B`, A-cotangent is proportional to `-ct @ X^H` or the equivalent
  orientation used by the existing dot conventions,
- for side/right variants, use the corresponding transposed orientation.

The implementation must respect existing adjoint, transpose, and conjugation
conventions in `tenferro-ops`.

### Pad

`Pad` transpose should crop/slice the cotangent back to the input region when
the padding configuration is statically representable by existing ops.

If dynamic or negative padding semantics cannot be represented correctly, return
a normal unsupported-AD error or stop before adding a partial rule.

### Missing linalg transpose rules

Do not implement all seven linalg transpose rules by default. Classify each op:

- implement only rules with clear math and oracle coverage,
- replace `todo!()` panics with explicit unsupported-rule errors where the
  runtime can represent that cleanly,
- otherwise leave a documented stop report.

### Indexing and piecewise ops

For Slice, DynamicSlice, Concatenate, Reverse, Select, Clamp, Maximum, and
Minimum:

- implement simple structural transpose rules only when correctness is clear and
  tests can cover them,
- do not implement non-smooth derivative rules for Maximum/Minimum/Clamp without
  a documented subgradient policy and tests.

### Symbolic shapes

`transpose_scatter` must not panic on symbolic shape dimensions. If inverse
slice sizes require concrete values, return a structured unsupported-shape error
or defer the rule for symbolic cases.

## Design

Use a two-stage dispatch:

1. eliminate silent wrong gradients and runtime panics where an explicit error
   path exists,
2. add a small number of correct transpose rules with tests and oracle coverage.

Do not let DeepSeek grow this into a broad AD project. Correctness beats issue
count here.

## Testing

Required tests:

- finite-difference coverage for every changed transpose rule,
- a regression showing active A in triangular solve no longer drops cotangent if
  implemented,
- a Pad VJP test if Pad transpose is implemented,
- a symbolic scatter case that no longer panics, or a test for the explicit
  unsupported error.

Run at least:

```bash
cargo test -p tenferro-ops ad
cargo test -p tenferro ad
cargo fmt --all --check
```

If oracle replay is touched, run the relevant oracle replay command documented
near the changed tests.

## Dispatch Prompt

```text
Implement the AD rule boundary dispatch from
docs/plans/2026-05-02-ad-rule-boundaries-design.md.

First reread REPOSITORY_RULES.md. Fix only AD transpose/linearize boundary
issues where correctness and test coverage are available. Do not add a mainline
AD rule without finite-difference or oracle coverage. Prefer explicit
unsupported-rule errors over todo!/panic paths when a correct rule is out of
scope. Stop and report any rule that needs new oracle families before it can be
supported.
```

## Review Checklist

- `REPOSITORY_RULES.md` was followed for every AD rule.
- No silent `None` cotangent remains for an active differentiable input in a
  changed rule.
- No new AD rule lacks finite-difference or oracle coverage.
- No `todo!()` is added on a reachable AD path.
- Linalg rules match existing adjoint/conjugation conventions.

## Stop Conditions

Stop and report if:

- a rule requires new oracle data that does not exist,
- a mathematically correct transpose cannot be expressed with existing ops,
- replacing `todo!()` with `Err` requires changing core AD trait signatures.
