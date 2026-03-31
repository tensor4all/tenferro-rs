# Repository Rules

## Public Surface Drift

- `README`, rustdoc, and examples must not claim capabilities beyond the current public surface.
- When the public API changes, check for stale names, stale capability claims, and deleted paths in `README`, rustdoc, and examples before considering the work complete.

## Linearized Seam Coverage

- If `LinearizedOp::jvp` or `LinearizedOp::vjp` is not a thin delegation to the existing `frule` or `rrule`, add a focused seam test.
- The seam test must exercise the runtime packaging that the rule-math tests do not cover, such as saved linearization, schema, optional tangents/cotangents, or multi-output packaging.

## No Ad Hoc Fixes

- Do not add ad hoc fixes that violate DRY, KISS, or layering.
- Do not introduce compatibility shims, duplicated logic, or downstream reach-through into lower layers when the correct fix belongs in an existing seam or high-level API.
