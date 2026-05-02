# Docs And Einsum Cleanup Dispatch Design

**Date:** 2026-05-02

**Status:** Proposed dispatch spec

## Issues

Primary:

- #742: docs: realign GPU design docs with cubecl reality
- #746: Cleanup: vestigial ShapeGuardContext / global_metadata APIs
- #748: Document einsum repeated-label semantics

Related if the same files are already being edited:

- #798: BUG: multiple .unwrap() panics in einsum builder label lookups
- #799: BUG: debug_assert in contraction_cost compiles away in release
- #800: DESIGN: Subscripts::parse silently strips parentheses
- #801: BUG: NestedEinsum::parse_group sorts intermediate output labels
- #808: BUG: error type erasure in einsum planning
- #792: BUG: DimExpr::Sub can underflow usize
- #820: DOC: Stale OutFloat doc reference and minor test quality issues

## Goal

Bring docs and small cleanup surfaces back in line with current implementation
without changing historical plan records or starting a broad einsum parser
redesign.

## Scope

This dispatch covers:

- `docs/design/gpu-backend-design.md`,
- `docs/design/einsum.md`,
- rustdoc or guide text for repeated-label einsum semantics,
- vestigial metadata APIs after lazy lookup if cleanup stays narrow,
- small einsum safety fixes only when directly adjacent to docs/tests being
  updated.

It does not cover:

- changing files under `docs/plans/` other than this dispatch spec,
- implementing new GPU functionality,
- redesigning NestedEinsum parsing,
- eliminating the global metadata leak (#745),
- broad error-type migration for einsum planning.

## Acceptance Specification

### GPU design docs

Forward-looking GPU design docs must describe the CubeCL-based implementation:

- `tenferro-tensor/src/cubecl/` is the active GPU backend under the `cubecl`
  feature,
- CUDA execution flows through CubeCL/CubeCL-CUDA and supporting CUDA libraries,
- ROCm remains a stub unless explicitly implemented later,
- deleted `CudaBackend` direct-backend architecture must not be presented as the
  current target.

Historical references are acceptable only when clearly marked as historical.

### Repeated-label einsum docs

User-facing docs and rustdoc should state:

- repeated labels in one input select a diagonal,
- repeated labels omitted from output are reduced after diagonal extraction,
- repeated labels in the output perform diagonal embedding,
- public einsum APIs accept repeated labels,
- strict binary/GEMM lowering may decline these patterns as a fast path and fall
  back to the general path.

Examples should include:

- `ii->`,
- `ii->i`,
- `i->ii`,
- one higher-rank example such as `iij->ij` or `iii->i`.

### Metadata cleanup

If `snapshot_global_metadata` is unused outside tests, remove or demote it to a
test-only helper. `refresh_global_metadata` should either become a documented
no-op or be removed if all callers can be updated safely.

This dispatch must not attempt to solve the unbounded registry leak.

### Eager einsum safety

If editing einsum docs reveals nearby `unwrap()` or `debug_assert` failures in
current non-historical code, fix only the narrow cases with obvious `Result`
propagation. Parser-order design issues should be documented for a later
dispatch unless tests already define the intended behavior.

## Design

Separate user-facing docs from internal design pointers:

- user docs use `tenferro::{...}` imports only,
- internal design docs may point to implementation files,
- historical `docs/plans/` files remain unchanged.

For cleanup, prefer deletion of vestigial APIs over compatibility shims unless
there is a current public caller.

## Testing

Required checks:

```bash
cargo test -p tenferro-einsum
cargo test -p tenferro --doc
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
cargo fmt --all --check
```

If only Markdown files changed and no rustdoc examples changed, explain which
code tests were skipped and why.

## Dispatch Prompt

```text
Implement the docs and einsum cleanup dispatch from
docs/plans/2026-05-02-docs-einsum-cleanup-design.md.

Update current design docs and user-facing repeated-label einsum docs. Do not
edit historical docs/plans records. Keep metadata cleanup narrow and do not
solve the global metadata leak. Fix only adjacent obvious einsum unwrap/assert
issues if tests and local error propagation are straightforward; otherwise
document them for a later dispatch.
```

## Review Checklist

- `CudaBackend` is not described as the current GPU backend target.
- Repeated-label semantics are explicit, not only implicit in examples.
- User-facing docs import from `tenferro`, not internal crates.
- `docs/plans/` historical records are not rewritten.
- Metadata cleanup does not change the leak design problem.

## Stop Conditions

Stop and report if:

- docs accuracy requires resolving an implementation ambiguity,
- metadata API removal breaks public or downstream callers,
- NestedEinsum parser fixes require a broader design decision.
