# Repository Rules

## Public Surface Drift

- `README`, rustdoc, and examples must not claim capabilities beyond the current public surface.
- When the public API changes, check for stale names, stale capability claims, and deleted paths in `README`, rustdoc, and examples before considering the work complete.

## Oracle Gate

- Do not add or keep an AD `frule` or `rrule` in the mainline without a corresponding oracle family.
- Prefer oracle families with both Torch reference data and finite-difference checks.
- If a Torch reference is not available, a finite-difference-only oracle is acceptable.
- If no corresponding oracle exists yet, add it to `tensor-ad-oracles` before treating the rule as a supported mainline AD rule.

## Rule Source Of Truth

- `PrimitiveOp::linearize` and `PrimitiveOp::transpose_rule` (in `tenferro-ops/src/ad/`)
  are the semantic source of truth for first-order AD.
- These are graph-level rules that emit ops into a `FragmentBuilder`.
  `tidu::differentiate` calls `linearize`; `tidu::transpose` calls `transpose_rule`.
- Reference JAX's implementations (`jax/_src/lax/lax.py`, `jax/_src/lax/linalg.py`)
  when implementing new AD rules.

## AD Rule Coverage

- Every `linearize` / `transpose_rule` implementation must have a corresponding
  finite-difference integration test that verifies numerical correctness.
- For linalg ops, prefer oracle families with both Torch reference data and
  finite-difference checks when available in `third_party/tensor-ad-oracles/`.

## No Ad Hoc Fixes

- Do not add ad hoc fixes that violate DRY, KISS, or layering.
- Do not introduce compatibility shims, duplicated logic, or downstream reach-through into lower layers when the correct fix belongs in an existing seam or high-level API.

## Public API Convention

- **Unary single-output ops**: methods on `TracedTensor` (e.g., `x.exp()`, `x.reshape(shape)`)
- **Binary single-output ops**: operator overloads where natural (`&a + &b`, `&a * &b`), methods otherwise (`a.dot_general(&b, config)`)
- **Multi-output ops**: free functions (e.g., `svd(&a)`, `qr(&a)`, `eigh(&a)`)
- **Linalg ops**: free functions (e.g., `solve(&a, &b)`, `cholesky(&a)`)
- **Einsum**: free function `einsum(engine, inputs, subscripts)`
- No `traced_` prefix on methods. `TracedTensor` methods are inherently traced.
