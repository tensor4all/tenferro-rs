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

## CPU Threading Contract

- For faer-backed CPU ops, `CpuContext` is the single source of truth for thread-pool policy.
- Do not derive faer parallelism independently inside individual ops or helper functions.
- Execute faer-backed work only inside `ctx.install(...)` so the owned rayon context is preserved.
- Use `Par::Seq` for one-thread contexts and `Par::rayon(0)` for multi-thread contexts so faer follows the current `CpuContext`.

## Documentation Policy

### Source of Truth

- **Source code** is the source of truth for internal design (op catalog, backend contract, AD rules, compilation pipeline).
- **Online docs** are user-facing only — how to use the `tenferro` facade crate.
- **AGENTS.md** is the entry point for developers and AI agents. It contains pointers to source code locations.
- Do NOT duplicate source-code-level information in online docs. If it can be learned by reading the source, put a pointer instead of a copy.
- Development assumes AI agentic coding. Keep machine-readable sources (code + doc comments) authoritative.

### User-Facing Docs

- User docs target PyTorch/JAX users who interact with the `tenferro` facade crate.
- All imports must use `use tenferro::{...}` — never reference internal crates (`tenferro-tensor`, `tenferro-ops`, `computegraph`, etc.) in user-facing docs.
- Do NOT expose internal jargon (Fragment, StableHLO, ExecOp, ValRef, etc.) in user-facing pages.
- Provide PyTorch/JAX equivalents when introducing tenferro concepts.

### Doc Examples

- Doc examples (`/// # Examples`) must NOT use `ignore` or `no_run` attributes.
- Every example must compile AND run as a doctest.
- Use `compile_fail` only for examples that intentionally demonstrate compile errors.
- If an example cannot run as a doctest, refactor it until it can.

## Generic Over Scalar Type

- Use generic constructors with sealed traits instead of per-type functions.
- Bad: `TracedTensor::from_f64(...)`, `TracedTensor::from_f32(...)`, etc.
- Good: `TracedTensor::new<T: TensorScalar>(shape, data)` — type inference selects the variant.
- Sealed traits (`TensorScalar`, `PoolScalar`, etc.) restrict to supported dtypes (f64, f32, Complex64, Complex32).

## Public API Convention

- **Unary single-output ops**: methods on `TracedTensor` (e.g., `x.exp()`, `x.reshape(shape)`)
- **Binary single-output ops**: operator overloads where natural (`&a + &b`, `&a * &b`), methods otherwise (`a.dot_general(&b, config)`)
- **Multi-output ops**: free functions (e.g., `svd(&a)`, `qr(&a)`, `eigh(&a)`)
- **Linalg ops**: free functions (e.g., `solve(&a, &b)`, `cholesky(&a)`)
- **Einsum**: free function `einsum(engine, inputs, subscripts)`
- No `traced_` prefix on methods. `TracedTensor` methods are inherently traced.
