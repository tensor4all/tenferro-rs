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
  are the semantic source of truth for AD rules.
- These are graph-level rules that emit ops into a `FragmentBuilder`.
  `tidu::differentiate` calls `linearize`; `tidu::transpose` calls `transpose_rule`.
- Reverse-mode support is not always a direct `transpose_rule` arm on the
  primal op. Some ops are supported by first applying `linearize` and then
  transposing the emitted linear primitive graph. Before filing or closing an
  AD support issue, check the machine-readable AD support manifest in
  `tenferro-ops/src/ad/support.rs`; `SupportedViaLinearize` means a missing
  direct transpose arm is intentional.
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

## Complexity Budget

- Do not introduce accidental `O(n^2)` behavior in graph construction,
  metadata propagation, key hashing/equality, compilation, or execution
  scheduling. If a quadratic algorithm is intentional, document why the input
  size is bounded or why the tradeoff is acceptable.
- Avoid repeatedly cloning, hashing, formatting, or scanning whole graph
  histories, metadata scope lists, tensor input maps, or structural keys inside
  per-node/per-op loops. Prefer stable IDs, interning, cached fingerprints with
  exact equality checks, or persistent/shared data structures.
- When optimizing compiler or graph-build overhead, measure scaling across
  increasing graph sizes, not only one fixed benchmark case.

## Cache Ownership

- Long-lived runtime/compiler caches must be owned by `Engine` or another
  explicit top-level runtime object, not hidden in thread-local/global state or
  buried in backend internals.
- Every cache must have a bounded default, a user-facing way to configure that
  bound, and a user-facing way to clear it.
- Backend resource pools such as buffer pools may live on the backend, but they
  still need explicit limit/clear controls and documentation.
- Do not add a new cache without documenting its owner, lifetime, default
  capacity, memory behavior, and clear/configuration path.

## CPU Threading Contract

- For faer-backed CPU ops, `CpuContext` is the single source of truth for thread-pool policy.
- Do not derive faer parallelism independently inside individual ops or helper functions.
- Execute faer-backed work only inside `ctx.install(...)` so the owned rayon context is preserved.
- Use `Par::Seq` for one-thread contexts and `Par::rayon(0)` for multi-thread contexts so faer follows the current `CpuContext`.

## GPU Backend Contract

- Before touching CubeCL/GPU backend code, read
  [`docs/design/gpu-backend-design.md`](docs/design/gpu-backend-design.md).
- That document is the developer-facing source for CubeCL kernel ownership,
  runtime shape/stride metadata conventions, launch configuration rules, and
  device transfer behavior. Any change to those conventions must update that
  document in the same PR.

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

### User-Facing Code Snippet Consistency

- Non-trivial user-facing code snippets must have an executable source of truth:
  prefer including a checked example, test, or doctest instead of copying code
  into Markdown by hand.
- If a Markdown page must contain a copied snippet, add an automated sync or
  extraction check that fails when the snippet drifts from the executable
  source.
- Guide code that demonstrates a workflow should compile in CI. When runtime
  execution requires special hardware or external libraries, CI must still
  compile-check the example with the required feature flags, and the guide must
  document the command that runs the example on a correctly configured machine.
- CPU and GPU workflow examples should include meaningful assertions on shapes,
  dtypes, or values whenever the result is deterministic. Avoid examples that
  only print output unless the output itself is the documented behavior.
- GPU quickstart examples must use the same executable source as the guide,
  explicitly show upload/download boundaries, and assert downloaded CPU results
  for at least one supported operation.

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
