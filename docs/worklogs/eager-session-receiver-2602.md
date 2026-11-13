# Eager AD, value and leaf entry points on `EagerSession` (#2602)

Durable rule: [api-grammar.md](../design/api-grammar.md), section "Eager
operations that need an engine take the session as receiver"; removals and their
evidence: [api-consolidation-policy.md](../design/api-consolidation-policy.md).

## Decisions

- `ExecutionEngine` is the one engine type of the public eager boundary.
  `with_eager_session` already took it, AD needs its target and preparation, and
  `ExecutionEngine: Engine` covers concrete work through `s.engine()`;
  `backend_session()` (the `Engine` projection of the same object) is removed.
- Only the entry points moved. The session methods call the unchanged
  `semantic_eager_vjp_many` / `semantic_eager_jvp_optional` /
  `backward_from_seed`; the new code is the order of checks: every operand and
  seed is checked against the session's runtime first, then the non-scalar
  check, then seed materialization, then gradient-slot writes.
- `backward_with(&loss, &seed)` keeps its name as specified on the issue. Under
  the #2552 rule `_with` means explicit options, so it is registered as
  exception E11 together with the `_optional` result forms rather than as an
  options form.
- `detach_into` is removed rather than moved: it was `to_tensor` plus an
  engine-backed `new_leaf`, which is now spelled `duplicate_value` in the source
  session and `constant_from` in the target session. A session method that
  accepts a foreign-runtime operand would have been the one exception to the
  runtime check.
- Leaf construction keeps two documented paths: the engine-free
  `EagerRuntime::{constant_from, variable_from}` (owned host tensors, rejects
  device and view inputs) and the engine-backed session constructors (device
  resident and strided inputs, plus `constant_from_host`). The `EagerTensor`
  constructors forwarded to the same engine-free helper and were removed.
- `GradientValue::to_tensor(exec)` stays: a stored gradient carries no
  eager-runtime identity, so it follows the concrete grammar (E12).
  `extension::apply_eager(exec, ..)` stays as the extension-authoring entry
  (E13) until that facade is revised.
- Enforcement reads the registry: the rows of the `eager_ad` `api_ops!`
  registry in `tenferro-ad` (not the core registry, which `tenferro-runtime`
  owns and which must not name AD types)
  declare their compiled signature (`signature eager:`), and
  `scripts/check-api-registry-conformance.py` checks the compiled default and
  CUDA surfaces against the signatures and exception symbols. It runs on the
  surface compiled from the PR head inside a job the required
  `CI gate (PR workspace tests)` depends on. The `placement` job that ran the
  snapshot check was not a required check, so the snapshot check moved with it.

## Verification conclusions and constraints

- The checker rejects the `origin/main` surface (130 findings) and passes the
  migrated one; its self-test reintroduces each removed shape into a copy of the
  regenerated snapshots (default and CUDA profile) and requires a failure.
- Runtime contracts (foreign operands and seeds, check-before-seed, inactive
  inputs, higher order, accumulation, seeded backward, cross-runtime copy, leaf
  paths) are pinned by `tenferro-ad` `eager_session_ad_contract`; compile
  contracts by the `eager_backend_capability_contract` trybuild cases, which
  the nextest workspace lanes skip, so the `api-surface` job runs them with
  plain `cargo test`.
- An independent review found checker bypasses (an excepted name with an
  engine argument, engines as generic bounds or concrete types, copy aliases
  outside the name vocabulary). E11–E13 now list exact signatures, engine
  detection reads the whole rendered line, and copies are recognized by shape;
  each bypass is a self-test case. A renamed re-export is still checked under
  the type's own name, and a blanket `impl<T> Trait for T` is not attributed to
  the eager receivers.
- The CUDA call sites were migrated and compile with the CUDA feature set, but
  were not executed on this host (no GPU); the device-resident leaf path is
  exercised by the existing CUDA tests.
- The removed tensor methods accepted any `&mut dyn Engine`; the session path
  goes through `with_eager_session`, which registers the engine kind on first
  use (CPU and CUDA can both be registered on one runtime) and rejects only an
  engine kind the eager runtime cannot register. Every migrated call site passes
  the CPU or CUDA engine it passed before.
