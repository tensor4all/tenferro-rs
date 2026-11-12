# Public API grammar (#2552)

**[Open the illustrated proposal →](./api-grammar/proposal.html)** (Japanese, one page)

**[Implementation instructions →](./api-grammar/implementation.md)** (Japanese): what to build and delete, in which order, and the definition of done.

The proposal for one call grammar across the whole public API
([#2552](https://github.com/tensor4all/tenferro-rs/issues/2552)), drawn for human readers:
the grammar strip, before/after examples for each layer, the seven rules and the mechanism
that holds each one, the trait matrix (31 → 12), the migration steps, and the decisions.

It reflects the decisions on the issue up to 2026-11-10, including the withdrawal of the
materialization policy and the decision not to require a public copy counter. The normative
rules for layout adaptation are R3 and R8 in
[api-consolidation-policy.md](./api-consolidation-policy.md); where the two differ, the
policy document and the issue thread win.

```text
recv.op(exec)
recv.op_with(operands…, &opts, exec)
recv.op_into(operands…, &opts, exec, dest)

session.op(&x, operands…)          // eager
traced.op_with(operands…, &opts)   // traced, then prepare → run
```

## Eager operations that need an engine take the session as receiver (#2602)

The eager spelling `session.op(&x, operands…)` covers every eager operation that
needs an engine, not only the forward operations of the registries:

- **AD.** `s.grad(&output, &wrt)`, `s.vjp(&output, &wrt, &cotangent)`,
  `s.jvp(&output, &wrt, &tangent)`, their `_optional` forms, `s.backward(&loss)`
  and `s.backward_with(&loss, &seed)`.
- **Value extraction.** `s.duplicate_value(&x)` (the one owned-copy spelling) and
  `s.detach(&x)`. A value moves between runtimes as an owned copy:
  `duplicate_value` in the source session, `constant_from` in the target
  session.
- **Leaf construction.** `s.constant_from(t)`, `s.constant_from_host(t)` and
  `s.variable_from(t)` read the input through the engine, so device-resident and
  strided inputs become leaves without a host round trip.

The engine enters once, through `EagerRuntime::with_eager_session(&mut engine,
|s| ..)`, and the session checks every eager operand and seed against its
runtime (`ContextMismatch`) before anything is materialized or a gradient slot
is written. No `EagerTensor`, `EagerRuntime` or eager extension method takes an
engine.

Operations that need no engine stay on the handle or the runtime: `x.grad()`
(slot lookup), `x.clear_grad()`, `x.into_value()` (consuming extraction without a
copy), `shape`/`dtype`, `no_grad`, cache controls, and the engine-free import of
an owned host tensor, `EagerRuntime::{constant_from, variable_from}`, which
rejects device and view inputs and points to the session constructors.

**Engine type.** The public eager boundary uses `&mut dyn ExecutionEngine`:
`with_eager_session` takes it and `EagerSession::engine()` lends it. AD needs
the runtime target and program preparation of `ExecutionEngine`, and
`ExecutionEngine: Engine`, so the lent engine also runs every concrete tensor
operation. There is one accessor; `backend_session()` is removed.

**Exceptions** are registry rows of the core registry: E11 (`_optional` result
forms and the seeded `backward_with`, whose seed is an operand rather than
options), E12 (engine-free values and the stored-gradient copy
`GradientValue::to_tensor(exec)`, which carries no eager-runtime identity and
follows the concrete grammar) and E13 (the session boundary, the `engine()`
accessor and the extension-authoring `apply_eager`).

**Enforcement.** The eager rows of the core `api_ops!` registry declare their
compiled signature (`signature eager: "..."`).
`scripts/check-api-registry-conformance.py` checks the compiled surface of the
default and CUDA profiles against them: a declared signature must exist, an
eager AD/value/leaf/accessor name on an eager receiver must be declared, and an
engine argument on an eager item must be an exception. CI runs it on the surface
compiled from the PR head inside the required `CI gate (PR workspace tests)`, so
regenerating `api/` does not make an undeclared spelling pass.
