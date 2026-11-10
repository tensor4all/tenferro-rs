# Public API grammar (#2552)

**[Open the illustrated proposal →](./api-grammar/proposal.html)** (Japanese, one page)

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
