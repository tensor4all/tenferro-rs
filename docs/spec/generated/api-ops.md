# Public operation registry (generated)

This page is generated from the operation registries (`api_ops!` in
`crates/<family>/src/api/registry.rs`; core in `tenferro-runtime`). Do not
edit it by hand: each family's section is rewritten by
`UPDATE_API_DOCS=1 cargo test -p <family> api_docs_are_current`, and the same
test fails when the committed section differs from the registry.

## Call grammar

```text
recv.op(operands…, exec)
recv.op_with(operands…, &opts, exec)
recv.op_into(operands…, &opts, exec, dest)

session.op(&x, operands…)          // eager: the session is the receiver
traced.op_with(operands…, &opts)   // traced: no exec, run through prepare → run
```

## Suffixes

| Suffix | Meaning |
|---|---|
| (none) | Default options; returns a new result. |
| `_with` | Explicit `XxxOptions`. |
| `_into` | Writes into a caller-provided `Dest` (`Overwrite`, `Accumulate` or `Scaled`), only where the registry declares it. |
| `_in_place` | Consuming destructive update; exception E01 (in-place FFT). |

Every other spelling of input kind, parameters, layer or output mode is not a
suffix: it is a receiver type, an options field or a `Dest` value.
