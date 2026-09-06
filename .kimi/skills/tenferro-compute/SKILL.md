---
name: tenferro-compute
description: Write Rust programs that use tenferro for tensor computation, autodiff, einsum, linear algebra, and explicit backend execution.
---

# tenferro-compute

Load this skill when the task is to **use** tenferro from downstream Rust code,
not when changing tenferro itself.

## Fast path

1. Choose the API tier: direct concrete tensors, eager tensors, or traced
   graphs. The same operation has different receivers and arities at each tier.
2. Add the direct crates for that tier. There is no root `tenferro` facade;
   import operation families such as `tenferro-einsum` and `tenferro-linalg`
   directly.
3. Bring the operation's public `*Ext` trait into scope. An E0599 saying that
   a method does not exist usually means the right extension trait is missing.
4. If an older example mentions a removed module/free function or a
   constructor signature that no longer compiles, read the [API migration
   guide](../../../docs/getting-started/api-migration.md).
5. Read only the relevant reference below before writing the program.

| Need | Read |
| --- | --- |
| Crates, features, CPU providers, scratch crates | [crate selection](references/crate-selection.md) |
| Tier arities and extension-trait imports | [API cheatsheet](references/api-cheatsheet.md) |
| Backend/executor reuse and compile-once/run-many | [performance idioms](references/performance-idioms.md) |
| Column-major data, einsum syntax, registration, and setup traps | [pitfalls](references/pitfalls.md) |

## Non-negotiable defaults

- **Column-major storage.** Dense buffers are column-major: the leftmost
  dimension varies fastest. Row-major data passed to `from_vec_col_major` is
  silently reinterpreted as column-major — permuted/wrong values, never
  rejected.
- **No facade crate.** `cargo add tenferro` fails by design; depend on the
  crates you need (`tenferro-runtime`, `tenferro-cpu`, and operation crates).
- **Explicit execution owner.** Concrete operations take a borrowed session
  inside `backend.with_backend_session(...)` (`BackendSessionHost` import).
  Construct the backend/runtime once and reuse it — per-call construction
  discards the buffer pool. Eager tensors retain their runtime instead.
- **Representation is not reuse.** Integer einsum labels still plan. For a
  repeated compatible equation/input count/dtype/shape, prepare a
  `ConcreteEinsumPlan` once, even from a string; see
  [performance idioms](references/performance-idioms.md). Do not flatten
  parenthesized contraction order into label arrays.
- **Einsum dialect.** Equations need the explicit arrow (`"ij,jk->ik"`).
  Flat notation supports one right-aligned, broadcastable `...` ellipsis per
  term; `EinsumNotation` provides the programmatic form.
- **Result-returning operators.** Traced operators return `Result`; propagate
  with `?`.
- CPU/GPU transfers are explicit; unsupported GPU operations do not silently
  fall back to CPU.
- Traced standard extensions need an explicitly installed extension module and
  a matching registered runtime engine.
- Keep a scratch crate in its own Cargo workspace (use an empty
  `[workspace]` table when it lives inside a checkout), and enable exactly one
  BLAS provider when using `cpu-blas`.

The executable Rust examples in the references are extracted from
`docs/tutorial-code/src/bin/tenferro_compute_skill.rs` and compiled by the
existing tutorial-binary test.
