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
4. Read only the relevant reference below before writing the program.

| Need | Read |
| --- | --- |
| Crates, features, CPU providers, scratch crates | [crate selection](references/crate-selection.md) |
| Tier arities and extension-trait imports | [API cheatsheet](references/api-cheatsheet.md) |
| Backend/executor reuse and compile-once/run-many | [performance idioms](references/performance-idioms.md) |
| Column-major data, einsum syntax, registration, and setup traps | [pitfalls](references/pitfalls.md) |

## Non-negotiable defaults

- Dense flat buffers are column-major: the leftmost dimension varies fastest.
- Bind one backend/runtime/compiler and reuse it across related work.
- Compile traced programs once outside repeated execution loops.
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
