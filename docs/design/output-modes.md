# Output Modes And Write Surfaces

This note records the output-update vocabulary shared by tensor, einsum, and
dot/GEMM APIs.

## Suffix Vocabulary

| Form | Meaning |
| --- | --- |
| `op(...) -> Tensor` | Backend allocates the output. |
| `op_read(...) -> Tensor` | Inputs are borrowed through `TensorRead`; output is still allocated. |
| `op_into(..., out: TensorWrite) -> Result<()>` | Overwrite caller output. No resize and no accumulation. |
| `op_read_into(..., out: TensorWrite) -> Result<()>` | Borrowed inputs plus overwritten caller output. This is the main backend hook shape. |
| `op_in_place(...)` | Destructive update of an input tensor. This is separate from `_into`. |
| `op_add_to(..., out: TensorWrite) -> Result<()>` | Read-modify-write update, `out += op(...)`. |
| `op_read_into_accum(..., accum, out: TensorWrite) -> Result<()>` | Dot/GEMM-only update, `out = alpha * op(lhs, rhs) + beta * out`. |

The load-bearing rule is that bare `_into` means overwrite. Any semantic use of
the previous output value must be visible in the method name (`_add_to`) or in
the explicit dot/GEMM accumulation argument (`_into_accum`).

## Dot And Einsum

`DotGeneralConfig` describes contraction dimensions only. Output-update
semantics live in `DotGeneralAccumulation`, which carries conjugation flags and
`alpha`/`beta` coefficients. The existing `_into_accum` spelling remains the
official dot/GEMM read-modify-write surface; there is no separate
`_linear_into` vocabulary.

Einsum has both typed and erased output surfaces. Typed methods such as
`einsum_into` may take `&mut TypedTensor<T>` because the dtype is statically
known. Erased plan execution methods use `TensorWrite<'_>` so dtype and view
validation happen at the backend boundary.

## Elementwise

Elementwise `_into` methods overwrite caller-provided outputs. `_in_place`
methods are allowed only when the aliased input/output contract is explicitly
implemented and tested. Generic `*_into` kernels must not be assumed
alias-safe unless their owner documents that guarantee.

Elementwise `_add_to` is reserved for future accumulation APIs and must remain
distinct from overwrite `_into` methods.
