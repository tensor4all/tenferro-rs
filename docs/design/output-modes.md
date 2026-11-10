# Output Modes And Write Surfaces

This note records the output-update vocabulary shared by tensor, einsum, and
dot/GEMM APIs. Since the 0.12 call grammar (#2552) the public operation
spellings are generated from the operation registries; the authoritative list
of which operation declares which output mode is
[`docs/spec/generated/api-ops.md`](../spec/generated/api-ops.md).

## Public Suffix Vocabulary

| Form | Meaning |
| --- | --- |
| `op(..., exec) -> T` | Default options; the backend allocates the output. |
| `op_with(..., &opts, exec) -> T` | Explicit `XxxOptions`; the backend allocates the output. |
| `op_into(..., &opts, exec, dest) -> Result<()>` | Writes into a caller-provided `Dest`, only where the registry declares it. |
| `op_in_place(...)` | Consuming destructive update (exception E01, in-place FFT). This is separate from `_into`. |

Whether an input is owned, borrowed, typed, or a strided view is the receiver
type (`Tensor`, `TensorRead`, `TypedTensor<T>`, `TypedTensorView<T>`), never a
suffix. There is no public `_read`, `_add_to`, or `_into_accum` operation name.

`_into` takes a `Dest`, whose mode is the output-update semantics:

| `Dest` | Meaning |
| --- | --- |
| `Dest::overwrite(out)` | Overwrite the caller output; the previous value is never read. |
| `Dest::accumulate(out)` | Read-modify-write, `out += op(...)`. |
| `Dest::scaled(out, alpha, beta)` | `out = alpha * op(...) + beta * out`; `beta = 0` never reads `out`. |

The load-bearing rule is that any semantic use of the previous output value is
visible in the `Dest` value, not hidden in the method. A mode the registry does
not declare for an operation is `Unsupported` before anything is written. All
`_into` paths validate output dtype and shape before writing and never resize
the destination.

## Dot And Einsum

`DotGeneralConfig` describes contraction dimensions only. Conjugation is an
operation option (`DotGeneralOptions`, `EinsumOptions::conj`); the
output-update coefficients are the `Dest` value. Einsum's `einsum_into` and
`ConcreteEinsumPlan::execute_into` declare `Overwrite`, `Accumulate`, and
`Scaled`.

Typed inputs take typed destinations (an owned `TypedTensor<T>` or a mutable
typed view) because the dtype is statically known; erased inputs take
`TensorWrite<'_>`, so dtype and view validation happen at the backend boundary.

## Provider Hooks

Backend and provider traits (`Engine`, `TensorDot`, `LinalgBackend`,
`FftBackend`) keep their own hook vocabulary; these are provider contracts
(exception E10), not public operation grammar:

| Hook form | Meaning |
| --- | --- |
| `op_read(...)` | Borrowed `TensorRead` inputs; output is allocated. |
| `op_read_into(..., out: TensorWrite)` | Borrowed inputs plus overwritten caller output. This is the main backend hook shape. |
| `op_read_into_accum(..., accum, out: TensorWrite)` | Dot/GEMM-only update, `out = alpha * op(lhs, rhs) + beta * out`, with `DotGeneralAccumulation` carrying conjugation flags and `alpha`/`beta`. |

A public `Dest::accumulate` or `Dest::scaled` on a dot/GEMM-shaped operation is
lowered onto the `_read_into_accum` hook (`Dest::into_accumulation`, exception
E06); there is no separate `_linear_into` vocabulary.

## Elementwise

`_into` writes are overwrite unless the `Dest` says otherwise. `_in_place`
methods are allowed only when the aliased input/output contract is explicitly
implemented and tested. Generic `*_into` kernels must not be assumed
alias-safe unless their owner documents that guarantee.
