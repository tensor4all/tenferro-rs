# Pitfalls

## Column-major input

tenferro owns compact dense buffers in column-major order. For shape `[2, 3]`,
the physical sequence `[a00, a10, a01, a11, a02, a12]` is correct. A row-major
literal has the wrong values in the right shape; construction does not detect
that semantic mistake. Reorder external NumPy/PyTorch/JAX buffers explicitly
before `from_vec_col_major`. The checked column-major example is in the [API
cheatsheet](api-cheatsheet.md#direct-concrete-tensors).

## Einsum syntax

Use the explicit arrow in every equation: `"ij,jk->ik"`. The tenferro dialect
is intentionally smaller than NumPy's: `...` ellipsis is not supported, and
`"i->ii"` is a tenferro extension rather than a general NumPy spelling. Read
the `tenferro-einsum` guide before porting a large equation.

## Extension registration

A traced extension operation needs both sides of the runtime boundary:

1. register `tenferro_cpu::runtime_engine_registration(&backend)`;
2. install the matching operation module, for example
   `tenferro_einsum::extension_module::<CpuBackend>(runtime_engine_id()?)`.

A missing module is a runtime error, not a signal to silently construct another
backend or fall back to a concrete implementation. The complete checked setup
is in the [traced API example](api-cheatsheet.md#traced-tensors-and-extensions).

## Cargo setup traps

- A scratch crate inside the checkout needs an empty `[workspace]` table.
- `cpu-faer` and `cpu-blas` are CPU capability features; at least one is needed.
- Choose exactly one BLAS provider feature (`blas-openblas`, `blas-mkl`, or
  `blas-accelerate`) when using `cpu-blas`.
- CUDA is explicit: enable `tenferro-gpu`'s `cuda` feature and upload CPU
  tensors before CUDA operations. There is no implicit transfer.
