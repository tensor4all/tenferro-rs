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
- Use `CpuBackend::with_threads(n)` for faer/native CPU work; configure BLAS
  provider threads with the provider's environment variables.
- Choose exactly one BLAS provider feature (`blas-openblas`, `blas-mkl`, or
  `blas-accelerate`) when using `cpu-blas`.
- CUDA is explicit: enable `tenferro-gpu`'s `cuda` feature and upload CPU
  tensors before CUDA operations. There is no implicit transfer.

## Per-origin traps

Same traps, keyed by the priors you arrived with.

### If you come from ndarray / NumPy

- Your buffers are row-major; passing them to `from_vec_col_major` silently
  reinterprets them as column-major (no error, permuted/wrong values). Reorder
  explicitly or wrap with
  `TypedTensorView::from_slice` (see the [API cheatsheet](api-cheatsheet.md#borrowing-external-memory)).
- `cargo add tenferro` fails: there is no facade crate. Add
  `tenferro-runtime` + `tenferro-cpu` (plus operation crates).
- `Array2::dot` is a free-standing habit; tenferro direct ops run inside an
  explicit session: `backend.with_backend_session(|s| a.matmul(&b, s))`.
- Your priors do not include a dtype-erased tensor; `Tensor` (runtime dtype)
  has no ndarray counterpart — use `TypedTensor<T>`.

### If you come from nalgebra

- nalgebra is column-major like tenferro, so `.data` / `as_slice()` buffers map
  directly to `from_vec_col_major`.
- `Matrix::dot` / `gemm` are methods without an execution context; tenferro
  direct ops need the explicit backend session. Eager and traced tiers drop
  it: `EagerTensor` methods run through the `EagerRuntime`, traced methods
  build a graph.
- A single `DMatrix<T>` maps to `TypedTensor<T>`; there is no separate runtime
  dtype type unless you actually need runtime dtype selection.

### If you come from PyTorch / JAX

- Trace one level of `?`: traced operators return `Result`; `a.matmul(&b)` is
  a `Result`, unlike `torch.matmul` / `jnp.matmul`.
- Einsum needs the explicit arrow and rejects `...` (see
  [Einsum syntax](#einsum-syntax)).
- The backend is not ambient: eager code must own an `EagerRuntime`, traced
  code must register an engine plus extension modules (see
  [Extension registration](#extension-registration)).

## Retired names

Searching for the retired `TypedStridedTensorView` (tenferro-rs#886) leads
nowhere: the name survives only in the obsolete-names vocabulary test. The
zero-copy view API is `TypedTensorView::from_slice` / `TypedTensorViewMut::from_slice`
(see the [API cheatsheet](api-cheatsheet.md#borrowing-external-memory)).
