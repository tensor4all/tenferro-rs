# Operation Categories And Surface Parity

**Status:** v0.1 public-surface contract. This document is the **user-facing**
operation contract so the public surface grows by spec, not ad hoc.

Related:

- `primitive-catalog.md` — the **IR / primitive** vocabulary (internal ops). This
  document sits above it: the user-facing operations and where they are exposed.
- `supported-ops.md` — per-crate **status** (CPU/CUDA). This document is the
  **contract**, not a status snapshot.

## Surfaces

| Surface | Type | Role |
|---|---|---|
| **TypedTensor** | `TypedTensor<T, R>` (non-AD) | Static/dynamic-rank typed value; the "lightweight array" path |
| **Tensor** | `Tensor` (runtime dtype) | Dynamic-dtype value with explicit backend |
| **Eager** | `EagerTensor` | PyTorch-style immediate execution + stateful `backward()` plus functional `grad`/`vjp`/`jvp` |
| **Traced** | `TracedTensor` | JAX-style traced graph + `grad`/`vjp`/`jvp` |

**Rank model.** `TypedTensor<T, R = DynRank>` offers `Rank<N>` (opt-in static rank)
and `DynRank` (default). `DynRank` is the canonical rank for AD and traced execution
(`TracedTensor`/`EagerTensor` are rank-erased); do not push const-generic `Rank<N>`
through the AD/traced surfaces. This is current design intent, not a frozen contract.

**API style.** Core AD operations on `EagerTensor` and `TracedTensor` are methods
or associated functions, not module free functions. Single-output operations use
methods (`x.exp()`, `x.matmul(&y)`, `x.gather(&idx, config)`); operations with no
natural receiver use associated functions (`TracedTensor::concatenate(...)`,
`EagerTensor::where_select(...)`). Non-AD concrete operations use
backend-explicit crate-root extension traits (`TensorOpsExt`,
`TypedTensorOpsExt`, and `TypedTensorMaskOpsExt`) because `Tensor` and
`TypedTensor` are owned by `tenferro-tensor`, not `tenferro-runtime`. Extension
families likewise use crate-root extension traits because Rust does not let
extension crates add inherent methods to external tensor types: linalg/FFT are
tensor receiver methods, eager einsum is an input slice/array method, and
traced einsum is a `GraphCompiler` method.

**Parity rule (the core of the contract):** every operation in the Elementwise,
Reductions, Shape/structural, and Indexing categories **must be exposed on both
`Eager` and `Traced`**. "Exists on Eager but not Traced" (or vice versa) is a spec
violation, to be caught by `scripts/check-operation-categories.py`.

Legend: ✅ exposed today (verified) · ⬜ required by this contract, currently missing
(gap to implement) · · audit pending · — not applicable by design · (ext) provided by
an extension crate.

## 1. Construction

| Operation | TypedTensor | Tensor | Eager | Traced |
|---|---|---|---|---|
| `zeros` / `ones` | ✅ | · | (`constant_from`) | · |
| `full` | · | · | · | · |
| `from_vec_col_major` | ✅ | ✅ | (`constant_from`/`variable_from`) | ✅ |
| `arange` | · | · | — | — |
| `eye` / `identity` | · | · | · | · |

## 2. Elementwise — arithmetic, comparison, selection

| Operation | Eager | Traced | Notes |
|---|---|---|---|
| `add` `sub` `mul` `div` `neg` | ✅ | ✅ | |
| `abs` `sign` | ✅ | ✅ | |
| `conj` | ✅ | ✅ | complex |
| `pow` | ✅ | ✅ | |
| `compare(dir)` | ✅ | ✅ | produces `Bool` |
| `select` / `where_select` | ✅ | ✅ | associated function |
| `clamp` | ✅ | ✅ | |
| `maximum` `minimum` | ✅ | ✅ | |

## 3. Elementwise — analytic (ufunc catalog)

The named analytic set is the supported elementwise surface **on the AD surfaces
(`Eager`/`Traced`)**. Arbitrary-closure `map`/`mapv` is available only on the non-AD
surfaces — see Section 10.

| Operation | Eager | Traced |
|---|---|---|
| `exp` `log` `sin` `cos` `tanh` `sqrt` `rsqrt` | ✅ | ✅ |
| `expm1` (`expm`) | ✅ | ✅ |
| `log1p` | ✅ | ✅ |

**Parallelism.** The elementwise categories (2 and 3) are embarrassingly parallel.
Their CPU kernels are data-parallel via rayon behind the `parallel` feature (the
same feature that gates the strided parallel kernels), so the built-in named ufuncs
parallelize automatically. For user-supplied closures, parallelism is explicit:
`map` vs `par_map` (Section 10).

## 4. Reductions

| Operation | Eager | Traced |
|---|---|---|
| `reduce_sum` `reduce_prod` `reduce_max` `reduce_min` | ✅ | ✅ |
| `mean` | · | · |
| `argmax` / `argmin` | · | · |

Reduction axes use one convention on the eager and traced surfaces:

- `None` reduces over every axis;
- `Some(&[])` is the identity operation; and
- `Some(axes)` reduces over exactly those validated axes.

The public boundary normalizes `None` to the explicit axis list before graph or
eager operation construction. Primitive IR and backend reduction contracts
therefore always contain explicit axes rather than a second all-axes spelling.
For a rank-zero tensor, `None` normalizes to an empty list and remains the
identity.

## 5. Shape / structural

| Operation | Eager | Traced | Notes |
|---|---|---|---|
| `reshape` | ✅ | ✅ | Traced also `reshape_sym` |
| `transpose` / `permute` | ✅ | ✅ | |
| `broadcast` / `broadcast_in_dim` | ✅ | ✅ | Traced also `_sym` |
| `concatenate` | ✅ | ✅ | associated function |
| `stack` | ✅ | ✅ | associated function (`dim: isize`) |
| `split` | ⬜ | ⬜ | |
| `pad` | ✅ | ✅ | |
| `reverse` / `flip` | ✅ | ✅ | |
| `repeat` / `tile` | ⬜ | ⬜ | |

## 5a. DType / value conversion

| Operation | Eager | Traced | Notes |
|---|---|---|---|
| `convert` | ✅ | ✅ | checked conversion accepted by the dtype-promotion lattice; returns a typed error otherwise |
| `cast` | ✅ | ✅ | explicit lossy dtype projection; required when callers intentionally request truncation, precision narrowing, complex projection, or boolean truthiness |

## 6. Indexing / data movement

| Operation | Eager | Traced | Notes |
|---|---|---|---|
| `slice` | ✅ | ✅ | |
| `dynamic_slice` | ✅ | ✅ | |
| `dynamic_update_slice` | · | · | |
| `gather` | ✅ | ✅ | |
| `scatter` | ✅ | ✅ | |
| `take` | · | · | |
| `extract_diag` / `embed_diag` | ✅ | ✅ | |
| `tril` / `triu` | ✅ | ✅ | |

## 7. Contraction core

| Operation | Eager | Traced |
|---|---|---|
| `dot_general` | ✅ | ✅ |
| `matmul` (rank-2 sugar over `dot_general`) | ✅ | ✅ |

## 8. Extension operation families

Provided by extension crates, operating on the same tensor types on both the eager
and traced surfaces (subject to the same parity rule within each family):

- **Linalg** (`tenferro-linalg`): `svd`, `qr`, `eig`, `eigh`, `solve`,
  `triangular_solve`, `cholesky`, `lu`, `full_piv_lu` through
  `TracedTensorLinalgExt` / `EagerTensorLinalgExt`.
- **Einsum** (`tenferro-einsum`): `einsum` + contraction planning.
  Concrete inputs use `TensorEinsumExt` and `TypedTensorEinsumExt` for owned
  tensors, `TensorReadEinsumExt` and `TypedTensorReadEinsumExt` for borrowed
  views, and `ConcreteEinsumPlan` for repeated execution; traced graph
  construction uses `TraceContextEinsumExt`; autodiff eager inputs use
  `EagerEinsumExt`; `tensordot` sugar uses tensor extension traits.
- **FFT** (`tenferro-fft`): `fft`, `ifft`, `rfft`, and `irfft`.
  Concrete inputs use `TensorFftExt` and `TensorReadFftExt`; traced graph
  construction uses `TracedTensorFftExt`.

## 9. AD transforms

Not value operations; listed for completeness. `EagerRuntime` exposes stateful
`backward()` / `backward_with(seed)` plus functional `grad` / `vjp` / `jvp` /
HVP composition on eager tensors. `TracedTensor` and `AdContext` expose graph
`grad` / `vjp` / `jvp` / HVP workflows for compiled execution and graph reuse.

## 10. Host iteration and closure map (non-AD surfaces only)

These are convenient for the "lightweight array" path and are provided on the
**non-AD** surfaces (`TypedTensor`, `Tensor`, and their views). They are **not**
on `Eager`/`Traced`: an opaque closure cannot be GPU-lowered or differentiated, so
the AD surfaces use the named ufunc catalog (Section 3) instead.

| Operation | TypedTensor | Tensor | Eager | Traced | Notes |
|---|---|---|---|---|---|
| `map` (closure) → new tensor | ⬜ | ⬜ | — | — | sequential; `FnMut`; ordered; available in minimal builds |
| `map_inplace` / `map_into` | ⬜ | ⬜ | — | — | in-place / out-param variants |
| `par_map` / `par_map_inplace` | ⬜ | ⬜ | — | — | rayon data-parallel; `Fn + Sync` closure, `Send` elements; behind `parallel` feature |
| `iter` / `iter_mut` (elements) | ⬜ | ⬜ | — | — | over a view; respects layout |
| `par_iter` / `par_iter_mut` | ⬜ | ⬜ | — | — | rayon parallel iterators; behind `parallel` feature |
| `indexed_iter` | ⬜ | ⬜ | — | — | yields `(index, &value)` |
| axis / lane iteration | · | · | — | — | optional, ndarray-style |

**Sequential vs parallel is a deliberate API split** (different trait bounds, not a
flag): `map` takes `FnMut`, preserves order, and works even in minimal/`no_std`
builds; `par_map` requires a `Fn + Sync` closure over `Send` elements, is
order-independent (the closure must be pure / position-independent), uses rayon, and
is **std-only, feature-gated** (`parallel`). Keeping them distinct avoids silently
requiring `Send`/`Sync` bounds on the sequential path and keeps the `no_std` data
layer (the no_std epic) free of rayon.

Constraints (document in the API):

- **Host-only.** `map`/iterators operate on host data; a GPU-resident tensor must
  be downloaded first (explicit, per the no-silent-transfer rule).
- **Non-AD.** These live on the non-AD value types by definition; gradients do not
  flow through a closure. Use the AD surfaces + named ufuncs when you need autodiff.
- **Layout-aware.** Iteration order follows column-major storage; logical-order
  traversal is via a view (e.g. a transposed/permuted view).

## Argument convention (fixed across categories)

Shape arguments accept owned or borrowed shape-like values: fixed-rank APIs use
`impl Into<R::Shape>`, while dynamic-rank APIs use `impl IntoShapeVec`. Axis lists,
dimension mappings, and similar collections use `impl AsRef<[T]>`.

Axis sign is part of each operation's public contract, not a global coercion.
Negative `isize` axes are currently accepted only by FFT (`fft`, `ifft`, `rfft`,
`irfft`), `index_select`, `stack`, and explicit `TensorDotAxes::Axes`; each is
normalized relative to the input rank and rejects an out-of-range result. Other
axis-taking APIs, including reductions, `concatenate`, and `reverse`, accept
non-negative `usize` axes only. New negative-axis support must be added
deliberately and consistently across the relevant concrete, eager, and traced
surfaces.

Multi-input structural ops (`concatenate`, `stack`, `split`, …) must accept owned
values, views, and `IntoIterator` ergonomically, and define behavior for an empty
input. This pre-empts the owned-vs-view boilerplate complaint seen in NumPy/ndarray
ecosystems (e.g. rust-ndarray#1591).

## Non-goals

- **No arbitrary-closure `map`/`mapv` on the AD surfaces (`Eager`/`Traced`).** An
  opaque Rust closure cannot be lowered to GPU kernels and is not differentiable
  through the traced/AD path; those surfaces use the named ufunc catalog (Section 3).
  Closure `map` and element iterators **are** provided on the non-AD surfaces
  (`TypedTensor`/`Tensor`) — see Section 10.

## Enforcement

`scripts/check-operation-categories.py` verifies that the implemented public surface
matches this contract (especially the Eager/Traced parity rule), in the same spirit
as the other repository boundary checks. The contract and its parity matrix are
frozen together with the public API (see the API-freeze issue).
