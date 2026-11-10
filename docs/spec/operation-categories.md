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

**API style.** Traced core operations use `TracedTensor` methods or associated
functions, not module free functions (`x.exp()`, `x.matmul(&y)`,
`TracedTensor::concatenate(...)`). Eager core operations use methods on a
runtime-bound `EagerSession` within `EagerRuntime::with_eager_session`:
`session.exp(&x)`, `session.dynamic_slice(&x, &starts, &sizes)`, and
`session.where_select(&condition, &on_true, &on_false)`; `EagerTensor` is the
value/trace handle and does not implicitly enter the backend for each operation.
Non-AD concrete core operations use the crate-root `CoreOps` trait of
`tenferro-runtime` (`x.exp(exec)`, `x.matmul(&y, exec)`), implemented for every
`AsView` receiver (`Tensor`, `TensorRead`, `TypedTensor<T>`,
`TypedTensorView<T>`), because those types are owned by `tenferro-tensor`, not
`tenferro-runtime`. Extension families likewise use crate-root operation traits
because Rust does not let extension crates add inherent methods to external
tensor types: `LinalgOps`, `FftOps` and `TensordotOps` are tensor receiver
methods, concrete einsum (`EinsumOps`) is an input slice/array method, traced
einsum (`TracedEinsumOps`) is a `TraceContext` method, and the eager forms
(`EagerLinalgOps`, `EagerEinsumOps`, `EagerFftOps`) are `EagerSession`
methods. Every family follows the call grammar of
[API conventions](api-conventions.md): `op`, `op_with(&opts)` and, where the
registry declares it, `op_into(.., dest)`, with the engine last on the
concrete layer. The operation list per layer is generated in
[`generated/api-ops.md`](generated/api-ops.md).

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
| `erf` | ✅ | ✅ |
| `sigmoid` `silu` `softplus` `gelu` `gelu_tanh` | ✅ | ✅ |

`erf` is real-only. The activations are composites (also on the concrete-session
surface); their formulation and edge-case policy are specified in
[`tensor-semantics.md`](tensor-semantics.md) Section VIII.

**Parallelism.** The elementwise categories (2 and 3) are embarrassingly parallel.
Their CPU kernels are data-parallel via rayon behind the `parallel` feature (the
same feature that gates the strided parallel kernels), so the built-in named ufuncs
parallelize automatically. For user-supplied closures, parallelism is explicit:
`map` vs `par_map` (Section 10).

## 4. Reductions

| Operation | Eager | Traced |
|---|---|---|
| `reduce_sum` `reduce_prod` `reduce_max` `reduce_min` | ✅ | ✅ |
| `reduce_mean` | ✅ | ✅ |
| `softmax` `log_softmax` `masked_softmax` `masked_log_softmax` | ✅ | ✅ |
| `layer_norm` `rms_norm` | ✅ | ✅ |
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
| `take_along_axis` | ✅ | ✅ | NumPy semantics over `gather`; batch-varying indices |
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

- **Linalg** (`tenferro-linalg`): `svd`, `svdvals`, `qr`, `householder_qr`,
  `rank_revealing_qr`, `lu`, `full_piv_lu`, `full_piv_lu_solve`, `solve`,
  `lstsq`, `triangular_solve`, `cholesky`, `eig`, `eigvals`, `eigh`,
  `eigvalsh`, `slogdet`, `det`, `inv`, `pinv` and `norm` through `LinalgOps`
  (concrete), `TracedLinalgOps` (on `TracedTensor`) and `EagerLinalgOps` (on
  `EagerSession`). Multi-output results are structs (`Svd`, `Qr`, `Lu`,
  `FullPivLu`, `Eig`, `Eigh`, `Slogdet`).
- **Einsum** (`tenferro-einsum`): `einsum` + contraction planning, and
  `tensordot`. Concrete inputs, owned, borrowed or typed, use `EinsumOps` on an
  input slice or array (`[&a, &b].einsum(spec, exec)`, `einsum_into` with a
  `Dest`) and `ConcreteEinsumPlan` for repeated execution; traced graph
  construction uses `TracedEinsumOps` on `TraceContext`; eager einsum and
  `tensordot` use `EagerEinsumOps` on an `EagerSession`; concrete and traced
  `tensordot` use `TensordotOps` and `TracedTensordotOps`.
- **FFT** (`tenferro-fft`): `fft`, `ifft`, `rfft`, `irfft`, `fftn`, `ifftn`,
  `rfftn` and `irfftn` through `FftOps` (concrete), `TracedFftOps` (on
  `TracedTensor`) and `EagerFftOps` (on `EagerSession`, which also has the
  consuming `fft_in_place` / `ifft_in_place`).

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

Public axis arguments are signed `isize` on the concrete (`CoreOps`,
`LinalgOps`, `FftOps`, `TensordotOps`), traced and eager surfaces: reductions,
`transpose`, `reverse`, `concatenate`, `stack`, `index_select`, `softmax` and
the other composites, `norm` axes, FFT axes and `TensorDotAxes::Axes`. A
negative axis counts from the end (`-1` is the last axis); each axis is
normalized relative to the input rank, and an out-of-range or repeated axis is
a validation error. Configuration structs that mirror the IR
(`DotGeneralConfig`, `GatherConfig` and the like) and the dimension map of
`broadcast_in_dim` keep non-negative `usize` dimensions.

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

`scripts/check-operation-categories.py` checks the required public method names
on `EagerSession` (across `eager.rs` and `shape_packing.rs`) and `TracedTensor`,
including the Eager/Traced parity rule. It does not validate runtime behavior;
focused operation tests cover that separately. The contract and its parity matrix are
frozen together with the public API (see the API-freeze issue).
