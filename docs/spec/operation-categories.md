# Operation Categories And Surface Parity

**Status:** draft contract, to be ratified and implemented (see the operation-category
issue). This document is the **user-facing** operation contract. It is fixed here
*before* implementation so the public surface grows by spec, not ad hoc.

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
| **Eager** | `EagerTensor` | PyTorch-style immediate execution + `backward()` |
| **Traced** | `TracedTensor` | JAX-style traced graph + `grad`/`vjp`/`jvp` |

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
| `from_vec_row_major` / `from_vec_col_major` | ✅ | ✅ | (`constant_from`/`variable_from`) | ✅ |
| `arange` | · | · | — | — |
| `eye` / `identity` | · | · | · | · |

## 2. Elementwise — arithmetic, comparison, selection

| Operation | Eager | Traced | Notes |
|---|---|---|---|
| `add` `sub` `mul` `div` `neg` | ✅ | ✅ | |
| `abs` `sign` | ✅ | ✅ | |
| `conj` | ✅ | ✅ | complex |
| `pow` | ✅ | ✅ | |
| `compare(dir)` | · | ✅ | produces `Bool` |
| `select` | ✅ | ⬜ | gap on Traced |
| `clamp` | ✅ | ⬜ | gap on Traced |
| `maximum` `minimum` | ✅ | ⬜ | gap on Traced |

## 3. Elementwise — analytic (ufunc catalog)

The named analytic set is the supported elementwise surface. **There is no
arbitrary-closure `map`/`mapv`** — see Non-goals.

| Operation | Eager | Traced |
|---|---|---|
| `exp` `log` `sin` `cos` `tanh` `sqrt` `rsqrt` | ✅ | ✅ |
| `expm1` (`expm`) | ✅ | ✅ |
| `log1p` | · | · |

## 4. Reductions

| Operation | Eager | Traced |
|---|---|---|
| `reduce_sum` `reduce_prod` `reduce_max` `reduce_min` | ✅ | ✅ |
| `mean` | · | · |
| `argmax` / `argmin` | · | · |

## 5. Shape / structural

| Operation | Eager | Traced | Notes |
|---|---|---|---|
| `reshape` | ✅ | ✅ | Traced also `reshape_sym` |
| `transpose` / `permute` | ✅ | ✅ | |
| `broadcast` / `broadcast_in_dim` | ✅ | ✅ | Traced also `_sym` |
| `concatenate` | ✅ | ⬜ | op exists in vocabulary; Traced method missing |
| `stack` | ✅ | ⬜ | currently only via `shape_packing` (`dim: isize`); needs a clean general form |
| `split` | ⬜ | ⬜ | |
| `pad` | ✅ | ⬜ | Traced only has `pad_to_match`; needs general `pad` |
| `reverse` / `flip` | ✅ | ⬜ | gap on Traced |
| `repeat` / `tile` | ⬜ | ⬜ | |

## 6. Indexing / data movement

| Operation | Eager | Traced | Notes |
|---|---|---|---|
| `slice` | ✅ | ⬜ | Traced has `dynamic_truncate`, not general `slice` |
| `dynamic_slice` | ✅ | ⬜ | |
| `dynamic_update_slice` | · | · | |
| `gather` | ✅ | ⬜ | |
| `scatter` | ✅ | ⬜ | |
| `take` | · | · | |
| `extract_diag` / `embed_diag` | ✅ | ✅ | |
| `tril` / `triu` | ✅ | ⬜ | |

## 7. Contraction core

| Operation | Eager | Traced |
|---|---|---|
| `dot_general` | ✅ | ✅ |
| `matmul` (sugar over `dot_general`) | ✅ | ⬜ |

## 8. Extension operation families

Provided by extension crates, operating on the same tensor types on both the eager
and traced surfaces (subject to the same parity rule within each family):

- **Linalg** (`tenferro-linalg`): `svd`, `qr`, `eig`, `eigh`, `solve`,
  `triangular_solve`, `cholesky`, `lu`, `full_piv_lu`.
- **Einsum** (`tenferro-einsum`): `einsum` + contraction planning.
- **FFT** (`tenferro-fft`): `fft`, `rfft`, `irfft`.

## 9. AD transforms

Not value operations; listed for completeness. `grad` / `vjp` / `jvp` / HVP on
`Traced`; `backward` on `Eager` scalar losses.

## Argument convention (fixed across categories)

Multi-input structural ops (`concatenate`, `stack`, `split`, …) must accept owned
values, views, and `IntoIterator` ergonomically, and define behavior for an empty
input. This pre-empts the owned-vs-view boilerplate complaint seen in NumPy/ndarray
ecosystems (e.g. rust-ndarray#1591).

## Non-goals

- **No arbitrary-closure `map` / `mapv`.** An opaque Rust closure cannot be lowered
  to GPU kernels and is not differentiable through the traced/AD path. The supported
  elementwise surface is the named ufunc catalog (Section 3). A CPU-only, non-AD
  closure map could be considered separately but is explicitly out of this contract.

## Enforcement

`scripts/check-operation-categories.py` verifies that the implemented public surface
matches this contract (especially the Eager/Traced parity rule), in the same spirit
as the other repository boundary checks. The contract and its parity matrix are
frozen together with the public API (see the API-freeze issue).
