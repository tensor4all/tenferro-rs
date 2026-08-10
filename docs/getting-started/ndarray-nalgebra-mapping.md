# Coming from ndarray / nalgebra / ndarray-linalg

This page is a translation guide for Rust users who arrive with `ndarray`,
`nalgebra`, or `ndarray-linalg` priors. PyTorch/JAX users have their own
[mapping page](./pytorch-jax-mapping.md); read
[Core Concepts](./core-concepts.md) for tenferro's own mental model.

The short version is the five conventions every tenferro program must follow:

- **Column-major storage.** Dense buffers are column-major: the leftmost
  dimension varies fastest. Row-major data passed to `from_vec_col_major` is
  silently reinterpreted as column-major — permuted/wrong values, never
  rejected.
- **No facade crate.** `cargo add tenferro` fails by design; depend on the
  crates you need (`tenferro-runtime`, `tenferro-cpu`, and operation crates).
- **Explicit backend.** Direct operations take an explicit backend argument;
  construct the backend/runtime once and reuse it — per-call construction
  discards the buffer pool.
- **Einsum dialect.** Equations need the explicit arrow (`"ij,jk->ik"`); `...`
  ellipsis is unsupported.
- **Result-returning operators.** Traced operators return `Result`; propagate
  with `?`.

Each is explained below with the concrete ndarray/nalgebra divergence.

## The column-major asymmetry

| Prior | Storage order | `from_vec_col_major` hazard |
| --- | --- | --- |
| `ndarray` / NumPy | Row-major (C order) | **Dangerous**: a row-major flat buffer is silently reinterpreted as column-major (permuted/wrong values), never rejected |
| `nalgebra` | Column-major (F order) | Safe: `as_slice()` / `.data` map directly |

This is the one prior that actively hurts. `ndarray` and NumPy store rows
contiguously, so the natural flat buffer you already have is in the wrong
physical order for tenferro — and construction does **not** detect the
mistake. A `[2, 3]` tenferro tensor reads elements down each column first.
Reorder the buffer explicitly before `from_vec_col_major`, or wrap it without
copying (see [Zero-copy interop](#zero-copy-interop-keep-your-faerndarray-buffers)).
`nalgebra` uses the same Fortran/column-major order as tenferro, LAPACK, and
Julia, so a `Matrix`'s `as_slice()` or `.data` maps directly into
`from_vec_col_major` with its shape.

## Crate selection: `cargo add tenferro` does not exist

There is deliberately no facade crate, so `cargo add tenferro` fails. Add the
smallest set of crates for your API tier directly:

| Program | Minimum direct crates |
| --- | --- |
| Concrete `Tensor` / `TypedTensor` compute | `tenferro-runtime`, `tenferro-cpu` |
| Eager forward / AD | `tenferro-ad`, `tenferro-cpu` |
| Traced graph | `tenferro-runtime`, `tenferro-cpu` (+ `tenferro-ad` for graph transforms) |
| Linear algebra / einsum / FFT | add `tenferro-linalg` / `tenferro-einsum` / `tenferro-fft` |
| CUDA | the value/op crates plus `tenferro-gpu` with the `cuda` feature |

See the [crate selection reference](https://tensor4all.org/tenferro-rs/skill-references/crate-selection.md)
for full dependency blocks and feature rules.

## The backend is an explicit value

In `ndarray`/`nalgebra`/`ndarray-linalg` the execution context is ambient: the
BLAS that was linked at build time is whatever your ops run through. tenferro
reifies it as a value — `CpuBackend` (or an `EagerRuntime`, or a `Runtime` for
traced graphs) — because the backend owns device placement, provider
selection, and buffer pools.

The companion idiom is **construct once, reuse**. A backend is owned state:

```rust
let mut backend = CpuBackend::new();
// ... many operations through the same `backend` ...
```

Constructing `CpuBackend::new()` per call discards the buffer pool and cache
each time, and defeats reuse of compiled programs. The same rule applies
across the tiers — see the [performance idioms reference](https://tensor4all.org/tenferro-rs/skill-references/performance-idioms.md).

## faer vs BLAS providers

tenferro's CPU backend has two provider families controlled by additive
features: the `cpu-faer` provider and the `cpu-blas` provider. `ndarray-linalg`
users know feature-based BLAS selection; tenferro's knobs live in the
`tenferro-cpu` (and `tenferro-runtime`) features.

| Provider | Features | When to use |
| --- | --- | --- |
| faer (default) | `cpu-faer` | Portable, pure Rust, no system dependencies; the right default for most workloads |
| BLAS / LAPACK | `cpu-blas` plus exactly one explicit provider feature | Large GEMM-dominated workloads, or to reuse an already-tuned system BLAS |

`cpu-faer` and `cpu-blas` are additive, and `CpuBackend::new()` selects the
compiled default — BLAS when `cpu-blas` is compiled, otherwise faer — with
`CpuBackend::with_kind` for explicit selection when both are compiled. Within
the BLAS family the three explicit provider features (`blas-openblas`,
`blas-mkl`, `blas-accelerate`) are mutually exclusive, and tenferro rejects a
build that enables more than one.

## Operation arity: `.dot()` to `matmul`

The receiver-and-arity shape changes across the three tenferro tiers, and the
backends are pushed into method signatures. Compare with `ndarray`'s
`a.dot(&b)`:

| Tier | Tenferro | Notes |
| --- | --- | --- |
| Direct | `a.matmul(&b, &mut backend)?` | explicit mutable backend argument |
| Eager | `a.matmul(&b)?` | `EagerRuntime` owns the backend |
| Traced | `a.matmul(&b)?` | builds a graph; returns `Result` |

<!-- snippet-source: docs/tutorial-code/src/bin/tenferro_compute_skill.rs#concrete-operation -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};

let mut backend = CpuBackend::new();
// The leftmost dimension varies fastest: this is a 2 x 3 column-major tensor.
let x = TypedTensor::<f64>::from_vec_col_major(
    vec![2, 3],
    vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
)?;
let weights = TypedTensor::<f64>::from_vec_col_major(
    vec![3, 2],
    vec![0.5, -1.0, 1.5, 1.0, 2.0, -0.5],
)?;
let projected = x.matmul(&weights, &mut backend)?;
assert_eq!(projected.shape(), &[2, 2]);
assert_eq!(projected.host_data()?, &[3.0, 6.0, 3.5, 11.0]);
```
<!-- end-snippet-source -->

In the direct tier the explicit backend moves from "linked somehow" to "passed
per call"; in the eager and traced tiers it is owned by the runtime and the
signature matches what `ndarray` users expect.

## Tensor vs TypedTensor

`ndarray`'s `Array<A, IxN>` and `nalgebra`'s `Matrix<T, ...>` are generic over
the element type — your priors map to `TypedTensor<T>`:

| Your prior | Tenferro |
| --- | --- |
| `ndarray::Array2<f64>` | `TypedTensor<f64>` (rank-generic) |
| `nalgebra::DMatrix<f64>` | `TypedTensor<f64>` |
| dtype chosen at runtime | `Tensor` (dtype-erased, has no ndarray counterpart) |

`Tensor` has no `ndarray`/`nalgebra` analogue: its element type is selected at
runtime. Reach for it when you need runtime dtype dispatch or direct backend
dispatch; for the ordinary fixed-element-type case the generic `TypedTensor<T>`
is the direct match.

## Zero-copy interop: keep your faer/ndarray buffers

Adopting tenferro kernels does **not** require migrating tensor ownership. You
can wrap an existing column-major buffer — a `faer::Mat`, an `ndarray` view, or
any slice plus strides — zero-copy with `TypedTensorView::from_slice`, and
write through it with `TypedTensorViewMut::from_slice`:

```text
your faer::Mat / ndarray view  ->  TypedTensorView::from_slice(shape, strides, offset, data)
```

The full runnable recipe (faer column padding, ndarray row-major wrap, and the
mutable variant) is in the [API cheatsheet "Borrowing external memory"](https://tensor4all.org/tenferro-rs/skill-references/api-cheatsheet.md#borrowing-external-memory).
Because strides are arbitrary, row-major data wraps **without** transposition —
but kernels are tuned for column-major contiguity, so materialize a copy when
performance matters and the wrapped buffer is row-major.

The "do I have to own a new tensor type" objection is partly cost, and the
measured answer is small: adding tenferro on top of an existing `faer`
dependency costs about **+10 unique crates and +28 s of one-time cold build**
(see [tenferro-rs#1602](https://github.com/tensor4all/tenferro-rs/issues/1602)).
For a larger GEMM, the kernel-run portion is what you are adopting; the memory
stays yours.

## Next steps

- [Core Concepts](./core-concepts.md) — tenferro's own mental model.
- [Choosing a Tensor API](../guides/choosing-an-api.md) — `TypedTensor`, `Tensor`, `EagerTensor`, or `TracedTensor`.
- [PyTorch and JAX Mapping](./pytorch-jax-mapping.md) — the same translation for torch/JAX priors.
- [API cheatsheet](https://tensor4all.org/tenferro-rs/skill-references/api-cheatsheet.md) — tier-by-tier arities and recipes.
